import ast, re, os, time
import numpy as np
import scipy

import logging, threading
import textwrap, copy
from typing import Sequence, Any, Dict, Tuple
METRIC_NAME = "n_solved_eval"
NOISE_SCALE = 1e-5

# custom imports
from evolution import code_types, code_utils, code_parser

class database_config(object):
    functions_per_prompt: int = 2
    num_islands: int = 8
    reset_period: int = 60 * 60 * 1  # seconds
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 10000

def _get_score(metrics: Dict, score_key: str = METRIC_NAME) -> float:
    # higher is better
    return metrics.get(score_key, 0.0)

def _get_signature(metrics: Dict, n = 10) -> Tuple[float]:
    # round score to multiple of n for clusters of similar performance
    score = _get_score(metrics)
    return (score // n) * n

def _softmax(w: np.ndarray, temperature: float = 1.):
    result = scipy.special.softmax(w / temperature, axis=-1)
    idx = np.argmax(result)
    result[idx] = 1. - np.sum(result[:idx]) - np.sum(result[idx+1:])
    return result

class ProgramDatabase:
    """Population of correct programs, sampled to create prompts. 
    Diversity encouraged by division into island subpopulations."""

    def __init__(self, config, template: code_types.Program, function_to_evolve: str):
        self.config = config
        self.template = template
        self.function_to_evolve = function_to_evolve
        self.eval_metric_name = METRIC_NAME
        self._lock = threading.Lock()

        self.last_reset_time = time.time()
        self.islands = [Island(template, function_to_evolve, config.functions_per_prompt,
                               config.cluster_sampling_temperature_init, 
                               config.cluster_sampling_temperature_period)
                        for _ in range(config.num_islands)]
        self.best_score_per_island = [float('-inf')] * config.num_islands
        self.best_program_per_island = [None] * config.num_islands
        self.best_metrics_per_island = [{}] * config.num_islands
    
    def generate_prompts(self):
        with self._lock:
            island_id = np.random.randint(self.config.num_islands)
            code, version_generated = self.islands[island_id].generate_prompt()
            return code_types.Prompt(code=code, island_id=island_id, version_generated=version_generated)

    def _register_program_in_island(self, program, island_id, metrics):
        self.islands[island_id]._register_program(program, metrics)
        score = _get_score(metrics, self.eval_metric_name)
        if score > self.best_score_per_island[island_id]:
            self.best_score_per_island[island_id] = score
            self.best_program_per_island[island_id] = program
            self.best_metrics_per_island[island_id] = metrics
            logging.info("Best score of island %d updated to %.2f", island_id, score)

    def _register_program(self, program: code_types.Function, island_id: int, metrics: Dict):
        with self._lock:
            score = _get_score(metrics, self.eval_metric_name)
            if island_id is None:  # start: seed all islands with program
                for i in range(self.config.num_islands):
                    self._register_program_in_island(program, i, metrics)
            else:
                self._register_program_in_island(program, island_id, metrics)

            if time.time() - self.last_reset_time > self.config.reset_period:
                self.reset_islands()
                self.last_reset_time = time.time()
            return
    
    def reset_islands(self):
        """Reset worst-performing half with individuals drawn from better half."""
        idx_sorted_by_score = np.argsort(self.best_score_per_island + NOISE_SCALE * np.random.randn(self.config.num_islands))
        reset_n = self.config.num_islands // 2
        good_pool = idx_sorted_by_score[reset_n:]
        reset_pool = idx_sorted_by_score[:reset_n]
        for i in reset_pool:
            self.islands[i] = Island(self.template, self.function_to_evolve, self.config.functions_per_prompt,
                                     self.config.cluster_sampling_temperature_init,
                                     self.config.cluster_sampling_temperature_period)
            seed_id = np.random.choice(good_pool)
            self.best_score_per_island[i] = -float('inf')
            self._register_program_in_island(self.best_program_per_island[seed_id], i, self.best_metrics_per_island[seed_id])

class Island:
    """Subpopulation of programs.
       Responsible for synthesising new prompts from programs."""
    
    def __init__(self, template, function_to_evolve: str, functions_per_prompt: int,
                 cluster_sampling_temperature_init: float = 0.1, cluster_sampling_temperature_period: int = 10000):
        self.template = template
        self.function_to_evolve = function_to_evolve
        self.functions_per_prompt = functions_per_prompt
        self.cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self.cluster_sampling_temperature_period = cluster_sampling_temperature_period

        self._n_programs = 0
        self._clusters = {}  # signature -> Cluster

    def _register_program(self, program, metrics):
        signature = _get_signature(metrics)
        if signature not in self._clusters:
            self._clusters[signature] = Cluster(signature, program)
        else:
            self._clusters[signature].register_program(program)
        self._n_programs += 1
        return

    def generate_prompt(self) -> tuple[str, int]:
        scores = np.array(list(self._clusters.keys()))
        period = self.cluster_sampling_temperature_period
        temperature = self.cluster_sampling_temperature_init * (1 - (self._n_programs % period) / period)
        
        # sample between clusters based on scores
        sample_probs = _softmax(scores, temperature)
        n_samples = min(self.functions_per_prompt, len(self._clusters))
        idx = np.random.choice(len(scores), size=n_samples, replace=False, p=sample_probs)
        sampled_scores = [scores[i] for i in idx]

        implementations = []
        for score in sampled_scores:
            _cluster = self._clusters[score]
            implementations.append(_cluster.sample_program())
        
        sorted_idx = np.argsort(sampled_scores)
        sorted_implementations = [implementations[i] for i in sorted_idx]
        version_generated = n_samples + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_prompt(self, implementations) -> str:
        # stitch together implementations into single prompt, ordered 
        # descending by score
        implementations = copy.deepcopy(implementations)
        versioned_functions = []
        for i, impl in enumerate(implementations):
            function_i_name = f'{self.function_to_evolve}_v{i}'
            impl.name = function_i_name
            if i >= 1:
                preface = f"Improved version of `{self.function_to_evolve}_v{i-1}`.\n"
                impl.docstring = textwrap.indent(preface, '    ') + impl.docstring if impl.docstring else preface
            versioned_functions.append(impl)

        version_num = len(versioned_functions)
        new_function_name = f'{self.function_to_evolve}_v{version_num}'
        current_program_prompt = implementations[-1].replace(
            name=new_function_name,
            docstring=f"Improved version of `{self.function_to_evolve}_v{version_num-1}`.\n",
            body="")
        versioned_functions.append(current_program_prompt)

        prompt = self.template.replace(
            functions=versioned_functions)
        return str(prompt)




class Cluster:
    """Programs on the same island with the same coarse--grained score."""

    def __init__(self, score, implementation: code_types.Function):
        self._score = score  # coarse--grained, not exact
        self._programs = [implementation]
        self._program_lengths = [len(str(implementation))]

    @property
    def score(self):
        return self._score

    def register_program(self, program):
        self._programs.append(program)
        self._program_lengths.append(len(str(program)))

    def sample_program(self, temperature: float = 1.) -> code_types.Function:
        # sample within cluster based on lengths
        # Assign higher weight to shorter programs.
        weights = np.array(self._program_lengths)
        norm_weights = (weights - np.min(weights)) / np.max(weights)
        probs = _softmax(-norm_weights, temperature)
        idx = np.random.choice(len(probs), p=probs)
        return self._programs[idx]