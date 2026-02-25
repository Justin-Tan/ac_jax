import asyncio
import time
from openai import AsyncOpenAI

from typing import Sequence, Tuple

from code_parser import Evaluator

# Configuration
BATCH_SIZE = 32
BASE_URL = "http://localhost:8003/v1/"
MODEL_NAME = "unsloth/Qwen3-Coder-Next"
MAX_MODEL_LEN = 8192  # must agree with server config

MAX_TOKENS = 2048
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 40
MIN_P = 1e-2
REP_PENALTY = 1.0


class LLM:
    def __init__(self, samples_per_prompt=BATCH_SIZE,
                 model_name=MODEL_NAME, base_url=BASE_URL,
                 instructions_path="context/instructions_AC.md"):
        self.client = AsyncOpenAI(base_url=base_url)
        self.model_name = model_name
        self.samples_per_prompt = samples_per_prompt

        with open(instructions_path, "r", encoding="utf-8") as f:
            self._INSTRUCTIONS = f.read()

    async def _generate_sample(self, prompt: str) -> Tuple[float, str]:
        """Single--sample generation."""

        start = time.time()
        try:
            result = await self.client.responses.create(
                model=self.model_name,
                instructions=self._INSTRUCTIONS,
                input=prompt,
                max_tokens=MAX_MODEL_LEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                extra_body = {"top_k": TOP_K, "min_p": MIN_P,
                            "repetition_penalty": REP_PENALTY,}
            )
            latency = time.time() - start
            return result.output.text, latency
        except Exception as e:
            print(f"Error generating sample: {e}")
            return None, time.time() - start

    async def generate_samples(self, prompts: Sequence[str]) -> Tuple[float, Sequence[str]]:
        """Multi--sample generation."""
        tasks = [self._generate_sample(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        samples, latencies = zip(*results)
        return samples, latencies


class Sampler:
    """Interfaces with database. Enqueues procedurally generated prompts in queue, receives
    asynchronous generations from LLMs, registers them in database."""
    def __init__(self, database, template, samples_per_prompt=BATCH_SIZE,
                 max_generations=256, heuristics_queue=None, shutdown_sentinel=None):
        self.database = database
        self.max_generations = max_generations
        self.llm = LLM(samples_per_prompt=samples_per_prompt)
        self._heuristics_queue = heuristics_queue
        self._shutdown_sentinel = shutdown_sentinel
        if heuristics_queue is None:
            self.evaluator = Evaluator(database, template, function_to_evolve="heuristic_fn",
                                       function_to_run="heuristic_fn")

    async def sample(self):
        """Coordinates sampling/evaluation/database interaction."""
        _generation = 0
        pending_eval = None
        while _generation < self.max_generations:

            prompts = self.database.generate_prompts()
            samples, latencies = await self.llm.generate_samples(prompts)

            if self._heuristics_queue is not None:
                # Pub/sub: publish one sample per queue item for evaluator workers.
                for i, sample in enumerate(samples):
                    if sample is not None:
                        await self._heuristics_queue.put((_generation, i, sample))
            else:
                # Original: wait for previous eval, then run this batch.
                if pending_eval is not None:
                    await pending_eval
                pending_eval = asyncio.create_task(self.evaluator.analyse(samples))

            _generation += 1

        if self._heuristics_queue is not None and self._shutdown_sentinel is not None:
            await self._heuristics_queue.put(self._shutdown_sentinel)
