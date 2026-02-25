import ast, re, os, time
import multiprocessing
import concurrent.futures
import asyncio
import traceback

if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from copy import copy
from typing import Sequence, Any, Dict
_SHUTDOWN = "___SHUTDOWN___"

# custom imports
from evolution import code_types, code_utils, code_parser

N_SIMULATIONS = 16
MAX_DEPTH = 4
INITIAL_POOL = "/home/jt796/github/ac-jax/data/initial_presentations.npy"
HORIZON_LENGTH = 256
WORKER_MEM_FRACTION = 0.01

def _sandbox_worker_eval(
    task: code_types.Task,
    search_evaluator,
    config: dict
) -> Dict:
    """
    Worker function to execute untrusted code in a separate process.
    Executed in parallel on accelerator.
    """
    import jax
    import jax.numpy as jnp

    result = {"score": None, "complete": False, "error": None}
    program, function_to_run = task.program, task.function_to_run
    key = jax.random.PRNGKey(task.seed)
    try:
        # Create a local namespace execution context
        safe_globals = {"jax": jax, "jnp": jnp}
        local_scope = {}

        # Execute the program string to define functions
        exec(str(program), safe_globals, local_scope)

        if function_to_run not in local_scope:
            result["error"] = f"Function '{function_to_run}' not found."
            return result

        evolved_heuristic_fn = local_scope[function_to_run]

        # swap heuristic (no recompile)
        search_evaluator._heuristic_cb.update(evolved_heuristic_fn)
        metrics = search_evaluator._evaluate_batch_mcts_custom_heuristic_callback(key,
            n_simulations=config["n_simulations"], max_depth=config["max_depth"])

        result["scores"] = {k: float(v.block_until_ready()) for k, v in metrics.items()}
        result["complete"] = True
        return result

    except Exception:
        result["error"] = f"Sandbox execution failed: {traceback.format_exc()}"
        return result

def _evaluator_loop(task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue,
                    config: dict):
    """
    Persistent worker function running in single spawned process. Initialises evaluator class and pulls
    jobs from task queue.
    """
    # Disable JAX preallocation in worker to allow multiple processes
    # to share GPU memory. Must be set BEFORE any JAX import (or re-import).
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(WORKER_MEM_FRACTION)

    # We must import jax/numpy here to ensure they pick up the env var
    # inside this fresh process.
    import jax
    import jax.numpy as jnp
    from evolution import mcts_evaluation

    # Setup
    print(f"[EvalWorker {os.getpid()}] Initialising Evaluator...")
    evaluator = mcts_evaluation.SearchEvaluator(config["initial_pool_path"], config["horizon_length"])
    evaluator._heuristic_cb.update(evaluator.heuristic_fn)

    init_key = jax.random.PRNGKey(42)
    print(f"[EvalWorker {os.getpid()}] JIT--compling evaluation function...")
    _ = evaluator._evaluate_batch_mcts_custom_heuristic_callback(init_key,
        n_simulations=config["n_simulations"], max_depth=config["max_depth"])
    print(f"[EvalWorker {os.getpid()}] Ready.")
    # End setup

    while True:
        task = task_queue.get()

        if task == _SHUTDOWN:
            print(f"[Eval worker {os.getpid()}] Shutting down.")
            break

        result = _sandbox_worker_eval(task, evaluator, config)
        result_queue.put((task.candidate_id, result))


class Sandbox:
    """
    Sandbox for executing generated code in parallel. Manages N persistent evaluator processes.

    Each process holds its own evaluator class with fixed VRAM consumption.
    Tasks are distributed round-robin or via queue contention.
    Workers can be killed and restarted without affecting the main process.
    """

    def __init__(
        self,
        num_workers: int,
        initial_pool_path: str = INITIAL_POOL,
        horizon_length: int = HORIZON_LENGTH,
        n_simulations: int = N_SIMULATIONS,
        max_depth: int | None = MAX_DEPTH,
        timeout_eval: float = 32.,
    ):
        self.num_workers = num_workers
        self._config = {
            "initial_pool_path": initial_pool_path,
            "horizon_length": horizon_length,
            "n_simulations": n_simulations,
            "max_depth": max_depth,
            "timeout_seconds": timeout_eval,
        }

        # Shared queues — all workers compete for tasks from the same queue
        ctx = multiprocessing.get_context("spawn")
        self._task_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._processes: list[multiprocessing.Process] = []

        self._start_workers(ctx)

    def _start_workers(self, ctx):
        for i in range(self.num_workers):
            p = ctx.Process(
                target=_evaluator_loop,
                args=(self._task_queue, self._result_queue, self.config),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            print(f"[EvalWorkerPool] Started worker {i} (pid={p.pid})")

    def collect(self) -> dict:
        """Blocking: wait for one result. Returns the result dict."""
        return self._result_queue.get(timeout=self.config["timeout_seconds"])

    async def submit_batch_and_collect(
        self,
        candidates: list[code_types.Task],
    ) -> list[dict]:
        """
        Submit a batch of (candidate_id, program_str, function_name, rng_seed)
        and asynchronously collect all results.
        """
        loop = asyncio.get_running_loop()

        # Enqueue all candidates
        for task in candidates:
            self._task_queue.put(task)

        # Collect results without blocking the event loop
        results = []
        for _ in range(len(candidates)):
            result = await loop.run_in_executor(
                None,  # default thread pool
                self._result_queue.get,
                True,  # block=True
                self.config["timeout_seconds"],
            )
            results.append(result)

        return results

    def kill_workers(self):
        """Force-kill all workers immediately to reclaim VRAM."""
        for p in self._processes:
            if p.is_alive():
                p.kill()  # SIGKILL — immediate, no cleanup
                print(f"[EvalWorkerPool] Killed worker pid={p.pid}")
        self._processes.clear()

    def restart_workers(self):
        """Kill existing workers and start fresh ones."""
        self.kill_workers()
        ctx = multiprocessing.get_context("spawn")
        # Drain stale items from queues
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
            except Exception:
                break
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Exception:
                break
        self._start_workers(ctx)

    def shutdown(self):
        """Graceful shutdown: tell workers to exit, then join."""
        for _ in self._processes:
            self._task_queue.put(_SHUTDOWN)
        for p in self._processes:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()
        self._processes.clear()
        print("[EvalWorkerPool] All workers shut down.")


class Evaluator:
    """
    Parses raw string generated by LLM and scores resulting program via execution
    on test suite in sandbox.
    """
    def __init__(
        self, database, template: code_types.Program, function_to_evolve: str, function_to_run: str,
        inputs: Sequence[Any] = None, timeout_seconds: int = 60, num_workers: int = 8):
        """
        template: base program scaffold (imports, helper fundefs, target docstring)
        function_to_evolve: name of function to be evolved by LLM
        function_to_run: name of function to be executed for scoring, wrapper around function_to_evolve
        inputs: test cases used for evaluation and scoring.
        num_workers: number of concurrent sandbox processes. Adjust based on VRAM.
        """
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = Sandbox(num_workers=num_workers, timeout_eval=timeout_seconds)

    async def analyse(self, samples, candidate_id_offset: int = 0):
         """Compiles the sample into a program and executes it on test inputs."""
         tasks = []
         task_to_function = {}
         for i, sample in enumerate(samples):
            try:
                evolved_function, program = code_parser._sample_to_program(sample,
                    None, self._template, self._function_to_evolve)
                seed = int(time.time() % (2**32 - i))
                task = code_types.Task(candidate_id=candidate_id_offset + i,
                                       program=program,
                                       function_to_run=self._function_to_run,
                                       seed=seed)
                tasks.append(task)
                task_to_function[candidate_id_offset + i] = evolved_function
            except (ValueError, SyntaxError):
                print(f"Failed to parse sample {i}: {sample}")
                continue

         results = await self._sandbox.submit_batch_and_collect(tasks)
         for candidate_id, result in results:
             if candidate_id is None or not result.get("complete", False):
                 print(f"Candidate {candidate_id} failed evaluation: {result.get('error', 'Unknown error')}")
                 continue
             evolved_function = task_to_function.get(candidate_id)
             if evolved_function is not None:  # register evolved function plus score in db
                 self._database.register_evolved_function(candidate_id, evolved_function, result["scores"])

    async def subscribe_and_evaluate(self, heuristics_queue: asyncio.Queue, shutdown_sentinel=None):
        """
        Subscribe to the heuristics queue: pull (generation_id, sample_index, sample) items
        (one sample per item). Run analyse([sample]) for each. Exit when shutdown_sentinel
        is received.
        """
        while True:
            item = await heuristics_queue.get()
            if item is shutdown_sentinel:
                break
            generation_id, sample_index, sample = item
            try:
                await self.analyse([sample], candidate_id_offset=sample_index)
            except Exception as e:
                print(f"Evaluator: analyse failed for generation {generation_id} sample {sample_index}: {e}")
                continue

    def shutdown(self):
        self._sandbox.shutdown()
