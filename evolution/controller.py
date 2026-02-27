"""
Example runner for the pub/sub flow: Sampler publishes heuristics (one per queue item),
Evaluator workers subscribe and validate.

Run from project root:
  python -m evolution.run_pubsub
Or from the evolution directory:
  cd evolution && python run_pubsub.py
  (ensure evolution is on PYTHONPATH or run from repo root)

Requires: LLM server (see sampler BASE_URL).
"""
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

if "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# When run as script from evolution/, add project root so "from evolution.*" resolves
_evolution_dir = Path(__file__).resolve().parent
_project_root = _evolution_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evolution import code_types, code_utils
from evolution.evaluator import Evaluator
from evolution.sampler import Sampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-22s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evolution.controller")


class MockDatabase:
    """Minimal database for the example: provides prompts and records results."""

    def __init__(self):
        self.registered = []

    def generate_prompts(self):
        # One prompt per generation for a short example.
        return ["Complete the heuristic_fn body to score a group presentation."]

    def register_evolved_function(self, candidate_id, evolved_function, scores):
        self.registered.append((candidate_id, evolved_function, scores))
        print(f"[DB] registered candidate_id={candidate_id} scores={scores}")


def load_template():
    scaffold_path = Path(__file__).parent / "context" / "scaffold.md"
    text = scaffold_path.read_text(encoding="utf-8")
    return code_utils.text_to_program(text)


# -- Pipeline configuration --------------------------------------------------
class config(object):
    NUM_SAMPLERS = 2          # LLM samplers (1 produces ~32 samples/30s)
    SAMPLES_PER_PROMPT = 4    # samples per LLM call
    GENERATION_BUDGET = 256     # total generations across all samplers before shutdown
    NUM_EVAL_WORKERS = 4      # GPU eval workers (~1 sample/20s each)
    QUEUE_MAXSIZE = 128       # back-pressure threshold
    BUDGET_POLL_INTERVAL = 5  # seconds between budget checks


async def run_controller(
    samplers: list,
    evaluator: Evaluator,
    database,
    queue: asyncio.Queue,
    stop_event: asyncio.Event,
    generation_budget: int,
    poll_interval: float = 5.0,
    stall_threshold: int = 3,
):
    """Monitor pipeline health and throughput. Trigger shutdown on budget.

    Periodically logs throughput metrics, detects stalled workers, and
    restarts dead GPU processes. Returns when generation_budget is reached.
    """
    prev_registered = 0
    prev_generations = 0
    prev_time = time.time()
    stall_count = 0

    while True:
        await asyncio.sleep(poll_interval)
        now = time.time()
        dt = now - prev_time

        total_generations = sum(s.generations_completed for s in samplers)
        n_registered = len(database.registered)
        in_flight = evaluator.in_flight
        alive = evaluator.healthy_workers

        # Throughput deltas
        gen_delta = total_generations - prev_generations
        evals_delta = n_registered - prev_registered
        samples_per_min = (gen_delta / dt) * 60 if dt > 0 else 0
        evals_per_min = (evals_delta / dt) * 60 if dt > 0 else 0

        logger.info(
            "gen=%d/%d registered=%d queue=%d in_flight=%d workers=%d/%d "
            "samples=%.1f/min evals=%.1f/min",
            total_generations, generation_budget,
            n_registered, queue.qsize(),
            in_flight, alive, evaluator._sandbox.num_workers,
            samples_per_min, evals_per_min,
        )

        # Stall detection: in-flight work but no new results for consecutive polls
        if in_flight > 0 and evals_delta == 0:
            stall_count += 1
            if stall_count >= stall_threshold:
                logger.warning(
                    "Pipeline stalled: %d in-flight, 0 evals for %.0fs",
                    in_flight, stall_count * poll_interval,
                )
        else:
            stall_count = 0

        # Worker health: restart pool if any workers have died
        if alive < evaluator._sandbox.num_workers:
            logger.warning(
                "%d dead workers detected, restarting pool",
                evaluator._sandbox.num_workers - alive,
            )
            evaluator._sandbox.restart_workers()

        prev_registered = n_registered
        prev_generations = total_generations
        prev_time = now

        # Budget check
        if total_generations >= generation_budget:
            logger.info(
                "Budget reached (%d generations). Shutting down.", total_generations
            )
            stop_event.set()
            return


async def main(config):
    template = load_template()
    database = MockDatabase()

    queue = asyncio.Queue(maxsize=config.QUEUE_MAXSIZE)
    shutdown_sentinel = code_types.HEURISTICS_QUEUE_SHUTDOWN
    stop_event = asyncio.Event()

    # Launch K concurrent samplers publishing to the same queue.
    # Expect eval side to bottleneck.
    samplers = [
        Sampler(
            database,
            heuristics_queue=queue,
            stop_event=stop_event,
            samples_per_prompt=config.SAMPLES_PER_PROMPT,
            sampler_id=k,
        )
        for k in range(config.NUM_SAMPLERS)
    ]

    evaluator = Evaluator(
        database,
        template,
        function_to_evolve="heuristic_fn",
        function_to_run="heuristic_fn",
        num_workers=config.NUM_EVAL_WORKERS,
    )

    # Start evaluator consumer
    evaluator_task = asyncio.create_task(
        evaluator.subscribe_and_evaluate(queue, shutdown_sentinel)
    )

    # Start all persistent samplers
    sampler_tasks = [asyncio.create_task(s.sample()) for s in samplers]

    # Controller monitors throughput and health until budget is exhausted
    await run_controller(
        samplers, evaluator, database, queue, stop_event,
        generation_budget=config.GENERATION_BUDGET,
        poll_interval=config.BUDGET_POLL_INTERVAL,
    )

    # Wait for samplers to finish their current iteration
    await asyncio.gather(*sampler_tasks)

    # All samplers done — signal evaluator to drain remaining work and stop
    await queue.put(shutdown_sentinel)
    await evaluator_task

    evaluator.shutdown()
    total_generations = sum(s.generations_completed for s in samplers)
    logger.info(
        "Done. Registered %d results from %d generations.",
        len(database.registered), total_generations,
    )


if __name__ == "__main__":
    asyncio.run(main(config))
