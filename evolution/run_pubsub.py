"""
Example runner for the pub/sub flow: Sampler publishes heuristics (one per queue item),
Evaluator workers subscribe and validate.

Run from project root:
  python -m evolution.run_pubsub
Or from the evolution directory:
  cd evolution && python run_pubsub.py
  (ensure evolution is on PYTHONPATH or run from repo root)

Requires: LLM server (see sampler BASE_URL), and optional GPU for evaluator sandbox.
"""
import asyncio
import os
import sys
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


async def main():
    template = load_template()
    database = MockDatabase()

    queue = asyncio.Queue(maxsize=64)
    shutdown_sentinel = code_types.HEURISTICS_QUEUE_SHUTDOWN

    sampler = Sampler(
        database,
        template,
        samples_per_prompt=2,
        max_generations=2,
        heuristics_queue=queue,
        shutdown_sentinel=shutdown_sentinel,
    )
    evaluator = Evaluator(
        database,
        template,
        function_to_evolve="heuristic_fn",
        function_to_run="heuristic_fn",
        num_workers=2,
    )

    evaluator_task = asyncio.create_task(
        evaluator.subscribe_and_evaluate(queue, shutdown_sentinel)
    )
    await sampler.sample()
    await evaluator_task

    evaluator.shutdown()
    print(f"[Done] Registered {len(database.registered)} results.")


if __name__ == "__main__":
    asyncio.run(main())
