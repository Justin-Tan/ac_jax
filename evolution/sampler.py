import asyncio
import time
from openai import AsyncOpenAI

from typing import Sequence, Tuple

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
    """Produces LLM samples and publishes them to the heuristics queue.

    Interfaces with the database for prompt generation. Multiple Sampler
    instances can publish to the same queue concurrently to keep the
    evaluation pipeline saturated.
    """
    def __init__(self, database, heuristics_queue: asyncio.Queue,
                 stop_event: asyncio.Event,
                 samples_per_prompt: int = BATCH_SIZE,
                 sampler_id: int = 0):
        self.database = database
        self.llm = LLM(samples_per_prompt=samples_per_prompt)
        self._heuristics_queue = heuristics_queue
        self._stop_event = stop_event
        self._sampler_id = sampler_id
        self.generations_completed: int = 0

    async def sample(self):
        """Generate samples and publish to queue indefinitely.

        Shutdown signalling is responsibility of orchestrator. Checks for 
        stop signal between LLM generations.
        """
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            # Offload to thread pool so the event loop isn't blocked while
            # the database builds prompts (CPU-bound as DB grows).
            prompts = await loop.run_in_executor(None, self.database.generate_prompts)
            samples, latencies = await self.llm.generate_samples(prompts)

            for i, sample in enumerate(samples):
                if sample is not None:
                    await self._heuristics_queue.put(
                        (self.generations_completed, i, sample))
            self.generations_completed += 1