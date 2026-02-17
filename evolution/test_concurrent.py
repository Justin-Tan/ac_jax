import asyncio
import time
from openai import AsyncOpenAI
import numpy as np

import jax.numpy as jnp

# Configuration
CONCURRENCY = 16
BASE_URL = "http://localhost:8003/v1/"
MODEL_NAME = "unsloth/Qwen3-Coder-Next"
MAX_MODEL_LEN = 8192  # must agree with server config

MAX_TOKENS = 2048
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 40
MIN_P = 1e-2
REP_PENALTY = 1.0

with open("evolution/instructions_AC.md", "r", encoding="utf-8") as f: 
    _INSTRUCTIONS = f.read()

client = AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")

def heuristic_fn_v0(observation: jnp.ndarray) -> float:
    return -(jnp.shape(observation)[-1] - 2.)

async def get_heuristic(i):
    start = time.time()
    _CONTEXT_i = f"Modify the function `heuristic_fn_v0` to a more suitable heuristic, adhering to the signature and renaming the function to `heuristic_fn_v{i}`."
    try:
        await client.responses.create(
            model=MODEL_NAME,
            instructions=_INSTRUCTIONS,
            input=_CONTEXT_i,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            extra_body = {"top_k": TOP_K, "min_p": MIN_P, "repetition_penalty": REP_PENALTY,}
        )
        latency = time.time() - start
        return latency
    except Exception as e:
        print(f"Request {i} failed: {e}")
        return None

async def main():
    print(f"Firing {CONCURRENCY} parallel requests...")
    start_total = time.time()
    
    # Launch all requests effectively "at once"
    tasks = [get_heuristic(i) for i in range(CONCURRENCY)]
    latencies = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_total
    valid_latencies = [l for l in latencies if l is not None]
    
    print(f"\n--- Results on GH200 ---")
    print(f"Total Wall Time:  {total_time:.4f}s")
    print(f"Avg Latency:      {np.mean(valid_latencies):.4f}s")
    print(f"Est. Throughput:  {CONCURRENCY / total_time:.2f} requests/sec")

if __name__ == "__main__":
    asyncio.run(main())