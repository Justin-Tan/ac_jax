import asyncio
import os
import time
import jax.numpy as jnp
from unittest.mock import MagicMock

# Ensure this is set in the main process too, though it's critical for workers
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from evolution.code_parser import Evaluator, Sandbox
from evolution import code_types

# Mock Database
class MockDatabase:
    def __init__(self):
        self.programs = []
    
    def register_program(self, function, island_id, scores):
        print(f"Registering program: {function.name} with scores: {scores}")
        self.programs.append((function, island_id, scores))

async def main():
    print("Starting Sandbox Concurrency Test...")
    
    # 1. Setup minimal mocks
    mock_db = MockDatabase()
    
    # Create a minimal template program
    template_str = """
import jax.numpy as jnp

def heuristic(x):
    return jnp.sum(x)
"""
    # We need a valid Program object. 
    # Since we can't easily construct the full AST structure required by code_utils without existing files,
    # let's try to minimaly mock the pieces needed by Evaluator.analyse
    
    # Evaluator.analyse calls _sample_to_program
    # _sample_to_program calls copy.deepcopy(template) and template.get_function
    
    # Let's bypass the full AST complexity by mocking the template object
    mock_template = MagicMock()
    mock_function_obj = MagicMock()
    mock_function_obj.body = ""
    mock_template.get_function.return_value = mock_function_obj
    mock_template.__str__.return_value = template_str # This is what gets executed if we don't change it

    # However, Sandbox receives `program` string from _sample_to_program 
    # which is `str(program)`.
    # And Evaluator.analyse reconstructs it.
    
    # Let's use the real classes if possible to avoid integration issues later.
    # But for a quick test of the *Sandbox* logic specifically, we can invoke Sandbox directly first.
    
    print("\n--- Test 1: Direct Sandbox.run (JAX Computation) ---")
    sandbox = Sandbox(num_workers=4)
    
    program_code = """
import jax.numpy as jnp
import time

def run_me(x):
    # Simulate work
    y = jnp.array([x, x*2.0])
    return jnp.sum(y)
"""
    
    t0 = time.time()
    futures = []
    # Launch 4 concurrent tasks
    for i in range(4):
        futures.append(sandbox.run(program_code, "run_me", float(i), timeout_seconds=5))
    
    results = await asyncio.gather(*futures)
    dt = time.time() - t0
    
    print(f"Results: {results}")
    print(f"Time taken: {dt:.4f}s")
    
    # Verify results
    for i, (res, success) in enumerate(results):
        expected = i + i*2.0
        assert success, f"Task {i} failed"
        assert abs(res - expected) < 1e-5, f"Task {i} value mismatch: got {res}, exp {expected}"
        assert isinstance(res, float), f"Task {i} type mismatch: {type(res)}"

    print("Direct Sandbox Test Passed!")
    
    print("\n--- Test 2: OOM / Isolation Check ---")
    # Launch 8 tasks that allocate some memory (small enough for 4GB GPU split, but check concurrent overhead)
    program_memory = """
import jax
import jax.numpy as jnp

def memory_hog(n):
    # Allocate something that might crash if preallocation was on for everyone
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    return jnp.sum(x).item()
"""
    futures = []
    for i in range(8):
        futures.append(sandbox.run(program_memory, "memory_hog", 0, timeout_seconds=10))
        
    results = await asyncio.gather(*futures)
    success_count = sum(1 for r, s in results if s)
    print(f"Successful memory tasks: {success_count}/8")
    
    sandbox.close()

if __name__ == "__main__":
    asyncio.run(main())
