from openai import OpenAI
from pprint import pprint

BASE_URL = "http://localhost:8003/v1/"
MODEL_NAME = "unsloth/Qwen3-Coder-Next"
MAX_MODEL_LEN = 8192  # must agree with server config
MAX_TOKENS = 2048
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 40
MIN_P = 1e-2
REP_PENALTY = 1.0

# _TEST_PROMPT = r"Write a pure JAX function to compute the generalised advantage estimate given a rollout trajectory, for general $\lambda$ and $\gamma$."
# _TEST_PROMPT = r"Write a pure JAX function which implements the flashattention logic."
_TEST_PROMPT = r"Write a pure JAX function which computes the nth Fibonacci number using memoisation."
_INSTRUCTIONS = "You are a Python coding assistant with deep expertise in JAX."

client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY"
)

response = client.responses.create(
    model=MODEL_NAME,
    instructions=_INSTRUCTIONS,
    input=_TEST_PROMPT,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    extra_body = {"top_k": TOP_K, "min_p": MIN_P, "repetition_penalty": REP_PENALTY,}
)

# response = client.chat.completions.create(
#     model=MODEL_NAME,
#     messages=[
#         {"role": "system", "content": },
#         {"role": "user", "content": _TEST_PROMPT}
#     ],
#     max_tokens=MAX_TOKENS,
#     temperature=TEMPERATURE,
#     top_p=TOP_P,
#     extra_body = {"top_k": TOP_K, "min_p": MIN_P, "repetition_penalty": REP_PENALTY,}
# )

print("Model output:")
print(response.output_text)
# print(response.choices[0].message.content)