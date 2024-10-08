from outlines import models, generate
from vllm import LLM, SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
import datetime as dt

# Outdated vllm not using Outlines

arithmetic_grammar = """
    ?start: sum

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: atom
        | product "*" atom  -> mul
        | product "/" atom  -> div

    ?atom: NUMBER           -> number
         | "-" atom         -> neg
         | "(" sum ")"

    %import common.NUMBER
    %import common.WS_INLINE

    %ignore WS_INLINE
"""


model = models.vllm('study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8', gpu_memory_utilization=0.9, dtype="half", tensor_parallel_size=8, enforce_eager=True, quantization="gptq")
t0 = dt.datetime.now()
generator = generate.cfg(model, arithmetic_grammar)

result = generator("Question: How can you write 5*5 using addition?\nAnswer:")
print(result)
# 5+5+5+5+5

time_elapsed = (dt.datetime.now() - t0).total_seconds()
print(f"Generation took {time_elapsed:,} seconds.")


