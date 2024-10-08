from vllm import LLM, SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from pydantic import BaseModel, conlist
import datetime as dt

# from https://github.com/vllm-project/vllm/issues/3087


# Json Schema
class Output(BaseModel):
    names: conlist(str, max_length=5)
    organizations: conlist(str, max_length=5)
    places: conlist(str, max_length=5)
    locations: conlist(str, max_length=5)
    prepositions: conlist(str, max_length=5)

llm = LLM('study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8', gpu_memory_utilization=0.9, dtype="half", tensor_parallel_size=8, enforce_eager=True, quantization="gptq")
logits_processor = JSONLogitsProcessor(schema=Output, llm=llm.llm_engine)
prompt = """
Locate all the names, organizations, locations and other miscellaneous entities in the following sentence: 
"Charles went and saw Anna at the coffee shop Starbucks, which was based in a small town in Germany called Essen."
"""
sampling_params = SamplingParams(max_tokens=500, temperature=0, logits_processors=[logits_processor])

t0 = dt.datetime.now()
outputs = llm.generate(prompt, sampling_params=sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

time_elapsed = (dt.datetime.now() - t0).total_seconds()
print(f"Generation took {time_elapsed:,} seconds.")