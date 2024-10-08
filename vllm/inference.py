
from vllm import LLM, SamplingParams
from transformers import BitsAndBytesConfig
from dotenv import load_dotenv


load_dotenv()


prompts = [
    "Create a salute as 'Avē Imperātor, moritūrī tē salūtant.' but for Mark Zuckerberg and It's open source AI attitude. Do it in latin"
]
sampling_params = SamplingParams(max_tokens=500, temperature=1, top_p=0.95)

llm = LLM(model="study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8", dtype="half", tensor_parallel_size=8, enforce_eager=True, quantization="gptq")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
