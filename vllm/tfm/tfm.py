from vllm import LLM, SamplingParams
from dotenv import load_dotenv
from lark import Lark, exceptions
from lark.indenter import Indenter
from random import randint
import time
import sympy as sp
import json
import os
import threading
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo

load_dotenv()
# Initialize NVML
nvmlInit()

# Function to monitor GPU usage
def monitor_gpu(device_id, stop_event, interval=0.01, results=None):
    # Handle to GPU
    handle = nvmlDeviceGetHandleByIndex(device_id)
    
    # Lists to store GPU usage metrics
    gpu_usage = []
    memory_usage = []
    
    start_time = time.time()
    while not stop_event.is_set():  # Continue until stop_event is set:
        # Get GPU utilization
        util = nvmlDeviceGetUtilizationRates(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        
        # Append data to lists
        gpu_usage.append(util.gpu)         # GPU usage in %
        memory_usage.append(mem_info.used / 1024 ** 2)  # Memory usage in MB
        
        # Wait for the next interval
        time.sleep(interval)

    if results is not None:
        results['gpu_usage'] = gpu_usage
        results['memory_usage'] = memory_usage


stop_event = threading.Event()

results = {}
gpu_id = 0
interval = 0.1

# Start the GPU load test and monitoring in parallel
monitor_thread = threading.Thread(target=monitor_gpu, args=(gpu_id, stop_event, interval, results))
monitor_thread.start()

# Model loading
SAMPLES = 100

# llama-3-70 quantized INT8
MODEL = "study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8"
llm = LLM(MODEL, gpu_memory_utilization=0.9, tensor_parallel_size=8, enforce_eager=True, quantization="gptq") 

# Llama-3-8B
#MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
#llm = LLM(MODEL, gpu_memory_utilization=1, tensor_parallel_size=8, enforce_eager=False, dtype="half") 

# llama-3.1-70B INT4
#MODEL = "hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"
#llm = LLM(MODEL, gpu_memory_utilization=1.0, tensor_parallel_size=8, enforce_eager=False, quantization="gptq", enable_chunked_prefill=False, cpu_offload_gb=70.0, max_model_len=10384) # LLama 3.1 70B

# Auxiliar functions
class Generation:
    def __init__(self, seed, elapsed_time, gen):
        """
        arrival_time: The time when the request arrived. 
         first_scheduled_time: The time when the request was first scheduled. 
         first_token_time: The time when the first token was generated. 
         time_in_queue: The time the request spent in the queue. 
         finished_time: The time when the request was finished. 
        """
        # Elapsed time its based on vLLM values finished_time - arrival_time
        self.seed = seed
        self.elapsed_time = elapsed_time
        self.gen = gen

class Result:
    def __init__(self, seeds: list[int], elapsed_time_gen: list[int], syntax_validation: list[int], semantic_validation: list[int]):
        self.model = MODEL.replace("/","::")
        self.experiment_type = EXPERIMENT_TYPE
        self.samples = SAMPLES
        self.max_tokens= MAX_TOKENS
        self.seeds = [seed for seed in seeds[:SAMPLES]]
        self.elapsed_time_gen = [elapsed for elapsed in elapsed_time_gen[:SAMPLES]]
        self.syntax_validation = syntax_validation
        self.semantic_validation = semantic_validation

    def save(self):
        with open(f"./results/{self.model}/{self.experiment_type}/{self.samples}e_{self.max_tokens}t.jsonl", 'w') as file:
            json_line = json.dumps(self.__dict__)
            file.write(json_line + "\n")
        
    
    def load(self):
        with open(f"./results/{self.model}/{self.experiment_type}/{self.samples}e_{self.max_tokens}t.jsonl", "r") as file:
            for line in file:
                results = json.loads(line, object_hook=result_encoder)
            return results
    
def result_encoder(r):
    return Result(model=r['model'], experiment_type=r['experiment_type'], samples=r['samples'], max_tokens=r['max_tokens'],
                    seeds=r['seeds'], syntax_validation= r['syntax_validation'], semantic_validation=r['semantic_validation'])

def gen_encoder(g):
    return Generation(seed=g['seed'], elapsed_time=g['elapsed_time'], gen=g['gen'])

def fixed_seeds():
    # Taking the 100 samples seed from the first 100 samples experiment
    fixed_seeds = []
    with open(f"./seeds/100.jsonl", "r") as file:
        for line in file:
            fixed_seeds = json.loads(line)
    return fixed_seeds

def semantic_test(generation: str):
    sp.sympify(generation)

def syntax_text(generation: str, parser):
    parser.parse(generation).pretty()

def gen_preproc(generation:str):
    eot_id_comparison = None
    match EXPERIMENT_TYPE:
        case "nogbd":
            # Treatment for returning generation until <|eot_id|>
            eot_id_gen = generation.split("<|eot_id|>")[0]
            # Treatment for replace =(comparision) with == (used in no gbd generation), As using simpify (Python code simulation)
            eot_id_comparison = eot_id_gen.replace("=", "==")

        case "gbd":
            # Treatment for returning generation until <|eot_id|>
            eot_id_comparison = generation.split("<|eot_id|>")[0]

        case "gbd+fewshots":
            # Treatment for returning generation until <|eot_id|>
            eot_id_comparison = generation.split("<|eot_id|>")[0]

    return eot_id_comparison

tokens = [50, 100, 200]
experiments = ["gbd","nogbd","gbd+fewshots"]

for experiment_type in experiments:

    EXPERIMENT_TYPE = experiment_type

    for token in tokens:

        print(f"Experiment type: {experiment_type}")

        MAX_TOKENS = token

        # Relevant changes: == instead of =, ("==" expression)* instead of ("==" expression)? to allow multiple comparisions
        arithmetic_grammar = """
        ?start: comparison

        ?comparison: expression ("==" expression)* "<|eot_id|>"?

        ?expression: term (("+" | "-") term)*

        ?term: factor (("*" | "/") factor)*

        ?factor: NUMBER
            | "-" factor
            | "(" comparison ")"

        %import common.NUMBER
        %ignore " "  // Ignore spaces

        // Define <|eot_id|> as a terminal
        EOT_ID: "<|eot_id|>"
        """

        arithmetic_prompt = None

        match EXPERIMENT_TYPE:
            case "gbd":
                arithmetic_prompt=f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for creating gramatically and sintactically arithmetic expression<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Rewrite 9 * 15 as others equivalents expressions:
        Follow this example:
        (5*5)=(5+5+5+5+5)=(25*1)=(5*3)+(5*2). 
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
                
            case "nogbd":
                arithmetic_prompt=f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for creating gramatically and sintactically arithmetic expression<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Rewrite 9 * 15 as other equivalent expression, for the response, do not use text.
        Just only characters available in this grammar: {arithmetic_grammar}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
                
            case "gbd+fewshots":
                arithmetic_prompt="""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for creating gramatically, equivalent and correct arithmetical expression<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Given the following examples:\n
        (5*5)=(5+5+5+5+5)=(25*1)=(5*3)+(5*2).\n
        (3*3)=(3+3+3)=(3+6)=(9*1).\n
        (3*4*5)=3*(2+2)*5=15*4=15*(2+2)=(12*5)=(20*3).\n
        Rewrite 9 * 15 as others equivalents expressions:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

            case "grammar_in_prompt":
                arithmetic_prompt=f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for creating gramatically and sintactically expression given this specific grammar: {arithmetic_grammar}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        Rewrite 9 * 15 as others equivalents expressions:
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """

        grammar = arithmetic_grammar

        # Samples generations
        #seeds = [randint(1,SAMPLES*10e9) for i in range(SAMPLES)]
        seeds = fixed_seeds()
        only_generations = []
        elapsed_time_gens = []

        # Iterate experiments to generate completions
        for i in range(SAMPLES):
            seed = seeds[i]

            sampling_params = SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=1,
            top_p=0.95,
            seed= seed
            )

            start_time = time.perf_counter()

            outputs = None

            match EXPERIMENT_TYPE:
                case "gbd":
                    outputs = llm.generate(
                        prompts=arithmetic_prompt,
                        sampling_params=sampling_params,
                        guided_options_request=dict(guided_grammar=grammar))
                    
                case "nogbd":
                    outputs = llm.generate(
                        prompts=arithmetic_prompt,
                        sampling_params=sampling_params,
                        )
                    
                case "gbd+fewshots":
                    outputs = llm.generate(
                        prompts=arithmetic_prompt,
                        sampling_params=sampling_params,
                        guided_options_request=dict(guided_grammar=grammar))

            elapsed_time = time.perf_counter() - start_time
            print(f'Elapsed time for generation nº{i}: {elapsed_time} seconds')

            # Stop event for the monitoring thread
            stop_event.set()

            elapsed_time = None
            gen_text = None
            try:
                elapsed_time = outputs[0].metrics.finished_time - outputs[0].metrics.arrival_time
            except:
                elapsed_time = outputs.metrics.finished_time - outputs.metrics.arrival_time
            
            gen_text = None
            try:
                gen_text = outputs[0].outputs[0].text
            except:
                gen_text = outputs.outputs[0].text
                            
            elapsed_time_gens.append(elapsed_time)
            only_generations.append(Generation(seed=seed, elapsed_time=elapsed_time, gen=gen_text))


        # Wait for monitoring to finish
        monitor_thread.join()

        # Output the collected GPU usage and memory usage data
        print("GPU Usage (%):", results["gpu_usage"])
        print("Memory Usage (MB):", results["memory_usage"])

        print("Max GPU usage: ", max(results["gpu_usage"]))
        print("Max Memory Usage (MB): ", max(results["memory_usage"]))

        model_foldername = MODEL.replace("/","::")
        os.makedirs(f"./results/{model_foldername}/{EXPERIMENT_TYPE}/gpu/", exist_ok=True)
        with open(f"./results/{model_foldername}/{EXPERIMENT_TYPE}/gpu/{SAMPLES}e_{MAX_TOKENS}t_gpu.jsonl", 'w') as file:
                json_line = json.dumps(results)
                file.write(json_line + "\n")

        # Save and load results
        # Write the jsonl and serialize the gens
        os.makedirs(f"./samples/{model_foldername}/{EXPERIMENT_TYPE}/", exist_ok=True)
        os.makedirs(f"./results/{model_foldername}/{EXPERIMENT_TYPE}/", exist_ok=True)

        
        with open(f"./samples/{model_foldername}/{EXPERIMENT_TYPE}/{SAMPLES}e_{MAX_TOKENS}t.jsonl", 'w') as file:
            for gen in only_generations:
                json_line = json.dumps(gen.__dict__)
                file.write(json_line + "\n")

        # Read the jsonl and deserialize back
        generation_from_file = []
        with open(f"./samples/{model_foldername}/{EXPERIMENT_TYPE}/{SAMPLES}e_{MAX_TOKENS}t.jsonl", "r") as file:
            for line in file:
                gen = json.loads(line, object_hook=gen_encoder)
                generation_from_file.append(gen)


        # Checking syntax
        syntactic_results=[]
        parser = Lark(grammar, parser='lalr')

        for gen in generation_from_file:

            try:
                # Parse a generation
                gen_preprocesed = gen_preproc(gen.gen)
                syntax_text(generation=gen_preprocesed, parser=parser)
                syntactic_results.append(1)

            except:
                syntactic_results.append(0)

        print(f"total syntactically valid: {syntactic_results.count(1)}" )
        print(f"total syntactically invalid: {syntactic_results.count(0)}" )

        print(f"Percentaje syntactically valid: {(syntactic_results.count(1)/SAMPLES) * 100}%" )


        # Checking semantic
        semantic_results=[]

        for gen in generation_from_file:

            try:
                # Parse a generation
                gen_preprocesed = gen_preproc(gen.gen)
                semantic_test(generation=gen_preprocesed)
                semantic_results.append(1)

            except:
                semantic_results.append(0)

        print(f"total semantically valid: {semantic_results.count(1)}" )
        print(f"total semantically invalid: {semantic_results.count(0)}" )

        print(f"Percentaje semantically valid: {(semantic_results.count(1)/SAMPLES) * 100}%" )

        res = Result(seeds=seeds, elapsed_time_gen= elapsed_time_gens, syntax_validation= syntactic_results, semantic_validation=semantic_results)
        res.save()