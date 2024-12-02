from vllm import LLM, SamplingParams
from dotenv import load_dotenv

load_dotenv()

# llama-3-70 quantized
#llm = LLM('study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8', gpu_memory_utilization=0.9, tensor_parallel_size=8, enforce_eager=False, quantization="gptq")
#llm = LLM('meta-llama/Llama-3.2-1B-Instruct' enforce_eager=False, dtype="half")
llm = LLM("meta-llama/Llama-3.2-3B-Instruct", enforce_eager=False, dtype="half")


import random
random_seed = random.randint(0, 10000)
random.seed(random_seed)

print(f"seed: {random_seed}")

uvl_grammar_no_indent = r"""
?start: _NL* featuremodel

featuremodel: "features" _NL [feature+]
feature: NAME _NL (group+)?

group: "or" groupspec          -> or_group
    | "alternative" groupspec -> alternative_group
    | "optional" groupspec    -> optional_group
    | "mandatory" groupspec   -> mandatory_group
    | cardinality groupspec    -> cardinality_group

groupspec: _NL feature+

cardinality: "[" INT (".." (INT | "*"))? "]"

%import common.INT
%import common.CNAME -> NAME
%import common.WS_INLINE
%ignore WS_INLINE

_NL: /\r?\n[ \t]*/
"""

uvl_grammar = r"""
?start: _NL* featuremodel

featuremodel: "features" _NL ["->" feature+ "<-"]
feature: NAME _NL ("->" group+ "<-")?

group: "or" groupspec          -> or_group
    | "alternative" groupspec -> alternative_group
    | "optional" groupspec    -> optional_group
    | "mandatory" groupspec   -> mandatory_group
    | cardinality groupspec    -> cardinality_group

groupspec: _NL "->" feature+ "<-"

cardinality: "[" INT (".." (INT | "*"))? "]"

%import common.INT
%import common.CNAME -> NAME
%import common.WS_INLINE
%ignore WS_INLINE

_NL: /\r?\n[ \t]*/
"""

uvl_prompt=f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for creating sintactically correct expressions similar to these examples:

features
    Sandwich
        mandatory
            Bread
        optional
            Sauce
                alternative
                    Ketchup
                    Mustard
            Cheese
constraints
    Ketchup => Cheese
,
features
	SmartWatch
		mandatory
			Functionalities 
				mandatory
					FitnessMonitor
					SleepTracker
					VibrateAlert
						mandatory
							Call
							Notification
			Sensors
				mandatory
					Pedometer
					Accelerometer
				optional
					HeartRateSensor
			Connectivity
				mandatory
					BT40


<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Write new, valid expressions for a computer: 
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


import time
start_time = time.perf_counter()
grammar = uvl_grammar_no_indent
prompt = uvl_prompt
# llama-3 8B
#MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
#llm = LLM(MODEL, gpu_memory_utilization=1, tensor_parallel_size=8, enforce_eager=False, dtype="half") 


sampling_params = SamplingParams(
        max_tokens=200,
        temperature=1,
        top_p=0.95,
        min_tokens=200,
    )


outputs = llm.generate(
    prompts=prompt,
    sampling_params=sampling_params,
    guided_options_request=dict(guided_grammar=grammar))

elapsed_time = time.perf_counter() - start_time
print(f'Elapsed time for inference: {elapsed_time} seconds')

from lark import Lark, exceptions
from lark.indenter import Indenter


def test(generation: str, parser):
    print(parser.parse(generation).pretty())

"""
class TreeIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8
"""

#parser = Lark(grammar, parser='lalr', postlex=TreeIndenter())
parser = Lark(grammar, parser='lalr')

print("Grammar is well-written.")

for output in outputs:
    prompt = output.prompt

    generated_text = output.outputs[0].text
    print(f"Generated text without parser: {generated_text!r}")

    try:
        # Parse a generation
        test(generation=generated_text, parser=parser)

    except:
        print("Generation not grammatically valid")

print(f"nº of generated tokens: {str(len(outputs[0].outputs[0].token_ids))}")
print(f"generated tokens id: {outputs[0].outputs[0].token_ids}")


gen = outputs[0].outputs[0].text
print(f"{str(gen)}")

print(parser.parse(gen).pretty())