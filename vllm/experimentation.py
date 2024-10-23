from vllm import LLM, SamplingParams
from dotenv import load_dotenv

load_dotenv()

uvl_grammar = r"""
?start: _NL* featuremodel

featuremodel: "features" _NL [_INDENT feature+ _DEDENT]
feature: NAME _NL (_INDENT group+ _DEDENT)?

group: "or" groupspec          -> or_group
| "alternative" groupspec -> alternative_group
| "optional" groupspec    -> optional_group
| "mandatory" groupspec   -> mandatory_group
| cardinality groupspec   -> cardinality_group

groupspec: _NL _INDENT feature+ _DEDENT

cardinality: "[" INT (".." (INT | "*"))? "]"

%import common.INT
%import common.CNAME -> NAME
%import common.WS_INLINE
%declare _INDENT _DEDENT
%ignore WS_INLINE

_NL: /(\r?\n[\t ]*)+/
"""

uvl_prompt=f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for creating gramatically and sintactically expression given this specific grammar: {uvl_grammar}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Write new expressions separated by \n:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

arithmetic_grammar = """
?start: comparison

?comparison: expression ("=" expression)?

?expression: term (("+" | "-") term)*

?term: factor (("*" | "/") factor)*

?factor: NUMBER
       | "-" factor
       | "(" comparison ")"

%import common.NUMBER
%ignore " "  // Ignore spaces
"""

arithmetic_prompt="Rewrite 5*5 as another expression"

arithmetic_prompt_fewshots="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for creating gramatically, equivalent and correct arithmetical expression<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Rewrite the following expressions into equivalent ones as show in this example(5*5)=(5+5+5+5+5)=(25*1)=(5*3)+(5*2).\n(3*3)=\n(3*4*5)=\n(7*3)=
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

sql_grammar="""
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
"""
sql_prompt="Generate a sql state that select col_1 from table_1 where it is equals to 1"

import time
start_time = time.perf_counter()
grammar = arithmetic_grammar
# llama-3-70 quantized
llm = LLM('study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8', gpu_memory_utilization=0.9, tensor_parallel_size=8, enforce_eager=False, quantization="gptq")
#llm = LLM('meta-llama/Llama-3.2-1B-Instruct', gpu_memory_utilization=0.9, tensor_parallel_size=8, enforce_eager=False, dtype="half")

sampling_params = SamplingParams(
        max_tokens=10,
        temperature=1,
        top_p=0.95,
    )

outputs = llm.generate(
    prompts=arithmetic_prompt,
    sampling_params=sampling_params,
    guided_options_request=dict(guided_grammar=grammar))

elapsed_time = time.perf_counter() - start_time
print(f'Elapsed time for inference: {elapsed_time} seconds')

from lark import Lark, exceptions
from lark.indenter import Indenter


def test(generation: str, parser):
    print(parser.parse(generation).pretty())

class TreeIndenter(Indenter):
    NL_type = '_NL'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8

parser = Lark(grammar, parser='lalr', postlex=TreeIndenter())
print("Grammar is well-written.")

print(f"len for outputs{str(len(outputs))}")
print(f"outputs{str(outputs[0].outputs[0].text)}")
print(f"outputs{str(outputs[0])}")


for output in outputs:
    prompt = output.prompt

    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text without parser: {generated_text!r}")

    try:
        # Parse a generation
        test(generation=generated_text, parser=parser)

    except:
        print("Generation not grammatically valid")

