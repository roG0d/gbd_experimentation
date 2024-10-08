from vllm import LLM, SamplingParams

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

llm = LLM('study-hjt/Meta-Llama-3-70B-Instruct-GPTQ-Int8', gpu_memory_utilization=0.9, tensor_parallel_size=8, enforce_eager=False, quantization="gptq")
sampling_params = SamplingParams(
        max_tokens=100,
        temperature=1,
        top_p=0.95,
    )
outputs = llm.generate(
    prompts=arithmetic_prompt_fewshots,
    sampling_params=sampling_params,
    guided_options_request=dict(guided_grammar=arithmetic_grammar))

for output in outputs:
    prompt = output.prompt

    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text without parser: {generated_text!r}")

    """
    # use Lark to parse the output, and make sure it's a valid parse tree
    from lark import Lark
    parser = Lark(arithmetic_grammar)
    parser.parse(generated_text)

    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    """

