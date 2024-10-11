from vllm import LLM, SamplingParams

uvl_grammar = """
?start: featureModel

featureModel: namespace? _NEWLINE? includes? _NEWLINE? imports? _NEWLINE? features? _NEWLINE? constraints? EOF

includes: "include" _NEWLINE _INDENT includeLine* _DEDENT
includeLine: languageLevel _NEWLINE

namespace: "namespace" reference

imports: "imports" _NEWLINE _INDENT importLine* _DEDENT
importLine: reference ("as" reference)? _NEWLINE -> import_line

features: "features" _NEWLINE _INDENT feature _DEDENT

group
    : "or" groupSpec          -> or_group
    | "alternative" groupSpec -> alternative_group
    | "optional" groupSpec    -> optional_group
    | "mandatory" groupSpec   -> mandatory_group
    | CARDINALITY groupSpec   -> cardinality_group

groupSpec: _NEWLINE _INDENT feature+ _DEDENT

feature: featureType? reference featureCardinality? attributes? _NEWLINE (_INDENT group+ _DEDENT)?

featureCardinality: "cardinality" CARDINALITY

attributes: "{" (attribute ("," attribute)*)? "}"

attribute
    : valueAttribute
    | constraintAttribute

valueAttribute: key value?

key: id
value: BOOLEAN | FLOAT | INTEGER | STRING | attributes | vector
vector: "[" (value ("," value)*)? "]"

constraintAttribute
    : "constraint" constraint               -> single_constraint_attribute
    | "constraints" constraintList          -> list_constraint_attribute

constraintList: "[" (constraint ("," constraint)*)? "]"

constraints: "constraints" _NEWLINE _INDENT constraintLine* _DEDENT

constraintLine: constraint _NEWLINE

constraint
    : equation                              -> equation_constraint
    | reference                             -> literal_constraint
    | "(" constraint ")"                    -> parenthesis_constraint
    | "!" constraint                        -> not_constraint
    | constraint "&" constraint             -> and_constraint
    | constraint "|" constraint             -> or_constraint
    | constraint "=>" constraint            -> implication_constraint
    | constraint "<=>" constraint           -> equivalence_constraint

equation
    : expression "==" expression            -> equal_equation
    | expression "<" expression             -> lower_equation
    | expression ">" expression             -> greater_equation
    | expression "<=" expression             -> lower_equals_equation
    | expression ">=" expression             -> greater_equals_equation
    | expression "!=" expression             -> not_equals_equation

expression
    : FLOAT                                 -> float_literal_expression
    | INTEGER                               -> integer_literal_expression
    | STRING                                -> string_literal_expression
    | aggregateFunction                     -> aggregate_function_expression
    | reference                             -> literal_expression
    | "(" expression ")"                    -> bracket_expression
    | expression "+" expression             -> add_expression
    | expression "-" expression             -> sub_expression
    | expression "*" expression             -> mul_expression
    | expression "/" expression             -> div_expression

aggregateFunction
    : "sum" "(" (reference ",")? reference ")"    -> sum_aggregate_function
    | "avg" "(" (reference ",")? reference ")"    -> avg_aggregate_function
    | stringAggregateFunction                     -> string_aggregate_function_expression
    | numericAggregateFunction                    -> numeric_aggregate_function_expression

stringAggregateFunction
    : "len" "(" reference ")"                -> length_aggregate_function

numericAggregateFunction
    : "floor" "(" reference ")"              -> floor_aggregate_function
    | "ceil" "(" reference ")"               -> ceil_aggregate_function

reference: (id ".")* id
id: ID_STRICT | ID_NOT_STRICT
featureType: "String" | "Integer" | "Boolean" | "Real"

languageLevel: majorLevel ("." (minorLevel | "*"))?
majorLevel: "Boolean" | "Arithmetic" | "Type"
minorLevel: "group-cardinality" | "feature-cardinality" | "aggregate-function" | "string-constraints"

%import common.CNAME -> ID_STRICT
%import common.ESCAPED_STRING -> ID_NOT_STRICT
%import common.FLOAT
%import common.INT -> INTEGER
%import common.WORD -> STRING
%import common.WS_INLINE

NOT: "!"
AND: "&"
OR: "|"
EQUIVALENCE: "<=>"
IMPLICATION: "=>"

EQUAL: "=="
LOWER: "<"
LOWER_EQUALS: "<="
GREATER: ">"
GREATER_EQUALS: ">="
NOT_EQUALS: "!="

DIV: "/"
MUL: "*"
ADD: "+"
SUB: "-"

BOOLEAN: "true" | "false"

CARDINALITY: "[" INTEGER (".." (INTEGER | "*"))? "]"

COMMENT: "//" /[^\n]*/
%ignore COMMENT
%ignore WS_INLINE

_NEWLINE: /\\r?\\n/
_INDENT: /<INDENT>/
_DEDENT: /<DEDENT>/
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

elapsed_time = time.perf_counter() - start_time
print(f'Elapsed time: {elapsed_time} seconds')
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

