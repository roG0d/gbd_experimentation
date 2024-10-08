import os
from llama_cpp import Llama, LlamaGrammar
from llama_cpp.llama_grammar import JSON_GBNF
from dotenv import load_dotenv

load_dotenv()

#GBNF
arithmetic_grammar="""
root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
"""

uvl_grammar="""
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
# lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF // *Q8_0.gguf
# lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF // https://github.com/abetlen/llama-cpp-python/pull/1457/files

# Fix print grammar https://github.com/abetlen/llama-cpp-python/commit/0998ea0deea076a547d54bd598d6b413b588ee2b
arithmetic_grammar = LlamaGrammar.from_string(arithmetic_grammar, verbose=False)
arithmetic_prompt="Rewrite 5*5 as another mathematically correct expression"
#response = llama(arithmetic_prompt, max_tokens=100, temperature=1, top_p=0.95, seed=10,grammar=arithmetic_grammar)
#response = llama(prompt="Rewrite 5*5 as different equivalent mathematical expressions", max_tokens=100, temperature=1, top_p=0.95, grammar=arithmetic_grammar)
#print(response["choices"][0]["text"])


# local inference with every layer offloaded to GPU (n_gpu_layers=-1)
#llama.__init__(model_path=llama.model_path, n_gpu_layers=-1)

# Code refactor: added n_gpus_layers to from_pretrained function
uvl_grammar = LlamaGrammar.from_string(uvl_grammar, verbose=False)
uvl_prompt = "Write syntactically valid expression given a formal grammar"
llama = Llama.from_pretrained(
    repo_id="lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF",
    filename="Meta-Llama-3.1-70B-Instruct-Q8_0-00001-of-00002.gguf",
    additional_files=["Meta-Llama-3.1-70B-Instruct-Q8_0-00002-of-00002.gguf"],
    n_gpu_layers=-1,
    verbose=False,
)
response = llama(echo=True, prompt=uvl_prompt, max_tokens=500, temperature=1, top_p=0.95, grammar=uvl_grammar)
print(response["choices"][0]["text"])





