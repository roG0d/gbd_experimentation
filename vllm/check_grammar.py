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

from lark import Lark, exceptions
from lark.indenter import Indenter

"""
class TreeIndenter(Indenter):
    NL_type = '_NEWLINE'
    OPEN_PAREN_types = []
    CLOSE_PAREN_types = []
    INDENT_type = '_INDENT'
    DEDENT_type = '_DEDENT'
    tab_len = 8
"""

try:
    # Create a Lark parser with the grammar
    # parser = Lark(uvl_grammar, parser='lalr',  propagate_positions=True, postlex=TreeIndenter()) TreeIndenter try
    parser = Lark(uvl_grammar, parser='lalr')
    print("Grammar is well-written.")
except exceptions.LarkError as e:
    # If there is an error in the grammar, it will raise an exception
    print("Error in the grammar definition:")
    print(e)