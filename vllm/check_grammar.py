uvL_full_grammar = r"""
?start: _NL* featuremodel

featuremodel: "features" _NL [_INDENT feature+ _DEDENT]
feature: NAME _NL (_INDENT group+ _DEDENT)?

group: "or" groupspec          -> or_group
    | "alternative" groupspec -> alternative_group
    | "optional" groupspec    -> optional_group
    | "mandatory" groupspec   -> mandatory_group
    | cardinality groupspec    -> cardinality_group

groupspec: _NL _INDENT feature+ _DEDENT

cardinality: "[" INT (".." (INT | "*"))? "]"

namespace: "namespace" NAME _NL?

includes: "include" _NL _INDENT includeLine* _DEDENT
includeLine: languageLevel _NL

imports: "imports" _NL _INDENT importLine* _DEDENT
importLine: NAME ("as" NAME)? _NL

constraints: "constraints" _NL _INDENT constraintLine* _DEDENT
constraintLine: constraint _NL

constraint: equation                              -> equation_constraint
          | reference                             -> literal_constraint
          | "(" constraint ")"                    -> parenthesis_constraint
          | "!" constraint                        -> not_constraint
          | constraint "&" constraint             -> and_constraint
          | constraint "|" constraint             -> or_constraint
          | constraint "=>" constraint            -> implication_constraint
          | constraint "<=>" constraint           -> equivalence_constraint

equation: expression "==" expression            -> equal_equation
        | expression "<" expression             -> lower_equation
        | expression ">" expression             -> greater_equation
        | expression "<=" expression            -> lower_equals_equation
        | expression ">=" expression            -> greater_equals_equation
        | expression "!=" expression            -> not_equals_equation

expression: FLOAT -> float_literal_expression
          | INT -> integer_literal_expression
          | STRING -> string_literal_expression
          | aggregateFunction -> aggregate_function_expression
          | reference -> literal_expression
          | "(" expression ")" -> bracket_expression
          | expression "+" expression -> add_expression
          | expression "-" expression -> sub_expression
          | expression "*" expression -> mul_expression
          | expression "/" expression -> div_expression

aggregateFunction: "sum" "(" (reference ("," reference)*)? ")" -> sum_aggregate_function
                 | "avg" "(" (reference ("," reference)*)? ")" -> avg_aggregate_function
                 | stringAggregateFunction -> string_aggregate_function_expression
                 | numericAggregateFunction -> numeric_aggregate_function_expression

stringAggregateFunction: "len" "(" reference ")" -> length_aggregate_function

numericAggregateFunction: "floor" "(" reference ")" -> floor_aggregate_function
                        | "ceil" "(" reference ")" -> ceil_aggregate_function

reference: (NAME ".")* NAME

languageLevel: majorLevel ("." (minorLevel | "*"))?
majorLevel: "Boolean" | "Arithmetic" | "Type"
minorLevel: "group-cardinality" | "feature-cardinality" | "aggregate-function" | "string-constraints"

%import common.CNAME -> NAME
%import common.FLOAT
%import common.INT
%import common.ESCAPED_STRING -> STRING
%import common.WS_INLINE
%declare _INDENT _DEDENT
%ignore WS_INLINE

_NL: /\r?\n[ \t]*/
"""

uvl_level1_grammar = r"""
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

scratch_grammar = r""""
?start: _NL* tree

tree: NAME _NL [_INDENT tree+ _DEDENT]

%import common.CNAME -> NAME
%import common.WS_INLINE
%declare _INDENT _DEDENT
%ignore WS_INLINE

_NL: /(\r?\n[\t ]*)+/
"""

generation = f"""
features
    b
        or
            a
            c
    c
        alternative
            test
        [1..2]
            pepe
"""


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

# Create a Lark parser with the grammar
parser = Lark(uvl_level1_grammar, parser='lalr', postlex=TreeIndenter())
print("Grammar is well-written.")

# Parse a generation
test(generation=generation, parser=parser)
    
