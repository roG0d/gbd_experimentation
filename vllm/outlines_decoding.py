import outlines.generate as generate
import outlines.models as models

# from https://github.com/dottxt-ai/outlines/blob/main/examples/cfg.py

arithmetic_grammar = r"""
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

model = models.transformers("meta-llama/Meta-Llama-3-8B-Instruct")
batch_size = 10
print(model.model.config)
for grammar in  [arithmetic_grammar]:
    generator = generate.cfg(model, grammar)
    sequences = generator([" "] * batch_size)
    for seq in sequences:
        try:
            parse = generator.fsm.parser.parse(seq)
            assert parse is not None
            print("SUCCESS", seq)
        except Exception:  # will also fail if goes over max_tokens / context window
            print("FAILURE", seq)