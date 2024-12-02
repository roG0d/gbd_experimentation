import xgrammar as xgr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

messages = [
    {"role": "system", "content": """
You are a helpful AI assistant for creating sintactically correct expressions similar to these examples:
features
INDENT
    SmartWatch
    INDENT
        mandatory
        INDENT
            Functionalities
            INDENT
                mandatory
                INDENT
                    FitnessMonitor
                    SleepTracker
                    VibrateAlert
                    INDENT
                        mandatory
                        INDENT
                            Call
                            Notification
                        DEDENT
                    DEDENT
                DEDENT
            DEDENT
            Sensors
            INDENT
                mandatory
                INDENT
                    Pedometer
                    Accelerometer
                DEDENT
                optional
                INDENT
                    HeartRateSensor
                DEDENT
            DEDENT
            Connectivity
            INDENT
                mandatory
                INDENT
                    BT40
                DEDENT
            DEDENT
        DEDENT
    DEDENT
DEDENT
     """},
    {"role": "user", "content": """
     Create a new one representing a computer:
     """},
]
texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)

tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

uvl_gbnf_no_indents = r"""
root ::= "features" _NL feature_list?
feature_list ::= feature+

feature ::= NAME _NL group_list?
group_list ::= group+

group ::= or_group | alternative_group | optional_group | mandatory_group | cardinality_group
or_group ::= "or" groupspec
alternative_group ::= "alternative" groupspec
optional_group ::= "optional" groupspec
mandatory_group ::= "mandatory" groupspec
cardinality_group ::= cardinality groupspec

groupspec ::= _NL feature_list

cardinality ::= "[" INT (".." (INT | "*"))? "]"

NAME ::= "CNAME"
INT ::= "INT"

_NL ::= "(\r?\n[\t ]*)+"
"""

uvl_gbnf_indents = r"""
root ::= (_NL* featuremodel)

featuremodel ::= "features" _NL "INDENT" feature+ "DEDENT"
feature ::= NAME _NL ("INDENT" group+ "DEDENT")?

group ::= or_group | alternative_group | optional_group | mandatory_group | cardinality_group
or_group ::= "or" groupspec
alternative_group ::= "alternative" groupspec
optional_group ::= "optional" groupspec
mandatory_group ::= "mandatory" groupspec
cardinality_group ::= cardinality groupspec 

groupspec ::= _NL "INDENT" feature+ "DEDENT"

cardinality ::= "[" INT (".." (INT | "*"))? "]"

NAME ::= "CNAME"
INT ::= "INT"

_NL ::= /(\r?\n[\t ]*)+/
"""
compiled_grammar = grammar_compiler.compile_grammar(uvl_gbnf_indents)

xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
generated_ids = model.generate(
    **model_inputs, max_new_tokens=512, logits_processor=[xgr_logits_processor]
)
generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :] 
print(tokenizer.decode(generated_ids, skip_special_tokens=False))
