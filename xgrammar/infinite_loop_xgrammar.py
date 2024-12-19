# from: https://xgrammar.mlc.ai/docs/how_to/ebnf_guided_generation.html#try-out-via-hf-transformers
import xgrammar as xgr

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
import random

# Generates a random number between
# a given positive range
r1 = 2 #random.randint(0, 1e5)
set_seed(r1)
print(f"seed: {r1}")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

messages = [
    {"role": "system", "content": """
You are a helpful AI assistant for creating sintactically correct expressions similar to these example:
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
    Create a new one representing a computer
     """},
]
texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(texts, return_tensors="pt").to(model.device)

tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=config.vocab_size)
grammar_compiler = xgr.GrammarCompiler(tokenizer_info)

uvl_gbnf_no_indents = r"""
start ::= "features" _NL feature_list?
feature_list ::= feature+

feature ::= chars+ _NL group_list?
group_list ::= group+

group ::= or_group | alternative_group | optional_group | mandatory_group | cardinality_group
or_group ::= "or" groupspec
alternative_group ::= "alternative" groupspec
optional_group ::= "optional" groupspec
mandatory_group ::= "mandatory" groupspec
cardinality_group ::= cardinality groupspec

groupspec ::= _NL feature_list

cardinality ::= "[" digit+ (".." (digit+ | "*"))? "]"

chars ::= letter | digit 
letter ::= [a-zA-Z]
digit ::= [0-9]

_NL ::= [ \t\n]+
"""

uvl_base = r"""
root ::= "\n features \n\t" feature ws
feature ::= reference "\n"( ("\t")+ group)? 
group    ::= ( "or " | "alternative " | "optional " | "mandatory ") ("\n" ("\t")+  feature)+ 
reference ::= [a-zA-Z_] [a-zA-Z_0-9]*
ws ::= [ \t\n]
"""

compiled_grammar = grammar_compiler.compile_grammar(uvl_base)

xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

import datetime
init_time = datetime.datetime.now()

generated_ids = model.generate(
    **model_inputs, max_new_tokens=500, logits_processor=[xgr_logits_processor]
)
generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :] 
print(tokenizer.decode(generated_ids, skip_special_tokens=False))

print(f"generation elapsed time :{datetime.datetime.now()-init_time}")
