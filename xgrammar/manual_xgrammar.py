import xgrammar as xgr
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Get tokenizer info
model_id = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
# This can be larger than tokenizer.vocab_size due to paddings
full_vocab_size = config.vocab_size
tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)


ebnf_grammar_str = """
root ::= "\n features \n\t" feature ws
feature ::= reference "\n"( ("\t")+ group)? 
group    ::= ( "or " | "alternative " | "optional " | "mandatory ") ("\n" ("\t")+  feature)+ 
reference ::= [a-zA-Z_] [a-zA-Z_0-9]*
ws ::= [ \t\n]
"""

compiled_grammar = compiler.compile_grammar(ebnf_grammar_str)