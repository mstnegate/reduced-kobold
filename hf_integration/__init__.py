from .opt import register as register_opt
from .gpt_neox import register as register_gpt_neox
from .llama import register as register_llama

def register():
    register_opt()
    register_gpt_neox()
    register_llama()
