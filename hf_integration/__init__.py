from .opt import register as register_opt
try:
    from .llama import register as register_llama
    _LLAMA_LOADED = True
except ImportError:
    _LLAMA_LOADED = False
    print("LLaMA model code not found; proceeding anyway.")

def register():
    register_opt()

    if _LLAMA_LOADED:
        register_llama()
