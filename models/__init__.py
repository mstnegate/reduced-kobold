from .opt import MemoryOPTModelWrapper, ShardedOPTModelWrapper
try:
    from .llama import MemoryLLaMAModelWrapper, ShardedLLaMAModelWrapper
except ImportError:
    print("LLaMA model code not found; proceeding anyway.")

def get_loader(model_type, is_sharded):
    if model_type == "opt":
        return ShardedOPTModelWrapper if is_sharded else MemoryOPTModelWrapper
    elif model_type == "llama":
        return ShardedLLaMAModelWrapper if is_sharded else MemoryLLaMAModelWrapper
    elif model_type in ("opt-sq-reduced", "llama-sq-reduced"):
        raise ValueError("Cannot quantize or sparsify already quantized/sparsified models; use the base model.")
    else:
        raise ValueError("Unknown model type %r passed" % model_type)
