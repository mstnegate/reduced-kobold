from .opt import MemoryOPTModelWrapper, ShardedOPTModelWrapper
from .gpt_neox import MemoryGPTNeoXModelWrapper, ShardedGPTNeoXModelWrapper
from .llama import MemoryLlamaModelWrapper, ShardedLlamaModelWrapper

def get_loader(model_type, is_sharded):
    if model_type == "opt":
        return ShardedOPTModelWrapper if is_sharded else MemoryOPTModelWrapper
    elif model_type == "llama":
        return ShardedLlamaModelWrapper if is_sharded else MemoryLlamaModelWrapper
    elif model_type == "gpt_neox":
        return ShardedGPTNeoXModelWrapper if is_sharded else MemoryGPTNeoXModelWrapper
    elif model_type in ("opt-sq-reduced", "llama-sq-reduced", "gpt_neox-sq-reduced"):
        raise ValueError("Cannot quantize or sparsify already quantized/sparsified models; use the base model.")
    else:
        raise ValueError("Unknown model type %r passed" % model_type)
