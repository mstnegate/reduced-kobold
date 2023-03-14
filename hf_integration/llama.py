import torch
import transformers
from transformers import LLaMAConfig, LLaMAForCausalLM
import layers

LLAMA_MODEL_TYPE_KEY = "llama-sq-reduced"

class SQReducedLLaMAConfig(LLaMAConfig):
    model_type = LLAMA_MODEL_TYPE_KEY
    def __init__(self, quantization_bits=4, is_sparse=False, qwopqwop_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.quantization_bits = quantization_bits
        self.qwopqwop_mode = qwopqwop_mode
        self.is_sparse = is_sparse

class SQReducedLLaMAForCausalLM(LLaMAForCausalLM):
    config_class = SQReducedLLaMAConfig

    def _dummy_mask_func(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return torch.ones((1,), dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    def __init__(self, config):
        super().__init__(config)

        quantization = getattr(config, "quantization_bits", -1)
        if quantization == -1:
            # do nothing
            return

        if quantization != 4:
            raise ValueError("Only 4-bit quantization supported currently")

        # evil hacky code below this point
        d = self.model

        if getattr(config, "qwopqwop_mode", False):
            print("Activating compatibility layer...")
            if getattr(config, "is_sparse", False):
                raise RuntimeError("Those models weren't ever sparsified")
            else:
                layer_cls = layers.CompatLinear_Int4
        else:
            if getattr(config, "is_sparse", False):
                layer_cls = layers.Linear_SparseInt4
            else:
                layer_cls = layers.Linear_Int4

        for layer in d.layers:
            attn = layer.self_attn
            proj_mtxes = ["k_proj", "v_proj", "q_proj", "o_proj"]
            for proj_mtx_name in proj_mtxes:
                lin_lay = getattr(attn, proj_mtx_name)
                setattr(attn, proj_mtx_name, layer_cls(lin_lay.in_features, lin_lay.out_features, bias=False))

            mlp = layer.mlp
            proj_mtxes = ["gate_proj", "down_proj", "up_proj"]
            for proj_mtx_name in proj_mtxes:
                lin_lay = getattr(mlp, proj_mtx_name)
                setattr(mlp, proj_mtx_name, layer_cls(lin_lay.in_features, lin_lay.out_features, bias=False))


def register():
    transformers.AutoConfig.register(LLAMA_MODEL_TYPE_KEY, SQReducedLLaMAConfig)
    transformers.AutoModelForCausalLM.register(SQReducedLLaMAConfig, SQReducedLLaMAForCausalLM)
