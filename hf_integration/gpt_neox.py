import torch
import transformers
import layers

MODEL_TYPE_KEY = "gpt_neox-sq-reduced"

class SQReducedGPTNeoXConfig(transformers.GPTNeoXConfig):
    model_type = MODEL_TYPE_KEY
    def __init__(self, quantization_bits=4, is_sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.quantization_bits = quantization_bits
        self.is_sparse = is_sparse

class SQReducedGPTNeoXForCausalLM(transformers.GPTNeoXForCausalLM):
    config_class = SQReducedGPTNeoXConfig

    def __init__(self, config):
        super().__init__(config)

        quantization = getattr(config, "quantization_bits", -1)
        if quantization == -1:
            # do nothing
            return

        if quantization != 4:
            raise ValueError("Only 4-bit quantization supported currently")

        # evil hacky code below this point
        d = self.gpt_neox

        if getattr(config, "is_sparse", False):
            layer_cls = layers.Linear_SparseInt4
        else:
            layer_cls = layers.Linear_Int4

        for layer in d.layers:
            attn = layer.attention
            proj_mtxes = ["query_key_value", "dense"]
            for proj_mtx_name in proj_mtxes:
                lin_lay = getattr(attn, proj_mtx_name)
                setattr(attn, proj_mtx_name, layer_cls(lin_lay.in_features, lin_lay.out_features))

            mlp = layer.mlp
            proj_mtxes = ["dense_4h_to_h", "dense_h_to_4h"]
            for proj_mtx_name in proj_mtxes:
                lin_lay = getattr(mlp, proj_mtx_name)
                setattr(mlp, proj_mtx_name, layer_cls(lin_lay.in_features, lin_lay.out_features))

def register():
    transformers.AutoConfig.register(MODEL_TYPE_KEY, SQReducedGPTNeoXConfig)
    transformers.AutoModelForCausalLM.register(SQReducedGPTNeoXConfig, SQReducedGPTNeoXForCausalLM)
