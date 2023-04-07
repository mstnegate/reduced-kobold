import torch
import transformers
import layers

MODEL_TYPE_KEY = "opt-sq-reduced"

class SQReducedOptConfig(transformers.OPTConfig):
    model_type = MODEL_TYPE_KEY
    def __init__(self, quantization_bits=4, is_sparse=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.quantization_bits = quantization_bits
        self.is_sparse = is_sparse

class SQReducedOPTForCausalLM(transformers.OPTForCausalLM):
    config_class = SQReducedOptConfig

    def __init__(self, config):
        super().__init__(config)

        quantization = getattr(config, "quantization_bits", -1)
        if quantization == -1:
            # do nothing
            return

        if quantization != 4:
            raise ValueError("Only 4-bit quantization supported currently")

        group_size = getattr(config, "quantization_group_size", -1)

        kwargs = dict(quantization_bits=quantization, group_size=group_size)

        # evil hacky code below this point
        d = self.model.decoder

        if getattr(config, "is_sparse", False):
            layer_cls = layers.Linear_SparseInt4
        else:
            layer_cls = layers.Linear_Int4

        for layer in d.layers:
            attn = layer.self_attn
            proj_mtxes = ["k_proj", "v_proj", "q_proj", "out_proj"]
            for proj_mtx_name in proj_mtxes:
                lin_lay = getattr(attn, proj_mtx_name)
                setattr(attn, proj_mtx_name, layer_cls(lin_lay.in_features, lin_lay.out_features, **kwargs))

            layer.fc1 = layer_cls(layer.fc1.in_features, layer.fc1.out_features, **kwargs)
            layer.fc2 = layer_cls(layer.fc2.in_features, layer.fc2.out_features, **kwargs)

            for module in layer.modules():
                if not isinstance(module, (layers.Linear_Int4, layers.Linear_SparseInt4)):
                    continue

                for k,v in module.state_dict().items():
                    if isinstance(v, torch.Tensor):
                        if (v.dtype == torch.float32):
                            getattr(module, k).data = v.to(torch.float16)

def register():
    transformers.AutoConfig.register(MODEL_TYPE_KEY, SQReducedOptConfig)
    transformers.AutoModelForCausalLM.register(SQReducedOptConfig, SQReducedOPTForCausalLM)
