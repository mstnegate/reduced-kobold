import os

import torch
import transformers

from .base import SQBaseModelWrapper, SQShardedMixin

class GPTNeoXModelWrapper(SQBaseModelWrapper):
    QUANTIZATION_NODES = [
        "mlp.dense_4h_to_h",
        "mlp.dense_h_to_4h",
        "attention.dense",
        "attention.query_key_value",
    ]
    HAS_BIAS = True

    def __init__(self, fld):
        super().__init__(fld)
        self.prefix = "gpt_neox."

    def embed(self, tokens, batch_size):
        if self._token_embedder is None:
            self.load_embedder()

        attn_mask = torch.zeros_like(tokens)[:, None, None, :]
        self._token_embedder.to(tokens.device)

        return self._token_embedder(tokens), attn_mask


class MemoryGPTNeoXModelWrapper(GPTNeoXModelWrapper):
    # loads in a full model in one shot; thin wrapper around HF

    def __init__(self, fld):
        super().__init__(fld)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            fld, torch_dtype=torch.float16)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(fld)
        self._token_embedder = self.model.gpt_neox.embed_in

    def decoder_layers(self):
        for layer in self.model.gpt_neox.layers:
            yield layer

    def write(self, layer, pth, do_sparse, debug_quantize_mode):
        pass # no-op; we store everything in self.model

    def finalize_items(self):
        pass # no-op again; everything is in self.model

    @property
    def save_container(self):
        # force clone to prevent view-saving; might be possible to disable this
        return {k:v.clone() for k,v in self.model.state_dict().items()}


class ShardedGPTNeoXModelWrapper(GPTNeoXModelWrapper, SQShardedMixin):
    # loads in a sharded model one layer at a time, disposing after each layer

    def __init__(self, fld):
        super().__init__(fld)

        self.setup_sharding()

        from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
        self.fake_model_conf = GPTNeoXConfig(**self.model_conf)
        self._token_embedder = None

    def load_embedder(self):
        self._token_embedder = torch.nn.Embedding(
            self.fake_model_conf.vocab_size,
            self.fake_model_conf.hidden_size
        )
        self._token_embedder.weight.data = self.fetch_weights("%sembed_in.weight" % self.prefix)

    def decoder_layers(self):
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

        # idea: generate each layer as needed ex-nihilo then delete it once used
        n_layers = self.model_conf["num_hidden_layers"]

        for i in range(n_layers):
            prefix = "%slayers.%d." % (self.prefix, i)

            # and now construct the decoding layer
            decode_layer = GPTNeoXLayer(self.fake_model_conf)
            for k, v in self.get_weights_with_prefix(prefix).items():
                root, *path, leaf = k[len(prefix):].split(".")

                dig = getattr(decode_layer, root)
                for part in path:
                    dig = getattr(dig, part)

                if isinstance(getattr(dig, leaf), torch.nn.Parameter):
                    # slightly non-standard assign since you need to do param.data
                    dig = getattr(dig, leaf)
                    dig.data = self.fetch_weights(k).clone()
                else:
                    setattr(dig, leaf, self.fetch_weights(k).clone())

            yield decode_layer

            del decode_layer

            self.flush_loaded()

    def finalize_items(self):
        # everything that we didn't get before, we're going to load now
        already_hit = set(self.save_container.keys())
        for k,v in self.map["weight_map"].items():
            if k in already_hit:
                continue

            if k in self.dont_save_these_keys:
                continue

            self.save_container[k] = self.fetch_weights(k)

        self.flush_loaded()

    def write(self, layer, pth, do_sparse, debug_quantize_mode):
        pth = "%slayers.%s" % (self.prefix, pth)

        self.save_container["%s.bias" % pth] = layer.bias.data.to("cpu").clone()
        # kinda ugly; should probably load in named_buffers/params from a module instead
        if not debug_quantize_mode:
            self.save_container["%s.quantized_weights" % pth] = layer.quantized_weights.data.to("cpu").clone()
            if do_sparse:
                self.save_container["%s.quantized_mask" % pth] = layer.quantized_mask.data.to("cpu").clone()

            self.dont_save_these_keys.add("%s.weight" % pth)
            self.save_container["%s.zeros" % pth] = layer.zeros.data.to("cpu").clone()
            self.save_container["%s.scales" % pth] = layer.scales.data.to("cpu").clone()
        else:
            self.save_container["%s.weight" % pth] = layer.weight.data.to("cpu").clone()
