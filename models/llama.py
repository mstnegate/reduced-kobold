import os

import torch
import transformers

from .base import SQBaseModelWrapper, SQShardedMixin

class LlamaModelWrapper(SQBaseModelWrapper):
    QUANTIZATION_NODES = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
    ]
    HAS_BIAS = False

    def __init__(self, fld):
        super().__init__(fld)
        self.prefix = "model."

    def embed(self, tokens, batch_size):
        if self._token_embedder is None:
            self.load_embedder()

        token_embeds = self._token_embedder(tokens)
        # positional embedding apparently happens up in the attention module

        attn_mask = torch.ones(
            token_embeds.shape[:2],
            dtype=torch.bool,
            device=token_embeds.device
        )
        from transformers.models.llama.modeling_llama import LlamaModel
        attn_mask = LlamaModel._prepare_decoder_attention_mask(
            None, attn_mask, (batch_size, tokens.shape[1]), token_embeds, 0
        )
        seq_len = token_embeds.shape[1]
        pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=token_embeds.device).unsqueeze(0)

        return token_embeds, dict(
            attention_mask=attn_mask.to("cuda:0"),
            position_ids=pos_ids.to("cuda:0"),
        )

class MemoryLlamaModelWrapper(LlamaModelWrapper):
    def __init__(self, fld):
        # just shard the model; there aren't models small enough to make
        # this worthwhile
        raise NotImplementedError

class ShardedLlamaModelWrapper(LlamaModelWrapper, SQShardedMixin):
    class FakeLlamaConfig:
        def __init__(self, dct):
            super().__init__()
            self._dct = dct
        def __getattr__(self, k):
            return self._dct[k]

    def __init__(self, fld):
        super().__init__(fld)

        self.setup_sharding()

        self.fake_model_conf = ShardedLlamaModelWrapper.FakeLlamaConfig(self.model_conf)
        if "max_position_embeddings" not in self.fake_model_conf._dct:
            self.fake_model_conf._dct["max_position_embeddings"] = self.fake_model_conf.max_sequence_length

        self._token_embedder = None

    def load_embedder(self):
        self._token_embedder = torch.nn.Embedding(
            self.model_conf["vocab_size"],
            self.model_conf["hidden_size"],
            self.model_conf["pad_token_id"]
        )
        self._token_embedder.weight.data = self.fetch_weights("%sembed_tokens.weight" % self.prefix)
        self.flush_loaded()

    def decoder_layers(self):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        n_layers = self.model_conf["num_hidden_layers"]

        for i in range(n_layers):
            prefix = "%slayers.%d." % (self.prefix, i)

            decode_layer = LlamaDecoderLayer(self.fake_model_conf)
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
        already_hit = set(self.save_container.keys())
        for k,v in self.map["weight_map"].items():
            if k in already_hit:
                continue

            if k in self.dont_save_these_keys:
                continue

            self.save_container[k] = self.fetch_weights(k, autodispose=True)

        self.flush_loaded()

    def write(self, layer, pth, do_sparse, debug_quantize_mode):
        pth = "%slayers.%s" % (self.prefix, pth)

        if not debug_quantize_mode:
            self.save_container["%s.quantized_weights" % pth] = layer.quantized_weights.data.to("cpu").clone()
            if do_sparse:
                self.save_container["%s.quantized_mask" % pth] = layer.quantized_mask.data.to("cpu").clone()

            self.dont_save_these_keys.add("%s.weight" % pth)
            self.save_container["%s.zeros" % pth] = layer.zeros.data.to("cpu").clone()
            self.save_container["%s.scales" % pth] = layer.scales.data.to("cpu").clone()
        else:
            self.save_container["%s.weight" % pth] = layer.weight.data.to("cpu").clone()
