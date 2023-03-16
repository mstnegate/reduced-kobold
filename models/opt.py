import os

import torch
import transformers

from .base import SQBaseModelWrapper, SQShardedMixin

class OPTModelWrapper(SQBaseModelWrapper):
    QUANTIZATION_NODES = [
        "fc1",
        "fc2",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
    ]
    HAS_BIAS = True

    def __init__(self, fld):
        super().__init__(fld)
        self.prefix = "model."

    def embed(self, tokens, batch_size):
        # TODO: cleaner way to embed tokens

        if self._pos_embedder is None:
            self.load_embedder()

        token_embeds = (self._token_embedder)(tokens)

        attn_mask = torch.ones(
            token_embeds.shape[:2],
            dtype=torch.bool,
            device=token_embeds.device
        )
        pos_embeds = (self._pos_embedder)(attn_mask, 0)

        from transformers.models.opt.modeling_opt import OPTDecoder
        attn_mask = OPTDecoder._prepare_decoder_attention_mask(
            None, attn_mask, (batch_size, tokens.shape[1]), token_embeds, 0
        )

        return token_embeds + pos_embeds, attn_mask.to("cuda:0")


class MemoryOPTModelWrapper(OPTModelWrapper):
    # loads in a full model in one shot; thin wrapper around HF

    def __init__(self, fld):
        super().__init__(fld)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            fld, torch_dtype=torch.float16)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(fld)
        self._token_embedder = self.model.model.decoder.embed_tokens
        self._pos_embedder = self.model.model.decoder.embed_positions

    def decoder_layers(self):
        for layer in self.model.model.decoder.layers:
            yield layer

    def write(self, layer, pth, do_sparse, debug_quantize_mode):
        pass # no-op; we store everything in self.model

    def finalize_items(self):
        pass # no-op again; everything is in self.model

    @property
    def save_container(self):
        # force clone to prevent view-saving; might be possible to disable this
        return {k:v.clone() for k,v in self.model.state_dict().items()}


class ShardedOPTModelWrapper(OPTModelWrapper, SQShardedMixin):
    # loads in a sharded model one layer at a time, disposing after each layer

    class FakeOPTConfig:
        def __init__(self, dct):
            super().__init__()
            self._dct = dct
        def __getattr__(self, k):
            return self._dct[k]

    def __init__(self, fld):
        super().__init__(fld)

        self.setup_sharding()

        self.fake_model_conf = ShardedOPTModelWrapper.FakeOPTConfig(self.model_conf)
        self._pos_embedder = None
        self._token_embedder = None

    def load_embedder(self):
        from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
        self._pos_embedder = OPTLearnedPositionalEmbedding(
            self.model_conf["max_position_embeddings"],
            self.model_conf["hidden_size"],
        )
        self._token_embedder = torch.nn.Embedding(
            self.model_conf["vocab_size"],
            self.model_conf["word_embed_proj_dim"],
            self.model_conf["pad_token_id"]
        )

        self._pos_embedder.weight.data = self.fetch_weights("%sdecoder.embed_positions.weight" % self.prefix)
        self._token_embedder.weight.data = self.fetch_weights("%sdecoder.embed_tokens.weight" % self.prefix)

    def decoder_layers(self):
        from transformers.models.opt.modeling_opt import OPTDecoderLayer

        # idea: generate each layer as needed ex-nihilo then delete it once used
        n_layers = self.model_conf["num_hidden_layers"]

        for i in range(n_layers):
            prefix = "%sdecoder.layers.%d." % (self.prefix, i)

            # and now construct the decoding layer
            decode_layer = OPTDecoderLayer(self.fake_model_conf)
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

            self.save_container[k] = self.fetch_weights(k, autodispose=True)

        self.flush_loaded()

    def write(self, layer, pth, do_sparse, debug_quantize_mode):
        pth = "%sdecoder.layers.%s" % (self.prefix, pth)

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
