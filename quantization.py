import gc
import json
import os
import random

import numpy as np
import torch
import transformers

import layers
from layers import unquantize, quantize, fake_quantize, MAX_QUANTIZATION_VALUE

################################################################################
# general config stuff

# stuff you will probably want to change
PERFORM_QUANTIZATION = True
PERFORM_SPARSIFICATION = True
LOAD_FILE_PATH = "/pth/to/your/model/folder"
LOAD_MODEL_IS_SHARDED = False

# stuff you probably don't need to change
N_SAMPLES = 128
CALIBRATION_BATCH_SIZE = 1

CALIBRATION_DATA_SEED = 1234 # set to None for random seed
# dataset mode
CALIBRATION_DATA_SETTINGS = dict(
    mode="dataset",
    args=dict(
        path="allenai/c4",
        name="allenai--c4",
        data_files=dict(train="en/c4-train.00000-of-01024.json.gz"),
        split="train",
    )
)
# file mode
# CALIBRATION_DATA_SETTINGS = dict(
    # mode="file",
    # args="/path/to/your/file"
# )

SAVE_FILE_PATH = "pytorch_model.bin"

# more technical stuff with gptq/sparsegpt; don't change these unless you
# know what you're doing
GPTQ_BLOCK_SIZE = 128           # suggested by gptq paper on p5
SPARSEGPT_BLOCK_SPARSITY = 32   # keep at 32 unless benchmarking fp16 sparse
GPTQ_LAMBDA_FACTOR = 0.01       # suggested by gptq paper on p6
CALIBRATION_TOKEN_LENGTH = 2048 # mostly model-dependent

# debug options; don't touch unless you know what you're doing
CHECK_EIGS_JUST_TO_BE_SURE = False
ONLY_FAKE_QUANTIZE = False
QUANTIZATION_RANGE_MUST_INCLUDE_ZERO = True
PERFORM_RTN_QUANTIZATION = False

################################################################################

if PERFORM_SPARSIFICATION and not PERFORM_QUANTIZATION:
    print("WARNING: there is no model or kernel support currently for sparse "
          + "fp16; you will _NOT_ see memory or speed improvements. This is "
          + "purely for benchmarking purposes.")

    if not ONLY_FAKE_QUANTIZE:
        raise ValueError("Only fake quantization/sparsification supported.")

################################################################################

def get_quantization_params(W):
    mmin = W.min(axis=1).values
    mmax = W.max(axis=1).values

    if QUANTIZATION_RANGE_MUST_INCLUDE_ZERO:
        # zero-point corresponds to the real value 0 when dequantized; the
        # quantization range kinda needs to include 0 for this to be true
        mmin = torch.minimum(mmin, torch.zeros_like(mmin))
        mmax = torch.maximum(mmax, torch.zeros_like(mmax))

    mscl = (mmax - mmin) / MAX_QUANTIZATION_VALUE

    zeros = torch.round(-mmin/mscl)
    scales = mscl

    return zeros, scales


def gptq_quantization(H, W, sparsify=False):
    # config parameters:
    # block size; size of the chunks we iterate by when quantization/sparisfying
    B = GPTQ_BLOCK_SIZE
    # block sparsity structure; this is size of matrix spanned, not weights
    # (i.e. it's a (B_s//2):B_s sparsity structure)
    B_s = SPARSEGPT_BLOCK_SPARSITY
    lambda_factor = GPTQ_LAMBDA_FACTOR

    if PERFORM_QUANTIZATION and PERFORM_SPARSIFICATION:
        # we have kernel support currently so these sparsity checks matter
        assert B_s == 32, "CUDA kernel currently only supports 16:32 sparsity structure"
        assert B_s <= B, "Sparsity block size cannot exceed quantization block size"
        assert B % B_s == 0, "Sparsity block size must divide quantization block size"
    else:
        # we're either plain int4 (no sparsification) or sparse fp16 (no kernel)
        # in both cases, the param checks are pointless
        pass

    d_row, d_col = W.shape

    zeros, scales = get_quantization_params(W)

    H *= 2 # H = 2XX^T+\lambdaI, not XX^T+\lambdaI

    lambda_ = H.diagonal().mean(axis=0) * lambda_factor
    diag_idxes = list(range(H.shape[-1]))

    # TODO: group quantization

    attempts = 0
    while True:
        attempts += 1
        if (attempts > 10):
            # bad numerics, somehow
            raise ValueError("H matrix not posdef after %d lambda adjustment[s]; check your calibration data and/or model" % attempts)

        H[diag_idxes, diag_idxes] += lambda_

        if CHECK_EIGS_JUST_TO_BE_SURE:
            # check to make sure H is actually posdef; chol is *supposed* to
            # error if that happens but it didn't work for me (...somehow)
            eigs = torch.linalg.eigvalsh(H)
            if (eigs.min() < 1e-6):
                continue
            else:
                break
        else:
            # pray
            break

    # first, invert H; H is posdef by design (...according to the paper, anyway)
    H = torch.linalg.cholesky(H)
    # invert via choleskying; cholesky_inverse requires a pre-choleskyed matrix
    H = torch.cholesky_inverse(H)
    # upper chol (figure 2 in gptq paper, also sparsegpt algo)
    H_inv = torch.linalg.cholesky(H, upper=True)

    Q = torch.zeros((d_row, d_col), dtype=W.dtype, device=W.device)
    E = torch.zeros((d_row, B), dtype=W.dtype, device=W.device)
    if sparsify:
        M = torch.ones((d_row, d_col), dtype=torch.uint8, device=W.device)

    for i in range(0, d_col, B):
        temp_b = min(B, d_col-i)

        if (temp_b < B):
            Q = Q[:, :(d_col-temp_b)]
            E = E[:, :temp_b]

        for j in range(i, i+temp_b):
            if sparsify:
                if (j % B_s) == 0:
                    H_inv_d = torch.diagonal(H_inv[j:(j+B_s), j:(j+B_s)])
                    scores = W[:, j:(j+B_s)]**2 / H_inv_d**2

                    g_mask = torch.zeros_like(scores)
                    g_mask.scatter_(1, torch.argsort(scores)[:, (B_s // 2):], 1)

                    M[:, j:(j+B_s)] = g_mask.to(M.dtype)
                    assert((M[:, j:(j+B_s)].sum(axis=1) == (B_s // 2)).all())

                    del g_mask

            W_q = fake_quantize(W[:, j], zeros, scales)
            Q[:, j] = W_q

            if sparsify and PERFORM_QUANTIZATION:
                if PERFORM_QUANTIZATION:
                    # paper has a square in the numerator; that seemed like a typo
                    E[:, j-i] = (W[:, j] - M[:,j]*Q[:, j]) / H_inv[j, j]
                else:
                    # remember that W(1 - M) == (W - MW) (up to numerics)
                    E[:, j-i] = W[:, j] / H_inv[j, j]
                    E[:, j-i] = (1 - M[:, j]) * E[:, j-i]
            else:
                E[:, j-i] = (W[:, j] - Q[:, j]) / H_inv[j, j]

            # original paper said just to assign, but:
            # 1. just assigning doesn't work
            # 2. sparsegpt's paper says it can reuse the overall loop, and that
            #    paper subtracted
            W[:, j:(i+temp_b)] -= E[:, (j-i)][:, None] @ H_inv[j, j:(i+temp_b)][None, :]

        W[:, (i+temp_b):] -= E @ H_inv[i:(i+temp_b), (i+temp_b):]

    if sparsify:
        if PERFORM_QUANTIZATION:
            Q *= M
        else:
            W *= M

    del E
    del H
    del H_inv
    del lambda_

    if sparsify and not PERFORM_QUANTIZATION:
        return W, zeros, scales, M
    elif sparsify and PERFORM_QUANTIZATION:
        return Q, zeros, scales, M
    else:
        return Q, zeros, scales, None

def rtn_quantization(H, W, sparsify=False):
    assert not sparsify
    zeros, scales = get_quantization_params(W)

    W = fake_quantize(W, zeros.unsqueeze(1), scales.unsqueeze(1))
    return W, zeros, scales, None

if PERFORM_RTN_QUANTIZATION:
    if PERFORM_SPARSIFICATION:
        raise ValueError("Cannot perform sparsification with RTN quantization")
    quantize_func = rtn_quantization
else:
    quantize_func = gptq_quantization

################################################################################

# TODO: remove cuda:0 assumption
def quantize_opt_layer(arg_stack, attn_mask, layer, layer_key, outpath=None):
    layer.to("cuda:0")

    def capture_layer_input(module):
        original_fwd = module.forward
        output_container = [(None, 0)]

        def _capture(*args, **kwargs):
            v = original_fwd(*args, **kwargs)

            raw_tensor = args[0]
            if len(raw_tensor.shape) == 3:
                batch_size = raw_tensor.shape[0]
                input_tensor = raw_tensor
            else:
                batch_size = 1
                input_tensor = raw_tensor.unsqueeze(0)
            del raw_tensor

            # higher precision mult since no access to fp16 => fp32 accum
            input_tensor = input_tensor.to(torch.float32)
            proc_tensor = input_tensor.permute([0, 2, 1]) @ input_tensor

            del input_tensor

            mean_tensor, n = output_container[0]
            for i in range(batch_size):
                batch_itm = proc_tensor[i, :, :]
                # online mean to get "average" input tensor
                if (n == 0):
                    mean_tensor = batch_itm
                    n = 1
                else:
                    mean_tensor += (batch_itm - mean_tensor)/(n+1)
                    n += 1
                del batch_itm

            del proc_tensor

            output_container[0] = (mean_tensor, n)
            return v

        module.forward = _capture
        return output_container

    def swap_out_layer(calibration_tensor, module_key, layer):
        original_device = layer.weight.device
        original_weights = layer.weight.to("cuda:0").to(torch.float32)

        weights, zeros, scales, mask = quantize_func(
            calibration_tensor.to(torch.float32).to(original_weights.device),
            original_weights,
            sparsify=PERFORM_SPARSIFICATION
        )

        weights = weights.to(original_device)
        zeros = zeros.to(original_device)
        scales = scales.to(original_device)
        layer.weight.to(original_device)

        del original_weights

        if ONLY_FAKE_QUANTIZE:
            layer.weight.data = weights.clone().to(layer.bias.dtype)
            replacement_layer = layer
        else:
            if PERFORM_SPARSIFICATION:
                replacement_layer = layers.Linear_SparseInt4(layer.in_features, layer.out_features)
            else:
                replacement_layer = layers.Linear_Int4(layer.in_features, layer.out_features)

            replacement_layer.construct(zeros, scales, layer.bias, weights, mask)

            del layer.weight
            del layer.bias
            del layer

        if outpath is not None:
            conv_model.write(replacement_layer, "%d.%s" % (layer_key, module_key))

        del zeros
        del scales
        del weights

        torch.cuda.empty_cache()
        gc.collect()

        return replacement_layer


    # TODO: lift this later for more generic handling
    quantization_targets = [
        "fc1",
        "fc2",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
    ]
    module_accessors = {}
    leaf_targets = {}

    for target_identifier in quantization_targets:
        *sub, leaf_label = target_identifier.split(".")

        target = layer
        for k in sub:
            target = getattr(target, k)

        module_accessors[target_identifier] = {
            "get" : lambda t=target, l=leaf_label: getattr(t, l),
            "set" : lambda x, t=target, l=leaf_label: setattr(t, l, x),
        }

    capture_container = {}
    # set up data capture
    for module, acc in module_accessors.items():
        capture_container[module] = capture_layer_input(acc["get"]())

    print("Capturing calibration tensors")
    for calib in arg_stack:
        calib = calib.to("cuda:0")
        irrelevant = layer.forward(calib, attention_mask=attn_mask)[0]
        del irrelevant

    print("Quantizing and replacing modules")
    capture_container = {k:v[0][0] for k,v in capture_container.items()}
    for module, acc in module_accessors.items():
        new_layer = swap_out_layer(capture_container[module], module, acc["get"]()).to("cuda:0")
        acc["set"](new_layer)

    print("Recalculating calibration tensors")
    # run the whole thing back through the forward pass now to get
    # calibration values that take quantization into account
    final_states = []
    for calib in arg_stack:
        calib = calib.to("cuda:0")
        v = layer.forward(calib, attention_mask=attn_mask)
        v = v[0].to("cpu")
        final_states.append(v)

    print("Cleaning up")
    for v in capture_container.values():
        del v
    capture_container = None

    layer.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

    return final_states


################################################################################

class OPTModelWrapper:
    def __init__(self, fld):
        super().__init__()

        self.base_path = fld
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.base_path)
        self.prefix = "model."

    def embed(self, tokens):
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
            None, attn_mask, (CALIBRATION_BATCH_SIZE, tokens.shape[1]), token_embeds, 0
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

    def write(self, layer, pth):
        pass # no-op; we store everything in self.model

    def finalize_items(self):
        pass # no-op again; everything is in self.model

    @property
    def save_container(self):
        # force clone to prevent view-saving; might be possible to disable this
        return {k:v.clone() for k,v in self.model.state_dict().items()}


class ShardedOPTModelWrapper(OPTModelWrapper):
    # loads in a sharded model one layer at a time, disposing after each layer

    class FakeOPTConfig:
        def __init__(self, dct):
            super().__init__()
            self._dct = dct
        def __getattr__(self, k):
            return self._dct[k]

    def __init__(self, fld):
        super().__init__(fld)

        with open(os.path.join(self.base_path, "pytorch_model.bin.index.json"), "rb") as f:
            self.map = json.load(f)
        with open(os.path.join(self.base_path, "config.json"), "rb") as f:
            self.model_conf = json.load(f)

        self.fake_model_conf = ShardedOPTModelWrapper.FakeOPTConfig(self.model_conf)
        self._pos_embedder = None
        self._token_embedder = None
        self.dont_save_these_keys = set()

        self.save_container = {}

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

        ep_w = self.map["weight_map"]["%sdecoder.embed_positions.weight" % self.prefix]
        et_w = self.map["weight_map"]["%sdecoder.embed_tokens.weight" % self.prefix]

        ep_wts = torch.load(os.path.join(self.base_path, ep_w))
        if (et_w == ep_w):
            et_wts = ep_wts
        else:
            et_wts = torch.load(os.path.join(self.base_path, et_w))

        self._pos_embedder.weight.data = ep_wts["%sdecoder.embed_positions.weight" % self.prefix]
        self._token_embedder.weight.data = et_wts["%sdecoder.embed_tokens.weight" % self.prefix]

    def decoder_layers(self):
        from transformers.models.opt.modeling_opt import OPTDecoderLayer

        # idea: generate each layer as needed ex-nihilo then delete it once used
        n_layers = self.model_conf["num_hidden_layers"]

        for i in range(n_layers):
            prefix = "%sdecoder.layers.%d." % (self.prefix, i)
            containers = {}
            params_to_path = {}

            for k,v in self.map["weight_map"].items():
                if not k.startswith(prefix):
                    continue

                if v not in containers:
                    containers[v] = torch.load(os.path.join(self.base_path, v))

                params_to_path[k] = v # store for convenience since we're already here

            # and now construct the decoding layer
            decode_layer = OPTDecoderLayer(self.fake_model_conf)
            for k, v in params_to_path.items():
                root, *path, leaf = k[len(prefix):].split(".")

                dig = getattr(decode_layer, root)
                for part in path:
                    dig = getattr(dig, part)

                if isinstance(getattr(dig, leaf), torch.nn.Parameter):
                    # slightly non-standard assign since you need to do param.data
                    dig = getattr(dig, leaf)
                    dig.data = containers[v][k].clone()
                else:
                    setattr(dig, leaf, containers[v][k].clone())

            yield decode_layer

            del decode_layer

            memo_keys = list(containers.keys())
            for k in memo_keys:
                del containers[k]
            del containers

    def finalize_items(self):
        # everything that we didn't get before, we're going to load now
        already_hit = set(self.save_container.keys())
        containers = {}
        for k,v in self.map["weight_map"].items():
            if k in already_hit:
                continue

            if k in self.dont_save_these_keys:
                continue

            if v not in containers:
                containers[v] = torch.load(os.path.join(self.base_path, v))

            self.save_container[k] = containers[v][k].to("cpu")

        memo_keys = list(containers.keys())
        for k in memo_keys:
            del containers[k]
        del containers

    def write(self, layer, pth):
        pth = "%sdecoder.layers.%s" % (self.prefix, pth)

        self.save_container["%s.bias" % pth] = layer.bias.data.to("cpu").clone()
        if not ONLY_FAKE_QUANTIZE:
            self.save_container["%s.quantized_weights" % pth] = layer.quantized_weights.data.to("cpu").clone()
            if PERFORM_SPARSIFICATION:
                self.save_container["%s.quantized_mask" % pth] = layer.quantized_mask.data.to("cpu").clone()

            self.dont_save_these_keys.add("%s.weight" % pth)
            self.save_container["%s.zeros" % pth] = layer.zeros.data.to("cpu").clone()
            self.save_container["%s.scales" % pth] = layer.scales.data.to("cpu").clone()
        else:
            self.save_container["%s.weight" % pth] = layer.weight.data.to("cpu").clone()


################################################################################

def _load_from_file(fsrc, n_samples, tokenizer):
    with open(fsrc, "r", encoding="utf-8") as f:
        prompt = f.read()

    iten = tokenizer.encode(prompt)
    starts = np.random.randint(0, len(iten)-CALIBRATION_TOKEN_LENGTH, size=(n_samples,))
    iten = torch.tensor([iten[x:x+CALIBRATION_TOKEN_LENGTH] for x in starts])

    return iten

def _load_from_dataset(dset, n_samples, tokenizer):
    import datasets

    data = datasets.load_dataset(**dset)

    idx = list(range(len(data)))
    random.shuffle(idx)

    tok_set = []
    for _ in range(n_samples):
        while True:
            txt = data[idx.pop()]["text"]
            tokens = tokenizer.encode(txt)

            if len(tokens) < CALIBRATION_TOKEN_LENGTH:
                continue

            last_idx = len(tokens) - CALIBRATION_TOKEN_LENGTH - 1
            start_idx = 0
            if last_idx > 0:
                start_idx = random.randint(0, last_idx)
            tok_set.append(tokens[start_idx:(start_idx+CALIBRATION_TOKEN_LENGTH)])
            break

    return torch.tensor(tok_set)

def prepare_calibration_data(fsettings, n_samples, conv_model):
    if (n_samples % CALIBRATION_BATCH_SIZE) != 0:
        # TODO: lift this constraint
        raise ValueError("samples must be multiple of calibration batch size currently")

    fmode = fsettings["mode"]

    if fmode == "file":
        iten = _load_from_file(fsettings["args"], n_samples, conv_model.tokenizer)
    elif fmode == "dataset":
        iten = _load_from_dataset(fsettings["args"], n_samples, conv_model.tokenizer)
    else:
        raise ValueError("Unknown calibration data load mode %r" % fmode)

    arg_stack = []
    for i in range(0, n_samples, CALIBRATION_BATCH_SIZE):
        val, mask = conv_model.embed(iten[i:(i+CALIBRATION_BATCH_SIZE), ...])

        arg_stack.append(val)

    return arg_stack, mask.to("cuda:0")

################################################################################

if __name__ == "__main__":
    if os.path.exists(SAVE_FILE_PATH):
        raise ValueError("File under %r already exists; aborting." % SAVE_FILE_PATH)

    if CALIBRATION_DATA_SEED is not None:
        random.seed(CALIBRATION_DATA_SEED)
        np.random.seed(CALIBRATION_DATA_SEED)
        torch.manual_seed(CALIBRATION_DATA_SEED)

    if LOAD_MODEL_IS_SHARDED:
        conv_model = ShardedOPTModelWrapper(LOAD_FILE_PATH)
    else:
        conv_model = MemoryOPTModelWrapper(LOAD_FILE_PATH)

    arg_stack, attn_mask = prepare_calibration_data(
        CALIBRATION_DATA_SETTINGS,
        N_SAMPLES,
        conv_model=conv_model
    )

    states = arg_stack

    for i,layer in enumerate(conv_model.decoder_layers()):
        print("Running on layer %d" % i)
        with torch.no_grad():
            new_states = quantize_opt_layer(states, attn_mask, layer, i, outpath=conv_model)

        # clear memory very very thoroughly; any sort of leak = bad
        for a in states:
            del a
        del states

        for t in gc.get_objects():
            if torch.is_tensor(t):
                if t.device.type == "cuda":
                    del t
        gc.collect()
        torch.cuda.empty_cache()

        states = new_states

    conv_model.finalize_items()

    torch.save(conv_model.save_container, open(SAVE_FILE_PATH, "wb"))
    print("Model prepared and saved under %s" % SAVE_FILE_PATH)
