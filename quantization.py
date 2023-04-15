import gc
import json
import os
import random

import numpy as np
import torch

import layers
import quantizers
import models

################################################################################
# general config stuff

# stuff you will probably want to change
QUANTIZATION_BITS = 4
PERFORM_QUANTIZATION = True
PERFORM_SPARSIFICATION = True
GROUP_QUANTIZATION_SIZE = -1 # disable with -1 or None (either works)
QUANTIZATION_METHOD = "zero-point" # see quantizers.py for options

LOAD_FILE_PATH = "/pth/to/your/model/folder"
MODEL_TYPE = "your-model-type" # "opt", "llama", "gpt_neox"
LOAD_MODEL_IS_SHARDED = True


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

# experimental options
# quantizes on a per-layer instead of decode-unit level when passing data in;
# turning it on increases runtime and should theoretically improve accuracy
# (it didn't for me though)
PERFORM_STAGED_CALIBRATION = True
# makes a best guess at sparsifying a group quantization bucket before
# calculating quantization params; theoretically improves accuracy for if it
# predicts well and worsens it if it doesn't; experimental
# set to 0 to disable
GUESS_SPARSE_PERCENTAGE = 0.5
# based off actorder from reference GPTQ impl; this rearranges quantization
# to happen in descending activation magnitude; only recommend using this for
# mostly-unstructured configurations (no group quantization, no sparsity)
PERFORM_ACTIVATION_SORTING = True

# more technical stuff with gptq/sparsegpt; don't change these unless you
# know what you're doing
GPTQ_BLOCK_SIZE = 128           # suggested by gptq paper on p5
SPARSEGPT_BLOCK_SPARSITY = 32   # keep at 32 unless benchmarking fp16 sparse
GPTQ_LAMBDA_FACTOR = 0.01       # suggested by gptq paper on p5
CALIBRATION_TOKEN_LENGTH = 2048 # mostly model-dependent

# debug options; don't touch unless you know what you're doing
CHECK_EIGS_JUST_TO_BE_SURE = False
ONLY_FAKE_QUANTIZE = False
PERFORM_RTN_QUANTIZATION = False
REALLY_LARGE_NUMBER = int(1e9)

################################################################################

if PERFORM_SPARSIFICATION and not PERFORM_QUANTIZATION:
    print("WARNING: there is no model or kernel support currently for sparse "
          + "fp16; you will _NOT_ see memory or speed improvements. This is "
          + "purely for benchmarking purposes.")

    if not ONLY_FAKE_QUANTIZE:
        raise ValueError("Only fake quantization/sparsification supported.")

# 2-bit technically supported but so niche i don't want to support it
if QUANTIZATION_BITS not in (4, 3) and not ONLY_FAKE_QUANTIZE:
    raise ValueError("Only 3-bit and 4-bit quantization supported currently.")

if GROUP_QUANTIZATION_SIZE not in (-1, None) and PERFORM_RTN_QUANTIZATION:
    raise ValueError("Group RTN not supported currently.")

if PERFORM_RTN_QUANTIZATION and PERFORM_SPARSIFICATION:
    # should probably impl mag sparsification at some point as a baseline
    raise ValueError("Cannot perform sparsification with RTN quantization")

################################################################################

def _get_quantizer():
    G_S = GROUP_QUANTIZATION_SIZE

    if (G_S == -1) or (G_S is None):
        # to avoid special-casing logic, just set group size to an absurd number
        G_S = int(1e9)

    if QUANTIZATION_METHOD not in quantizers.methods:
        raise ValueError("Invalid quantization method %r; choose one of %r"
            % (QUANTIZATION_METHOD, quantizer.methods.keys()))
    quantizer_method = quantizers.methods[QUANTIZATION_METHOD](QUANTIZATION_BITS, GROUP_QUANTIZATION_SIZE)

    return quantizer_method


def gptq_quantization(H, W, sparsify=False):
    # config parameters:
    # block size; size of the chunks we iterate by when quantization/sparisfying
    B = GPTQ_BLOCK_SIZE
    # block sparsity structure; this is size of matrix spanned, not weights
    # (i.e. it's a (B_s//2):B_s sparsity structure)
    B_s = SPARSEGPT_BLOCK_SPARSITY
    lambda_factor = GPTQ_LAMBDA_FACTOR

    quantizer_method = _get_quantizer()
    G_S = quantizer_method.group_size

    if (G_S == -1) or (G_S is None):
        # to avoid special-casing logic, just set group size to some absurd
        # number (i.e. treat no group quant as group quant w/ one group)
        G_S = B * REALLY_LARGE_NUMBER

    # technically not impossible, just a headache
    assert (B <= G_S), "Group quantization chunk size cannot be smaller than quantization chunk size"
    # see above
    assert (G_S % B == 0), "Group quantization chunk size must be multiple of quantization chunk size"

    if PERFORM_SPARSIFICATION:
        assert G_S >= B_s, "Group quantization size must be greater than sparsity size"
        assert (G_S % B_s == 0), "Group quantization size must be multiple of sparsity size"

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

    zero_stack = []
    scale_stack = []

    H *= 2 # H = 2XX^T+\lambdaI, not XX^T+\lambdaI

    diag_idxes = list(range(H.shape[-1]))

    if PERFORM_ACTIVATION_SORTING:
        if (G_S >= REALLY_LARGE_NUMBER):
            g_inc = 1
        else:
            g_inc = G_S

        act_scores = H[diag_idxes, diag_idxes]
        aN = len(act_scores)

        # TODO: genericize this logic to N-level block sorting
        if g_inc == 1:
            if not sparsify:
                ordering = torch.argsort(act_scores, descending=True)
            else:
                grped_act = act_scores.reshape(aN//B_s, B_s)
                grp_scores = grped_act.sum(axis=1)
                gp_ordering = torch.argsort(grp_scores, descending=True)

                increment = gp_ordering * B_s

                ordering = (grped_act.argsort(axis=-1, descending=True)[gp_ordering, ...] + increment[..., None]).flatten()
        else:
            # activation sorting swaps over the dimension we group-quantize
            # *and* sparsify over; we need to go in multiple steps where we
            # sort over smaller and smaller steps
            if not sparsify:
                grped_act = act_scores.reshape(aN//g_inc, g_inc)
                grp_scores = grped_act.sum(axis=1)
                gp_ordering = torch.argsort(grp_scores, descending=True)

                increment = gp_ordering * g_inc

                ordering = (grped_act.argsort(axis=-1, descending=True)[gp_ordering, ...] + increment[..., None]).flatten()
            else:
                grped_act = act_scores.reshape(aN//g_inc, g_inc//B_s, B_s)
                grp_scores = grped_act.sum(axis=[1,2])
                gp_ordering = torch.argsort(grp_scores, descending=True)

                subgp_ordering = grped_act.sum(axis=2)[gp_ordering, ...].argsort(axis=-1, descending=True)

                increment = (gp_ordering * g_inc)[:, None] + (subgp_ordering * B_s)
                sub_increment = (gp_ordering * subgp_ordering.shape[-1])[:, None] + subgp_ordering

                # perform sorting within each sparse block
                sparse_ordering = grped_act.reshape(aN//B_s, B_s).argsort(axis=-1, descending=True)
                # rearrange the sparse orderings into position then shape it back
                sparse_ordering = sparse_ordering[sub_increment.reshape(aN//B_s, 1), :]
                # properly shape it back and apply block-level offsets
                ordering = sparse_ordering.reshape(grped_act.shape) + increment[..., None]
                ordering = ordering.flatten()

        W = W[:, ordering]
        H = H[ordering][:, ordering]

    lambda_ = H.diagonal().mean(axis=0) * lambda_factor

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
            def _sparse_mask(s, e, pct):
                slc = slice(s, e)
                H_inv_d = torch.diagonal(H_inv[slc, slc])
                scores = W[:, slc]**2 / H_inv_d**2

                start_idx = int(scores.shape[1] * pct)

                g_mask = torch.zeros_like(scores)
                g_mask.scatter_(1, torch.argsort(scores)[:, start_idx:], 1)

                return g_mask

            if sparsify:
                if (j % B_s) == 0:
                    g_mask = _sparse_mask(j, j+B_s, 0.5)
                    M[:, j:(j+B_s)] = g_mask.to(M.dtype)
                    assert((M[:, j:(j+B_s)].sum(axis=1) == (B_s // 2)).all())

                    del g_mask

            if (j % G_S) == 0:
                if sparsify and B_s < G_S and GUESS_SPARSE_PERCENTAGE > 0.0:
                    # make a guess at where the sparsified entires will be
                    g_mask = _sparse_mask(j+B_s, j+G_S, GUESS_SPARSE_PERCENTAGE)
                    sub_M = M[:, j:(j+G_S)]
                    sub_M[:, B_s:] = g_mask
                    del g_mask

                    new_zeros, new_scales = quantizer_method.solve_params(
                        W[:, j:(j+G_S)], sub_M)
                    del sub_M
                else:
                    new_zeros, new_scales = quantizer_method.solve_params(
                        W[:, j:(j+G_S)], M[:, j:(j+G_S)] if sparsify else None)

                zero_stack.append(new_zeros)
                scale_stack.append(new_scales)

            W_q = quantizer_method.fake_quantize(
                W[:, j][:, None],
                M[:, j][:, None] if sparsify else None,
                (zero_stack[-1], scale_stack[-1])
            )
            Q[:, j] = W_q[:, 0]

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

    zeros = torch.concat(zero_stack, axis=0)
    scales = torch.concat(scale_stack, axis=0)

    if PERFORM_ACTIVATION_SORTING:
        reverse_ordering = torch.argsort(ordering)
        Q = Q[:, reverse_ordering]
        W = W[:, reverse_ordering]
        if sparsify:
            M = M[:, reverse_ordering]

        if g_inc > 1:
            reverse_gp_ordering = torch.argsort(gp_ordering)
            zeros = zeros[reverse_gp_ordering, :]
            scales = scales[reverse_gp_ordering, :]

    if sparsify and not PERFORM_QUANTIZATION:
        return W, zeros, scales, M
    elif sparsify and PERFORM_QUANTIZATION:
        return Q, zeros, scales, M
    else:
        return Q, zeros, scales, None

def rtn_quantization(H, W, sparsify=False):
    assert not sparsify

    quantizer_method = _get_quantizer()
    zeros, scales = quantizer_method.solve_params(W, None)

    W = quantizer_method.fake_quantize(W, None, (zeros, scales))

    return W, zeros, scales, None

if PERFORM_RTN_QUANTIZATION:
    quantize_func = rtn_quantization
else:
    quantize_func = gptq_quantization

################################################################################

# TODO: remove cuda:0 assumption
def quantize_opt_layer(arg_stack, gen_kwargs, layer, layer_key, conv_model=None):
    layer.to("cuda:0")

    class StopForwardPass(Exception):
        pass

    def capture_layer_input(module, label):
        original_fwd = module.forward
        output_container = [(None, 0)]

        def _capture(*args, **kwargs):
            if label not in pending_capture_targets:
                return original_fwd(*args, **kwargs)

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

            if PERFORM_STAGED_CALIBRATION:
                raise StopForwardPass()

            v = original_fwd(*args, **kwargs)
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
                layer_class = layers.Linear_SparseInt4
            else:
                layer_class = layers.Linear_Int4

            replacement_layer = layer_class(
                layer.in_features, layer.out_features,
                quantization_bits=QUANTIZATION_BITS, group_size=GROUP_QUANTIZATION_SIZE,
                bias=conv_model.HAS_BIAS)

            quantizer_method = _get_quantizer()
            n_weights, n_mask = quantizer_method.pack_weights(weights, mask, (zeros, scales))
            replacement_layer.construct(zeros, scales, layer.bias, n_weights, n_mask)

            del layer.weight
            del layer.bias
            del layer
            del n_weights
            if n_mask is not None:
                del n_mask

        if conv_model is not None:
            conv_model.write(
                replacement_layer,
                "%d.%s" % (layer_key, module_key),
                PERFORM_SPARSIFICATION,
                ONLY_FAKE_QUANTIZE
            )

        del zeros
        del scales
        del weights

        torch.cuda.empty_cache()
        gc.collect()

        return replacement_layer

    quantization_targets = conv_model.QUANTIZATION_NODES
    pending_capture_targets = set()
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
            "label" : target_identifier,
        }
        pending_capture_targets.add(target_identifier)

    capture_container = {}
    # set up data capture
    for module, acc in module_accessors.items():
        capture_container[module] = capture_layer_input(acc["get"](), label=acc["label"])

    while pending_capture_targets:
        # perform forward pass on layer
        print("Capturing calibration tensors")
        for calib in arg_stack:
            calib = calib.to("cuda:0")
            try:
                irrelevant = layer.forward(calib, **gen_kwargs)[0]
                del irrelevant
            except StopForwardPass:
                pass

        captured = {k:v[0][0] for k,v in capture_container.items() if v[0][0] is not None}
        for module, calib_tensor in captured.items():
            acc = module_accessors[module]

            new_layer = swap_out_layer(calib_tensor, module, acc["get"]()).to("cuda:0")
            acc["set"](new_layer)
            pending_capture_targets.remove(module)
            del capture_container[module]

    print("Recalculating calibration tensors")
    # run the whole thing back through the forward pass now to get
    # calibration values that take quantization into account
    final_states = []
    for i, calib in enumerate(arg_stack):
        calib = calib.to("cuda:0")
        v = layer.forward(calib, **gen_kwargs)[0].contiguous()
        v = v.clone().to("cpu")

        if torch.isnan(v).any():
            raise ValueError("Input tensor %d had nan element[s] on forward pass" % i)

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
        val, gen_kwargs = conv_model.embed(iten[i:(i+CALIBRATION_BATCH_SIZE), ...], batch_size=CALIBRATION_BATCH_SIZE)

        arg_stack.append(val)

    return arg_stack, gen_kwargs

################################################################################

if __name__ == "__main__":
    if os.path.exists(SAVE_FILE_PATH):
        raise ValueError("File under %r already exists; aborting." % SAVE_FILE_PATH)

    if CALIBRATION_DATA_SEED is not None:
        random.seed(CALIBRATION_DATA_SEED)
        np.random.seed(CALIBRATION_DATA_SEED)
        torch.manual_seed(CALIBRATION_DATA_SEED)

    conv_model = models.get_loader(MODEL_TYPE, LOAD_MODEL_IS_SHARDED)(LOAD_FILE_PATH)

    arg_stack, gen_kwargs = prepare_calibration_data(
        CALIBRATION_DATA_SETTINGS,
        N_SAMPLES,
        conv_model=conv_model
    )

    states = arg_stack

    for i,layer in enumerate(conv_model.decoder_layers()):
        print("Running on layer %d" % i)
        with torch.no_grad():
            new_states = quantize_opt_layer(states, gen_kwargs, layer, i, conv_model=conv_model)

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
