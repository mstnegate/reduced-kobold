import torch
import transformers
import int4matmul
import math

################################################################################

# TODO: remove the cuda:0 assumption
class Linear_Int4(torch.nn.Module):
    # HACK: older GPTQ quantizations used "qweight" for the weight tensor and
    #       i chose my name before i knew about that; we'll switch this in a
    #       subclass to provide compatibility;
    #       we only need to support it here since those models were never sparsified
    WEIGHT_KEY = "quantized_weights"

    def __init__(self, in_features, out_features, quantization_bits=4, sparsity=1.0, group_size=-1, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # register as buffers to try and avoid baggage with weight updates
        # and autograd; no plans currently to try and support quantized training
        # TODO: switch zeros to packed int format

        if self.group_size in (None, -1):
            self._n_groups = 1
        else:
            self._n_groups = int(math.ceil(in_features / group_size))

        q_shape = (self._n_groups, out_features)
        if (self._n_groups == 1):
            q_shape = (out_features,)
        self.register_buffer("zeros", torch.zeros(q_shape, dtype=torch.float16))
        self.register_buffer("scales", torch.zeros(q_shape, dtype=torch.float16))

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

        # *don't* call this weights just in case some code gets confused and
        #  thinks we're a legitimate linear layer
        # also note that the weights are stored pre-transposed for convenience
        self.register_buffer(
            self.WEIGHT_KEY,
            torch.zeros(
                math.ceil(sparsity * in_features // (32 / quantization_bits)),
                out_features,
                dtype=torch.int
            )
        )

    def construct(self, zeros, scales, bias, q_weights, mask=None):
        # TODO: move this out into its own thing

        # we don't need to construct into the older format
        assert self.WEIGHT_KEY == "quantized_weights"

        self.zeros = zeros.clone().to(torch.float16).contiguous()
        self.scales = scales.clone().to(torch.float16).contiguous()

        # small detail: if there's only one group, store it as a flat vector
        # to keep compatibility with older models
        if self._n_groups == 1:
            self.zeros = self.zeros[0, :].contiguous()
            self.scales = self.scales[0, :].contiguous()

        if self.bias is not None:
            self.bias = bias.clone().to(torch.float16).contiguous()

        self.quantized_weights.data[...] = q_weights
        self.quantized_weights = self.quantized_weights.to("cuda:0").contiguous()

    def forward(self, x):
        # TODO: CPU forward pass (probably not enabled by default though)

        if len(x.shape) == 3:
            shape = (x.shape[0], x.shape[1], self.out_features)
        elif len(x.shape) == 2:
            # force batch axis to keep kernel coordinates consistent
            shape = (1, x.shape[0], self.out_features)

        with torch.no_grad():
            outs = torch.zeros(shape, dtype=x.dtype, device=x.device)

            int4matmul.quant_int4_linear_mult(
                outs, getattr(self, self.WEIGHT_KEY),
                x if len(x.shape) == 3 else x.view(1, *x.shape),
                self.scales if self._n_groups > 1 else self.scales[None, :],
                self.zeros if self._n_groups > 1 else self.zeros[None, :],
                self.group_size,
                getattr(self, "quantized_mask", None)
            )

            if self.bias is not None:
                outs += self.bias.view((1,1,len(self.bias)))

            # match shape going out
            if len(x.shape) == 3:
                return outs
            elif len(x.shape) == 2:
                return outs.view((outs.shape[1], outs.shape[2]))


class CompatLinear_Int4(Linear_Int4):
    WEIGHT_KEY = "qweight"


class Linear_SparseInt4(Linear_Int4):
    def __init__(self, in_features, out_features, quantization_bits=4, group_size=-1, bias=True):
        super().__init__(
            in_features, out_features,
            quantization_bits=quantization_bits, group_size=group_size,
            sparsity=0.5,
            bias=bias
        )

        self.register_buffer(
            "quantized_mask",
            torch.zeros(in_features // 32, out_features, dtype=torch.int)
        )

    def construct(self, zeros, scales, bias, q_weights, mask):
        super().construct(zeros, scales, bias, q_weights)
        self.quantized_mask.data[...] = mask
