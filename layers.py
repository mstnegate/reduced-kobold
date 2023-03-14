import torch
import transformers
import int4matmul

################################################################################

# only int4 quantization supported currently; need to write separate kernels
# for other quantization sizes
MAX_QUANTIZATION_VALUE = (1 << 4) - 1

# standard RTN quant/dequant
def unquantize(block, zeros, scales):
    # return scales*block + zeros
    return scales * (block - zeros)

def quantize(block, zeros, scales):
    return torch.clamp(
        torch.round( (block/scales) + zeros ),
        0,
        MAX_QUANTIZATION_VALUE
    )

def fake_quantize(block, zeros, scales):
    return unquantize(quantize(block, zeros, scales), zeros, scales)

################################################################################

# TODO: remove the cuda:0 assumption
class Linear_Int4(torch.nn.Module):
    # HACK: older GPTQ quantizations used "qweight" for the weight tensor and
    #       i chose my name before i knew about that; we'll switch this in a
    #       subclass to provide compatibility;
    #       we only need to support it here since those models were never sparsified
    WEIGHT_KEY = "quantized_weights"

    def __init__(self, in_features, out_features, weight_packing_factor=8, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # register as buffers to try and avoid baggage with weight updates
        # and autograd; no plans currently to try and support quantized training
        self.register_buffer("zeros", torch.zeros(out_features, dtype=torch.float16))
        self.register_buffer("scales", torch.zeros(out_features, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

        # *don't* call this weights just in case some code gets confused and
        #  thinks we're a legitimate linear layer
        # also note that the weights are stored pre-transposed for convenience
        self.register_buffer(
            self.WEIGHT_KEY,
            torch.zeros(in_features // weight_packing_factor, out_features, dtype=torch.int)
        )

    def construct(self, zeros, scales, bias, fake_q_weights, mask=None):
        # we don't need to construct into the older format
        assert self.WEIGHT_KEY == "quantized_weights"

        self.zeros = zeros.clone().to(torch.float16).contiguous()
        self.scales = scales.clone().to(torch.float16).contiguous()
        if self.bias is not None:
            self.bias = bias.clone().to(torch.float16).contiguous()

        fake_q_weights = quantize(fake_q_weights, self.zeros[:, None], self.scales[:, None]).to(torch.int).T

        for i in range(0, fake_q_weights.shape[0], 8):
            write_idx = i // 8

            v = fake_q_weights[i, :].clone()
            v |= fake_q_weights[i+1, :] << 4
            v |= fake_q_weights[i+2, :] << 8
            v |= fake_q_weights[i+3, :] << 12
            v |= fake_q_weights[i+4, :] << 16
            v |= fake_q_weights[i+5, :] << 20
            v |= fake_q_weights[i+6, :] << 24
            v |= fake_q_weights[i+7, :] << 28

            self.quantized_weights.data[write_idx, :] = v

        self.quantized_weights = self.quantized_weights.to("cuda:0")

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
                self.scales,
                self.zeros,
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
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, weight_packing_factor=16, bias=bias)

        self.register_buffer(
            "quantized_mask",
            torch.zeros(in_features // 32, out_features, dtype=torch.int)
        )

    def construct(self, zeros, scales, bias, fake_q_weights, mask):
        self.zeros = zeros.clone().to(torch.float16).contiguous()
        self.scales = scales.clone().to(torch.float16).contiguous()
        if self.bias is not None:
            self.bias = bias.clone().to(torch.float16).contiguous()

        # don't transpose
        fake_q_weights = quantize(fake_q_weights, self.zeros[:, None], self.scales[:, None]).to(torch.uint8)

        self.quantized_weights = self.quantized_weights.to("cuda:0")
        self.quantized_mask = self.quantized_mask.to("cuda:0")

        int4matmul.weight_matrix_packing(
            self.quantized_weights, self.quantized_mask,
            fake_q_weights.contiguous(), mask.contiguous()
        )

