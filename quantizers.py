import numpy as np
import torch

import int4matmul

################################################################################

def _translate_gs(group_size):
    if group_size in (-1, None):
        return int(1e9)
    return group_size

# TODO: resolve device stuff
def _pack_dense_int4(weights, wt_out=None, mask_out=None):
    weights = weights.T.contiguous().to(torch.int)

    if wt_out is None:
        wt_out = torch.zeros(
            (weights.shape[0] // 8, weights.shape[1]),
            dtype=torch.int,
            device=weights.device
        )

    for i in range(0, weights.shape[0], 8):
        write_idx = i // 8

        v = weights[i, :].clone()
        v |= weights[i+1, :] << 4
        v |= weights[i+2, :] << 8
        v |= weights[i+3, :] << 12
        v |= weights[i+4, :] << 16
        v |= weights[i+5, :] << 20
        v |= weights[i+6, :] << 24
        v |= weights[i+7, :] << 28

        wt_out.data[write_idx, :] = v

    return wt_out, mask_out

def _pack_sparse_int4(weights, mask, wt_out=None, mask_out=None):
    if wt_out is None:
        wt_out = torch.zeros(
            (weights.shape[1] // 16, weights.shape[0]),
            dtype=torch.int,
            device=weights.device
        )
    if mask_out is None:
        mask_out = torch.zeros(
            (weights.shape[1] // 32, weights.shape[0]),
            dtype=torch.int,
            device=weights.device
        )

    int4matmul.weight_matrix_packing(
        wt_out, mask_out,
        weights.contiguous(), mask.contiguous()
    )

    return wt_out, mask_out

def pack_weights(weights, mask, bits, wt_out=None, mask_out=None):
    if (bits == 4):
        if mask is not None:
            return _pack_sparse_int4(weights, mask, wt_out, mask_out)
        else:
            return _pack_dense_int4(weights, wt_out, mask_out)
    else:
        raise ValueError("Unknown weight packing arguments")


################################################################################

class Quantizer(object):
    # basically just a wrapper for methods relating to quantizing (actual
    # numeric quantization [as in setting up the grid], not overarching
    # methods like RTN/GPTQ/etc.

    def __init__(self, bits, group_size):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self._group_tick = _translate_gs(self.group_size)


    def quantize(self, W, M, params):
        raise NotImplementedError

    def unquantize(self, W, M, params):
        raise NotImplementedError

    def solve_params(self, W, M):
        raise NotImplementedError

    def fake_quantize(self, W, M, params):
        return self.unquantize(self.quantize(W, M, params), M, params)

    def pack_weights(self, W, M, params, wt_out=None, mask_out=None):
        raise NotImplementedError


class ZeroPoint(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._must_include_zero = True
        self._max_q = (1 << self.bits) - 1

    def quantize(self, W, M, params):
        zeros, scales = params

        Q = W.clone()
        for i in range(0, W.shape[1], self._group_tick):
            j = i // self._group_tick
            slc = slice(i, i+self._group_tick)

            Q[..., slc] = torch.clamp(
                torch.round(
                    (W[..., slc]/scales[j, :][:, None])
                    + zeros[j, :][:, None]
                ),
                0,
                self._max_q
            )

        if (M is not None):
            Q *= M

        return Q

    def unquantize(self, W, M, params):
        zeros, scales = params
        Wq = W.clone()
        for i in range(0, W.shape[1], self._group_tick):
            j = i // self._group_tick
            slc = slice(i, i+self._group_tick)

            Wq[..., slc] = (
                scales[j, :][:, None]
                * (W[..., slc] - zeros[j, :][:, None])
            )

        if (M is not None):
            Wq *= M
        return Wq

    def solve_params(self, W, M):
        # TODO: some option to auto group-quant here

        if (M is not None):
            W = W[M > 0].reshape((W.shape[0], M.sum() // W.shape[0]))

        mmin = W.min(axis=1).values
        mmax = W.max(axis=1).values

        if self._must_include_zero:
            # zero-point corresponds to the real value 0 when dequantized; the
            # quantization range kinda needs to include 0 for this to be true
            mmin = torch.minimum(mmin, torch.zeros_like(mmin))
            mmax = torch.maximum(mmax, torch.zeros_like(mmax))

        mscl = (mmax - mmin) / self._max_q

        zeros = torch.round(-mmin/mscl)
        scales = mscl

        return zeros[None, :], scales[None, :]

    def pack_weights(self, W, M, params, wt_out=None, mask_out=None):
        Wq = self.quantize(W, M, params).to(torch.uint8)
        return pack_weights(Wq, M, self.bits, wt_out, mask_out)


class ZeroPointButNotReally(ZeroPoint):
    # doesn't include all zeros so questionable if it's really zeropoint
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._must_include_zero = False


class ZPSolver(ZeroPoint):
    # try to directly optimize zero/scale; didn't work well when i tested it
    # but leaving it for the sake of thoroughness
    def solve_params(self, W, M):
        # TODO: implement proper solver
        base_params = super().solve_params(W, M)
        if (M is not None):
            W = W[M > 0].reshape((W.shape[0], M.sum() // W.shape[0]))

        basezeros, basescales = base_params

        def _calc_s(vmin, vmax):
            scales = (vmax - vmin) / self._max_q
            zeros = torch.round(-vmin / scales)

            return zeros[None, :], scales[None, :]

        vmin = None
        vmax = None
        with torch.enable_grad():
            vmin = torch.autograd.Variable(W.min(axis=1).values, requires_grad=True)
            vmax = torch.autograd.Variable(W.max(axis=1).values, requires_grad=True)

            # TODO: test directional sum (non-abs sum then abs)
            loss_fn = torch.nn.MSELoss()
            optim = torch.optim.Adam([vmin, vmax], lr=1e-3)

            base_score = loss_fn(self.fake_quantize(W, None, (torch.round(basezeros), basescales)), W)

            def fn():
                optim.zero_grad()

                zeros, scales = _calc_s(vmin, vmax)

                out = self.fake_quantize(W, None, (zeros, scales))
                loss = loss_fn(out, W)
                loss.backward()
                return loss

            # TODO: early stopping
            for _ in range(100):
                optim.step(fn)

        return _calc_s(vmin, vmax)


class SymmetricExponent(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._max_q = np.power(2, self.bits - 1) - 1
        self._max_val = np.power(2, self._max_q)

    def quantize(self, W, M, params):
        zeros, scales = params

        Q = W.clone()
        for i in range(0, W.shape[1], self._group_tick):
            j = i // self._group_tick
            slc = slice(i, i+self._group_tick)

            # TODO: test mapping 0b0000 to 0
            Wbit = W[..., slc]
            sgn = torch.sign(Wbit - zeros[j, :][:, None])

            Q[..., slc] = torch.clamp(
                torch.round(
                    torch.log2(
                        (Wbit - zeros[j, :][:, None])
                        / (sgn * scales[j, :][:, None])
                    ) - 0.085 # correction factor for log2 rounding
                ),
                0, self._max_q
            )
            Q[..., slc] += (sgn < 0).to(torch.int) << (self.bits - 1)

        if (M is not None):
            Q *= M

        return Q

    def unquantize(self, W, M, params):
        zeros, scales = params
        Wq = W.clone()
        for i in range(0, W.shape[1], self._group_tick):
            j = i // self._group_tick
            slc = slice(i, i+self._group_tick)

            Wbit = W[..., slc].to(torch.int)
            exp = Wbit & ((1 << (self.bits - 1)) - 1)
            sgn = 1 - ((Wbit >> (self.bits - 2)) & 0x2)

            Wq[..., slc] = torch.pow(2, exp) * sgn * scales[j, :][:, None] + zeros[j, :][:, None]

        if (M is not None):
            Wq *= M
        return Wq

    def solve_params(self, W, M):
        # TODO: some option to auto group-quant here

        if (M is not None):
            W = W[M > 0].reshape((W.shape[0], M.sum() // W.shape[0]))

        mmin = W.min(axis=1).values
        mmax = W.max(axis=1).values

        centre = W.mean(axis=1)

        mag_up = mmax-centre
        mag_down = centre-mmin
        mag = torch.maximum(mag_up, mag_down)

        scales = mag / self._max_val

        return centre[None, :], scales[None, :]

    def pack_weights(self, W, M, params, wt_out=None, mask_out=None):
        Wq = self.quantize(W, M, params).to(torch.uint8)
        return pack_weights(Wq, M, self.bits, wt_out, mask_out)


class DynamicExponent(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._linear = ZeroPoint(self.bits, self.group_size)
        self._exponent = SymmetricExponent(self.bits, self.group_size)

        self._q_cross_block_size = 64

    def unquantize(self, W, M, params):
        Q = W.clone()

        zeros, scales = params

        lQ = self._linear.unquantize(W, M, (zeros, scales.abs()))
        eQ = self._exponent.unquantize(W, M, (zeros, scales.abs()))

        for i in range(0, W.shape[1], self._group_tick):
            j = i // self._group_tick
            slc = slice(i, i+self._group_tick)

            exp_msk = scales[j, :] < 0

            Q[~exp_msk, slc] = lQ[~exp_msk, slc]
            Q[exp_msk, slc] = eQ[exp_msk, slc]

        if (M is not None):
            Q *= M

        return Q

    def quantize(self, W, M, params):
        Q = W.clone()

        zeros, scales = params

        lQ = self._linear.quantize(W, M, (zeros, scales.abs()))
        eQ = self._exponent.quantize(W, M, (zeros, scales.abs()))

        for i in range(0, W.shape[1], self._group_tick):
            j = i // self._group_tick
            slc = slice(i, i+self._group_tick)

            exp_msk = scales[j, :] < 0

            Q[~exp_msk, slc] = lQ[~exp_msk, slc]
            Q[exp_msk, slc] = eQ[exp_msk, slc]

        if (M is not None):
            Q *= M

        return Q

    def solve_params(self, W, M):
        lparams = self._linear.solve_params(W, M)
        eparams = self._exponent.solve_params(W, M)

        l_sqerr = (self._linear.fake_quantize(W, M, lparams) - W) ** 2
        e_sqerr = (self._exponent.fake_quantize(W, M, eparams) - W) ** 2

        gstack = []
        for i in range(0, l_sqerr.shape[0], self._q_cross_block_size):
            slc = slice(i, i+self._q_cross_block_size)
            lc = l_sqerr[slc, :].sum()
            ec = e_sqerr[slc, :].sum()

            gstack.append(
                (lparams[0][:, slc], lparams[1][:, slc])
                if lc < ec
                else (eparams[0][:, slc], -eparams[1][:, slc])
            )

        return (
            torch.concat([x[0] for x in gstack], axis=1),
            torch.concat([x[1] for x in gstack], axis=1),
        )

    def pack_weights(self, W, M, params, wt_out=None, mask_out=None):
        Wq = self.quantize(W, M, params).to(torch.uint8)
        return pack_weights(Wq, M, self.bits, wt_out, mask_out)


methods = {
    "zero-point" : ZeroPoint,
    "zero-point-sorta" : ZeroPointButNotReally,
    "exponent-sym" : SymmetricExponent,
    "exponent-dyn" : DynamicExponent,
    "zp-solver" : ZPSolver, # don't use this one; it's bad
}
