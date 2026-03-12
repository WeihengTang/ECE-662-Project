"""
DCF (Decomposed Convolutional Filter) layer for PyTorch.

Implements the filter decomposition:
    W[i, j] = sum_{k=1}^{K}  a_k[i, j] * psi_k

where:
    psi_k in R^{l x l}  are filter atoms (basis elements)
    a_k   in R^{c' x c} are atom coefficients
    c', c                are out/in channels, l x l is kernel size

Two forward modes (following the original DCFNet implementation):
    mode0 : two-conv path  — conv with each atom, then 1x1 conv on channel stack
    mode1 : reconstruct-then-conv — explicitly compute W, then standard conv2d

mode1 is the default and matches the mathematical description in the paper exactly.

Reference: Qiu et al., "DCFNet: Deep Neural Network with Decomposed Convolutional
Filters," ICML 2018. https://arxiv.org/abs/1802.04145
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fb import get_fb_bases_tensor, max_fb_bases, calculate_FB_bases


class Conv_DCF(nn.Module):
    """
    2-D convolution with decomposed filters.

    Args:
        in_channels  (int) : Input channels.
        out_channels (int) : Output channels.
        kernel_size  (int) : Spatial size of each atom (must be odd for FB init).
        stride       (int) : Convolution stride. Default 1.
        padding      (int) : Zero-padding. Default 0.
        num_bases    (int) : K — number of atoms.  -1 → use all FB bases.
        bias         (bool): Learnable bias. Default True.
        bases_grad   (bool): If True, atoms are learnable (DCF from scratch).
                             If False, atoms are fixed (e.g. pre-set to FB).
        initializer  (str) : 'FB' or 'random'.
        mode         (str) : 'mode0' or 'mode1'. Default 'mode1'.
    """

    __constants__ = ['in_channels', 'out_channels', 'kernel_size',
                     'stride', 'padding', 'num_bases', 'bases_grad', 'mode']

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, num_bases=-1, bias=True,
                 bases_grad=False, dilation=1, initializer='FB', mode='mode1'):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.bases_grad   = bases_grad
        assert mode in ('mode0', 'mode1')
        self.mode = mode
        assert initializer in ('FB', 'random')

        # ── Build initial bases ──────────────────────────────────────────────
        if initializer == 'FB':
            if kernel_size % 2 == 0:
                raise ValueError("Kernel size must be odd for FB initialisation.")
            L1 = (kernel_size - 1) // 2
            base_np, _, _ = calculate_FB_bases(L1)  # (k*k, total)
            total_bases = base_np.shape[1]
            if num_bases == -1:
                num_bases = total_bases
            elif num_bases > total_bases:
                raise ValueError(
                    f"Requested {num_bases} FB bases but only {total_bases} "
                    f"available for kernel_size={kernel_size}."
                )
            base_np = base_np[:, :num_bases]               # (k*k, K)
            base_np = base_np.T.reshape(num_bases, kernel_size, kernel_size)
            base_np = base_np.astype(np.float32)
        else:  # random
            if num_bases <= 0:
                raise ValueError("num_bases must be > 0 for random initialisation.")
            base_np = np.random.randn(num_bases, kernel_size, kernel_size).astype(np.float32)

        self.num_bases = num_bases

        # ── Register bases ───────────────────────────────────────────────────
        if bases_grad:
            self.bases = Parameter(torch.tensor(base_np), requires_grad=True)
        else:
            self.register_buffer('bases', torch.tensor(base_np, requires_grad=False))

        # ── Atom coefficients  a_k in R^{c' x c} ─────────────────────────────
        # Shape: (out_channels, in_channels, num_bases)
        self.weight = Parameter(torch.Tensor(out_channels, in_channels, num_bases))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.num_bases
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # ── Forward helpers ──────────────────────────────────────────────────────

    def forward(self, x):
        if self.mode == 'mode1':
            return self._forward_mode1(x)
        return self._forward_mode0(x)

    def _forward_mode1(self, x):
        """Reconstruct full filter tensor, then standard conv2d."""
        # weight: (C_out, C_in, K)  bases: (K, kH, kW)
        # rec_kernel[out, in, kH, kW] = sum_k weight[out, in, k] * bases[k, kH, kW]
        rec_kernel = torch.einsum('oik,khw->oihw', self.weight, self.bases)
        return F.conv2d(x, rec_kernel, self.bias, self.stride, self.padding, self.dilation)

    def _forward_mode0(self, x):
        """Two-conv path: convolve with each atom, then combine with 1x1 conv."""
        N, C, H, W = x.size()
        # Expand input: (N*C, 1, H, W)
        x_exp = x.view(N * C, 1, H, W)
        # Conv with all atoms at once: bases (K,1,kH,kW) → (N*C, K, H', W')
        bases_4d = self.bases.unsqueeze(1)  # (K, 1, kH, kW)
        feat = F.conv2d(x_exp, bases_4d, None, self.stride, self.padding, self.dilation)
        H2, W2 = feat.shape[2], feat.shape[3]
        feat = feat.view(N, C * self.num_bases, H2, W2)
        # 1x1 conv to mix: weight (C_out, C_in*K, 1, 1)
        weight_4d = self.weight.view(self.out_channels, self.in_channels * self.num_bases, 1, 1)
        return F.conv2d(feat, weight_4d, self.bias)

    # ── Parameter count utilities ────────────────────────────────────────────

    def num_trainable_params(self):
        n = self.weight.numel()
        if self.bias is not None:
            n += self.bias.numel()
        if self.bases_grad:
            n += self.bases.numel()
        return n

    def extra_repr(self):
        return (
            f"in={self.in_channels}, out={self.out_channels}, "
            f"k={self.kernel_size}, K={self.num_bases}, "
            f"bases_grad={self.bases_grad}, mode={self.mode}"
        )
