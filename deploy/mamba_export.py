"""Export-friendly Mamba block for ONNX / TensorRT.

mamba_ssm.Mamba runs its selective scan and causal conv1d through custom CUDA kernels
that neither torch.onnx.export nor TensorRT understand. MambaExport keeps the exact
parameter names of mamba_ssm.Mamba (a checkpoint loads unchanged) but computes the
forward pass with plain tensor ops:

  * causal depthwise conv1d  -> nn.Conv1d with left padding, sliced to seqlen
  * selective scan           -> one of two paths, chosen by MambaExport.SCAN:
      "closed_form": chunked closed form in plain tensor ops (selective_scan_chunked);
                     builds with stock TensorRT layers but materialises T x T decay
                     matrices per (channel, state) -> memory-bound on the Jetson.
      "plugin":      a single custom ONNX node per block (bios::SelectiveScan) that the
                     TensorRT plugin in deploy/plugin/ implements as one CUDA kernel doing
                     the real recurrence (O(L) traffic, fp32 state). In PyTorch the node
                     falls back to selective_scan_chunked, so the model still runs and
                     tests on a machine without CUDA.

Selective scan, per channel d and state n, with delta_t > 0 and A < 0:
    h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
    y_t = sum_n C_t[n] * h_t[n]  (+ D * u_t),  then y *= silu(z)
Inside a chunk of T tokens the recurrence unrolls to
    h_t = exp(L_t) * h_0 + sum_{s<=t} exp(L_t - L_s) * delta_s * B_s * u_s,
    L_t = sum_{r<=t} delta_r * A
a masked (T x T) matrix product per (d, n). The carries h_0 between chunks obey the
same recurrence over chunk-end states and are solved the same way (two levels, no loop).
Every exponent is <= 0, so nothing overflows, also in fp16. The cumulative sum L is a
matmul with a constant lower-triangular matrix, so the exported graph only needs
MatMul, Exp, Where, Mul, Add, ReduceSum, Softplus and Sigmoid -- all native TensorRT layers.
"""

import importlib.util
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIG = {"num_classes": 1, "input_channels": 3, "c_list": [16, 32, 48, 64, 96, 128]}


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                           delta_softplus=True, chunk=16):
    """u, delta, z: (b, d, l)   A: (d, n)   B, C: (b, l, n)   D, delta_bias: (d,)
    Returns y: (b, d, l). Same semantics as mamba_ssm's selective_scan_fn.

    Two-level closed form, no sequential loop at all:
      level 1: every chunk of T tokens in parallel, assuming a zero state at its start;
      level 2: the chunk-end states form the same recurrence over nc = l / T steps,
               solved with the same masked-matmul trick; the resulting entering state of
               each chunk is then folded back in with exp(L_t).
    Peak intermediate is (b, d, n, l, T) -- 16 tokens per chunk keeps that under ~70 MB
    for the longest (4096-token) MambaLiteUNet sequences."""
    b, d, l = (int(v) for v in u.shape)     # python ints: keeps the traced ONNX graph static
    n = int(A.shape[1])
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, d, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    T = min(chunk, l)
    pad = (-l) % T                                          # zero-pad to a multiple of T
    if pad:
        # delta = 0 on padded steps: decay exp(0) = 1 and no input, so they are inert
        u_p, delta_p = F.pad(u, (0, pad)), F.pad(delta, (0, pad))
        B_p, C_p = F.pad(B, (0, 0, 0, pad)), F.pad(C, (0, 0, 0, pad))
    else:
        u_p, delta_p, B_p, C_p = u, delta, B, C
    lp = l + pad
    nc = lp // T
    dA = (delta_p.unsqueeze(-1) * A.view(1, d, 1, n)).transpose(2, 3)          # (b, d, n, lp) <= 0
    dBu = ((delta_p * u_p).unsqueeze(-1) * B_p.unsqueeze(1)).transpose(2, 3)   # (b, d, n, lp)
    Cn = C_p.unsqueeze(1).transpose(2, 3)                                        # (b, 1, n, lp)
    neg = u.new_tensor(-1e4)                                # exp(-1e4) == 0, safe in fp16
    tril_T = torch.tril(torch.ones(T, T, dtype=u.dtype, device=u.device))
    tril_c = torch.tril(torch.ones(nc, nc, dtype=u.dtype, device=u.device))

    # level 1: within-chunk states with zero entering state
    dA_r = dA.reshape(b, d, n, nc, T)
    L = dA_r @ tril_T.t()                                   # (b, d, n, nc, T) cumulative log-decay
    diff = L.unsqueeze(-1) - L.unsqueeze(-2)                # (b, d, n, nc, T, T): [t, s] = L_t - L_s
    decay = torch.exp(torch.where(tril_T.bool(), diff, neg))
    x_r = dBu.reshape(b, d, n, nc, T, 1)
    h_intra = (decay @ x_r).squeeze(-1)                     # (b, d, n, nc, T)

    # level 2: carry across chunks
    e = h_intra[..., -1]                                    # (b, d, n, nc) chunk-end states
    G = L[..., -1] @ tril_c.t()                             # (b, d, n, nc) cumulative chunk log-decay
    diff2 = G.unsqueeze(-1) - G.unsqueeze(-2)               # (b, d, n, nc, nc): [c, c'] = G_c - G_c'
    decay2 = torch.exp(torch.where(tril_c.bool(), diff2, neg))
    S = (decay2 @ e.unsqueeze(-1)).squeeze(-1)              # (b, d, n, nc) true state at chunk ends
    S_prev = torch.cat([S.new_zeros(b, d, n, 1), S[..., :-1]], dim=-1)   # state entering each chunk
    h = h_intra + torch.exp(L) * S_prev.unsqueeze(-1)       # (b, d, n, nc, T)

    y = (h.reshape(b, d, n, lp) * Cn).sum(2)                # (b, d, lp)
    y = y[..., :l]
    if D is not None:
        y = y + u * D.view(1, d, 1)
    if z is not None:
        y = y * F.silu(z)
    return y


PLUGIN_DOMAIN = "bios"
PLUGIN_OP = "SelectiveScan"
PLUGIN_VERSION = "1"


class SelectiveScanFn(torch.autograd.Function):
    """One selective scan as a custom ONNX op for the TensorRT plugin.

    Contract (all float32, matches deploy/plugin/selective_scan_plugin.cu):
        inputs  u (b, d, l)  delta (b, d, l)  A (d, n)  B (b, l, n)  C (b, l, n)
                D (d)  z (b, d, l)  dt_bias (d)
        output  y (b, d, l) = (scan(u, softplus(delta + dt_bias), A, B, C) + D * u) * silu(z)
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, z, dt_bias):
        return selective_scan_chunked(u, delta, A, B, C, D, z=z, delta_bias=dt_bias,
                                      delta_softplus=True, chunk=16)

    @staticmethod
    def symbolic(g, u, delta, A, B, C, D, z, dt_bias):
        out = g.op(f"{PLUGIN_DOMAIN}::{PLUGIN_OP}", u, delta, A, B, C, D, z, dt_bias,
                   delta_softplus_i=1, plugin_version_s=PLUGIN_VERSION, plugin_namespace_s="")
        out.setType(u.type())
        return out


class MambaExport(nn.Module):
    """Drop-in for mamba_ssm.Mamba (same constructor kwargs, same parameter names)."""

    CHUNK = 16
    SCAN = "closed_form"     # or "plugin" (see module docstring)

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 conv_bias=True, bias=False, **_ignored):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                                            .repeat(self.d_inner, 1)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (b, l, d_model) -> (b, l, d_model)"""
        l = int(hidden_states.shape[1])
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)        # (b, l, d_inner) each
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :l])        # causal conv, (b, d_inner, l)
        x_dbl = self.x_proj(x.transpose(1, 2))                     # (b, l, R + 2n)
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.linear(dt, self.dt_proj.weight).transpose(1, 2)  # (b, d_inner, l); bias added in scan
        A = -torch.exp(self.A_log.float())
        if self.SCAN == "plugin":
            y = SelectiveScanFn.apply(x, delta, A, Bm, Cm, self.D.float(), z.transpose(1, 2),
                                      self.dt_proj.bias.float())
        else:
            y = selective_scan_chunked(x, delta, A, Bm, Cm, self.D.float(), z=z.transpose(1, 2),
                                       delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
                                       chunk=self.CHUNK)
        return self.out_proj(y.transpose(1, 2))


def install_shim():
    """Let `from mamba_ssm import Mamba` succeed on machines without mamba_ssm (e.g. a Mac)."""
    if "mamba_ssm" not in sys.modules and importlib.util.find_spec("mamba_ssm") is None:
        fake = types.ModuleType("mamba_ssm")
        fake.Mamba = MambaExport
        sys.modules["mamba_ssm"] = fake


def load_state(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return state["model_state_dict"] if "model_state_dict" in state else state


def build_export_model(ckpt_path=None, chunk=16, scan="closed_form"):
    """MambaLiteUNet with every Mamba block replaced by MambaExport; weights loaded strictly.
    scan: "closed_form" (stock ONNX ops) or "plugin" (bios::SelectiveScan custom nodes)."""
    if scan not in ("closed_form", "plugin"):
        raise ValueError(f"scan must be 'closed_form' or 'plugin', got {scan!r}")
    install_shim()
    import models.MambaLiteUNet as M

    class _MambaExport(MambaExport):
        CHUNK = chunk
        SCAN = scan

    original = M.Mamba
    M.Mamba = _MambaExport
    try:
        model = M.MambaLiteUNet(**MODEL_CONFIG)
    finally:
        M.Mamba = original
    if ckpt_path is not None:
        model.load_state_dict(load_state(ckpt_path), strict=True)
    return model.eval()
