"""Pure-PyTorch reference for the tests, copied from mamba_ssm v2.2.4
(state-spaces/mamba, Apache-2.0): selective_scan_ref and the Mamba.forward slow path.
Sequential over the sequence, so only used to check selective_scan_chunked."""
import torch
import torch.nn.functional as F
from einops import rearrange


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """u, delta, z: (b d l)   A: (d n)   B, C: (b n l)   D, delta_bias: (d)  ->  (b d l)"""
    u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    x = A.new_zeros((batch, dim, dstate))
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        ys.append(torch.einsum('bdn,bn->bd', x, C[:, :, i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    return out


def mamba_slow_forward(m, hidden_states):
    """mamba_ssm.Mamba.forward, non-fused path, run on an object with the same attributes."""
    batch, seqlen, dim = hidden_states.shape
    xz = rearrange(m.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
                   "d (b l) -> b d l", l=seqlen)
    if m.in_proj.bias is not None:
        xz = xz + rearrange(m.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
    A = -torch.exp(m.A_log.float())
    x, z = xz.chunk(2, dim=1)
    x = F.silu(m.conv1d(x)[..., :seqlen])
    x_dbl = m.x_proj(rearrange(x, "b d l -> (b l) d"))
    dt, B, C = torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = m.dt_proj.weight @ dt.t()
    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    y = selective_scan_ref(x, dt, A, B, C, m.D.float(), z=z,
                           delta_bias=m.dt_proj.bias.float(), delta_softplus=True)
    return m.out_proj(rearrange(y, "b d l -> b l d"))
