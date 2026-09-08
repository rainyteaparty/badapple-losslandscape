"""Heightmap generation: loss landscape over first-hidden-layer perturbations.

z(i,j) = || d L / d a[:, i, j] ||  -- the small-eps limit of

    z(i,j) = L(head(a + eps * e_ij)) - L(head(a))

where e_ij perturbs spatial location (i,j) across all channels. One backward
pass per batch instead of H*W forward passes.

Filter normalisation (Li et al. sec. 4) carries over from weight space to
activation space as a per-channel scaling by that channel's activation norm.
"""
import numpy as np
import torch
import torch.nn.functional as F

from losses import REGISTRY


def _channel_scales(a, eps=1e-8):
    """Filter-norm analogue: channel c's perturbation is scaled by ||a_c||."""
    s = a.detach().flatten(2).norm(dim=2)              # (B,C)
    return s / (s.mean(1, keepdim=True) + eps)


@torch.enable_grad()
def frame_heights(model, x, loss="entropy", filter_norm=True, loss_kw=None):
    """x: (B,1,448,448) -> (z_grad, z_act) each (B,111,111) float32 numpy."""
    loss_fn = REGISTRY[loss]
    a = model.pre_relu(x)
    a.retain_grad()
    L = loss_fn(model.from_pre_relu(a), **(loss_kw or {}))
    model.zero_grad(set_to_none=True)
    L.sum().backward()                                  # samples are independent (no BN)

    g = a.grad
    if filter_norm:
        g = g * _channel_scales(a)[:, :, None, None]
    z_grad = g.norm(dim=1)
    z_act = F.relu(a.detach()).mean(dim=1)
    return z_grad.detach().cpu().numpy(), z_act.cpu().numpy()


def run_all(model, frames, device, loss="entropy", batch=8, filter_norm=True, loss_kw=None):
    """frames: (N,448,448) -> (N,111,111) grad and act stacks."""
    gs, as_ = [], []
    for i in range(0, len(frames), batch):
        x = torch.from_numpy(frames[i:i + batch]).unsqueeze(1).to(device)
        zg, za = frame_heights(model, x, loss, filter_norm, loss_kw)
        gs.append(zg); as_.append(za)
    return np.concatenate(gs), np.concatenate(as_)


# --- normalisation, blending, temporal stability -----------------------------

def norm_global(stack, lo=1.0, hi=99.0):
    """Fixed scaling from whole-clip percentiles.

    Per-frame min-max makes the terrain pump and breathe on every cut; global
    percentiles let quiet frames stay quiet.
    """
    a, b = np.percentile(stack, [lo, hi])
    return np.clip((stack - a) / (b - a + 1e-8), 0, 1).astype(np.float32)


def blend(z_grad, z_act, w1=1.0, w2=1.0):
    """Gradient term gives ReLU-gated edges; activation term fills interiors."""
    return norm_global(w1 * norm_global(z_grad) + w2 * norm_global(z_act))


def gaussian_blur(stack, sigma=1.0):
    if sigma <= 0:
        return stack
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = stack
    for ax in (1, 2):
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), ax, out)
    return out.astype(np.float32)


def temporal_ema(stack, alpha=0.6):
    """Raw per-frame saliency flickers badly; this is the biggest quality win."""
    out = np.empty_like(stack)
    acc = stack[0].copy()
    for i, f in enumerate(stack):
        acc = alpha * f + (1 - alpha) * acc
        out[i] = acc
    return out
