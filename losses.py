"""The candidate L definitions. Each maps logits (B,C) -> per-sample loss (B,).

Entropy is the default: no labels required, and "how much does perturbing here
change the model's uncertainty" is a clean thing to be plotting.
"""
import torch
import torch.nn.functional as F


def entropy(logits, **kw):
    logp = F.log_softmax(logits, -1)
    return -(logp.exp() * logp).sum(-1)


def self_ce(logits, **kw):
    """CE against the unperturbed frame's own argmax -- label-free, sharper."""
    return F.cross_entropy(logits, logits.argmax(-1).detach(), reduction="none")


def fixed_ce(logits, target=0, **kw):
    t = torch.full((logits.shape[0],), target, device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, t, reduction="none")


def neg_max_logit(logits, **kw):
    return -logits.max(-1).values


def margin(logits, **kw):
    """top2 - top1: high when the model is confused."""
    top2 = logits.topk(2, -1).values
    return top2[:, 1] - top2[:, 0]


def free_energy(logits, **kw):
    return -torch.logsumexp(logits, -1)


def logit_l2(logits, **kw):
    return 0.5 * (logits ** 2).sum(-1)


REGISTRY = {
    "entropy": entropy,
    "self_ce": self_ce,
    "fixed_ce": fixed_ce,
    "neg_max_logit": neg_max_logit,
    "margin": margin,
    "free_energy": free_energy,
    "logit_l2": logit_l2,
}
