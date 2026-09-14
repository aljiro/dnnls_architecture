"""Loss functions shared by all levels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def kl_divergence(stats: dict[str, Tensor], free_bits: float = 0.0) -> Tensor:
    """KL(q || p) between the diagonal Gaussians of the posterior and the conditional prior, per window.

    free_bits > 0: each latent dimension may carry up to that many nats without penalty (Kingma et al.
    2016), so the posterior keeps a fixed information budget instead of being squeezed toward the prior."""
    mu_p, lv_p, mu_q, lv_q = stats["mu_p"], stats["logvar_p"], stats["mu_q"], stats["logvar_q"]
    kl = 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p) - 1)
    if free_bits > 0:
        kl = kl.clamp(min=free_bits)
    return kl.sum(-1).mean()



def centred_cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    """1 - cosine similarity after subtracting the batch mean of the (detached) targets.

    Encoder latents share a large mean vector (raw pairwise cosine 0.31; MiniLM vectors 0.38), so a
    raw cosine loss is nearly blind to the informative part and can be satisfied by shrinking all
    latents toward the mean. Centring removes that solution."""
    target = target.detach()
    mu = target.mean(0, keepdim=True)
    return 1 - F.cosine_similarity(pred - mu, target - mu, dim=-1).mean()


latent_loss = centred_cosine_loss
