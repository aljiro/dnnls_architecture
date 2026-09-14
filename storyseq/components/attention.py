"""Attention, selection and structured heads (Levels 4, 6, 7).

Attention interface (over the sequence): forward(sequence [B, K, H], h [B, H]) -> weights [B, K] summing to 1.
Mixture-attention interface (over frame latents): forward(zv [B, K, D]) -> weights [B, K].
EntityPooling: forward(x [B, K, H], ents [B, K, M, H], valid [B, K, M]) -> [B, K, H].
SlotHead: forward(hc, chars_in, slot_emb, ents, ent_slot, valid) -> logits [B, S].
PixelCopyPath: forward(generated, frames, alpha) -> (image, gate map)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGE_HW = (60, 125)
N_SLOTS = 8            # character slots per story (see storyseq/precompute/annotations.py)


class FixedQueryAttention(nn.Module):
    """The notebook's attention: one learned query, the same importance profile for any input."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: Tensor, h: Tensor) -> Tensor:    # -> alpha [B, K]
        return torch.softmax(self.score(sequence).squeeze(-1), dim=1)


class ContentAttention(nn.Module):
    """Stage A: the query comes from the final GRU state, so the weights depend on the sequence."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, sequence: Tensor, h: Tensor) -> Tensor:    # -> alpha [B, K]
        scores = torch.einsum("bd,bkd->bk", self.q(h), self.k(sequence)) / math.sqrt(sequence.size(-1))
        return torch.softmax(scores, dim=1)


class LatentSimilarityAttention(nn.Module):
    """Stage D: weights over the inputs from the pattern of similarities between their frame latents
    (window-centred cosine), plus position. Independent of the GRU state, so it can be supervised
    toward the closest input without reshaping what the text heads read."""

    def __init__(self, k: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * k, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.register_buffer("eye", torch.eye(k))

    def forward(self, zv: Tensor) -> Tensor:                     # [B, K, D] -> alpha [B, K]
        zc = F.normalize(zv - zv.mean(1, keepdim=True), dim=-1)
        sim = zc @ zc.transpose(1, 2)                               # [B, K, K]
        feats = torch.cat((sim, self.eye.expand(len(zv), -1, -1)), -1)
        return torch.softmax(self.mlp(feats).squeeze(-1), dim=1)


class SlotHead(nn.Module):
    """Stage D: one logit per character slot from that slot's own evidence: its presence in each of
    the K inputs, how often, its slot embedding, its pooled entity token, and the sequence state."""

    def __init__(self, hidden_dim: int, slot_dim: int = 64, k: int = 4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(k + 1 + slot_dim + hidden_dim + 2 * hidden_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, hc: Tensor, chars_in: Tensor, slot_emb: Tensor, ents: Tensor, ent_slot: Tensor, valid: Tensor) -> Tensor:
        b, k, n_slots = chars_in.shape
        hist = chars_in.float().transpose(1, 2)                     # [B, S, K]
        count = hist.sum(-1, keepdim=True) / k
        onehot = F.one_hot(ent_slot.clamp(min=0), n_slots).float() * valid[..., None].float()   # [B, K, M, S]
        pooled = torch.einsum("bkms,bkmh->bsh", onehot, ents) / onehot.sum((1, 2)).clamp(min=1)[..., None]
        x = torch.cat((hist, count, slot_emb[None].expand(b, -1, -1), pooled, hc[:, None].expand(-1, n_slots, -1)), -1)
        return self.mlp(x).squeeze(-1)                              # [B, S]


class EntityPooling(nn.Module):
    """Stage C: each frame token attends over the entity tokens of its frame and adds the result."""

    def __init__(self, hidden_dim: int, ent_dim: int) -> None:
        super().__init__()
        self.k = nn.Linear(ent_dim, hidden_dim)
        self.v = nn.Linear(ent_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, ents: Tensor, valid: Tensor) -> Tensor:
        # x [B, K, H], ents [B, K, M, E], valid [B, K, M]
        scores = torch.einsum("bkh,bkmh->bkm", x, self.k(ents)) / math.sqrt(x.size(-1))
        scores = scores.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(scores, -1).nan_to_num(0.0)        # frames without entities -> zero
        return self.norm(x + torch.einsum("bkm,bkmh->bkh", alpha, self.v(ents)))




class PixelCopyPath(nn.Module):
    """Blend the input frames in pixel space with the attention weights, then let a per-pixel gate
    choose between that blend and the generated image.

    Inputs to the gate: the generated image, the blend, and the attention-weighted variance of the
    inputs around the blend (high where the inputs disagree, so copying is risky)."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(9, width, 3, padding=1), nn.LeakyReLU(0.1),
                                 nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.1),
                                 nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, generated: Tensor, frames: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
        blend = torch.einsum("bk,bkchw->bchw", alpha, frames)
        var = torch.einsum("bk,bkchw->bchw", alpha, (frames - blend[:, None]) ** 2)
        gate = torch.sigmoid(self.net(torch.cat((generated, blend, var), 1)))      # [B, 1, H, W]
        return gate * blend + (1 - gate) * generated, gate


