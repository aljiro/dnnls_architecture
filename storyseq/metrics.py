"""Semantic and realism metrics for image predictions, with the usual floors.

Pixel L1 cannot distinguish "right layout, blurred" from "right tint, no structure". These can:
  clip_sim   cosine between CLIP ViT-B/32 embeddings of the decoded image and of the target
             (semantic content: same scene / people / lighting), centred by the mean test embedding
  clip_fd    Frechet distance between the CLIP-feature distributions of a set of images and of the
             targets (realism of the *set*, independent of alignment; lower is better)
  sharpness  variance of the Laplacian of the grayscale image, as a ratio to the target's

Reported for: the model's prior-mean prediction, prior samples, the blob (median image),
copy-last, the best-of-4 input, and the autoencoder reconstruction of the target.

Used by the trainer (--semantic) and by each level's visualize.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel

from storyseq.data import gather, load_split, model_kwargs, tokenizer, windows

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ClipEmbed(torch.nn.Module):
    """Differentiable CLIP image embedding of a [B, 3, H, W] tensor in [0, 1] (resize + normalise in-graph)."""

    def __init__(self) -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        for p in self.clip.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        return super().train(False)

    @staticmethod
    def augment(x: torch.Tensor) -> torch.Tensor:
        """Random resized crop (70-100 % of the frame), horizontal flip, slight brightness / contrast
        jitter; batched and differentiable (affine grid), so a loss through CLIP cannot be satisfied
        by a fixed pattern at a fixed position (the CLIP-guided-generation remedy for adversarial motifs)."""
        b = x.size(0)
        dev = x.device
        scale = torch.empty(b, device=dev).uniform_(0.7, 1.0)
        tx = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale)
        ty = (torch.rand(b, device=dev) * 2 - 1) * (1 - scale)
        flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
        theta = torch.zeros(b, 2, 3, device=dev)
        theta[:, 0, 0] = scale * flip
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        contrast = torch.empty(b, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
        brightness = torch.empty(b, 1, 1, 1, device=dev).uniform_(-0.05, 0.05)
        return ((x - 0.5) * contrast + 0.5 + brightness).clamp(0, 1)

    def forward(self, x: torch.Tensor, augment: bool = False, n_views: int = 2) -> torch.Tensor:
        """[B, 3, H, W] in [0, 1] -> unit embeddings [B, 512]; augment=True averages n_views random views."""
        if augment:
            return F.normalize(sum(self.forward(self.augment(x)) for _ in range(n_views)) / n_views, dim=-1)
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        f = self.clip.get_image_features(pixel_values=x)
        f = f if torch.is_tensor(f) else f.pooler_output
        return F.normalize(f, dim=-1)


def frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frechet distance between Gaussians fitted to feature sets a and b ([N, D], float64)."""
    a, b = a.double(), b.double()
    mu_a, mu_b = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    # sqrt(ca cb) via eigen-decomposition of the symmetric product sqrt(ca) cb sqrt(ca)
    ea, va = torch.linalg.eigh(ca)
    sa = (va * ea.clamp(min=0).sqrt()) @ va.T
    m = sa @ cb @ sa
    em = torch.linalg.eigvalsh((m + m.T) / 2).clamp(min=0).sqrt().sum()
    return float((mu_a - mu_b).pow(2).sum() + ca.trace() + cb.trace() - 2 * em)


def sharpness(x: torch.Tensor) -> torch.Tensor:
    g = x.mean(1, keepdim=True)
    k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=x.device).view(1, 1, 3, 3)
    return F.conv2d(g, k).flatten(1).var(1)


@torch.no_grad()
def semantic_table(model, d: dict, s, t, n_samples: int = 2, max_windows: int = 0) -> dict[str, dict[str, float]]:
    """CLIP similarity (centred and raw), CLIP Frechet distance and sharpness ratio for the model's
    prediction, its samples (variational models), and the floors. Returns {candidate: {metric: value}}."""
    model.eval()
    if max_windows:
        s, t = s[:max_windows], t[:max_windows]
    emb = ClipEmbed().to(DEVICE)
    median = (d["pix"][s.cpu(), t.cpu()].float() / 255).median(0).values.to(DEVICE)
    feats: dict[str, list] = {k: [] for k in ("target", "prediction", "sample", "blob", "copy_last", "best_input", "reconstruction")}
    sharp: dict[str, list] = {k: [] for k in feats}
    for b in range(0, len(s), 64):
        batch = gather(d, s[b:b + 64], t[b:b + 64])
        kw = model_kwargs(model, batch)
        o = model(batch["frames"], batch["txt"], batch["target_ids"][:, :-1], **kw)
        per_in = (batch["frames"] - batch["target"][:, None]).abs().mean(dim=(2, 3, 4))
        best = batch["frames"][torch.arange(len(per_in), device=DEVICE), per_in.argmin(1)]
        imgs = {"target": batch["target"], "prediction": o["image"], "blob": median.expand_as(batch["target"]),
                "copy_last": batch["frames"][:, -1], "best_input": best,
                "reconstruction": model.image_decoder(model.image_encoder(batch["target"]))}
        if model.variational and n_samples > 0:
            imgs["sample"] = torch.cat([model(batch["frames"], batch["txt"], batch["target_ids"][:, :-1], sample=True, **kw)["image"]
                                        for _ in range(n_samples)])
        for k, v in imgs.items():
            feats[k].append(emb(v).cpu()); sharp[k].append(sharpness(v).cpu())
    fe = {k: torch.cat(v) for k, v in feats.items() if v}
    sh = {k: torch.cat(v).mean().item() for k, v in sharp.items() if v}
    mu = fe["target"].mean(0, keepdim=True)
    tgt_c = F.normalize(fe["target"] - mu, dim=-1)
    table = {}
    for k in ("prediction", "sample", "blob", "copy_last", "best_input", "reconstruction", "target"):
        if k not in fe:
            continue
        f = fe[k]
        rep = len(f) // len(fe["target"])
        table[k] = {"clip_sim": (F.normalize(f - mu, dim=-1) * tgt_c.repeat(rep, 1)).sum(-1).mean().item(),
                    "clip_sim_raw": (f * fe["target"].repeat(rep, 1)).sum(-1).mean().item(),
                    "clip_frechet": frechet(f, fe["target"]), "sharpness": sh[k] / sh["target"]}
    return table


def print_table(table: dict[str, dict[str, float]]) -> None:
    print(f"{'candidate':16s}{'CLIP sim (centred)':>20s}{'CLIP sim (raw)':>16s}{'CLIP Frechet':>14s}{'sharpness/target':>18s}")
    for k, v in table.items():
        print(f"{k:16s}{v['clip_sim']:20.3f}{v['clip_sim_raw']:16.3f}{v['clip_frechet']:14.3f}{v['sharpness']:18.2f}")
