"""Semantic and realism metrics for image predictions, with the usual floors.

Pixel L1 cannot distinguish "right layout, blurred" from "right tint, no structure". These can:
  clip_sim   cosine between CLIP ViT-B/32 embeddings of the decoded image and of the target
             (semantic content: same scene / people / lighting), centred by the mean test embedding
  clip_fd    Frechet distance between the CLIP-feature distributions of a set of images and of the
             targets (realism of the *set*, independent of alignment; lower is better)
  sharpness  variance of the Laplacian of the grayscale image, as a ratio to the target's

Reported for: the model's prior-mean prediction, prior samples, the blob (median image),
copy-last, the best-of-4 input, and the autoencoder reconstruction of the target.

Run: python v2/semantic_metrics.py --tag _scale --ae-width 2 --clip-input --entity-features clip
     python v2/semantic_metrics.py --tag _kl1e-2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import CLIPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.data import gather, load_split, model_kwargs, tokenizer, windows  # noqa: E402
from v2.models import SequencePredictor, VisualAutoencoder  # noqa: E402

OUT = ROOT / "v2" / "out"
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="D")
    ap.add_argument("--text-encoder", default="minilm")
    ap.add_argument("--tag", default="")
    ap.add_argument("--ae-width", type=int, default=1)
    ap.add_argument("--clip-input", action="store_true")
    ap.add_argument("--entity-features", choices=["ae", "clip"], default="ae")
    ap.add_argument("--no-copy-path", action="store_true")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--max-windows", type=int, default=0)
    args = ap.parse_args()
    tok = tokenizer()
    te = load_split("test", DEVICE, annotations=args.stage in ("C", "D"))
    s, t = windows(te)
    if args.max_windows:
        s, t = s[:args.max_windows], t[:args.max_windows]
    model = SequencePredictor(VisualAutoencoder(width=args.ae_width), 384, tok.vocab_size, stage=args.stage,
                              copy_path=not args.no_copy_path, clip_input=args.clip_input,
                              entity_features=args.entity_features).to(DEVICE)
    model.load_state_dict(torch.load(OUT / f"predictor_stage{args.stage}_{args.text_encoder}{args.tag}.pt", map_location=DEVICE))
    model.eval()
    emb = ClipEmbed().to(DEVICE)
    median = (te["pix"][s.cpu(), t.cpu()].float() / 255).median(0).values.to(DEVICE)

    feats: dict[str, list] = {k: [] for k in ("target", "prediction", "sample", "blob", "copy_last", "best_input", "reconstruction")}
    sharp: dict[str, list] = {k: [] for k in feats}
    for b in range(0, len(s), 64):
        batch = gather(te, s[b:b + 64], t[b:b + 64])
        kw = model_kwargs(model, batch)
        o = model(batch["frames"], batch["txt"], batch["target_ids"][:, :-1], **kw)
        per_in = (batch["frames"] - batch["target"][:, None]).abs().mean(dim=(2, 3, 4))
        best = batch["frames"][torch.arange(len(per_in), device=DEVICE), per_in.argmin(1)]
        imgs = {"target": batch["target"], "prediction": o["image"], "blob": median.expand_as(batch["target"]),
                "copy_last": batch["frames"][:, -1], "best_input": best,
                "reconstruction": model.image_decoder(model.image_encoder(batch["target"]))}
        if args.stage == "D":
            imgs["sample"] = torch.cat([model(batch["frames"], batch["txt"], batch["target_ids"][:, :-1], sample=True, **kw)["image"]
                                        for _ in range(args.n_samples)])
        for k, v in imgs.items():
            feats[k].append(emb(v).cpu()); sharp[k].append(sharpness(v).cpu())
    fe = {k: torch.cat(v) for k, v in feats.items() if v}
    sh = {k: torch.cat(v).mean().item() for k, v in sharp.items() if v}
    mu = fe["target"].mean(0, keepdim=True)
    tgt_c = F.normalize(fe["target"] - mu, dim=-1)
    print(f"{len(s)} test windows, checkpoint predictor_stage{args.stage}_{args.text_encoder}{args.tag}.pt\n")
    print(f"{'candidate':16s}{'CLIP sim (centred)':>20s}{'CLIP sim (raw)':>16s}{'CLIP Frechet':>14s}{'sharpness/target':>18s}")
    for k in ("prediction", "sample", "blob", "copy_last", "best_input", "reconstruction", "target"):
        if k not in fe:
            continue
        f = fe[k]
        rep = len(f) // len(fe["target"])
        tgt_rep = fe["target"].repeat(rep, 1); tgt_c_rep = tgt_c.repeat(rep, 1)
        sim_c = (F.normalize(f - mu, dim=-1) * tgt_c_rep).sum(-1).mean().item()
        sim_r = (f * tgt_rep).sum(-1).mean().item()
        fd = frechet(f, fe["target"])
        print(f"{k:16s}{sim_c:20.3f}{sim_r:16.3f}{fd:14.3f}{sh[k] / sh['target']:18.2f}")


if __name__ == "__main__":
    main()
