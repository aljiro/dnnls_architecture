"""Proof of concept: next-frame prediction in embedding space on cached features.

Reformulation of the notebook task:
  inputs : K frames as (CLIP image embedding, MiniLM text embedding)
  target : embedding of frame K (image and text)
  loss   : InfoNCE, i.e. "pick the true next frame among candidates"
           candidates = other targets in the batch (+ the story's own other frames as hard negatives)
  metric : retrieval recall on the held-out test split, against copy-last and chance baselines.

Optional: a pixel decoder (embedding -> 60x125 image, L1) to visualise what an embedding
prediction looks like as pixels, and why L1-in-pixel-space alone gives a blob.

Run:  python poc/train.py            (after python poc/precompute.py)
      python poc/train.py --modality text   # ablation: text inputs only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import VisualDecoder  # noqa: E402

CACHE = ROOT / "poc" / "cache"
OUT = ROOT / "poc" / "out"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 4


def windows(d: dict, k: int = K) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (story_index, target_frame) pairs for every valid window."""
    idx = []
    for i, n in enumerate(d["n_frames"].tolist()):
        for t in range(k, n):
            idx.append((i, t))
    idx = torch.tensor(idx)
    return idx[:, 0], idx[:, 1]


def gather_window(d: dict, s: torch.Tensor, t: torch.Tensor, k: int = K):
    ar = torch.arange(k)
    pos = t[:, None] - k + ar[None]                            # [B, K]
    img = d["img"][s[:, None], pos]                            # [B, K, 512]
    txt = d["txt"][s[:, None], pos]                            # [B, K, 384]
    return img, txt, d["img"][s, t], d["txt"][s, t]


class NextFramePredictor(nn.Module):
    def __init__(self, modality: str = "both", d: int = 256, layers: int = 2, k: int = K):
        super().__init__()
        self.modality = modality
        in_dim = {"both": 512 + 384, "image": 512, "text": 384}[modality]
        self.inp = nn.Linear(in_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, k, d))
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.img_head = nn.Linear(d, 512)
        self.txt_head = nn.Linear(d, 384)
        self.log_tau = nn.Parameter(torch.tensor(-2.0))       # tau ~ 0.135, learnable

    def forward(self, img: torch.Tensor, txt: torch.Tensor):
        x = {"both": torch.cat((img, txt), -1), "image": img, "text": txt}[self.modality]
        h = self.norm(self.encoder(self.inp(x) + self.pos))[:, -1]
        return F.normalize(self.img_head(h), dim=-1), F.normalize(self.txt_head(h), dim=-1)


def info_nce(pred: torch.Tensor, target: torch.Tensor, tau: torch.Tensor,
             hard: torch.Tensor | None = None, hard_mask: torch.Tensor | None = None) -> torch.Tensor:
    logits = pred @ target.T / tau                              # [B, B]
    if hard is not None:                                        # [B, M] story's own frames
        extra = torch.einsum("bd,bmd->bm", pred, hard) / tau
        extra = extra.masked_fill(~hard_mask, float("-inf"))
        logits = torch.cat((logits, extra), 1)
    return F.cross_entropy(logits, torch.arange(len(pred), device=pred.device))


@torch.no_grad()
def evaluate(model: NextFramePredictor | None, d: dict, tag: str, baseline: str | None = None) -> dict[str, float]:
    s, t = windows(d)
    img, txt, tgt_img, tgt_txt = gather_window(d, s, t)
    if baseline == "copy-last":
        p_img, p_txt = img[:, -1], txt[:, -1]
    else:
        model.eval()
        p_img, p_txt = model(img.to(DEVICE), txt.to(DEVICE))
        p_img, p_txt = p_img.cpu(), p_txt.cpu()
    res = {}
    for name, p, tgt in (("img", p_img, tgt_img), ("txt", p_txt, tgt_txt)):
        # global pool: every test target
        rank = (p @ tgt.T).argsort(1, descending=True)
        hit = rank == torch.arange(len(p))[:, None]
        res[f"{name}_R@1"] = hit[:, 0].float().mean().item()
        res[f"{name}_R@10"] = hit[:, :10].any(1).float().mean().item()
        res[f"{name}_medrank"] = hit.float().argmax(1).median().item() + 1
        # within-story pool: all frames of the same story except the K inputs
        pool = d[name][s]                                        # [B, M, D]
        m = torch.arange(pool.shape[1])[None] < d["n_frames"][s][:, None]
        ar = torch.arange(pool.shape[1])[None]
        m &= ~((ar >= (t - K)[:, None]) & (ar < t[:, None]))     # drop the input positions
        sim = torch.einsum("bd,bmd->bm", p, pool).masked_fill(~m, float("-inf"))
        res[f"{name}_story_R@1"] = (sim.argmax(1) == t).float().mean().item()
        res[f"{name}_story_chance"] = (1.0 / m.sum(1).float()).mean().item()
    res["n_windows"] = len(s)
    print(f"[{tag}] " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    return res


def train_predictor(tr: dict, te: dict, args) -> NextFramePredictor:
    model = NextFramePredictor(args.modality).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    s_all, t_all = windows(tr)
    steps = args.epochs * (len(s_all) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.1)
    print(f"predictor params: {sum(p.numel() for p in model.parameters()):,}   train windows: {len(s_all)}")
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(s_all))
        tot = 0.0
        for b in range(len(s_all) // args.batch_size):
            j = perm[b * args.batch_size:(b + 1) * args.batch_size]
            s, t = s_all[j], t_all[j]
            img, txt, tgt_img, tgt_txt = (x.to(DEVICE) for x in gather_window(tr, s, t))
            p_img, p_txt = model(img, txt)
            tau = model.log_tau.exp()
            if args.hard_neg:
                hm = torch.arange(tr["img"].shape[1])[None] < tr["n_frames"][s][:, None]
                hm &= torch.arange(tr["img"].shape[1])[None] != t[:, None]      # exclude the true target
                hm = hm.to(DEVICE)
                loss = (info_nce(p_img, tgt_img, tau, tr["img"][s].to(DEVICE), hm)
                        + info_nce(p_txt, tgt_txt, tau, tr["txt"][s].to(DEVICE), hm))
            else:
                loss = info_nce(p_img, tgt_img, tau) + info_nce(p_txt, tgt_txt, tau)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d}  loss {tot / (b + 1):.3f}  tau {tau.item():.3f}  {time.time() - t0:.0f}s")
            evaluate(model, te, f"test e{epoch + 1}")
    return model


def train_pixel_decoder(tr: dict, epochs: int = 8) -> VisualDecoder:
    """Embedding -> pixels with L1. Shows what any L1 pixel head can do given a good embedding."""
    dec = VisualDecoder(latent_dim=512).to(DEVICE)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
    n = tr["n_frames"]
    fi = torch.nonzero(torch.arange(tr["img"].shape[1])[None] < n[:, None])     # [F, 2] valid frames
    print(f"pixel decoder: {len(fi)} training frames")
    for ep in range(epochs):
        perm = fi[torch.randperm(len(fi))]
        tot = 0.0
        for b in range(0, len(perm), 256):
            j = perm[b:b + 256]
            z = tr["img"][j[:, 0], j[:, 1]].to(DEVICE)
            x = tr["pix"][j[:, 0], j[:, 1]].to(DEVICE).float() / 255
            loss = F.l1_loss(dec(z)[0], x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
        print(f"  decoder epoch {ep + 1}: L1 {tot / len(fi):.4f}")
    return dec


@torch.no_grad()
def visualise(model: NextFramePredictor, dec: VisualDecoder, tr: dict, te: dict, n_rows: int = 5) -> None:
    model.eval(); dec.eval()
    s, t = windows(te)
    pick = torch.randperm(len(s))[:n_rows]
    s, t = s[pick], t[pick]
    img, txt, tgt_img, _ = gather_window(te, s, t)
    p_img, _ = model(img.to(DEVICE), txt.to(DEVICE))
    # nearest training frame to the predicted embedding
    n = tr["n_frames"]
    valid = torch.nonzero(torch.arange(tr["img"].shape[1])[None] < n[:, None])
    bank = tr["img"][valid[:, 0], valid[:, 1]].to(DEVICE)
    nn_idx = (p_img @ bank.T).argmax(1).cpu()
    dec_true = dec(tgt_img.to(DEVICE))[0].cpu()
    dec_pred = dec(p_img)[0].cpu()
    cols = ["in 1", "in 2", "in 3", "in 4", "target", "decode(true emb)", "decode(pred emb)", "retrieved (pred emb)"]
    fig, ax = plt.subplots(n_rows, len(cols), figsize=(2.2 * len(cols), 1.3 * n_rows))
    for r in range(n_rows):
        tiles = [te["pix"][s[r], t[r] - K + i] for i in range(K)] + [te["pix"][s[r], t[r]]]
        tiles = [x.permute(1, 2, 0).numpy() for x in tiles]
        tiles += [dec_true[r].permute(1, 2, 0).numpy(), dec_pred[r].permute(1, 2, 0).numpy(),
                  tr["pix"][valid[nn_idx[r], 0], valid[nn_idx[r], 1]].permute(1, 2, 0).numpy()]
        for c, tile in enumerate(tiles):
            ax[r, c].imshow(tile); ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(cols[c], fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / f"predictions_{model.modality}.png", dpi=110); plt.close()
    print(f"figure: {OUT / f'predictions_{model.modality}.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["both", "image", "text"], default="both")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hard-neg", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pixel-decoder", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    tr = torch.load(CACHE / "train.pt")
    te = torch.load(CACHE / "test.pt")
    print(f"train stories {len(tr['n_frames'])}  test stories {len(te['n_frames'])}  device {DEVICE}")

    evaluate(None, te, "baseline copy-last", baseline="copy-last")
    model = train_predictor(tr, te, args)
    torch.save(model.state_dict(), OUT / f"predictor_{args.modality}.pt")
    if args.pixel_decoder:
        dec = train_pixel_decoder(tr)
        visualise(model, dec, tr, te)


if __name__ == "__main__":
    main()
