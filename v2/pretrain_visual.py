"""Pretrain the v2 visual autoencoder with a reconstruction loss on every training frame.

This is the notebook's "To-Do" cell. It gives the sequence predictor an image encoder whose
latent already carries the frame, and a decoder that can draw more than the median image.

Reports validation L1 (test-split frames) next to the constant-median floor, and writes
  v2/out/visual_ae.pt          weights
  v2/out/reconstructions.png   originals vs reconstructions
Run: python v2/pretrain_visual.py [--epochs 12]
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.data import all_frames, load_split  # noqa: E402
from v2.models import VisualAutoencoder  # noqa: E402

OUT = ROOT / "v2" / "out"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(0)

    tr, te = load_split("train", DEVICE), load_split("test", DEVICE)
    s_tr, f_tr = all_frames(tr)
    s_te, f_te = all_frames(te)
    val = te["pix"][s_te, f_te].float() / 255
    median = val.median(0).values
    floor = F.l1_loss(median.expand_as(val), val).item()
    print(f"train frames {len(s_tr)}  val frames {len(s_te)}  constant-median floor L1 = {floor:.4f}  device {DEVICE}")

    ae = VisualAutoencoder(args.latent_dim).to(DEVICE)
    print(f"autoencoder params: {sum(p.numel() for p in ae.parameters()):,}")
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * (len(s_tr) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)

    t0 = time.time()
    for epoch in range(args.epochs):
        ae.train()
        perm = torch.randperm(len(s_tr), device=DEVICE)
        tot = 0.0
        for b in range(len(s_tr) // args.batch_size):
            j = perm[b * args.batch_size:(b + 1) * args.batch_size]
            x = tr["pix"][s_tr[j], f_tr[j]].float() / 255
            loss = F.l1_loss(ae(x), x)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        ae.eval()
        with torch.no_grad():
            rec = torch.cat([ae(val[i:i + 256]) for i in range(0, len(val), 256)])
            val_l1 = F.l1_loss(rec, val).item()
            z = torch.cat([ae.encoder(val[i:i + 256]) for i in range(0, len(val), 256)])
            dead = (z.abs().max(0).values < 1e-6).float().mean().item()
        print(f"epoch {epoch + 1:2d}/{args.epochs}  train L1 {tot / (b + 1):.4f}  val L1 {val_l1:.4f} "
              f"(median floor {floor:.4f})  latent spread {z.std(0).mean():.3f}  dead {dead:.0%}  {time.time() - t0:.0f}s")

    torch.save(ae.state_dict(), OUT / "visual_ae.pt")
    fig, ax = plt.subplots(2, 8, figsize=(17, 3.2))
    pick = torch.randperm(len(val))[:8]
    for c, i in enumerate(pick.tolist()):
        ax[0, c].imshow(val[i].permute(1, 2, 0).cpu()); ax[0, c].axis("off")
        ax[1, c].imshow(rec[i].permute(1, 2, 0).cpu()); ax[1, c].axis("off")
    ax[0, 0].set_title("original", fontsize=9); ax[1, 0].set_title("reconstruction", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "reconstructions.png", dpi=110)
    print(f"saved {OUT / 'visual_ae.pt'} and {OUT / 'reconstructions.png'}")


if __name__ == "__main__":
    main()
