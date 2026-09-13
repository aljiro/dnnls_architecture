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
CACHE = ROOT / "poc" / "cache"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--width", type=int, default=1, help="channel multiplier")
    ap.add_argument("--extra", choices=["", "groundcap"], default="", help="add GroundCap frames to the training set")
    ap.add_argument("--out", default="", help="weights file (default v2/out/visual_ae.pt, or visual_ae_w<width>.pt)")
    args = ap.parse_args()
    torch.manual_seed(0)

    tr, te = load_split("train", DEVICE), load_split("test", DEVICE)
    s_tr, f_tr = all_frames(tr)
    s_te, f_te = all_frames(te)
    frames = tr["pix"][s_tr.cpu(), f_tr.cpu()]                              # [F, 3, H, W] uint8, CPU
    if args.extra == "groundcap":
        frames = torch.cat((frames, torch.load(CACHE / "groundcap.pt")["pix"]))
    n_train = len(frames)
    val = te["pix"][s_te.cpu(), f_te.cpu()].to(DEVICE).float() / 255
    median = val.median(0).values
    floor = F.l1_loss(median.expand_as(val), val).item()
    print(f"train frames {n_train}  val frames {len(s_te)}  constant-median floor L1 = {floor:.4f}  device {DEVICE}  width {args.width}")

    ae = VisualAutoencoder(args.latent_dim, args.width).to(DEVICE)
    print(f"autoencoder params: {sum(p.numel() for p in ae.parameters()):,}")
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * (n_train // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)

    t0 = time.time()
    for epoch in range(args.epochs):
        ae.train()
        perm = torch.randperm(n_train)
        tot = 0.0
        for b in range(n_train // args.batch_size):
            j = perm[b * args.batch_size:(b + 1) * args.batch_size]
            x = frames[j].to(DEVICE, non_blocking=True).float() / 255
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

    out_path = Path(args.out) if args.out else OUT / ("visual_ae.pt" if args.width == 1 else f"visual_ae_w{args.width}.pt")
    torch.save(ae.state_dict(), out_path)
    fig, ax = plt.subplots(2, 8, figsize=(17, 3.2))
    pick = torch.randperm(len(val))[:8]
    for c, i in enumerate(pick.tolist()):
        ax[0, c].imshow(val[i].permute(1, 2, 0).cpu()); ax[0, c].axis("off")
        ax[1, c].imshow(rec[i].permute(1, 2, 0).cpu()); ax[1, c].axis("off")
    ax[0, 0].set_title("original", fontsize=9); ax[1, 0].set_title("reconstruction", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "reconstructions.png", dpi=110)
    print(f"saved {out_path} and {OUT / 'reconstructions.png'}")


if __name__ == "__main__":
    main()
