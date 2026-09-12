"""Data diagnostics for StoryReasoning.

1. Basic stats: stories, frames per story, image sizes, description length in
   BERT tokens (is max_length=120 truncating?).
2. The position-pattern hypothesis: average frames by position in the story
   and compare the between-position differences against split-half noise of
   the same position. Also train a linear probe: can pixels predict position?
3. The L1 floor: what a constant per-pixel median image scores against the
   5th frame, with and without equalize, so the notebook's im=0.215 has a
   reference.
4. Data-loading cost of the current SequencePredictionDataset per item.

Writes figures to diagnostics/out/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import SequencePredictionDataset, parse_gdi_text  # noqa: E402

OUT = ROOT / "diagnostics" / "out"
OUT.mkdir(parents=True, exist_ok=True)
HW = (60, 125)
N_STORIES = int(sys.argv[1]) if len(sys.argv) > 1 else 800
torch.manual_seed(0)
np.random.seed(0)


def main() -> None:
    ds = load_dataset("daniel3303/StoryReasoning", split="train")
    print(f"train stories: {len(ds)}   columns: {ds.column_names}")

    # ---------- 1. basic stats ----------
    tok = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    idx = np.random.choice(len(ds), N_STORIES, replace=False)
    sizes, counts, tok_lens = {}, [], []
    resize = transforms.Compose([transforms.Resize(HW), transforms.ToTensor()])
    raw = np.zeros((5, 3, *HW), dtype=np.float64)   # per-position sums, raw
    eq = np.zeros((5, 3, *HW), dtype=np.float64)    # per-position sums, equalized
    per_pos_imgs = [[] for _ in range(5)]           # small copies for split-half + probe
    fifth = []
    t0 = time.time()
    for k, i in enumerate(idx):
        row = ds[int(i)]
        imgs = row["images"]
        counts.append(len(imgs))
        for im in imgs[:5]:
            sizes[im.size] = sizes.get(im.size, 0) + 1
        for d in parse_gdi_text(row["story"])[:5]:
            tok_lens.append(len(tok(d["description"]).input_ids))
        for p in range(5):
            x = resize(imgs[p]).numpy()
            raw[p] += x
            eq[p] += resize(TF.equalize(imgs[p])).numpy()
            per_pos_imgs[p].append(x[:, ::4, ::5].mean(0).ravel())  # 15x25 gray
            if p == 4:
                fifth.append(x)
        if k % 200 == 0:
            print(f"  scanned {k}/{N_STORIES} ({time.time() - t0:.0f}s)")
    counts = np.array(counts)
    tok_lens = np.array(tok_lens)
    print(f"\nframes/story: min {counts.min()} median {np.median(counts):.0f} max {counts.max()}")
    print("most common source sizes (W,H):", sorted(sizes.items(), key=lambda x: -x[1])[:6])
    print(f"description length in BERT tokens: median {np.median(tok_lens):.0f}, "
          f"p90 {np.percentile(tok_lens, 90):.0f}, max {tok_lens.max()}, "
          f"share > 120 (truncated): {(tok_lens > 120).mean():.0%}")

    # ---------- 2. position pattern ----------
    raw_mean = raw / N_STORIES
    eq_mean = eq / N_STORIES
    grand = raw_mean.mean(0, keepdims=True)
    between = np.abs(raw_mean - grand).mean(axis=(1, 2, 3))            # per position, vs grand mean
    # split-half noise: same position, two random halves
    half = N_STORIES // 2
    noise = []
    for p in range(5):
        a = np.stack(per_pos_imgs[p])
        perm = np.random.permutation(len(a))
        noise.append(np.abs(a[perm[:half]].mean(0) - a[perm[half:]].mean(0)).mean())
    print("\nposition pattern (mean abs deviation of per-position mean image, 0..1 scale):")
    print("  position  vs-grand-mean   split-half-noise(gray, same position)")
    for p in range(5):
        print(f"     {p}        {between[p]:.4f}            {noise[p]:.4f}")
    print(f"  std of pixel values inside the grand-mean image: {grand.std():.4f}  "
          f"(range {grand.min():.3f}..{grand.max():.3f})")

    fig, ax = plt.subplots(3, 6, figsize=(18, 6))
    for p in range(5):
        ax[0, p].imshow(raw_mean[p].transpose(1, 2, 0)); ax[0, p].set_title(f"raw mean, pos {p}")
        ax[1, p].imshow(eq_mean[p].transpose(1, 2, 0)); ax[1, p].set_title(f"equalized mean, pos {p}")
        d = raw_mean[p] - grand[0]
        ax[2, p].imshow(np.clip(d.transpose(1, 2, 0) * 10 + 0.5, 0, 1)); ax[2, p].set_title("(pos - grand) x10")
    ax[0, 5].imshow(grand[0].transpose(1, 2, 0)); ax[0, 5].set_title("grand mean")
    ax[1, 5].imshow(np.clip(np.median(np.stack(fifth), 0).transpose(1, 2, 0), 0, 1)); ax[1, 5].set_title("median of 5th frames")
    a = np.stack(per_pos_imgs[0]); perm = np.random.permutation(len(a))
    ax[2, 5].imshow(np.clip((a[perm[:half]].mean(0) - a[perm[half:]].mean(0)).reshape(15, 25) * 10 + 0.5, 0, 1), cmap="gray")
    ax[2, 5].set_title("split-half noise x10 (pos 0)")
    for a_ in ax.ravel():
        a_.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "position_means.png", dpi=110); plt.close()
    print(f"  figure: {OUT / 'position_means.png'}")

    # linear probe: pixels -> position (chance = 20%)
    X = torch.tensor(np.concatenate([np.stack(v) for v in per_pos_imgs]), dtype=torch.float32)
    y = torch.tensor(np.concatenate([[p] * len(v) for p, v in enumerate(per_pos_imgs)]))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    perm = torch.randperm(len(X)); tr, te = perm[: int(0.8 * len(X))], perm[int(0.8 * len(X)):]
    clf = torch.nn.Linear(X.shape[1], 5); opt = torch.optim.Adam(clf.parameters(), 1e-2, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad(); torch.nn.functional.cross_entropy(clf(X[tr]), y[tr]).backward(); opt.step()
    acc_tr = (clf(X[tr]).argmax(1) == y[tr]).float().mean().item()
    acc_te = (clf(X[te]).argmax(1) == y[te]).float().mean().item()
    print(f"  linear probe pixels->position: train {acc_tr:.1%}, held-out {acc_te:.1%} (chance 20%)")

    # ---------- 3. L1 floor ----------
    fifth = np.stack(fifth)
    med = np.median(fifth, 0, keepdims=True)
    print(f"\nL1 floor: constant median image vs 5th frame (raw)      = {np.abs(fifth - med).mean():.4f}")
    print(f"          constant mid-gray 0.5 vs 5th frame (raw)       = {np.abs(fifth - 0.5).mean():.4f}")
    fifth_eq = np.stack([resize(TF.equalize(ds[int(i)]["images"][4])).numpy() for i in idx[:200]])
    med_eq = np.median(fifth_eq, 0, keepdims=True)
    print(f"          constant median image vs 5th frame (equalized)= {np.abs(fifth_eq - med_eq).mean():.4f}")
    print(f"          copy 4th frame as prediction of 5th (raw)     = "
          f"{np.mean([np.abs(resize(ds[int(i)]['images'][3]).numpy() - fifth[k]).mean() for k, i in enumerate(idx[:200])]):.4f}")
    print("  (notebook reached im=0.215 after 5 epochs, on equalized frames)")

    # ---------- 4. loading cost ----------
    sp = SequencePredictionDataset(ds, tok)
    t0 = time.time()
    for i in idx[:20]:
        sp[int(i)]
    per_item = (time.time() - t0) / 20
    print(f"\nSequencePredictionDataset.__getitem__: {per_item * 1000:.0f} ms/item -> "
          f"{per_item * 0.8 * len(ds) / 60:.1f} min per epoch of data loading alone (single worker)")


if __name__ == "__main__":
    main()
