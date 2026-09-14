"""v0: the floors every later number is judged against, and the cut / continue split.

For every test window (4 input frames -> frame 5), pixel L1 of three trivial predictors:
  median   the constant per-pixel median image of all test targets
  copy     the last input frame
  best     the input frame closest to the target (an oracle: it peeks at the target)
reported for all windows and split into "shot continues" (best < 0.06) and "shot cuts".
Also: which input is the closest one (the A-B-A-B editing rhythm), and a figure of the median
image next to a few targets.

From this directory:  python floors.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storyseq.data import gather, load_split, windows  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def main() -> None:
    te = load_split("test", "cpu")
    s, t = windows(te)
    target = te["pix"][s, t].float() / 255
    median = target.median(0).values
    per = lambda a, b: (a - b).abs().mean(dim=(1, 2, 3))
    l1_median = per(median.expand_as(target), target)
    copy_l1, best_l1, best_idx = [], [], []
    for b in range(0, len(s), 256):
        batch = gather(te, s[b:b + 256], t[b:b + 256])
        per_in = (batch["frames"] - batch["target"][:, None]).abs().mean(dim=(2, 3, 4))
        copy_l1.append(per_in[:, -1]); best_l1.append(per_in.min(1).values); best_idx.append(per_in.argmin(1))
    copy_l1, best_l1, best_idx = map(torch.cat, (copy_l1, best_l1, best_idx))
    near = best_l1 < 0.06
    print(f"{len(s)} test windows; shot continues (some input within L1 0.06) in {near.float().mean():.1%}\n")
    print(f"{'predictor':22s}{'all':>8s}{'continues':>12s}{'cuts':>8s}")
    for name, v in (("median image (blob)", l1_median), ("copy last input", copy_l1), ("best input (oracle)", best_l1)):
        print(f"{name:22s}{v.mean():8.3f}{v[near].mean():12.3f}{v[~near].mean():8.3f}")
    print("\nwhich input is closest to the target:", "  ".join(f"input {i + 1}: {(best_idx == i).float().mean():.0%}" for i in range(4)))
    print("(frame 3 above frame 4 is the A-B-A-B rhythm of dialogue editing)")
    fig, ax = plt.subplots(1, 5, figsize=(17, 2.6))
    ax[0].imshow(median.permute(1, 2, 0)); ax[0].set_title("median image: what an input-blind model converges to", fontsize=8)
    for i, j in enumerate(torch.randperm(len(s), generator=torch.Generator().manual_seed(0))[:4].tolist()):
        ax[i + 1].imshow(target[j].permute(1, 2, 0)); ax[i + 1].set_title(f"a target (L1 to median {l1_median[j]:.2f})", fontsize=8)
    for a in ax:
        a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "floors.png", dpi=110)
    print(f"\nfigure: {OUT / 'floors.png'}")


if __name__ == "__main__":
    main()
