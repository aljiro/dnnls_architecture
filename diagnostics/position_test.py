"""Is there a real per-position pattern in the frames, or is (pos_mean - grand_mean) sampling noise?

Uses all cached stories (poc/cache/train.pt), positions 0..4, 60x125 RGB.

Test 1 - split-half reproducibility. Split stories at random into halves A and B. For each
  position p compute the deviation map d_p^A = mean_A(pos p) - grand_A, likewise d_p^B.
  If the pattern is real, corr(d_p^A, d_p^B) (same position, different stories) is high and
  corr(d_p^A, d_q^B) for p != q is lower. If it is noise, the whole 5x5 matrix is ~0.

Test 2 - permutation test. Statistic = mean over positions of mean|d_p| (full data).
  Shuffle frame order within each story (destroys any position signal, keeps everything else),
  recompute the statistic 300 times. p-value = share of shuffles >= observed.

Figure diagnostics/out/position_test.png: row 1 = (d_p) x10 for all data,
  row 2 = half A, row 3 = half B. Real signal would make rows 2 and 3 match column by column.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out"
P = 5
rng = np.random.default_rng(0)

cache = torch.load(ROOT / "poc" / "cache" / "train.pt")
keep = cache["n_frames"] >= P
X = cache["pix"][keep][:, :P].float().div_(255).numpy()          # [N, P, 3, 60, 125]
N = len(X)
print(f"stories: {N}, frames per position: {N}")


def deviations(x: np.ndarray) -> np.ndarray:
    pos_mean = x.mean(0)                                          # [P, 3, H, W]
    return pos_mean - pos_mean.mean(0, keepdims=True)             # [P, 3, H, W]


def stat(x: np.ndarray) -> float:
    return float(np.abs(deviations(x)).mean())


# ---- Test 1: split-half correlation matrix ----
perm = rng.permutation(N)
A, B = X[perm[: N // 2]], X[perm[N // 2:]]
dA, dB = deviations(A).reshape(P, -1), deviations(B).reshape(P, -1)
corr = np.array([[np.corrcoef(dA[p], dB[q])[0, 1] for q in range(P)] for p in range(P)])
print("\nTest 1: correlation of deviation maps, half A (rows) vs half B (cols)")
print("        same position on the diagonal; a real pattern shows up as a positive diagonal")
np.set_printoptions(precision=3, suppress=True)
print(corr)
print(f"  mean diagonal {np.mean(np.diag(corr)):+.3f}   mean off-diagonal {np.mean(corr[~np.eye(P, dtype=bool)]):+.3f}")

# ---- Test 2: permutation test ----
obs = stat(X)
null = []
for _ in range(300):
    order = np.argsort(rng.random((N, P)), axis=1)               # independent shuffle per story
    null.append(stat(np.take_along_axis(X, order[:, :, None, None, None], axis=1)))
null = np.array(null)
p_val = float((null >= obs).mean())
print(f"\nTest 2: observed mean|pos - grand| = {obs:.5f};  shuffled-position null = {null.mean():.5f} "
      f"+/- {null.std():.5f} (min {null.min():.5f}, max {null.max():.5f});  p = {p_val:.3f}")

# ---- Figure ----
rows = [("all stories", deviations(X)), ("half A", deviations(A)), ("half B", deviations(B))]
fig, ax = plt.subplots(3, P, figsize=(2.6 * P, 4.6))
for r, (name, d) in enumerate(rows):
    for p in range(P):
        ax[r, p].imshow(np.clip(d[p].transpose(1, 2, 0) * 10 + 0.5, 0, 1)); ax[r, p].axis("off")
        ax[r, p].set_title(f"{name}: (pos {p} - grand) x10", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "position_test.png", dpi=110)
print(f"figure: {OUT / 'position_test.png'}")
