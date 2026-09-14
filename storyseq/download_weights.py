"""Download the published weights so every version can be visualised or fine-tuned without training.

The weights live as assets of a GitHub release of this repository (tag WEIGHTS_TAG), not in git:
  pretrained/visual_ae.pt, pretrained/visual_ae_w2.pt, pretrained/text_lm.pt   (v1, v2 outputs)
  <version>/out/predictor_<version>.pt                                          (v3 ... v10)

Run from the repository root:
    python -m storyseq.download_weights            # everything
    python -m storyseq.download_weights v4_attention v10_semantic --pretrained
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "aljiro/dnnls_architecture"
WEIGHTS_TAG = "weights-v1"
VERSIONS = ["v3_multimodal_fusion", "v4_attention", "v5_protect", "v6_annotations", "v7_names",
            "v8_variational", "v9_scaling", "v10_semantic"]
PRETRAINED = ["visual_ae.pt", "visual_ae_w2.pt", "text_lm.pt"]


def fetch(asset: str, dest: Path) -> None:
    url = f"https://github.com/{REPO}/releases/download/{WEIGHTS_TAG}/{asset}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"exists   {dest.relative_to(ROOT)}")
        return
    print(f"download {asset} -> {dest.relative_to(ROOT)}", flush=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="*", default=VERSIONS, help="version directories (default: all)")
    ap.add_argument("--pretrained", action="store_true", help="also the v1 / v2 outputs (implied when no versions are given)")
    a = ap.parse_args()
    if a.pretrained or not sys.argv[1:]:
        for name in PRETRAINED:
            fetch(name, ROOT / "pretrained" / name)
    for v in a.versions:
        fetch(f"predictor_{v}.pt", ROOT / v / "out" / f"predictor_{v}.pt")


if __name__ == "__main__":
    main()
