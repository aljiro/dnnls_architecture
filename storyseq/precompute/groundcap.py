"""Cache GroundCap (daniel3303/GroundCap), the single-frame dataset StoryReasoning was built from,
for component pretraining (Level 9): more frames for the visual autoencoder, more
captions for the text language model. No sequences, so it never touches the predictor's windows.

Output: cache/groundcap.pt with
  pix [N, 3, 60, 125] uint8   the frame, resized like the story frames
  ids [N, T] long             BERT token ids of the caption with the grounding tags stripped (T = 100)
Run from the repository root: python -m storyseq.precompute.groundcap
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms

from storyseq.data import T, tokenizer

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
HW = (60, 125)


def strip_tags(caption: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"</?gd[oal][^>]*>", "", caption or "")).strip()


def main() -> None:
    tok = tokenizer()
    to_pix = transforms.Compose([transforms.Resize(HW), transforms.PILToTensor()])
    pix_all, ids_all = [], []
    t0 = time.time()
    for split in ("train", "test"):
        ds = load_dataset("daniel3303/GroundCap", split=split)
        n = len(ds)
        pix = torch.zeros(n, 3, *HW, dtype=torch.uint8)
        caps = []
        for i in range(n):
            row = ds[i]
            pix[i] = to_pix(row["image"].convert("RGB"))
            caps.append(strip_tags(row["caption"]))
            if i % 5000 == 0:
                print(f"[{split}] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
        ids = tok(caps, padding="max_length", truncation=True, max_length=T, return_tensors="pt").input_ids
        pix_all.append(pix); ids_all.append(ids)
        print(f"[{split}] {n} frames", flush=True)
    torch.save({"pix": torch.cat(pix_all), "ids": torch.cat(ids_all)}, CACHE / "groundcap.pt")
    print(f"saved {sum(len(p) for p in pix_all)} frames and captions to {CACHE / 'groundcap.pt'}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
