"""Cache StoryReasoning frames and frozen features once, so every level trains on tensors.

Run from the repository root: python -m storyseq.precompute.frames

Per story (train and test splits), for up to MAX_FRAMES frames:
  - CLIP ViT-B/32 image embedding of the frame          (512-d, L2-normalised)
  - MiniLM sentence embedding of the frame description  (384-d, L2-normalised)
  - the frame resized to 60x125 as uint8                (for visualisation / pixel decoder)

Output: cache/{split}.pt with tensors
  img [N, MAX_FRAMES, 512], txt [N, MAX_FRAMES, 384], n_frames [N], pix [N, MAX_FRAMES, 3, 60, 125] uint8

Runtime: a few minutes on a laptop GPU, dominated by JPEG decoding.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from storyseq.precompute.parsing import parse_gdi_text

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
MAX_FRAMES = 22   # covers every story (max frame count 22); was 10 until the scaling stage
HW = (60, 125)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(16)


@torch.no_grad()
def main() -> None:
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    if DEVICE == "cuda":
        clip = clip.half()
    dtype = next(clip.parameters()).dtype
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    st_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    st = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE).eval()
    to_pix = transforms.Compose([transforms.Resize(HW), transforms.PILToTensor()])

    def embed_text(texts: list[str]) -> torch.Tensor:
        enc = st_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        h = st(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu()

    for split in ("train", "test"):
        ds = load_dataset("daniel3303/StoryReasoning", split=split)
        n = len(ds)
        img = torch.zeros(n, MAX_FRAMES, 512)
        txt = torch.zeros(n, MAX_FRAMES, 384)
        pix = torch.zeros(n, MAX_FRAMES, 3, *HW, dtype=torch.uint8)
        n_frames = torch.zeros(n, dtype=torch.long)
        t0 = time.time()
        for i in range(n):
            row = ds[i]
            frames = row["images"][:MAX_FRAMES]
            descs = [d["description"] for d in parse_gdi_text(row["story"])][:MAX_FRAMES]
            k = min(len(frames), len(descs))
            if k < 2:
                continue
            frames, descs = frames[:k], descs[:k]
            n_frames[i] = k
            pv = proc(images=[f.convert("RGB") for f in frames], return_tensors="pt")["pixel_values"].to(DEVICE, dtype)
            feats = clip.get_image_features(pixel_values=pv)
            feats = feats if torch.is_tensor(feats) else feats.pooler_output   # transformers >= 5 returns an output object
            img[i, :k] = torch.nn.functional.normalize(feats.float(), dim=-1).cpu()
            txt[i, :k] = embed_text(descs)
            pix[i, :k] = torch.stack([to_pix(f.convert("RGB")) for f in frames])
            if i % 250 == 0:
                print(f"[{split}] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
        torch.save({"img": img, "txt": txt, "pix": pix, "n_frames": n_frames}, CACHE / f"{split}.pt")
        print(f"[{split}] saved {n} stories, {int(n_frames.sum())} frames, {time.time() - t0:.0f}s")
    print("frame-count histogram (train):", np.bincount(n_frames.numpy()).tolist())


if __name__ == "__main__":
    main()
