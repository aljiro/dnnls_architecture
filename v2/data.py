"""Data access for the v2 pipeline, built on the cache written by poc/precompute.py.

poc/cache/{split}.pt holds, per story and frame (up to 10 frames):
  pix  [N, 10, 3, 60, 125] uint8   raw frames, no equalisation
  txt  [N, 10, 384]                MiniLM sentence embedding of the description
  img  [N, 10, 512]                CLIP image embedding (unused here, kept for comparison)
  n_frames [N]
This module adds BERT token ids of every description (for the LSTM text encoder option and
for the text decoder's teacher forcing), cached as poc/cache/tokens_{split}.pt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import parse_gdi_text  # noqa: E402

CACHE = ROOT / "poc" / "cache"
K = 4            # input frames per window
T = 100          # tokens per description (median is 98 BERT tokens; longer ones are truncated)
MAX_FRAMES = 10
TOKENIZER_NAME = "google-bert/bert-base-uncased"


def tokenizer() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(TOKENIZER_NAME)


def _build_tokens(split: str, n: int) -> torch.Tensor:
    from datasets import load_dataset
    tok = tokenizer()
    ds = load_dataset("daniel3303/StoryReasoning", split=split)
    ids = torch.full((n, MAX_FRAMES, T), tok.pad_token_id, dtype=torch.long)
    for i in range(n):
        descs = [d["description"] for d in parse_gdi_text(ds[i]["story"])][:MAX_FRAMES]
        if descs:
            ids[i, :len(descs)] = tok(descs, padding="max_length", truncation=True,
                                      max_length=T, return_tensors="pt").input_ids
    return ids


def load_split(split: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    d = torch.load(CACHE / f"{split}.pt")
    tok_path = CACHE / f"tokens_{split}.pt"
    if not tok_path.exists():
        print(f"tokenising {split} descriptions once ...", flush=True)
        torch.save(_build_tokens(split, len(d["n_frames"])), tok_path)
    d["ids"] = torch.load(tok_path)
    d.pop("img", None)
    return {k: v.to(device) for k, v in d.items()}


def windows(d: dict[str, torch.Tensor], k: int = K) -> tuple[torch.Tensor, torch.Tensor]:
    """(story index, target frame index) for every window of k inputs followed by a target."""
    n = d["n_frames"].cpu()
    idx = [(i, t) for i, c in enumerate(n.tolist()) for t in range(k, c)]
    idx = torch.tensor(idx, device=d["n_frames"].device)
    return idx[:, 0], idx[:, 1]


def all_frames(d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(story index, frame index) of every real frame, for autoencoder pretraining."""
    valid = torch.arange(MAX_FRAMES, device=d["n_frames"].device)[None] < d["n_frames"][:, None]
    nz = valid.nonzero()
    return nz[:, 0], nz[:, 1]


def gather(d: dict[str, torch.Tensor], s: torch.Tensor, t: torch.Tensor, k: int = K) -> dict[str, torch.Tensor]:
    """Assemble one batch of windows. Frames are float in [0, 1]."""
    pos = t[:, None] - k + torch.arange(k, device=t.device)[None]        # [B, K]
    return {
        "frames": d["pix"][s[:, None], pos].float() / 255,                # [B, K, 3, H, W]
        "target": d["pix"][s, t].float() / 255,                           # [B, 3, H, W]
        "txt": d["txt"][s[:, None], pos],                                 # [B, K, 384]
        "ids": d["ids"][s[:, None], pos],                                 # [B, K, T]
        "target_txt": d["txt"][s, t],                                     # [B, 384]
        "target_ids": d["ids"][s, t],                                     # [B, T]
    }
