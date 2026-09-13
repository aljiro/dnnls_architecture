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
MAX_FRAMES = 22          # all frames of every story (scaling stage; was 10)
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


CPU_RESIDENT = ("pix", "ent_pix", "ent_clip")     # large tensors stay in CPU memory; gather() moves batches


def load_split(split: str, device: str = "cpu", annotations: bool = False) -> dict[str, torch.Tensor]:
    d = torch.load(CACHE / f"{split}.pt")
    if annotations:                       # stage C: chain-of-thought annotations (v2/precompute_annotations.py)
        d.update(torch.load(CACHE / f"annot_{split}.pt"))
    tok_path = CACHE / f"tokens_{split}.pt"
    if tok_path.exists() and torch.load(tok_path).shape[1] != MAX_FRAMES:
        tok_path.unlink()                 # built for a different MAX_FRAMES
    if not tok_path.exists():
        print(f"tokenising {split} descriptions once ...", flush=True)
        torch.save(_build_tokens(split, len(d["n_frames"])), tok_path)
    d["ids"] = torch.load(tok_path)
    d["clip"] = d.pop("img")              # CLIP frame embeddings [N, MAX_FRAMES, 512]
    out = {}
    for k, v in d.items():
        if k in CPU_RESIDENT:
            out[k] = v.pin_memory() if device != "cpu" and torch.cuda.is_available() else v
        else:
            out[k] = v.to(device)
    return out


def model_kwargs(model, batch: dict) -> dict:
    """The optional forward() arguments a given model variant needs, taken from a gathered batch."""
    kw = {}
    if getattr(model, "annotated", False):
        kw.update(set_emb=batch["set_emb"], ent_slot=batch["ent_slot"])
        if getattr(model, "entity_features", "ae") == "clip":
            kw["ent_clip"] = batch["ent_clip"]
        else:
            kw["ent_pix"] = batch["ent_pix"]
    if getattr(model, "stage", "0") == "D":
        kw.update(chars_in=batch["chars_in"], slot_name_ids=batch["slot_name_ids"])
    if getattr(model, "clip_input", False):
        kw["clip"] = batch["clip"]
    return kw


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
    """Assemble one batch of windows on the device of the small tensors. Frames are float in [0, 1]."""
    dev = d["txt"].device
    pos = t[:, None] - k + torch.arange(k, device=t.device)[None]        # [B, K]
    sc, tc, posc = s.cpu(), t.cpu(), pos.cpu()                           # indices for the CPU-resident tensors
    out = {
        "frames": d["pix"][sc[:, None], posc].to(dev, non_blocking=True).float() / 255,   # [B, K, 3, H, W]
        "target": d["pix"][sc, tc].to(dev, non_blocking=True).float() / 255,             # [B, 3, H, W]
        "txt": d["txt"][s[:, None], pos],                                 # [B, K, 384]
        "ids": d["ids"][s[:, None], pos],                                 # [B, K, T]
        "clip": d["clip"][s[:, None], pos],                               # [B, K, 512]
        "target_txt": d["txt"][s, t],                                     # [B, 384]
        "target_ids": d["ids"][s, t],                                     # [B, T]
        "target_clip": d["clip"][s, t],                                   # [B, 512]
    }
    if "set_emb" in d:                                                    # stage C fields
        out.update({
            "set_emb": d["set_emb"][s[:, None], pos],                     # [B, K, 384]
            "ent_pix": d["ent_pix"][sc[:, None], posc].to(dev, non_blocking=True),   # [B, K, M, 3, h, w] uint8
            "ent_clip": d["ent_clip"][sc[:, None], posc].to(dev, non_blocking=True).float() if "ent_clip" in d else None,
            "ent_slot": d["ent_slot"][s[:, None], pos],                   # [B, K, M]
            "chars_in": d["char_present"][s[:, None], pos],               # [B, K, S] bool
            "target_chars": d["char_present"][s, t],                      # [B, S] bool
            "target_set": d["set_emb"][s, t],                             # [B, 384]
            "n_chars": d["n_chars"][s],                                   # [B]
            "slot_name_ids": d["slot_name_ids"][s],                       # [B, S, 6]
        })
    return out
