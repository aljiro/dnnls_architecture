"""Cache the chain-of-thought annotations for stage C, once per split.

Per story and frame (up to MAX_FRAMES = 10):
  set_emb      [N, 10, 384]              MiniLM embedding of the frame's setting table
                                         ("Location: ...; Lighting: ...; Time: ...; Mood: ...")
  char_present [N, 10, S] bool           which of the story's characters (slot 0..S-1, S = 8,
                                         ordered by first appearance) are in the frame
  ent_pix      [N, 10, M, 3, 30, 62] u8  crops of up to M = 4 character boxes per frame
  ent_slot     [N, 10, M] long           the slot index of each crop, -1 if none
  n_chars      [N] long                  number of distinct characters in the story (capped at S)

Output: poc/cache/annot_{split}.pt.  Run: python v2/precompute_annotations.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import _markdown_rows, parse_cot_grounding  # noqa: E402

CACHE = ROOT / "poc" / "cache"
MAX_FRAMES, S, M = 10, 8, 4
CROP_HW = (30, 62)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setting_text(cot: str) -> dict[int, str]:
    """Frame index -> one sentence built from the '### Setting' table of that frame."""
    out: dict[int, str] = {}
    matches = list(re.finditer(r"^##\s*Image\s+(\d+)", cot or "", re.MULTILINE))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(cot)
        section = cot[m.end():end]
        table = re.search(r"###\s*Setting(.*?)(?=\n###|\n##|$)", section, re.DOTALL)
        if not table:
            continue
        parts = []
        for row in _markdown_rows(table.group(1)):
            elem, desc = row.get("Setting Element", ""), row.get("Description", "")
            extra = ", ".join(v for k in ("Mood", "Time") if (v := row.get(k, "")) and v != "N/A")
            if elem or desc:
                parts.append(f"{elem}: {desc}" + (f" ({extra})" if extra else ""))
        out[int(m.group(1)) - 1] = "; ".join(parts)
    return out


@torch.no_grad()
def main() -> None:
    st_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    st = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE).eval()
    to_pix = transforms.Compose([transforms.Resize(CROP_HW), transforms.PILToTensor()])

    def embed(texts: list[str]) -> torch.Tensor:
        enc = st_tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        h = st(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu()

    for split in ("train", "test"):
        ds = load_dataset("daniel3303/StoryReasoning", split=split)
        n = len(ds)
        set_emb = torch.zeros(n, MAX_FRAMES, 384)
        char_present = torch.zeros(n, MAX_FRAMES, S, dtype=torch.bool)
        ent_pix = torch.zeros(n, MAX_FRAMES, M, 3, *CROP_HW, dtype=torch.uint8)
        ent_slot = torch.full((n, MAX_FRAMES, M), -1, dtype=torch.long)
        n_chars = torch.zeros(n, dtype=torch.long)
        t0 = time.time()
        for i in range(n):
            row = ds[i]
            cot = row.get("chain_of_thought", "") or ""
            frames = parse_cot_grounding(cot)
            settings = setting_text(cot)
            slots: dict[str, int] = {}
            for f in sorted(frames):
                for det in frames[f]["characters"]:
                    if det["id"] not in slots and len(slots) < S:
                        slots[det["id"]] = len(slots)
            n_chars[i] = len(slots)
            texts, idx = [], []
            for f in range(min(MAX_FRAMES, len(row["images"]))):
                if f in settings and settings[f]:
                    texts.append(settings[f]); idx.append(f)
                if f not in frames:
                    continue
                img = row["images"][f]
                W, H = img.size
                k = 0
                for det in frames[f]["characters"]:
                    slot = slots.get(det["id"])
                    if slot is None:
                        continue
                    char_present[i, f, slot] = True
                    if k < M:
                        x1, y1, x2, y2 = det["bbox"]
                        x1, x2 = max(0, min(x1, W - 1)), max(1, min(x2, W))
                        y1, y2 = max(0, min(y1, H - 1)), max(1, min(y2, H))
                        if x2 > x1 and y2 > y1:
                            ent_pix[i, f, k] = to_pix(img.crop((x1, y1, x2, y2)).convert("RGB"))
                            ent_slot[i, f, k] = slot
                            k += 1
            if texts:
                set_emb[i, idx] = embed(texts)
            if i % 500 == 0:
                print(f"[{split}] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
        torch.save({"set_emb": set_emb, "char_present": char_present, "ent_pix": ent_pix,
                    "ent_slot": ent_slot, "n_chars": n_chars}, CACHE / f"annot_{split}.pt")
        have = (ent_slot >= 0).sum().item()
        print(f"[{split}] saved: {have} character crops, mean chars/story {n_chars.float().mean():.2f}, "
              f"frames with setting text {(set_emb.abs().sum(-1) > 0).float().mean():.0%}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
