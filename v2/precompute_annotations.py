"""Cache the chain-of-thought annotations for stage C, once per split.

Per story and frame (up to MAX_FRAMES = 10):
  set_emb      [N, 10, 384]              MiniLM embedding of the frame's setting table
                                         ("Location: ...; Lighting: ...; Time: ...; Mood: ...")
  char_present [N, 10, S] bool           which of the story's characters (slot 0..S-1, S = 8,
                                         ordered by first appearance) are in the frame
  ent_pix      [N, 10, M, 3, 30, 62] u8  crops of up to M = 4 character boxes per frame
  ent_slot     [N, 10, M] long           the slot index of each crop, -1 if none
  n_chars      [N] long                  number of distinct characters in the story (capped at S)
  slot_name_ids [N, S, 6] long           BERT token ids of each slot's name (from the "Name" column), padded
  ent_clip     [N, 10, M, 512] f16       CLIP ViT-B/32 embedding of each crop (scaling stage)

Output: poc/cache/annot_{split}.pt.  Run: python v2/precompute_annotations.py
"""

from __future__ import annotations

import collections
import re
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer, BertTokenizerFast, CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import _markdown_rows, parse_cot_grounding  # noqa: E402

CACHE = ROOT / "poc" / "cache"
MAX_FRAMES, S, M = 22, 8, 4   # all frames (was 10 until the scaling stage)
CROP_HW = (30, 62)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PRONOUNS = {"he", "she", "him", "her", "his", "hers", "they", "them", "their", "it", "himself", "herself", "themselves", "who",
            "i", "me", "my", "mine", "myself", "we", "us", "our", "you", "your"}


def character_names(cot: str, story: str = "") -> dict[str, str]:
    """Character ID -> name. Prefer the story's grounded mentions (<gdo char2>Mrs. Patel</gdo>): the most
    frequent mention that is not a pronoun. Fall back to the chain-of-thought table's Name column (a role)."""
    mentions: dict[str, collections.Counter] = {}
    for cid, text in re.findall(r"<gdo\s+(char\d+)[^>]*>([^<]+)</gdo>", story or ""):
        t = re.sub(r"\s+", " ", text).strip().strip(".,;:'\"")
        if t and t.lower() not in PRONOUNS:
            mentions.setdefault(cid, collections.Counter())[t] += 1
    names = {cid: c.most_common(1)[0][0] for cid, c in mentions.items()}
    for table in re.finditer(r"###\s*Characters(.*?)(?=\n###|\n##|$)", cot or "", re.DOTALL):
        for row in _markdown_rows(table.group(1)):
            cid, name = row.get("Character ID", ""), row.get("Name", "")
            if cid and name and cid not in names:
                names[cid] = name
    return names


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
    bert = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def clip_embed(images: list) -> torch.Tensor:
        pv = clip_proc(images=images, return_tensors="pt")["pixel_values"].to(DEVICE)
        feats = clip.get_image_features(pixel_values=pv)
        feats = feats if torch.is_tensor(feats) else feats.pooler_output
        return torch.nn.functional.normalize(feats.float(), dim=-1).half().cpu()

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
        slot_name_ids = torch.zeros(n, S, 6, dtype=torch.long)
        ent_clip = torch.zeros(n, MAX_FRAMES, M, 512, dtype=torch.float16)
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
            names = character_names(cot, row.get("story", ""))
            for cid, slot in slots.items():
                name = names.get(cid, "").lower().replace("the ", "")
                if name:
                    ids = bert(name, add_special_tokens=False).input_ids[:6]
                    slot_name_ids[i, slot, :len(ids)] = torch.tensor(ids)
            texts, idx = [], []
            crops, crop_pos = [], []
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
                            crop = img.crop((x1, y1, x2, y2)).convert("RGB")
                            ent_pix[i, f, k] = to_pix(crop)
                            ent_slot[i, f, k] = slot
                            crops.append(crop); crop_pos.append((f, k))
                            k += 1
            if texts:
                set_emb[i, idx] = embed(texts)
            if crops:
                ce = clip_embed(crops)
                for (f, k), e in zip(crop_pos, ce):
                    ent_clip[i, f, k] = e
            if i % 500 == 0:
                print(f"[{split}] {i}/{n}  {time.time() - t0:.0f}s", flush=True)
        torch.save({"set_emb": set_emb, "char_present": char_present, "ent_pix": ent_pix,
                    "ent_slot": ent_slot, "n_chars": n_chars, "slot_name_ids": slot_name_ids, "ent_clip": ent_clip},
                   CACHE / f"annot_{split}.pt")
        have = (ent_slot >= 0).sum().item()
        print(f"[{split}] saved: {have} character crops, mean chars/story {n_chars.float().mean():.2f}, "
              f"frames with setting text {(set_emb.abs().sum(-1) > 0).float().mean():.0%}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
