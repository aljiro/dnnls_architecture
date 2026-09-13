"""Figure of v2 predictions in the notebook's layout: a row of images filling the width, the
description under each one; the last two columns are the target and the prediction, with the
true and the generated description.

Used by v2/train.py at the end of training, and runnable on its own from a checkpoint:
  python v2/visualize.py [--text-encoder minilm] [--tag ""] [--rows 4] [--sample]
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.data import K, gather, load_split, tokenizer, windows  # noqa: E402
from v2.models import SequencePredictor, TextEncoderLSTM, VisualAutoencoder  # noqa: E402

OUT = ROOT / "v2" / "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def make_figure(model: SequencePredictor, d: dict, tok, mode: str, path: Path, rows: int = 4,
                seed: int = 0, max_chars: int = 230, sample: bool = False) -> None:
    model.eval()
    s, t = windows(d)
    g = torch.Generator(device="cpu").manual_seed(seed)
    pick = torch.randperm(len(s), generator=g)[:rows].to(s.device)
    batch = gather(d, s[pick], t[pick])
    text = batch["txt"] if mode == "minilm" else batch["ids"]
    extra = {k: batch[k] for k in ("set_emb", "ent_pix", "ent_slot") if model.stage == "C"}
    o = model(batch["frames"], text, batch["target_ids"][:, :-1], **extra)
    img = o["image"]
    gen = model.text_decoder.generate(o["cond"], tok.cls_token_id, tok.sep_token_id, max_len=70, sample=sample)

    cols = K + 2
    fig, ax = plt.subplots(2 * rows, cols, figsize=(3.4 * cols, 3.0 * rows),
                           gridspec_kw={"height_ratios": [1.0, 0.75] * rows, "wspace": 0.03, "hspace": 0.06})
    for r in range(rows):
        tiles = [batch["frames"][r, i] for i in range(K)] + [batch["target"][r], img[r]]
        texts = ([tok.decode(batch["ids"][r, i], skip_special_tokens=True) for i in range(K)]
                 + [tok.decode(batch["target_ids"][r], skip_special_tokens=True), tok.decode(gen[r])])
        for c in range(cols):
            a = ax[2 * r, c]
            a.imshow(tiles[c].permute(1, 2, 0).clamp(0, 1).cpu(), aspect="auto")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            name = ["input 1", "input 2", "input 3", "input 4", "target", "prediction"][c] if r == 0 else ""
            if model.stage != "0" and c < K:
                name = f"{name}   attention {o['alpha'][r, c]:.2f}".strip()
            if model.stage != "0" and c == K + 1:
                name = f"{name}   gate {o['gate'][r]:.2f}".strip()
            if name:
                a.set_title(name, fontsize=9 if r else 10)
            b = ax[2 * r + 1, c]
            b.axis("off")
            b.text(0.5, 1.0, textwrap.fill(texts[c][:max_chars], 46), ha="center", va="top", fontsize=7.5, wrap=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"figure: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "A", "B", "C"], default="A")
    ap.add_argument("--text-encoder", choices=["minilm", "lstm"], default="minilm")
    ap.add_argument("--tag", default="")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", action="store_true", help="nucleus sampling instead of greedy decoding")
    ap.add_argument("--out", default="", help="output file (default v2/out/predictions_<name>.png)")
    ap.add_argument("--copy-path", action="store_true")
    args = ap.parse_args()
    tok = tokenizer()
    te = load_split("test", DEVICE, annotations=args.stage == "C")
    text_encoder = TextEncoderLSTM(tok.vocab_size, tok.pad_token_id) if args.text_encoder == "lstm" else None
    model = SequencePredictor(VisualAutoencoder(), 384 if text_encoder is None else text_encoder.out_dim,
                              tok.vocab_size, text_encoder, stage=args.stage, copy_path=args.copy_path).to(DEVICE)
    name = f"stage{args.stage}_{args.text_encoder}{args.tag}"
    model.load_state_dict(torch.load(OUT / f"predictor_{name}.pt", map_location=DEVICE))
    out = Path(args.out) if args.out else OUT / f"predictions_{name}.png"
    make_figure(model, te, tok, args.text_encoder, out, args.rows, args.seed, sample=args.sample)


if __name__ == "__main__":
    main()
