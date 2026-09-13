"""Probe a trained v2 predictor: how much do its outputs depend on the inputs?

Text: cross-entropy of the target description under the decoder when conditioned on
  (a) the predicted latent z for that window, (b) z of a random other window (shuffled),
  (c) a zero vector. If (a) is close to (b), the decoder is an unconditional language model
  and the conditioning is ignored.
Image: L1 of the prediction against its own target vs against a random other target.
  If the two are equal, the prediction is not specific to its input.

Run: python v2/probe.py [--text-encoder minilm|lstm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.data import gather, load_split, tokenizer, windows  # noqa: E402
from v2.models import SequencePredictor, TextEncoderLSTM, VisualAutoencoder  # noqa: E402

OUT = ROOT / "v2" / "out"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-encoder", choices=["minilm", "lstm"], default="minilm")
    args = ap.parse_args()
    tok = tokenizer()
    te = load_split("test", DEVICE)
    s, t = windows(te)
    text_encoder = TextEncoderLSTM(tok.vocab_size, tok.pad_token_id) if args.text_encoder == "lstm" else None
    model = SequencePredictor(VisualAutoencoder(), 384 if text_encoder is None else text_encoder.out_dim,
                              tok.vocab_size, text_encoder).to(DEVICE)
    model.load_state_dict(torch.load(OUT / f"predictor_{args.text_encoder}.pt", map_location=DEVICE))
    model.eval()

    ce = {"true z": 0.0, "shuffled z": 0.0, "zero z": 0.0}
    n_tok = 0
    l1_own, l1_other = 0.0, 0.0
    for b in range(0, len(s), 64):
        batch = gather(te, s[b:b + 64], t[b:b + 64])
        text = batch["txt"] if args.text_encoder == "minilm" else batch["ids"]
        img, _, z, _ = model(batch["frames"], text, batch["target_ids"][:, :-1])
        tgt = batch["target_ids"][:, 1:]
        n_tok += (tgt != tok.pad_token_id).sum().item()
        perm = torch.randperm(len(z), device=DEVICE)
        for name, cond in (("true z", z), ("shuffled z", z[perm]), ("zero z", torch.zeros_like(z))):
            logits = model.text_decoder(batch["target_ids"][:, :-1], cond)
            ce[name] += F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=tok.pad_token_id, reduction="sum").item()
        l1_own += F.l1_loss(img, batch["target"], reduction="sum").item()
        l1_other += F.l1_loss(img, batch["target"][perm], reduction="sum").item()
    n_pix = len(s) * 3 * 60 * 125
    print(f"text CE / perplexity on {len(s)} test windows:")
    for name, v in ce.items():
        print(f"  {name:11s} CE {v / n_tok:.3f}  ppl {torch.exp(torch.tensor(v / n_tok)).item():6.1f}")
    print(f"image L1: vs own target {l1_own / n_pix:.4f}   vs a random other target {l1_other / n_pix:.4f}")

    # what does the decoder generate for a few windows, greedy vs sampled
    batch = gather(te, s[:3], t[:3])
    text = batch["txt"] if args.text_encoder == "minilm" else batch["ids"]
    z = model.predict_latent(batch["frames"], text)
    for i, g in enumerate(model.text_decoder.generate(z, tok.cls_token_id, tok.sep_token_id, max_len=40)):
        print(f"greedy {i}: {tok.decode(g)[:160]}")


if __name__ == "__main__":
    main()
