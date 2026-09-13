"""Pretrain the text decoder as an unconditional language model on every cached description.

The predictor's text decoder only ever sees the 13.6k window targets; the cache holds about
31k descriptions (all frames of all training stories). This trains the same TextDecoder with a
zero condition on all of them, reports test perplexity, and saves v2/out/text_lm.pt, which
v2/train.py loads into the decoder (embedding, LSTM, output layer) before conditional training.

Run: python v2/pretrain_text.py [--epochs 8]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.data import load_split, tokenizer  # noqa: E402
from v2.models import TextDecoder  # noqa: E402

OUT = ROOT / "v2" / "out"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def descriptions(d: dict, pad_id: int) -> torch.Tensor:
    ids = d["ids"].flatten(0, 1)                                   # [N*10, T]
    return ids[(ids != pad_id).sum(1) > 2]                         # drop empty slots


def perplexity(model: TextDecoder, ids: torch.Tensor, pad_id: int) -> float:
    model.eval()
    ce, n = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(ids), 256):
            x = ids[b:b + 256]
            logits = model(x[:, :-1], torch.zeros(len(x), 256, device=DEVICE))
            ce += F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), ignore_index=pad_id, reduction="sum").item()
            n += (x[:, 1:] != pad_id).sum().item()
    return float(torch.exp(torch.tensor(ce / n)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--extra", choices=["", "groundcap"], default="", help="add GroundCap captions to the training set")
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = tokenizer()
    tr = descriptions(load_split("train", DEVICE), tok.pad_token_id)
    if args.extra == "groundcap":
        gc = torch.load(ROOT / "poc" / "cache" / "groundcap.pt")["ids"].to(DEVICE)
        tr = torch.cat((tr, gc[(gc != tok.pad_token_id).sum(1) > 2]))
    te = descriptions(load_split("test", DEVICE), tok.pad_token_id)
    print(f"descriptions: train {len(tr)}  test {len(te)}  device {DEVICE}")
    model = TextDecoder(tok.vocab_size, cond_dim=256).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(tr) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    zero = torch.zeros(args.batch_size, 256, device=DEVICE)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr), device=DEVICE)
        tot = 0.0
        for b in range(len(tr) // args.batch_size):
            x = tr[perm[b * args.batch_size:(b + 1) * args.batch_size]]
            loss = F.cross_entropy(model(x[:, :-1], zero).flatten(0, 1), x[:, 1:].flatten(), ignore_index=tok.pad_token_id)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} ({time.time() - t0:.0f}s)  train CE {tot / (b + 1):.3f}  "
              f"test perplexity {perplexity(model, te, tok.pad_token_id):.1f}", flush=True)
    torch.save(model.state_dict(), OUT / "text_lm.pt")
    model.eval()
    gen = model.generate(torch.zeros(3, 256, device=DEVICE), tok.cls_token_id, tok.sep_token_id, max_len=40, sample=True, temperature=0.6)
    for g in gen:
        print("sample:", tok.decode(g))
    print(f"saved {OUT / 'text_lm.pt'}")


if __name__ == "__main__":
    main()
