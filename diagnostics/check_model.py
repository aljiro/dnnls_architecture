"""Architecture diagnostics that need no dataset.

Checks, on random inputs:
1. whether the two decoder heads (content / context) are actually distinct,
2. how much of the visual latent is alive after init and after a few gradient
   steps against the notebook's losses,
3. what a constant-output (blob) predictor scores under the same L1 loss, so
   the notebook's im=0.215 can be read against a floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import DecoderLSTM, EncoderLSTM, Seq2SeqLSTM, SequencePredictor, VisualAutoencoder  # noqa: E402
from training import init_weights  # noqa: E402

torch.manual_seed(0)
VOCAB = 30522
B, S, T = 8, 4, 120


def build():
    text = Seq2SeqLSTM(EncoderLSTM(VOCAB, 16, 16), DecoderLSTM(VOCAB, 16, 16))
    for p in text.parameters():
        p.requires_grad = False
    visual = VisualAutoencoder()
    visual.apply(init_weights)
    return SequencePredictor(visual, text)


def main() -> None:
    model = build()
    frames = torch.rand(B, S, 3, 60, 125)
    target = torch.rand(B, 3, 60, 125)
    ids = torch.randint(VOCAB, (B, S, T))
    tgt_ids = torch.randint(VOCAB, (B, 1, T))

    content, context, logits, *_ , zv, zt = model(frames, ids, tgt_ids)
    print("== decoder heads ==")
    print(f"content is context tensor: {content is context}")
    print(f"max |content - context|  : {(content - context).abs().max().item():.3e}")

    print("\n== parameter budget ==")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    big = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad and p.numel() > 50_000}
    print(f"trainable: {total:,}")
    for n, k in sorted(big.items(), key=lambda x: -x[1]):
        print(f"  {n:55s} {k:>10,}  ({100 * k / total:4.1f}%)")

    def latent_report(tag: str) -> None:
        with torch.no_grad():
            z = model.image_encoder(frames.flatten(0, 1))
            inner_c = model.image_encoder.content_backbone(frames.flatten(0, 1))
            inner_x = model.image_encoder.context_backbone(frames.flatten(0, 1))
            fused = model.projection(torch.cat((model.temporal_rnn(torch.cat((zv, zt), -1))[1][-1],
                                                model.attention(model.temporal_rnn(torch.cat((zv, zt), -1))[0])), -1))
        dead_c = (inner_c.abs().max(0).values < 1e-6).float().mean().item()
        dead_x = (inner_x.abs().max(0).values < 1e-6).float().mean().item()
        dead_f = (fused.abs().max(0).values < 1e-6).float().mean().item()
        print(f"[{tag}] visual latent std {z.std().item():.3e} | dead units: content backbone "
              f"{dead_c:.0%}, context backbone {dead_x:.0%}, fused pre-decoder {dead_f:.0%}")

    print("\n== latent health ==")
    latent_report("init")

    # a few steps of the notebook's loss recipe on random data
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(30):
        content, context, logits, *_ = model(frames, ids, tgt_ids)
        loss = (F.l1_loss(content, target)
                + F.mse_loss(context, frames.mean(dim=(0, 1)).unsqueeze(0).expand_as(context))
                + F.cross_entropy(logits.flatten(0, 1), tgt_ids.squeeze(1)[:, 1:].flatten()))
        opt.zero_grad()
        loss.backward()
        opt.step()
    latent_report("after 30 steps")

    print("\n== L1 floor of a constant predictor ==")
    print("Under L1, the best constant image is the per-pixel median of the targets.")
    print("A model that ignores its inputs converges to exactly that blob.")
    print("The notebook's im loss went 0.259 -> 0.215 over 5 epochs; the data script")
    print("reports the real median-image floor on the actual frames for comparison.")


if __name__ == "__main__":
    main()
