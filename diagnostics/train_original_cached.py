"""Train the ORIGINAL architecture (models.py) on real data, with the logging the notebook lacked.

Uses the cached 60x125 frames from poc/precompute.py (so an epoch takes a minute on CPU)
and the notebook's loss recipe: L1(image) + MSE(context vs batch-mean) + CE(text).
The text autoencoder is trained jointly because the notebook's pretrained weights live
on Google Drive; that only makes this variant more capable than the notebook, not less.

Reports per epoch:
  - each loss component averaged over the epoch (the notebook printed the last batch only)
  - validation L1 of the predicted image vs the target, next to two floors:
      * constant per-pixel median image (what a model that ignores its input converges to)
      * copy the 4th input frame
  - spread of the predicted image across the validation batch (0 = every input gives the same blob)
  - fraction of dead units in the visual latent

Flags: --epochs N, --no-ctx (drop the context loss), --no-equalize (skip histogram equalisation)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data import parse_gdi_text  # noqa: E402
from models import DecoderLSTM, EncoderLSTM, Seq2SeqLSTM, SequencePredictor, VisualAutoencoder  # noqa: E402
from training import init_weights  # noqa: E402

OUT = ROOT / "diagnostics" / "out"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "poc" / "cache"
K, T = 4, 120
torch.set_num_threads(16)
torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tokens(tok: BertTokenizer, n_stories: int) -> torch.Tensor:
    path = OUT / "tokens_train.pt"
    if path.exists():
        return torch.load(path)
    from datasets import load_dataset
    ds = load_dataset("daniel3303/StoryReasoning", split="train")
    ids = torch.full((n_stories, K + 1, T), tok.pad_token_id, dtype=torch.long)
    for i in range(n_stories):
        descs = [d["description"] for d in parse_gdi_text(ds[i]["story"])][:K + 1]
        ids[i, :len(descs)] = tok(descs, return_tensors="pt", padding="max_length", truncation=True, max_length=T).input_ids
    torch.save(ids, path)
    return ids


def prep_frames(pix: torch.Tensor, equalize: bool) -> torch.Tensor:
    """uint8 [B, 5, 3, H, W] -> float [B, 5, 3, H, W] in [0, 1], optionally equalised like the notebook."""
    if equalize:
        pix = torch.stack([torch.stack([TF.equalize(f) for f in story]) for story in pix])
    return pix.float() / 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--no-equalize", action="store_true")
    args = ap.parse_args()
    equalize = not args.no_equalize

    cache = torch.load(CACHE / "train.pt")
    keep = cache["n_frames"] >= K + 1
    pix = cache["pix"][keep][:, :K + 1]
    tok = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    ids = tokens(tok, len(cache["n_frames"]))[keep]
    n = len(pix)
    perm = torch.randperm(n)
    tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n):]
    print(f"stories: {n}  train {len(tr)}  val {len(va)}  equalize={equalize}  ctx_loss={not args.no_ctx}")

    # validation floors (on the same preprocessing the model sees)
    va_frames = prep_frames(pix[va], equalize)
    tgt = va_frames[:, K]
    median = tgt.median(0).values
    floor_median = F.l1_loss(median.expand_as(tgt), tgt).item()
    floor_copy = F.l1_loss(va_frames[:, K - 1], tgt).item()
    print(f"validation floors: constant median image L1 = {floor_median:.4f}   copy-4th-frame L1 = {floor_copy:.4f}")

    text = Seq2SeqLSTM(EncoderLSTM(tok.vocab_size, 16, 16), DecoderLSTM(tok.vocab_size, 16, 16))
    visual = VisualAutoencoder()
    visual.apply(init_weights)
    model = SequencePredictor(visual, text).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 8

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        sums = {"im": 0.0, "ctx": 0.0, "txt": 0.0}
        order = tr[torch.randperm(len(tr))]
        for b in range(0, len(order), bs):
            j = order[b:b + bs]
            frames = prep_frames(pix[j], equalize).to(DEVICE)
            inp, target = frames[:, :K], frames[:, K]
            desc, tgt_ids = ids[j, :K].to(DEVICE), ids[j, K:K + 1].to(DEVICE)
            content, context, logits, *_ = model(inp, desc, tgt_ids)
            l_im = F.l1_loss(content, target)
            l_ctx = F.mse_loss(context, inp.mean(dim=(0, 1)).unsqueeze(0).expand_as(context))
            l_txt = F.cross_entropy(logits.flatten(0, 1), tgt_ids.squeeze(1)[:, 1:].flatten(), ignore_index=tok.pad_token_id)
            loss = l_im + l_txt + (0 if args.no_ctx else l_ctx)
            opt.zero_grad(); loss.backward(); opt.step()
            sums["im"] += l_im.item() * len(j); sums["ctx"] += l_ctx.item() * len(j); sums["txt"] += l_txt.item() * len(j)

        model.eval()
        with torch.no_grad():
            preds, zs, fz = [], [], []
            for b in range(0, len(va), 32):
                j = va[b:b + 32]
                frames = prep_frames(pix[j], equalize).to(DEVICE)
                content, *_ = model(frames[:, :K], ids[j, :K].to(DEVICE), ids[j, K:K + 1].to(DEVICE))
                preds.append(content.cpu())
                zs.append(model.image_encoder(frames[:, :K].flatten(0, 1)).cpu())
                # fused latent that feeds the decoder: is it input-dependent?
                zv = zs[-1].to(DEVICE).view(len(j), K, -1)
                _, h, _ = model.text_encoder(ids[j, :K].to(DEVICE).flatten(0, 1))
                fused_in = torch.cat((zv, h[-1].view(len(j), K, -1)), -1)
                temporal, hN = model.temporal_rnn(fused_in)
                fz.append(model.projection(torch.cat((hN[-1], model.attention(temporal)), -1)).cpu())
            preds, zs, fz = torch.cat(preds), torch.cat(zs), torch.cat(fz)
            fused_spread = fz.std(0).mean().item()
            fused_dead = (fz.abs().max(0).values < 1e-6).float().mean().item()
            val_l1 = F.l1_loss(preds, tgt).item()
            spread = preds.std(0).mean().item()          # how different predictions are across inputs
            tgt_spread = tgt.std(0).mean().item()
            dead = (zs.abs().max(0).values < 1e-6).float().mean().item()
            l2_to_median = (preds - median).abs().mean().item()
        print(f"epoch {epoch + 1}/{args.epochs} ({time.time() - t0:.0f}s)  train im={sums['im'] / len(tr):.4f} "
              f"ctx={sums['ctx'] / len(tr):.4f} txt={sums['txt'] / len(tr):.3f} | val L1={val_l1:.4f} "
              f"(median floor {floor_median:.4f}, copy floor {floor_copy:.4f}) | pred spread {spread:.4f} "
              f"vs target spread {tgt_spread:.4f} | |pred - median| {l2_to_median:.4f} | dead backbone units {dead:.0%} | "
              f"fused latent: spread {fused_spread:.4f}, dead {fused_dead:.0%}")

    fig, ax = plt.subplots(4, K + 3, figsize=(2.2 * (K + 3), 5.5))
    for r in range(4):
        j = va[r]
        frames = prep_frames(pix[j:j + 1], equalize)[0]
        for c in range(K + 1):
            ax[r, c].imshow(frames[c].permute(1, 2, 0)); ax[r, c].axis("off")
        ax[r, K + 1].imshow(preds[r].permute(1, 2, 0).clamp(0, 1)); ax[r, K + 1].axis("off")
        ax[r, K + 2].imshow(median.permute(1, 2, 0)); ax[r, K + 2].axis("off")
    for c, t_ in enumerate(["in 1", "in 2", "in 3", "in 4", "target", "prediction", "median floor"]):
        ax[0, c].set_title(t_, fontsize=9)
    tag = ("eq" if equalize else "raw") + ("_noctx" if args.no_ctx else "")
    plt.tight_layout(); plt.savefig(OUT / f"original_{tag}.png", dpi=110)
    print(f"figure: {OUT / f'original_{tag}.png'}")


if __name__ == "__main__":
    main()
