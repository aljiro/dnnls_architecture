"""Train the v2 sequence predictor: 4 (frame, description) pairs -> next frame image and description.

Pipeline (see v2/models.py):
  image encoder (pretrained v2 autoencoder) + text encoder (frozen MiniLM vectors, or --text-encoder lstm)
  -> fuse -> GRU -> attention -> latent z
  -> image decoder (pretrained, fine-tuned)      pixel L1 to the true next frame
  -> latent target                               1 - cos(z, encoder(true next frame))
  -> text decoder conditioned on z               cross-entropy with teacher forcing

Logged per epoch on the validation split (20 % of train stories, split by story):
  image L1 vs constant-median floor and copy-last floor; spread of predictions across inputs
  (0.0000 = the old failure); retrieval of the true next frame by latent cosine among all
  validation targets (R@1, R@10, chance); text cross-entropy and perplexity.
Final numbers on the test split; figure v2/out/predictions_{text-encoder}.png with generated text.

Run after v2/pretrain_visual.py:
  python v2/train.py                       # MiniLM text encoder
  python v2/train.py --text-encoder lstm   # from-scratch LSTM text encoder
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v2.data import K, gather, load_split, tokenizer, windows  # noqa: E402
from v2.models import SequencePredictor, TextEncoderLSTM, VisualAutoencoder, latent_loss  # noqa: E402

OUT = ROOT / "v2" / "out"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def text_input(batch: dict, mode: str) -> torch.Tensor:
    return batch["txt"] if mode == "minilm" else batch["ids"]


@torch.no_grad()
def evaluate(model: SequencePredictor, d: dict, s: torch.Tensor, t: torch.Tensor, mode: str,
             pad_id: int, tag: str) -> dict[str, float]:
    model.eval()
    preds, zs, z_true, last_frames, e_pred, e_true, ce, ce_shuf, n_tok = [], [], [], [], [], [], 0.0, 0.0, 0
    for b in range(0, len(s), 64):
        batch = gather(d, s[b:b + 64], t[b:b + 64])
        img, logits, z, e_txt = model(batch["frames"], text_input(batch, mode), batch["target_ids"][:, :-1])
        preds.append(img); zs.append(z); e_pred.append(e_txt); e_true.append(batch["target_txt"])
        z_true.append(model.image_encoder(batch["target"]))
        last_frames.append(batch["frames"][:, -1])
        tgt = batch["target_ids"][:, 1:]
        ce += F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=pad_id, reduction="sum").item()
        logits_shuf = model.text_decoder(batch["target_ids"][:, :-1], z[torch.randperm(len(z), device=z.device)])
        ce_shuf += F.cross_entropy(logits_shuf.flatten(0, 1), tgt.flatten(), ignore_index=pad_id, reduction="sum").item()
        n_tok += (tgt != pad_id).sum().item()
    preds, zs, z_true, last_frames, e_pred, e_true = map(torch.cat, (preds, zs, z_true, last_frames, e_pred, e_true))
    sim_t = F.normalize(e_pred, dim=-1) @ F.normalize(e_true, dim=-1).T
    hit_t = sim_t.argsort(1, descending=True) == torch.arange(len(s), device=DEVICE)[:, None]
    target = d["pix"][s, t].float() / 255
    median = target.median(0).values
    sim = F.normalize(zs, dim=-1) @ F.normalize(z_true, dim=-1).T
    rank = sim.argsort(1, descending=True)
    hit = rank == torch.arange(len(s), device=DEVICE)[:, None]
    res = {
        "img_L1": F.l1_loss(preds, target).item(),
        "floor_median": F.l1_loss(median.expand_as(target), target).item(),
        "floor_copy": F.l1_loss(last_frames, target).item(),
        "pred_spread": preds.std(0).mean().item(),
        "target_spread": target.std(0).mean().item(),
        "latent_R@1": hit[:, 0].float().mean().item(),
        "latent_R@10": hit[:, :10].any(1).float().mean().item(),
        "latent_chance": 1 / len(s),
        "text_R@1": hit_t[:, 0].float().mean().item(),
        "text_R@10": hit_t[:, :10].any(1).float().mean().item(),
        "text_CE": ce / n_tok,
        "text_CE_shuffled_z": ce_shuf / n_tok,
        "text_ppl": float(torch.exp(torch.tensor(ce / n_tok))),
    }
    print(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" if v < 1 else f"{k}={v:.2f}" for k, v in res.items()), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-encoder", choices=["minilm", "lstm"], default="minilm")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pixel-weight", type=float, default=1.0)
    ap.add_argument("--latent-weight", type=float, default=1.0)
    ap.add_argument("--text-weight", type=float, default=1.0)
    ap.add_argument("--text-embed-weight", type=float, default=1.0, help="cosine loss on the predicted MiniLM vector of the next description")
    ap.add_argument("--word-dropout", type=float, default=0.3, help="share of teacher-forced tokens replaced by [UNK]")
    ap.add_argument("--ae-weights", default=str(OUT / "visual_ae.pt"))
    ap.add_argument("--freeze-image-encoder", action="store_true",
                    help="keep the pretrained image encoder fixed, so the latent target cannot drift")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = tokenizer()

    tr_all, te = load_split("train", DEVICE), load_split("test", DEVICE)
    # split train/val by story so no story leaks across the split
    n = len(tr_all["n_frames"])
    perm = torch.randperm(n, device=DEVICE)
    val_stories = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    val_stories[perm[: n // 5]] = True
    s_all, t_all = windows(tr_all)
    s_tr, t_tr = s_all[~val_stories[s_all]], t_all[~val_stories[s_all]]
    s_va, t_va = s_all[val_stories[s_all]], t_all[val_stories[s_all]]
    s_te, t_te = windows(te)
    print(f"windows: train {len(s_tr)}  val {len(s_va)}  test {len(s_te)}  device {DEVICE}  text encoder {args.text_encoder}")

    ae = VisualAutoencoder()
    if Path(args.ae_weights).exists():
        ae.load_state_dict(torch.load(args.ae_weights, map_location="cpu"))
        print(f"loaded pretrained autoencoder from {args.ae_weights}")
    else:
        print("WARNING: no pretrained autoencoder found, training the CNN from scratch (run v2/pretrain_visual.py first)")
    text_encoder = TextEncoderLSTM(tok.vocab_size, tok.pad_token_id) if args.text_encoder == "lstm" else None
    text_dim = 384 if text_encoder is None else text_encoder.out_dim
    model = SequencePredictor(ae, text_dim, tok.vocab_size, text_encoder, word_dropout=args.word_dropout).to(DEVICE)
    if args.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(s_tr) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(s_tr), device=DEVICE)
        sums = {"pixel": 0.0, "latent": 0.0, "text": 0.0, "emb": 0.0}
        for b in range(len(s_tr) // args.batch_size):
            j = order[b * args.batch_size:(b + 1) * args.batch_size]
            batch = gather(tr_all, s_tr[j], t_tr[j])
            img, logits, z, e_txt = model(batch["frames"], text_input(batch, args.text_encoder), batch["target_ids"][:, :-1])
            l_pix = F.l1_loss(img, batch["target"])
            l_lat = latent_loss(z, model.image_encoder(batch["target"]))
            l_txt = F.cross_entropy(logits.flatten(0, 1), batch["target_ids"][:, 1:].flatten(), ignore_index=tok.pad_token_id)
            l_emb = 1 - F.cosine_similarity(e_txt, batch["target_txt"], dim=-1).mean()
            loss = (args.pixel_weight * l_pix + args.latent_weight * l_lat + args.text_weight * l_txt
                    + args.text_embed_weight * l_emb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            sums["pixel"] += l_pix.item(); sums["latent"] += l_lat.item(); sums["text"] += l_txt.item(); sums["emb"] += l_emb.item()
        print(f"epoch {epoch + 1:2d}/{args.epochs} ({time.time() - t0:.0f}s)  train pixel {sums['pixel'] / (b + 1):.4f}  "
              f"latent {sums['latent'] / (b + 1):.4f}  text {sums['text'] / (b + 1):.3f}  text-emb {sums['emb'] / (b + 1):.4f}", flush=True)
        evaluate(model, tr_all, s_va, t_va, args.text_encoder, tok.pad_token_id, f"val e{epoch + 1}")

    res = evaluate(model, te, s_te, t_te, args.text_encoder, tok.pad_token_id, "TEST")
    name = args.text_encoder + args.tag
    torch.save(model.state_dict(), OUT / f"predictor_{name}.pt")

    # figure: inputs, target, prediction, with true and generated descriptions
    model.eval()
    pick = torch.randperm(len(s_te), device=DEVICE)[:5]
    batch = gather(te, s_te[pick], t_te[pick])
    with torch.no_grad():
        img, _, z, _ = model(batch["frames"], text_input(batch, args.text_encoder), batch["target_ids"][:, :-1])
        gen = model.text_decoder.generate(z, tok.cls_token_id, tok.sep_token_id)
    fig, ax = plt.subplots(10, K + 2, figsize=(2.6 * (K + 2), 12), gridspec_kw={"height_ratios": [2, 1.6] * 5})
    for r in range(5):
        tiles = [batch["frames"][r, i] for i in range(K)] + [batch["target"][r], img[r]]
        for c, tile in enumerate(tiles):
            ax[2 * r, c].imshow(tile.permute(1, 2, 0).clamp(0, 1).cpu()); ax[2 * r, c].axis("off")
            ax[2 * r + 1, c].axis("off")
        ax[2 * r, K].set_title("target", fontsize=9); ax[2 * r, K + 1].set_title("prediction", fontsize=9)
        true_txt = tok.decode(batch["target_ids"][r], skip_special_tokens=True)
        ax[2 * r + 1, K].text(0, 1, textwrap.fill(true_txt[:260], 34), fontsize=6.5, va="top")
        ax[2 * r + 1, K + 1].text(0, 1, textwrap.fill(tok.decode(gen[r])[:260], 34), fontsize=6.5, va="top")
    plt.tight_layout(); plt.savefig(OUT / f"predictions_{name}.png", dpi=110)
    print(f"figure: {OUT / f'predictions_{name}.png'}")
    print("TEST summary:", {k: round(v, 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
