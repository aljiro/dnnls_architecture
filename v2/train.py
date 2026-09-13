"""Train the v2 sequence predictor: 4 (frame, description) pairs -> next frame image and description.

Pipeline (see v2/models.py):
  image encoder (pretrained v2 autoencoder) + text encoder (frozen MiniLM vectors, or --text-encoder lstm)
  -> fuse -> GRU -> attention -> latents
  -> image decoder                               pixel L1 to the true next frame
  -> latent target                               1 - cos(z, encoder(true next frame))
  -> text decoder                                cross-entropy with teacher forcing (word dropout)
  -> text-embedding head                         1 - cos to the MiniLM vector of the next description
  stage C adds: which characters appear next (BCE over story slots), next setting embedding (cosine)

--stage 0|A|B|C selects the model variant (cumulative; see models.py and ASSESSMENT.md section 9).

Logged per epoch on the validation split (20 % of train stories, split by story) and finally on test:
  image L1 vs constant-median and copy-last floors; prediction spread; latent cosine between windows
  (the constant-component problem); attention weight on the input closest to the target; gate on
  near-copy vs cut windows; image-latent and text-embedding retrieval; text CE with the true vs a
  shuffled condition; stage C: character F1 vs "same as frame 4" and "all seen so far" baselines.

Run after v2/pretrain_visual.py (and v2/precompute_annotations.py for stage C):
  python v2/train.py --stage A
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
from v2.data import K, gather, load_split, tokenizer, windows  # noqa: E402
from v2.models import SequencePredictor, TextEncoderLSTM, VisualAutoencoder, latent_loss  # noqa: E402
from v2.visualize import make_figure  # noqa: E402

OUT = ROOT / "v2" / "out"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def text_input(batch: dict, mode: str) -> torch.Tensor:
    return batch["txt"] if mode == "minilm" else batch["ids"]


def run_model(model: SequencePredictor, batch: dict, mode: str) -> dict:
    extra = {k: batch[k] for k in ("set_emb", "ent_pix", "ent_slot") if model.stage == "C"}
    return model(batch["frames"], text_input(batch, mode), batch["target_ids"][:, :-1], **extra)


def f1(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> float:
    """micro F1 over valid character slots."""
    tp = (pred & true & mask).sum().item()
    fp = (pred & ~true & mask).sum().item()
    fn = (~pred & true & mask).sum().item()
    return 2 * tp / max(1, 2 * tp + fp + fn)


@torch.no_grad()
def evaluate(model: SequencePredictor, d: dict, s: torch.Tensor, t: torch.Tensor, mode: str,
             pad_id: int, tag: str) -> dict[str, float]:
    model.eval()
    acc: dict[str, list] = {k: [] for k in ("img", "z", "z_true", "last", "e_pred", "e_true", "alpha", "gate", "recon",
                                             "best_in", "copy_best", "char_logits", "target_chars", "chars_in", "n_chars")}
    ce = ce_shuf = 0.0
    n_tok = 0
    for b in range(0, len(s), 64):
        batch = gather(d, s[b:b + 64], t[b:b + 64])
        o = run_model(model, batch, mode)
        per_in = (batch["frames"] - batch["target"][:, None]).abs().mean(dim=(2, 3, 4))   # [B, K] L1 of each input
        acc["img"].append(o["image"]); acc["z"].append(o["z"]); acc["z_true"].append(model.target_latent(batch["target"]))
        acc["last"].append(batch["frames"][:, -1]); acc["e_pred"].append(o["e_txt"]); acc["e_true"].append(batch["target_txt"])
        acc["recon"].append(model.image_decoder(model.image_encoder(batch["target"])))   # drift of the fine-tuned autoencoder
        acc["alpha"].append(o["alpha"]); acc["gate"].append(o["gate"])
        acc["best_in"].append(per_in.argmin(1)); acc["copy_best"].append(per_in.min(1).values)
        tgt = batch["target_ids"][:, 1:]
        ce += F.cross_entropy(o["logits"].flatten(0, 1), tgt.flatten(), ignore_index=pad_id, reduction="sum").item()
        perm = torch.randperm(len(tgt), device=DEVICE)
        logits_shuf = model.text_decoder(batch["target_ids"][:, :-1], o["cond"][perm])
        ce_shuf += F.cross_entropy(logits_shuf.flatten(0, 1), tgt.flatten(), ignore_index=pad_id, reduction="sum").item()
        n_tok += (tgt != pad_id).sum().item()
        if model.stage == "C":
            acc["char_logits"].append(o["char_logits"]); acc["target_chars"].append(batch["target_chars"])
            acc["chars_in"].append(batch["chars_in"]); acc["n_chars"].append(batch["n_chars"])
    a = {k: torch.cat(v) for k, v in acc.items() if v}
    target = d["pix"][s, t].float() / 255
    median = target.median(0).values
    ar = torch.arange(len(s), device=DEVICE)
    # retrieval and cosine on centred vectors (subtract the mean of the true targets), see centred_cosine_loss
    def centred(x, ref):
        return F.normalize(x - ref.mean(0, keepdim=True), dim=-1)
    hit_z = (centred(a["z"], a["z_true"]) @ centred(a["z_true"], a["z_true"]).T).argsort(1, descending=True) == ar[:, None]
    hit_t = (centred(a["e_pred"], a["e_true"]) @ centred(a["e_true"], a["e_true"]).T).argsort(1, descending=True) == ar[:, None]
    zn = centred(a["z"][:512], a["z_true"])
    cos = (zn @ zn.T)
    near = a["copy_best"] < 0.06
    res = {
        "img_L1": F.l1_loss(a["img"], target).item(),
        "recon_L1": F.l1_loss(a["recon"], target).item(),          # pretrained autoencoder: 0.040
        "img_L1_near_copy": F.l1_loss(a["img"][near], target[near]).item() if near.any() else float("nan"),
        "floor_median": F.l1_loss(median.expand_as(target), target).item(),
        "floor_copy": F.l1_loss(a["last"], target).item(),
        "pred_spread": a["img"].std(0).mean().item(),
        "latent_cos": ((cos.sum() - cos.diag().sum()) / (len(zn) * (len(zn) - 1))).item(),
        "alpha_on_best_input": a["alpha"].gather(1, a["best_in"][:, None]).mean().item(),
        "alpha_argmax_acc_near": (a["alpha"].argmax(1) == a["best_in"])[near].float().mean().item() if near.any() else float("nan"),
        "gate_near_copy": a["gate"][near].mean().item() if near.any() else float("nan"),
        "gate_cut": a["gate"][~near].mean().item(),
        "latent_R@10": hit_z[:, :10].any(1).float().mean().item(),
        "text_R@10": hit_t[:, :10].any(1).float().mean().item(),
        "text_CE": ce / n_tok,
        "text_CE_shuffled": ce_shuf / n_tok,
    }
    if model.stage == "C":
        mask = torch.arange(a["target_chars"].size(1), device=DEVICE)[None] < a["n_chars"][:, None]
        true = a["target_chars"]
        res["char_F1"] = f1(a["char_logits"] > 0, true, mask)
        res["char_F1_same_as_frame4"] = f1(a["chars_in"][:, -1], true, mask)
        res["char_F1_all_seen"] = f1(a["chars_in"].any(1), true, mask)
    print(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" if abs(v) < 1 else f"{k}={v:.2f}" for k, v in res.items()), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["0", "A", "B", "C"], default="A")
    ap.add_argument("--text-encoder", choices=["minilm", "lstm"], default="minilm")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pixel-weight", type=float, default=1.0)
    ap.add_argument("--latent-weight", type=float, default=1.0)
    ap.add_argument("--text-weight", type=float, default=1.0)
    ap.add_argument("--text-embed-weight", type=float, default=1.0)
    ap.add_argument("--char-weight", type=float, default=1.0)
    ap.add_argument("--setting-weight", type=float, default=1.0)
    ap.add_argument("--word-dropout", type=float, default=0.3)
    ap.add_argument("--attn-weight", type=float, default=0.0,
                    help="supervise the attention toward the input closest to the target (known at training time)")
    ap.add_argument("--gate-weight", type=float, default=0.0,
                    help="supervise the gate: 1 when some input is within --sup-threshold of the target, else 0")
    ap.add_argument("--sup-threshold", type=float, default=0.10,
                    help="windows whose closest input has L1 below this get attention supervision")
    ap.add_argument("--ae-weights", default=str(OUT / "visual_ae.pt"))
    ap.add_argument("--text-lm-weights", default=str(OUT / "text_lm.pt"),
                    help="unconditional language-model weights for the text decoder (v2/pretrain_text.py); '' to skip")
    ap.add_argument("--pretrained-lr-scale", type=float, default=0.1,
                    help="learning-rate multiplier for the pretrained image encoder/decoder (discriminative LR)")
    ap.add_argument("--text-lm-lr-scale", type=float, default=0.3,
                    help="learning-rate multiplier for the pretrained text decoder layers")
    ap.add_argument("--freeze-image-encoder", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = tokenizer()
    annot = args.stage == "C"

    tr_all, te = load_split("train", DEVICE, annot), load_split("test", DEVICE, annot)
    n = len(tr_all["n_frames"])
    perm = torch.randperm(n, device=DEVICE)                     # same split for every stage (seed 0)
    val_stories = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    val_stories[perm[: n // 5]] = True
    s_all, t_all = windows(tr_all)
    s_tr, t_tr = s_all[~val_stories[s_all]], t_all[~val_stories[s_all]]
    s_va, t_va = s_all[val_stories[s_all]], t_all[val_stories[s_all]]
    s_te, t_te = windows(te)
    print(f"stage {args.stage}  windows: train {len(s_tr)}  val {len(s_va)}  test {len(s_te)}  device {DEVICE}  text encoder {args.text_encoder}")

    ae = VisualAutoencoder()
    if Path(args.ae_weights).exists():
        ae.load_state_dict(torch.load(args.ae_weights, map_location="cpu"))
        print(f"loaded pretrained autoencoder from {args.ae_weights}")
    else:
        print("WARNING: no pretrained autoencoder found (run v2/pretrain_visual.py first)")
    text_encoder = TextEncoderLSTM(tok.vocab_size, tok.pad_token_id) if args.text_encoder == "lstm" else None
    text_dim = 384 if text_encoder is None else text_encoder.out_dim
    model = SequencePredictor(ae, text_dim, tok.vocab_size, text_encoder, word_dropout=args.word_dropout, stage=args.stage).to(DEVICE)
    text_lm = bool(args.text_lm_weights) and Path(args.text_lm_weights).exists()
    if text_lm:
        model.text_decoder.load_language_model(args.text_lm_weights)
        print(f"loaded pretrained language model into the text decoder from {args.text_lm_weights}")
    if args.freeze_image_encoder:
        for p in model.image_encoder.parameters():
            p.requires_grad = False
    # discriminative learning rates: pretrained parts move slowly, new parts at the full rate
    image_params = [p for m in (model.image_encoder, model.image_decoder) for p in m.parameters() if p.requires_grad]
    lm_params = [p for n, p in model.text_decoder.named_parameters() if n.split(".")[0] in ("embedding", "lstm", "out")]
    seen = {id(p) for p in image_params + lm_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups = [{"params": other, "lr": args.lr},
              {"params": image_params, "lr": args.lr * args.pretrained_lr_scale},
              {"params": lm_params, "lr": args.lr * (args.text_lm_lr_scale if text_lm else 1.0)}]
    params = other + image_params + lm_params
    print(f"trainable params: {sum(p.numel() for p in params):,}  (image enc/dec at {args.pretrained_lr_scale}x, "
          f"text LM layers at {args.text_lm_lr_scale if text_lm else 1.0}x)")
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(s_tr) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=steps, pct_start=0.1)

    t0 = time.time()
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(s_tr), device=DEVICE)
        sums: dict[str, float] = {}
        for b in range(len(s_tr) // args.batch_size):
            j = order[b * args.batch_size:(b + 1) * args.batch_size]
            batch = gather(tr_all, s_tr[j], t_tr[j])
            o = run_model(model, batch, args.text_encoder)
            losses = {
                "pixel": args.pixel_weight * F.l1_loss(o["image"], batch["target"]),
                "latent": args.latent_weight * latent_loss(o["z"], model.target_latent(batch["target"])),
                "text": args.text_weight * F.cross_entropy(o["logits"].flatten(0, 1), batch["target_ids"][:, 1:].flatten(),
                                                           ignore_index=tok.pad_token_id),
                "emb": args.text_embed_weight * latent_loss(o["e_txt"], batch["target_txt"]),
            }
            if args.attn_weight > 0 or args.gate_weight > 0:
                per_in = (batch["frames"] - batch["target"][:, None]).abs().mean(dim=(2, 3, 4))   # [B, K]
                close = per_in.min(1).values < args.sup_threshold
                if args.attn_weight > 0 and close.any():
                    losses["attn"] = args.attn_weight * F.nll_loss(torch.log(o["alpha"][close] + 1e-6), per_in.argmin(1)[close])
                if args.gate_weight > 0:
                    losses["gate"] = args.gate_weight * F.binary_cross_entropy(o["gate"].clamp(1e-6, 1 - 1e-6), close.float())
            if args.stage == "C":
                mask = (torch.arange(o["char_logits"].size(1), device=DEVICE)[None] < batch["n_chars"][:, None]).float()
                bce = F.binary_cross_entropy_with_logits(o["char_logits"], batch["target_chars"].float(), reduction="none")
                losses["chars"] = args.char_weight * (bce * mask).sum() / mask.sum().clamp(min=1)
                has_set = batch["target_set"].abs().sum(-1) > 0
                losses["setting"] = args.setting_weight * latent_loss(o["setting"][has_set], batch["target_set"][has_set])
            loss = sum(losses.values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        print(f"epoch {epoch + 1:2d}/{args.epochs} ({time.time() - t0:.0f}s)  train " +
              "  ".join(f"{k} {v / (b + 1):.4f}" for k, v in sums.items()), flush=True)
        val = evaluate(model, tr_all, s_va, t_va, args.text_encoder, tok.pad_token_id, f"val e{epoch + 1}")
        # checkpoint selection on retrieval (text + image latent), not on image L1: the pixel head
        # overfits from epoch 1 while the other heads are still untrained (ASSESSMENT.md 10c)
        score = -(val["text_R@10"] + val["latent_R@10"])
        if score < best_val:
            best_val, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    name = f"stage{args.stage}_{args.text_encoder}{args.tag}"
    res_last = evaluate(model, te, s_te, t_te, args.text_encoder, tok.pad_token_id, "TEST last epoch")
    torch.save(model.state_dict(), OUT / f"predictor_{name}_last.pt")
    model.load_state_dict(best_state)
    res = evaluate(model, te, s_te, t_te, args.text_encoder, tok.pad_token_id, f"TEST best val retrieval (epoch {best_epoch})")
    torch.save(model.state_dict(), OUT / f"predictor_{name}.pt")
    make_figure(model, te, tok, args.text_encoder, OUT / f"predictions_{name}.png", sample=args.stage in ("B", "C"))
    print("TEST summary:", {k: round(v, 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
