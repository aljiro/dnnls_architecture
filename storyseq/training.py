"""The trainer shared by Levels 3-10. A level supplies a TrainConfig and a build_model() function;
the trainer does the rest: data, losses enabled by the configuration, validation with floors,
checkpoint selection on retrieval, test evaluation, loss / metric curves, figures for 3 seeds.

Usage from a level directory:
    from storyseq.training import TrainConfig, run
    run(CONFIG, build_model)            # CONFIG: TrainConfig, build_model: (tokenizer, components) -> model

Command line (every level's train.py accepts these):
    --epochs N --batch-size B --lr LR --seed S --tag NAME --max-windows N (debug) --no-plot
    --no-semantic (skip the CLIP / Frechet / sharpness table at the end) --device cpu|cuda
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from storyseq.components import PredictorConfig, SequencePredictor, TextEncoderLSTM, VisualAutoencoder, kl_divergence, latent_loss
from storyseq.data import K, gather, load_split, model_kwargs, tokenizer, windows

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "pretrained"


@dataclass
class TrainConfig:
    name: str                                   # e.g. "v4_attention"
    model: PredictorConfig = field(default_factory=PredictorConfig)
    out_dir: str = ""                           # default <name>/out, e.g. v4_attention/out
    # components
    text_encoder: str = "minilm"                # "minilm" (frozen, cached) or "lstm" (from scratch)
    ae_width: int = 1
    ae_weights: str = "pretrained/visual_ae.pt"
    text_lm_weights: str = ""                   # "" = none; e.g. "pretrained/text_lm.pt" (Level 5+)
    # optimisation
    epochs: int = 15
    batch_size: int = 32
    lr: float = 3e-4
    pretrained_lr_scale: float = 1.0            # image encoder / decoder learning-rate multiplier (0.1 at Level 5+)
    text_lm_lr_scale: float = 1.0               # pretrained language-model layers multiplier (0.3 at Level 5+)
    # loss weights (0 disables a term)
    pixel_weight: float = 1.0
    latent_weight: float = 1.0
    text_weight: float = 1.0
    text_embed_weight: float = 1.0
    recon_weight: float = 0.0                   # keep the autoencoder's reconstruction skill (Level 5+)
    char_weight: float = 1.0
    setting_weight: float = 1.0
    attn_weight: float = 0.0                    # supervise the mixture attention toward the closest input (Level 7)
    gate_weight: float = 0.0
    sup_threshold: float = 0.10
    kl_weight: float = 1e-2                     # variational (Level 8)
    kl_warmup: float = 3.0
    free_bits: float = 0.0
    clip_loss_weight: float = 0.0               # semantic loss through frozen CLIP (Level 10)
    clip_loss_mode: str = "contrastive"         # "contrastive": InfoNCE over the batch (relative; a generic template gains
                                                # nothing); "cosine": centred cosine to the target (absolute; exploitable)
    clip_augment: bool = False
    clip_temperature: float = 0.07
    # evaluation
    n_samples: int = 5                          # prior samples per window for variational models
    semantic: bool = True                       # CLIP similarity / Frechet / sharpness table at the end
    plot: bool = True
    seeds_for_figures: tuple = (0, 1, 2)
    max_windows: int = 0                        # debug: limit train / val / test windows
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tag: str = ""


# ----------------------------------------------------------------------------------------------
def build_predictor(cfg: TrainConfig, tok, components: dict | None = None) -> SequencePredictor:
    """Default model assembly used by every level's build_model(); components override defaults."""
    ae = VisualAutoencoder(width=cfg.ae_width)
    w = ROOT / cfg.ae_weights
    if w.exists():
        ae.load_state_dict(torch.load(w, map_location="cpu"))
    else:
        print(f"WARNING: no pretrained autoencoder at {w} (run: python -m storyseq.pretrain_visual)")
    text_encoder = TextEncoderLSTM(tok.vocab_size, tok.pad_token_id) if cfg.text_encoder == "lstm" else None
    model = SequencePredictor(ae, tok.vocab_size, cfg.model, text_encoder, components)
    if cfg.text_lm_weights and (ROOT / cfg.text_lm_weights).exists():
        model.text_decoder.load_language_model(ROOT / cfg.text_lm_weights)
    return model


def f1(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> float:
    tp = (pred & true & mask).sum().item(); fp = (pred & ~true & mask).sum().item(); fn = (~pred & true & mask).sum().item()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def run_model(model: SequencePredictor, batch: dict, train: bool = False, sample: bool = False) -> dict:
    kw = model_kwargs(model, batch)
    if model.variational:
        kw["sample"] = sample
        if train:
            kw["target_latent"] = model.target_latent(batch["target"])
    if model.cfg.text_memory and train:
        kw["names_present"] = batch["target_chars"]
    return model(batch["frames"], batch["txt"] if model.text_encoder is None else batch["ids"], batch["target_ids"][:, :-1], **kw)


_MEDIAN: dict = {}


def median_image(d: dict, s, t, key: str, device: str) -> torch.Tensor:
    if key not in _MEDIAN:
        _MEDIAN[key] = (d["pix"][s.cpu(), t.cpu()].float() / 255).median(0).values.to(device)
    return _MEDIAN[key]


@torch.no_grad()
def evaluate(model: SequencePredictor, d: dict, s, t, cfg: TrainConfig, pad_id: int, tag: str,
             clip_embed=None) -> dict[str, float]:
    """Streaming evaluation: per-window scalars are accumulated, images never are."""
    model.eval()
    dev = cfg.device
    if dev == "cuda":
        torch.cuda.empty_cache()
    median = median_image(d, s, t, f"{id(d)}:{tag.split()[0]}", dev)
    n = len(s)
    per = lambda a, b: (a - b).abs().mean(dim=(1, 2, 3))
    W: dict[str, list] = {}
    Z: dict[str, list] = {k: [] for k in ("z", "z_true", "e_pred", "e_true")}
    add = lambda k, v: W.setdefault(k, []).append(v)
    img_sum = torch.zeros(3, 60, 125, device=dev); img_sq = torch.zeros_like(img_sum)
    ce = ce_shuf = 0.0; n_tok = 0
    for b in range(0, n, 64):
        batch = gather(d, s[b:b + 64], t[b:b + 64])
        o = run_model(model, batch)
        target, img = batch["target"], o["image"]
        per_in = (batch["frames"] - target[:, None]).abs().mean(dim=(2, 3, 4))
        recon = model.image_decoder(model.image_encoder(target))
        if model.variational:
            samples = torch.stack([run_model(model, batch, sample=True)["image"] for _ in range(cfg.n_samples)])
            l1_s = (samples - target[None]).abs().mean(dim=(2, 3, 4))
            add("l1_best_of_k", l1_s.min(0).values); add("l1_sample_mean", l1_s.mean(0))
            add("diversity", (samples[:, None] - samples[None]).abs().mean(dim=(3, 4, 5)).sum((0, 1)) / max(1, cfg.n_samples * (cfg.n_samples - 1)))
            o_post = run_model(model, batch, train=True)
            add("l1_posterior", per(o_post["image"], target)); add("kl", kl_divergence(o_post).expand(len(target)))
            del samples, o_post
        add("l1", per(img, target)); add("l1_blob", per(median.expand_as(target), target))
        add("l1_copy", per(batch["frames"][:, -1], target)); add("l1_recon", per(recon, target))
        add("copy_best", per_in.min(1).values); add("best_in", per_in.argmin(1))
        add("alpha", o["alpha"]); add("gate", o["gate"])
        if "copy_gate" in o:
            add("copy_gate", o["copy_gate"].mean(dim=(1, 2, 3)))
        if clip_embed is not None:
            mu = batch["target_clip"].mean(0, keepdim=True)
            add("clip_sim", F.cosine_similarity(clip_embed(img) - mu, batch["target_clip"] - mu, dim=-1))
        img_sum += img.sum(0); img_sq += (img ** 2).sum(0)
        Z["z"].append(o["z"]); Z["z_true"].append(model.target_latent(target)); Z["e_pred"].append(o["e_txt"]); Z["e_true"].append(batch["target_txt"])
        tgt = batch["target_ids"][:, 1:]
        ce += F.cross_entropy(o["logits"].flatten(0, 1), tgt.flatten(), ignore_index=pad_id, reduction="sum").item()
        perm = torch.randperm(len(tgt), device=dev)
        logits_shuf = model.text_decoder(batch["target_ids"][:, :-1], o["cond"][perm], o.get("memory", None) if "memory" not in o else o["memory"][perm],
                                         o["memory_mask"][perm] if "memory_mask" in o else None)
        ce_shuf += F.cross_entropy(logits_shuf.flatten(0, 1), tgt.flatten(), ignore_index=pad_id, reduction="sum").item()
        n_tok += (tgt != pad_id).sum().item()
        if model.annotated:
            add("char_logits", o["char_logits"]); add("target_chars", batch["target_chars"]); add("chars_in", batch["chars_in"]); add("n_chars", batch["n_chars"])
        del o, img, recon, batch
    a = {k: torch.cat(v) for k, v in {**W, **Z}.items() if v}
    ar = torch.arange(n, device=dev)
    centred = lambda x, ref: F.normalize(x - ref.mean(0, keepdim=True), dim=-1)
    hit_z = (centred(a["z"], a["z_true"]) @ centred(a["z_true"], a["z_true"]).T).argsort(1, descending=True) == ar[:, None]
    hit_t = (centred(a["e_pred"], a["e_true"]) @ centred(a["e_true"], a["e_true"]).T).argsort(1, descending=True) == ar[:, None]
    zn = centred(a["z"][:512], a["z_true"]); cos = zn @ zn.T
    near = a["copy_best"] < 0.06
    mean_img = img_sum / n
    res = {
        "img_L1": a["l1"].mean().item(), "recon_L1": a["l1_recon"].mean().item(),
        "img_L1_near_copy": a["l1"][near].mean().item() if near.any() else float("nan"),
        "floor_median": a["l1_blob"].mean().item(), "floor_copy": a["l1_copy"].mean().item(),
        "pred_spread": (img_sq / n - mean_img ** 2).clamp(min=0).sqrt().mean().item(),
        "latent_cos": ((cos.sum() - cos.diag().sum()) / (len(zn) * (len(zn) - 1))).item(),
        "alpha_on_best_input": a["alpha"].gather(1, a["best_in"][:, None]).mean().item(),
        "alpha_argmax_acc_near": (a["alpha"].argmax(1) == a["best_in"])[near].float().mean().item() if near.any() else float("nan"),
        "gate_near_copy": a["gate"][near].mean().item() if near.any() else float("nan"), "gate_cut": a["gate"][~near].mean().item(),
        "latent_R@10": hit_z[:, :10].any(1).float().mean().item(), "text_R@10": hit_t[:, :10].any(1).float().mean().item(),
        "text_CE": ce / n_tok, "text_CE_shuffled": ce_shuf / n_tok,
    }
    for k in ("clip_sim",):
        if k in a:
            res[k] = a[k].mean().item()
    if "l1_best_of_k" in a:
        res.update({"img_L1_best_of_k": a["l1_best_of_k"].mean().item(), "img_L1_sample_mean": a["l1_sample_mean"].mean().item(),
                    "sample_diversity": a["diversity"].mean().item(), "img_L1_posterior": a["l1_posterior"].mean().item(), "kl": a["kl"].mean().item()})
    if "copy_gate" in a:
        res["copy_gate_near"] = a["copy_gate"][near].mean().item() if near.any() else float("nan"); res["copy_gate_cut"] = a["copy_gate"][~near].mean().item()
    if model.annotated:
        mask = torch.arange(a["target_chars"].size(1), device=dev)[None] < a["n_chars"][:, None]
        true = a["target_chars"]
        res.update({"char_F1@0.3": f1(torch.sigmoid(a["char_logits"]) > 0.3, true, mask),
                    "char_F1_same_as_frame4": f1(a["chars_in"][:, -1], true, mask), "char_F1_all_seen": f1(a["chars_in"].any(1), true, mask),
                    "char_F1_in_2plus": f1(a["chars_in"].sum(1) >= 2, true, mask)})
    print(f"[{tag}] " + "  ".join(f"{k}={v:.4f}" if abs(v) < 1 else f"{k}={v:.2f}" for k, v in res.items()), flush=True)
    return res


# ----------------------------------------------------------------------------------------------
def parse_overrides(cfg: TrainConfig) -> TrainConfig:
    ap = argparse.ArgumentParser(description=f"train {cfg.name}")
    ap.add_argument("--epochs", type=int); ap.add_argument("--batch-size", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int); ap.add_argument("--tag"); ap.add_argument("--max-windows", type=int)
    ap.add_argument("--device"); ap.add_argument("--no-plot", action="store_true"); ap.add_argument("--no-semantic", action="store_true")
    ap.add_argument("--n-samples", type=int)
    a = ap.parse_args()
    for k in ("epochs", "batch_size", "lr", "seed", "tag", "max_windows", "device", "n_samples"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    if a.no_plot:
        cfg.plot = False
    if a.no_semantic:
        cfg.semantic = False
    return cfg


def plot_curves(history: list[dict], path: Path, name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    loss_keys = sorted({k for h in history for k in h["train"]})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k in loss_keys:
        ax[0].plot(ep, [h["train"].get(k, float("nan")) for h in history], label=k)
    ax[0].set_title("training losses"); ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=7); ax[0].set_yscale("log")
    v = lambda k: [h["val"].get(k, float("nan")) for h in history]
    ax[1].plot(ep, v("img_L1"), label="image L1 (prior mean)")
    if "img_L1_best_of_k" in history[-1]["val"]:
        ax[1].plot(ep, v("img_L1_best_of_k"), label="best of K samples")
    ax[1].axhline(history[-1]["val"]["floor_median"], ls="--", c="gray", label="floor: median image")
    ax[1].axhline(history[-1]["val"]["floor_copy"], ls=":", c="gray", label="floor: copy last")
    ax[1].plot(ep, v("recon_L1"), label="reconstruction (AE)")
    ax[1].set_title("validation image L1"); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=7)
    ax[2].plot(ep, v("text_R@10"), label="text retrieval top-10"); ax[2].plot(ep, v("latent_R@10"), label="image-latent retrieval top-10")
    for k, lab in (("clip_sim", "CLIP similarity"), ("char_F1@0.3", "character F1")):
        if k in history[-1]["val"]:
            ax[2].plot(ep, v(k), label=lab)
    ax[2].set_title("validation retrieval / semantics"); ax[2].set_xlabel("epoch"); ax[2].legend(fontsize=7)
    fig.suptitle(name); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"curves: {path}")


def run(cfg: TrainConfig, build_model, overrides: bool = True) -> dict:
    if overrides:
        cfg = parse_overrides(cfg)
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    out = Path(cfg.out_dir) if cfg.out_dir else ROOT / cfg.name / "out"
    out.mkdir(parents=True, exist_ok=True)
    name = cfg.name + cfg.tag
    tok = tokenizer()
    annot = cfg.model.annotations
    tr_all, te = load_split("train", dev, annot), load_split("test", dev, annot)
    n = len(tr_all["n_frames"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)    # same story split for every level
    val_stories = torch.zeros(n, dtype=torch.bool, device=dev); val_stories[perm[: n // 5]] = True
    s_all, t_all = windows(tr_all)
    s_tr, t_tr = s_all[~val_stories[s_all]], t_all[~val_stories[s_all]]
    s_va, t_va = s_all[val_stories[s_all]], t_all[val_stories[s_all]]
    s_te, t_te = windows(te)
    if cfg.max_windows:
        s_tr, t_tr, s_va, t_va, s_te, t_te = (x[:cfg.max_windows] for x in (s_tr, t_tr, s_va, t_va, s_te, t_te))
    print(f"{name}: windows train {len(s_tr)} val {len(s_va)} test {len(s_te)}  device {dev}")

    model = build_model(tok).to(dev)
    clip_embed = None
    if cfg.clip_loss_weight > 0 or cfg.semantic:
        from storyseq.metrics import ClipEmbed
        clip_embed = ClipEmbed().to(dev)
    image_params = [p for m in (model.image_encoder, model.image_decoder) for p in m.parameters()]
    lm_params = [p for k, p in model.text_decoder.named_parameters() if k.split(".")[0] in ("embedding", "lstm", "out")]
    seen = {id(p) for p in image_params + lm_params}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups = [{"params": other, "lr": cfg.lr}, {"params": image_params, "lr": cfg.lr * cfg.pretrained_lr_scale},
              {"params": lm_params, "lr": cfg.lr * (cfg.text_lm_lr_scale if cfg.text_lm_weights else 1.0)}]
    params = other + image_params + lm_params
    print(f"trainable parameters: {sum(p.numel() for p in params):,}")
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(s_tr) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)

    history, best, best_state, best_epoch, t0 = [], -1.0, None, 0, time.time()
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(s_tr), device=dev)
        sums: dict[str, float] = {}
        for b in range(steps_per_epoch):
            j = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            batch = gather(tr_all, s_tr[j], t_tr[j])
            o = run_model(model, batch, train=True)
            L = {"pixel": cfg.pixel_weight * F.l1_loss(o["image"], batch["target"]),
                 "latent": cfg.latent_weight * latent_loss(o["z"], model.target_latent(batch["target"])),
                 "text": cfg.text_weight * F.cross_entropy(o["logits"].flatten(0, 1), batch["target_ids"][:, 1:].flatten(), ignore_index=tok.pad_token_id),
                 "text_embed": cfg.text_embed_weight * latent_loss(o["e_txt"], batch["target_txt"])}
            if cfg.recon_weight > 0:
                L["recon"] = cfg.recon_weight * F.l1_loss(model.image_decoder(model.image_encoder(batch["target"])), batch["target"])
            if cfg.attn_weight > 0 or cfg.gate_weight > 0:
                per_in = (batch["frames"] - batch["target"][:, None]).abs().mean(dim=(2, 3, 4))
                close = per_in.min(1).values < cfg.sup_threshold
                if cfg.attn_weight > 0 and close.any():
                    L["attn"] = cfg.attn_weight * F.nll_loss(torch.log(o["alpha"][close] + 1e-6), per_in.argmin(1)[close])
                if cfg.gate_weight > 0:
                    L["gate"] = cfg.gate_weight * F.binary_cross_entropy(o["gate"].clamp(1e-6, 1 - 1e-6), close.float())
            if model.variational:
                warm = min(1.0, (epoch + b / steps_per_epoch) / max(cfg.kl_warmup, 1e-6))
                L["kl"] = cfg.kl_weight * warm * kl_divergence(o, cfg.free_bits)
            if cfg.clip_loss_weight > 0:
                pred_clip = clip_embed(o["image"], augment=cfg.clip_augment)
                if cfg.clip_loss_mode == "contrastive":
                    mu = batch["target_clip"].mean(0, keepdim=True)
                    logits = F.normalize(pred_clip - mu, dim=-1) @ F.normalize(batch["target_clip"] - mu, dim=-1).T / cfg.clip_temperature
                    L["clip"] = cfg.clip_loss_weight * F.cross_entropy(logits, torch.arange(len(logits), device=dev))
                else:
                    L["clip"] = cfg.clip_loss_weight * latent_loss(pred_clip, batch["target_clip"])
            if model.annotated:
                mask = (torch.arange(o["char_logits"].size(1), device=dev)[None] < batch["n_chars"][:, None]).float()
                bce = F.binary_cross_entropy_with_logits(o["char_logits"], batch["target_chars"].float(), reduction="none")
                L["chars"] = cfg.char_weight * (bce * mask).sum() / mask.sum().clamp(min=1)
                has = batch["target_set"].abs().sum(-1) > 0
                if has.any():
                    L["setting"] = cfg.setting_weight * latent_loss(o["setting"][has], batch["target_set"][has])
            loss = sum(L.values())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            for k, v in L.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        train_losses = {k: v / steps_per_epoch for k, v in sums.items()}
        print(f"epoch {epoch + 1:2d}/{cfg.epochs} ({time.time() - t0:.0f}s)  " + "  ".join(f"{k} {v:.4f}" for k, v in train_losses.items()), flush=True)
        val = evaluate(model, tr_all, s_va, t_va, cfg, tok.pad_token_id, f"val e{epoch + 1}", clip_embed if cfg.clip_loss_weight > 0 else None)
        history.append({"epoch": epoch + 1, "train": train_losses, "val": val})
        score = val["text_R@10"] + val["latent_R@10"]
        if score > best:
            best, best_epoch = score, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if cfg.plot:
            plot_curves(history, out / f"curves_{name}.png", name)
        (out / f"history_{name}.json").write_text(json.dumps(history, indent=1))

    torch.save(model.state_dict(), out / f"predictor_{name}_last.pt")
    torch.save(best_state, out / f"predictor_{name}.pt")
    model.load_state_dict(best_state)
    res = evaluate(model, te, s_te, t_te, cfg, tok.pad_token_id, f"TEST (best validation retrieval, epoch {best_epoch})",
                   clip_embed if cfg.clip_loss_weight > 0 else None)
    summary = {"config": {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in asdict(cfg).items()},
               "best_epoch": best_epoch, "test": res}
    if cfg.semantic:
        from storyseq.metrics import print_table, semantic_table
        table = semantic_table(model, te, s_te, t_te, n_samples=2, max_windows=cfg.max_windows)
        print_table(table)
        summary["semantic"] = table
    (out / f"summary_{name}.json").write_text(json.dumps(summary, indent=1))
    from storyseq.visualize import render_seeds
    render_seeds(model, te, tok, out, name, seeds=cfg.seeds_for_figures, train=tr_all)
    return summary
