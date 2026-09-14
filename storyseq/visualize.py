"""Figures: for a handful of test windows, the four inputs with their descriptions, the target
with its description, the prediction with the generated description, and, depending on the
model, two prior samples (variational levels) and the nearest training frame to the predicted
latent (retrieval). Three seeds by default, so a reader sees twelve windows rather than four.

From a level directory:  python visualize.py [--seeds 0 1 2] [--rows 4] [--greedy]
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from storyseq.data import K, gather, load_split, model_kwargs, tokenizer, windows

ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def make_figure(model, d: dict, tok, path: Path, rows: int = 4, seed: int = 0, max_chars: int = 230,
                sample_text: bool = True, train: dict | None = None) -> None:
    model.eval()
    dev = d["txt"].device
    s, t = windows(d)
    pick = torch.randperm(len(s), generator=torch.Generator().manual_seed(seed))[:rows].to(dev)
    batch = gather(d, s[pick], t[pick])
    text = batch["txt"] if model.text_encoder is None else batch["ids"]
    kw = model_kwargs(model, batch)
    o = model(batch["frames"], text, batch["target_ids"][:, :-1], **kw)
    gen = model.text_decoder.generate(o["cond"], tok.cls_token_id, tok.sep_token_id, max_len=70, sample=sample_text,
                                      memory=o.get("memory"), memory_mask=o.get("memory_mask"))
    extra_tiles: list[list] = [[] for _ in range(rows)]
    extra_names: list[str] = []
    if model.variational:
        for i in range(2):
            si = model(batch["frames"], text, batch["target_ids"][:, :-1], sample=True, **kw)["image"]
            for r in range(rows):
                extra_tiles[r].append(si[r])
            extra_names.append(f"prior sample {i + 1}")
    if train is not None:
        n = train["n_frames"]
        valid = torch.nonzero(torch.arange(train["pix"].shape[1], device=n.device)[None] < n[:, None]).cpu()
        bank = torch.cat([model.target_latent(train["pix"][valid[i:i + 512, 0], valid[i:i + 512, 1]].to(dev).float() / 255)
                          for i in range(0, len(valid), 512)])
        mu = bank.mean(0, keepdim=True)
        nn_idx = (F.normalize(o["z"] - mu, dim=-1) @ F.normalize(bank - mu, dim=-1).T).argmax(1).cpu()
        for r in range(rows):
            extra_tiles[r].append(train["pix"][valid[nn_idx[r], 0], valid[nn_idx[r], 1]].float() / 255)
        extra_names.append("retrieved (train)")

    cols = K + 2 + len(extra_names)
    fig, ax = plt.subplots(2 * rows, cols, figsize=(3.4 * cols, 3.0 * rows),
                           gridspec_kw={"height_ratios": [1.0, 0.75] * rows, "wspace": 0.03, "hspace": 0.06})
    for r in range(rows):
        tiles = [batch["frames"][r, i] for i in range(K)] + [batch["target"][r], o["image"][r]] + extra_tiles[r]
        texts = ([tok.decode(batch["ids"][r, i], skip_special_tokens=True) for i in range(K)]
                 + [tok.decode(batch["target_ids"][r], skip_special_tokens=True), tok.decode(gen[r])] + [""] * len(extra_names))
        for c in range(cols):
            a = ax[2 * r, c]
            a.imshow(tiles[c].permute(1, 2, 0).clamp(0, 1).cpu(), aspect="auto")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            name = (["input 1", "input 2", "input 3", "input 4", "target", "prediction"] + extra_names)[c] if r == 0 else ""
            if c < K:
                name = f"{name}   attention {o['alpha'][r, c]:.2f}".strip()
            if c == K + 1 and o["gate"].abs().sum() > 0:
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


def render_seeds(model, d: dict, tok, out: Path, name: str, seeds=(0, 1, 2), rows: int = 4,
                 sample_text: bool = True, train: dict | None = None) -> None:
    for seed in seeds:
        make_figure(model, d, tok, out / f"predictions_{name}_seed{seed}.png", rows, seed, sample_text=sample_text, train=train)


def run(cfg, build_model) -> None:
    """Entry point for a level's visualize.py: loads the level's checkpoint and renders the figures."""
    ap = argparse.ArgumentParser(description=f"visualise {cfg.name}")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cfg.seeds_for_figures))
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--greedy", action="store_true", help="greedy decoding instead of nucleus sampling for the text")
    ap.add_argument("--tag", default=cfg.tag)
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--device", default=cfg.device)
    a = ap.parse_args()
    out = Path(cfg.out_dir) if cfg.out_dir else ROOT / cfg.name / "out"
    name = cfg.name + a.tag
    tok = tokenizer()
    te = load_split("test", a.device, cfg.model.annotations)
    tr = None if a.no_retrieval else load_split("train", a.device, annotations=False)
    model = build_model(tok).to(a.device)
    model.load_state_dict(torch.load(out / f"predictor_{name}.pt", map_location=a.device))
    render_seeds(model, te, tok, out, name, seeds=a.seeds, rows=a.rows, sample_text=not a.greedy, train=tr)
