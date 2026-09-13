"""How much of the next frame is predictable in pixel space, and how much of that does v2 get?

Per test window, compare four L1 scores against the true next frame:
  blob     : the constant median image (what an input-blind model converges to)
  copy     : the 4th input frame (perfect when the shot continues, poor when it cuts)
  oracle   : min(blob, copy) per window, i.e. a model that knew whether the shot continues
  model    : the trained v2 predictor (--text-encoder minilm by default)
  recon    : the autoencoder given the true target (what the decoder can draw with a perfect latent)
Then split the windows by whether the shot continues (copy L1 < threshold) and report each group.

Run: python v2/analyze_predictability.py [--text-encoder minilm] [--tag ""]
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
    ap.add_argument("--tag", default="")
    ap.add_argument("--same-shot-threshold", type=float, default=0.06)
    args = ap.parse_args()
    tok = tokenizer()
    te = load_split("test", DEVICE)
    s, t = windows(te)
    ae = VisualAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(OUT / "visual_ae.pt", map_location=DEVICE)); ae.eval()
    text_encoder = TextEncoderLSTM(tok.vocab_size, tok.pad_token_id) if args.text_encoder == "lstm" else None
    model = SequencePredictor(VisualAutoencoder(), 384 if text_encoder is None else text_encoder.out_dim,
                              tok.vocab_size, text_encoder).to(DEVICE)
    model.load_state_dict(torch.load(OUT / f"predictor_{args.text_encoder}{args.tag}.pt", map_location=DEVICE)); model.eval()

    target = te["pix"][s, t].float() / 255
    median = target.median(0).values
    per = lambda a, b: (a - b).abs().mean(dim=(1, 2, 3))          # per-window L1
    blob = per(median.expand_as(target), target)
    copy, mdl, recon = [], [], []
    for b in range(0, len(s), 64):
        batch = gather(te, s[b:b + 64], t[b:b + 64])
        text = batch["txt"] if args.text_encoder == "minilm" else batch["ids"]
        img, *_ = model(batch["frames"], text, batch["target_ids"][:, :-1])
        copy.append(per(batch["frames"][:, -1], batch["target"]))
        mdl.append(per(img, batch["target"]))
        recon.append(per(ae(batch["target"]), batch["target"]))
    copy, mdl, recon = map(torch.cat, (copy, mdl, recon))
    oracle = torch.minimum(blob, copy)
    same = copy < args.same_shot_threshold

    print(f"{len(s)} test windows; shot continues (copy L1 < {args.same_shot_threshold}) in {same.float().mean():.1%} of them\n")
    print(f"{'':28s}{'all':>8s}{'shot continues':>16s}{'shot cuts':>12s}")
    for name, v in (("blob (median image)", blob), ("copy last frame", copy), ("oracle min(blob, copy)", oracle),
                    ("v2 model", mdl), ("autoencoder w/ true target", recon)):
        print(f"{name:28s}{v.mean():8.4f}{v[same].mean():16.4f}{v[~same].mean():12.4f}")
    q = torch.quantile(copy, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=DEVICE))
    print("\ncopy-last L1 quantiles 10/25/50/75/90 %:", [round(x, 3) for x in q.tolist()])
    print(f"model beats copy on {(mdl < copy).float().mean():.1%} of windows, beats blob on {(mdl < blob).float().mean():.1%}")
    print(f"on windows where the shot continues, model L1 - copy L1 = {(mdl[same] - copy[same]).mean():+.4f}")


if __name__ == "__main__":
    main()
