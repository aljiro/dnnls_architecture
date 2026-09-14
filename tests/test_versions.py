"""Every version's model builds and runs forward / backward on random inputs (CPU, no caches needed)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from storyseq.components import PredictorConfig, SequencePredictor, VisualAutoencoder, kl_divergence  # noqa: E402

VOCAB, B, K, M, S = 30522, 2, 4, 4, 8


class FakeTok:
    vocab_size, pad_token_id = VOCAB, 0


def load_config(version: str):
    spec = importlib.util.spec_from_file_location(f"{version}.model", ROOT / version / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


def random_batch(cfg: PredictorConfig) -> dict:
    kw = dict(frames=torch.rand(B, K, 3, 60, 125), text=torch.rand(B, K, 384), target_ids_in=torch.randint(1, VOCAB, (B, 99)))
    if cfg.annotations:
        kw.update(set_emb=torch.rand(B, K, 384), ent_slot=torch.tensor([[[0, 1, -1, -1]] * K, [[-1] * M] * K]),
                  chars_in=torch.rand(B, K, S) > 0.5, slot_name_ids=torch.randint(0, 3000, (B, S, 6)))
        if cfg.entity_features == "clip":
            kw["ent_clip"] = torch.rand(B, K, M, 512)
        else:
            kw["ent_pix"] = torch.randint(0, 255, (B, K, M, 3, 30, 62), dtype=torch.uint8)
    if cfg.clip_input:
        kw["clip"] = torch.rand(B, K, 512)
    return kw


@pytest.mark.parametrize("version", ["v3_multimodal_fusion", "v4_attention", "v5_protect", "v6_annotations", "v7_names",
                                     "v8_variational", "v9_scaling", "v10_semantic"])
def test_version_forward_backward(version):
    cfg = load_config(version)
    ae = VisualAutoencoder(width=cfg.ae_width)
    model = SequencePredictor(ae, VOCAB, cfg.model)
    kw = random_batch(cfg.model)
    model.train()
    if cfg.model.latent_mode == "variational":
        kw["target_latent"] = model.target_latent(torch.rand(B, 3, 60, 125))
    if cfg.model.text_memory:
        kw["names_present"] = torch.rand(B, S) > 0.5
    o = model(**kw)
    assert o["image"].shape == (B, 3, 60, 125) and o["logits"].shape == (B, 99, VOCAB) and o["alpha"].shape == (B, K)
    loss = o["image"].mean() + o["logits"].mean() + o["z"].mean() + o["e_txt"].mean()
    if cfg.model.latent_mode == "variational":
        loss = loss + kl_divergence(o)
    if cfg.model.annotations:
        assert o["char_logits"].shape == (B, S)
        loss = loss + o["char_logits"].mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        kw.pop("target_latent", None); kw.pop("names_present", None)
        o = model(**kw)
        gen = model.text_decoder.generate(o["cond"], 101, 102, max_len=5, sample=True,
                                          memory=o.get("memory"), memory_mask=o.get("memory_mask"))
    assert len(gen) == B


def test_component_injection():
    from storyseq.components import ContentAttention

    class Uniform(ContentAttention):
        def forward(self, sequence, h):
            return torch.full(sequence.shape[:2], 1.0 / sequence.shape[1], device=sequence.device)

    cfg = load_config("v4_attention")
    model = SequencePredictor(VisualAutoencoder(), VOCAB, cfg.model, components={"context_attention": Uniform})
    o = model(**random_batch(cfg.model))
    assert torch.allclose(o["alpha_ctx"], torch.full((B, K), 0.25))
