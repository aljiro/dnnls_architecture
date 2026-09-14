"""The sequence predictor: encoders -> fusion -> sequence model -> attention -> latents -> decoders.

Every level builds the same class with a different `PredictorConfig` and, optionally, its own
component classes injected through `components=`. A component is any nn.Module with the
interface documented in the module that holds the default (autoencoder.py, text.py,
attention.py). The predictor only calls those interfaces, so a student can replace, say, the
attention or the text decoder without touching this file.

Outputs of forward() (a dict):
  image, generated       decoded prediction (after the copy path if enabled) and before it
  logits                 text decoder logits (teacher forcing)
  z, z_txt, e_txt, cond  image latent, text latent, predicted text embedding, decoder condition
  alpha, alpha_ctx, gate mixture weights over the inputs, context weights, gate
  char_logits, setting   (annotations) which characters appear next; next setting embedding
  mu_p, logvar_p, mu_q, logvar_q   (variational) prior / posterior statistics for the KL
  memory, memory_mask    (text memory) what the text decoder attended over
  copy_gate              (copy path) per-pixel gate map
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import (ContentAttention, EntityPooling, FixedQueryAttention, LatentSimilarityAttention,
                        PixelCopyPath, SlotHead)
from .autoencoder import IMAGE_HW, N_SLOTS, VisualAutoencoder
from .text import TextDecoder


@dataclass
class PredictorConfig:
    latent_dim: int = 256
    hidden_dim: int = 256
    text_dim: int = 384            # MiniLM vectors; a from-scratch text encoder sets its own
    text_embed_dim: int = 384
    setting_dim: int = 384
    clip_dim: int = 512
    word_dropout: float = 0.3
    # --- what the level uses ---
    latent_mode: str = "residual"  # "shared": one vector for every head (Level 3, the failure case)
                                   # "residual": separate image / text latents, no mixture
                                   # "mixture": gate * mixture of input latents + (1 - gate) * residual (Level 4)
                                   # "variational": mixture with a stochastic residual (Level 8)
    attention: str = "fixed"       # context attention over the GRU outputs: "fixed" | "content"
    mix_attention: str = "same"    # weights for the mixture / copy path: "same" (as context) | "similarity" (Level 7)
    cond_with_embedding: bool = False   # text decoder also conditioned on the predicted text embedding (Level 5+)
    annotations: bool = False      # setting vectors, entity tokens, character and setting heads (Level 6)
    entity_features: str = "ae"    # "ae": encode crops with the image encoder; "clip": cached CLIP crop embeddings (Level 9)
    per_slot_head: bool = False    # character head per slot from its own history (Level 7)
    text_memory: bool = False      # decoder cross-attention over descriptions, embedding and names (Level 7)
    copy_path: bool = False        # pixel copy path over the inputs
    clip_input: bool = False       # cached CLIP frame embeddings as an extra input (Level 9)


DEFAULT_COMPONENTS = {
    "context_attention": {"fixed": FixedQueryAttention, "content": ContentAttention},
    "mix_attention": LatentSimilarityAttention,
    "text_decoder": TextDecoder,
    "entity_pool": EntityPooling,
    "slot_head": SlotHead,
    "copy_path": PixelCopyPath,
    "sequence_model": nn.GRU,
}


class SequencePredictor(nn.Module):
    def __init__(self, autoencoder: VisualAutoencoder, vocab_size: int, cfg: PredictorConfig,
                 text_encoder: nn.Module | None = None, components: dict | None = None) -> None:
        super().__init__()
        c = {**DEFAULT_COMPONENTS, **(components or {})}
        self.cfg = cfg
        L, H = cfg.latent_dim, cfg.hidden_dim
        self.image_encoder = autoencoder.encoder
        self.image_decoder = autoencoder.decoder
        self.target_encoder = copy.deepcopy(autoencoder.encoder).eval()     # frozen: the latent target cannot drift
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.text_encoder = text_encoder                                    # None -> precomputed MiniLM vectors
        text_dim = cfg.text_dim if text_encoder is None else text_encoder.out_dim
        in_dim = L + text_dim + (cfg.setting_dim if cfg.annotations else 0) + (cfg.clip_dim if cfg.clip_input else 0)
        self.fuse = nn.Sequential(nn.Linear(in_dim, H), nn.LayerNorm(H), nn.GELU())
        self.sequence_model = c["sequence_model"](H, H, batch_first=True)
        att = c["context_attention"]
        self.attention = (att[cfg.attention] if isinstance(att, dict) else att)(H)
        if cfg.mix_attention == "similarity":
            self.mix_attention = c["mix_attention"]()
        if cfg.latent_mode == "shared":
            self.projection = nn.Sequential(nn.Linear(2 * H, L), nn.LayerNorm(L))
        else:
            self.text_latent = nn.Sequential(nn.Linear(2 * H, L), nn.LayerNorm(L))
            if cfg.latent_mode == "variational":
                self.prior = nn.Linear(2 * H, 2 * L)
                self.posterior = nn.Linear(2 * H + L, 2 * L)
                self.res_proj = nn.Sequential(nn.Linear(L, L), nn.LayerNorm(L))
            else:
                self.residual = nn.Sequential(nn.Linear(2 * H, L), nn.BatchNorm1d(L, affine=False))
            if cfg.latent_mode in ("mixture", "variational"):
                self.gate = nn.Linear(H, 1)
        self.text_embed_head = nn.Linear(L, cfg.text_embed_dim)
        cond_dim = L + (cfg.text_embed_dim if cfg.cond_with_embedding else 0)
        self.text_decoder = c["text_decoder"](vocab_size, cond_dim=cond_dim, word_dropout=cfg.word_dropout,
                                              memory_dim=H if cfg.text_memory else None)
        if cfg.annotations:
            self.slot_embedding = nn.Embedding(N_SLOTS, 64)
            ent_in = (cfg.clip_dim if cfg.entity_features == "clip" else L) + 64
            self.entity_proj = nn.Sequential(nn.Linear(ent_in, H), nn.LayerNorm(H), nn.GELU())
            self.entity_pool = c["entity_pool"](H, H)
            self.char_head = c["slot_head"](H) if cfg.per_slot_head else nn.Linear(2 * H, N_SLOTS)
            self.setting_head = nn.Linear(2 * H, cfg.setting_dim)
            self.setting_to_latent = nn.Linear(cfg.setting_dim, L)
        if cfg.text_memory:
            self.mem_txt = nn.Linear(text_dim, H)
            self.mem_emb = nn.Linear(cfg.text_embed_dim, H)
            self.mem_name = nn.Linear(self.text_decoder.embedding.embedding_dim, H)
        self.copy_path = c["copy_path"]() if cfg.copy_path else None

    # ---- conveniences used by the trainer / metrics ----
    @property
    def annotated(self) -> bool:
        return self.cfg.annotations

    @property
    def entity_features(self) -> str:
        return self.cfg.entity_features

    @property
    def clip_input(self) -> bool:
        return self.cfg.clip_input

    @property
    def variational(self) -> bool:
        return self.cfg.latent_mode == "variational"

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def target_latent(self, frames: Tensor) -> Tensor:
        return self.target_encoder(frames)

    # ---- pieces ----
    def encode_text(self, text: Tensor) -> Tensor:
        if self.text_encoder is None:
            return text
        b, k = text.shape[:2]
        return self.text_encoder(text.flatten(0, 1)).view(b, k, -1)

    def encode_entities(self, ent_slot: Tensor, ent_pix: Tensor | None, ent_clip: Tensor | None) -> tuple[Tensor, Tensor]:
        b, k, m = ent_slot.shape
        valid = ent_slot >= 0
        if self.cfg.entity_features == "clip":
            z = ent_clip
        else:
            crops = F.interpolate(ent_pix.flatten(0, 2).float() / 255, size=IMAGE_HW, mode="bilinear", align_corners=False)
            z = self.image_encoder(crops).view(b, k, m, -1)
        return self.entity_proj(torch.cat((z, self.slot_embedding(ent_slot.clamp(min=0))), -1)), valid

    def variational_residual(self, hc: Tensor, target_latent: Tensor | None, sample: bool) -> tuple[Tensor, dict[str, Tensor]]:
        mu_p, lv_p = self.prior(hc).chunk(2, -1)
        lv_p = lv_p.clamp(-8, 4)
        stats = {"mu_p": mu_p, "logvar_p": lv_p}
        if target_latent is not None:                                   # training: posterior sample
            mu_q, lv_q = self.posterior(torch.cat((hc, target_latent), -1)).chunk(2, -1)
            lv_q = lv_q.clamp(-8, 4)
            z = mu_q + torch.exp(0.5 * lv_q) * torch.randn_like(mu_q)
            stats.update({"mu_q": mu_q, "logvar_q": lv_q})
        elif sample:
            z = mu_p + torch.exp(0.5 * lv_p) * torch.randn_like(mu_p)
        else:
            z = mu_p
        return self.res_proj(z), stats

    def name_memory(self, slot_name_ids: Tensor) -> Tensor:
        emb = self.text_decoder.embedding(slot_name_ids)                # [B, S, 6, E]
        m = (slot_name_ids > 0).float()[..., None]
        return self.mem_name((emb * m).sum(2) / m.sum(2).clamp(min=1))

    # ---- forward ----
    def forward(self, frames: Tensor, text: Tensor, target_ids_in: Tensor, *,
                set_emb: Tensor | None = None, ent_pix: Tensor | None = None, ent_slot: Tensor | None = None,
                ent_clip: Tensor | None = None, chars_in: Tensor | None = None, slot_name_ids: Tensor | None = None,
                clip: Tensor | None = None, target_latent: Tensor | None = None, names_present: Tensor | None = None,
                sample: bool = False) -> dict[str, Tensor]:
        cfg = self.cfg
        b, k = frames.shape[:2]
        zv = self.image_encoder(frames.flatten(0, 1)).view(b, k, -1)
        zt = self.encode_text(text)
        parts = [zv, zt]
        if cfg.annotations:
            parts.append(set_emb)
        if cfg.clip_input:
            parts.append(clip)
        x = self.fuse(torch.cat(parts, -1))
        if cfg.annotations:
            ents, valid = self.encode_entities(ent_slot, ent_pix, ent_clip)
            x = self.entity_pool(x, ents, valid)
        seq, h = self.sequence_model(x)
        h = h[-1]
        alpha_ctx = self.attention(seq, h)
        ctx = torch.einsum("bk,bkd->bd", alpha_ctx, seq)
        hc = torch.cat((h, ctx), -1)
        alpha = self.mix_attention(zv) if cfg.mix_attention == "similarity" else alpha_ctx
        out: dict[str, Tensor] = {"alpha": alpha, "alpha_ctx": alpha_ctx, "gate": torch.zeros(b, device=frames.device)}
        if cfg.latent_mode == "shared":
            z = z_txt = self.projection(hc)
        else:
            if cfg.latent_mode == "variational":
                residual, stats = self.variational_residual(hc, target_latent, sample)
                out.update(stats)
            else:
                residual = self.residual(hc)
            if cfg.latent_mode in ("mixture", "variational"):
                mix = torch.einsum("bk,bkd->bd", alpha, zv)
                g = torch.sigmoid(self.gate(h)).squeeze(-1)
                z = F.layer_norm(g[:, None] * mix + (1 - g[:, None]) * residual, (zv.size(-1),))
                out["gate"] = g
            else:
                z = residual
            z_txt = self.text_latent(hc)
        e_txt = self.text_embed_head(z_txt)
        cond = torch.cat((z_txt, e_txt), -1) if cfg.cond_with_embedding else z_txt
        z_img = z
        memory = memory_mask = None
        if cfg.annotations:
            if cfg.per_slot_head:
                out["char_logits"] = self.char_head(hc, chars_in, self.slot_embedding.weight, ents, ent_slot, valid)
            else:
                out["char_logits"] = self.char_head(hc)
            out["setting"] = self.setting_head(hc)
            z_img = z + self.setting_to_latent(out["setting"])
        if cfg.text_memory:
            present = names_present if names_present is not None else torch.sigmoid(out["char_logits"]) > 0.3
            present = present & (slot_name_ids.sum(-1) > 0)
            memory = torch.cat((self.mem_txt(zt), self.mem_emb(e_txt)[:, None], self.name_memory(slot_name_ids)), 1)
            memory_mask = torch.cat((torch.ones(b, k + 1, dtype=torch.bool, device=frames.device), present), 1)
            out["memory"], out["memory_mask"] = memory, memory_mask
        generated = self.image_decoder(z_img)
        out.update({"image": generated, "generated": generated,
                    "logits": self.text_decoder(target_ids_in, cond, memory, memory_mask),
                    "z": z, "z_txt": z_txt, "e_txt": e_txt, "cond": cond})
        if self.copy_path is not None:
            out["image"], out["copy_gate"] = self.copy_path(generated, frames, alpha)
        return out
