"""v2 components. Same pipeline shape as the notebook, with the fixes from ASSESSMENT.md.

  ConvEncoder / ConvDecoder / VisualAutoencoder   from-scratch CNN, latent 256, LayerNorm (no ReLU) on the latent
  TextEncoderLSTM                                 optional from-scratch text encoder (default is frozen MiniLM)
  TextDecoder                                     LSTM conditioned at every step; word dropout; greedy or nucleus sampling
  SequencePredictor                               encoders -> fuse -> GRU -> attention -> latents -> decoders

Stages (ASSESSMENT.md section 9), cumulative:
  "0": the pass-2 model of section 7 (fixed-query attention, one shared latent z)
  "A": constant component removed from the residual (BatchNorm without affine); content-dependent
       attention (query from the final GRU state); image latent = gate * mixture of input frame
       latents + (1 - gate) * residual; separate text latent
  "B": A + text decoder conditioned on (text latent, predicted text embedding)
  "C": B + per-frame setting embedding as input; character crops as entity tokens pooled into each
       frame token; heads predicting which characters appear next (multi-label) and the next
       setting embedding; image decoder conditioned on the predicted setting embedding
  "D": C + (1) variational residual: prior N(mu_p, s_p) from the sequence state, posterior from the
       state and the frozen target latent, KL(q || p), samples at test time; (2) mixture / copy-path
       attention from the similarities between the input frame latents, decoupled from the GRU;
       (3) per-slot character head on each character's own presence history and entity token;
       (4) text decoder with cross-attention over the input descriptions, the predicted text
       embedding and the names of the characters predicted present
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGE_HW = (60, 125)
N_SLOTS = 8            # character slots per story (see v2/precompute_annotations.py)


class ConvEncoder(nn.Module):
    """60x125 image -> latent. Four stride-2 convs to a 4x8 map, then a linear layer."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        ch = [3, 32, 64, 128, 256]
        layers: list[nn.Module] = []
        for i in range(4):
            k, p = (5, 2) if i == 0 else (3, 1)
            layers += [nn.Conv2d(ch[i], ch[i + 1], k, stride=2, padding=p), nn.GroupNorm(8, ch[i + 1]), nn.LeakyReLU(0.1)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(256 * 4 * 8, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.fc(self.conv(x).flatten(1)))


class ConvDecoder(nn.Module):
    """latent -> 60x125 image in [0, 1]."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 8)
        ch = [256, 128, 64, 32]
        layers: list[nn.Module] = []
        for i in range(3):
            layers += [nn.ConvTranspose2d(ch[i], ch[i + 1], 4, stride=2, padding=1), nn.GroupNorm(8, ch[i + 1]), nn.LeakyReLU(0.1)]
        layers += [nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        x = self.deconv(self.fc(z).view(-1, 256, 4, 8))          # [B, 3, 64, 128]
        return x[:, :, :IMAGE_HW[0], :IMAGE_HW[1]]


class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


class TextEncoderLSTM(nn.Module):
    """From-scratch alternative to MiniLM: embed tokens, LSTM, mean-pool over real tokens."""

    def __init__(self, vocab_size: int, pad_id: int, embedding_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.out_dim = 2 * hidden_dim

    def forward(self, ids: Tensor) -> Tensor:                    # [B, T] -> [B, out_dim]
        mask = (ids != self.pad_id).float().unsqueeze(-1)
        out, _ = self.lstm(self.embedding(ids))
        return self.norm((out * mask).sum(1) / mask.sum(1).clamp(min=1))


class TextDecoder(nn.Module):
    """LSTM language model conditioned on a vector at every step.

    The condition is first projected to a fixed COND_PROJ_DIM, so the embedding, LSTM and output
    layers are the same for every stage and can be pretrained as an unconditional language model
    (v2/pretrain_text.py, condition = zeros) and then loaded into any predictor."""

    COND_PROJ_DIM = 256

    def __init__(self, vocab_size: int, cond_dim: int, embedding_dim: int = 128, hidden_dim: int = 384,
                 word_dropout: float = 0.0, unk_id: int = 100, memory_dim: int | None = None) -> None:
        super().__init__()
        self.word_dropout, self.unk_id = word_dropout, unk_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cond_proj = nn.Linear(cond_dim, self.COND_PROJ_DIM)
        self.init_h = nn.Linear(self.COND_PROJ_DIM, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + self.COND_PROJ_DIM, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.memory_dim = memory_dim
        if memory_dim:                                   # stage D: Luong-style attention over a memory of vectors
            self.mem_q = nn.Linear(hidden_dim, hidden_dim)
            self.mem_k = nn.Linear(memory_dim, hidden_dim)
            self.mem_v = nn.Linear(memory_dim, hidden_dim)
            self.mem_out = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())

    def _attend(self, out: Tensor, memory: Tensor, mask: Tensor) -> Tensor:
        """out [B, L, H], memory [B, M, Dm], mask [B, M] -> [B, L, H]"""
        scores = torch.einsum("blh,bmh->blm", self.mem_q(out), self.mem_k(memory)) / math.sqrt(out.size(-1))
        scores = scores.masked_fill(~mask[:, None], float("-inf"))
        att = torch.softmax(scores, -1).nan_to_num(0.0)
        return self.mem_out(torch.cat((out, torch.einsum("blm,bmh->blh", att, self.mem_v(memory))), -1))

    def _step_input(self, ids: Tensor, c: Tensor) -> Tensor:
        return torch.cat((self.embedding(ids), c[:, None].expand(-1, ids.size(1), -1)), -1)

    def forward(self, ids_in: Tensor, cond: Tensor, memory: Tensor | None = None,
                memory_mask: Tensor | None = None) -> Tensor:  # teacher forcing: [B, L] -> [B, L, vocab]
        if self.training and self.word_dropout > 0:
            # replace a share of the teacher-forced tokens with [UNK] so the decoder cannot rely on
            # the previous words alone and has to use the conditioning vector (Bowman et al. 2016)
            drop = torch.rand_like(ids_in, dtype=torch.float) < self.word_dropout
            ids_in = ids_in.masked_fill(drop, self.unk_id)
        c = self.cond_proj(cond)
        h0 = torch.tanh(self.init_h(c))[None]
        out, _ = self.lstm(self._step_input(ids_in, c), (h0, torch.zeros_like(h0)))
        if memory is not None:
            out = self._attend(out, memory, memory_mask)
        return self.out(out)

    def load_language_model(self, path) -> None:
        """Load the embedding / LSTM / output layers from an unconditional pretraining run."""
        state = torch.load(path, map_location="cpu")
        keep = {k: v for k, v in state.items() if k.split(".")[0] in ("embedding", "lstm", "out")}
        missing, unexpected = self.load_state_dict(keep, strict=False)
        assert not unexpected and all(m.split(".")[0] in ("cond_proj", "init_h", "mem_q", "mem_k", "mem_v", "mem_out")
                                      for m in missing), (missing, unexpected)

    @torch.no_grad()
    def generate(self, cond: Tensor, cls_id: int, sep_id: int, max_len: int = 80, sample: bool = False,
                 top_p: float = 0.9, temperature: float = 0.8, repetition_penalty: float = 1.3,
                 memory: Tensor | None = None, memory_mask: Tensor | None = None) -> list[list[int]]:
        """Greedy by default; sample=True gives nucleus sampling with a repetition penalty."""
        cp = self.cond_proj(cond)
        h = torch.tanh(self.init_h(cp))[None]
        c = torch.zeros_like(h)
        b = cond.size(0)
        ids = torch.full((b, 1), cls_id, dtype=torch.long, device=cond.device)
        out_ids, done = [], torch.zeros(b, dtype=torch.bool, device=cond.device)
        seen = torch.zeros(b, self.out.out_features, dtype=torch.bool, device=cond.device)
        for _ in range(max_len):
            o, (h, c) = self.lstm(self._step_input(ids, cp), (h, c))
            if memory is not None:
                o = self._attend(o, memory, memory_mask)
            logits = self.out(o[:, -1])
            if sample:
                logits = torch.where(seen, logits / repetition_penalty, logits) / temperature
                probs = torch.softmax(logits, -1)
                sp, si = probs.sort(-1, descending=True)
                keep = (sp.cumsum(-1) - sp) < top_p
                sp = sp * keep
                ids = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
            else:
                ids = logits.argmax(-1, keepdim=True)
            seen.scatter_(1, ids, True)
            out_ids.append(ids)
            done |= ids.squeeze(1) == sep_id
            if done.all():
                break
        seq = torch.cat(out_ids, 1).tolist()
        return [s[: s.index(sep_id)] if sep_id in s else s for s in seq]


class FixedQueryAttention(nn.Module):
    """The notebook's attention: one learned query, the same importance profile for any input."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: Tensor, h: Tensor) -> Tensor:    # -> alpha [B, K]
        return torch.softmax(self.score(sequence).squeeze(-1), dim=1)


class ContentAttention(nn.Module):
    """Stage A: the query comes from the final GRU state, so the weights depend on the sequence."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, sequence: Tensor, h: Tensor) -> Tensor:    # -> alpha [B, K]
        scores = torch.einsum("bd,bkd->bk", self.q(h), self.k(sequence)) / math.sqrt(sequence.size(-1))
        return torch.softmax(scores, dim=1)


class LatentSimilarityAttention(nn.Module):
    """Stage D: weights over the inputs from the pattern of similarities between their frame latents
    (window-centred cosine), plus position. Independent of the GRU state, so it can be supervised
    toward the closest input without reshaping what the text heads read."""

    def __init__(self, k: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * k, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.register_buffer("eye", torch.eye(k))

    def forward(self, zv: Tensor) -> Tensor:                     # [B, K, D] -> alpha [B, K]
        zc = F.normalize(zv - zv.mean(1, keepdim=True), dim=-1)
        sim = zc @ zc.transpose(1, 2)                               # [B, K, K]
        feats = torch.cat((sim, self.eye.expand(len(zv), -1, -1)), -1)
        return torch.softmax(self.mlp(feats).squeeze(-1), dim=1)


class SlotHead(nn.Module):
    """Stage D: one logit per character slot from that slot's own evidence: its presence in each of
    the K inputs, how often, its slot embedding, its pooled entity token, and the sequence state."""

    def __init__(self, hidden_dim: int, slot_dim: int = 64, k: int = 4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(k + 1 + slot_dim + hidden_dim + 2 * hidden_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, hc: Tensor, chars_in: Tensor, slot_emb: Tensor, ents: Tensor, ent_slot: Tensor, valid: Tensor) -> Tensor:
        b, k, n_slots = chars_in.shape
        hist = chars_in.float().transpose(1, 2)                     # [B, S, K]
        count = hist.sum(-1, keepdim=True) / k
        onehot = F.one_hot(ent_slot.clamp(min=0), n_slots).float() * valid[..., None].float()   # [B, K, M, S]
        pooled = torch.einsum("bkms,bkmh->bsh", onehot, ents) / onehot.sum((1, 2)).clamp(min=1)[..., None]
        x = torch.cat((hist, count, slot_emb[None].expand(b, -1, -1), pooled, hc[:, None].expand(-1, n_slots, -1)), -1)
        return self.mlp(x).squeeze(-1)                              # [B, S]


class EntityPooling(nn.Module):
    """Stage C: each frame token attends over the entity tokens of its frame and adds the result."""

    def __init__(self, hidden_dim: int, ent_dim: int) -> None:
        super().__init__()
        self.k = nn.Linear(ent_dim, hidden_dim)
        self.v = nn.Linear(ent_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, ents: Tensor, valid: Tensor) -> Tensor:
        # x [B, K, H], ents [B, K, M, E], valid [B, K, M]
        scores = torch.einsum("bkh,bkmh->bkm", x, self.k(ents)) / math.sqrt(x.size(-1))
        scores = scores.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(scores, -1).nan_to_num(0.0)        # frames without entities -> zero
        return self.norm(x + torch.einsum("bkm,bkmh->bkh", alpha, self.v(ents)))


class PixelCopyPath(nn.Module):
    """Blend the input frames in pixel space with the attention weights, then let a per-pixel gate
    choose between that blend and the generated image.

    Inputs to the gate: the generated image, the blend, and the attention-weighted variance of the
    inputs around the blend (high where the inputs disagree, so copying is risky)."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(9, width, 3, padding=1), nn.LeakyReLU(0.1),
                                 nn.Conv2d(width, width, 3, padding=1), nn.LeakyReLU(0.1),
                                 nn.Conv2d(width, 1, 3, padding=1))

    def forward(self, generated: Tensor, frames: Tensor, alpha: Tensor) -> tuple[Tensor, Tensor]:
        blend = torch.einsum("bk,bkchw->bchw", alpha, frames)
        var = torch.einsum("bk,bkchw->bchw", alpha, (frames - blend[:, None]) ** 2)
        gate = torch.sigmoid(self.net(torch.cat((generated, blend, var), 1)))      # [B, 1, H, W]
        return gate * blend + (1 - gate) * generated, gate


class SequencePredictor(nn.Module):
    def __init__(self, autoencoder: VisualAutoencoder, text_dim: int, vocab_size: int,
                 text_encoder: nn.Module | None = None, latent_dim: int = 256, hidden_dim: int = 256,
                 word_dropout: float = 0.0, text_embed_dim: int = 384, stage: str = "0",
                 setting_dim: int = 384, copy_path: bool = False) -> None:
        super().__init__()
        assert stage in ("0", "A", "B", "C", "D")
        self.stage = stage
        self.annotated = stage in ("C", "D")
        self.copy_path = PixelCopyPath() if copy_path else None
        self.image_encoder = autoencoder.encoder
        self.image_decoder = autoencoder.decoder
        # frozen copy of the pretrained encoder: the latent target cannot drift while the online encoder is fine-tuned
        self.target_encoder = copy.deepcopy(autoencoder.encoder).eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.text_encoder = text_encoder                        # None -> inputs are precomputed MiniLM vectors
        in_dim = latent_dim + text_dim + (setting_dim if self.annotated else 0)
        self.fuse = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.temporal_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = FixedQueryAttention(hidden_dim) if stage == "0" else ContentAttention(hidden_dim)
        if stage == "0":
            self.projection = nn.Sequential(nn.Linear(2 * hidden_dim, latent_dim), nn.LayerNorm(latent_dim))
        else:
            self.residual = nn.Sequential(nn.Linear(2 * hidden_dim, latent_dim), nn.BatchNorm1d(latent_dim, affine=False))
            self.gate = nn.Linear(hidden_dim, 1)
            self.text_latent = nn.Sequential(nn.Linear(2 * hidden_dim, latent_dim), nn.LayerNorm(latent_dim))
        self.text_embed_head = nn.Linear(latent_dim, text_embed_dim)
        cond_dim = latent_dim + (text_embed_dim if stage in ("B", "C", "D") else 0)
        self.text_decoder = TextDecoder(vocab_size, cond_dim=cond_dim, word_dropout=word_dropout,
                                        memory_dim=hidden_dim if stage == "D" else None)
        if stage == "D":
            self.prior = nn.Linear(2 * hidden_dim, 2 * latent_dim)
            self.posterior = nn.Linear(2 * hidden_dim + latent_dim, 2 * latent_dim)
            self.res_proj = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.LayerNorm(latent_dim))
            self.mix_attention = LatentSimilarityAttention()
            self.slot_head = SlotHead(hidden_dim)
            self.mem_txt = nn.Linear(text_dim, hidden_dim)
            self.mem_emb = nn.Linear(text_embed_dim, hidden_dim)
            self.mem_name = nn.Linear(self.text_decoder.embedding.embedding_dim, hidden_dim)
        if self.annotated:
            self.slot_embedding = nn.Embedding(N_SLOTS, 64)
            self.entity_proj = nn.Sequential(nn.Linear(latent_dim + 64, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
            self.entity_pool = EntityPooling(hidden_dim, hidden_dim)
            self.char_head = nn.Linear(2 * hidden_dim, N_SLOTS)
            self.setting_head = nn.Linear(2 * hidden_dim, setting_dim)
            self.setting_to_latent = nn.Linear(setting_dim, latent_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()                              # never in train mode (GroupNorm is fine, but be explicit)
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

    def encode_entities(self, ent_pix: Tensor, ent_slot: Tensor) -> tuple[Tensor, Tensor]:
        """ent_pix [B, K, M, 3, h, w] uint8 crops, ent_slot [B, K, M] -> tokens [B, K, M, H], valid [B, K, M]"""
        b, k, m = ent_slot.shape
        valid = ent_slot >= 0
        crops = ent_pix.flatten(0, 2).float() / 255
        crops = F.interpolate(crops, size=IMAGE_HW, mode="bilinear", align_corners=False)
        z = self.image_encoder(crops).view(b, k, m, -1)
        slot = self.slot_embedding(ent_slot.clamp(min=0))
        return self.entity_proj(torch.cat((z, slot), -1)), valid

    def variational_residual(self, hc: Tensor, target_latent: Tensor | None, sample: bool) -> tuple[Tensor, dict[str, Tensor]]:
        """Stage D. Training (target_latent given): sample from the posterior q(z | state, target).
        Evaluation: the prior mean (deterministic) or a prior sample (sample=True)."""
        mu_p, lv_p = self.prior(hc).chunk(2, -1)
        lv_p = lv_p.clamp(-8, 4)
        stats = {"mu_p": mu_p, "logvar_p": lv_p}
        if target_latent is not None:
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
        """[B, S, 6] token ids -> [B, S, H] mean of the decoder's token embeddings, projected."""
        emb = self.text_decoder.embedding(slot_name_ids)            # [B, S, 6, E]
        m = (slot_name_ids > 0).float()[..., None]
        return self.mem_name((emb * m).sum(2) / m.sum(2).clamp(min=1))

    def forward(self, frames: Tensor, text: Tensor, target_ids_in: Tensor,
                set_emb: Tensor | None = None, ent_pix: Tensor | None = None, ent_slot: Tensor | None = None,
                chars_in: Tensor | None = None, slot_name_ids: Tensor | None = None,
                target_latent: Tensor | None = None, names_present: Tensor | None = None,
                sample: bool = False) -> dict[str, Tensor]:
        b, k = frames.shape[:2]
        zv = self.image_encoder(frames.flatten(0, 1)).view(b, k, -1)
        zt = self.encode_text(text)
        parts = [zv, zt]
        if self.annotated:
            parts.append(set_emb)
        x = self.fuse(torch.cat(parts, -1))
        if self.annotated:
            ents, valid = self.encode_entities(ent_pix, ent_slot)
            x = self.entity_pool(x, ents, valid)
        seq, h = self.temporal_rnn(x)
        h = h[-1]
        alpha_ctx = self.attention(seq, h)                         # [B, K], from the sequence state
        ctx = torch.einsum("bk,bkd->bd", alpha_ctx, seq)
        hc = torch.cat((h, ctx), -1)
        alpha = self.mix_attention(zv) if self.stage == "D" else alpha_ctx    # weights for the mixture / copy path
        out: dict[str, Tensor] = {"alpha": alpha, "alpha_ctx": alpha_ctx}
        if self.stage == "0":
            z = self.projection(hc)
            z_txt = z
            out["gate"] = torch.zeros(b, device=frames.device)
        else:
            mix = torch.einsum("bk,bkd->bd", alpha, zv)             # a point in the frame-latent space
            g = torch.sigmoid(self.gate(h)).squeeze(-1)
            if self.stage == "D":
                residual, stats = self.variational_residual(hc, target_latent, sample)
                out.update(stats)
            else:
                residual = self.residual(hc)
            z = F.layer_norm(g[:, None] * mix + (1 - g[:, None]) * residual, (zv.size(-1),))
            z_txt = self.text_latent(hc)
            out["gate"] = g
        e_txt = self.text_embed_head(z_txt)
        cond = torch.cat((z_txt, e_txt), -1) if self.stage in ("B", "C", "D") else z_txt
        z_img = z
        memory = memory_mask = None
        if self.annotated:
            if self.stage == "D":
                out["char_logits"] = self.slot_head(hc, chars_in, self.slot_embedding.weight, ents, ent_slot, valid)
            else:
                out["char_logits"] = self.char_head(hc)
            out["setting"] = self.setting_head(hc)
            z_img = z + self.setting_to_latent(out["setting"])
        if self.stage == "D":
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

    @torch.no_grad()
    def generate_text(self, cond: Tensor, cls_id: int, sep_id: int, **kw) -> list[list[int]]:
        return self.text_decoder.generate(cond, cls_id, sep_id, **kw)


def kl_divergence(stats: dict[str, Tensor]) -> Tensor:
    """KL(q || p) between the diagonal Gaussians of the posterior and the conditional prior, per window."""
    mu_p, lv_p, mu_q, lv_q = stats["mu_p"], stats["logvar_p"], stats["mu_q"], stats["logvar_q"]
    kl = 0.5 * (lv_p - lv_q + (torch.exp(lv_q) + (mu_q - mu_p) ** 2) / torch.exp(lv_p) - 1)
    return kl.sum(-1).mean()


def centred_cosine_loss(pred: Tensor, target: Tensor) -> Tensor:
    """1 - cosine similarity after subtracting the batch mean of the (detached) targets.

    Encoder latents share a large mean vector (raw pairwise cosine 0.31; MiniLM vectors 0.38), so a
    raw cosine loss is nearly blind to the informative part and can be satisfied by shrinking all
    latents toward the mean. Centring removes that solution."""
    target = target.detach()
    mu = target.mean(0, keepdim=True)
    return 1 - F.cosine_similarity(pred - mu, target - mu, dim=-1).mean()


latent_loss = centred_cosine_loss
