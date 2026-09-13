"""v2 components. Same pipeline shape as the notebook, with the fixes from ASSESSMENT.md.

  ConvEncoder / ConvDecoder / VisualAutoencoder   from-scratch CNN, latent 256, LayerNorm (no ReLU) on the latent
  TextEncoderLSTM                                 optional from-scratch text encoder (default is frozen MiniLM)
  TextDecoder                                     LSTM conditioned on the fused vector at every step
  Attention                                       the notebook's learned-query attention, unchanged
  SequencePredictor                               encoders -> fuse -> GRU -> attention -> latent -> decoders

What changed and why (numbers refer to ASSESSMENT.md):
  - no context head: the two heads shared one tensor and trained the output to be the mean image
  - latent 16 -> 256, ReLU -> LayerNorm: the 16-d ReLU bottleneck was constant across inputs from epoch 1
  - the predictor returns its latent so training can add a latent-space target (cosine to the
    encoder's latent of the true next frame), which is informative where pixel L1 is not
  - the text decoder sees the fused vector at every step, not only through h0/c0
  - second pass, after the probe showed the decoder ignored its condition (CE 3.061 with the true
    latent vs 3.070 with a shuffled one): word dropout on the teacher-forced tokens, and a head that
    predicts the MiniLM embedding of the next description so the latent must carry the text
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGE_HW = (60, 125)


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
    """LSTM language model conditioned on a vector at every step (concatenated to the token embedding)."""

    def __init__(self, vocab_size: int, cond_dim: int, embedding_dim: int = 128, hidden_dim: int = 384,
                 word_dropout: float = 0.0, unk_id: int = 100) -> None:
        super().__init__()
        self.word_dropout, self.unk_id = word_dropout, unk_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.init_h = nn.Linear(cond_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim + cond_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def _step_input(self, ids: Tensor, cond: Tensor) -> Tensor:
        return torch.cat((self.embedding(ids), cond[:, None].expand(-1, ids.size(1), -1)), -1)

    def forward(self, ids_in: Tensor, cond: Tensor) -> Tensor:  # teacher forcing: [B, L] -> [B, L, vocab]
        if self.training and self.word_dropout > 0:
            # replace a share of the teacher-forced tokens with [UNK] so the decoder cannot rely on
            # the previous words alone and has to use the conditioning vector (Bowman et al. 2016)
            drop = torch.rand_like(ids_in, dtype=torch.float) < self.word_dropout
            ids_in = ids_in.masked_fill(drop, self.unk_id)
        h0 = torch.tanh(self.init_h(cond))[None]
        out, _ = self.lstm(self._step_input(ids_in, cond), (h0, torch.zeros_like(h0)))
        return self.out(out)

    @torch.no_grad()
    def generate(self, cond: Tensor, cls_id: int, sep_id: int, max_len: int = 80) -> list[list[int]]:
        h = torch.tanh(self.init_h(cond))[None]
        c = torch.zeros_like(h)
        ids = torch.full((cond.size(0), 1), cls_id, dtype=torch.long, device=cond.device)
        out_ids, done = [], torch.zeros(cond.size(0), dtype=torch.bool, device=cond.device)
        for _ in range(max_len):
            o, (h, c) = self.lstm(self._step_input(ids, cond), (h, c))
            ids = self.out(o[:, -1]).argmax(-1, keepdim=True)
            out_ids.append(ids)
            done |= ids.squeeze(1) == sep_id
            if done.all():
                break
        seq = torch.cat(out_ids, 1).tolist()
        return [s[: s.index(sep_id)] if sep_id in s else s for s in seq]


class Attention(nn.Module):
    """Learned-query attention over the GRU outputs (as in the notebook)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: Tensor) -> Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)


class SequencePredictor(nn.Module):
    def __init__(self, autoencoder: VisualAutoencoder, text_dim: int, vocab_size: int,
                 text_encoder: nn.Module | None = None, latent_dim: int = 256, hidden_dim: int = 256,
                 word_dropout: float = 0.0, text_embed_dim: int = 384) -> None:
        super().__init__()
        self.image_encoder = autoencoder.encoder
        self.image_decoder = autoencoder.decoder
        self.text_encoder = text_encoder                        # None -> inputs are precomputed MiniLM vectors
        self.fuse = nn.Sequential(nn.Linear(latent_dim + text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.temporal_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.projection = nn.Sequential(nn.Linear(2 * hidden_dim, latent_dim), nn.LayerNorm(latent_dim))
        self.text_decoder = TextDecoder(vocab_size, cond_dim=latent_dim, word_dropout=word_dropout)
        self.text_embed_head = nn.Linear(latent_dim, text_embed_dim)   # predicts the MiniLM vector of the next description

    def encode_text(self, text: Tensor) -> Tensor:
        """text is [B, K, 384] MiniLM vectors, or [B, K, T] token ids when an LSTM encoder is set."""
        if self.text_encoder is None:
            return text
        b, k = text.shape[:2]
        return self.text_encoder(text.flatten(0, 1)).view(b, k, -1)

    def predict_latent(self, frames: Tensor, text: Tensor) -> Tensor:
        b, k = frames.shape[:2]
        zv = self.image_encoder(frames.flatten(0, 1)).view(b, k, -1)
        zt = self.encode_text(text)
        seq, h = self.temporal_rnn(self.fuse(torch.cat((zv, zt), -1)))
        return self.projection(torch.cat((h[-1], self.attention(seq)), -1))

    def forward(self, frames: Tensor, text: Tensor, target_ids_in: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z = self.predict_latent(frames, text)
        return self.image_decoder(z), self.text_decoder(target_ids_in, z), z, self.text_embed_head(z)


def latent_loss(z_pred: Tensor, z_true: Tensor) -> Tensor:
    """1 - cosine similarity to the encoder's latent of the true next frame (target detached)."""
    return 1 - F.cosine_similarity(z_pred, z_true.detach(), dim=-1).mean()
