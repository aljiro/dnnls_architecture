"""Text components (Level 2): a from-scratch LSTM encoder (the frozen MiniLM vectors are precomputed and
need no module) and the LSTM decoder used for generation.

Encoder interface: forward(ids [B, T]) -> vector [B, out_dim]; attribute out_dim.
Decoder interface: forward(ids_in [B, L], cond [B, cond_dim], memory=None, memory_mask=None) -> logits [B, L, vocab];
                   generate(cond, cls_id, sep_id, ...) -> list of token-id lists; load_language_model(path)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGE_HW = (60, 125)
N_SLOTS = 8            # character slots per story (see storyseq/precompute/annotations.py)


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


