"""Neural network components for StoryReasoning sequence prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        outputs, (hidden, cell) = self.lstm(self.embedding(input_ids))
        return outputs, hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: Tensor, hidden: Tensor,
                cell: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        output, (hidden, cell) = self.lstm(self.embedding(input_ids), (hidden, cell))
        return self.output(output), hidden, cell


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        _, hidden, cell = self.encoder(input_ids)
        logits, _, _ = self.decoder(target_ids[:, :-1], hidden, cell)
        return logits


class Backbone(nn.Module):
    def __init__(self, latent_dim: int = 16, output_height: int = 8,
                 output_width: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3), nn.GroupNorm(8, 16), nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.1),
        )
        self.flatten_dim = 64 * output_height * output_width
        self.projection = nn.Sequential(nn.Linear(self.flatten_dim, latent_dim), nn.ReLU())

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.features(images).flatten(1))


class VisualEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16, output_height: int = 8,
                 output_width: int = 16) -> None:
        super().__init__()
        self.context_backbone = Backbone(latent_dim, output_height, output_width)
        self.content_backbone = Backbone(latent_dim, output_height, output_width)
        self.projection = nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, images: Tensor) -> Tensor:
        content = self.content_backbone(images)
        context = self.context_backbone(images)
        return self.projection(torch.cat((content, context), dim=1))


class VisualDecoder(nn.Module):
    def __init__(self, latent_dim: int = 16, output_height: int = 8,
                 output_width: int = 16, image_size: tuple[int, int] = (60, 125)) -> None:
        super().__init__()
        self.image_height, self.image_width = image_size
        self.output_height, self.output_width = output_height, output_width
        self.projection = nn.Linear(latent_dim, 64 * output_height * output_width)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=2, output_padding=1),
            nn.GroupNorm(8, 16), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 3, 7, stride=2, padding=3, output_padding=1), nn.Sigmoid(),
        )

    def decode_image(self, latent: Tensor) -> Tensor:
        image = self.decoder(latent.view(-1, 64, self.output_height, self.output_width))
        return image[:, :, :self.image_height, :self.image_width]

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.projection(latent)
        return self.decode_image(projected), self.decode_image(projected)


class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16, image_size: tuple[int, int] = (60, 125)) -> None:
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim, image_size=image_size)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        return self.decoder(self.encoder(images))


class Attention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: Tensor) -> Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1)


class SequencePredictor(nn.Module):
    def __init__(self, visual_autoencoder: VisualAutoencoder,
                 text_autoencoder: Seq2SeqLSTM, latent_dim: int = 16,
                 gru_hidden_dim: int = 16, text_layers: int = 1) -> None:
        super().__init__()
        self.image_encoder = visual_autoencoder.encoder
        self.text_encoder = text_autoencoder.encoder
        self.temporal_rnn = nn.GRU(2 * latent_dim, gru_hidden_dim, batch_first=True)
        self.attention = Attention(gru_hidden_dim)
        self.projection = nn.Sequential(nn.Linear(2 * gru_hidden_dim, latent_dim), nn.ReLU())
        self.image_decoder = visual_autoencoder.decoder
        self.text_decoder = text_autoencoder.decoder
        text_hidden_dim = text_autoencoder.decoder.lstm.hidden_size
        self.text_layers = text_layers
        self.fused_to_h0 = nn.Linear(latent_dim, text_hidden_dim * text_layers)
        self.fused_to_c0 = nn.Linear(latent_dim, text_hidden_dim * text_layers)

    def forward(self, image_seq: Tensor, text_seq: Tensor,
                target_text: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, sequence_length = image_seq.shape[:2]
        image_latents = self.image_encoder(image_seq.flatten(0, 1))
        _, hidden, _ = self.text_encoder(text_seq.flatten(0, 1))
        text_latents = hidden[-1]
        image_latents = image_latents.view(batch_size, sequence_length, -1)
        text_latents = text_latents.view(batch_size, sequence_length, -1)
        fused = torch.cat((image_latents, text_latents), dim=-1)
        temporal, final_hidden = self.temporal_rnn(fused)
        context = self.attention(temporal)
        latent = self.projection(torch.cat((final_hidden[-1], context), dim=-1))
        predicted_content, predicted_context = self.image_decoder(latent)
        hidden0 = self.fused_to_h0(latent).view(self.text_layers, batch_size, -1)
        cell0 = self.fused_to_c0(latent).view(self.text_layers, batch_size, -1)
        text_logits, _, _ = self.text_decoder(target_text.squeeze(1)[:, :-1], hidden0, cell0)
        return predicted_content, predicted_context, text_logits, hidden0, cell0, image_latents, text_latents