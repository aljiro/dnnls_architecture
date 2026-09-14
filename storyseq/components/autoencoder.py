"""Frame representation: a convolutional autoencoder (Level 1).

Interface for a replacement encoder: forward(images [B, 3, 60, 125] in [0, 1]) -> latent [B, latent_dim].
Interface for a replacement decoder: forward(latent [B, latent_dim]) -> images [B, 3, 60, 125] in [0, 1].
Any pair with these signatures can be injected into a level's build_model()."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IMAGE_HW = (60, 125)
N_SLOTS = 8            # character slots per story (see storyseq/precompute/annotations.py)


class ConvEncoder(nn.Module):
    """60x125 image -> latent. Four stride-2 convs to a 4x8 map, then a linear layer. width scales the channels."""

    def __init__(self, latent_dim: int = 256, width: int = 1) -> None:
        super().__init__()
        ch = [3] + [c * width for c in (32, 64, 128, 256)]
        layers: list[nn.Module] = []
        for i in range(4):
            k, p = (5, 2) if i == 0 else (3, 1)
            layers += [nn.Conv2d(ch[i], ch[i + 1], k, stride=2, padding=p), nn.GroupNorm(8, ch[i + 1]), nn.LeakyReLU(0.1)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(ch[-1] * 4 * 8, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.fc(self.conv(x).flatten(1)))


class ConvDecoder(nn.Module):
    """latent -> 60x125 image in [0, 1]."""

    def __init__(self, latent_dim: int = 256, width: int = 1) -> None:
        super().__init__()
        ch = [c * width for c in (256, 128, 64, 32)]
        self.c0 = ch[0]
        self.fc = nn.Linear(latent_dim, ch[0] * 4 * 8)
        layers: list[nn.Module] = []
        for i in range(3):
            layers += [nn.ConvTranspose2d(ch[i], ch[i + 1], 4, stride=2, padding=1), nn.GroupNorm(8, ch[i + 1]), nn.LeakyReLU(0.1)]
        layers += [nn.ConvTranspose2d(ch[-1], 3, 4, stride=2, padding=1), nn.Sigmoid()]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        x = self.deconv(self.fc(z).view(-1, self.c0, 4, 8))      # [B, 3, 64, 128]
        return x[:, :, :IMAGE_HW[0], :IMAGE_HW[1]]


class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 256, width: int = 1) -> None:
        super().__init__()
        self.encoder = ConvEncoder(latent_dim, width)
        self.decoder = ConvDecoder(latent_dim, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))


