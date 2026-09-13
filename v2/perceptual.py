"""Perceptual (feature-space) distance with a frozen VGG16, in the style of LPIPS.

Two 60x125 images are upsampled x2, normalised with ImageNet statistics, passed through VGG16
up to relu2_2, relu3_3 and relu4_3; each feature map is unit-normalised along channels and the
mean absolute difference is summed over the three layers. No learned layer weights (LPIPS
learns them from human judgements; here all three count equally).

Floors measured on test windows (v2/out/train_stageC_run2.log has the full set): with early layers
(relu1_2..relu3_3) the blob scores 0.133, copy-last 0.144 and the best-of-4 input 0.134, i.e. as
alignment-sensitive as pixel L1; with relu4_3+relu5_3 everything collapses to 0.03 and the
distance no longer separates a reconstruction from a blob. The middle set is the compromise.

Used as an extra training loss (`--perceptual-weight` in v2/train.py) and as a metric with the
same floors as pixel L1: the blob, copy-last, the best input, the reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import VGG16_Weights, vgg16

LAYERS = {8: "relu2_2", 15: "relu3_3", 22: "relu4_3"}


class VGGPerceptual(nn.Module):
    def __init__(self, upsample: int = 2) -> None:
        super().__init__()
        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[: max(LAYERS) + 1].eval()
        for p in features.parameters():
            p.requires_grad = False
        self.features = features
        self.upsample = upsample
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        return super().train(False)                                 # always in eval mode

    def _feats(self, x: Tensor) -> list[Tensor]:
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in LAYERS:
                out.append(F.normalize(x, dim=1))
        return out

    def forward(self, a: Tensor, b: Tensor, reduce: bool = True) -> Tensor:
        """Perceptual distance between image batches a and b in [0, 1]; per-image if reduce=False."""
        d = sum((fa - fb).abs().mean(dim=(1, 2, 3)) for fa, fb in zip(self._feats(a), self._feats(b)))
        return d.mean() if reduce else d
