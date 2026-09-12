"""Losses, checkpointing, and training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "loss": loss}, path)


def load_checkpoint(path: str | Path, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def sequence_loss(outputs: tuple[torch.Tensor, ...], batch: tuple[torch.Tensor, ...],
                  pad_token_id: int, image_encoder: nn.Module | None = None,
                  temperature: float = 0.07) -> dict[str, torch.Tensor]:
    predicted_image, predicted_context, text_logits, _, _, image_latents, text_latents = outputs
    frames, _, target_image, target_text, roi1, roi2, roi_valid, roi_frame, _ = batch
    losses = {
        "image": F.l1_loss(predicted_image, target_image),
        "context": F.mse_loss(predicted_context, frames.mean(dim=(0, 1), keepdim=True).expand_as(predicted_context)),
        "text": F.cross_entropy(text_logits.flatten(0, 1), target_text.squeeze(1)[:, 1:].flatten(), ignore_index=pad_token_id),
    }
    zero = predicted_image.new_zeros(())
    losses.update({"reid": zero, "grounding": zero, "contrastive": zero})
    mask = roi_valid.bool()
    if mask.any() and image_encoder is not None:
        roi1_latents = image_encoder(roi1[mask])
        roi2_latents = image_encoder(roi2[mask])
        frame_indices = roi_frame[mask].clamp(0, text_latents.size(1) - 1)
        matched_text = text_latents[mask].gather(1, frame_indices[:, None, None].expand(-1, 1, text_latents.size(-1))).squeeze(1)
        losses["reid"] = F.mse_loss(roi1_latents, roi2_latents)
        losses["grounding"] = F.mse_loss(roi1_latents, matched_text)
        similarity = F.normalize(roi1_latents, dim=-1) @ F.normalize(matched_text, dim=-1).T / temperature
        losses["contrastive"] = F.cross_entropy(similarity, torch.arange(similarity.size(0), device=similarity.device))
    losses["total"] = losses["image"] + losses["context"] + losses["text"] + 0.1 * (losses["reid"] + losses["grounding"] + losses["contrastive"])
    return losses