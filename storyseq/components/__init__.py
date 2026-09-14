"""Components of the story-continuation predictor. Each module documents the interface a
replacement must implement; a level's build_model() is where a replacement is injected."""

from .attention import (ContentAttention, EntityPooling, FixedQueryAttention, LatentSimilarityAttention,
                        PixelCopyPath, SlotHead)
from .autoencoder import IMAGE_HW, N_SLOTS, ConvDecoder, ConvEncoder, VisualAutoencoder
from .losses import centred_cosine_loss, kl_divergence, latent_loss
from .predictor import DEFAULT_COMPONENTS, PredictorConfig, SequencePredictor
from .text import TextDecoder, TextEncoderLSTM

__all__ = ["ContentAttention", "EntityPooling", "FixedQueryAttention", "LatentSimilarityAttention", "PixelCopyPath",
           "SlotHead", "IMAGE_HW", "N_SLOTS", "ConvDecoder", "ConvEncoder", "VisualAutoencoder", "centred_cosine_loss",
           "kl_divergence", "latent_loss", "DEFAULT_COMPONENTS", "PredictorConfig", "SequencePredictor", "TextDecoder",
           "TextEncoderLSTM"]
