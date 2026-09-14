"""v7: Names in the text, selection from the frames.

This file is the injection point. build_model() assembles the version from the shared components;
to try your own component, pass a replacement class in `components`, for example

    from storyseq.components import ContentAttention
    class MyAttention(ContentAttention):        # same interface: forward(sequence, h) -> weights [B, K]
        ...
    model = build_model(tok, components={"context_attention": MyAttention})

Component keys and their interfaces are documented in storyseq/components/predictor.py.
Every other knob is a field of CONFIG (see storyseq/training.py: TrainConfig and PredictorConfig).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storyseq.components import PredictorConfig  # noqa: E402
from storyseq.training import TrainConfig, build_predictor  # noqa: E402

CONFIG = TrainConfig(
    name="v7_names",
    model=PredictorConfig(latent_mode='mixture', attention='content', mix_attention='similarity', cond_with_embedding=True, annotations=True, per_slot_head=True, text_memory=True),
    text_lm_weights='pretrained/text_lm.pt',
    pretrained_lr_scale=0.1,
    text_lm_lr_scale=0.3,
    recon_weight=1.0,
    attn_weight=1.0,
)


def build_model(tok, components: dict | None = None):
    return build_predictor(CONFIG, tok, components)
