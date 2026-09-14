"""v3: Multimodal fusion and a sequence model: the first version that takes frames and descriptions in and predicts both out.

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
    name="v3_multimodal_fusion",
    model=PredictorConfig(latent_mode='residual', attention='fixed'),
    
)


def build_model(tok, components: dict | None = None):
    return build_predictor(CONFIG, tok, components)
