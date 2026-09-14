"""v1: pretrain the visual autoencoder. From this directory:
    python train.py [--epochs 12] [--width 2 --extra groundcap]
Writes pretrained/visual_ae.pt (or visual_ae_w2.pt) and pretrained/reconstructions.png."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storyseq.pretrain_visual import main  # noqa: E402

if __name__ == "__main__":
    main()
