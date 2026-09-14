"""v2: pretrain the text decoder as an unconditional language model. From this directory:
    python train.py [--epochs 8] [--extra groundcap]
Writes pretrained/text_lm.pt and prints held-out perplexity and a few samples."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storyseq.pretrain_text import main  # noqa: E402

if __name__ == "__main__":
    main()
