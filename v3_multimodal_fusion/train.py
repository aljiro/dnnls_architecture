"""Train v3. From this directory:  python train.py [--epochs 15] [--no-plot] [--max-windows 200]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CONFIG, build_model  # noqa: E402
from storyseq.training import run  # noqa: E402

if __name__ == "__main__":
    run(CONFIG, build_model)
