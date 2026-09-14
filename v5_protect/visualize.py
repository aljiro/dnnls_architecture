"""Render figures for v5 from its checkpoint. From this directory:  python visualize.py [--seeds 0 1 2]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CONFIG, build_model  # noqa: E402
from storyseq.visualize import run  # noqa: E402

if __name__ == "__main__":
    run(CONFIG, build_model)
