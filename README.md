# StoryReasoning architecture

The notebook architecture is organized into a small Python repository:

- `data.py`: GDI/CoT parsing and the sequence, text, and image datasets.
- `models.py`: text autoencoder, visual autoencoder, attention, and sequence predictor.
- `training.py`: initialization, checkpoints, and the main loss calculation.
- `train.py`: command-line training entry point.
- `tests/test_models.py`: CPU shape and forward-pass tests.

Install dependencies and run the focused checks:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Train against the Hugging Face dataset with:

```bash
python3 train.py --epochs 5 --batch-size 8
```

The original notebook remains available for exploration and visualization. The training script writes checkpoints under `checkpoints/` by default rather than requiring Google Drive.