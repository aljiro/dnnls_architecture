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

## Assessment and follow-ups

`ASSESSMENT.md` explains why the original model only predicts a blob and what was measured.

- `diagnostics/`: checks of the original model and data (`check_model.py`, `check_data.py`,
  `position_test.py`, `train_original_cached.py`).
- `poc/`: next-frame prediction in embedding space with frozen CLIP + MiniLM features.
  Run `poc/precompute.py` once (writes `poc/cache/`, needed by `v2/` as well), then `poc/train.py`.
- `v2/`: the original pipeline with the fixes applied, one script per component:

```bash
python poc/precompute.py            # cache frames and text embeddings (once)
python v2/pretrain_visual.py        # pretrain the visual autoencoder
python v2/train.py                  # sequence predictor with MiniLM text encoder
python v2/train.py --text-encoder lstm --freeze-image-encoder
python v2/probe.py                  # does the decoder use its condition?
python v2/precompute_groundcap.py   # scaling stage: GroundCap frames + captions for pretraining
python v2/pretrain_visual.py --width 2 --extra groundcap
python v2/pretrain_text.py --extra groundcap
python v2/train.py --stage D --copy-path --recon-weight 1 --attn-weight 1 --kl-weight 1e-2 \
    --clip-input --entity-features clip --ae-width 2 --ae-weights v2/out/visual_ae_w2.pt
python v2/visualize.py --tag _frozen # prediction figure from a checkpoint
```

Use `./venv/bin/python` (Python 3.12 venv in the repo).

The original notebook remains available for exploration and visualization. The training script writes checkpoints under `checkpoints/` by default rather than requiring Google Drive.