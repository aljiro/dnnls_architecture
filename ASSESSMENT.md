# Assessment of the StoryReasoning sequence predictor

Everything below was measured on this machine on 2026-09-13 with the scripts in
`diagnostics/` and `poc/`. Figures are in `diagnostics/out/` and `poc/out/`.

## Short answer

The model produces a blob because the training objective asks for one, not because the
data is too disjoint and not because of a noise pattern in the frames. Three independent
problems stack up, and none of them is a matter of scale:

1. **The blob is built into the loss.** The decoder returns the same tensor twice, so the
   "context" loss (MSE to the batch-mean image, weight 1) trains the only image output to be
   the mean image. On top of that, L1 in pixel space against a next frame that comes from a
   different shot is minimised by the per-pixel median image. On the real frames a constant
   median image scores L1 = 0.155, while copying the previous frame scores 0.174, so the loss
   itself prefers the blob over a plausible frame. A bigger model trained this way only
   estimates the median more precisely.
2. **The prediction is input-independent.** Retraining the original model on real frames with
   proper logging (`diagnostics/train_original_cached.py`, 10 epochs) gives a validation L1 of
   0.2586 against a constant-median-image floor of 0.2580, and a spread of the predicted
   image across the validation set of exactly 0.0000: every input produces the same picture.
   The collapse point is measurable: the 16-d fused vector that feeds the decoder has a
   spread of 0.0000 across inputs and 44 % dead units from epoch 1. Rerunning without the
   context loss and without equalisation changes nothing (validation L1 0.1537 vs floor
   0.1533, spread still 0.0000, 50 % dead). The decoder reaches the median image through its
   biases alone, so the gradient into the latent vanishes and the latent never learns.
3. **The refactored repo never ran.** `training.py` computes the context target with
   `keepdim=True`, which yields a 5-D tensor that cannot be expanded to the 4-D prediction, so
   `train.py` crashes on its first batch. The tests only check shapes.

The task *is* learnable on a laptop budget if it is reformulated as prediction in an
embedding space with frozen pretrained encoders. A 2M-parameter model trained for two minutes
on cached features picks the true next frame out of 2,974 held-out candidates 6.8 % of the
time (chance 0.03 %, copy-the-last-frame 2.3 %) and puts it in the top 10 40 % of the time.
The same figures for the description are 8.7 % and 58 %. Details in section 4.

## 1. What the notebook's own numbers were telling you

The loss print showed only the last batch of each epoch, so nothing could be read from it.
Reconstructed properly (see section 3), the relevant comparison is:

| quantity (60x125, L1, 0-1 scale)                         | value |
|----------------------------------------------------------|-------|
| constant per-pixel median image vs 5th frame, raw        | 0.155 |
| copy 4th frame as prediction of 5th, raw                 | 0.174 |
| constant mid-gray 0.5, raw                               | 0.321 |
| constant median image vs 5th frame, *equalised* frames   | 0.241 |
| notebook after 5 epochs (equalised frames, last batch)   | 0.215 |

And the original model retrained here for 10 epochs with per-epoch averages and a held-out
split of 711 stories (`diagnostics/out/original_eq.log`, `diagnostics/out/original_eq.png`):

| quantity (equalised, validation split)                   | value  |
|----------------------------------------------------------|--------|
| constant median image floor                              | 0.2580 |
| copy 4th frame floor                                     | 0.3259 |
| model, every epoch 1-10                                  | 0.2585-0.2590 |
| spread of the prediction across inputs (target: 0.299)   | 0.0000 |
| mean distance of the prediction from the median image    | 0.02   |

The model is a constant image. It matches the median floor from epoch 1 and never moves.
The training image loss is flat at 0.2585 while the text loss keeps falling (7.3 -> 4.9), so
the optimiser is busy with the text head and the image head has nothing to gain.

Two things follow. First, `TF.equalize` makes the target strictly harder (0.155 -> 0.241 for
the same trivial predictor) while destroying the colour statistics; it is applied to the
frames but not to the ROI crops, and not in the autoencoder dataset. Second, the notebook's
0.215 sits just under the constant-image floor for equalised frames, which is exactly what a
blob with a little input-dependent shading scores.

## 2. The position-pattern hypothesis

`diagnostics/check_data.py` and `diagnostics/position_test.py` test it on all 3,552 training
stories.

- Per-position mean images are the same brownish blob (`diagnostics/out/position_means.png`).
- Amplified deviations from the grand mean do differ per position, and part of that is
  reproducible: split the stories into two random halves and the deviation maps of the same
  position correlate at r = 0.12 on average (r = 0.30 for position 0) versus r = -0.03 for
  different positions (`diagnostics/out/position_test.png`).
- What is reproducible is a **global tint**: red-minus-blue rises monotonically from
  position 0 to 4 by 0.005 and brightness by 0.002, on a 0-1 scale. Stories open slightly
  cooler and darker. Positions 0 and 1 also keep a faint spatial component after the tint is
  removed, consistent with opening frames being wider establishing shots. This is a content
  trend of how the stories were cut, not a watermark.
- The cloudy texture that the eye reads as "a distinct pattern" changes between the two
  halves. It is sampling noise of averaging a few hundred natural images.
- The whole effect is about one gray level out of 255, against a between-frame brightness
  spread of 0.115, twenty times larger. A permutation test on 3,552 stories gives p = 0.05,
  i.e. barely detectable with the entire training set; a linear probe from pixels to position
  scores 20.6 % held-out (chance 20 %).

So the blob is not the position pattern. The blob is the median image, identical for every
position, and the model's latents were dead so it could not have exploited a one-gray-level
tint anyway.

## 3. What is wrong, item by item

### Objective

- `VisualDecoder.forward` returns `decode_image(x), decode_image(x)`: content and context
  predictions are the same tensor (verified: max |content - context| = 0). The context loss
  therefore trains the prediction to be the batch-mean image with weight 1.
- `validation()` displays `context_image`, i.e. the head that is *supposed* to be the blob.
- L1/MSE in pixel space on a multimodal target (next shot could be anything) is minimised by
  the median/mean image. This is the classic result behind adversarial and perceptual losses
  for video prediction (Mathieu, Couprie, LeCun 2016). No amount of capacity changes it.
- The "two pathways to disentangle the mean pattern" idea cannot work as wired: content and
  context backbones are concatenated and projected into one 16-d vector, and nothing in the
  loss distinguishes them.

### Capacity and optimisation

- Latent 16 for a 60x125x3 image, GRU hidden 16, text hidden 16. 78 % of the 507k trainable
  parameters sit in three 8192<->16 linear layers.
- `nn.ReLU()` on the latents (`Backbone.projection`, `SequencePredictor.projection`): 31 % of
  the 16 fused units are dead at initialisation, and on random inputs 62-69 % of backbone
  units die within 30 Adam steps at lr 1e-3 (`diagnostics/check_model.py`). On real frames
  the backbone units stay alive, but the fused pre-decoder vector is constant across inputs
  (spread 0.0000, 44-50 % dead units) from the first epoch, with or without the context loss
  and equalisation. Kaiming init on a 8192->16 layer plus ReLU plus lr 1e-3 is a textbook
  dying-ReLU setup, and a decoder that can output the median through its biases gives the
  latent no gradient to recover with.
- The visual encoder is never pretrained (the autoencoder cell is a To-Do) and receives
  gradient only through the prediction loss, i.e. through the blob.
- The text autoencoder is frozen at 16 hidden units. The predictor's only handle on the
  description is a 16-d initial state fed to a decoder that was trained to copy its input, not
  to generate from a fused state. A 16-d LSTM state cannot carry a 100-token description.
- The grounding losses are effectively off: with a single valid ROI in a batch the InfoNCE
  over a 1x1 logit matrix is exactly 0, ROI crops are not equalised while frames are, and both
  pass through a collapsed encoder.

### Data pipeline

- `TF.equalize` (see section 1). Remove it.
- `USE_COT_TEXT` appends `# # # characters # # # objects # # # setting` to every description:
  the helper strips the markdown tables and keeps their headings. 17 % of descriptions already
  exceed the 120-token cap (median 98, p90 131), so the appended text was mostly truncated
  away. The split repo dropped this, correctly.
- Only frames 0-4 of each story are used, but the median story has 13 frames (min 5, max 22).
  Sliding windows give 17,018 training windows instead of 2,842 for free.
- No validation metric, no baseline, last-batch loss printing. This is the reason the failure
  mode was invisible for so long: with a per-epoch validation L1 next to the constant-image
  floor, the blob would have been diagnosed in the first run.
- Loading is fine: 16 ms per item, under a minute per epoch single-threaded.

### The refactor

- `training.py:43`: `frames.mean(dim=(0, 1), keepdim=True).expand_as(predicted_context)`
  raises at the first batch. The notebook used `.mean(dim=[0, 1]).unsqueeze(0)`.
- `tests/test_models.py` only checks output shapes on random tensors; it cannot catch any of
  the above.

## 4. Is it a matter of scale?

Mostly no. Separate the three causes:

- The objective (blob by construction, L1 on a multimodal target) is scale-invariant.
- The dead latents and the 16-d bottleneck are design bugs, fixed by a normalisation instead
  of a ReLU and a wider latent, not by more data.
- From-scratch visual features on 2.8k stories is where scale would matter. The cheap way
  around it is frozen pretrained features, not a larger CNN.

`poc/` implements that reformulation to show it works within budget:

- `poc/precompute.py`: CLIP ViT-B/32 image embeddings and MiniLM sentence embeddings for
  every frame and description of every story, once. 3 min on the laptop GPU, 30 min on CPU.
  Cached as tensors, so every later experiment runs in seconds.
- `poc/train.py`: a 2-layer transformer (2.0M parameters) over the 4 input (image, text)
  embeddings predicts the next frame's image and text embeddings. Loss is InfoNCE: "pick the
  true next frame among the other targets in the batch plus the story's own other frames".
  30 epochs on 17,018 windows take 2 minutes on the GPU.

Test split, 2,974 windows, 626 stories never seen in training:

| metric                                       | chance | copy last frame | model (30 ep) |
|----------------------------------------------|--------|-----------------|---------------|
| image: true next frame ranked 1st of 2,974   | 0.03 % | 2.3 %           | **6.8 %**     |
| image: in top 10 of 2,974                    | 0.3 %  | 31.7 %          | **40.3 %**    |
| image: median rank of true frame             | 1487   | 65              | **19**        |
| image: ranked 1st among the story's own frames | 21 %  | 24.4 %          | **27.0 %**    |
| text: true next description ranked 1st       | 0.03 % | 3.1 %           | **8.7 %**     |
| text: in top 10                              | 0.3 %  | 54.2 %          | **58.0 %**    |
| text: ranked 1st among the story's own       | 21 %   | 27.2 %          | **30.8 %**    |

Reading it honestly:

- The model learns something real: 3x the copy-last baseline at rank 1 and a median rank of
  19 versus 65. Test metrics plateau after epoch 10 while training loss keeps falling, so this
  2M-parameter model already overfits 17k windows; more capacity would not help, more windows
  or regularisation would.
- The within-story numbers are only modestly above copy-last (27 % vs 24 %, chance 21 %).
  That is the part of your intuition which is right: *which of this story's remaining shots
  comes next* is close to unpredictable from four frames, because consecutive shots share a
  story, not appearance. What the model can do well is predict the *kind* of frame (scene,
  characters, lighting), which is what the global-pool numbers measure.
- `poc/out/predictions_both.png` shows the decisive picture. Column "decode(true emb)" is a
  small L1 pixel decoder applied to the *ground-truth* target embedding: it is still a blob
  with the right brightness and colour. So even a perfect predictor would give you a blob
  through an L1 pixel head. The "retrieved" column, the training frame nearest to the
  predicted embedding, shows the right characters, setting and lighting, often from the same
  film.
- Ablations (`--modality image` / `--modality text`) show that each modality predicts
  itself and the fusion adds nothing yet:

  | rank-1 of 2,974 test targets | image target | text target |
  |------------------------------|--------------|-------------|
  | image inputs only            | 7.5 %        | 1.5 %       |
  | text inputs only             | 1.6 %        | 9.0 %       |
  | both                         | 6.8 %        | 8.7 %       |

  Image-only is even slightly better at images than the fused model (top-10 45 % vs 40 %,
  median rank 14 vs 19), i.e. with 17k windows the joint model overfits a little more and
  learns no cross-modal transfer. If the course question is "does the story help predict the
  picture", this is the experiment to build on: the honest current answer is "not with this
  much data and a generic sentence embedding", and the lever is the text side (entity-aware
  embeddings, the chain-of-thought grounding), not the CNN.
- That last point is also a warning about the dataset: several stories come from the same
  movie and are split across train and test, so part of any model's skill on this benchmark
  is recognising the film. Report it if you use retrieval.

## 5. What to do

### If you keep the assessment's architecture (LSTM/CNN from scratch)

Minimal fixes, in order of impact:

1. Give the context head its own decoder or drop the context loss. Do not train the
   prediction to be a mean image.
2. Replace the `ReLU` on latents with `LayerNorm` (or nothing); use latent 128-256.
3. Pretrain the visual autoencoder with a reconstruction loss on all frames (the To-Do cell),
   without equalisation, and keep a small reconstruction term during sequence training.
4. Use all sliding windows, not frames 0-4.
5. Train the text autoencoder jointly with hidden 256+, and condition the decoder on the
   fused state at every step (concatenate it to the token embedding), not only through h0/c0.
6. Log per-epoch averages of each loss, a validation L1 next to the median-image and
   copy-last floors, and the spread of predictions across a validation batch (0 means "same
   blob for every input").
7. Expect: images that are still blurry but change with the input (scene brightness, colour,
   rough layout) and a validation L1 clearly below 0.155. Note that fix 1 on its own does
   nothing: the `--no-ctx --no-equalize` run in `diagnostics/out/original_raw_noctx.log` is
   still a constant image. Fixes 2 and 3 are what break the collapse; check with the
   prediction-spread number in the log, which must move off 0.0000 in the first epoch. Sharp images need a perceptual or
   adversarial loss and are out of a from-scratch budget.

### If you can change the formulation

Use `poc/` as the base. It already beats every baseline in minutes. The natural extensions:

- Pixel output: retrieval (nearest training frame to the predicted embedding) is honest and
  looks right; a generative decoder from CLIP embeddings is a research project, not a course
  assessment.
- Text output: fine-tune a small pretrained decoder (`distilgpt2`, 82M parameters, fits in
  8 GB with batch 8 and 128 tokens) with the predicted embedding projected to a prefix token,
  or a 256-512 hidden LSTM decoder trained jointly. Evaluate with perplexity on the test split
  plus the retrieval recall above.
- Grounding: the bounding-box ROIs from the chain-of-thought can be embedded with the same
  frozen CLIP and matched to entity names with the same InfoNCE, which is a cleaner version of
  what the notebook attempted.

## 6. Files added

| file | purpose |
|------|---------|
| `diagnostics/check_model.py` | identical decoder heads, parameter budget, dead-unit fractions |
| `diagnostics/check_data.py` | dataset stats, position means, L1 floors, loader timing |
| `diagnostics/position_test.py` | split-half correlation and permutation test for the position pattern |
| `diagnostics/train_original_cached.py` | the original model on cached frames with proper logging |
| `poc/precompute.py` | frozen CLIP + MiniLM features and 60x125 frames, cached to `poc/cache/` |
| `poc/train.py` | embedding-space next-frame predictor, baselines, metrics, figure |
| `venv/` | Python 3.12 environment; `./venv/bin/python` |
