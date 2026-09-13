# Assessment of the StoryReasoning sequence predictor

Everything below was measured on this machine on 2026-09-13 with the scripts in
`diagnostics/` and `poc/`. Figures are in `diagnostics/out/` and `poc/out/`.

## Short answer

The model produces a blob because the training objective asks for one, not because the
data is too disjoint and not because of a noise pattern in the frames. Three independent
problems stack up, and none of them is a matter of scale:

1. **The blob is built into the loss.** The context head was added later as an attempt to
   absorb the mean image, and it is not part of the intended architecture, but as wired the
   decoder returns the same tensor twice, so the "context" loss (MSE to the batch-mean image,
   weight 1) trains the only image output to be the mean image. Independently of that, L1 in
   pixel space against a next frame that comes from a
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
- `poc/out/predictions_both.png` separates the two sources of blur. Column "decode(true emb)"
  is a small L1 pixel decoder (the notebook's `VisualDecoder` with a 512-d latent) applied to
  the *ground-truth* target embedding: a blurry, input-dependent image with the right colour,
  brightness and coarse layout (a figure on the left, a bright background on the right) but
  no sharp detail. Its training L1 of 0.116 is well below the 0.153 of a constant median
  image, so this is not a blob; it is what L1 gives when the embedding does not pin down the
  fine structure. "decode(pred emb)" looks much the same, so the prediction is about as good
  as the truth at the resolution an L1 head can express. Sharper output needs a different
  pixel loss (perceptual, adversarial, or a diffusion decoder), not a better predictor. The
  "retrieved" column, the training frame nearest to the predicted embedding, shows the right
  characters, setting and lighting, often from the same film.
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

## 7. v2: the original pipeline with the fixes applied

`v2/` keeps the pipeline shape of the notebook (from-scratch CNN autoencoder, text encoder,
GRU + attention fusion, image decoder, LSTM text decoder) and applies section 5. It was
built to teach the components as parts of one system, so each component has its own
script and its own numbers.

| component | what changed | evidence |
|-----------|--------------|----------|
| visual autoencoder (`v2/pretrain_visual.py`) | latent 16 -> 256, LayerNorm instead of ReLU on the latent, pretrained with L1 on all 31k training frames, no equalisation | val L1 0.040 vs 0.152 constant-image floor after 12 epochs (1 min on the GPU); `v2/out/reconstructions.png` |
| text encoder | frozen MiniLM sentence vectors by default; `--text-encoder lstm` trains a bidirectional LSTM from scratch for comparison | MiniLM gives 3x the text retrieval of the LSTM (37 % vs 13 % top-10) |
| fusion | fuse -> GRU(256) -> attention -> LayerNorm latent, no context head | prediction spread 0.10 (target 0.20; original model 0.00) |
| image head | pretrained decoder, pixel L1 plus a cosine target to the encoder latent of the true next frame | test L1 0.130-0.132 vs floors 0.151 (median) and 0.170 (copy last); L1 vs a random other target 0.166, so predictions are input-specific |
| text head | LSTM conditioned on the latent at every step; second pass added word dropout 0.3 and a head predicting the MiniLM vector of the next description | see below |
| training data | all sliding windows, split by story | 13.6k train / 3.4k val / 3.0k test windows |
| logging | per-epoch averages, floors, spread, retrieval, shuffled-condition CE | `v2/out/train_*.log` |

**First pass** (`v2/out/train_*_pass1.log`): images worked, text did not. The text decoder
reached perplexity 21 but its cross-entropy was 3.061 with the true latent and 3.070 with a
shuffled one (`v2/probe.py`), i.e. it had become an unconditional language model, and every
window greedy-decoded to the same sentence. This is the standard "decoder ignores the
condition" failure of conditional LSTMs.

**Second pass** (word dropout + text-embedding target), test split:

| metric | MiniLM text | LSTM text |
|--------|-------------|-----------|
| image L1 (floors 0.151 / 0.170) | 0.132 | 0.132 |
| prediction spread (target 0.204) | 0.101 | 0.096 |
| text-embedding retrieval, top-1 / top-10 of 2,974 | 5.9 % / 37.5 % | 1.9 % / 13.1 % |
| text CE, true latent vs shuffled latent | 3.28 vs 3.38 | 3.23 vs 3.34 |
| text perplexity | 26.6 | 25.2 |
| image-latent retrieval, top-10 | 0.9 % | 4.4 % |

The decoder now uses its condition (a 0.10-nat gap instead of 0.01), the generated text
varies with the window and tracks the scene (`v2/out/predictions_minilm.png`), and the
latent carries the next description well enough for 37 % top-10 retrieval, close to the
CLIP-based proof of concept's 58 %. The price of word dropout is a higher perplexity.

One new failure appeared and is worth teaching: image-latent retrieval collapsed from 7.5 %
(first pass) to 0.9 % while the training latent loss fell to 0.035. The image encoder is
fine-tuned, and a cosine target to a detached copy of its own output can be satisfied by
shrinking the latent space so every frame looks alike. The target must come from a fixed
encoder: `--freeze-image-encoder` (results below), or an EMA copy.

**Frozen image encoder** (`--freeze-image-encoder`, MiniLM text), test split: image L1
0.133, spread 0.112, image-latent top-10 3.4 % (training latent loss 0.33, so the target
is no longer satisfied trivially), text top-10 16 %, text CE 3.23 vs 3.34 shuffled. The
image latent recovers part of its discriminative power, but text retrieval drops from 37 %
to 16 %. That is the third lesson from this pipeline: one 256-d vector `z` feeds three heads
(pixels, image latent, text embedding), and with a fixed image target the heads compete for
it. The second pass "won" text retrieval by letting the encoder collapse, which made the image
target free. The clean fix is to stop sharing a single bottleneck: give the text and image
heads their own projections from the GRU state, or lower `--latent-weight`, and compare the
three runs on the same table. The flags are in `v2/train.py`; each run is 7-9 minutes.

Summary of the three v2 runs with the MiniLM text encoder (test split):

| run | image L1 | spread | image-latent top-10 | text top-10 | text CE true / shuffled |
|-----|----------|--------|---------------------|-------------|-------------------------|
| pass 1 (no word dropout, no text target) | 0.130 | 0.100 | 7.5 % | n/a | 3.06 / 3.07 |
| pass 2 (word dropout 0.3 + text target)  | 0.132 | 0.101 | 0.9 % | 37.5 % | 3.28 / 3.38 |
| pass 2 + frozen image encoder            | 0.133 | 0.112 | 3.4 % | 16.2 % | 3.23 / 3.34 |
| original notebook model (section 1)      | at floor | 0.000 | n/a | n/a | n/a |

## 8. How far pixel L1 can go on this data

`v2/analyze_predictability.py` splits the 2,974 test windows by whether the shot continues
(copy-last L1 < 0.06, true for 5.9 % of windows) and compares per-window L1:

| L1 vs the true next frame | all | shot continues (6 %) | shot cuts (94 %) |
|---|---|---|---|
| blob (median image) | 0.151 | 0.111 | 0.154 |
| copy last frame | 0.170 | 0.041 | 0.178 |
| oracle: best of blob or copy per window | 0.137 | 0.041 | 0.142 |
| v2 model (MiniLM, pass 2) | 0.132 | 0.065 | 0.137 |
| autoencoder given the true target | 0.039 | 0.021 | 0.041 |

The decoder can draw the target at 0.039 when handed the true latent, so the gap to 0.132
is not capacity. When the shot cuts, a real frame from the same story scores worse under L1
than the blob (0.178 vs 0.154): pixel L1 penalises any sharp guess that is not pixel-aligned
more than it penalises a blob, so reconstruction-quality predictions are unreachable under
L1 in 94 % of windows for any architecture. The only predictable pixel content is the
continuing-shot case, where the model is currently worse than copying; a skip path from the
last frame with a learned gate would close that, with a ceiling near the oracle row. Anything
beyond needs a loss that rewards plausibility instead of alignment (perceptual, adversarial,
diffusion) or the retrieval formulation of section 4.

The text head collapses to the corpus mode under greedy decoding for the analogous reason:
the conditioning adds 0.10 nats per token against a strong style prior. Judge it by the
cross-entropy gap and retrieval, and sample (temperature or nucleus, repetition penalty) to
see the conditional signal in generated strings.

**Which input frame to copy.** The last frame is not the usual source. Across the test
windows the closest input to the 5th frame is frame 1 in 20 %, frame 2 in 20 %, frame 3 in
33 % and frame 4 in 27 % of cases, the A-B-A-B rhythm of dialogue editing. Near-copies (some
input within L1 0.06) exist in 15 % of windows, not 6 %, and the source is frame 3 in 44 % of
them. Copying the best of the four inputs scores 0.130 (vs 0.170 for the last frame), and the
best of blob or best-of-4 scores 0.112. A copy path should therefore attend over all four
inputs (a softmax over frames, or per pixel) and gate the blend against the generated image;
that pattern is learnable from the inputs alone, since the alternation is visible in them.

**Why the generated text and images look alike across windows (checked, not a bug).**
Batched generation equals one-at-a-time generation and reversing the batch reverses the
outputs. The phrases are corpus modes: 34 % of the 44k training descriptions contain "the
tension", 14 % "palpable", 2.4 % open with "the tension reached a". The underlying cause of
the sameness is that the predicted latents are nearly identical: pairwise cosine 0.98
between windows. The 256-d vector feeding both decoders is a large constant plus a small
input-dependent part; a linear head reads the small part (37 % text retrieval), but the LSTM
decoder and the image decoder mostly see the constant. Next fix: remove the constant
component before the decoders (batch normalisation without affine on `z`, or running-mean
subtraction), or condition the text decoder on the predicted text embedding, which the cosine
loss makes discriminative.

## 9. Roadmap for v2 (agreed 2026-09-13)

Ordered by expected impact per line of code. Each step has a metric that shows whether it
worked, independent of pixel L1. The pixel copy path (section 8) is deferred: if step 2
works, it should be unnecessary.

### A. Fix the latent (small edits to `v2/models.py`)

1. **Remove the constant component of `z`.** Batch normalisation without affine (or a running
   mean subtracted) on the vector that feeds the decoders. Check: pairwise cosine between
   predicted latents on a test batch drops well below the current 0.98; prediction spread
   rises from 0.10; generated openings stop repeating.
2. **Content-dependent attention and latent mixture.** Query from the final GRU state, keys
   from the GRU outputs; `z = g * sum_i alpha_i zv_i + (1 - g) * delta`, with the mixture over
   the image-encoder latents `zv_i`, the residual `delta` from `(h, context)`, and a gate `g`
   from `h`. Check: on windows where an input is a near-copy of the target, image L1 falls
   from 0.065 toward 0.04; `alpha` concentrates on frame 3 in alternating A-B-A-B windows;
   `g` is high on continuing shots and low on cuts.
3. **Separate heads.** Own projections from `(h, context)` for the text-embedding head and
   the image latent, so they stop competing for one vector (section 7, frozen run). Check:
   image-latent top-10 and text top-10 both stay high in the same run (today it is one or
   the other: 7.5 % / n/a, 0.9 % / 37 %, 3.4 % / 16 %).

### B. Fix the text head (`v2/models.py`, `v2/visualize.py`)

4. **Judge and decode properly.** Report the true-vs-shuffled cross-entropy gap and text
   retrieval as the text metrics; generate with nucleus sampling and a repetition penalty
   instead of greedy search, which returns the corpus mode ("the tension reached a ...",
   34 % of descriptions contain "the tension").
5. **Condition on the predicted text embedding** (the cosine-trained head) in addition to
   `z`; later, cross-attention over the four input descriptions instead of a single vector.

### C. Use the annotations (new inputs and targets; parser exists in `data.py`)

The chain-of-thought gives, per frame: characters (persistent ID, name, description,
emotions, actions, narrative function, bounding box), objects (same fields), and setting
(location, lighting, time of day, mood). The story text is grounded: each mention carries
the entity ID.

6. **Setting fields as conditioning for the image decoder** (location type, lighting, time,
   mood, as small embeddings). Cheap, and aimed at the only pixel properties that are
   predictable across a cut: brightness and tint. Check: image L1 on cut windows below the
   current 0.137.
7. **Entity crops as inputs.** Encode each character box with the pretrained image encoder
   and give the fusion a set of entity tokens per frame alongside the frame latent
   (attention over entities, not only over frames). Check: `alpha` on entities, image-latent
   and text retrieval up.
8. **Structured next-frame target: which characters appear next, and which setting.** A
   multi-label head over the story's character IDs plus a setting/mood head, trained with
   cross-entropy. This is the most predictable aspect of the next frame and gives the GRU and
   attention a genuine sequence task with an honest metric (precision/recall of the
   characters in frame 5, vs "same as frame 4" and "all seen so far" baselines). Largest
   impact, largest build; depends on 7.
9. **Grounded names in the text.** Condition the text decoder on the names of the predicted
   next-frame characters (from 8), so descriptions say "Mrs. Patel" instead of "john and
   john". Check: name accuracy against the target description's grounded mentions.

### Deferred

- Pixel copy path over the four inputs (section 8): add only if step 2 does not reach the
  0.04 level on near-copy windows.
- Plausibility losses for the image (perceptual, adversarial, diffusion decoder) or the
  retrieval formulation of section 4: the only way past the L1 ceiling on cut windows.

## 10. Stages A, B, C: results (2026-09-13)

`v2/train.py --stage {0,A,B,C}`; MiniLM text encoder, same story split, 15 epochs each
(8, 8 and 12 minutes on the laptop GPU). Logs `v2/out/train_stage*.log`, figures
`v2/out/predictions_stage*_minilm.png` (B and C use nucleus sampling for the text).

Before these runs a measurement problem had to be fixed: the encoder's latents share a large
mean vector (raw pairwise cosine 0.31 between frames; 0.38 for MiniLM vectors), so a raw
cosine loss is nearly blind to the informative part and can be satisfied by shrinking every
latent toward the mean, which is exactly what the fine-tuned encoder did in section 7 (latent
loss 0.005 within two epochs, "latent cosine 0.98"). Fix, applied to all stages: the latent
target comes from a frozen copy of the pretrained encoder, and cosine losses and retrieval
metrics are computed after subtracting the batch mean of the targets (`centred_cosine_loss`).

| test split, 2,974 windows | stage 0 (sec. 7 pass 2) | A | B | C |
|---|---|---|---|---|
| image L1 (floors 0.151 median / 0.170 copy) | 0.132 | 0.135 | 0.134 | 0.133 |
| image L1 on near-copy windows (15 %) | 0.065 | 0.077 | 0.077 | 0.075 |
| prediction spread (target 0.204) | 0.101 | 0.124 | 0.125 | 0.123 |
| latent cosine between windows, centred | (0.98 raw) | 0.033 | 0.033 | 0.028 |
| image-latent retrieval, top-10 | 0.9 % | 6.1 % | 6.8 % | 7.6 % |
| text-embedding retrieval, top-10 | 37.5 %* | 49.2 % | 48.0 % | 43.5 % |
| text CE, true / shuffled condition | 3.28 / 3.38 | 3.29 / 3.48 | 3.28 / 3.48 | 3.30 / 3.48 |
| attention argmax = closest input (chance 25 %) | n/a | 29 % | 29 % | 30 % |
| gate on near-copy / cut windows | n/a | 0.85 / 0.78 | 0.86 / 0.79 | 0.92 / 0.88 |
| next-frame character F1 (best threshold) | n/a | n/a | n/a | 0.41 |

\* raw-cosine retrieval; the other columns are centred.

**Stage A is the step that mattered.** The constant component is gone (centred cosine 0.03),
both retrievals rise in the same run (the heads no longer compete), the text decoder's
dependence on its condition doubles (gap 0.19 vs 0.10 nats), and the prediction spread rises
from 0.10 to 0.12. The one regression is on the near-copy windows, 0.065 -> 0.077, and it
tells you what did not happen: the attention learned the *position prior* (mean weights
[0.12, 0.26, 0.36, 0.26], frame 3 first, the A-B-A-B rhythm) but not the *per-window*
selection (its argmax hits the closest input 29 % of the time, chance 25 %). With near-uniform
weights the mixture is an average of four frames, which decodes to a blend, and the gate stays
high everywhere (0.85 / 0.78) because that blend still beats the residual under L1. The
mechanism is wired but untrained: nothing in the loss says which frame to pick, and the pixel
gradient from 15 % of windows is too weak to teach it.

**Stage B changed nothing measurable.** Conditioning on the predicted text embedding leaves
the CE gap at 0.20. Nucleus sampling replaces the corpus-mode sentence with varied, mostly
readable descriptions that carry names and settings ("maria sat near the door", "sarah
asked", "the reader"), with the grammar slips of a small LSTM sampled at temperature 0.8.

**Stage C: the structured heads learned the prior, not the pattern.** The next-frame
character head reaches F1 0.41 at its best threshold (0.29 at 0.5: the head is
under-confident because only 1.6 of 6.9 slots are positive per frame), equal to the "every
character seen so far" baseline (0.41) and below "present in at least 2 of the 4 inputs"
(0.47) and "same as frame 3" (0.44). Two facts from the annotations are worth keeping: frame
3's characters predict frame 5's better than frame 4's (0.44 vs 0.38), the editing rhythm
again; and the per-character history is the signal, which the current head, fed only the
pooled sequence state, cannot exploit slot by slot. The setting conditioning has no visible
effect on L1, and the extra losses cost text retrieval (49 % -> 43 %).

**Next fixes, in order:**
1. Teach the attention: the closest input is known at training time (per-input pixel
   distance to the target), so add a cross-entropy from `alpha` to it. This is the shot
   pattern as an explicit target; check argmax accuracy well above 29 % and near-copy L1
   back under 0.065.
2. Per-slot character head: for each character slot, a small classifier on that slot's
   presence pattern over the four inputs plus its entity token, instead of one head on the
   pooled state. The 0.47 baseline is the floor it must beat. Calibrate the threshold on
   validation.
3. Drop the setting conditioning of the image decoder (no effect) and keep the setting
   embedding as an input only.

### 10b. Supervised attention and early stopping (stage A + `--attn-weight 1 --gate-weight 0.5`)

The closest input is known at training time (per-input pixel L1 to the target), so the
attention is trained toward it on windows where that input is within L1 0.10, and the gate
toward "such an input exists". The checkpoint is chosen by validation image L1.

| test split | A | A + supervised attention, last epoch | same, best-val checkpoint (epoch 5) |
|---|---|---|---|
| image L1 | 0.135 | 0.132 | **0.130** |
| L1 on near-copy windows | 0.077 | 0.075 | 0.075 |
| attention argmax = closest input, near-copy windows | 29 % | 44 % | 52 % |
| gate on near-copy / cut windows | 0.85 / 0.78 | 0.63 / 0.32 | 0.67 / 0.38 |
| image-latent retrieval top-10 | 6.1 % | 6.6 % | 5.8 % |
| text retrieval top-10 | 49 % | 25 % | 16 % |
| text CE true / shuffled | 3.29 / 3.48 | 3.48 / 3.59 | 3.94 / 3.98 |

The mechanism now works as designed: the attention finds the closest input half the time
instead of a quarter, and the gate separates continuing shots from cuts. The pixel gain is
real but small, 0.135 -> 0.130 overall and 0.077 -> 0.075 on the near-copy windows, for two
reasons. First, the latent route has its own floor on those windows: decoding the best
input's latent reproduces that input at 0.04, and that input differs from the target by up
to 0.06, so about 0.06 is the best the mixture can do, against 0.041 for pixel copying.
Second, the weights are soft (0.36 on the best input on average) and the gate still mixes in
the residual. The cost was large: text retrieval fell from 49 % to 16-25 % and the decoder's
dependence on its condition shrank, because the attention query and the gate are computed
from the same GRU state the text heads rely on, and the supervision reshaped it. If this
path is pursued, the selection should be computed from the frame latents directly (pairwise
similarities between the inputs, which is where the A-B-A-B pattern lives) rather than from
the shared sequence state, and the remaining gap on near-copy windows is the pixel copy path.
For the L1 version of the architecture, 0.130 with the selection working is close to what
this data allows without a pixel copy path; the 0.112 ceiling assumes perfect selection and
perfect copying.

### 10c. Stage C for 30 epochs

Same configuration as stage C, 30 epochs instead of 15 (`v2/out/train_stageC_e30.log`).

| test split | C, 15 epochs | C, 30 epochs (last) | C, 30 epochs (best val image L1 = epoch 1) |
|---|---|---|---|
| image L1 | 0.133 | 0.140 | 0.130 |
| prediction spread | 0.123 | 0.138 | 0.100 |
| image-latent retrieval top-10 | 7.6 % | 6.3 % | 4.3 % |
| text retrieval top-10 | 43.5 % | 39.6 % | 1.2 % |
| text CE, true / shuffled | 3.30 / 3.48 | 3.00 / 3.29 | 5.91 / 5.91 |
| character F1 (threshold 0.5) | 0.29 | 0.29 | 0.00 |

Validation image L1 is lowest at epoch 1 (0.1288) and rises monotonically to 0.1396 at epoch
30: the pixel head overfits from the first epoch. Retrieval peaks around epochs 12-18 and
then declines; only the text decoder keeps improving (its condition gap widens to 0.29 nats),
at the cost of text retrieval. Longer training therefore helps nothing but the language model,
and the checkpoint with the best image L1 is a model whose other heads have not trained yet,
so "best validation image L1" is the wrong selection rule for a multi-head model: select on
retrieval, or stop each head separately. Under L1 with 13.6k windows, 12-15 epochs is the
right budget.

### 10d. Steps 1-3: discriminative learning rates, retrieval-based selection, pretrained text decoder

`v2/pretrain_text.py` trains the text decoder as an unconditional language model on all 31k
cached descriptions (8 epochs, 6 minutes; test perplexity 17.9). `v2/train.py` then loads its
embedding, LSTM and output layers, trains them at 0.3x the learning rate, the pretrained image
encoder and decoder at 0.1x, selects the checkpoint on validation retrieval, and logs the
reconstruction L1 of the fine-tuned autoencoder (`recon_L1`, pretrained reference 0.040).
Tag `_s123`, 15 epochs.

| test split | C | C + steps 1-3 | A | A + steps 1-3 |
|---|---|---|---|---|
| image L1 | 0.133 | **0.131** | 0.135 | **0.132** |
| L1 on near-copy windows | 0.075 | 0.073 | 0.077 | 0.076 |
| image-latent retrieval top-10 | 7.6 % | 7.8 % | 6.1 % | 6.9 % |
| text retrieval top-10 | 43.5 % | **45.7 %** | 49.2 % | **52.0 %** |
| text CE, true / shuffled | 3.30 / 3.48 | **2.75** / 2.81 | 3.29 / 3.48 | **2.75** / 2.81 |
| character F1 (0.5) | 0.29 | 0.28 | n/a | n/a |
| reconstruction L1 of the fine-tuned autoencoder | n/a | 0.080 | n/a | 0.081 |

Every metric improved or held, and the pretrained decoder is the clearest gain: text
cross-entropy 2.75 (perplexity 15.6) instead of 3.30, and the sampled descriptions read as
sentences. Two things to read carefully. The decoder's dependence on its condition shrank
(gap 0.06 nats vs 0.18): the pretrained language model is a stronger prior, so the same
conditioning signal moves it less, while the *latent* carries more (text retrieval up). And the
reconstruction monitor shows that a 10x lower learning rate does not stop the autoencoder from
drifting: reconstruction of the target through the fine-tuned encoder and decoder is 0.077
after the first epoch and 0.080 at the end, twice the pretrained 0.040. The decoder is being
trained to draw smooth predictions and forgets how to draw sharp reconstructions. The fix is
the one from section 5, item 3: keep a reconstruction loss term on the target frame during
sequence training (or freeze the decoder), so the component keeps the skill it was pretrained
for. That is the next single change.

### 10e. Run 1 (reconstruction term + pixel copy path) and run 2 (+ VGG perceptual loss)

Both on stage C with steps 1-3 (`_run1`, `_run2`). The copy path blends the four inputs in
pixel space with the attention weights and a per-pixel gate chooses between the blend and the
generated image. The perceptual loss is the LPIPS-style distance of `v2/perceptual.py`
(VGG16 relu2_2 / relu3_3 / relu4_3, weight 1.0), also reported as a metric with floors.

| test split | C + steps 1-3 | run 1 | run 2 |
|---|---|---|---|
| image L1 | 0.131 | 0.132 | **0.131** |
| L1 on near-copy windows | 0.073 | **0.071** | 0.074 |
| reconstruction L1 of the fine-tuned autoencoder | 0.080 | **0.041** | **0.041** |
| copy gate, near-copy / cut windows | n/a | 0.28 / 0.16 | 0.01 / 0.00 |
| perceptual distance: model / blob / copy-last / reconstruction | n/a | n/a | 0.083 / 0.086 / 0.098 / 0.077 |
| image-latent retrieval top-10 | 7.8 % | 8.7 % | **8.9 %** |
| text retrieval top-10 | 45.7 % | 45.0 % | 43.6 % |
| text CE, true / shuffled | 2.75 / 2.81 | 2.75 / 2.81 | 2.75 / 2.81 |

**The reconstruction term works exactly as intended**: the fine-tuned autoencoder stays at
0.041 instead of drifting to 0.080, at no cost elsewhere. It should stay on.

**The copy path does what the ceiling analysis said**: near-copy windows improve a little
(0.073 -> 0.071), the per-pixel gate opens more on near-copy windows than on cuts (0.28 vs
0.16), and overall L1 does not move. In the figure the gate shows as faint ghosted structure
from the inputs on cut windows, which is what a pixel blend does when no input is right.

**The perceptual loss does not change the picture.** The model's feature distance beats the
blob's by a small margin (0.083 vs 0.086) and copy-last by more (0.098), pixel L1 stays at its
best value, and the images look as smooth as before. The copy gate closes (0.01), because in
feature space a misaligned copy costs more than a blur. Calibration explains why: no VGG layer
set ranks a real but different shot above the blob (early layers: blob 0.133, best input
0.134; deep layers: everything at 0.03), so a feature-space loss still averages over the
plausible next shots on the 85 % of windows that cut. Sharper output on those windows needs a
loss that scores samples rather than expectations (adversarial or diffusion), or the retrieval
formulation. Under the objectives available here, run 2 is the final state of the L1-family
image head: 0.131 against floors of 0.151 and 0.170, with the autoencoder intact.

### 10f. Stage D: variational latent, decoupled attention, per-slot head, cross-attention decoder

Built on run 1 (reconstruction term, copy path). Four changes (`v2/models.py`, stage "D"):
(1) the residual path is a conditional Gaussian: a prior from the sequence state, a posterior
that also sees the frozen target latent, KL(q || p) with a 3-epoch warm-up, posterior samples
during training, prior mean or prior samples at test; (2) the mixture / copy-path attention is
computed from the similarities between the input frame latents, independent of the GRU, and
supervised toward the closest input; (3) one logit per character slot from that slot's own
presence history, pooled entity token and the sequence state; (4) the text decoder attends
over a memory of the four input description vectors, the predicted text embedding and the
names of the characters predicted present (true names during training; names come from the
story's grounded mentions). Two runs, KL weight 1e-3 and 1e-2.

| test split | run 1 (deterministic) | D, KL 1e-3 | D, KL 1e-2 |
|---|---|---|---|
| image L1, prior mean | 0.132 | 0.134 | 0.132 |
| image L1, best of 5 prior samples | n/a | 0.136 | **0.125** |
| image L1, average prior sample (blob 0.151) | n/a | 0.158 | 0.146 |
| image L1, posterior sample (sees the target) | n/a | 0.059 | 0.096 |
| sample diversity (mean L1 between samples) | n/a | 0.123 | 0.094 |
| KL, nats | n/a | 124 | 10.8 |
| L1 on near-copy windows | 0.071 | 0.077 | 0.073 |
| attention picks the closest input, near-copy | 36 % | 44 % | 44 % |
| text retrieval top-10 | 45.0 % | **49.6 %** | 47.8 % |
| text CE, true / shuffled condition | 2.75 / 2.81 | 2.72 / 3.01 | 2.72 / 2.99 |
| character F1 at 0.3 (baselines 0.38 / 0.41 / 0.47) | 0.41 | **0.45** | 0.44 |
| image-latent retrieval top-10 | 8.7 % | 2.3 % | 3.5 % |

**The variational latent is the first thing to move the image past the deterministic ceiling.**
At KL 1e-3 the posterior smuggles the target (0.059 with 124 nats) and the prior never learns
to match it: samples are diverse and structured but their average scores below the blob. At
KL 1e-2 the channel is constrained (10.8 nats), the prior mean matches run 1, the average
prior sample beats the blob (0.146 vs 0.151), and the best of five samples scores 0.125, below
every deterministic model. The samples themselves are the visible change: blurred but
composed images (figures against a lit background, a face-shaped warm region, a dark room
with a lighter figure) instead of a smooth field, because the decoder is allowed to commit
to one plausible next shot. Sharpness is still bounded by the L1-trained decoder.

**The other three changes worked as designed.** Cross-attention with names quadruples the
decoder's dependence on its condition (gap 0.27-0.29 nats vs 0.06) and the generated text uses
the story's names; the per-slot head beats "all seen so far" (0.45 vs 0.41) and sits 0.02 below
"in at least two inputs"; the decoupled supervised attention reaches 44 % without touching
text retrieval, which the coupled version could not (section 10b). The prior mean is a poor
point estimate for image-latent retrieval (3.5 %): with a stochastic latent, retrieval should
use samples, or the posterior mean of the training set.

The figures show the trade-off the KL weight sets: at 1e-3
(`predictions_stageD_minilm.png`) the samples are visibly composed and different from each
other but not tied to the inputs; at 1e-2 (`predictions_stageD_minilm_kl1e-2.png`) they are
tied to the inputs, score better, and sit visibly closer to the mean. A value between the two
(3e-3), or free bits, is the natural next test.

Costs and open items: KL weight is a real hyper-parameter (1e-3 and 1e-2 differ in kind, not
degree); the evaluation with 5 samples per window doubles validation time; and the deepest
lesson of the run, that sampling and not a new distance is what turns averages into pictures,
is the argument for an adversarial or diffusion decoder if sharper samples are the goal.

### 10g. Stage D with free bits (KL 3e-3, warm-up 6 epochs, 0.1 nats/dim free), 25 epochs

| test split | D, KL 1e-3 (15 ep) | D, KL 1e-2 (15 ep) | D, 3e-3 + free bits (best val, epoch 12) |
|---|---|---|---|
| image L1, prior mean | 0.134 | 0.132 | 0.133 |
| image L1, best of 5 samples | 0.136 | **0.125** | 0.133 |
| image L1, average sample (blob 0.151) | 0.158 | **0.146** | 0.159 |
| posterior sample | 0.059 | 0.096 | 0.074 |
| sample diversity | 0.123 | 0.094 | 0.127 |
| KL, nats | 124 | 10.8 | 55 |
| text retrieval top-10 | 49.6 % | 47.8 % | 48.8 % |
| character F1 at 0.3 | 0.45 | 0.44 | 0.42 |

Free bits at 0.1 nats per dimension (25.6 nats total) plus a 3e-3 weight settled at 55 nats
and behaved like the 1e-3 run: high diversity, posterior carrying most of the target, samples
whose average scores below the blob. On this data the prior only matches the posterior when
the channel is squeezed to about 10 nats; the intermediate setting did not give an
intermediate result, it fell on the leaky side. The 1e-2 run remains the reference
configuration for stage D.

**Running longer does not help, and the curve says where each head peaks** (validation,
25 epochs): best-of-5 sample L1 is lowest at epoch 6 (0.129) and worsens afterwards; the
average-sample L1 likewise; text retrieval peaks at epoch 12 (0.476) and drifts down to
0.463; character F1 peaks at epoch 9 (0.466) and falls to 0.413; the KL is flat from epoch 9.
The image heads finish earlier than the text heads, so a single checkpoint chosen on
retrieval (epoch 12) is already past the best image epoch. Twelve to fifteen epochs with
selection on the metric you care about is the budget; there is nothing left to gain from
epochs 15-25 on 13.6k windows.

## 11. Scaling stage

After section 10 the model's remaining limit was data and the strength of the frozen
components, not the sequence model (every head plateaued by epoch 12 on 13.6k windows; MiniLM
and CLIP outperformed their from-scratch counterparts by 3-4x). Three scaling steps, all
documented here so that a reader can see what changed and why.

1. **All frames of every story.** The caches were built with at most 10 frames per story;
   stories have up to 22 (median 13). Raising `MAX_FRAMES` to 22 in `poc/precompute.py` and
   `v2/precompute_annotations.py` uses all 44,199 training frames instead of 31,226 and gives
   29,991 training windows instead of 17,018 (+76 %). The large uint8 tensors (frames, crops)
   now stay in CPU memory and `gather()` moves each batch to the GPU (`v2/data.py`).
2. **GroundCap for component pretraining.** `daniel3303/GroundCap` is the single-frame
   dataset StoryReasoning was built from: 52,350 movie frames with grounded captions in the
   same tag format. `v2/precompute_groundcap.py` caches frames at 60x125 and caption token ids.
   It has no sequences, so it only feeds the two component pretraining jobs: the visual
   autoencoder (`--extra groundcap`, ~96k frames) and the text language model (`--extra
   groundcap`, ~83k captions). It never touches the predictor's windows or the test split.
3. **Frozen CLIP as an extra input, and a wider decoder.** `--clip-input` concatenates the
   cached CLIP ViT-B/32 embedding of each frame to the fusion input; `--entity-features clip`
   builds the entity tokens from cached CLIP embeddings of the character crops (also removes
   the crop encoding from the training step); `--width 2` doubles the channels of the
   autoencoder (`--ae-width 2` in the trainer). The mixture, the latent target and the decoder
   still live in the autoencoder's latent space, so the pipeline's shape is unchanged; CLIP
   adds what the frames look like semantically, which the from-scratch encoder lacked.

Resolution (120x250) was deliberately left out of this stage: it multiplies compute by four,
raises only the reconstruction reference and the near-copy windows, and does nothing for the
cut windows. It is worth one run only if the samples of this stage look limited by decoder
detail rather than by content.

Run: `python poc/precompute.py`, `python v2/precompute_annotations.py`,
`python v2/precompute_groundcap.py`, then `v2/pretrain_visual.py --width 2 --extra groundcap`,
`v2/pretrain_text.py --extra groundcap`, and `v2/train.py --stage D --copy-path --recon-weight 1
--attn-weight 1 --kl-weight 1e-2 --clip-input --entity-features clip --ae-width 2 --ae-weights
v2/out/visual_ae_w2.pt`.
