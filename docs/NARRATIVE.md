# Building a story-continuation model, one component at a time

A pedagogical path from a minimal baseline to the full architecture in `v2/`, arranged so that
each level adds one idea, teaches one concept, and can be checked with one number. It is not
the historical order in which the project was built (that is in `ARCHITECTURE_EVOLUTION.md`);
it is the order in which the ideas depend on each other. Every level ends with what to
measure, the value reached in this repository, and exercise ideas.

**The task.** StoryReasoning gives stories of 5 to 22 consecutive movie frames, each with a
description. Given four frames and their descriptions, predict the fifth frame (an image at
60x125) and its description (text). Annotations give, per frame, the characters with a
persistent identity and a bounding box, the objects, and the setting.

**The trap the whole path is built around.** The next frame is usually a different camera
shot (85 % of windows). Its pixels are not determined by the inputs, so any loss that compares
the output with the target pixel by pixel is minimised by the *average* of all plausible next
frames: a blob. Learning to see that trap, measure it, and work around it is the arc of the
course.

---

## Level 0: Data, floors and an honest metric

**Concept.** Before any model: what does a trivial predictor score? Without floors, a model
that outputs a constant looks like it is learning.

**Build.**
- A cache of frames at 60x125 (uint8) and of the descriptions' token ids, so an epoch takes
  seconds. Sliding windows over each story (4 inputs, 1 target), split by story.
- Three floors for the image, per window: the constant per-pixel median image; copy the last
  input; copy the best of the four inputs (an oracle that peeks at the target).
- A split of the windows into "shot continues" (some input within L1 0.06 of the target) and
  "shot cuts".

**Measure.** Pixel L1 of each floor on the test windows.

| floor | all windows | shot continues (15 %) | shot cuts (85 %) |
|---|---|---|---|
| median image | 0.151 | 0.111 | 0.154 |
| copy last frame | 0.170 | 0.041 | 0.178 |
| best of the 4 inputs | 0.130 | 0.040 | 0.142 |

**Lesson.** On a cut, a real frame from the same story scores *worse* than the blob. Pixel L1
cannot reward plausibility. Everything downstream is judged against these numbers.

**Exercises.** Compute the floors. Show that the median image beats copy-last. Find the share
of windows where frame 3, not frame 4, is the closest input (it is 33 % vs 27 %: the A-B-A-B
rhythm of dialogue editing).

---

## Level 1: Representing a frame — the convolutional autoencoder

**Concept.** Reconstruction as pretraining; the latent as the currency the rest of the system
will trade in; the reconstruction score as the ceiling any prediction could reach.

**Build.** A four-block strided CNN encoder (60x125 -> 4x8x256 -> linear -> 256-d latent,
LayerNorm on the latent) and the mirror transposed-convolution decoder with a sigmoid output.
Train with L1 on every frame in the training stories.

**Measure.** Validation reconstruction L1 vs the median floor. Reached: 0.040 vs 0.152. Check
that no latent unit is dead and the latent spread is stable.

**Lesson.** A 16-d latent with a ReLU (the original notebook) dies; 256-d with LayerNorm does
not. Reconstruction at 0.040 is the best any decoder-based prediction can do, and only if the
predicted latent equals the true one.

**Exercises.** Vary the latent width (16, 64, 256) and plot dead units and L1. Replace
LayerNorm with ReLU and watch the collapse. Interpolate between two latents and decode.

---

## Level 2: Representing a description — a language model, and a frozen encoder

**Concept.** Two different jobs for text: *encoding* a description into a vector (for the
inputs) and *generating* one (for the output). Pretrained frozen components versus from
scratch.

**Build.**
- Encoder: frozen MiniLM sentence embeddings (384-d), cached once. Keep a from-scratch
  bidirectional LSTM encoder as the comparison.
- Generator: an LSTM language model over BERT word pieces, pretrained unconditionally on all
  descriptions (and, at Level 9, on GroundCap captions too).

**Measure.** Text perplexity of the language model (reached 17.9 on held-out descriptions).
Later, retrieval with each encoder.

**Lesson.** The corpus has a house style: 34 % of descriptions contain "the tension", and greedy
decoding of any model trained on it returns the modal sentence. Use sampling (nucleus, with a
repetition penalty) to see what a conditional model actually knows, and judge it by numbers,
not by the greedy string.

**Exercises.** Count the most common 4-word openings. Decode greedily and by sampling from
the same model. Compare MiniLM and the LSTM encoder by nearest-neighbour retrieval of
descriptions.

---

## Level 3: Fusion and a sequence model

**Concept.** Early fusion by concatenation; a GRU as the sequence summary; predicting in
latent space; why a shared vector gets captured by the strongest loss.

**Build.** Per input frame, concatenate the frame latent (256) and the description vector
(384), pass through Linear + LayerNorm + GELU, run a GRU over the four tokens. From the final
state, predict: the latent of frame 5 (decoded to pixels with the pretrained decoder, pixel
L1, plus a cosine term to the true latent from a *frozen copy* of the encoder), the MiniLM
vector of description 5 (cosine), and description 5 itself (the LSTM decoder conditioned on the
state at every step, teacher forcing, cross-entropy).

**Measure.** Image L1 vs floors; spread of the predictions across inputs (0 = the blob);
retrieval of the true next frame / description among all test targets by centred cosine
(top-10); text cross-entropy with the true vs a *shuffled* condition.

**Lesson.** Three things go wrong here, and each is a classic:
1. *The decoder ignores its condition.* Cross-entropy is the same with a shuffled condition.
   Fix: word dropout on the teacher-forced tokens, plus the text-embedding target.
2. *Encoder latents share a mean.* Raw cosine between any two latents is 0.31, so a raw cosine
   loss is nearly blind and the fine-tuned encoder satisfies it by shrinking everything toward
   the mean. Fix: centre by the batch mean of the targets; take the target from a frozen encoder.
3. *One vector, three heads.* Pixels, image latent and text embedding compete for the same
   256 numbers. Fix: separate projections per head.

**Reached** (stage A in the repository): image L1 0.135 (floors 0.151 / 0.170), spread 0.12,
text retrieval top-10 49 %, condition gap 0.19 nats.

**Exercises.** Reproduce the shuffled-condition test. Measure the mean component of the
latents and show what centring does to the cosine. Train with a single shared latent and
with separate heads and compare both retrievals.

---

## Level 4: Attention that can select an input

**Concept.** Fixed-query attention (a learned importance profile) versus content-dependent
attention (query from the state, keys from the sequence); a differentiable "copy one of the
inputs" via a mixture of latents and a learned gate.

**Build.** Query = W_q h, keys = W_k seq_i, alpha = softmax; `z = g * sum(alpha_i zv_i) +
(1 - g) * residual`, with the mixture over the *encoder latents* so that a one-hot alpha
decodes to the chosen frame at reconstruction quality. Return alpha and g for inspection.

**Measure.** Mean alpha per input position; how often argmax(alpha) is the closest input on
near-copy windows (chance 25 %); the gate on near-copy vs cut windows.

**Lesson.** Without an explicit target the attention learns the *positional prior* (frame 3
highest) but not the per-window pattern (29 % vs 25 %); the pixel gradient from 15 % of
windows is too weak to teach selection. Supervising alpha with the known closest input works
(52 %), but if the query is computed from the GRU state the supervision reshapes what the text
heads read and text retrieval collapses (49 % -> 16 %). The selection should be computed from
the frame latents' pairwise similarities, decoupled from the GRU (Level 7).

**Exercises.** Plot alpha per window and look for A-B-A-B. Add the supervision and watch text
retrieval. Compute the ceiling of any selector (best-of-4 oracle, 0.130).

---

## Level 5: Protect what was pretrained

**Concept.** Fine-tuning drift; discriminative learning rates; keeping the pretraining
objective alive; selecting checkpoints on the right metric.

**Build.** Image encoder and decoder at 0.1x the learning rate, pretrained language-model
layers at 0.3x; a reconstruction term `L1(decoder(encoder(target)), target)` during sequence
training; a per-epoch reconstruction monitor; checkpoint selection on validation retrieval,
not on image L1.

**Measure.** The reconstruction monitor (pretrained reference 0.040).

**Lesson.** Lowering the learning rate does not stop the drift (0.040 -> 0.080 after one
epoch); only keeping the objective does (stays at 0.041). The pixel head overfits from epoch
1 while the other heads are still learning, so "best validation image L1" selects an
untrained model; select on retrieval and stop at 12-15 epochs.

**Exercises.** Log the reconstruction monitor with and without the term. Plot the validation
curve of each head and find where each peaks.

---

## Level 6: Using the annotations

**Concept.** Structured targets with honest baselines; entity tokens; set attention inside a
sequence model.

**Build.** From the chain-of-thought tables: a MiniLM vector of each frame's setting (input and
target); character crops encoded and tagged with a story-local slot embedding, pooled into
each frame token by attention; a head that predicts which of the story's characters appear in
frame 5 (multi-label BCE, masked to the story's slots).

**Measure.** Character F1 at a calibrated threshold, against "same as frame 4" (0.38),
"everyone seen so far" (0.41) and "present in at least two inputs" (0.47).

**Lesson.** A head on the pooled sequence state learns the prior (0.41) and not the pattern;
the per-character history is the signal. Frame 3's characters predict frame 5's better than
frame 4's, the editing rhythm again. A per-slot head that sees each slot's own history reaches
0.45 (Level 7).

**Exercises.** Compute the three baselines. Build the per-slot head. Calibrate the threshold on
validation and show why 0.5 is wrong when 1.6 of 7 slots are positive.

---

## Level 7: Names in the text, selection from the frames

**Concept.** Cross-attention over a memory in a recurrent decoder; conditioning through
content rather than a single vector; decoupling a mechanism from a shared state.

**Build.**
- The text decoder attends, at every step, over a memory of the four input description
  vectors, the predicted text embedding, and the names of the characters predicted present
  (true names during training, predicted at test; names come from the story's grounded
  mentions).
- Mixture attention computed from the window-centred cosines between the four frame latents
  plus position, supervised toward the closest input; the GRU keeps its own attention for the
  context vector.
- The per-slot character head from Level 6.

**Measure.** Condition gap (reached 0.27-0.29 nats, from 0.06), names appearing in generated
text, closest-input accuracy 44 % with text retrieval intact, character F1 0.45.

**Exercises.** Ablate the memory (no names / no descriptions). Compare the coupled and
decoupled attention on text retrieval.

---

## Level 8: A distribution over next frames — the variational latent

**Concept.** Why sampling beats averaging on a multimodal target; the conditional VAE recipe
(prior, posterior, reparameterisation, KL); the KL weight as the knob between "posterior leaks
the target" and "prior matches"; free bits.

**Build.** The residual path becomes Gaussian: a prior from the state, a posterior from the
state and the frozen target latent; train with L1 on the decoded posterior sample plus
`beta * KL(q || p)` with a warm-up; at test decode the prior mean (deterministic prediction) or
prior samples.

**Measure.** Prior-mean L1; best-of-5 and average sample L1 against the blob; sample
diversity; posterior-sample L1; KL in nats.

| KL weight | KL | best of 5 | average sample (blob 0.151) | diversity |
|---|---|---|---|---|
| 1e-3 | 124 | 0.136 | 0.158 | 0.123 |
| 1e-2 | 10.8 | **0.125** | **0.146** | 0.094 |
| 3e-3 + free bits 0.1 | 55 | 0.133 | 0.159 | 0.127 |

**Lesson.** At 1e-3 the posterior carries the target and the prior never matches it: samples
are composed pictures but not tied to the inputs. At 1e-2 the average sample beats the blob
and the best of five is below every deterministic model, the first number past the L1
ceiling; the samples sit closer to the mean. The intermediate setting fell on the leaky side.
Sampling, not a new distance, is what turns averages into pictures; a VGG perceptual loss was
tried and changed nothing, because no feature distance ranks a different shot above the blob.

**Exercises.** Sweep the KL weight and plot KL against best-of-K. Implement free bits. Show
the posterior-collapse test (posterior L1 vs prior-mean L1). Add a pixel copy path over the
four inputs and measure its effect on the near-copy windows only.

---

## Level 9: Scaling what has capacity

**Concept.** Data-limited versus capacity-limited; frozen pretrained features as the lever;
using the dataset a benchmark was built from.

**Build.** All 22 frames per story (30k windows instead of 17k); GroundCap (52k single frames
with captions, the source of StoryReasoning) for pretraining the autoencoder and the language
model only; frozen CLIP embeddings of frames as an extra fusion input and of character crops as
entity tokens; a twice-wider autoencoder. Resolution left out (four times the compute for
gains confined to reconstruction and near-copy windows).

**Measure.** The same table as Level 8, plus the reconstruction reference of the wider
autoencoder and text perplexity of the language model trained on both corpora.

**Reached** (same 5,254 test windows, scaled vs reference): text retrieval 51.6 % vs 41.5 %,
text cross-entropy 2.40 vs 2.65, character F1 0.46 vs 0.44, reconstruction 0.036 vs 0.042;
image prediction unchanged (prior mean 0.132 vs 0.131, best of 5 samples 0.124 vs 0.123).

**Lesson.** Every head's validation curve was flat by epoch 12 on 13.6k windows; scaling the
GRU or the attention would fit sooner and generalise no better. What moves is data and the
strength of the frozen components, the pattern seen from the first proof of concept (CLIP
retrieval 40 % vs 9 % for the from-scratch encoder). And what does not move, even with a
stronger encoder, more windows and a better decoder, is the image prediction: the cleanest
demonstration that it is bounded by the objective, not the model.

**Exercises.** Ablate each of the three scaling steps. Compare the wide and narrow autoencoder
on reconstruction and on the near-copy windows.

---

## Level 10: Measuring what the eye sees, and training for it

**Concept.** Metric validity: a metric must separate the outputs you can tell apart. Pixel L1
and VGG feature distance cannot separate "right layout, blurred" from "right tint, no
structure"; a semantic embedding can. Then the same embedding as a loss.

**Build** (`v2/semantic_metrics.py`). Three measures with the usual floors: CLIP similarity
between the decoded image and the target (centred by the mean test embedding); the Frechet
distance between CLIP-feature distributions of a set of images and of the targets (realism of
the set, independent of alignment); and the variance of the Laplacian as a sharpness ratio to
the target. Then a training term: centred cosine between the CLIP embedding of the decoded
image (differentiable through a frozen CLIP) and the cached CLIP embedding of frame 5.

**Measure** (5,254 test windows, before the loss):

| candidate | CLIP similarity | CLIP Frechet | sharpness / target |
|---|---|---|---|
| blob | 0.00 | 0.65 | 0.00 |
| deterministic prediction (Level 5 model) | 0.074 | 0.26 | 0.02 |
| Level 9 prediction | 0.056 | 0.33 | 0.02 |
| Level 8 prior samples | 0.00 | 0.33 | 0.03 |
| reconstruction, narrow / wide autoencoder | 0.08 / 0.15 | 0.33 / 0.24 | 0.07 / 0.10 |
| copy last frame | 0.37 | 0.00 | 1.01 |

**Lesson.** The predictions carried weak target semantics and scaling did not add to it; what
looked better was the wide autoencoder's reconstruction. The variational samples had structure
but not the target's. Everything drawn was at 2 % of the target's sharpness. And the decisive
row: copy-last scores 0.37 against 0.00 for the blob, so under CLIP similarity a different but
related frame beats the average by a wide margin, the property no pixel or VGG distance had.
That makes it a loss that rewards plausibility. Trained with it (15 epochs): CLIP similarity
of the prediction 0.249 (from 0.056), sharpness 0.34 of the target's (from 0.02), Frechet
distance 0.25 (from 0.33), pixel L1 0.134 (from 0.132). The predictions become textured,
scene-like images. Two caveats to teach with it: the samples still carry no target semantics,
and the decoder exploits the frozen CLIP: with an absolute similarity loss it paints a fixed
motif; with random augmentations before the encoder the motif becomes a translation-robust
template (a faint face in a suit in every scene). The cure is a *relative* loss: contrastive
InfoNCE over the batch, where the prediction must be closer to its own target than to the other
targets, so a generic template gains nothing.

**Exercises.** Compute the three measures for the floors and show which metric ranks copy-last
above the blob. Add the CLIP loss and plot CLIP similarity against pixel L1 over epochs. Judge
a set of samples by Frechet distance and by best-of-K, and explain why they can disagree.

---

## Reading the numbers, at every level

- Report every image number for all windows and for the near-copy / cut split.
- Report the floors next to the model; never a loss alone.
- For any conditional output, report the shuffled-condition score.
- For any latent, report the centred pairwise cosine between windows (collapse detector).
- For any multi-label head, report three trivial baselines.
- Choose checkpoints on the metric that answers the question, not on the training loss.
- Before trusting a metric, check that it separates the floors the way your eye does.

## Where the path ends, and what lies beyond

Under pixel L1 the deterministic ceiling on this data is about 0.13 for a model that
understands the inputs; the variational latent gets samples below it. Sharper samples on cut
windows need a loss over samples rather than expectations (adversarial, diffusion), or the
retrieval formulation, where a two-minute model on frozen CLIP features already picks the true
next frame out of 2,974 candidates 7 % of the time and puts it in the top 10 40 % of the time.
