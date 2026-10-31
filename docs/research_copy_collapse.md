# Research: why the dynamics model copies first and overfits after

**Doc class: Live.** A quick survey, dated 2026-10-31, of published remedies for
the pattern both Phase 2 dynamics runs showed. It is a survey, not a decision;
the decision it feeds is decision 4's next step in `phase2_structural_plan.md`.
Claims about external work are labelled **fact** (the source states it) or
**inference** (my reading of how it applies here). Several sources are 2026
preprints read through their abstracts or secondary summaries, and are marked
so. Nothing here has been run on mirage except what the last section records.

## The pattern it addresses

About 87% of target cells repeat the frame before (persistence, 86.69% on the
full val population, r54). Both runs learn that copy first - the best checkpoint
of each sits at an early step with a copy overlap near 100% - and then memorise
the training windows: held-out cross-entropy rises from epoch 1 without
regularisation (r62) and still rises with dropout 0.1 and weight decay 0.1
(the 2026-10-30 run). Neither checkpoint beats persistence, and the best one's
rollouts freeze on every frame.

## Findings

1. **The failure is common, and our metric is the right one.** A NeurIPS 2026
   paper reports a world model explaining 97.9% of its targets' variance where
   copying the last frame explains 99.6%, and that 9 of the 12 papers in its
   survey reporting a prediction score give no baseline for it. It recommends
   scoring against persistence - which gate row 1 already does. **Fact, abstract
   only.** <https://neurips.cc/virtual/2026/166122>
2. **Weight the loss toward what changes.** CAER reweights training toward the
   tokens the action causally affects, found by comparing predictions with and
   without the action, normalised so the total weight is unchanged, and claims
   better physical consistency and controllability than uniform training.
   **Fact, abstract only - no numbers seen.** <https://arxiv.org/abs/2608.30897>
   The cheap version here weights target cells whose token differs from the
   frame before. **Inference**: under a plain mean the unchanged 87% of cells
   dominate the gradient, which is the copy-first signature.
3. **Corrupt input tokens.** In data-limited autoregressive pretraining,
   replacing a share of input tokens with random ones regularised best at 15%
   (validation loss 3.841 against 4.015 without), and random replacement beat
   masking at every rate. **Fact, language models only - secondary summary.**
   <https://www.emergentmind.com/papers/2606.16246> GameNGen's noise on context
   frames stops its rollouts degrading after 20-30 steps. **Fact.**
   <https://arxiv.org/pdf/2408.14837v1.pdf> Context corruption is already
   decision 7's named first remedy for exposure bias; **inference**: it also
   makes copying the context unreliable, which is the behaviour to break.
4. **More data is the cure for overfitting proper.** FitVid overfits on small
   video benchmarks and not on RoboNet, which is far larger. **Fact.**
   <https://arxiv.org/pdf/2106.13195> It stays the last remedy here because
   regenerating the set moves `data_hash`.
5. **Delta tokens are the structural fix, at a tokenizer's cost.** Δ-IRIS and
   DeltaTok encode only what changes between frames, so "no change" is free.
   **Fact.** <https://arxiv.org/html/2406.19320v1>,
   <https://arxiv.org/pdf/2604.04913> Δ-IRIS reports predicting delta tokens was
   "much more difficult" than full-frame tokens. A new tokenizer moves
   `tokenizer_hash` and re-opens Phase 1.
6. **MaskGIT-style dynamics (Genie) do not fit.** They change the attention
   mask, an irreversible decision already taken, and TECO reports MaskGIT
   collapsing in later frames on DMLab. **Fact.**
   <https://arxiv.org/pdf/2210.02396>

## Gaps

No source tests token corruption on a robot world model; the 15% rate comes
from language models. CAER's results could not be read (its PDF did not
extract). IRIS and TECO publish nothing on overfitting. About nine searches and
three full reads - a quick pass, not an exhaustive one.

## What was taken into the project, 2026-10-31

Both cheap remedies, as training knobs that move no hash, **off by default**
until a run shows they help (`mirage/dynamics.py`):

- `--change-weight W`: target cells whose token differs from the frame before
  weigh `W`, others 1, as a weighted mean so the loss keeps its scale.
- `--corrupt P`: each input frame token is replaced by a uniform random code
  with probability `P` during training; actions and targets are never touched.

The self-check asserts the weights land on exactly the changed cells, uniform
weights reproduce the plain mean, and corruption at 15% hits 15% of frame
inputs and no action. Held-out loss stays the plain, uncorrupted mean, so the
train/val gap stays comparable with r62. **Unmeasured** until a run lands.

## Measured 2026-10-31 (`runs.jsonl` r67)

At 2.0M parameters, dropout 0.1, against r66's runs of the same setting:
**corruption works on the loss and not on the gate.** At 15% it lowers the best
held-out loss from 0.475-0.478 to 0.4625-0.4629 on both seeds and moves the
minimum from epoch 2-3 to 5-6; at 30% it reaches 0.4592, halves the final gap
and runs to the cap without tripping. No checkpoint beats persistence. **The
change weight fails** at 4 and 8: the model predicts change too often, and every
measure is worse. Finding 2's inference - that the unchanged cells' share of the
gradient is what holds the model at copying - is not supported in this form.

