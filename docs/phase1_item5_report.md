# Phase 1 progress report - item 5, first slice: R0 and the held-out floor

Session of **2026-08-28**, continuing from `docs/phase1_progress_report.md`
(items 1 to 4). Covers build order **item 5 stages 5a, 5b and 5c**, the **R0**
rung, and one correction to a pre-work number that R0 forced into the open.

Item 5 stages **5d** (the R1/R2/R3 ladder and the gate table) and **5e** (the
token cache) are **not started**, deliberately. The handoff for them is the
second half of this file.

**This file is a narrative, not a source of truth.** Every number here is
transcribed from `runs.jsonl` rows **32 and 33**, from
`runs/20260828-185053-r0/result.json`, and from the verification log at the end
of `world_model_architecture.md`. Those are authoritative - if a figure here
disagrees with them, they win and this file is what is wrong. Do not correct a
number here without correcting it there first, or this becomes the third place a
stale figure can hide.

---

## Outcome

Two commits, tree clean. Neither is on `origin/main`, which still sits at
`92e2193` - the branch is **3 ahead**, because the previous session's docs commit
was never pushed either.

```
70145a6 fix: measure the k-means floor on the held-out split - 28.27 dB, not 29.02
26492b9 feat: add fsq.py 5a-5c, and R0 clears 30 dB with 1.228 dB to spare
00e4de9 docs: add the Phase 1 progress report for items 1-4, and the item 5 handoff  (unpushed, prior session)
```

626 insertions across five files: `mirage/fsq.py` new at 499 lines,
`bench/patch_probe.py` +124, and one line each in `AGENDA.md`,
`world_model_architecture.md` and `runs.jsonl` (two rows).

**Both headline questions are answered, and they answer in opposite directions.**
The architecture is fine. The bar it has to clear was wrong.

---

## The number R0 was run for

**Held-out PSNR 31.228 dB against Q-1's 30 dB bar. Clears it by +1.228 dB.
Train-val gap +0.810 dB.**

Run `20260828-185053-r0`, `data_hash 18a76531aaa8b609`,
`tokenizer_hash 978246d7157caa27`, 15 epochs, 1,313.3 s.

| Epoch | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| val dB | 26.353 | 27.925 | 28.717 | 29.363 | 29.733 | **30.098** | 30.433 | 30.591 | 30.762 | 30.889 | 30.998 | 31.047 | 31.149 | 31.196 | **31.228** |
| gap dB | +0.291 | +0.314 | +0.371 | +0.443 | +0.477 | +0.540 | +0.595 | +0.634 | +0.685 | +0.712 | +0.737 | +0.766 | +0.782 | +0.795 | +0.810 |

R0 was run **before any FSQ code was wired in**, which was the point: continuous
bottleneck, quantizer constructed but bypassed, no attention. If it had missed
30 dB, no levels table would have helped and the encoder would have been what
needed work - and writing `codes_to_indices` and the token cache first would have
spent a day discovering that the expensive way.

**Three things the curve says beyond the final number.**

- **It crossed the bar at epoch 6 and was still climbing at 15.** So 15 epochs is
  a floor on R0, not its ceiling, and the plan's "15 epochs is enough to *rank*
  architectures" has not yet been tested on a curve that has flattened.
- **The gap is small and growing slowly.** +0.291 dB at epoch 1, +0.810 at 15.
  The measured gap is the evidence; the ratio behind it is that 283,800 frames of
  12,288 values each is **3.49 billion target values against 744,966
  parameters - 4,681 per parameter**. (Frames per parameter is 0.38, i.e. *fewer
  frames than weights*, which is why the per-frame ratio is the wrong one to
  quote for an autoencoder: the target is every pixel, not every frame.) So a
  future shortfall will be a **capacity** problem, not an overfitting one, and
  that decides which lever to reach for: wider channels or residual blocks, not
  regularisation.
- **R0 is a loose upper bound and should be read as one.** Its bottleneck is
  192 fp32 numbers per frame; R1's will be 64 tokens x 9 bits = 576 bits. A
  comfortable R0 does not promise R1 clears 30 dB. It only rules out the encoder
  as the suspect.

---

## The one finding that changes the plan

**The 29.02 dB k-means floor was never a held-out number, and gate row 2 was
comparing against it.**

`data.is_val` splits **by episode** - hashed per `episode_id`, 473 train and 27
val of 500. `bench/patch_probe.py`'s `sample_frames` spreads its sample evenly
*within every shard*. Those two facts are incompatible: the sample the floor was
fit and scored on straddles the split. So a tokenizer's held-out PSNR was being
measured against an in-sample dictionary, and the comparison flattered the
dictionary.

The item 5 handoff predicted the two might disagree. They disagree
**structurally**, not by sampling luck, which is why this was worth fixing rather
than noting.

The probe now carries a split-aware section that takes the split from
`data.is_val` over `data.episode_index` and **never from a fraction recomputed in
the probe**. It reports two numbers, because two separate effects were inflating
29.02 dB and only measuring both separates them.

| Fit on | Scored on | k=240 | **k=512** | k=1024 | live @512 | live @1024 |
|---|---|---|---|---|---|---|
| **train** | **val** | 27.09 | **28.27** | 29.39 | 486/512 | 912/1024 |
| val | val | 28.06 | 29.88 | 31.81 | 512/512 | 1024/1024 |
| whole set *(unchanged)* | itself | 27.53 | 29.02 | 30.51 | 512/512 | 1024/1024 |

- **The honest floor at 512 codes is 28.27 dB.** Train-fit, val-scored - the same
  treatment a tokenizer gets. **Gate row 2's bar is `+1.73 dB`, i.e.
  `30.0 - 28.27`, not `+0.98`.** `AGENDA.md`'s row now says so.
- **The in-sample advantage is 1.61 dB**, of which the straddled whole-set sample
  was carrying **0.75 dB**. The remaining 0.86 dB is the advantage a codebook
  gets from being fit on the very patches it scores, which no split change can
  remove - it is why the two rows exist rather than one.

**R0's conclusion is unaffected and its margin is wider than first reported:
31.228 dB against 28.27 dB is +2.958 dB, not the +2.208 dB computed against the
old floor.**

### Two further claims died with it

Both were in-sample artifacts, and both were load-bearing somewhere.

- **"1,024 codes clear Q-1 outright" is dead.** Held out, 1,024 reaches
  **29.39 dB** and *misses* 30. The old 30.51 dB was the reason vocabulary counted
  as a proven lever the token budget merely forbids. It is now not proven at all -
  which, read the useful way, **removes the main reason to regret the fixed
  512-code budget** the Phase 2 handoff imposes.
- **"All 512 centroids stay live" is in-sample too.** Scored on held-out patches,
  **486 of 512** and **912 of 1,024** are used. This does not revive the Q-2
  collapse risk - Q-2 is a statement about a trained tokenizer's code usage, not
  about k-means - but the sentence as written was not a held-out claim.

### Two things that did not move

- **The 96x96 fork's evidence is untouched.** The edge-error split reads
  **99.98%** on val under the train-fit codebook, against 99.95% whole-set. The
  fork diagnostic does not depend on any of this.
- **Float versus uint8 centroids cost at most 0.01 dB** (29.88 vs 29.87 at
  k=512). That caveat can be retired rather than carried, which matters because
  the documented floor was a float number while every tokenizer PSNR is uint8.

---

## Everything else measured

| Measure | Number | Note |
|---|---|---|
| Parameters, no attention | **744,966** | the plan estimated ~1.5M; **refuted** |
| Parameters, with attention | **1,008,646** | `GridAttention` adds 263,680 |
| Latent | `(3, 8, 8)` | `len(levels)` channels, so R0 and R1 share the bottleneck geometry |
| 15-epoch rung | **1,313.3 s = 21.9 min** | the plan predicted ~6 min; **refuted** |
| Per epoch | 87.6 s | 2,217 steps of batch 128, plus 2.4 s of eval |
| Per step | **39.5 ms** | aggregate, `1,313.3 / 33,255`; the 16 eval passes are ~38 s of the total, so training alone is ~38.4 ms |
| GPU state during training | **99% util, SM 2,542-2,677 of 3,090 MHz, 82.8-92.5 W** | sampled 4x mid-run at 20 s intervals; 75-87 C, 4.0-4.7 GiB used |
| Preload, both splits | 24.9 s | 283,800 + 16,200 frames |
| Untrained baseline | **9.823 dB** | `--epochs 0`, 2.4 s; a sanity floor for the eval path |
| Final train PSNR | 32.037 dB | the val figure plus the gap |
| Final val MSE, `[0,1]` units | 0.00075594 | reported beside PSNR so loss and gate read on one line |
| STE gradient at zero | **0.858 / 1.001 / 0.668** | `[8,8,8]` / `[5,5,5]` / `[4,4,4]`, all three reproduced |
| STE gradient, mixed table | 0.858 / 0.8009 / 1.001 | `[8,6,5]`, per-channel - a mixed table has three different LRs |
| Output range check | +3.0 bias reaches **3.773** | proves no `tanh` and no clamp survived |
| Gradient coverage | **28** tensors, **34** with attention | all reached through the quantizer, all finite |

**The 3.6x timing miss is not a batching problem.** The GPU sample settles that:
99% utilisation with the SMs at 82-87% of max clock drawing 82-92 W of a ~100 W
cap is a compute-bound run, so the loader is not the constraint and no work on the
input path would recover anything. Per `CLAUDE.md`, a compute number is gated on
SM clock and power draw, and both are recorded above. Budget **~22 min per rung**,
which makes R1+R2 at both resolutions about **1.5 to 2 hours**, not 36 minutes.

---

## Judgement calls worth knowing

- **`levels` is a function argument, not config.** Adding it to
  `configs/base.json`'s `tokenizer` section would change `tokenizer_hash`, which
  chains off `data_hash`, and `config._check_keys` rejects unknown keys anyway.
  Since a rung's identity already encodes its resolution through that chain,
  putting a *sweep* parameter inside it would make every ladder step a different
  tokenizer identity. It stays out until a winning table is chosen, and then it
  goes in deliberately with the hash bump acknowledged.
- **Palette indices stay in CPU RAM.** 1.16 GB for the train split against ~5 GB
  free on an 8 GB card that already had 3.1 GiB in use at session start. One batch
  is 512 KB, so the transfer is noise next to a 35 ms step, and the LUT expansion
  runs GPU-side as a gather over 7 rows. The GPU sample above confirms nothing was
  lost to this.
- **The train-val gap is measured on a fixed 4,096-frame train subsample**, drawn
  once with the run's seed and evaluated exactly like the val split. Comparing a
  moving loss average against a full-split eval would report a gap that is partly
  an artifact of the two being measured differently.
- **`GridAttention` was written because 5b specifies it, and is marked UNVERIFIED
  in the source.** R0 does not use it. Only its shapes, finiteness and gradient
  flow are checked. Do not quote it as a working quality lever.
- **The self-check reproduces recorded measurements rather than asserting
  shapes.** Reproducing all three STE gradients at once pins `eps`, `offset`,
  `shift` and the normalisation *simultaneously* - a shape assertion would pass on
  a quantizer with any of the four wrong.
- **`codes_to_indices` is not written**, even though the plan lists it under 5a
  and its bijection is part of 5a's working-when. It is not needed until R1, and
  the instruction was to read R0's number before writing it.
- **Two floor numbers, not one.** Reporting only the honest floor would have left
  "why did 29.02 become 28.27" unanswerable - split leak or in-sample advantage,
  no way to tell. The second row costs one extra k-means fit per `k` and turns a
  correction into an explanation.
- **The probe's new invariant has documented slack.** A codebook fit on the
  patches it scores should not lose to one fit elsewhere, but k-means++ is not
  globally optimal, so the assert allows 0.15 dB rather than claiming an identity.
  A real inversion would mean the two splits are not the same distribution, which
  is a data bug and not a fit artifact - the message says so.
- **The `final` jsonl record now carries `epoch`.** It also matches a
  `val_psnr_db` filter, so a reader grouping by epoch would have tripped over its
  absence. Found by tripping over it.

---

## Open, not blocking

- **About 16 stale restatements of `29.02` / `0.98 dB` remain in prose.**
  `AGENDA.md` 64-77, 151, 162-190; `docs/phase1_progress_report.md` 54-74, 176,
  227; `docs/phase1_structural_plan.md` 261, 315, 404-405;
  `docs/world_model_architecture.md` 1070, 1082-1084. The **decision-bearing**
  sites are corrected: `AGENDA.md`'s gate row 2, `fsq.py`'s `GridAttention`
  docstring, and a new verification-log row. The rest are narrative and were left
  rather than swept mid-session, which is exactly the "stale copies outnumber the
  decision" pattern `CLAUDE.md` warns about - so sweep them before they are quoted
  again.
- **`bench/patch_probe.py`'s receptive-field section still hardcodes 64.**
  `grid = 64 // PATCH` and the `<= 64` bound. Fine today because the probe loads
  `base.json` explicitly, but it will silently mis-measure at 96x96. The new
  split-aware section takes its shape from `shards[0].pixels.shape[1:]` and does
  not have this problem.
- **Two method failures worth knowing, both caught only by checking.**
  (a) A bulk `str.replace` patch silently failed to apply, and two `fsq.py`
  self-check assertions kept their old, type-unsafe form for several steps - the
  script asserted on some substitutions and not that one. Assert every
  substitution, or use a real edit. (b) This report was first written with three
  wrong figures: `87.5` s/epoch for `1313.3/15 = 87.6`, `35.1` ms/step for
  `1313.3/33255 = 39.5`, and a **"381:1 data-to-parameter ratio"** that is off by
  three orders of magnitude - frames per parameter is `283800/744966 = 0.38`. The
  first two had already been written into `runs.jsonl` row 32 and were corrected
  in place there, which is the one place this log has been edited rather than
  appended; the row records the correction in an `arithmetic_corrected` field. A
  script that recomputes every derived figure in a report against `result.json`
  and the notebook found all three in one pass, and is worth re-running on the
  next report rather than proof-reading by eye.
- **R0 was still climbing at epoch 15.** The plan's ranking assumption - "ordering
  at 15 epochs survives longer training" - has not been tested on a flattened
  curve, and rungs landing within ~0.5 dB of each other are tied by the plan's own
  rule.
- **W&B remains unexercised.** `wandb` is absent; `logging.py` says so.

---

# Handoff - item 5 stages 5d and 5e

**The item 5 handoff in `docs/phase1_progress_report.md` is spent.** Its first
slice is done and two of its stated numbers have moved. Use this section.

## Start here

`docs/phase1_structural_plan.md` sections 5d and 5e. Trust its *structure*; do
**not** trust its `29.02 dB` / `0.98 dB` figures - see the floor correction above,
and prefer `runs.jsonl` row 33.

## State you can rely on

```
64x64   data 18a76531aaa8b609   tokenizer 978246d7157caa27   validator 48882ee24c2278b6
96x96   data 35e5b8627987a2bb   tokenizer 6e689ccce0c6d994   validator 15a3e80e6566d114
```

Seven self-checks green: `config`, `logging`, `fsq`, and `data`/`validator`
against both configs. Probes green: `patch_probe` (now with the split-aware
section), `occlusion_probe`. Both datasets on disk, 3.5 GB and 7.8 GB.

R0's checkpoint is at `runs/20260828-185053-r0/model.pt` with its hashes inside
it. It is a diagnostic, not a candidate - its bottleneck is continuous, so it
cannot produce tokens.

## What this slice hands you

- `FSQ(levels)` - verified against all three recorded STE gradients.
- `Tokenizer(levels, attention=, quantize=, width=)` - `quantize=False` is R0.
- `train(rung, cfg, ...)` -> a dict, plus `runs/<run_id>/` holding
  `metrics.jsonl`, `meta.json`, `result.json` and `model.pt`.
- `reconstruction_psnr(model, idx, lut)` -> `(uint8 dB, float MSE)`.
- `python -m mirage.fsq` self-checks 5a and 5b without touching data.

## The next four things, in order

1. **`codes_to_indices`, with the bijection check** the plan's 5a working-when
   names. `prod(levels)` distinct values per dimension is already verified; the
   bijection onto `0..prod(levels)-1` is not.
2. **R1: FSQ `[8,8,8]`, no attention, 15 epochs, seed 0.** Same call as R0 with
   `quantize=True`. Compare to the **28.27 dB** held-out floor and the
   **+1.73 dB** bar. `[8,8,8]` is the reference table at `g = 0.858`, so **no
   paired LR run is needed until the levels change** - the paired check exists for
   levels comparisons, and R1 is not one.
3. **R2: R1 plus attention at 8x8.** This is where `GridAttention` stops being
   unverified. Row 2 is the row that matters: at R0's +1.228 dB over the bar, an
   R1 landing near 29 dB tells you little on its own.
4. **Only then 5e**, named by run id, per-shard `.npy` of `uint16`, with the
   manifest recording `tokenizer_hash` and the checkpoint.

## Traps this slice found the hard way

- **Budget ~22 min per rung, not 6.** R1+R2 at 64 and 96 is 1.5-2 hours of wall
  clock. Plan the session around that.
- **Python buffers stdout when redirected**, so a backgrounded run's log stays
  empty until it exits. `Run.log` flushes per line - read
  `runs/<run_id>/metrics.jsonl` instead. That is what per-line flushing is for.
- **Do not filter `metrics.jsonl` on `val_psnr_db` alone.** The `final` record
  matches too. It now carries `epoch`, but group on `final` explicitly.
- **Quote the held-out floor, not 29.02.** Sixteen prose sites still say
  otherwise; `runs.jsonl` row 33 and the verification log are right.
- **`bench/patch_probe.py` at 96x96** will need its receptive-field section
  de-hardcoded from 64 before its Q-2 ceiling means anything there.

## Inherited environment traps, still true

- `cd` into another directory resets the shell cwd; use `git -C` or re-`cd`.
- Git-Bash paths (`/c/...`) do not resolve in Windows Python.
- Heredocs carrying prose apostrophes break the Bash tool; use a real file write.
- `contact_mask` is two fields - bits 0..6 contact, bit 7 scripted. Use
  `mirage.data.contact_bits` / `.scripted`.
- The fixture carries its own `data_hash`; any change to `scene/arm_blocks.xml` or
  the `sim` section invalidates it.
- Do not build under `%TEMP%` - MSBuild `FTK1011`, which surfaces as a missing
  C++ compiler.
- `runs/` (gitignored directory) sits beside `runs.jsonl` (tracked notebook). The
  trailing slash in `.gitignore` is the only thing separating them.
