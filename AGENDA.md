# Mirage: Agenda

Design lives in `world_model_architecture.md`; requirements in
`world_model_requirements.md`. This file is only the ordered list of what to do
next. Keep it short - delete items as they land, do not accumulate history.

Plain-English versions for non-engineering readers: `timeline.md` (schedule,
gates, risks, with the study plan woven in) and `decision_notes.md` (every
decision with its trigger and fallbacks). Both are derived from the two docs
above - when a decision changes, change it there first.

**Phase 0 is complete.** Gate met 2026-08-27 and re-met twice on 2026-08-28 -
once for the `gear 6 / damping 1.5` scene change, once after a `data_hash` fix -
all three conditions every time. The set on disk is the frame count and hash in
`NUM-DATA-FRAMES` and `NUM-DATA-HASH64`. Write-up in `phase0_report.md`; every
number traces to the verification log at the end of `world_model_architecture.md`.

**Do not quote a figure from before 2026-08-28.** Three `data_hash` values exist
and only the last describes the data - the chain is in `canonical_numbers.md`
under `NUM-DATA-HASH64`, with what moved between each. `phase0_report.md` opens
with the same list.

The Phase 0 debt checklist is **closed, 12 of 12**. Three of its items change how
Phase 1 starts:

- **The validator's palette thresholds are calibrated and in config** - item 6,
  landed 2026-08-29. `NUM-VAL-TAU` and `NUM-VAL-FRACMAX` now describe *decoder
  output*, which is a different regime from renders, not a noisier one:
  `NUM-VAL-RECONDIST` against `NUM-VAL-WORSTDIST`. Changing either moves
  `validator_hash`, which is the whole reason they were moved out of code -
  **so calibrate once.** Retuning mid-phase fragments Q-3 into incomparable
  buckets, and the margin over `NUM-VAL-RECONFP` is deliberately thin.
- **The meta record's `contact_mask` is two fields.** Bits 0..6 are block
  contact, bit 7 is scripted-vs-random. `mirage.data.contact_bits` and
  `.scripted` exist; read the raw byte and F-6 reads over 50%.
- **`python -m mirage.data` and `python -m mirage.validator` run without a
  dataset**, falling back to a committed 40-frame fixture. A fresh clone or a new
  worktree can check F-8 and F-9 before generating anything.

Nothing from Phase 0 is pending. The F-9 threshold calibration, the last item,
landed as Phase 1 item 6 on 2026-08-29 - `runs.jsonl` r42.

---

## The writeup. Highest priority, and the only item here with an outside deadline

**Phase 1's gate passed 2026-08-29 and nothing has been published.** This is the
one item on this list that is not an engineering task. Its deadline is not a gate,
it is **2026-09-09**, when classes start and the hours this project runs on
collapse.

**It does not wait for the Phase 4 benchmark table.** Phase 2 is budgeted at two
weeks, Phase 3 at three days, Phase 4 at six - and those are full-time weeks that
stop existing on September 9. Holding the writeup for a table that far out is how
it does not get written. **Ship Phases 0-1 as part one.** The ladder table is part
two, and a published part one is what makes anyone read it.

Nothing here is assembled from scratch:

| Have | Where |
|---|---|
| The eight-row gate table, passing | `python -m mirage.fsq --eval` |
| Eight figures | `docs/figures/fig1_curves.png` through `fig8_token_entropy_map.png`, described in `tokenizer_figures.md` |
| Two narratives over the log | `phase1_progress_report.md`, `phase1_item5_report.md` |
| Every number with its provenance | `canonical_numbers.md`, `runs.jsonl` |
| Plain-English framing, already written | `timeline.md`, `decision_notes.md` |

**Lead with the refutations.** They are the part nobody else's writeup has, and
all three are this project's own measurements overturning its own predictions:

- **15 epochs is not convergence.** Every rung missed `NUM-BAR-Q1` at 15 epochs
  and every rung clears it at 60, with no change to the architecture - only
  `--epochs`. A day was lost reading ranking runs as gate verdicts
- **`NUM-HW-READBACK`, against the ~30 ms the MuJoCo discussion reported.** The
  single riskiest unknown in the plan, the one that could have turned Week 0 into
  three weeks, did not exist
- **`NUM-HW-FP16` after a chassis cooling fix, with no throttle flag ever set.**
  The instantaneous flags read `Not Active` through a 45 W cap and the evidence
  was in the counters
- **The 64/144 fork: the higher resolution wins on the metric everyone was
  watching and fails one nobody was.** `NUM-TOK-FORK` of Q-1 quality, and
  `NUM-TOK-ENT-R1-96` against `NUM-BAR-Q2`. One mechanism drives both signs, and
  **both remedies this project had written down die to arithmetic before either
  is run** - `NUM-TOK-MARGSUM-96` and `NUM-TOK-SHRINK240-UB`. The strongest
  refutation of the set, because the plan was wrong about *which number decides*
  rather than about the value of a number
- **`NUM-VAL-PCTL`: the resolution-free validator statistic that was obviously
  right and is wrong.** A quantile of palette distance replaces a pixel count
  and needs no per-resolution calibration - and it misses gaussian noise 99.9%
  of the time, because it is a *tail* statistic where the failures are *bulk*.
  The cheaper fix, the same count over the frame's pixels, keeps item 6's
  verdict bit-identical at 64x64. This one is the cleanest example of the
  project's own house rule working: the argument was airtight and the
  measurement killed it in one run

**Cite the register id, not the value** - the same rule this file already lives
under. A writeup that hard-codes figures joins the list of documents that went
stale the day the floor moved. Chosen bars are the stated exception.

**Done when** it is published and linked from `README.md`. Not "drafted".

---

## Phase 1: the tokenizer. Budgeted at 1 week

**Structural plan: `phase1_structural_plan.md`.** What each file owns, the build
order with named APIs, done-when per file, and the gotchas.

**Items 1-5 are done. Item 5's gate is met** - all five pass/fail rows pass, on
two checkpoints either of which could ship: R1 `20260829-005439-r1`
(`NUM-TOK-R1-60`, `NUM-TOK-ENT-R1`) and R2 `20260828-230015-r2`
(`NUM-TOK-R2-60`, `NUM-TOK-ENT-R2`), against **`NUM-BAR-Q1` = 30.0 dB** and
**`NUM-BAR-Q2` = 70%**. Outcomes are in `phase1_progress_report.md` (items 1-4)
and `phase1_item5_report.md` (item 5, complete). Both are narratives over
`runs.jsonl` and the verification log, which stay authoritative.

> **The lesson item 5 cost the most to learn: 15 epochs is not convergence, and
> the 15-epoch numbers were ranking runs read as gate verdicts.** Every rung
> failed Q-1 at 15 epochs and every rung passes at 60, with **no change to the
> architecture** - only `--epochs`. The cosine anneals to `lr_floor` by the final
> step, so a short run is frozen at its end *by construction* and its last-epoch
> gain says nothing about convergence. The structural plan already said to quote
> the 60-epoch run; it needed following, not correcting.

> **The six pre-work numbers were re-measured on 2026-08-28, and the k-means
> floor was refuted - by its own methodology, not by the regeneration.** The full
> account is `phase1_item5_report.md` and `runs.jsonl` rows 27 and 33; the value
> chain, with what moved at each step, is `canonical_numbers.md` under
> `NUM-TOK-FLOOR512`. What belongs in *this* file is three consequences and a
> caution.
>
> - **Gate row 2's bar is derived, not measured.** It is `NUM-BAR-Q1` minus
>   `NUM-TOK-FLOOR512`, recorded as `NUM-BAR-ROW2`. It has been published at three
>   different values, because the floor moved twice underneath it and each
>   restatement was correct arithmetic on a stale input. **If the floor moves
>   again this moves with it**, and every doc holding a copy is wrong that day.
> - **"1,024 codes clear Q-1 outright" is dead.** Held out, 1,024 codes reach only
>   `NUM-TOK-FLOOR1024` and still miss `NUM-BAR-Q1`. Read the useful way, that
>   **removes the main reason to regret the fixed 512-code budget** the Phase 2
>   handoff imposes.
> - **The Q-2 collapse evidence is gone.** `NUM-TOK-LIVE512` of 512 centroids stay
>   live on held-out patches; the far smaller count that started the worry was an
>   initialisation artifact, and its chain is in the register. The shrink ladder
>   keeps its mechanism but loses its reason to be pre-emptive.
>
> **The caution:** every floor here is a *lower* bound, since 25 Lloyd iterations
> may not have converged. Quote the held-out column - fit on the train episodes
> and scored on the val ones, `NUM-DATA-SPLIT`, which is the treatment a tokenizer
> actually gets. The gap to the whole-set fit is `NUM-TOK-LEAK` and stays on the
> record, because dropping it would leave "why did the floor move" answerable only
> as *split leak* or *the advantage of scoring a codebook on its own patches*,
> with no way to tell the two apart.
>
> **Unaffected by either refutation**, being properties of the palette or the
> config rather than of the trajectories: `NUM-DATA-FRAMES`,
> `NUM-DATA-VALFRAMES`, `NUM-D96-SIZE`, `NUM-LOAD-TRAIN64`, the 4,096 pixels of a
> frame, and `NUM-TOK-PIXELCOST`. The 96x96 fork's whole evidence base -
> `NUM-TOK-EDGESHARE` - survives both refutations untouched.

### The gate - one command, eight measurements

`python -m mirage.fsq --eval` prints the table and exits nonzero if any pass/fail
row misses.

| # | Measure | Bar | What the requirement actually says |
|---|---|---|---|
| 1 | Held-out PSNR, uint8, over the val frames (`NUM-DATA-VALFRAMES`) | **>= 30.0 dB** (`NUM-BAR-Q1`) | **Q-1** - "tokenizer reconstruction PSNR, held-out, at 64x64". The tokenizer must round-trip a frame it never trained on this well, or every later phase is learning from mush |
| 2 | That PSNR minus the k-means-512 floor on the same frames | **>= `NUM-BAR-ROW2`** - which *is* `NUM-BAR-Q1` minus `NUM-TOK-FLOOR512`, and moves whenever the floor does | **Not a requirement of its own.** It asks one question: is the conv context earning its keep over a codebook that sees each patch in isolation |
| 3 | Token entropy / `log2(codebook)`, all frames (`NUM-DATA-FRAMES`) | **>= 70%** (`NUM-BAR-Q2`) | **Q-2** - "token entropy vs uniform over 512 codes". The codebook must not collapse onto a handful of entries, or the 512-code budget is a fiction and Phase 2 inherits a smaller vocabulary than it was promised |
| 4 | Token cache rows == `shard.frames`, every shard | **exact** | Not a numbered requirement - the Phase 2 handoff indexes tokens by frame, so an off-by-one here is silent and corrupts everything downstream |
| 5 | Re-encode from one checkpoint twice | **bit-identical** | **E-1** - "deterministic sim given a seed", defined as *same as F-4*: same seed and action sequence give bit-identical frames. Encoding is inference, so no backward pass and no cuDNN nondeterminism |
| 6 | F-9 sweep against reconstructions, at `NUM-VAL-TAU` | **<= `NUM-VAL-FRACMAX`** | **F-9** - "frame validator reports block count, arm pose plausibility, palette adherence", accepted at *zero false positives*. Item 6 recalibrated it for decoder output, where "zero off-palette pixels" is unreachable - see `NUM-VAL-RECONFP`. The ground-truth half of the sweep runs alongside as an alignment assert |
| 7 | Edge-pixel PSNR vs flat-pixel PSNR | reported | Not a requirement - **this is the 64/144 fork**, and the numbers are `NUM-TOK-EDGE-R1` and `NUM-TOK-EDGE-R2` |
| 8 | Train-val PSNR gap; live codes at mass > 1e-4 | reported | Not a requirement - overfit and collapse canaries |

Rows 1-6 are pass/fail. **The eval charges row 2 against the recorded `NUM-TOK-FLOOR512`
rather than refitting k-means**, which is a change from the original design and
worth knowing why: the val split is fixed by `data.is_val` over a `data_hash` the
loader already refuses to mismatch, so a refit at seed 0 returns the same value
every time - a constant dressed as a measurement. The disagreement the refit was
meant to surface (row 1 passing while row 2 fails, meaning the val split got
easier) can only happen if the data moves, and a moved `data_hash` fails louder
and earlier. `bench/patch_probe.py` stays the one place k-means lives.

### Build order

Riskiest first, which here means "the thing that could invalidate 300,000
frames" first.

1. ~~`sim/main.cpp` - `offwidth`/`offheight` from config~~ **done 2026-08-28.**
   Every frame regenerated, all 14 blobs byte-identical, `git_sha` the only
   sidecar field that moved. Regeneration throughput is `NUM-DATA-GENFPS`.
   **Regenerate `mirage/fixtures/` too if the `sim` section moves at all** - the
   fixture carries its own `data_hash` and `load_shards` will refuse it
2. ~~`mirage/configs/base96.json`~~ **done 2026-08-28.** `NUM-D96-HASH`,
   `NUM-D96-SIZE` and `NUM-D96-TOKENS`, still inside **R-4** - "dataset on disk,
   <= 20 GB" (`NUM-BAR-R4`). Generation runs at `NUM-D96-GENFPS`, and both earlier
   guesses were wrong; the pixel-count extrapolation was the worse of them, since
   2.25x the pixels cost only 9% more wall clock. **Generation is physics-bound,
   not pixel-bound.** **F-6** - "arm-block contact events exceed 5% of frames" -
   is unchanged at `NUM-DATA-F6`. **F-7** - "block fully occluded in >= 3% of
   frames, counting only occlusion the block recovers from" - falls from
   `NUM-DATA-F7` to `NUM-D96-F7`. The fork buys edge fidelity and spends
   occlusion headroom
3. ~~`mirage/data.py` - `preload`~~ **done 2026-08-28.** Palette indices plus the
   byte LUT, lossless and asserted so: `NUM-LOAD-TRAIN64` and `NUM-LOAD-TRAIN96`
   for the train split, one `NUM-LOAD-LUT` serving both resolutions. Takes the
   palette rgb as an argument, because `validator` imports `data` and importing
   back would be a cycle. Build cost is `NUM-LOAD-BUILD64` / `NUM-LOAD-BUILD96`,
   paid once per run
4. ~~`mirage/logging.py`~~ **done 2026-08-28.** `Run.log(dict)` -> one jsonl line
   in `runs/<run_id>/`, always, each carrying the run id and the caller's hashes;
   W&B only when `wandb_project` is passed. **The W&B mirror is UNVERIFIED** -
   wandb is not installed here, which is the condition the rest was verified
   against. Do not quote it as working until someone runs it
5. ~~`mirage/fsq.py` and `mirage/fsq_eval.py`~~ **done 2026-08-29.** Quantizer,
   encoder/decoder, train loop, PSNR, token cache and the eight-row gate table,
   split at the plan's 500-line trigger. **The gate passes**; two 60-epoch
   checkpoints clear every pass/fail row. `--resume` and a resumable per-epoch
   checkpoint were added after a native-layer `nn.Upsample` use-after-free killed
   two runs mid-flight - diagnosed, not this project's bug, see the verification
   log. **`--resume` was itself broken on CUDA and had never once been executed**
   until it was fixed and tested on 2026-08-29 - `runs.jsonl` row 41
6. ~~F-9 recalibration against reconstructions~~ **done 2026-08-29**, r42.
   `NUM-VAL-TAU` and a new `NUM-VAL-FRACMAX` are in `configs/base.json`;
   `validator_hash` moved and `data_hash` did not. **The obvious recipe was
   wrong**: raising tau past `NUM-VAL-RECONDIST` needs ~160, a ball 6.5 million
   times the calibrated volume, so the verdict changed shape - `> N` off-palette
   pixels rather than `> 0`, because every clean reconstruction has some. tau was
   picked by detection rate at zero false positives and is an **interior**
   optimum. Three of the four thresholds this item expected to write were refuted
   and left out on purpose; the verification log has the table. **Still open:**
   `configs/base96.json` carries an area-scaled `offpalette_px_max` that is
   **unverified** - there is no 96x96 tokenizer yet, and the first one must
   re-run this

### The ladder - four runs, each answering one question

**Budget a gate rung at 60 epochs, not 15.** A 15-epoch rung is a *ranking* run
only - the plan says so, `epochs, the winner | 60 | the run whose PSNR is
quoted`, and item 5 lost a day to reading 15-epoch numbers as gate verdicts. A
60-epoch rung is 60 x `NUM-TOK-EPOCH64`, and **every wall clock measured since is
void**: thermal throttling took one run to 99.2 s/epoch and Modern Standby put a
7.5-hour hole in another. `NUM-PERF-STEP64` is the per-step figure behind it.

| Rung | Config | Answered |
|---|---|---|
| ~~R0~~ | continuous bottleneck, no FSQ, no attention | `NUM-TOK-R0`, **at 15 epochs**. A loose upper bound - its bottleneck is 192 fp32 numbers against R1's 64 tokens x 9 bits. **At 60 epochs it is UNMEASURED**, deliberately: the gate does not need it, and it is an input to the 64-vs-96 fork rather than to item 5 |
| ~~R1~~ | FSQ `[8,8,8]`, no attention | `NUM-TOK-R1-60` at 60 epochs (`runs.jsonl` row 34 for the 15-epoch run). Quantization is *not* the wall - at convergence R1 alone passes every gate row. The "quantization costs 1.322 dB" figure was R0 minus R1 with both under-trained |
| ~~R2~~ | R1 + self-attention on the 8x8 grid | `NUM-TOK-R2-60` at 60 epochs (row 35 for 15). Joint coding buys `NUM-TOK-ATTN` for `NUM-TOK-ATTNPARAM` extra parameters - about a sixth of the plan's own "tied" threshold, so a **measured non-lever for quality**. It buys `NUM-TOK-ATTNENT` of token entropy by decorrelating the three FSQ digits, which no document predicted |
| R3 | **not needed for the gate, and now closed** | The levels ladder is ruled out at 64x64 for the reason given here - **zero of 512 codes have zero count** in any rung, so nothing collapsed - and ruled out at 96x96 for a *different* reason, by arithmetic: `NUM-TOK-SHRINK240-UB`, r45. "If quality is pushed further it is capacity or resolution, and R0 at 60 says which" is **superseded**: resolution was measured directly by the 96x96 arm, which is a better answer than R0 would have given, and **R0 at 60 was deliberately skipped** |
| ~~R1 96~~ | FSQ `[8,8,8]`, no attention, **96x96** | `NUM-TOK-R1-96`. **The fork's answer** - and the run that closed it. See the box below |
| R2 96 | R1 96 + attention | **Never run, and not running it is the result.** `NUM-TOK-MARGSUM-96` is a hard ceiling on any decorrelator and sits below `NUM-BAR-Q2`, so a perfect one still fails. Refuted for the cost of reading a token cache |

> ## The 64/144 fork is RESOLVED: **64x64**. Decided 2026-08-29, `runs.jsonl` r44 + r45.
>
> The 96x96 arm ran. **It wins on Q-1 and fails Q-2**, which no document
> predicted: `NUM-TOK-R1-96`, a gain of `NUM-TOK-FORK` over the 64x64 rung, with
> every other knob identical - and `NUM-TOK-ENT-R1-96` against `NUM-BAR-Q2`, where
> the same architecture at 64x64 clears it.
>
> **One mechanism, two opposite signs.** `NUM-D96-FLATPATCH` of patches are one
> flat colour at 96x96. That raises `NUM-TOK-FLOOR512-96` - easier patches, which
> is also why gate row 2's bar there is a nearly vacuous +0.03 dB - and it
> concentrates the token distribution at the same time. It is **skew, not
> collapse**: `NUM-TOK-SKEW-96`, zero codes unused, 422 live.
>
> **Both remedies this file names are refuted by arithmetic, at zero GPU cost**
> (`bench/entropy_shrink_est.py`, r45):
>
> - **Attention cannot pass, ever.** `NUM-TOK-MARGSUM-96` is a hard ceiling on
>   any method that only decorrelates channels, and it sits below `NUM-BAR-Q2`.
>   The R2 rung at 96x96 was never run, and **not running it is the result.**
> - **The shrink ladder dies at its first step.** `NUM-TOK-SHRINK240-UB` bounds
>   `[8,6,5]` from above and is already under the bar.
>
> Neither lever touches skew, which is the actual failure. The one instrument that
> would is the entropy auxiliary loss ruled out below, **and that ruling stands.**
>
> **Not acted on, deliberately:** by Q-2's stated purpose - that Phase 2 not
> inherit a shrunken vocabulary - 96x96 delivers `NUM-TOK-BITSFRAME-96`, 1.68x the
> bits per frame, with zero dead codes. The statistic fails while the rationale is
> satisfied. **`NUM-BAR-Q2` is not moved.** Moving a bar because a run missed it
> is the failure mode this project's discipline exists to prevent.
>
> **Consequences, all of them good for the schedule.** DiagD stays in reserve,
> F-16 does not promote to M, and CUDA graphs stay a win rather than table stakes.
> Phase 2 is budgeted against the 64-token path. **No further tokenizer runs are
> planned** - the next item on this file is the writeup, and the 96x96 arm goes
> into it as a result rather than as a failed attempt.

---

## Phase 1's two risks, and the lever for each

**Q-1 - "tokenizer reconstruction PSNR, held-out, at 64x64, >= 30 dB" - is a real
risk, and narrower than this file was written around but wider than the
2026-08-28 morning's figure said.** A k-means codebook of 512 entries over real
8x8 patches, fit on the train episodes and scored on the val ones, reaches only
`NUM-TOK-FLOOR512` against `NUM-BAR-Q1`. At 1,024 entries it reaches
`NUM-TOK-FLOOR1024` and still misses. So a tokenizer that looks at one patch in
isolation cannot pass *at any vocabulary measured*, and at the one the plan uses
it misses by `NUM-BAR-ROW2` - that is what the 22x22 receptive field, the
attention layer and a shared decoder have to buy. In physical terms, a wrong
pixel costs `NUM-TOK-PIXELCOST` of squared error on this palette, so the floor
gets ~25 of a frame's 4,096 pixels wrong where the bar allows ~17. **Cut the
error count by a third.**

**This warning is retired by measurement.** At convergence both R1
(`NUM-TOK-R1-60`) and R2 (`NUM-TOK-R2-60`) clear `NUM-TOK-FLOOR512` by well over
the `NUM-BAR-ROW2` needed - so neither is ambiguous and row 2 does not have to
separate anything. Two cautions survive it.
~~**Run-to-run noise is still unmeasured**: no seed has ever been repeated, so
any sentence calling a margin "inside the noise" is asserting something nobody
has checked.~~ **A seed was finally repeated on 2026-08-29.** Two 1-epoch r1 runs
at seed 0, same machine, nothing else changed, landed `NUM-PERF-NOISE` apart,
from nondeterministic cuDNN backward reductions
(`torch.use_deterministic_algorithms` is not set, and setting it would cost
throughput for a property nothing here needs). **Read this carefully, because it
proves less than it looks like it proves.** It is a **1-epoch** figure and so a
*lower bound* on the 60-epoch spread, where 60x more steps of divergence
compound - it does **not** license calling R2's `NUM-TOK-ATTN` over R1
significant.
What it does is put the noise two orders of magnitude below that margin rather
than nowhere, which is a different sentence from the one this paragraph could
write before. **It is not an E-1 or a gate row 5 failure**: both are claims about
the simulator and about encoding from a fixed checkpoint, neither of which runs a
backward pass, and both still hold. Against **E-4** - "every bench number
reproducible from a config hash", accepted when a rerun matches within
`NUM-BAR-E4` - it passes with three orders of magnitude to spare. It stopped
mattering for the gate anyway because the margins got large: a margin above a dB
is safe under any plausible noise, where the sub-0.1 dB miss it replaced was not.
And **row 2 is now row 1 minus a constant**, since the eval charges against the
recorded `NUM-TOK-FLOOR512` rather than refitting, so the two rows can no longer
disagree at all.

**Q-2 - "token entropy vs uniform over 512 codes, >= 70%" - lost its evidence on
2026-08-28, and is now a live question rather than a prediction.** The data does
not force low entropy: only `NUM-TOK-FLAT` of interior cells have a fully flat
receptive field, so the provable ceiling is `NUM-TOK-Q2CEIL` of uniform,
comfortably above `NUM-BAR-Q2`. The reason to expect a collapse anyway was that
k-means kept only a fraction of its centroids alive; **under k-means++
`NUM-TOK-LIVE512` stay live on held-out patches**, so that reason is gone. Q-2 is a statement about
a trained tokenizer's code usage rather than about k-means, so those 26 unused
centroids do not revive it. Nothing says Q-2 will pass either - the exact-patch
distribution still carries just 4.40 bits of the 9 available (`runs.jsonl` row
27; single-use, so it has no register id). If Q-2 misses, shrink
the vocabulary, do not add an entropy loss - an auxiliary loss undoes the reason
FSQ was chosen over VQ.

### The Q-2 shrink ladder, in this order

`[8,8,8]`=512 -> `[8,6,5]`=240 -> `[5,5,5]`=125 -> `[4,4,4]`=64. **Take a step only
when row 3 actually misses**, not pre-emptively - **and at 96x96 it now does**,
which is the first time this condition has fired. The arithmetic does not
obviously reach: `NUM-TOK-ENT-R1-96`'s 4.982 bits over `log2(240)` is 63% and
over `log2(125)` is 71.5%, and **both assume the joint bits survive the shrink**,
which is exactly the thing nobody has measured. At 64x64 row 3 passes and the
ladder stays untouched - the measurement that used to
argue for shrinking was an initialisation artifact. Note the cost, too: the
held-out floor at 240 codes is `NUM-TOK-FLOOR240` against `NUM-TOK-FLOOR512` at
512, so shrinking the vocabulary spends over a dB of Q-1's headroom to buy Q-2
margin it may not need.

Token count never changes, so inference cost is untouched and the output head
gets smaller. **Each step needs a paired LR check**: the straight-through
gradient at zero is 0.858 / 1.001 / 0.668 across those tables, so a levels change
silently rescales the bottleneck learning rate by up to 1.5x and a single-LR
comparison reports a levels result that is partly an LR result.

### If Q-1 misses, the diagnosis is probably already made - and Q-1 did not miss

**Moot as written, kept for what it got right.** Q-1 passes on both 64x64 rungs,
so this section's "if" never fired. Its diagnosis was correct anyway:
`NUM-TOK-EDGESHARE` held up on re-measurement, edge placement *is* what 96x96
fixes, and the arm confirmed it - `NUM-TOK-FORK` of quality, essentially all of it
at edges.

**What it did not anticipate is that fixing edges would break something else.**
More resolution means flatter patches (`NUM-D96-FLATPATCH`), and flatter patches
concentrate the token distribution. The fork was decided against 96x96 on Q-2,
not on Q-1 - see the resolution box above. This section pointed at the right
lever and had no way to see its cost.

Consequences of the 144-token path, all now **avoided**: DiagD stays in reserve
rather than graduating to required, F-16 does not promote to **M**, and CUDA
graphs stay the headline win rather than table stakes. The fork table in
`world_model_architecture.md` has the arithmetic.

---

## Deferred - do not start

Phases 2 through 4. Phase 2's numbers wait on the Phase 1 PSNR; Phase 4's whole
plan derives from the Phase 3 profile, which does not exist yet. Draft a phase's
structural plan when you reach it, not before.

Connected-component labelling, parallel generation, `--replay` mode, and the
single-pass render each have an explicit trigger recorded in the architecture
doc. None of them is a judgement call - wait for the trigger.
