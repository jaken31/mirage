# Mirage: Agenda

Design lives in `world_model_architecture.md`; requirements in
`world_model_requirements.md`. This file is only the ordered list of what to do
next. Keep it short - delete items as they land, do not accumulate history.

Plain-English versions for non-engineering readers: `timeline.md` (schedule,
gates, risks, with the study plan woven in) and `decision_notes.md` (every
decision with its trigger and fallbacks). Both are derived from the two docs
above - when a decision changes, change it there first.

**Phase 0 is complete.** The gate was met 2026-08-27 and re-met twice on
2026-08-28 - once for the `gear 6 / damping 1.5` scene change, once after a
`data_hash` fix - on all three conditions every time. The set on disk is 300,000
frames at `data_hash` `18a76531`. The write-up is `phase0_report.md`; every
number traces to the verification log at the end of `world_model_architecture.md`.

**Do not quote a figure from before 2026-08-28.** Three `data_hash` values exist,
and only the last describes the data. The chain, with what moved between each, is
in `canonical_numbers.md` under the dataset hash entry. `phase0_report.md` opens
with the same list.

The Phase 0 debt checklist is **closed, 12 of 12**. Three of its items change how
Phase 1 starts:

- **The validator's palette thresholds are calibrated and in config** - item 6,
  landed 2026-08-29. tau (32.0) and the off-palette share limit (8.5449% of a
  frame) now describe *decoder output*, which is a different population from
  renders, not a noisier one: decoded pixels sit up to 154.9 RGB units from the
  palette, rendered ones at most 0.75. Changing either threshold moves
  `validator_hash`, which is the whole reason they were moved out of code - **so
  calibrate once.** Retuning mid-phase splits coherence-horizon results
  into buckets that cannot be compared, and the margin over the worst clean
  reconstruction (314 off-palette pixels) is deliberately thin.
- **The meta record's `contact_mask` is two fields.** Bits 0..6 are block
  contact; bit 7 is scripted-vs-random. `mirage.data.contact_bits` and `.scripted`
  exist. Read the raw byte instead, and the contact rate reads over 50%.
- **`python -m mirage.data` and `python -m mirage.validator` run without a
  dataset**, falling back to a committed 40-frame fixture. A fresh clone or a new
  worktree can check the round-trip and validator requirements before
  generating anything.

Nothing from Phase 0 is pending. The validator threshold calibration, the last
item, landed as Phase 1 item 6 on 2026-08-29.

---

## Phase 1: the tokenizer. Budgeted at 1 week

**Structural plan: `phase1_structural_plan.md`.** What each file owns, the build
order with named APIs, when each file counts as working, and the gotchas.

**Items 1-5 are done. Item 5's gate is met, and the checkpoint is chosen.** All
five pass/fail rows pass on both 60-epoch rungs: R1 `20260829-005439-r1`
(31.095 dB, 74.1% token entropy) and R2 `20260828-230015-r2` (31.182 dB, 77.6%),
against the **30.0 dB** PSNR bar and the **70%** entropy bar. So passing the gate
never decided between them. **Phase 2 takes R1.**

**Decided 2026-08-30**, with the reasoning and the trigger that would reverse it
in `world_model_architecture.md` under "Phase 2 inherits R1, and the encoder keeps
`GroupNorm`". R2's PSNR win, +0.087 dB, is a non-lever. What decides it is that
R2's tokens move with batch size while Phase 3 encodes a seed clip at batch 1,
and that R2 flips twice as many tokens with no local cause
(`bench/token_stability_probe.py`). **The trigger only works one way** - another
rung can promote itself above R1 by passing every gate row with fewer spurious
flips, and nothing demotes R1.

Outcomes are in `phase1_progress_report.md` (items 1-4) and
`phase1_item5_report.md` (item 5, complete). Both are narratives over
`runs.jsonl` and the verification log, which stay authoritative.

> **The lesson item 5 cost the most to learn: 15 epochs is not convergence, and
> the 15-epoch numbers were ranking runs read as gate verdicts.** Every rung
> failed the PSNR bar at 15 epochs and every rung passes at 60, with **no change
> to the architecture** - only `--epochs`. The cosine schedule anneals to
> `lr_floor` by the final step, so a short run is frozen at its end *by
> construction*, and its last-epoch gain says nothing about convergence. The
> structural plan already said to quote the 60-epoch run; it needed following,
> not correcting.

> **The six pre-work numbers were re-measured on 2026-08-28, and the k-means
> floor was refuted - by its own method, not by the regeneration.** The full
> account is in `phase1_item5_report.md` and in `runs.jsonl` (the
> `bench/patch_probe.py` re-measurement and its held-out follow-up). The value
> chain, with what moved at each step, is in `canonical_numbers.md` under the
> k-means floor entry. What belongs in *this* file is three consequences and a
> caution.
>
> - **Gate row 2's bar is derived, not measured.** It is the 30.0 dB bar minus the
>   held-out k-means floor (28.27 dB), so +1.73 dB. It has been published at three
>   different values, because the floor moved twice underneath it and each
>   restatement was correct arithmetic on a stale input. **If the floor moves
>   again, this moves with it**, and every doc holding a copy is wrong that day.
> - **"1,024 codes clear Q-1 outright" is dead.** Held out, 1,024 codes reach only
>   29.39 dB and still miss the 30.0 dB PSNR bar. Read the useful way, that
>   **removes the main reason to regret the fixed 512-code budget** the Phase 2
>   handoff imposes.
> - **The entropy-collapse evidence is gone.** 486 of 512 centroids stay live on
>   held-out patches. The far smaller count that started the worry was an
>   initialisation artifact, and its chain is in the register. The shrink ladder
>   keeps its mechanism but loses its reason to be used in advance.
>
> **The caution:** every floor here is a *lower* bound, since 25 Lloyd iterations
> may not have converged. Quote the held-out column - fit on the 473 train
> episodes and scored on the 27 val ones, which is how a tokenizer is actually
> scored. The gap to the whole-set fit is 0.75 dB and stays on the record.
> Dropping it would leave "why did the floor move" answerable only as *split
> leak* or *the advantage of scoring a codebook on its own patches*, with no way
> to tell the two apart.
>
> **Unaffected by either refutation**, being properties of the palette or the
> config rather than of the trajectories: the 300,000 frames, the 16,200 held-out
> frames, the 96x96 set's 8.294 GB, the 1.162 GB preloaded train split, the 4,096
> pixels of a frame, and the 47,814 cost of one wrong pixel. The 96x96 fork's whole
> evidence base - 99.95% of the floor's error sitting in non-flat patches -
> survives both refutations untouched.

### The gate - one command, eight measurements

`python -m mirage.fsq --eval` prints the table and exits nonzero if any pass/fail
row misses.

| # | Measure | Bar | What the requirement actually says |
|---|---|---|---|
| 1 | Held-out PSNR, uint8, over the 16,200 val frames | **>= 30.0 dB** | **The reconstruction bar** - "tokenizer reconstruction PSNR, held-out, at 64x64". The tokenizer must round-trip a frame it never trained on this well, or every later phase is learning from mush |
| 2 | That PSNR minus the k-means-512 floor on the same frames | **>= +1.73 dB** - which *is* 30.0 dB minus the 28.27 dB floor, and moves whenever the floor does | **Not a requirement of its own.** It asks one question: is the conv context earning its keep over a codebook that sees each patch in isolation |
| 3 | Token entropy / `log2(codebook)`, all 300,000 frames | **>= 70%** | **The entropy bar** - "token entropy vs uniform over 512 codes". The codebook must not collapse onto a handful of entries, or the 512-code budget is a fiction and Phase 2 inherits a smaller vocabulary than it was promised |
| 4 | Token cache rows == `shard.frames`, every shard | **exact** | Not a numbered requirement - the Phase 2 handoff indexes tokens by frame, so an off-by-one here is silent and corrupts everything downstream |
| 5 | Re-encode from one checkpoint twice | **bit-identical** | **Determinism** - "deterministic sim given a seed", defined as the same thing as the simulator's "deterministic given a seed": same seed and action sequence give bit-identical frames. Encoding is inference, so no backward pass and no cuDNN nondeterminism |
| 6 | Validator palette sweep against reconstructions, at tau 32.0 | **<= 8.5449% of a frame off-palette** | **The frame validator** - "frame validator reports block count, arm pose plausibility, palette adherence", accepted at *zero false positives*. Item 6 recalibrated it for decoder output, where "zero off-palette pixels" cannot be reached - every clean reconstruction has some, 314 px on the worst. The ground-truth half of the sweep runs alongside as an alignment assert |
| 7 | Edge-pixel PSNR vs flat-pixel PSNR | reported | Not a requirement - **this is the 64/144 fork**, and the numbers are 26.928 dB edge / 43.807 flat on R1 and 27.043 / 43.151 on R2 |
| 8 | Train-val PSNR gap; live codes at mass > 1e-4 | reported | Not a requirement - early warnings for overfitting and collapse |

Rows 1-6 are pass/fail. **The eval measures row 2 against the recorded 28.27 dB
floor rather than refitting k-means.** That is a change from the original design,
and worth knowing why. The val split is fixed by `data.is_val` over a `data_hash`
the loader already refuses to mismatch, so a refit at seed 0 returns the same
value every time - a constant dressed as a measurement. The disagreement the
refit was meant to catch (row 1 passing while row 2 fails, meaning the val split
got easier) can only happen if the data moves, and a moved `data_hash` fails
louder and earlier. `bench/patch_probe.py` stays the one place k-means lives.

### Build order

Riskiest first, which here means "the thing that could invalidate 300,000
frames" first.

1. ~~`sim/main.cpp` - `offwidth`/`offheight` from config~~ **done 2026-08-28.**
   Every frame regenerated, all 14 blobs byte-identical, `git_sha` the only
   sidecar field that moved. Regeneration runs at 4,980 fps.
   **Regenerate `mirage/fixtures/` too if the `sim` section moves at all** - the
   fixture carries its own `data_hash`, and `load_shards` will refuse it
2. ~~`mirage/configs/base96.json`~~ **done 2026-08-28.** `data_hash` `35e5b862`,
   8.294 GB, 144 tokens per frame - still inside the 20 GB dataset ceiling
   ("dataset on disk, <= 20 GB"). Generation runs at 4,560 fps, and both earlier
   guesses were wrong; the pixel-count extrapolation was the worse of them, since
   2.25x the pixels cost only 9% more wall clock. **Generation is limited by
   physics, not by pixels.** The contact rate ("arm-block contact events exceed
   5% of frames") is unchanged at 16.63%. Recoverable occlusion ("block fully
   occluded in >= 3% of frames, counting only occlusion the block recovers
   from") falls from 5.35% to 4.78%. The fork buys edge fidelity and spends
   occlusion headroom
3. ~~`mirage/data.py` - `preload`~~ **done 2026-08-28.** Palette indices plus the
   byte LUT, lossless and asserted to be: 1.162 GB (64x64) and 2.616 GB (96x96)
   for the train split, with one 7-entry LUT serving both resolutions. It takes
   the palette rgb as an argument, because `validator` imports `data`, and
   importing back would be a cycle. The build costs 37.5 s / 87.8 s, paid once per
   run
4. ~~`mirage/logging.py`~~ **done 2026-08-28.** `Run.log(dict)` -> one jsonl line
   in `runs/<run_id>/`, always, each carrying the run id and the caller's hashes;
   W&B only when `wandb_project` is passed. **The W&B mirror is verified offline**
   as of 2026-08-29, against wandb 0.29.0 - init, three logs, finish, with the
   jsonl path intact underneath. The signature risk that made it UNVERIFIED is
   closed. **The credential path is verified too, as of 2026-08-30, and it did not
   behave as this file predicted.** No key and a wrong key both fail at `Run`
   construction in under a second - but only when stdin/stderr are not a
   terminal. **On a terminal, no key means an untimed prompt**, so a run launched
   from a shell hangs before step 0 instead of failing, and the run-level
   `Settings(login_timeout=)` does not reach it. `Run` now bounds the login with
   `WANDB_LOGIN_TIMEOUT_S` and raises whenever that login does not complete - a
   lapsed prompt is only one of the causes wandb reports the same way - because
   the bounded failure is otherwise *silent*: wandb disables itself and the run
   mirrors nothing. The verification log has all three cases. **The upload is
   verified too, 2026-08-30, and this item is closed.**
   `python -m mirage.logging --network <project>` ran a three-record run against
   the real server. It authenticated, created the `mirage` project under the
   account's default entity, and read its own history back through
   `wandb.Api()` - state `finished`, three rows, values matching - with the local
   jsonl asserted intact underneath. It can be re-run at any time. It needs
   `WANDB_API_KEY` in the environment or a prior `wandb login`, and **never a key
   in a config, a fixture, or this repo**
5. ~~`mirage/fsq.py` and `mirage/fsq_eval.py`~~ **done 2026-08-29.** Quantizer,
   encoder/decoder, train loop, PSNR, token cache and the eight-row gate table,
   split at the plan's 500-line trigger. **The gate passes**; two 60-epoch
   checkpoints clear every pass/fail row. `--resume` and a resumable per-epoch
   checkpoint were added after a native-layer `nn.Upsample` use-after-free killed
   two runs mid-flight - diagnosed, and not this project's bug; see the
   verification log. **`--resume` was itself broken on CUDA and had never once
   been executed** until it was fixed and tested on 2026-08-29 (the `runs.jsonl`
   row "--resume was broken on CUDA and had never once been executed")
6. ~~Validator recalibration against reconstructions~~ **done 2026-08-29.**
   tau (32.0) and a new off-palette share limit (8.5449% of a frame) are in
   `configs/base.json`; `validator_hash` moved and `data_hash` did not. **The
   obvious recipe was wrong.** Raising tau past the worst decoded pixel (154.9 RGB
   units from its palette entry) needs ~160, a ball 6.5 million times the
   calibrated volume. So the verdict changed shape: more than `N` off-palette
   pixels rather than more than zero, because every clean reconstruction has some.
   tau was picked by detection rate at zero false positives, and it is an
   **interior** optimum. Three of the four thresholds this item expected to write
   were refuted and left out on purpose; the verification log has the table.
   **Still unverified, and now moot:** `configs/base96.json` carries an
   area-scaled `offpalette_px_max` that nothing has ever checked. The fork closed
   at 64x64 and no 96x96 tokenizer is planned, so this blocks nothing - but if a
   96x96 rung is ever revived, it must re-run this before quoting a validator
   number

### The ladder - four runs, each answering one question

**Budget a gate rung at 60 epochs, not 15.** A 15-epoch rung is a *ranking* run
only. The plan says so - `epochs, the winner | 60 | the run whose PSNR is
quoted` - and item 5 lost a day to reading 15-epoch numbers as gate verdicts. A
60-epoch rung is 60 x 87.6 s, and **every wall clock measured since is void**:
thermal throttling took one run to 99.2 s/epoch, and Modern Standby put a
7.5-hour hole in another. 37.7 ms is the per-step figure behind it.

| Rung | Config | Answered |
|---|---|---|
| ~~R0~~ | continuous bottleneck, no FSQ, no attention | 31.228 dB, **at 15 epochs**. A loose upper bound - its bottleneck is 192 fp32 numbers against R1's 64 tokens x 9 bits. **At 60 epochs it is UNMEASURED**, on purpose: the gate does not need it, and it is an input to the 64-vs-96 fork rather than to item 5 |
| ~~R1~~ | FSQ `[8,8,8]`, no attention | 31.095 dB at 60 epochs (see `runs.jsonl` for the 15-epoch run). Quantization is *not* the wall - at convergence R1 alone passes every gate row. The "quantization costs 1.322 dB" figure was R0 minus R1 with both under-trained |
| ~~R2~~ | R1 + self-attention on the 8x8 grid | 31.182 dB at 60 epochs. Joint coding buys +0.087 dB for 263,680 extra parameters - about a sixth of the plan's own "tied" threshold, so a **measured non-lever for quality**. It buys +3.5 pp of token entropy by decorrelating the three FSQ digits, which no document predicted |
| R3 | **not needed for the gate, and now closed** | The levels ladder is ruled out at 64x64 because **zero of 512 codes have zero count** in any rung, so nothing collapsed. It is ruled out at 96x96 for a *different* reason, by arithmetic: even the best case for its first step caps entropy at 63.0%, below the bar. "If quality is pushed further it is capacity or resolution, and R0 at 60 says which" is **superseded**: resolution was measured directly by the 96x96 arm, which is a better answer than R0 would have given, and **R0 at 60 was deliberately skipped** |
| ~~R1 96~~ | FSQ `[8,8,8]`, no attention, **96x96** | 32.501 dB. **The fork's answer** - and the run that closed it. See the box below |
| R2 96 | R1 96 + attention | **Never run, and not running it is the result.** At 96x96 the channel marginals sum to 6.078 bits, 67.5%, which is a hard ceiling on any decorrelator and sits below the 70% bar - so even a perfect one fails. Refuted for the cost of reading a token cache |

> ## The 64/144 fork is RESOLVED: **64x64**. Decided 2026-08-29.
>
> The 96x96 arm ran. **It wins on PSNR and fails the entropy bar**, which no
> document predicted: 32.501 dB, a gain of +1.406 dB over the 64x64 rung with
> every other knob identical - and 55.4% token entropy against the 70% bar, where
> the same architecture at 64x64 clears it.
>
> **One mechanism, two opposite effects.** 73.09% of patches are one flat colour
> at 96x96, against 63.47% at 64x64. That raises the held-out k-means floor to
> 29.97 dB - easier patches, which is also why gate row 2's bar there is a nearly
> empty +0.03 dB - and at the same time it concentrates the token distribution. It
> is **skew, not collapse**: 2.922 of the 4.018 missing bits are marginal skew,
> zero codes are unused, and 422 are live.
>
> **Both remedies this file names are ruled out by arithmetic, at zero GPU cost**
> (`bench/entropy_shrink_est.py`):
>
> - **Attention can never pass.** The channel marginals sum to 67.5%, a hard
>   ceiling on any method that only decorrelates channels, and it sits below the
>   70% bar. The R2 rung at 96x96 was never run, and **not running it is the
>   result.**
> - **The shrink ladder dies at its first step.** Its best case, 63.0%, caps
>   `[8,6,5]` from above and is already under the bar.
>
> Neither lever touches skew, which is the actual failure. The one tool that would
> is the entropy auxiliary loss ruled out below, **and that ruling stands.**
>
> **Deliberately not acted on:** by the entropy bar's stated purpose - that Phase
> 2 not inherit a shrunken vocabulary - 96x96 delivers 717.4 bits per frame
> against 426.9, 1.68x, with zero dead codes. The statistic fails while its
> purpose is met. **The 70% bar is not moved.** Moving a bar because a run missed
> it is the failure this project's rules exist to prevent.
>
> **Consequences, all of them good for the schedule.** Diagonal decoding (DiagD)
> stays in reserve, the fused Triton block does not promote to M, and CUDA
> graphs stay a win rather than table stakes. Phase 2 is budgeted against the
> 64-token path. **No further tokenizer runs are planned**, and the 96x96 arm is a
> result rather than a failed attempt.

---

## Phase 1's two risks, and the lever for each

**The PSNR bar ("tokenizer reconstruction PSNR, held-out, at 64x64,
>= 30 dB") was a real risk - narrower than this file was written around, but
wider than the figure on the morning of 2026-08-28 said.** A k-means codebook of
512 entries over real 8x8 patches, fit on the train episodes and scored on the
val ones, reaches only 28.27 dB against the 30.0 dB bar. At 1,024 entries it
reaches 29.39 dB and still misses. So a tokenizer that looks at one patch in
isolation cannot pass *at any vocabulary measured*, and at the one the plan uses
it misses by 1.73 dB. That gap is what the 22x22 receptive field, the attention
layer and a shared decoder have to buy. In physical terms, a wrong pixel costs
47,814 of squared error on this palette, so the floor gets ~25 of a frame's 4,096
pixels wrong where the bar allows ~17. **Cut the error count by a third.**

**This warning is retired by measurement.** At convergence both R1 (31.095 dB) and
R2 (31.182 dB) clear the 28.27 dB floor by well over the 1.73 dB needed, so
neither is ambiguous, and row 2 does not have to separate anything. Two cautions
survive it.

~~**Run-to-run noise is still unmeasured**: no seed has ever been repeated, so
any sentence calling a margin "inside the noise" is asserting something nobody
has checked.~~ **A seed was finally repeated on 2026-08-29.** Two 1-epoch r1 runs
at seed 0, same machine, nothing else changed, landed 0.00167 dB apart, from
nondeterministic cuDNN backward reductions (`torch.use_deterministic_algorithms`
is not set, and setting it would cost throughput for a property nothing here
needs). **Read this carefully, because it proves less than it seems to.** It is a
**1-epoch** figure, so it is a *lower bound* on the 60-epoch spread, where 60x
more steps of divergence compound. It does **not** make R2's +0.087 dB over R1
significant. What it does is put the noise two orders of magnitude below that
margin rather than nowhere, which is a different sentence from the one this
paragraph could write before.

**It is not a determinism or gate row 5 failure**: both are claims about the
simulator and about encoding from a fixed checkpoint, neither of which runs a
backward pass, and both still hold. Against bench reproducibility ("every bench
number reproducible from a config hash", accepted when a rerun matches within 5%)
it passes with three orders of magnitude to spare. It stopped mattering for the
gate anyway, because the margins got large: a margin above a dB is safe under any
plausible noise, where the sub-0.1 dB miss it replaced was not. And **row 2 is
now row 1 minus a constant**, since the eval measures against the recorded 28.27
dB floor rather than refitting, so the two rows can no longer disagree at all.

**The entropy bar ("token entropy vs uniform over 512 codes, >= 70%") lost
its evidence on 2026-08-28, and is now an open question rather than a
prediction.** The data does not force low entropy: only 20.28% of interior cells
have a fully flat receptive field, so the provable ceiling is 94.25% of uniform,
comfortably above 70%. The reason to expect a collapse anyway was that k-means
kept only a fraction of its centroids alive. **Under k-means++, 486 of 512 stay
live on held-out patches**, so that reason is gone. The entropy bar is about a
trained tokenizer's code usage, not about k-means, so those 26 unused centroids
do not revive it. Nothing says it will pass either: the exact-patch distribution
still carries just 4.40 bits of the 9 available (the `bench/patch_probe.py`
re-measurement in `runs.jsonl`; quoted in one place only, so it has no register
entry). If the entropy bar misses, shrink the vocabulary; do not add an entropy
loss - an auxiliary loss undoes the reason FSQ was chosen over VQ.

### The Q-2 shrink ladder, in this order

`[8,8,8]`=512 -> `[8,6,5]`=240 -> `[5,5,5]`=125 -> `[4,4,4]`=64. **Take a step only
when row 3 actually misses**, not in advance - **and at 96x96 it now does**, the
first time this condition has fired. The arithmetic does not obviously reach: the
96x96 entropy of 55.4% is 4.982 bits, which over `log2(240)` is 63% and over
`log2(125)` is 71.5%, and **both assume the joint bits survive the shrink**, which
is exactly what nobody has measured. At 64x64 row 3 passes and the ladder stays
untouched - the measurement that used to argue for shrinking was an
initialisation artifact. Note the cost, too: the held-out floor at 240 codes is
27.09 dB against 28.27 dB at 512, so shrinking the vocabulary spends over a dB of
PSNR headroom to buy entropy margin it may not need.

The token count never changes, so inference cost is untouched and the output head
gets smaller. **Each step needs a paired LR check**: the straight-through
gradient at zero is 0.858 / 1.001 / 0.668 across those tables, so a levels change
silently rescales the bottleneck learning rate by up to 1.5x, and a single-LR
comparison reports a levels result that is partly an LR result.

### If Q-1 misses, the diagnosis is probably already made - and Q-1 did not miss

**Moot as written, kept for what it got right.** The PSNR bar passes on both
64x64 rungs, so this section's "if" never fired. Its diagnosis was correct anyway:
99.95% of the error in non-flat patches held up on re-measurement, edge placement
*is* what 96x96 fixes, and the arm confirmed it - +1.406 dB of quality,
essentially all of it at edges.

**What it did not anticipate is that fixing edges would break something else.**
More resolution means flatter patches (73.09% flat at 96x96), and flatter patches
concentrate the token distribution. The fork was decided against 96x96 on the
entropy bar, not on PSNR - see the resolution box above. This section pointed at
the right lever and had no way to see its cost.

The consequences of the 144-token path are all now **avoided**: DiagD stays in
reserve rather than becoming required, the fused Triton block does not promote
to **M**, and CUDA graphs stay the headline win rather than table stakes. The
fork table in `world_model_architecture.md` has the arithmetic.

---

## Phase 2: the dynamics model. Budgeted at 2 weeks

**Structural plan: `phase2_structural_plan.md`.** What each file owns, the build
order with named APIs, when each item counts as working, the gotchas, and the
numbers already taken so the first item does not re-derive them. Its proposed
gate rows are proposals against the requirements that already exist, not new
requirements.

Phase 2 inherits R1 (named above), the fixed 512-code budget and the 64-token
path. All three are settled, and the trigger that would reverse them only works
one way.

**The mask is decided: block-causal**, by the measurement ordered 2026-09-22
(`bench/mask_probe.py`, 2026-09-23). The dynamics requirement's description
carries a second, dated amendment in `world_model_requirements.md`, and the plan's
"Before item 1" lists the four things it changes.

**Item 1 landed 2026-09-24**: the block-causal layout and the token/action
window sampler, in `mirage/dynamics.py`, self-checked by `python -m mirage.dynamics`
(and so by `python check.py`).

**Item 2 landed 2026-09-24**: the `dynamics` config section names every shape
knob - head count, MLP ratio, RoPE, the untied head and the block-causal mask
beside `d_model` and `n_layers` - so `dynamics_hash` changes when any of them
does.

**Item 3 landed 2026-09-24**: the model, in `mirage/dynamics.py` - the mask
measurement's block-causal model, built from the `dynamics` section, at the
sizing probe's 14,593,152 parameters exactly, with its loss scored at frame
targets only and its causality asserted per block. `dynamics.py` passed the
500-line trigger, so item 6's rollout and gate go in `dynamics_eval.py`.

**Item 4 landed 2026-09-25**: the training loop, in `mirage/dynamics.py`, with the
best checkpoint's full-population score in `mirage/dynamics_eval.py`. bf16, a
per-epoch resumable checkpoint with `--resume RUN_ID`, held-out loss and the
train/val gap every epoch, decision 4a's stopping rule recorded on the run, and
gate row 1's measure every 1,000 steps beside persistence on the same 512
windows. **The one-epoch run answered "no" again**: no sub-epoch point was
above persistence, and the best checkpoint (step 1,000) is 0.19 points below it
on the full population, 86.50% against 86.69%. Peak training VRAM is 2.20 GB at
batch 16. The plan's item 4 and the `runs.jsonl` row have the rest.

**Item 5 landed 2026-09-25**: the token-to-pixel path, `FSQ.indices_to_codes`
and `Tokenizer.decode`, in `mirage/fsq.py`, self-checked by `python -m mirage.fsq`
(every id, both ways, exact) and `python -m mirage.fsq_eval` (R1's cached rows
against `reconstruct`). **It found that `reconstruct` does not encode the way
the cache was written**: the input's memory layout flips 0.33% of tokens under
TF32 at the same batch. The plan's item 5 has the numbers and what it means for
item 6's calibration.

**Next:** item 6, rollout and the gate. **The first real run** -
`python -m mirage.dynamics --train`, 10-epoch cap, under
`systemd-inhibit --what=idle:sleep` - is an overnight window and waits for a
go. Decision 4 chooses a remedy only after it measures the gap.

**Decided 2026-09-21**, with the reasons in the plan: RoPE and the no-shift
interleaving - the irreversible pair, `action[t]` immediately before frame `t`'s
64 tokens, from the same record, with the phase assertion as the acceptance
test; an untied output head; greedy rollout; shared window addressing. Exposure
bias was decided 2026-09-23: no mitigation now, with a named trigger and remedy
order.

**The dynamics model's acceptance test is restated against the persistence
baseline** - decided 2026-09-18, and in `world_model_requirements.md` since
2026-09-22 - and the plan's gate row 1 is written against it. An acceptance test
that a zero-parameter baseline already passes cannot show the model learned
dynamics.

**The risk is data, not compute** (`bench/dyn_size_probe.py`): the model is 15.0x
under Chinchilla-optimal, and one epoch draws 13.8x the dataset from window
overlap alone. Manage overfitting, not throughput. The first run measures the
train/val gap before any remedy is chosen, and stops at 10 epochs, or two epochs
after held-out loss has risen two epochs running, whichever comes first.
Alongside that rule it scores gate row 1's measure against persistence on the
same windows at sub-epoch intervals, and keeps the best checkpoint by it
(amended 2026-09-24).

---

## Deferred - do not start

Phases 3 and 4. Phase 4's whole plan derives from the Phase 3 profile, which does
not exist yet. Draft a phase's structural plan when you reach it, not before.

Connected-component labelling, parallel generation, `--replay` mode, and the
single-pass render each have an explicit trigger recorded in the architecture
doc. None of them is a judgement call - wait for the trigger.
