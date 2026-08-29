# Phase 1 progress report - item 5: the tokenizer, complete

Sessions of **2026-08-28** and **2026-08-29**. Covers build order **item 5** end
to end - 5a the FSQ quantizer, 5b the encoder / `GridAttention` / decoder, 5c the
train loop and held-out PSNR, 5d the R0/R1/R2 ladder, 5e the token cache and the
eight-row gate table - plus the corrections item 5 forced into the open. One of
those corrections is to a claim made earlier in this same file.

**This file is a narrative, not a source of truth.** Every number here is
transcribed from `runs.jsonl` rows **32 to 39** and from the verification log at
the end of `world_model_architecture.md`. Those are authoritative - if a figure
here disagrees with them, they win and this file is wrong. Do not correct a
number here without correcting it there first, or this becomes a third place a
stale figure can hide.

---

## Outcome

**Item 5's gate is met.** Every pass/fail row passes, on **two** checkpoints
either of which could ship.

| # | Measure | R1 @60 | R2 @60 | Bar | |
|---|---|---|---|---|---|
| 1 | Held-out PSNR, uint8, 16,200 val frames | 31.095 dB | **31.182 dB** | >= 30.0 | **PASS** |
| 2 | Minus the 28.27 dB held-out k-means floor | +2.825 | **+2.912** | >= +1.73 | **PASS** |
| 3 | Token entropy / log2(512), all 300,000 frames | 74.1% | **77.6%** | >= 70% | **PASS** |
| 4 | Cache rows == `shard.frames` | exact | exact | exact | **PASS** |
| 5 | Re-encode shard 0 twice | identical | identical | bit-identical | **PASS** |
| 6 | F-9 sweep vs reconstructions | deferred | deferred | - | item 6 |
| 7 | Edge / flat PSNR, edge error share | 26.928 / 43.807, 96.63% | 27.043 / 43.151, 96.00% | reported | - |
| 8 | Train-val gap; live codes at mass > 1e-4 | +1.047; 485/512 | +1.025; 460/512 | reported | - |

**The one thing to carry out of item 5:** the 15-epoch ladder concluded that
*every rung fails Q-1*, and that conclusion was an artifact of training length,
not a property of the architecture. Nothing about the model changed between
29.969 dB and 31.182 dB - only `--epochs 15` became `--epochs 60`.

---

## The four rungs, and what each actually answered

| Rung | Config | Params | Epochs | Held-out | Entropy |
|---|---|---|---|---|---|
| R0 `20260828-185053-r0` | continuous bottleneck | 744,966 | 15 | 31.228 dB | n/a - no tokens |
| R1 `20260828-210620-r1` | FSQ `[8,8,8]` | 744,966 | 15 | 29.905 | 60.8% |
| R2 `20260828-213042-r2` | R1 + attention at 8x8 | 1,008,646 | 15 | 29.969 | 69.0% |
| **R1 `20260829-005439-r1`** | FSQ `[8,8,8]` | 744,966 | **60** | **31.095** | **74.1%** |
| **R2 `20260828-230015-r2`** | R1 + attention at 8x8 | 1,008,646 | **60** | **31.182** | **77.6%** |

### Why 15 epochs was not "nearly converged"

R2 at 15 epochs gained **+0.009 dB** on its final epoch, which reads like a
flattened curve. It is not. The cosine schedule anneals to `lr_floor` **by the
final step**, so a 15-epoch run is frozen at its end *by construction* and its
last-epoch gain says nothing about convergence. A 60-epoch run is not that curve
continued - it is a different trajectory with 4x the steps at useful learning
rates.

`warmup` is 5% of *total* steps, so lengthening the run lengthened warmup too:
the 60-epoch run is **behind** at epoch 1 (24.948 vs 26.183) and only level at
epoch 10. Anyone extrapolating the short curve would have been wrong in both
directions.

**Convergence at 60 is established, not assumed:** val PSNR never turned over,
the best epoch **is** the final epoch, and the last six sit inside +/-0.010 dB.
The train-val gap grew 0.618 -> 1.025 dB and cost nothing on val, so that gap is
a train-side effect and not overfitting.

### The plan already said this, and was not followed

`docs/phase1_structural_plan.md` section 5c says `epochs, the winner | 60 |` and
calls it *the run whose PSNR is quoted*, and its ranking rule says **"if two
rungs land within ~0.5 dB of each other at 15, they are tied and both need the
long run."** R1 and R2 landed **0.063 dB** apart. By the plan's own rule they
were tied and both required the long run. The 15-epoch numbers were ranking runs
read as gate verdicts. **The plan needs no correction; it needed following.**

---

## What attention buys, settled at convergence

The 15-epoch ladder reported +0.063 dB and called `GridAttention` a refuted
quality lever. That comparison was between two models nowhere near converged, so
R1 was rerun at 60 epochs as the control. **The verdict survives:**

| | R1 @60 | R2 @60 | delta |
|---|---|---|---|
| Held-out PSNR | 31.095 dB | 31.182 dB | **+0.087** |
| Parameters | 744,966 | 1,008,646 | +263,680 (+35%) |
| Token entropy | 74.1% | 77.6% | +3.5 pp |
| Flat PSNR | **43.807** | 43.151 | -0.656 |
| Edge PSNR | 26.928 | **27.043** | +0.115 |
| Live codes | **485**/512 | 460/512 | -25 |
| E-1 batch dependence | **none** | 10 in 512,000 | worse |

Matched-epoch deltas stay in +0.05 to +0.11 dB the whole way (+0.052 at epoch 10,
+0.105 at 20, +0.066 at 30, +0.094 at 40, +0.087 final). **+0.087 dB for a 35%
parameter increase is about a sixth of the plan's own "within ~0.5 dB means tied"
threshold**, so attention is a measured non-lever for reconstruction quality at
convergence as well as at 15 epochs.

**R1 alone passes every gate row.** `GridAttention` is not what buys Q-1 or Q-2.
The 744,966-parameter attention-free tokenizer meets item 5 on its own, and it is
the one with no E-1 caveat.

### Which checkpoint to ship

Not decided here, because nothing in item 5 forces it. **R1** is smaller, has no
batch-dependence caveat, better flat PSNR and more live codes. **R2** is
+0.087 dB overall, better on edges, and carries 3.5 pp more token entropy. Both
pass. If Phase 2 has no opinion, **R1 is the simpler artifact** and the one to
prefer on the no-unrequested-complexity rule.

---

## Where the token entropy goes - and a claim this file had to retract

Gate row 3 is one number, but the deficit behind it has two distinct parts, and
they fall out of the same 512-bin count vector the row already sums. A token id
is the mixed-radix number `d0 + 8*d1 + 64*d2`, so the per-channel digit
distributions come free - no GPU, no re-encode:

- **Marginal skew** - a channel's latent sits off centre in the `tanh` bound and
  never reaches most of its 8 levels.
- **Redundancy** - the three channels encode copies of each other.

`fsq_eval.entropy_split` prints both under row 3. The full 2x2, in bits:

| | R1 @15 | R2 @15 | R1 @60 | R2 @60 |
|---|---|---|---|---|
| Joint entropy | 5.476 | 6.210 | 6.670 | **6.987** |
| Sum of marginals | 6.815 | 6.991 | 7.560 | 7.791 |
| **Marginal skew** | 2.185 | 2.009 | 1.440 | **1.209** |
| **Redundancy** | 1.339 | 0.781 | 0.890 | 0.804 |

**The retraction.** With only three of these four cells measured, this session
recorded a clean mechanism into `runs.jsonl` row 37 and the verification log:
*training length cuts skew and leaves redundancy unmoved; attention does the
reverse; two independent levers.* It fit all three cells. **The fourth cell
refutes it** - without attention, training length cuts redundancy
**1.339 -> 0.890**. It was never unmoved.

**Corrected mechanism:**

- On **skew** the levers are roughly *additive*, and training dominates: training
  buys -0.745 bits without attention and -0.800 with it; attention buys -0.176 at
  15 epochs and -0.231 at 60.
- On **redundancy** they are **substitutes**. Whichever acts first takes almost
  the whole reduction and the second adds little: attention -0.558 at 15 epochs
  but only -0.086 at 60; training -0.449 without attention but **+0.023** with
  it. R2@60's redundancy looking untouched by training was never independence -
  it was attention having already taken the removable part at 15 epochs.
- **Redundancy floors near 0.78-0.89 bits in all four cells.** A structural
  correlation between the three FSQ digits survives every combination tried, and
  neither lever reaches it.

**Nothing collapsed.** Zero of 512 codes have zero count in any rung, so "live
codes" at the 1e-4 mass threshold measures thinness, not collapse - and the Q-2
shrink ladder (`[8,8,8]`=512 -> `[8,6,5]`=240), which exists to fix *collapse*,
was aimed at a failure mode that was never happening. It is not needed and was
not used.

**The methodological note worth keeping:** three cells of a 2x2 supported a clean
mechanism that the fourth destroyed, and the fourth cost exactly one run.

---

## The `nn.Upsample` crash, diagnosed

The 2026-08-28 handoff left a crash as *code exonerated, mechanism not*, and said
to investigate for real if it recurred on an idle card. **It recurred** - R1 at
60 epochs, seed 0, nothing else on the card, epoch 6, with
`AttributeError: 'str' object has no attribute 'align_corners'` inside
`nn.Upsample.forward`.

**The traceback's shape is the whole diagnosis.** `Upsample.forward` reads `self`
four times in source order - `self.size`, `self.scale_factor`, `self.mode`,
`self.align_corners` - and a `str` has **none** of the four. Had `self` been a
`str` on entry it would have failed on the **first**. It failed on the
**fourth**. So the frame's `self` slot held a valid `Upsample` for three
bytecodes and a `str` for the next, inside one call, with nothing in the function
assigning to `self`. **That is impossible under Python semantics** - it is a
use-after-free in the native layer, and **not a bug in `fsq.py`**.

**Ruled out:** contention (idle card); a CUDA fault (that raises
`OutOfMemoryError`, not an `AttributeError` on a CPU-side object); free-threading
(`Py_GIL_DISABLED` is 0, `sys._is_gil_enabled()` is True); an unsupported Python
*on paper* (torch 2.9.1+cu130 lists `Python :: 3.14` in its own metadata).
**Not ruled out:** torch's C++ layer, a CUDA library, or CPython 3.14 - n=2 does
not separate them.

**Two patterns for a third occurrence:** both landed in `nn.Upsample`, the
decoder's only parameterless and bufferless module, and both landed in the
epoch-boundary eval, never in training. **Untried and cheap:**
`PYTHONMALLOC=pymalloc_debug` would turn the silent reuse into a reported
use-after-free at ~2x slowdown; the CPython 3.13.13 `uv` already has would say
whether 3.14 is implicated.

**Mitigation shipped, not a fix,** because this is not the project's bug:
`fsq.py` now writes a resumable checkpoint every epoch carrying epoch, optimizer
state and the numpy/torch/cuda RNG states, and takes `--resume RUN_ID`. ~~**A
crash now costs one epoch instead of ninety minutes.**~~ RNG state is *saved*
rather than the epoch *reseeded* deliberately - reseeding would move the data
order and make a resumed rung incomparable to rungs already measured, which
matters when the comparison it exists for is 0.06 dB wide.

> **The struck sentence was false when written, and stayed false for the rest of
> that session.** `--resume` could not survive its own first line on CUDA.
> `train()` loads the checkpoint with `torch.load(map_location=dev)`, which moves
> **every** tensor in it to the GPU - the two RNG states included - and
> `torch.set_rng_state` accepts a CPU `ByteTensor` only, so the restore raised
> `TypeError: RNG state must be a torch.ByteTensor`. **Found 2026-08-29 by
> writing the first test that ever called it**, during the repo-wide audit;
> fixed with `.cpu()` on both restores.
>
> The tell was in this file the whole time and nobody read it as one: the R1
> 60-epoch rerun `20260829-005439-r1` **starts at epoch 0**. It did not resume
> `20260829-004116-r1`, which had died at epoch 6 an hour earlier. The machinery
> built to absorb that crash sat unused while the crash it was built for was
> being absorbed by hand, and this write-up recorded the mitigation as working
> because it had been *written*, not because it had been *run*. That is the
> `CLAUDE.md` failure mode - a claim with no measurement next to it - reproduced
> inside the report whose own subject is that failure mode.
>
> Fixed alongside it: the guard checked **7 of the 10** knobs that change the
> computation. `lr_floor`, `warmup` and `weight_decay` were unchecked, though the
> first two are read by `lr_at` on every step and the third by AdamW, so a resume
> with a different one silently changed the schedule while the flag's help
> promised "every knob must match". Six test cases now cover both fixes,
> including one that resumes a real checkpoint and requires the remaining epoch
> to actually train - **25.66792 dB to 26.51939**. See `runs.jsonl`, "--resume
> was broken on CUDA and had never once been executed".

---

## Two things that voided every wall-clock number

Both 60-epoch runs' timings are unusable. **PSNR is unaffected in both cases** -
neither changes arithmetic.

1. **Thermal throttling.** During R2 at 60 epochs, `nvidia-smi -q -d PERFORMANCE`
   read **SW Thermal Slowdown Active** at **87 C**, with the enforced power limit
   at **85 W** against the 100 W recorded after the chassis fix, and the SM clock
   falling 2565 -> 2340 MHz mid-run. 99.2 s/epoch is a throttled figure and is
   not comparable to the 87.6 s/epoch of record.
2. **The machine suspended itself.** R1 at 60 epochs logged one epoch at 3,051 s
   and another at **27,219 s (7.56 hours)** while its neighbours took 77-98 s.
   Windows `Kernel-Power` 506/507 pairs confirm Modern Standby at 01:07-01:56 and
   again overnight to a wake at 09:48:08. `fsq.py` now calls
   `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` at the top of
   `train` - a per-process request that lapses on exit and changes no user
   setting or power plan.

**The rule this suggests:** on this machine, never quote a wall clock without
checking the throttle counters *and* the Kernel-Power log for the run's window.

---

## Row 7, and the 64 vs 144 fork

Row 7 now has trained-model numbers at convergence, where before it had k-means
and 15-epoch rungs.

| | k-means floor | R2 @15 | R1 @60 | R2 @60 |
|---|---|---|---|---|
| Flat PSNR | - | 42.592 dB | 43.807 | 43.151 |
| Edge PSNR | - | 25.805 | 26.928 | 27.043 |
| Edge share of squared error | 99.95% | 96.56% | 96.63% | **96.00%** |

**Flat regions are solved** - 43 dB. **Longer training moves error off the edges
where attention did not**: R2 went 96.56% -> 96.00% on training length alone,
while attention at 15 epochs had moved the share slightly *up*. The fork's
evidence weakens a little and survives: **96% of all squared error is still edge
geometry**, which is the argument for 96x96, now measured on converged
tokenizers rather than on k-means.

---

## The first slice, as written on 2026-08-28

**Everything from here to "What remains" is the record of the first session** -
stages 5a/5b/5c and the R0 rung - kept as written. Three of its statements have
since moved, and all three are covered above:

- *"15 epochs is enough to rank architectures"* is **wrong for the gate**: at 15
  epochs both quantized rungs fail Q-1 and Q-2, at 60 both pass.
- *"R0 was still climbing at epoch 15"* is **resolved** - so was every rung.
- *"quantization costs 1.322 dB"* was **R0 minus R1 with both under-trained**.
  R2 quantized at 60 epochs reaches 31.182 dB, within 0.046 dB of R0 *continuous*
  at 15. The cost at matched training is bounded *below* by 0.046 dB and above by
  nothing measured. **R0 at 60 is UNMEASURED**, deliberately - the gate does not
  need it, and it is an input to the 64-vs-96 decision rather than to item 5.

The k-means floor correction below (**28.27 dB**, not 29.02) is **unchanged and
still authoritative** - it is the number gate row 2 charges against.

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

# What remains

1. **Item 6** - the F-9 recalibration against reconstructions, which is also gate
   row 6, the one row still deferred. `offpalette_tau` is already in
   `configs/base.json` at 8.0, chosen against *ground-truth* frames where the
   worst palette distance is 0.75; decoder artifacts will push that up and the
   sweep says how far. Both token caches exist to sweep against.
2. **Pick the shipping checkpoint** - R1 `20260829-005439-r1` or R2
   `20260828-230015-r2`. Comparison table above. Nothing in item 5 forces it;
   Phase 2's needs should.
3. **Run-to-run variance is still unmeasured, and is no longer gating.** It
   mattered when R2 missed Q-1 by 0.031 dB; at a +1.182 dB margin it decides
   nothing. If wanted, it is one 60-epoch run at `--seed 1`. Note n=2 gives one
   difference, not a standard deviation. Only `val_psnr_db` is comparable across
   seeds - `train_psnr_db` and `gap_db` use a seed-dependent 4,096-frame
   subsample.
4. **R0 at 60 epochs is the one run that prices quantization at convergence**, and
   it is an input to the 64-vs-96 fork rather than to item 5. Do it with the
   96x96 config when that decision is actually being made, not before.
5. **About 16 stale restatements of `29.02` / `0.98 dB` remain in prose** - see
   "Open, not blocking" above for the site list. Decision-bearing sites are
   corrected; the narrative ones were not swept.
6. **`bench/patch_probe.py`'s receptive-field section** was de-hardcoded from 64
   during this work: it takes `size` from the frames and asserts square. 64x64
   still reproduces 36 of 64 interior cells; 96x96 now gives 100 of 144, where
   the old code would have said 36 of 144.

# Traps

**On this machine, for any run over ~20 minutes:**

- **Modern Standby will suspend a long run.** `fsq.py` now requests keep-awake,
  but only for processes started after that change. A suspend does not corrupt a
  run - it voids its wall clock and stalls the session. Check `Kernel-Power`
  506/507 before trusting any timing.
- **Thermal throttling is invisible in the instantaneous flags.** Read the
  `nvidia-smi -q -d PERFORMANCE` counters and record SM clock and power draw next
  to any timing number.
- **One training run at a time.** 4.8 GiB each against 8.1 GiB. Two fit but
  time-slice; aggregate throughput is unchanged, so there is nothing to gain and
  both timing numbers are voided. Check the card is free, not merely that the
  previous run finished.
- **Expect the `nn.Upsample` crash** roughly one run in three. `--resume RUN_ID`
  costs one epoch. Verify a process is dead by artifact, not by a return value -
  a `TaskStop` once reported success while killing only the wrapper, leaving a
  detached process tree that went on to write a token cache and start a rung.

**On the runs and their artifacts:**

- **Pin the encode batch.** With attention, re-encoding a shard at a different
  batch changes ~2 tokens in 100,000 and breaks E-1. The manifest records `batch`
  and `evaluate` reads it. Without attention this does not happen at all.
- **`runs/20260828-210534-r1` is a 0-epoch smoke test** (9.721 dB) and
  `runs/20260828-184832-r0` a 1-epoch one (26.109 dB). Neither is a rung.
  `runs/20260828-213020-r1` and `runs/20260829-004116-r1` are crashes with no
  `result.json`, left as evidence.
- **Do not filter `metrics.jsonl` on `val_psnr_db` alone** - the `final` record
  matches too. Group on `final` explicitly.
- **Use `python -u`**, or a backgrounded run's log stays empty until it exits.

**Inherited, still true:**

- `cd` into another directory resets the shell cwd; use `git -C` or re-`cd`.
- Git-Bash paths (`/c/...`) do not resolve in Windows Python.
- **Heredocs carrying prose apostrophes break the Bash tool**; use a real file
  write. This bit twice more on 2026-08-29.
- `contact_mask` is two fields - bits 0..6 contact, bit 7 scripted. Use
  `mirage.data.contact_bits` / `.scripted`.
- The fixture carries its own `data_hash`; any change to `scene/arm_blocks.xml`
  or the `sim` section invalidates it.
- Do not build under `%TEMP%` - MSBuild `FTK1011`, which surfaces as a missing
  C++ compiler.
- `runs/` (gitignored directory) sits beside `runs.jsonl` (tracked notebook). The
  trailing slash in `.gitignore` is the only thing separating them.
- **W&B remains unexercised.** `wandb` is absent; `logging.py` says so.
