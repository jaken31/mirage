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
all three conditions every time. The set on disk is 300,000 frames at
**`data_hash 18a76531`**. Write-up in `phase0_report.md`; every number traces to
the verification log at the end of `world_model_architecture.md`.

**Do not quote a figure from before 2026-08-28.** Three hashes exist and only the
last describes the data: `0259947e` was the original physics, `219ab0af` the
scene change, `18a76531` the same scene with line endings normalised out of the
hash. `phase0_report.md` opens with the list.

The Phase 0 debt checklist is **closed, 12 of 12**. Three of its items change how
Phase 1 starts:

- **`validator.offpalette_tau` is already in config**, so item 6 below is a
  calibration, not a plumbing job. Changing a validator threshold now moves
  `validator_hash`, which is the whole reason it was moved out of code.
- **The meta record's `contact_mask` is two fields.** Bits 0..6 are block
  contact, bit 7 is scripted-vs-random. `mirage.data.contact_bits` and
  `.scripted` exist; read the raw byte and F-6 reads over 50%.
- **`python -m mirage.data` and `python -m mirage.validator` run without a
  dataset**, falling back to a committed 40-frame fixture. A fresh clone or a new
  worktree can check F-8 and F-9 before generating anything.

Nothing else from Phase 0 is pending except the F-9 threshold calibration, which
is Phase 1 item 6 by design.

---

## Phase 1: the tokenizer. Budgeted at 1 week

**Structural plan: `phase1_structural_plan.md`.** What each file owns, the build
order with named APIs, done-when per file, and the gotchas.

**Items 1-5 are done. Item 5's gate is met** - all five pass/fail rows pass, on
two checkpoints either of which could ship: R1 `20260829-005439-r1` at
**31.095 dB / 74.1%** and R2 `20260828-230015-r2` at **31.182 dB / 77.6%**,
against bars of 30.0 dB and 70%. Outcomes are in `phase1_progress_report.md`
(items 1-4) and `phase1_item5_report.md` (item 5, complete). Both are narratives
over `runs.jsonl` and the verification log, which stay authoritative.

> **The lesson item 5 cost the most to learn: 15 epochs is not convergence, and
> the 15-epoch numbers were ranking runs read as gate verdicts.** Every rung
> failed Q-1 at 15 epochs and every rung passes at 60, with **no change to the
> architecture** - only `--epochs`. The cosine anneals to `lr_floor` by the final
> step, so a short run is frozen at its end *by construction* and its last-epoch
> gain says nothing about convergence. The structural plan already said to quote
> the 60-epoch run; it needed following, not correcting.

> **The six pre-work numbers were re-measured on 2026-08-28, and the k-means
> floor was refuted - by its own methodology, not by the regeneration.**
> `bench/patch_probe.py`, at the current `data_hash 18a76531`.
>
> The regeneration was the suspect and it was the wrong one. Every init-free
> statistic reproduced: the exact-patch dictionary within 0.03 dB, the patch
> entropy within 0.03 bits, the flat-receptive-field share within 0.4 pp, the
> edge-error split within 0.1 pp. What moved was k-means, and it moved **twice**.
> First, **the original run never recorded its initialisation and that choice is
> worth 2.6 dB** - random seeding reproduces the old numbers, k-means++ beats
> them. Then the k-means++ rerun turned out to be scoring its codebook on a
> sample that **straddled the train/val split**, worth another 0.75 dB.
>
> | | superseded | random init | k-means++, whole set | **k-means++, held out** |
> |---|---|---|---|---|
> | k-means, 240 | 25.67 dB | 25.47 | 27.53 | **27.09** |
> | **k-means, 512 - the floor** | 26.39 dB | 26.59 | 29.02 | **28.27** |
> | k-means, 1024 | 27.60 dB | 27.65 | 30.51 | **29.39** |
> | live of 512 | 150 | 172 | 512 | **486** |
>
> **The held-out column is the one to quote.** It is fit on the 473 train
> episodes and scored on the 27 val ones, which is the treatment a tokenizer
> gets. The whole-set column stays because the 0.75 dB between the two *is* the
> leak, and dropping it would leave "why did 29.02 become 28.27" unanswerable -
> split leak or the advantage of scoring a codebook on its own patches, with no
> way to tell them apart. Both remain lower bounds: 25 Lloyd iterations may not
> have converged. Three consequences, and they are not small:
>
> - **Gate row 2's bar is `+1.73 dB`, i.e. `30.0 - 28.27`** - not the `+3.6` this
>   file was written around, and not the `+0.98` that briefly replaced it. The
>   conv context, the attention layer and the shared decoder have to buy about
>   half of the original figure.
> - **"1,024 codes clear Q-1 outright" is dead.** Held out, 1,024 reaches
>   **29.39 dB** and misses the bar. Read the useful way, that **removes the main
>   reason to regret the fixed 512-code budget** the Phase 2 handoff imposes.
> - **The Q-2 collapse evidence is gone.** 486 of 512 centroids stay live on
>   held-out patches, against the 150 that started the worry. The shrink ladder
>   keeps its mechanism but loses its reason to be pre-emptive.
>
> **Confirmed, unchanged:** flat receptive fields 20.28% (was 19.96%), still
> zero void; the Q-2 ceiling 94.3% (was 94.4%); the edge-error split 99.95% of
> error in 36.53% non-flat patches (was 99.86% / 37%), and **99.98% on the val
> split under the train-fit codebook** - so **the 96x96 fork's evidence survives
> both refutations untouched**. Also unchanged, being properties of the palette or
> the config rather than the trajectories: 300,000 frames, 16,200 val frames,
> 8.294 GB at 96x96, the 1.16 / 3.49 GB preload sizes, 4,096 pixels, and the
> **47,814** squared-error cost of a wrong pixel.

### The gate - one command, eight measurements

`python -m mirage.fsq --eval` prints the table and exits nonzero if any pass/fail
row misses.

| # | Measure | Bar | Req |
|---|---|---|---|
| 1 | Held-out PSNR, uint8, over the 16,200 val frames | **>= 30.0 dB** | Q-1 |
| 2 | That PSNR minus the k-means-512 floor on the same frames | **>= +1.73 dB**, i.e. `30.0 - 28.27`, the **held-out** floor - k-means++ fit on the 473 train episodes and scored on the 27 val ones, measured 2026-08-28. The old `+0.98` came from `29.02`, which was fit *and* scored on a sample straddling the split; `bench/patch_probe.py` prints both, and the 0.75 dB between them is that leak | is the conv context earning its keep |
| 3 | Token entropy / `log2(codebook)`, all 300,000 frames | **>= 70%** | Q-2 |
| 4 | Token cache rows == `shard.frames`, every shard | **exact** | the Phase 2 handoff |
| 5 | Re-encode from one checkpoint twice | **bit-identical** | E-1 |
| 6 | F-9 sweep against reconstructions | **zero false positives, set recorded** | F-9 |
| 7 | Edge-pixel PSNR vs flat-pixel PSNR | reported | **this is the 64/144 fork** |
| 8 | Train-val PSNR gap; live codes at mass > 1e-4 | reported | overfit and collapse canaries |

Rows 1-5 are pass/fail; row 6 is deferred to build order item 6, which is what
calibrates it. **The eval charges row 2 against the recorded 28.27 dB rather than
refitting k-means**, which is a change from the original design and worth knowing
why: the val split is fixed by `data.is_val` over a `data_hash` the loader
already refuses to mismatch, so a refit at seed 0 returns 28.27 every time - a
constant dressed as a measurement. The disagreement the refit was meant to
surface (row 1 passing while row 2 fails, meaning the val split got easier) can
only happen if the data moves, and a moved `data_hash` fails louder and earlier.
`bench/patch_probe.py` stays the one place k-means lives.

### Build order

Riskiest first, which here means "the thing that could invalidate 300,000
frames" first.

1. ~~`sim/main.cpp` - `offwidth`/`offheight` from config~~ **done 2026-08-28.**
   All 300,000 frames regenerated, all 14 blobs byte-identical, `git_sha` the only
   sidecar field that moved. A full 64x64 regeneration is **60.2 s at 4,980 fps**,
   measured on that run - the 45-50 s figure recorded earlier the same day did not
   reproduce. Regenerate `mirage/fixtures/` too if the `sim` section moves at all -
   the fixture carries its own `data_hash` and `load_shards` will refuse it
2. ~~`mirage/configs/base96.json`~~ **done 2026-08-28.** `data_hash 35e5b862`,
   7 shards, **8.294 GB**, 12x12 = 144 tokens, still under R-4's 20 GB.
   Generation **65.8 s at 4,560 fps** - both earlier guesses were wrong, and the
   pixel-count extrapolation was the worse of them: 2.25x the pixels cost **9%
   more wall clock**, so generation is physics-bound, not pixel-bound. F-6 is
   unchanged at 16.63%, but **F-7 fell 5.35% -> 4.78%**, margin 1.6x the floor
   against 1.8x. The fork buys edge fidelity and spends occlusion headroom
3. ~~`mirage/data.py` - `preload`~~ **done 2026-08-28.** Palette indices plus the
   byte LUT, lossless and asserted so. **1.16 GB** for the train split instead of
   3.49 at 64x64, **2.62 instead of 7.85** at 96x96, one 7-entry LUT for both.
   Takes the palette rgb as an argument - `validator` imports `data`. Building it
   costs 37.5 s at 64x64 and 87.8 s at 96x96, paid once per run
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
   log
6. F-9 recalibration against reconstructions, then the verdict thresholds finally
   go into `configs/base.json`. `offpalette_tau` is already there at 8.0, chosen
   against *ground-truth* frames where the worst palette distance is 0.75; decoder
   artifacts will push that up, and the sweep is what says how far

### The ladder - four runs, each answering one question

**Budget a gate rung at 60 epochs, not 15.** A 15-epoch rung is ~22 min and is a
*ranking* run only - the plan says so, `epochs, the winner | 60 | the run whose
PSNR is quoted`, and item 5 lost a day to reading 15-epoch numbers as gate
verdicts. A 60-epoch rung is ~88 min at the clean 87.6 s/epoch, and **every wall
clock measured since is void**: thermal throttling took one run to 99.2 s/epoch
and Modern Standby put a 7.5-hour hole in another.

| Rung | Config | Answered |
|---|---|---|
| ~~R0~~ | continuous bottleneck, no FSQ, no attention | **31.228 dB at 15 epochs.** A loose upper bound - its bottleneck is 192 fp32 numbers against R1's 64 tokens x 9 bits. **At 60 epochs it is UNMEASURED**, deliberately: the gate does not need it, and it is an input to the 64-vs-96 fork rather than to item 5 |
| ~~R1~~ | FSQ `[8,8,8]`, no attention | **29.905 dB at 15, 31.095 dB at 60.** Quantization is *not* the wall - at convergence R1 alone passes every gate row. The "quantization costs 1.322 dB" figure was R0 minus R1 with both under-trained |
| ~~R2~~ | R1 + self-attention on the 8x8 grid | **29.969 dB at 15, 31.182 dB at 60.** Joint coding buys **+0.087 dB for +263,680 parameters** - about a sixth of the plan's own "tied" threshold, so a measured non-lever for quality. It buys **+3.5 pp of token entropy**, by decorrelating the three FSQ digits, which no document predicted |
| R3 | **not needed for the gate** | The levels ladder is specifically ruled out: **zero of 512 codes have zero count** in any rung, so nothing collapsed and the shrink ladder addresses a failure mode that never happened. If quality is pushed further it is capacity or resolution, and **R0 at 60 is the run that says which** |

Then the 96x96 arm, which is what turns the fork into a measurement. Row 7 now
prices it on converged tokenizers: flat regions are solved at **43 dB**, and
**96.00% of all squared error is still edge geometry**.

**Budget it as ~3 hours a rung, not "one training run".** Measured 2026-08-29 on
this card, batch 128: a 60-epoch rung costs **2.80 h at R1 and 3.14 h at R2**,
against 1.39 and 1.48 at 64x64. The fork's other two costs were already known and
are the small ones - one config file and ~66 s of generation - so the line above
that reads "one config file, ~45 s of generation and one training run" was
carrying the expensive term unpriced. Add **87.8 s of preload per rung** and
**2.62 GB of RAM** for the train split, against 1.16 at 64x64; caching the
preloaded array to disk is the recorded answer if that bites, and this is the arm
where it would.

**bf16 autocast is the lever if 3 h a rung is too slow, and it is worth less here
than at 64x64**: 1.25x at 96x96 against 1.4-1.5x at 64x64, because it accelerates
the tensor-core matmuls and not the `nn.Upsample` / `GroupNorm` / `SiLU` chain,
which is bandwidth-bound. **It is deliberately not enabled** - R1 and R2 at 60
epochs differ by 0.087 dB and changing the arithmetic underneath makes every
later rung incomparable to both. If this arm takes it, re-run one baseline in
bf16 and quote that; do not assume bf16 is neutral. Numbers in `runs.jsonl`,
"the training step profiled".

---

## Phase 1's two risks, and the lever for each

**Q-1 is a real risk, and narrower than this file was written around but wider
than the 2026-08-28 morning's figure said.** A k-means codebook of 512 entries
over real 8x8 patches, fit on the train episodes and scored on the val ones,
reaches **28.27 dB** against the 30 dB bar. At 1,024 entries it reaches
**29.39** and still misses. So a tokenizer that looks at one patch in isolation
cannot pass *at any vocabulary measured*, and at the one the plan uses it misses
by **1.73 dB** - that is what the 22x22 receptive field, the attention layer and
a shared decoder have to buy. In physical terms, since a wrong pixel costs 47,814
squared error on this palette: the floor gets ~25 of 4,096 pixels wrong and the
bar is ~17. **Cut the error count by a third.**

**This warning is retired by measurement.** At convergence R1 clears the floor by
**+2.825 dB** and R2 by **+2.912**, against the +1.73 needed - so neither is
ambiguous and row 2 does not have to separate anything. Two cautions survive it.
~~**Run-to-run noise is still unmeasured**: no seed has ever been repeated, so
any sentence calling a margin "inside the noise" is asserting something nobody
has checked.~~ **A seed was finally repeated on 2026-08-29.** Two 1-epoch r1 runs
at seed 0, same machine, nothing else changed, read **25.66625** and **25.66792
dB** - **0.00167 dB apart**, from nondeterministic cuDNN backward reductions
(`torch.use_deterministic_algorithms` is not set, and setting it would cost
throughput for a property nothing here needs). **Read this carefully, because it
proves less than it looks like it proves.** It is a **1-epoch** figure and so a
*lower bound* on the 60-epoch spread, where 60x more steps of divergence
compound - it does **not** license calling R2's +0.087 dB over R1 significant.
What it does is put the noise two orders of magnitude below that margin rather
than nowhere, which is a different sentence from the one this paragraph could
write before. **It is not an E-1 or a gate row 5 failure**: both are claims about
the simulator and about encoding from a fixed checkpoint, neither of which runs a
backward pass, and both still hold. As an E-4 measurement it passes, 0.0065%
against the 5% bar. It stopped mattering for the gate anyway because the margins
got large - a +1.18 dB margin is safe under any plausible noise, where the
-0.031 dB miss it replaced was not. And **row 2 is now row 1 minus a constant**,
since the eval charges against the recorded 28.27 dB rather than refitting, so
the two rows can no longer disagree at all.

**Q-2's risk lost its evidence on 2026-08-28, and is now a live question rather
than a prediction.** The data does not force low entropy - only 20.28% of interior
cells have a fully flat receptive field, so the provable ceiling is 94.3% of
uniform, comfortably above the 70% bar. The reason to expect a collapse anyway was
that k-means kept only 150 of 512 centroids alive; **under k-means++ 486 of 512
stay live on held-out patches**, so that reason is gone. Q-2 is a statement about
a trained tokenizer's code usage rather than about k-means, so those 26 unused
centroids do not revive it. Nothing says Q-2 will pass either - the exact-patch
distribution still carries just 4.40 bits of the 9 available. If Q-2 misses, shrink
the vocabulary, do not add an entropy loss - an auxiliary loss undoes the reason
FSQ was chosen over VQ.

### The Q-2 shrink ladder, in this order

`[8,8,8]`=512 -> `[8,6,5]`=240 -> `[5,5,5]`=125 -> `[4,4,4]`=64. **Take a step only
when row 3 actually misses**, not pre-emptively - the measurement that used to
argue for shrinking was an initialisation artifact. Note the cost, too: the
held-out k-means floor at 240 is 27.09 dB against 28.27 at 512, so shrinking the
vocabulary spends about 1.2 dB of Q-1's headroom to buy Q-2 margin it may not need.

Token count never changes, so inference cost is untouched and the output head
gets smaller. **Each step needs a paired LR check**: the straight-through
gradient at zero is 0.858 / 1.001 / 0.668 across those tables, so a levels change
silently rescales the bottleneck learning rate by up to 1.5x and a single-LR
comparison reports a levels result that is partly an LR result.

### If Q-1 misses, the diagnosis is probably already made

**99.95% of the k-means floor's error sits in the 36.53% of patches that are not a
single flat colour** - re-measured 2026-08-28 and confirmed, the one pre-work
conclusion the re-run left completely intact. Edge placement is the one failure mode 96x96 fixes and the
one thing levels tuning does not, so the arch doc's fork rule points at 96x96
before Phase 1 has run a single step. That is why the 96x96 arm is in the ladder:
it costs one config file, ~45 s of generation and one training run, and it
replaces a prediction about Phase 4's difficulty with a number.

Consequence if the 144-token path is taken: DiagD graduates from reserve to
required, F-16 promotes to **M**, and CUDA graphs stop being the headline win and
become table stakes. The fork table in `world_model_architecture.md` has the
arithmetic.

---

## Deferred - do not start

Phases 2 through 4. Phase 2's numbers wait on the Phase 1 PSNR; Phase 4's whole
plan derives from the Phase 3 profile, which does not exist yet. Draft a phase's
structural plan when you reach it, not before.

Connected-component labelling, parallel generation, `--replay` mode, and the
single-pass render each have an explicit trigger recorded in the architecture
doc. None of them is a judgement call - wait for the trigger.
