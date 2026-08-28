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

> **The six pre-work numbers were re-measured on 2026-08-28, and the k-means
> floor was refuted - by its own methodology, not by the regeneration.**
> `bench/patch_probe.py`, at the current `data_hash 18a76531`.
>
> The regeneration was the suspect and it was the wrong one. Every init-free
> statistic reproduced: the exact-patch dictionary within 0.03 dB, the patch
> entropy within 0.03 bits, the flat-receptive-field share within 0.4 pp, the
> edge-error split within 0.1 pp. What moved was k-means, because **the original
> run never recorded its initialisation and that choice is worth 2.6 dB.**
> Random seeding reproduces the old numbers; k-means++ beats them:
>
> | | superseded | random init, now | **k-means++, now** |
> |---|---|---|---|
> | k-means, 240 | 25.67 dB | 25.47 | **27.53** |
> | **k-means, 512 - the floor** | 26.39 dB | 26.59 | **29.02** |
> | k-means, 1024 | 27.60 dB | 27.65 | **30.51** |
> | live of 512 | 150 | 172 | **512** |
>
> The floor is the *best* patch-independent tokenizer, so 29.02 dB is the honest
> one - and it is still a lower bound, since 25 Lloyd iterations may not have
> converged. Three consequences, and they are not small:
>
> - **Gate row 2's bar is `+0.98 dB`, not `+3.6`.** The conv context, the
>   attention layer and the shared decoder have to buy about a quarter of what
>   this file was written around.
> - **A patch-independent tokenizer nearly passes Q-1, and at 1,024 codes it
>   does.** 30.51 dB clears the bar. The refutation now holds only at 512 codes
>   and only by 0.98 dB.
> - **The Q-2 collapse evidence is gone.** All 512 centroids stay live. The
>   shrink ladder keeps its mechanism but loses its reason to be pre-emptive.
>
> **Confirmed, unchanged:** flat receptive fields 20.28% (was 19.96%), still
> zero void; the Q-2 ceiling 94.3% (was 94.4%); the edge-error split 99.95% of
> error in 36.53% non-flat patches (was 99.86% / 37%) - so **the 96x96 fork's
> evidence is untouched**. Also unchanged, being properties of the palette or
> the config rather than the trajectories: 300,000 frames, 16,200 val frames,
> 8.294 GB at 96x96, the 1.16 / 3.49 GB preload sizes, 4,096 pixels, and the
> **47,814** squared-error cost of a wrong pixel.

### The gate - one command, eight measurements

`python -m mirage.fsq --eval` prints the table and exits nonzero if any pass/fail
row misses.

| # | Measure | Bar | Req |
|---|---|---|---|
| 1 | Held-out PSNR, uint8, over the 16,200 val frames | **>= 30.0 dB** | Q-1 |
| 2 | That PSNR minus the k-means-512 floor on the same frames | **>= +0.98 dB**, i.e. `30.0 - 29.02`, the k-means++ floor re-measured 2026-08-28. Report the floor the eval computes beside it - if they disagree the val split is not the probe's sample | is the conv context earning its keep |
| 3 | Token entropy / `log2(codebook)`, all 300,000 frames | **>= 70%** | Q-2 |
| 4 | Token cache rows == `shard.frames`, every shard | **exact** | the Phase 2 handoff |
| 5 | Re-encode from one checkpoint twice | **bit-identical** | E-1 |
| 6 | F-9 sweep against reconstructions | **zero false positives, set recorded** | F-9 |
| 7 | Edge-pixel PSNR vs flat-pixel PSNR | reported | **this is the 64/144 fork** |
| 8 | Train-val PSNR gap; live codes at mass > 1e-4 | reported | overfit and collapse canaries |

Rows 1-6 are pass/fail. Rows 1 and 2 can disagree, and that is informative: row 1
passing while row 2 fails means the val split got easier, not that the model got
better.

### Build order

Riskiest first, which here means "the thing that could invalidate 300,000
frames" first.

1. `sim/main.cpp` - set `model->vis.global.offwidth`/`offheight` from config
   between `mj_loadXML` and the context. Three lines. **Then regenerate 64x64 and
   prove every blob is byte-identical to what is on disk** - that comparison is
   the proof the XML stayed frozen, and it is the only reason this goes first.
   A full 64x64 regeneration is **45-50 s**, measured 2026-08-28, so the proof is
   cheap. Regenerate `mirage/fixtures/` too if the `sim` section moves at all -
   the fixture carries its own `data_hash` and `load_shards` will refuse it
2. `mirage/configs/base96.json` - `sim.height`/`width` 96,
   `data.shard_dir` `data/shards96`. **8.294 GB**, still under R-4's 20 GB.
   **Measure the wall clock, do not quote one**: the "~45 s" carried here came
   from the superseded 6,775 fps and was never a 96x96 run. Extrapolating the
   measured 64x64 frame cost by pixel count puts it nearer 1.5-2 min, and that is
   an extrapolation too. F-5 is unaffected (same policy, same scene) but **F-6
   and F-7 must be re-verified** - both are measured off rendered pixels
3. `mirage/data.py` - `preload`, returning palette indices plus the byte LUT.
   1.16 GB for the train split instead of 3.49, and lossless
4. `mirage/logging.py` - `log(dict)` to jsonl always, W&B behind a flag
5. `mirage/fsq.py` - quantizer, encoder/decoder, train loop, eval, token cache.
   **Run rung R0 before FSQ is wired in at all**
6. F-9 recalibration against reconstructions, then the verdict thresholds finally
   go into `configs/base.json`. `offpalette_tau` is already there at 8.0, chosen
   against *ground-truth* frames where the worst palette distance is 0.75; decoder
   artifacts will push that up, and the sweep is what says how far

### The ladder - four runs, each answering one question

Training is ~6 min at 15 epochs, so the sweep is an afternoon, not a week.

| Rung | Config | Answers |
|---|---|---|
| R0 | continuous bottleneck, no FSQ, no attention | the architecture's ceiling. **If R0 misses 30 dB, no levels table will help** - fix the encoder |
| R1 | FSQ `[8,8,8]`, no attention | what quantization costs; comparable to the **29.02 dB** floor |
| R2 | R1 + self-attention on the 8x8 grid | what joint coding buys |
| R3 | only if R2 falls short | residual blocks, wider channels, or the levels ladder |

Then R1-R3 again at 96x96, which is what turns the fork into a measurement.

---

## Phase 1's two risks, and the lever for each

**Q-1 is a real risk, but a much narrower one than this file used to say.** A
k-means codebook of 512 entries over real 8x8 patches reaches **29.02 dB** against
the 30 dB bar; 1,024 entries reach **30.51** and clear it. So a tokenizer that
looks at one patch in isolation cannot pass *at the vocabulary the plan uses*, and
it misses by **0.98 dB** - that is what the 22x22 receptive field, the attention
layer and a shared decoder have to buy. In physical terms, since a wrong pixel
costs 47,814 squared error on this palette: the floor gets ~21 of 4,096 pixels
wrong and the bar is ~17. **Cut the error count by a fifth, not by half.**

The corollary is a warning, not a comfort. A margin of 0.98 dB is inside the noise
of a training run, so **R1 landing near 29 dB says nothing** - it could be the
tokenizer working or the floor being easy to reach. Row 2 is what separates them,
and it is now the row that matters more than row 1.

**Q-2's risk lost its evidence on 2026-08-28, and is now a live question rather
than a prediction.** The data does not force low entropy - only 20.28% of interior
cells have a fully flat receptive field, so the provable ceiling is 94.3% of
uniform, comfortably above the 70% bar. The reason to expect a collapse anyway was
that k-means kept only 150 of 512 centroids alive; **under k-means++ all 512 stay
live**, so that reason is gone. Nothing says Q-2 will pass either - the exact-patch
distribution still carries just 4.40 bits of the 9 available. If Q-2 misses, shrink
the vocabulary, do not add an entropy loss - an auxiliary loss undoes the reason
FSQ was chosen over VQ.

### The Q-2 shrink ladder, in this order

`[8,8,8]`=512 -> `[8,6,5]`=240 -> `[5,5,5]`=125 -> `[4,4,4]`=64. **Take a step only
when row 3 actually misses**, not pre-emptively - the measurement that used to
argue for shrinking was an initialisation artifact. Note the cost, too: the
k-means floor at 240 is 27.53 dB against 29.02 at 512, so shrinking the vocabulary
spends about 1.5 dB of Q-1's headroom to buy Q-2 margin it may not need.

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
