# Mirage: Agenda

Design lives in `world_model_architecture.md`; requirements in
`world_model_requirements.md`. This file is only the ordered list of what to do
next. Keep it short - delete items as they land, do not accumulate history.

Plain-English versions for non-engineering readers: `timeline.md` (schedule,
gates, risks, with the study plan woven in) and `decision_notes.md` (every
decision with its trigger and fallbacks). Both are derived from the two docs
above - when a decision changes, change it there first.

**Phase 0 is complete.** Gate met 2026-08-27, all three conditions. Write-up in
`phase0_report.md`; every number traces to the verification log at the end of
`world_model_architecture.md`. Nothing from Phase 0 is pending except the F-9
threshold calibration, which is Phase 1 item 6 by design.

---

## Phase 1: the tokenizer. Budgeted at 1 week

**Structural plan: `phase1_structural_plan.md`.** What each file owns, the build
order with named APIs, done-when per file, and the gotchas.

### The gate - one command, eight measurements

`python -m mirage.fsq --eval` prints the table and exits nonzero if any pass/fail
row misses.

| # | Measure | Bar | Req |
|---|---|---|---|
| 1 | Held-out PSNR, uint8, over the 16,200 val frames | **>= 30.0 dB** | Q-1 |
| 2 | That PSNR minus the k-means-512 floor on the same frames | **>= +3.6 dB** | is the conv context earning its keep |
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
   the proof the XML stayed frozen, and it is the only reason this goes first
2. `mirage/configs/base96.json` - `sim.height`/`width` 96,
   `data.shard_dir` `data/shards96`. Generate it: ~45 s, 8.3 GB
3. `mirage/data.py` - `preload`, returning palette indices plus the byte LUT.
   1.16 GB for the train split instead of 3.49, and lossless
4. `mirage/logging.py` - `log(dict)` to jsonl always, W&B behind a flag
5. `mirage/fsq.py` - quantizer, encoder/decoder, train loop, eval, token cache.
   **Run rung R0 before FSQ is wired in at all**
6. F-9 recalibration against reconstructions, then the thresholds finally go into
   `configs/base.json`

### The ladder - four runs, each answering one question

Training is ~6 min at 15 epochs, so the sweep is an afternoon, not a week.

| Rung | Config | Answers |
|---|---|---|
| R0 | continuous bottleneck, no FSQ, no attention | the architecture's ceiling. **If R0 misses 30 dB, no levels table will help** - fix the encoder |
| R1 | FSQ `[8,8,8]`, no attention | what quantization costs; comparable to the 26.39 dB floor |
| R2 | R1 + self-attention on the 8x8 grid | what joint coding buys |
| R3 | only if R2 falls short | residual blocks, wider channels, or the levels ladder |

Then R1-R3 again at 96x96, which is what turns the fork into a measurement.

---

## Phase 1's two risks, and the lever for each

**Q-1 is the real risk, and it is not close to free.** A k-means codebook of 512
entries over real 8x8 patches reaches **26.39 dB** against the 30 dB bar; 1,024
entries reach only 27.60. So a tokenizer that looks at one patch in isolation
cannot pass, and the whole 3.6 dB has to come from the 22x22 receptive field, the
attention layer, and a shared decoder - codes describing the frame jointly rather
than independently. In physical terms, since a wrong pixel costs 47,814 squared
error on this palette: the floor gets ~38 of 4,096 pixels wrong and the bar is
~17. **Halve the error count.**

**Q-2 is at risk for the opposite reason to the obvious one.** The data does not
force low entropy - only 19.96% of interior cells have a fully flat receptive
field, so the provable ceiling is 94.4% of uniform. What is at risk is that the
scene does not *need* 512 codes: k-means keeps only **150 of 512** centroids
alive. If Q-2 misses, shrink the vocabulary, do not add an entropy loss - an
auxiliary loss undoes the reason FSQ was chosen over VQ.

### The Q-2 shrink ladder, in this order

`[8,8,8]`=512 -> `[8,6,5]`=240 -> `[5,5,5]`=125 -> `[4,4,4]`=64.

Token count never changes, so inference cost is untouched and the output head
gets smaller. **Each step needs a paired LR check**: the straight-through
gradient at zero is 0.858 / 1.001 / 0.668 across those tables, so a levels change
silently rescales the bottleneck learning rate by up to 1.5x and a single-LR
comparison reports a levels result that is partly an LR result.

### If Q-1 misses, the diagnosis is probably already made

**99.86% of the k-means floor's error sits in the 37% of patches that are not a
single flat colour.** Edge placement is the one failure mode 96x96 fixes and the
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
