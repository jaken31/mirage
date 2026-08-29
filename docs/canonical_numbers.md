# Canonical numbers

**The current value of every number this project quotes in more than one place.**
Created 2026-08-29.

Named `canonical_numbers`, not `figures`, because `docs/figures/` holds PNG plots
and `docs/tokenizer_figures.md` is their index - "figure" was already taken.

## What this file is for, and the rule that makes it work

`runs.jsonl` is an **append-only event log**. It answers *what was measured, when,
and by what method*, and it is authoritative for that. It deliberately keeps
superseded values: `gate_row2_bar_db` appears in it as both `0.98` and `1.73`,
because both were once believed and the log does not rewrite history.

That makes it unable to answer a different question - *what is true now* - and
because nothing answered that question, **seventeen documents each kept their own
private answer.** 51% of the distinct figures in the doc corpus appear in two or
more files; the k-means floor moved three times and the bar derived from it was
published as `+3.6`, then `+0.98`, then `+1.73`, each one correct arithmetic on a
stale input.

This file is the current-state view over that log. The split that keeps it honest:

| Doc class | Examples | May state a number inline? |
|---|---|---|
| **Log** | `runs.jsonl`, the verification log in `world_model_architecture.md` | yes - it *is* the evidence |
| **Register** | this file | yes - exactly once, and it says where from |
| **Live** | `AGENDA.md`, both structural plans, `CLAUDE.md`, `README.md`, the derived explainers | **measured or derived values: no** - cite the name and the `NUM-` id. **Chosen bars: yes**, with the id beside them |
| **Frozen** | `phase0_report.md`, `phase1_item5_report.md`, `phase1_progress_report.md`, `tokenizer_figures.md` | yes - they are dated snapshots, and preserving what was believed at the time is their whole job |

The live row is split that way because the split is what the history actually
shows. **Every figure that went stale was measured or derived** - the k-means
floor moved three times, the bar derived from it moved three times, F-7 was
restated, `data_hash` moved twice. **No chosen bar has ever moved**: 30 dB, 70%,
3% and 20 GB are the same today as the day they were written, because changing
one is a decision someone makes on purpose rather than a number that drifts. A
gate table that hides its own threshold behind an id is unusable, and hiding it
would buy nothing.

So: `**>= 30.0 dB** (\`NUM-BAR-Q1\`)` is correct in a live doc. `28.27 dB` is
not - write *the held-out k-means floor (`NUM-TOK-FLOOR512`)*.

A frozen report holding a stale number is **correct behaviour**. A live doc
holding one is the bug this file exists to prevent.

## How to cite

Write the name and the id, never the bare value:

> the held-out k-means floor (`NUM-TOK-FLOOR512`)

`python check.py` verifies that every `NUM-` id cited anywhere exists here, and
that **no live doc cites a superseded id**. That last check is the one that would
have caught all three versions of the gate row 2 bar.

**Source column:** `r<N>` is `runs.jsonl` row N, 1-based, counting from the top of
the file - the convention `phase1_item5_report.md` already uses ("rows 32 to 39").
`VLOG` is the verification log at the end of `world_model_architecture.md`.
`REQ` is `world_model_requirements.md`. `CALC` is computed on demand from config.

---

## Bars and requirements

These are chosen, not measured. They move only by decision.

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-BAR-Q1` | **30.0 dB** | Held-out tokenizer PSNR, uint8, over the val frames | REQ Q-1 | current |
| `NUM-BAR-Q1B` | **35 dB** | The same, stretch tier | REQ Q-1b | current |
| `NUM-BAR-Q2` | **70%** | Token entropy as a share of `log2(codebook)` | REQ Q-2 | current |
| `NUM-BAR-R4` | **20 GB** | Ceiling on a generated dataset on disk | r16 | current |
| `NUM-BAR-F7` | **3%** | Recoverable-occlusion floor | r26 | current |
| `NUM-BAR-P6` | **500 fps** | Generation throughput floor | r23 | current |
| `NUM-BAR-P7` | **30 min** | Ceiling on one 300k-frame epoch | REQ P-7 | current |
| `NUM-BAR-E4` | **5%** | A rerun must match a recorded bench number within this | REQ E-4 | current |
| `NUM-BAR-ROW2` | **+1.73 dB** | Gate row 2's bar. **DERIVED**: `NUM-BAR-Q1` minus `NUM-TOK-FLOOR512` | r33 | **derived - recompute, do not restate** |

> `NUM-BAR-ROW2` is the one entry here that is not independent. It is written down
> only because `mirage/fsq_eval.py` charges against a recorded constant rather than
> refitting. **If `NUM-TOK-FLOOR512` ever moves, this moves with it** - that
> coupling is what three published values of this bar cost the project, and naming
> it here is the whole point of the register.

## Hardware

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-HW-READBACK` | **25.4 us** | `mjr_readPixels` RGB at 64x64, GLFW offscreen | r2 | current |
| `NUM-HW-RENDERREAD` | **75.8 us** | Render plus readback, same conditions | r2 | current |
| `NUM-HW-FP16` | **27.6 TFLOP/s** | fp16 matmul, after the chassis cooling fix | r4 | current |
| `NUM-HW-BW` | **308.3 GB/s** | Measured streaming read, against 384 real peak | r4 | current |
| `NUM-HW-POWER` | **99.86 W** | Enforced power limit after the cooling fix | r4 | current |
| `NUM-HW-VRAM` | **8 GB** | RTX 5060 **Laptop**, sm_120, capability (12,0) | CLAUDE.md | current |

## Dataset - the shipped 64x64 set

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-DATA-HASH64` | **`18a76531`** | `data_hash` of the set on disk | r23, CALC | current |
| `NUM-DATA-FRAMES` | **300,000** | Frames, 500 episodes x 600 steps | r23 | current |
| `NUM-DATA-SHARDS` | **7** | Shards | r23 | current |
| `NUM-DATA-SIZE64` | **3.686 GB** | On disk | r23 | current |
| `NUM-DATA-VALFRAMES` | **16,200** | Held-out frames | r30 | current |
| `NUM-DATA-SPLIT` | **473 / 27** | Train / val episodes of 500, split by hashed episode id | r33 | current |
| `NUM-DATA-COLOURS` | **7** | F-2 distinct byte triples over the whole set - a union, not per frame | r23 | current |
| `NUM-DATA-F6` | **16.63%** | Contact rate. Read the masked byte, not the raw one | r23 | current |
| `NUM-DATA-F7` | **5.35%** | Recoverable occlusion - blocks that return | r26 | current |
| `NUM-DATA-F5RATIO` | **2.15** | F-5 flatness ratio at the shipped physics | r20 | current |
| `NUM-DATA-GENFPS` | **4,980 fps** | Full regeneration, 60.2 s for the set | r29 | current |
| `NUM-DATA-Q4CEIL` | **83.1%** | Ground truth's own action-agreement score, i.e. **Q-4 sits above its own ceiling** | r20 | current |

## Dataset - the 96x96 fork

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-D96-HASH` | **`35e5b862`** | `data_hash` of `configs/base96.json` | CALC | current |
| `NUM-D96-SIZE` | **8.294 GB** | On disk, inside `NUM-BAR-R4` | r29 | current |
| `NUM-D96-TOKENS` | **144** | Tokens per frame, 12x12 grid | r29 | current |
| `NUM-D96-F7` | **4.78%** | Recoverable occlusion at 96x96 - the fork spends occlusion headroom | r29 | current |
| `NUM-D96-GENFPS` | **4,560 fps** | 65.8 s. 2.25x the pixels cost 9% more wall clock: generation is physics-bound | r29 | current |

## Loader and preload

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-LOAD-TRAIN64` | **1.162 GB** | Train split as palette indices, against 3.487 raw | r30 | current |
| `NUM-LOAD-TRAIN96` | **2.616 GB** | Same at 96x96, against 7.847 raw | r30 | current |
| `NUM-LOAD-VAL64` | **66.4 MB** | Val split, 64x64 | r30 | current |
| `NUM-LOAD-LUT` | **7 entries** | The inverting LUT - identical at both resolutions | r30 | current |
| `NUM-LOAD-BUILD64` | **37.5 s** | Cost of building the preload array, once per run | r30 | current |
| `NUM-LOAD-BUILD96` | **87.8 s** | Same at 96x96 | r30 | current |
| `NUM-LOAD-COLD` | **6,804 frames/s** | `WindowSampler(ctx=0)` from a cold page cache - **the reason `preload` exists** | r30 | current |
| `NUM-LOAD-WARM` | **109,682 frames/s** | The same read warm. Training needs ~13,000, so the cold path misses by 2x and the warm path cannot be relied on at a 3.5 GB working set | r30 | current |

## Validator thresholds - the F-9 inputs item 6 recalibrates

> **These four are the ones about to move.** Build order item 6 re-runs the F-9
> sweep against *reconstructions* rather than ground truth, and an FSQ decoder
> emits continuous colour where the renderer emitted seven exact triples. So
> `NUM-VAL-TAU` is expected to rise and `NUM-VAL-FALSEPOS` is expected to stop
> reading zero. **When it lands: update these rows, add the old values to the
> supersession table, and change `configs/base.json` - in that order.** Every doc
> that cites the ids then follows for free.

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-VAL-TAU` | **8.0** | `validator.offpalette_tau` - the RGB Euclidean radius inside which a pixel counts as on-palette. Lives in config, so editing it moves `validator_hash` | r18, r23 | current, **item 6 recalibrates** |
| `NUM-VAL-WORSTDIST` | **0.75** | Worst distance any *ground-truth* pixel sits from its palette entry. `rgba * 255` does not land on integers, which is the whole reason this is not zero | r18, r29 | current |
| `NUM-VAL-HEADROOM` | **11x** | `NUM-VAL-TAU` over `NUM-VAL-WORSTDIST`. The slack item 6 is about to spend on decoder artifacts | r18 | derived |
| `NUM-VAL-FALSEPOS` | **0 px** | Off-palette pixels over every ground-truth frame at that tau - the F-9 acceptance condition | r23 | current, **on ground truth only** |

## Tokenizer - the floor it must beat

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-TOK-FLOOR512` | **28.27 dB** | k-means++ 512 codes, **fit on train episodes, scored on val** | r33 | current |
| `NUM-TOK-FLOOR240` | **27.09 dB** | Same, 240 codes - the cost of the first Q-2 shrink step | r33 | current |
| `NUM-TOK-FLOOR1024` | **29.39 dB** | Same, 1024 codes - **still misses `NUM-BAR-Q1`** | r33 | current |
| `NUM-TOK-LIVE512` | **486 of 512** | Centroids alive on held-out patches | r33 | current |
| `NUM-TOK-LEAK` | **0.75 dB** | Whole-set floor minus held-out floor - the split leak | r33 | current |
| `NUM-TOK-FLAT` | **20.28%** | Interior cells whose 22x22 receptive field is one flat colour | r27 | current |
| `NUM-TOK-Q2CEIL` | **94.25%** | Provable ceiling on token entropy. Docs rounding to 94.3% are not wrong, but quote this | r27 | current |
| `NUM-TOK-EDGESHARE` | **99.95%** | Share of the floor's squared error in the **36.53%** of patches that are not flat | r27 | current |
| `NUM-TOK-PIXELCOST` | **47,814** | Mean squared distance between two distinct palette entries - the cost of one wrong pixel | VLOG | current |

## Tokenizer - the trained rungs

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-TOK-R0` | **31.228 dB** | R0, continuous bottleneck, **15 epochs**. At 60 it is unmeasured | r32 | current |
| `NUM-TOK-R1-60` | **31.095 dB** | R1, FSQ `[8,8,8]`, no attention, 60 epochs | r39 | current |
| `NUM-TOK-R2-60` | **31.182 dB** | R2, R1 plus `GridAttention`, 60 epochs | r37 | current |
| `NUM-TOK-ENT-R1` | **74.1%** | R1 token entropy at 60 epochs | r39 | current |
| `NUM-TOK-ENT-R2` | **77.6%** | R2 token entropy at 60 epochs | r37 | current |
| `NUM-TOK-ATTN` | **+0.087 dB** | What attention buys in quality at convergence - a measured non-lever | r39 | current |
| `NUM-TOK-ATTNPARAM` | **263,680** | What it costs in parameters | r39 | current |
| `NUM-TOK-ATTNENT` | **+3.5 pp** | What it buys in entropy, by decorrelating the FSQ digits | r37, r39 | current |
| `NUM-TOK-PARAMS-R1` | **744,966** | R1 parameter count | r32 | current |
| `NUM-TOK-PARAMS-R2` | **1,008,646** | R2 parameter count | r32 | current |
| `NUM-TOK-EPOCH64` | **87.6 s** | One clean 64x64 epoch, 2,217 steps at batch 128 | r32 | current |
| `NUM-TOK-EDGE-R1` | **26.928 dB** | R1 edge-pixel PSNR at 60 epochs, against 43.807 flat | r39 | current |
| `NUM-TOK-EDGE-R2` | **27.043 dB** | R2 edge-pixel PSNR at 60 epochs, against 43.151 flat | r37 | current |

## Throughput and reproducibility, measured 2026-08-29

| ID | Value | What it is | Source | Status |
|---|---|---|---|---|
| `NUM-PERF-STEP64` | **37.7 ms** | fp32 step, R1, batch 128, 64x64 | r40 | current |
| `NUM-PERF-BF16-64` | **1.46x** | bf16 autocast speedup at 64x64. **Not adopted** - see r40 | r40 | current |
| `NUM-PERF-BF16-96` | **1.25x** | Same at 96x96. Lower, because bf16 misses the bandwidth-bound upsample chain | r40 | current |
| `NUM-PERF-RUNG96-R1` | **2.80 h** | One 60-epoch 96x96 R1 rung, fp32 | r40 | current |
| `NUM-PERF-RUNG96-R2` | **3.14 h** | One 60-epoch 96x96 R2 rung, fp32 | r40 | current |
| `NUM-PERF-BATCHMS` | **0.47 ms** | `_batch`, i.e. **1.2%** of a step. The loader is not the bottleneck | r40 | current |
| `NUM-PERF-NOISE` | **0.00167 dB** | Run-to-run spread, two 1-epoch r1 runs at seed 0. **A 1-epoch lower bound** | r40 | current |

---

## Superseded - kept, because a chain is more useful than an overwrite

Nothing here is wrong to find in a **frozen** doc. Finding one in a **live** doc is
the failure `check.py` looks for.

| Chain | Values, oldest first | Why it moved |
|---|---|---|
| `NUM-TOK-FLOOR512` | 26.39 -> 29.02 -> **28.27 dB** | 26.39 was random k-means init and had no provenance row; k-means++ beat it by 2.6 dB; then 29.02 turned out to be fit *and* scored on a sample straddling the split, worth another 0.75. r27, r33 |
| `NUM-BAR-ROW2` | +3.6 -> +0.98 -> **+1.73 dB** | Purely a consequence of the row above. Three published values, each correct arithmetic on a stale input. **This is the incident this file exists to prevent.** |
| `NUM-DATA-F7` | 19.83% -> **5.35%** | The old counter scored any frame with zero visible pixels. 73% of it was blocks that never return, which Q-6 cannot score object permanence on. r25, r26 |
| `NUM-DATA-HASH64` | `0259947e` -> `219ab0af` -> **`18a76531`** | Original physics; then the `gear 6 / damping 1.5` scene change; then CRLF normalised out of the hash. Do not quote a figure taken before 2026-08-28. r20, r22, r23 |
| `NUM-HW-FP16` | 3.0 -> **27.6 TFLOP/s** | A chassis cooling fix moved the enforced power limit 55 -> 100 W. The throttle *flags* read `Not Active` the whole time it was capped; the evidence was in the counters. r4 |
| `NUM-TOK-LIVE512` | 150 -> **486 of 512** | An initialisation artifact. This was the entire evidence base for the Q-2 collapse risk, and it is gone. r27, r33 |
| `NUM-TOK-FLAT` | 19.96% -> **20.28%** | Re-measured after the regeneration. Confirms rather than refutes. r27 |
| `NUM-TOK-Q2CEIL` | 94.4% -> **94.25%** | Same re-measurement. r27 |

## Not in this register, on purpose

- **Per-run values that are supposed to differ** - a given run's `val_psnr_db`,
  wall clocks, run ids. Those live in `runs.jsonl` and `runs/<id>/result.json`,
  and collapsing them to one "current" value would be a category error.
- **Numbers quoted in exactly one place.** 85% of `runs.jsonl`'s 533 keys are used
  once. An entry for each would be a second copy of the log, which is the disease
  and not the cure. **Add an entry when a number reaches its second live doc**,
  not before.
