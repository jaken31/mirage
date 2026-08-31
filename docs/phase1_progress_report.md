# Phase 1 progress report - build order items 1 to 4

Session of **2026-08-28**. Covers the merge of `worktree-phase0-debt` and Phase 1
build order **items 1 through 4**. Item 5 (`mirage/fsq.py`) is **not started**;
the handoff for it is the second half of this file.

**This file is a narrative, not a source of truth.** Every number here is
transcribed from `runs.jsonl` rows 27-31 and the verification log at the end of
`world_model_architecture.md`. Those two are authoritative - if a figure here
disagrees with them, they win, and this file is what is wrong. Do not correct a
number here without correcting it there first, or this becomes the third place a
stale figure can hide.

**Erratum, 2026-08-30.** A later branch verified the W&B mirror: offline, on both
credential-failure paths, and with a real networked upload read back through
`wandb.Api()`. The lines below that call the mirror unverified or never executed
are left as transcribed from the 2026-08-28 session. `runs.jsonl` and the
verification log at the end of `world_model_architecture.md` carry the corrected
record and are authoritative.

---

## Outcome

Five commits on `main`, tree clean. All of them are **on `origin/main`**: a push
landed at 18:40 on 2026-08-28, carrying the branch through `92e2193`. It was not
run from this session - `origin/main`'s reflog records it as `update by push`
13 minutes after the last commit was authored.

```
92e2193 feat: add logging.py - jsonl always, W&B behind a flag
be0dcb6 feat: add preload, the palette-index array that fits in memory
1aa1e5b feat: add the 96x96 dataset, and refute both estimates of what it costs
f1f211c feat: take the offscreen render size from config, not the XML
158f76c feat: re-measure Phase 1's six pre-work numbers, and refute the k-means floor
6449414 Merge branch 'worktree-phase0-debt'   (the merge that opened the session)
```

`worktree-phase0-debt` is merged and deleted; `.claude/worktrees/phase0-debt` is
gone. `runs.jsonl` went 26 -> **31 rows**.

The merge needed care for a reason worth remembering: `data/` is gitignored, so
merging code that computes `data_hash 18a76531` onto a checkout holding
`0259947e` shards fails nothing at merge time and surfaces only later, when
`load_shards` refuses them by name. Both doc conflicts were resolved
hunk-by-hunk rather than with `git checkout --theirs`, which would have discarded
main-only content git had already auto-merged outside the conflict markers.

---

## The one finding that changes the plan

> **PARTLY SUPERSEDED the same day, and kept unedited below.** The finding itself
> holds - the initialisation was the cause, not the regeneration. What did not
> hold is the conclusion that **29.02 dB** is the honest floor: that number was
> fit *and* scored on a sample straddling the train/val split. Held out it is
> **28.27 dB**, so gate row 2's bar is **+1.73 dB**, not +0.98, and "1,024 codes
> clear Q-1" is dead - held out, 1,024 reaches 29.39 dB and misses. Quote
> `docs/phase1_item5_report.md` and `runs.jsonl` row 33, not this section.

**The k-means floor was wrong, and not because of the regeneration everyone
suspected.** The six pre-work numbers were flagged as measured against a dataset
that no longer exists. Re-running them says the dataset was never the problem:
the original run never recorded its **initialisation**.

| k | superseded | random init | k-means++ |
|---|---|---|---|
| 240 | 25.67 dB | 25.47 | 27.53 |
| **512 - the floor** | **26.39 dB** | 26.59 | **29.02** |
| 1024 | 27.60 dB | 27.65 | **30.51** |
| live of 512 | 150 | 172 | **512** |

Random seeding reproduces the superseded figures to within 0.2 dB. Every
init-free statistic reproduced - the exact-patch dictionary within 0.03 dB, the
patch entropy within 0.03 bits, the flat-receptive-field share within 0.4 pp, the
edge-error split within 0.1 pp - and that is what pins the cause on the seeding
rather than on the `gear 6 / damping 1.5` change.

The floor is the *best* patch-independent tokenizer, so 29.02 dB is the honest
one. It is still a lower bound: 25 Lloyd iterations may not have converged, and a
better seeding would only raise it.

Consequences, all propagated:

- **Gate row 2's bar is `+0.98 dB`, not `+3.6`.** The 22x22 receptive field, the
  attention layer and the shared decoder have to buy about a quarter of what the
  phase was planned around.
- **A patch-independent tokenizer clears Q-1 at 1,024 codes** (30.51 dB). The
  refutation now holds only at 512, and only by 0.98 dB.
- **The Q-2 collapse risk lost its only evidence.** All 512 centroids stay live.
  The shrink ladder keeps its mechanism but stops being something to do
  pre-emptively.
- **The 96x96 fork is untouched.** 99.95% of the floor's error still sits in the
  36.53% of patches that are not flat.

`bench/patch_probe.py` is the probe. It runs both seedings on purpose, because
the choice turned out to matter more than the thing it was written to check.

---

## Everything else measured

| Claim | Result |
|---|---|
| 64x64 regeneration after the `offwidth` change | **14 of 14 blobs byte-identical**; `git_sha` the only sidecar field that moved |
| Fixture under the ASan build | blobs byte-identical to the committed ones, `GL_RENDERER` names the RTX 5060, no sanitizer report |
| 64x64 full regeneration | **60.2 s at 4,980 fps** - the recorded 45-50 s did not reproduce |
| 96x96 generation | **65.8 s at 4,560 fps**. 2.25x the pixels costs **9% more wall clock** |
| 96x96 set | 7 shards, 300,000 frames, **8.294 GB**, largest shard 1.19 GB, split identical to 64x64, token grid **12x12 = 144** |
| **F-7 at 96x96** | **4.78%**, down from 5.35%. Margin **1.6x** the 3% floor against 1.8x |
| F-6 at 96x96 | **16.63%**, unchanged - contact is a physics fact, not a pixel one |
| F-2, F-8, F-9 at 96x96 | unchanged: 7 colours, worst palette distance 0.75, zero off-palette pixels, mode 2 exact against truth |
| `preload` train split | **1.16 GB** at 64x64 (3.49 raw), **2.62 GB** at 96x96 (7.85 raw), one 7-entry LUT for both |
| `preload` build cost | 37.5 s / 87.8 s, about 90 MB/s both times - packing-bound, not disk-bound |
| `logging.py` | jsonl path verified; **W&B mirror unverified**, wandb absent from this environment |

Two notes that outlive this session:

**Generation is physics-bound, not pixel-bound.** Both prior estimates for the
96x96 cost were wrong, in opposite directions - "~45 s at 6,775 fps" from a
superseded throughput number, and 1.5-2 min from extrapolating the 64x64 cost by
pixel count. Sizing a resolution change by pixel count is refuted; measure it.

**The 96x96 arm spends occlusion headroom to buy edge fidelity.** F-7 falling
from 5.35% to 4.78% is the resolution doing exactly what the arm exists to do - a
bigger frame makes total occlusion rarer - and the fork decision has to carry
that cost, not just the token-count arithmetic.

---

## Claims refuted in passing, and corrected at every site

1. **"Neither existing size check loses its teeth"** (`phase1_structural_plan.md`,
   item 1). False once the change landed. `gl_context.cpp` compares the viewport
   against `model->vis.global.offwidth` and `main.cpp` compares it against
   `cfg.width`; those are now the same number, so the "third corner" is gone and
   it is one fact checked twice. Both checks are kept - they still catch a driver
   returning a viewport other than the one requested - and `main.cpp`'s comment
   now says plainly that it is a duplicate, not corroboration.
2. **The pixel-count sizing rule** for resolution changes, above.
3. **`log()` returned the caller's dict, not the written line.** Pass a numpy
   array and the return value and the bytes on disk disagree while every
   assertion still passes. Now returns `json.loads` of the line it wrote.
4. **`Path` coerced through `str()` writes backslashes on Windows** - unreadable
   paths in a log meant to be read anywhere. Now `as_posix()`.

Items 3 and 4 were found by the self-check of the file they were in, within
minutes of it being written. That is the argument for the self-check convention,
stated as a measurement rather than as a preference.

---

## Judgement calls worth knowing

- **`preload`'s round-trip check runs on the val split** in `python -m
  mirage.data`. Materialising 1.16 GB (2.62 at 96x96) on every self-check run
  would buy no coverage the val pass does not already give - it is the same code
  over 17x the frames - and the size is arithmetic. The full train build is
  measured once into `runs.jsonl`.
- **`preload` takes `palette_rgb` as a parameter**, deviating from the plan's
  signature, because `mirage.validator` imports `mirage.data` and importing
  `load_palette` back would be a cycle.
- **Stale shards went to the Recycle Bin, not `rm`.** The superseded `0259947e`
  set and the regeneration baseline are both recoverable.
- **The fixture sidecar's `git_sha` was reverted** after the ASan smoke test. The
  blobs were byte-identical, so `8735d7f` - the commit that actually generated
  them - remains the honest value, and the alternative was gratuitous churn on
  every future run.

---

## Open, not blocking

- **The 60.2 s vs 45-50 s regeneration gap is unexplained.** Same binary path.
  Could be thermal, could be that the earlier figure was taken without the
  dry-runs. Recorded as a range rather than a budget; not chased.
- **The W&B mirror has never executed.** wandb is absent, which is the very
  condition the rest of `logging.py` was verified against. Do not quote it as
  working until someone installs wandb and runs it.
- **`origin/main` is at `92e2193`, so everything in this report is published.**
  The push did not come from the session that wrote these commits; if that was
  not deliberate, the remote is where to look first.

---

# Handoff - build order item 5, `mirage/fsq.py`

> **SPENT, 2026-08-28. Do not follow this section.** Its first slice - stages 5a,
> 5b and 5c plus rung R0 - is done, and two of the numbers it hands you have
> moved: the k-means floor is **28.27 dB** held out, not 29.02, so gate row 2's
> bar is **+1.73 dB**, not +0.98. See `docs/phase1_item5_report.md`, whose second
> half is the live handoff for stages **5d** and **5e**. Kept here unedited
> because the report above is the record of what this session worked from.

## Start here

`phase1_structural_plan.md`, section **5. `mirage/fsq.py`** - five stages,
roughly 360 lines. It already carries the corrected 29.02 dB / 0.98 dB / 99.95%
figures, so trust it as written.

**Do R0 first, before any FSQ code exists.** Continuous bottleneck, quantizer
bypassed, no attention. It answers whether the architecture can reach 30 dB at
all. If R0 misses, no levels table will help and the encoder is what needs work -
writing the quantizer first would spend a day finding that out the expensive way.

## State you can rely on

```
64x64   data 18a76531aaa8b609   tokenizer 978246d7157caa27   validator 48882ee24c2278b6
96x96   data 35e5b8627987a2bb   tokenizer 6e689ccce0c6d994   validator 15a3e80e6566d114
```

Both datasets on disk: `data/shards` 3.5 GB, `data/shards96` 7.8 GB. Six
self-checks green - `config`, `logging`, and `data` / `validator` against both
configs. Both probes green: `bench/patch_probe.py`, `bench/occlusion_probe.py`.

**The hash tree is chained**: `tokenizer_hash = sha256(data_hash +
canon(tokenizer))`. The two configs have identical `tokenizer` sections and still
get different `tokenizer_hash`es, purely because `data_hash` differs. That is
correct and it is load-bearing for stage 5e's manifest - a 64x64 and a 96x96
token cache can never collide, and a rung's identity already encodes its
resolution.

## What items 1 to 4 handed you

- `preload(shards, index, split, val_fraction, palette_rgb)` -> `(n, h, w)` uint8
  indices plus a `(7, 3)` LUT, rows already flipped. **Note the fifth
  parameter**, and that scaling to `[0, 1]` means `lut[idx] / 255`.
- `Run("r0", {"tokenizer_hash": ...})` -> `runs/<run_id>/metrics.jsonl` plus
  `meta.json`. Every record self-identifies. Never call it inside a timed region.
- Resolution is a config-only change. `mirage/configs/base96.json` needs no code
  change to train against.

## Traps specific to item 5

- **`ConvTranspose2d` is forbidden in the decoder.** Checkerboarding presents as
  misplaced edges, and misplaced edges are the exact signal deciding the 64-vs-144
  fork. It would send the project to 96x96 on a false diagnosis. Nearest-upsample
  plus a 3x3 conv.
- **No `tanh` on the output, no clamp inside the loss.** The scene is pure black
  void and saturated blocks - exactly where `tanh` saturates. Clamp only when
  materialising uint8.
- **No auxiliary loss on FSQ.** No commitment, codebook, EMA or dead-code
  restart. Adding one undoes the reason FSQ was chosen over VQ.
- **The paired LR check is not optional.** The straight-through gradient at zero
  is 0.858 / 1.001 / 0.668 across the levels tables, so a levels change silently
  rescales the bottleneck learning rate by up to 1.5x. Run each variant at the
  base LR *and* at base LR x `0.858 / g_new`.
- **Row 2 now matters more than row 1.** At a 0.98 dB margin, R1 landing near
  29 dB tells you nothing on its own - it could be the tokenizer working or the
  floor being easy to reach. The floor `bench/patch_probe.py` computes is
  k-means++ at seed 0; the eval must recompute it on the same val frames, and if
  the two disagree the val split is not the probe's sample.
- The hyperparameters in the plan's table are **explicitly unverified** - a
  reproducible starting point, expected to move.
- **`runs/` (directory, gitignored) sits beside `runs.jsonl` (notebook,
  tracked).** The trailing slash in `.gitignore` is the only thing separating
  them.

## Inherited environment traps

- `cd` into another directory resets the shell cwd; use `git -C` or re-`cd`.
- Git-Bash paths (`/c/...`) do not resolve in Windows Python.
- `contact_mask` is two fields - bits 0..6 contact, bit 7 scripted. Use
  `mirage.data.contact_bits` / `.scripted`.
- The fixture carries its own `data_hash`; any change to `scene/arm_blocks.xml`
  or the `sim` section invalidates it. The regeneration command is in the
  README's Build section.
- Do not build under `%TEMP%` - MSBuild `FTK1011`, which surfaces as a missing
  C++ compiler.

## Suggested first slice

Stages 5a and 5b plus R0 only: the quantizer written but bypassed, encoder and
decoder, MSE, a train loop over the preloaded 64x64 train split, logging through
`Run`. Report held-out PSNR against 30 dB, plus the train-val gap. **Stop there
and read the number** before writing `codes_to_indices`, the ladder, or the token
cache.
