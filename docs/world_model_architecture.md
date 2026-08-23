# Mirage: Architecture Decisions

## Context

`mirage` is greenfield: three planning docs, zero code, one commit. The docs
(`world_model_requirements.md`, `world_model_ingredients.md`,
`world_model_learning_roadmap.md`) settle the *ML and systems choices* - MuJoCo +
flat render + EGL, FSQ tokenizer at 512 codes / stride 8, a 384d x 8L decoder-only
dynamics transformer, the 5-rung inference ladder, a 12-week phase order, and the
target hardware (RTX 5060, sm_120, WSL2).

**Superseded 2026-08-21: EGL and WSL2 are out.** WSL2's GPU graphics path is broken
on this machine - no `/dev/dri` node, Mesa falls back to a CPU rasterizer - so the
project runs natively on Windows, where MuJoCo's only viable backend is GLFW.
Measured, not assumed; evidence in `CLAUDE.md` and `bench/egl_probe.py`. Read every
"EGL" below as "the offscreen GL context"; the render-path section records what this
changed.

They say nothing about **how the pieces meet**. Every remaining architectural
question is an interface question. This doc records those decisions so Phase 0 can
start without re-deriving them, and so the few cross-cutting constraints that are
expensive to retrofit are in place from the first commit.

## Constraints that must hold from commit 1

- **No hardcoded shapes.** Image size, token grid, context length, action count
  all derive from config; checkpoints store the config that produced them.
  Required by the live 64x64 / 96x96 fork and by F-13.
- **Model forward takes an explicit attention mask:** `(tokens, cache, mask)`. A
  hardcoded causal mask blocks Diagonal Decoding at the model level regardless of
  how the engine schedules, and DiagD is *required* on the 144-token path.
- **KV cache, when enabled, is statically allocated and masked - never growing.**
  A growing cache breaks CUDA graph capture (roadmap, Track E). This is not "the
  cache is always on": F-15 needs a no-cache full-recompute path as rung 1's
  baseline. Both ship; the cached one is static.
- **Every artifact's filename carries the hash of what determined its contents,
  and every artifact stores the git SHA that produced it.** The hash gives
  staleness-by-construction; the SHA gives provenance when config alone cannot
  explain a result.
- **Sanitizers available from the first C++ file** (`/W4 /WX /fsanitize=address`),
  as a separate build type rather than the default - the production run needs `/O2`
  to hit P-6. MSVC has no UBSan; see the sanitizer-cost section.

## Scope

**Phase 0 plus cross-cutting concerns, designed fully. Seams only for Phases 1-4.**
Per the roadmap's own "just-in-time, not up front" rule.

## C++/Python boundary: standalone binary, no bindings

`gen_data` is an executable writing shards to disk; Python only reads files. No
pybind11.

Decisive argument is **E-3** (ASan/UBSan clean on the full data-generation run):
ASan through a Python extension module requires preloading the ASan runtime before
the interpreter starts, plus typically a suppression file for interpreter
internals, whereas ASan on a standalone binary is `./gen_data`. The magnitude of
that gap is unverified (see verification log) and the decision does not depend on
it - the binary is simpler regardless, and it makes "delete the simulator"
literal: after Phase 0 nothing in the Python tree imports MuJoCo.

## Shard format: structure-of-arrays, three files per shard

| File | Contents | Size @ 300k frames, 64x64 |
|---|---|---|
| `shard_NNN.pixels` | raw `uint8`, contiguous, no header | 3.686 GB |
| `shard_NNN.meta` | fixed-width record per frame, 46 B | 13.8 MB |
| `shard_NNN.json` | frame count, H/W, dtype spec, seed, `data_hash`, git SHA | ~1 KB |

Meta record: `action` u8; `qpos[2]` f32; `block_xy[3][2]` f32; `visible_px[3]` u16;
`contact_mask` u8; `episode_id` u32; `step_idx` u16. **46 B, 0.37% of pixel bytes.**
Under R-4's 20 GB with room for the 96x96 fallback (8.294 GB).

Field justifications, one each - a field with no named consumer does not ship:

| Field | Consumer |
|---|---|
| `action` | F-5 histogram, training targets, Q-4 commanded direction |
| `qpos[2]` | calibration reference for pixel-measured link angles (Q-4, Q-5) |
| `block_xy` | Q-6 position error |
| `visible_px[3]` | F-7 occlusion rate, Q-6 occlusion events |
| `contact_mask` | F-6 contact rate |
| `episode_id`, `step_idx` | prevents a training window spanning a reset |

Three non-obvious choices:

- **Store `visible_px` counts, not an `is_occluded` boolean.** A boolean bakes in a
  threshold; counts let F-7 and Q-6 be re-derived at any threshold without
  regenerating.
- **`episode_id` + `step_idx` are mandatory.** With ctx=15 a window straddling a
  reset is noise, and without these fields the dataloader cannot detect it.
  Absence produces plausible-looking loss curves - the worst failure mode
  available.
- **SoA over AoS, for simplicity not speed.** Measured (numpy + torch 2.9.1): a
  structured field view is non-contiguous as expected (`C_CONTIGUOUS: False`), but
  `torch.from_numpy` on it *succeeds*; the cost is one extra host memcpy before
  H2D, ~197 KB at batch 16, i.e. noise. Separate files win because
  `np.memmap(...).reshape(-1,H,W,3)` needs no stride math and no structured-dtype
  subtleties - not because of throughput.

Sidecar JSON is written *after* the blobs close, so its existence is the commit
marker: a crashed run leaves a shard with no JSON and the loader skips it. No
lockfiles.

## Render path and occlusion measurement (F-7)

Verified against MuJoCo docs: `mjtRndFlag` includes `mjRND_SEGMENT` and
`mjRND_IDCOLOR` ("segmentation with segid+1 color"), set via `mjvScene.flags`. So
per frame: render RGB, render segmentation, count pixels per block segid, store
the counts, discard the mask. Full occlusion is `visible_px == 0` - exactly what
F-7 and Q-6 need, with no baseline-area render. Storing masks would double dataset
size for nothing.

**Two passes. Measured 2026-08-23, and the switch trigger did not fire.** P-6's
500 fps allows 2 ms per frame total; two passes cost 7.6% of it.

| Arm, 64x64, median of 5 x 1000 calls | us | spread |
|---|---|---|
| `mjr_readPixels`, RGB only | 25.4 | 22% |
| `mjr_readPixels`, RGB + depth | 49.6 | 10% |
| `mjr_render` + `mjr_readPixels`, RGB + depth | 75.8 | 65% |

Two passes per frame = 151.6 us of the 2000 us budget, **13x margin**. Dropping
depth would save 2.4% - take it only if `sim/truth.*` turns out not to need depth,
do not contort anything for it. Conditions: RTX 5060 Laptop, GLFW offscreen FBO,
Windows, **P2 / ~2500 MHz** (not P0 - readback is driver-bound, so the P0 rerun is
expected to move this little). Probe: `bench/readback_probe.py`.

**The ~30 ms figure from MuJoCo discussion #2222 does not reproduce here.** The real
fixed cost was MuJoCo's own visual defaults, `shadowsize=4096` and `offsamples=4`.
Setting both to 0 cut `mjr_render` from 316 to 26 us (12x) and readback from 71 to
25 us (2.8x). The readback half of that was an **MSAA resolve** - with `offsamples=4`,
`mjr_readPixels` must collapse a 4x multisampled buffer before it can transfer
anything. So `<quality offsamples="0" shadowsize="0"/>` plus `castshadow="false"`
are **required** in `scene/arm_blocks.xml`, not cosmetic: they are what buys the P-6
margin, and `offsamples="0"` is independently required for F-2's <=24-colour palette.

Two limits on reusing these numbers. Readback cost is per-pixel and per-call, so
25/50 us holds for any 64x64 scene - but the 26 us render is **one sphere and is a
floor**; two arm links plus blocks will cost more, so re-run the render arm once
`arm_blocks.xml` exists. And the 65% spread on the render arm is shrinking signal,
not growing jitter (absolute jitter stayed ~40 us throughout); at 13x margin, record
the spread next to the median and move on.

**Retired triggers, kept for the record.** Readback above ~0.5 ms/call would have
forced the single-pass palette render below; near ~30 ms would have forced a
hand-rolled WGL pbuffer context (`wglCreateContext`, `wglMakeCurrent`,
`WGL_ARB_pbuffer` via `ctypes` on `opengl32.dll`), the 3-to-5-day detour in
`timeline.md`. Neither fired, so neither gets built. OSMesa remains an explicit
anti-choice regardless: it is a CPU rasterizer.

**Single-pass alternative, held in reserve: render segmentation only and synthesize
RGB from a palette lookup.** Under F-2's config - ambient-only light, no shadows,
no textures, no skybox, `offsamples=0` - every pixel belongs to exactly one geom or
the background with no blended edges, so the RGB frame *is* a palette lookup over
the segmentation frame. Benefits: F-2's <=24-colour ceiling becomes structural
rather than tested, F-4 determinism strengthens because no float shading math
touches the pixel path, and occlusion counts stop costing a pass. Cost is
reversibility - a shaded or textured variant needs the second pass back. Switching
later costs one regeneration, ~10 minutes.

Either way, disable visualization decorations (contact points, joint axes) in
`mjvOption` - they pollute both the palette and the segmentation.

## Provenance and artifact naming

Config is a sectioned JSON (`sim`/`data`/`tokenizer`/`dynamics`/`engine`/`validator`),
`nlohmann/json` single header in C++, stdlib `json` in Python. Hashes form a **tree
rooted at `data_hash`**, which is what makes re-tuning a validator threshold and
retraining the tokenizer mutually non-invalidating:

```
data_hash = sha256(canon(sim) + canon(data) + arm_blocks.xml)
├── tokenizer_hash = sha256(data_hash + canon(tokenizer))
│   └── dynamics_hash = sha256(tokenizer_hash + canon(dynamics))
│       └── engine_hash = sha256(dynamics_hash + canon(engine))
└── validator_hash = sha256(data_hash + canon(validator))
```

The scene XML is inside `data_hash` or E-4 has a hole - a bench number from a
different scene is not comparable. `validator_hash` branches off `data_hash`
rather than standing alone because the palette lives in the XML, so a palette
change alters what the measurements *mean*; and it stays out of `dynamics_hash` so
that re-tuning a threshold does not invalidate a checkpoint whose rollouts never
changed. Eval rows are stamped `(dynamics_hash, validator_hash)`, so recomputing
Q-3 under new thresholds is a new row against the same checkpoint. ~20 lines total.

| Artifact | Name | Keyed by |
|---|---|---|
| Shard | `shard_NNN.*`, `data_hash` in sidecar | sim + data config + scene XML |
| Tokenizer checkpoint | `tok_{tokenizer_hash}_{run}.pt` | data + tokenizer config |
| Token cache | `tokens_{ckpt_hash}.bin` | **the checkpoint's bytes** |
| Dynamics checkpoint | `dyn_{dynamics_hash}_{run}.pt` | upstream + dynamics config |
| Eval row | `(dynamics_hash, validator_hash)` | checkpoint + threshold set |
| Bench row | `engine_hash` | full upstream chain + engine flags |

**The token cache is keyed by `ckpt_hash = sha256(checkpoint bytes)`, not by
`tokenizer_hash`.** The config-derived version is a bug: two runs with identical
config but different seeds share a `tokenizer_hash` and produce different tokens,
so Phase 2 would silently load the wrong cache - precisely the failure this scheme
exists to prevent. The checkpoint stores `tokenizer_hash` inside it for provenance.
Content-addressed by weights, provenance-linked by config.

What it buys: **no invalidation code exists** - a stale artifact is unreachable
because nothing computes its name; "already computed?" is `os.path.exists`;
concurrent runs cannot collide; every number walks back to its frames.

**The one real limit: the hash covers config, not code.** Change an
implementation without touching its config and the hash is identical while
behaviour differs. That splits into *detecting* staleness (not cheaply solvable)
and *diagnosing* provenance (solvable). Store the **git SHA inside each artifact**
- not in the hash, which would invalidate everything every commit - and accept
that detection stays manual. No `version` integers: seven hand-bumped ints across
seven sections will rot, and they only ever gave a partial answer to the
unsolvable half.

## Policy mixing is per-episode, not per-frame

A scripted reach needs consecutive steps to complete; coin-flipping per frame
destroys the property the 50/50 mix exists for. F-5's near-uniform action
histogram still holds in aggregate because the random half carries it. Corollary:
keep episodes >> ctx. At 300 steps, 4.7% of frames are lost to boundaries; at 100
steps, 14%.

## Parallelism and replay: both deleted

Single process to start. If P-6's 500 fps is missed, the fallback is N processes
with shard *i* seeded `base + i` - GL contexts are per-thread so processes are the
right unit regardless, and the shard being simultaneously the unit of determinism
and the unit of parallelism keeps this a config change. Whether it is needed
depends on the day-1 numbers. Readback is no longer a candidate cause - it is 7.6%
of the frame budget - so the trigger is now `mj_step` time and end-to-end fps.

No `--replay` mode: F-4 is tested by generating twice with the same seed and
`cmp`-ing the pixel blobs. README states the caveat - bit-exactness holds for a
fixed driver and build, and `/fp:fast` stays off.

## Validator: emit measurements, not verdicts

Python only. The validator is a feature extractor, not a predicate: per frame it
emits a fixed vector, and "the validator failed" is a threshold expression over
that vector, defined in config rather than in code.

| Field | Per | Detects |
|---|---|---|
| `px_count[color]` | colour | missing / dissolved objects |
| `bbox[color]` (4 ints) | colour | position, extent |
| `compactness[color]` | colour | fragmentation, smearing |
| `link_extent[2]` | arm link | Q-5 drift |
| `link_angle[2]` | arm link | **Q-4 action-following** |
| `offpalette_px` | frame | palette violation (F-9) |
| `n_unique_colors` | frame | F-2's <=24 bar, mode 1 only |

Three properties this buys:

- **Q-3 becomes recomputable without re-running rollouts.** Store the vectors per
  rollout frame and the coherence horizon re-derives under any threshold set. Cost:
  500 frames x 100 rollouts x ~15 floats ≈ 3 MB. Same principle as storing
  `visible_px` counts rather than a boolean.
- **Connected-component labelling never becomes a commitment.** It is deferred
  entirely, and adding it later as a *diagnostic on already-failing frames* -
  "block 2 fragmented into 5 pieces, largest 12 px" - invalidates no earlier
  measurement. Designing that against real Phase 2 failures beats designing it now
  against guesses.
- **Q-4 needs no inverse dynamics model.** Actions are joint deltas in
  `{-1,0,+1}`, and the PCA machinery that produces `link_extent` yields
  `link_angle` for free. Action-following is `sign(θ_t+1 − θ_t)` against the
  commanded sign. This is also *better* than an IDM: a model trained on
  ground-truth frames and applied to generated frames carries an unquantified
  generalization gap that a geometric measurement does not.

**Why colour counting and not CC for the verdict** - the argument is correctness,
not effort. F-7 requires full occlusion in >=3% of frames, so partial occlusion is
common. An arm crossing a block splits it into two disconnected same-colour blobs,
and CC reports that as **two blocks** - a phantom object. Suppressing it needs a
"same-colour blobs are one object" merge rule, which reduces CC back to colour
counting with a labelling pass in front. Colour counting is immune by construction.

`compactness` is the fragmentation detector CC would have provided: an intact
flat-rendered box sits near 1.0, confetti near 0.05. **It uses an oriented bbox
from PCA on the mask coordinates, not an axis-aligned one** - both arm links
revolve and a free-joint block rotates when pushed, and an axis-aligned bbox around
a square rotated 45 degrees has 2x the area, giving ~0.5 and colliding with the
occluded case. PCA is rotation-invariant by construction, ~5 lines, and shares
machinery `link_extent` and `link_angle` already need.

### Pixel-to-colour mapping order

**Nearest-palette assignment, not exact RGB equality** (`np.argmin` over squared
distances). If the model generates a block in a slightly off shade, exact equality
counts zero pixels and reports "block missing," conflating two distinct failures.
Nearest-palette separates *palette adherence* from *block count*.

Order matters and is fixed: **raw frame → `n_unique_colors`**, then
**nearest-palette-with-distance → `offpalette_px`, `px_count`, `bbox`,
`compactness`**. Post-mapping, `n_unique_colors` cannot exceed the palette size, so
it must be computed first to serve F-2; and `offpalette_px` means "distance to
nearest palette entry exceeds tau," since after mapping every pixel has a nearest
entry.

`n_unique_colors` is measured in both modes but **thresholded only in mode 1**. F-2's
<=24 bar is a statement about the renderer, and an FSQ decoder emits hundreds of
unique RGB values by construction, so the field is large on reconstructions and
rollouts without that being a violation. General rule: **the measurement vector is
generous; the verdict expression is minimal and built from uncorrelated fields.**
Under a verdict of "fail if any measurement exceeds its threshold," false-positive
risk rises monotonically with threshold count, so every added threshold makes F-9's
"zero false positives" strictly harder - and `offpalette_px` already responds to the
same colour-mush failure.

### Two modes, split by whether ground truth exists

This is what makes F-9's acceptance test executable rather than a judgement call.

| Mode | Inputs | Used in |
|---|---|---|
| `measure_with_truth(frame, meta)` | `visible_px`, `block_xy`, `qpos` from shard meta | Phase 0 |
| `measure_pixels_only(frame)` | nothing | Phases 2, 3, 4 |

F-9's "zero false positives on ground-truth frames" is literally a threshold sweep
of mode 2 against mode 1. Without both modes there is no procedure for that
criterion. Occlusion-aware calibration is legitimate because mode 1 knows when a
block is occluded; mode 2 inherits the calibrated number and needs no gate.

**Calibrate twice.** Ground-truth frames are perfectly rendered while Q-3's inputs
carry decoder artifacts, so zero false positives on clean frames does not bound the
rate on generated ones. Sweep against Phase 0 ground truth *and* against Phase 1
tokenizer reconstructions - Phase 1 produces those anyway for Q-1's PSNR, they carry
real artifacts, and they exist before Phase 2 needs them.

Build order: measurement vector plus both modes (~50 lines, no deps) → calibrate
against Phase 0 truth → recalibrate against Phase 1 reconstructions → ship.

## Observability: jsonl always, W&B behind a flag

Three layers, only the last two of which were in question:

| Layer | What | Status |
|---|---|---|
| 1 | Human-authored run log: config hash, change, number, conclusion | **E-5, Must.** A lab notebook, one line per run. No tool produces it. |
| 2 | Machine metrics stream (loss, lr, grad norm, VRAM) | jsonl, always |
| 3 | Live curves, image history, cross-run comparison | W&B, when flagged |

One `log(dict)` writes jsonl unconditionally and forwards to `wandb.log` when
enabled, ~15 lines. jsonl stays the source of truth so history survives independently
of the account, and F-17's ladder table is a jsonl-to-markdown script - no UI
produces it.

Three disciplines, all tied to Must-tier numbers:

- **Nothing logs inside the Phase 4 timed region.** P-2 (p99 <= 40 ms) and P-4
  (p99/p50 <= 1.3) are tail-latency requirements; an occasional 1 ms hitch is 2.5%
  of the frame budget and lands in p99, not p50. The bench loop records into a
  preallocated array and writes after.
- **Disable W&B's system-metrics collector for bench runs** (`x_disable_stats` or
  `mode="disabled"`). Verified against the W&B source: a background process samples
  CPU/GPU/disk/network on its own schedule, and sampling the GPU during a
  tail-latency measurement inflates p99 quietly.
- **R-1 / R-2 use `torch.cuda.max_memory_allocated()`, not W&B's GPU metric.** W&B
  samples driver-reported usage on an interval and can miss a transient peak; the
  allocator high-water mark cannot.

Offline mode is available if the network is a problem - verified to write a durable
local transaction log and sync later.

## Seams for Phases 1-4 (sketch only)

Model is a pure function `(tokens, cache, mask) -> logits`; engine owns the loop,
cache memory, graph capture, and decode schedule. That line assigns the ladder:
**KV cache, CUDA graphs, and DiagD are engine**; **INT8 and the fused Triton block
are model** (a weight transform at load, and a module swap). F-15's five flags
therefore never produce 32 code paths - they select two submodule implementations
plus three engine behaviours.

Two consequences traced during review:

- **Inference needs a seed clip, so the dataset outlives the simulator.** F-14
  drives the model with MuJoCo absent, but the model needs ~15 frames of context
  before predicting anything, and those come from a real episode. "Delete the
  simulator" is exact; "delete the data" is not. R-5's 10-second cold start
  includes loading that clip.
- **On the 144-token path, "plain MHA" must mean `F.scaled_dot_product_attention`.**
  See the fork table below - materialized attention caps training batch size there.

## The 64x64 vs 96x96 fork

Decided by Phase 1's PSNR. Recorded so the decision is mechanical when the number
arrives.

| | 64 tok/frame | 144 tok/frame |
|---|---|---|
| Per-token-step budget @ 30 fps | 520.8 us | 231.5 us |
| Compute floor | 95 us | 127 us |
| Launch overhead (~80 kernels) | 400 us | 400 us |
| **Naive total** | **~495 us (95% of budget)** | **~527 us (228% of budget)** |
| After CUDA graphs (~50 us overhead) | ~145 us (28%) | ~177 us (77%) |
| DiagD reduction | 15 diagonals vs 64 = 4.27x | 23 vs 144 = **6.26x** |
| DiagD status | reserve | **required** |
| Context seq len | ~1024 | ~2176 |
| KV cache (8L, 384d, bf16) | 12.58 MB | 26.74 MB |
| Inference VRAM vs R-2's 2 GB | non-issue | non-issue |
| Dataset on disk vs R-4's 20 GB | 3.686 GB | 8.294 GB |
| Token cache in RAM | 38.4 MB | 86.4 MB |

Materialized-attention memory, computed. The constraint is a **batch ceiling**, not
a pass/fail - nothing in the requirements specifies a training batch size:

| | batch 8 | batch 16 | batch 32 |
|---|---|---|---|
| seq 1024 | 0.81 GB | 1.61 GB | 3.22 GB |
| seq 2176 | 3.64 GB | **7.27 GB** | 14.55 GB |

Against R-1's 7.5 GB, the 144 path caps batch at ~16 with materialized attention
and the 64 path at 32+. **At inference the fork is a latency decision; at training
it is a batch-size decision** - which SDPA removes entirely.

**The compression ratio is provably identical on both paths**, exactly
`1536/9 = 170.667:1`, verified by computation: with stride 8 and 512 codes,
`3HW / ((HW/64) * 9/8)` cancels to a resolution-independent constant. So 96x96
cannot give the tokenizer an easier ratio. It helps because the scene becomes
**oversampled relative to its feature size** - a block occupying exactly one 8x8
patch at 64x64 spans ~2.25 patches at 96x96, so each patch carries a simpler piece
of an edge.

**Therefore the fallback only helps if Q-1 fails on edge placement.** If the failure
looks like colour drift or lost global structure, 96x96 will not fix it and the real
lever is the FSQ levels table. Worth knowing before spending a week on it.

## Sanitizer cost, and keeping E-3 cheap

E-3 requires ASan and UBSan clean on the **full** data-generation run. **MSVC
provides ASan only.** Microsoft lists `/fsanitize=undefined` and `/fsanitize=leak`
among sanitizers it may ship *later*, so on this toolchain E-3's UBSan half has no
implementation and there is no leak detection either. Two ways out, undecided:
add a clang-cl configuration carrying UBSan, or restate E-3 as ASan-clean plus the
UBSan-class defects `/W4 /WX` already rejects at compile time.

Published overheads, for the ASan half that does exist:

| Sanitizer | Measured |
|---|---|
| ASan | **73% average** on SPEC CPU2006 (original paper), worst case 2.6x; 103% average in a later study |
| UBSan, full set | **up to 228%** on SPEC2006 across all 19 sub-sanitizers; 59% average elsewhere - relevant only if the clang-cl build happens |

Both are pure-CPU benchmarks, and this loop is dominated by GPU render and pixel
readback which ASan does not instrument, so end-to-end slowdown should land well
below them. Measure; do not extrapolate from SPEC.

- **`/Zi` plus linker `/DEBUG` are not optional.** MSVC emits `C5072` without debug
  info and `/WX` makes it fatal; the LLVM symbolizer also needs the PDB, or the
  stacks are useless and you pay the run twice.
- **`/fsanitize=address` is incompatible with `/RTC`, incremental linking,
  edit-and-continue, and PGO.** The middle two are Debug defaults, and CMake sets
  `CMAKE_MSVC_DEBUG_INFORMATION_FORMAT` to EditAndContinue on new projects - leaving
  it there is a silent failure.
- **The ASan runtime DLL has to be findable.** MSVC links it dynamically regardless
  of `/MD` or `/MT`, and it sits next to `cl.exe`, on `PATH` only inside a Developer
  prompt. `sim/CMakeLists.txt` copies it beside the binary.
- **Collect every finding in one pass** for the gate run rather than stopping at the
  first. MSVC honours a subset of `ASAN_OPTIONS` and documents `ASAN_SAVE_DUMPS`;
  confirm the continue-on-error spelling against the MSVC docs before relying on it.
- **MuJoCo stays uninstrumented** (prebuilt `mujoco.lib`) - not your code, and the
  case that matters is still caught, because ASan intercepts `malloc` globally so
  MuJoCo writing past the end of *your* buffer trips a redzone.
- **No leak checking here**, so the LSan suppression file a Linux run would need
  against the NVIDIA driver does not apply. A process that exits after one shard is
  also where leaks matter least.
- **Phase 0 is the only phase where this is clean: no CUDA in it.** ASan and CUDA
  coexist poorly - another reason to confine all C++ to Phase 0.

## Benchmark validity: the measurement environment is a requirement, not a detail

Measured on the target machine (RTX 5060 **Laptop** GPU, driver 610.62, torch
2.9.1+cu130, sm_120 confirmed): device-copy bandwidth came out at 66-77 GB/s and
fp16 matmul at 1.4 TFLOP/s. **Both numbers are invalid as statements about the
hardware** - the diagnostic says why:

| Signal | Observed under sustained load | Expected |
|---|---|---|
| `power.draw` | **6.16 W** | 60-115 W for this class |
| `clocks.current.sm` | 1102 MHz | max 3090 MHz (36%) |
| `clocks.current.memory` | 9001 MHz | max 12001 MHz (75%) |
| `pstate` | **P4** | P0 |
| `temperature.gpu` | 55-59 C | cool, so not thermal |

6 W during an 8192-cube fp16 matmul is conclusive: the device never left a
low-power state. Two-second sustained warmups did not ramp it, and ~50 desktop
processes (browsers, Teams, Slack, Discord, NVIDIA and Overwolf overlays) hold GPU
contexts under WDDM. **The ingredients doc's 448 GB/s is therefore neither
confirmed nor refuted - it is unverified**, and every compute-floor term in the
fork table derives from it.

The finding that matters is not the bandwidth number:

1. **E-4 ("rerun matches within 5%") is unachievable without pinning the power
   state.** A GPU drifting between P4 and P0 varies by more than an order of
   magnitude, not 5%.
2. **P-4 (p99/p50 <= 1.3) can be blown by a single boost transition mid-run**,
   independent of any kernel. On a laptop with dynamic boost this is the most
   likely cause of a failed P-4, and it would look exactly like a kernel problem.
3. **Therefore the bench harness must record `pstate`, `clocks.current.sm`,
   `clocks.current.memory`, `power.draw`, and `temperature.gpu` on every timing
   row, and refuse to run when `pstate != P0` or clocks sit below a configured
   fraction of max.** A timing without its clock state is not reproducible, so by
   E-4 it is not a valid number. This is cheap - one `nvidia-smi` query per run,
   outside the timed region per the observability rules.
4. **Preconditions for any Phase 3/4 measurement, documented in the README:** mains
   power, Windows/NVIDIA performance profile, `nvidia-smi --lock-gpu-clocks` and
   `--lock-memory-clocks` for the run (needs admin), desktop GPU consumers closed.
5. **Day-1 addition: re-measure bandwidth and matmul throughput at P0** before
   trusting any budget term. If the laptop cannot *sustain* P0 under a 1000-frame
   run, that is a project-level constraint worth knowing in week 1 rather than
   week 10 - it would compress every margin in the fork table simultaneously.

Environment otherwise checks out: sm_120 is confirmed at capability (12, 0), and
CUDA 13.0 clears the docs' 12.8 requirement for Blackwell.

**Corrected 2026-08-21:** the earlier claim that "WSL2 Ubuntu-24.04 is running, so
Phase 0's EGL path has a home" was wrong. WSL2 has a home for CUDA, not for
rendering: `dxgkrnl` fails adapter enumeration (`Ioctl failed: -22`), no `/dev/dri`
node appears, and Mesa reports `Falling back to surfaceless swrast without DRM`.
Two restarts, `wsl --update --pre-release`, and every platform and driver override
were tried. Everything now runs natively on Windows. The C++ toolchain there - MSVC
or MinGW - is unverified and is the remaining Phase 0 prerequisite.

## Feasibility cross-check, and what the project argues

Computed from the roadmap's own MineWorld reference:

| | per-token latency | params |
|---|---|---|
| MineWorld 300M @ 5.9 fps, 340 tok/frame | 499 us/token | 300M |
| mirage target, 30 fps, 64 tok/frame | 521 us/token | ~15M |

**The target is essentially the same per-token latency as a 20x larger model.** That
is only possible - and only necessary - because both systems are
fixed-overhead-bound rather than compute-bound. It sharpens the thesis: not "a small
model can be fast," but "fixed overhead is what makes a small model no faster than a
big one, and here is how to remove it." Precisely what the ladder attacks.

- **At 64 tokens, P-1 is nearly free and P-5 is the binding requirement.** The naive
  path is ~400-495 us against a 520 us budget, so the Phase 3 baseline lands near
  30 fps *before* any optimization; the ladder's job is P-5's 3x speedup, not
  reaching playability. At 144 tokens this inverts - naive is 228% of budget, so P-1
  becomes hard and DiagD is mandatory.
- **Prediction to validate the instrumentation:** F-17's first measurement at 64
  tokens should show roughly 80% CPU dispatch, 20% GPU busy. A very different split
  means suspect the measurement before the model.

## Which later-phase plans can be drafted now

| Phase | Draftable now? | What gates it |
|---|---|---|
| 0 Sim + data | Yes, fully | nothing |
| 1 Tokenizer | Yes | only the shard format, now fixed |
| 2 Dynamics | Structure yes, numbers no | the 64/144 fork - a *Phase 1* result |
| 3 Playable | Yes | nothing |
| 4 Engine | **No** | the Phase 3 profile, which does not exist yet |

Phase 0's *results* gate almost nothing; what gates Phases 2 and 4 is Phase 1's PSNR
number. Next step: a full Phase 0 plan plus a thin Phase 1 plan, since the Phase 0
to 1 handoff is the token cache and that seam should be right. Leave 2-4 undrafted -
"profile before changing anything" applied to planning.

## Findings that change the scene XML

- **Colour the two arm links differently.** Q-5 (link-length drift), Q-4
  (`link_angle`), and `compactness[link]` all become few-line pixel measurements
  per link. A shared arm colour would require segmenting one blob into two links.
- Distinct saturated colours per block make "block count" a per-colour pixel
  threshold rather than connected-component labelling. **No scipy in the shipped
  design** - it appears only if the deferred CC diagnostic is added in Phase 2.
- **The palette has exactly one home: the XML's `rgba` attributes.** The validator
  reads them with `xml.etree.ElementTree` (stdlib, ~5 lines) rather than duplicating
  the list in config JSON. Duplication here is the same class of bug as two
  validator implementations - the copies drift, and the symptom is a validator
  reporting "block missing" for a block that is present.
- Keep the palette under F-2's 24-colour ceiling. Current count is ~7 (background,
  table, 2 arm links, 3 blocks), so there is headroom, but every new
  distinctly-coloured object spends from a budget that also protects the tokenizer.

## Repo layout

Two build systems, no monorepo tooling. `sim/` is deletable after Phase 0 and
nothing in `mirage/` imports MuJoCo.

```
mirage/
  configs/base.json          # sections: sim, data, tokenizer, dynamics, engine, validator
  scene/arm_blocks.xml       # flat-configured; rgba attrs are the palette's only home
  sim/                       # C++20, CMake, sanitizer build type
    CMakeLists.txt
    main.cpp                 # arg parse, seed, episode loop
    gl_context.{h,cpp}       # GLFW hidden window; asserts GL_RENDERER names the GPU
    policy.{h,cpp}           # per-episode 50/50 random vs scripted reach
    truth.{h,cpp}            # reads mjData: contact mask, segmentation px counts, poses
    shard_writer.{h,cpp}     # pixels + meta blobs, sidecar JSON written last
  mirage/                    # Python package, run via python -m mirage.X
    config.py                # load, hash tree, Shapes
    data.py                  # memmap reader, episode-aware sampler
    validator.py             # measurement vector, both modes, threshold sweep
    logging.py               # log(dict) -> jsonl always, wandb when flagged
    fsq.py                   # Phase 1
    dynamics.py              # Phase 2
    engine.py                # Phase 3-4
    bench.py                 # Phase 4; preallocated timing array, GPU power-state gate
  runs.jsonl                 # E-5 lab notebook, append-only, hand-authored
```

`truth.cpp` stays separate from `shard_writer.cpp` - one measures `mjData`, the
other serialises. `truth` is the only C++ that *must* exist, because its inputs
vanish when the sim is deleted.

## Verification

Each **M** requirement gets one runnable check. Phase 0's gate is these passing.

| Check | Method | Requirement |
|---|---|---|
| Clean build from scratch | `cmake -B build && cmake --build build`, documented in README | E-2 |
| GL context is hardware, not software | binary prints `glGetString(GL_RENDERER)` after context creation; assert it names the RTX 5060, and reject `GDI Generic` / `Microsoft Basic Render Driver`. A software fallback is ~50x slower and silently kills P-6. Deny by renderer name, not by vendor: vendor strings do not identify hardware. **PASSED 2026-08-23** in `bench/readback_probe.py`; the C++ port keeps the same assert | F-3 |
| **Per-call `mjr_readPixels` latency, in isolation** | time N readbacks in a tight loop, report us/call. Not end-to-end fps, which hides which term dominates. **PASSED 2026-08-23**: 25.4 us RGB / 49.6 us RGB+depth / 75.8 us with render, at P2. Two-pass render confirmed, 13x margin on P-6. `bench/readback_probe.py`. Re-run the render arm against the real scene | P-6 |
| MuJoCo step time for this scene | time `mj_step` alone, report us/step. **Day 1, and now the only day-1 number that can still threaten P-6** - render+readback leaves ~1850 us of the 2000 us frame | P-6 |
| **GPU at P0, and bandwidth re-measured there** | plugged in, performance profile, clocks locked, desktop GPU consumers closed; then re-run the copy and matmul benchmarks. **Day 1** - the fork table's compute floors all derive from the 448 GB/s assumption, currently unverified | E-4, P-4, and every P-row |
| Determinism | generate twice, same seed, `cmp` the pixel blobs | F-4, E-1 |
| Sanitizers clean | full generation run under ASan+UBSan, zero reports | E-3 |
| Throughput | binary reports frames/sec at exit, assert >= 500 | P-6 |
| Dataset size | `du -sh` the dataset dir, assert <= 20 GB | R-4 |
| Memmap round-trip | write a known buffer in C++, read via `np.memmap`, byte-compare | F-8 |
| Palette adherence | validator mode 1 over all shards, `n_unique_colors` <= 24 | F-2 |
| Action histogram | `np.bincount` over `action`, assert roughly uniform over 9 | F-5 |
| Contact rate | nonzero fraction of `contact_mask`, assert > 5% | F-6 |
| Occlusion rate | fraction of frames with any `visible_px[i] == 0`, assert >= 3% | F-7 |
| Validator false positives | threshold sweep of mode 2 against mode 1, zero FP on ground truth; repeat against Phase 1 reconstructions | F-9 |

The occlusion and contact checks are the two most likely to fail on the first scene,
and both are fixed by editing the XML - arm reach versus block placement - not by
changing code. Budget an iteration or two.

## Verification log

What has been checked and how, so future sessions neither re-derive nor over-trust.

| Claim | Method | Result |
|---|---|---|
| All dataset / meta / token-cache / KV / param / budget / diagonal arithmetic | computed | confirmed; values in the tables above |
| Ingredients doc's budget table | recomputed from first principles | KV 28.1/59.7 us vs its "~28/~60"; weights 65.0 us vs "~67"; params 14.56M vs "15M"; DiagD 4.27x/6.26x vs "4.3x/6.3x". **The doc's arithmetic is sound.** |
| Compression ratio identical across the fork | closed form + computed | exactly `1536/9` both, resolution-independent |
| `mjRND_SEGMENT` / `mjRND_IDCOLOR` exist, set via `mjvScene.flags` | MuJoCo API docs | confirmed; IDCOLOR encodes segid+1 |
| numpy structured field view / torch behaviour | **run empirically** (numpy + torch 2.9.1) | non-contiguous as expected, but `from_numpy` succeeds. **Corrected an overstated claim** - SoA is a simplicity argument, not a speed one |
| W&B offline mode, background metrics process, `x_disable_stats` | W&B source and docs | confirmed |
| ASan / UBSan overhead | published benchmarks | ASan 73% avg, UBSan full set up to 228%. **Corrected a too-optimistic guess** |
| `mjr_readPixels` at ~30 ms per call under GLFW | MuJoCo discussion #2222, then **measured and refuted** | **Does not reproduce.** 25.4 us RGB / 49.6 us RGB+depth at 64x64, GLFW offscreen on Windows - three orders of magnitude below the reported figure. **This was the highest-risk number in the plan; it is now the safest.** No pbuffer, no single-pass collapse |
| What the fixed per-call cost actually was | measured by toggling MuJoCo's visual defaults | `shadowsize=4096` and `offsamples=4`. Zeroing both cut `mjr_render` 316 -> 26 us and readback 71 -> 25 us; the readback share was an MSAA resolve. **Promotes `<quality offsamples="0" shadowsize="0"/>` + `castshadow="false"` from cosmetic to a hard `arm_blocks.xml` requirement with a measured justification** |
| Hardware GL context on Windows (F-3) | `bench/readback_probe.py` asserts `GL_RENDERER` contains "RTX 5060" | confirmed - assert passes, offscreen framebuffer selected and non-empty. Not `GDI Generic`, not `Microsoft Basic Render Driver` |
| Target machine: sm_120, CUDA version, WSL2 availability | queried | **partly refuted** - capability (12,0) and CUDA 13.0 > required 12.8 both confirmed, and it is a 5060 **Laptop** GPU, which the docs do not distinguish. But WSL2 running is not the same as WSL2 rendering: its GL path is broken here, so the project moved to Windows |
| **448 GB/s memory bandwidth** | attempted, **measurement invalid** | got 66-77 GB/s, but at 6.16 W / P4 / 36% SM clock - the GPU never left a low-power state. Neither confirms nor refutes 448. **Re-measure at P0.** Surfaced the E-4 / P-4 power-state requirement above |
| MuJoCo step time for this scene | **not verified** | the remaining day-1 measurement, and the only one that can still threaten P-6. ~1850 us of the 2000 us frame is left for it |
| Offscreen GL on WSL2 | attempted, **refuted** | `bench/egl_probe.py` reaches a context but `GL_RENDERER` is `llvmpipe` (CPU) under every platform and driver override. Root cause is below Mesa: `dxgkio_query_adapter_info: Ioctl failed: -22`, no `/dev/dri` node. CUDA is unaffected - different ioctl path. **Moved all rendering to Windows** |
| ASan-under-CPython specifics | **not verified in detail** | claim softened; the boundary decision does not depend on the magnitude |

Sources: [ASan paper](https://research.google.com/pubs/archive/37752.pdf),
[Debloating ASan (USENIX Sec '22)](https://www.usenix.org/system/files/sec22summer_zhang-yuchen.pdf),
[Reducing Redundant Sanitizer Checks (OSDI '21)](https://www.usenix.org/system/files/osdi21-zhang.pdf),
[MuJoCo discussion #2222](https://github.com/google-deepmind/mujoco/discussions/2222),
[MuJoCo mjtRndFlag reference](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).
