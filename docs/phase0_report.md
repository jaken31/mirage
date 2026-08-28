# Mirage: Phase 0 Report

**Status: Phase 0 complete. Gate met 2026-08-27 and re-met twice on 2026-08-28**,
all three conditions every time. 300,000 frames on disk at `data_hash 18a76531`.

> **Superseded hashes, and why there are three.** `0259947e` was the original
> physics; it read F-6 20.69%, F-7 16.18%, F-5 ratio 2.27 and 6,775 fps, and
> those figures describe a scene that no longer exists. `219ab0af` was the
> `gear 6 / damping 1.5` scene change. `18a76531` is the **same scene** as
> `219ab0af`: the hash moved not because anything about the data changed but
> because `data_hash` was being computed over the scene XML's **CRLF** bytes in a
> stale working tree, so no fresh clone could reproduce it. Line endings are now
> normalised inside `mirage.config.scene_bytes`. Every measured value is
> unchanged across that move; the meta record additionally gained D3's
> scripted-episode flag. All gate runs are in the verification log.

Derived document, same class as `timeline.md` and `decision_notes.md`. Every
number here traces to the verification log at the end of
`docs/world_model_architecture.md`; every requirement ID traces to
`docs/world_model_requirements.md`. When a decision changes, change it there
first, then re-derive this. Nothing in this file is authoritative on its own.

---

## 1. What the project is

Generate a robot-arm manipulation dataset in MuJoCo, tokenize it, train an
action-conditioned world model on it, **delete the simulator**, and drive the
learned model from the keyboard at 30 fps.

The thesis is a latency argument, not a modelling one:

> MineWorld runs 300M params at **499 us/token**. Mirage targets 15M params at
> **521 us/token**. Same latency, 20x fewer parameters. Neither system is
> compute-bound - launch overhead (~400 us) dwarfs the compute floor (~135 us).
> A small model is only fast once you remove the fixed overhead.

Five phases. Phase 0 is the simulator and the dataset - the only phase with C++
in it, and the only phase whose artifacts survive the simulator's deletion.

| Phase | What | State |
|---|---|---|
| **0** | MuJoCo scene, C++ generator, shard format, Python reader, validator | **done** |
| 1 | FSQ tokenizer, 64x64 to an 8x8 grid over 512 codes | not started, plan draftable |
| 2 | Dynamics transformer, 384d x 8L, interleaved frame/action tokens | structure only - numbers gated on Phase 1 PSNR |
| 3 | Playable loop with MuJoCo absent | not started |
| 4 | Inference ladder: KV cache, CUDA graphs, DiagD, INT8, fused Triton | **not draftable** - derives from the Phase 3 profile |

---

## 2. The output

### 2.1 The dataset

Verified on disk this session by reading all seven sidecars and stat-ing the blobs:

| | |
|---|---|
| Frames | **300,000** (43,200 x 3 shards + 42,600 x 4) |
| Episodes | **500**, ids 0..499, each exactly 600 steps, none split across a shard |
| Resolution | 64 x 64 x 3, `uint8` |
| Pixel bytes | **3,686,400,000** = 3.686 GB, against R-4's 20 GB ceiling |
| Meta bytes | 13,800,000 = 13.8 MB - **0.374% of pixel bytes** |
| Wall clock | **45.1-50.1 s at 5,987-6,653 fps** - 12-13x over P-6's 500 fps floor |
| `data_hash` | `18a76531...`, identical across all 7 sidecars |
| `git_sha` | `8735d7f`, identical across all 7 sidecars |

Three files per shard, and the write order is the crash-safety mechanism:

```
data/shards/
  shard_000.pixels    506 MB   raw uint8, 3*H*W per frame, no header
  shard_000.meta      1.9 MB   one fixed-width 46-byte record per frame
  shard_000.json      350 B    written LAST, after both blobs close
  ... x7
```

**The sidecar is the commit marker.** A crashed run leaves blobs with no JSON, and
`load_shards` globs the sidecars, so an incomplete shard is skipped by
construction. No lockfiles, no truncation bookkeeping.

### 2.2 The meta record - 46 bytes, every field with a named consumer

A field with no downstream consumer does not ship. This is the whole justification
table:

| Field | Type | Consumer |
|---|---|---|
| `action` | `u8` | F-5 histogram, training targets, Q-4 commanded direction |
| `qpos[2]` | `f32 x2` | calibration reference for pixel-measured link angles (Q-4, Q-5) |
| `block_xy[3][2]` | `f32 x6` | Q-6 position error |
| `visible_px[3]` | `u16 x3` | F-7 occlusion rate, Q-6 occlusion events |
| `contact_mask` bits 0..6 | `u8` | F-6 contact rate - bit *i* is "block *i* touches the arm" |
| `contact_mask` bit 7 | (same byte) | scripted-vs-random: which half of the 50/50 policy mix produced this episode |
| `episode_id` | `u32` | prevents a training window spanning a reset |
| `step_idx` | `u16` | same |

**`contact_mask` is two fields in one byte.** The scripted flag is packed into
bit 7 rather than given its own `u8`, which would have taken the record to 47 B
for one boolean. `sim/truth.cpp` caps a scene at seven blocks so the fields cannot
collide, and the writer aborts if a block bit ever reaches bit 7 anyway. **Every
reader must mask**: `contact_mask != 0` on the raw byte counts every scripted
frame as a contact, which reads F-6 as over 50% instead of 16.63% and fails
nothing. The constant lives once per language - `kScriptedBit` in
`sim/shard_writer.h`, `SCRIPTED_BIT` in `mirage/data.py`.

Before this the 50/50 coin - the single biggest structural choice in the policy -
was invisible in the dataset, and the ctx=15 action-coverage split had to be
inferred from a bimodal histogram with a hand-chosen cutoff. It now reads
directly: **53.2% of the 500 episodes are scripted**, constant across every frame
of each.

Two more non-obvious choices worth restating:

- **Counts, not booleans.** `visible_px` stores pixel counts rather than
  `is_occluded`. A boolean bakes a threshold into the dataset; counts let F-7 and
  Q-6 be re-derived at any threshold without regenerating 3.7 GB.
- **`episode_id` + `step_idx` are mandatory.** At ctx=15 a window straddling a
  reset is pure noise, and without these fields the loader cannot detect it. The
  failure mode is a plausible-looking loss curve, which is the worst kind.

### 2.3 Measured results against the Must-tier requirements

All figures over the **full 300k set** unless marked otherwise.

| ID | Requirement | Threshold | Measured | |
|---|---|---|---|---|
| F-2 | Palette adherence | <= 24 unique RGB | **7** | pass, whole set |
| F-3 | Hardware GL, not a software rasterizer | deny `GDI Generic` / `Basic Render Driver` | `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2` | pass |
| F-4 | Determinism given a seed | bit-identical | **two full runs, byte-identical `.pixels` and `.meta`, all 7 shards** | pass |
| F-5 | Action coverage / balance | min share >= 5%, max/min <= 2.5 | **7.17%**, ratio **2.15** | pass |
| F-6 | Arm-block contact | > 5% of frames | **16.63%** | pass, 3.3x |
| F-7 | Full block occlusion | >= 3% of frames | **19.83%** as the counter defines it, of which **5.35%** is recoverable occlusion and **14.48%** is a block that never returns | pass; quote **1.8x**, not 6.6x |
| F-8 | Shard round-trip byte-exact | numpy matches C++ | 448 records decode identically two independent ways | pass |
| F-9 | Validator, zero false positives | 0 FP on ground truth | **`offpalette_px` = 0 on every ground-truth frame at `validator.offpalette_tau` = 8.0** | pass, verdict thresholds not yet written |
| P-6 | Generation throughput | >= 500 fps | **5,987-6,653 fps** | pass, 12-13x |
| P-7 | Full 300k epoch | <= 30 min | **5.9 s sequential / 39 s random** | pass, 306-734x |
| R-4 | Dataset on disk | <= 20 GB | **3.686 GB** | pass |
| E-3 | ASan clean on the generation run | zero reports | clean, both build types | pass |
| E-2 | Clean build from scratch | documented in README, verified once | **both build types built and run from an empty directory 2026-08-28**; recipe and four measured gotchas in `README.md`, "Build" | pass |

**F-8 and F-9 are runnable without the dataset.** `python -m mirage.data` and
`python -m mirage.validator` fall back to a committed 40-frame fixture
(`mirage/fixtures/`, 4.9 KB packed, real writer output) when `data/shards` is
empty, so both acceptance tests run in a fresh clone. The three dataset-scale
checks they cannot honestly make there - F-6, F-7 and the episode-level
train/val split - are skipped and say so.

**F-7's headline number is 73% blocks that are gone.** `bench/occlusion_probe.py`
splits `visible_px == 0` into occlusion the block recovers from and a block that
never returns: 5.35% and 14.48% of the 19.83%. The honest occlusion rate clears
the floor at 1.8x rather than 6.6x. The recorded cause was wrong - **no block has
ever left the table**, 0 frames of 900,000, on a table whose half-extent is 1.2 m
against an arm that reaches 0.33 m. The cause is the camera: 12.66% of frames
have a block outside the frustum. Nothing needs regenerating to separate them.

**Two small-sample figures were corrected by the full run and should not be
quoted.** A single 1,200-frame shard read F-6 at 62.4% and F-7 at 40.3% -
roughly 3x the full-set values in both directions. A 2-episode window does not
estimate a 500-episode run.

### 2.4 Day-1 performance probes

Four numbers, each of which decided a design question. All in `bench/`.

| Measure | Result | What it decided |
|---|---|---|
| `mjr_readPixels` RGB @ 64x64 | **25.4 us** | Two-pass render, **13x margin**. No WGL pbuffer, no single-pass collapse |
| `mj_step`, driven arm | **10.5-10.8 us** median | Physics is 131-176x under budget. Not a P-6 risk |
| Full frame: step + 2-pass render + readback | **178.5 us = 5,602 fps** | **Parallel generation not needed** - the trigger never fired |
| Memory bandwidth, clocked up | **308.3 GB/s** read | Refuted the assumed 448. Every compute floor rose 45%; CUDA graphs promoted to required on both fork paths |

---

## 3. The architecture

### 3.1 Data flow, end to end

```
mirage/configs/base.json --+
scene/arm_blocks.xml ------+--> mirage/config.py --> data_hash (sha256)
                                                          |
                                                          v  passed as a flag
   +--------------- mirage_sim <config.json> --data-hash --git-sha --------------+
   |                                                                             |
   |  GlContext ----> hidden GLFW window, offscreen FBO, GL_RENDERER asserted     |
   |      |                                                                      |
   |      v                                                                      |
   |  per episode:  Policy::begin_episode ----> 50/50 coin: random | reach       |
   |      |                                                                      |
   |      v  per step (600 of them)                                              |
   |  Policy::step --> action 0..8 --> action_to_control --> mjData.ctrl         |
   |  mj_step                                                                    |
   |  mjv_updateScene                                                            |
   |  mjr_render + mjr_readPixels ---------> RGB frame     <-- MUST come first   |
   |  Truth::read (2nd pass, mjRND_SEGMENT) -> visible_px, contact, poses        |
   |  ShardWriter::append -------------------> .pixels + .meta                   |
   |                                                                             |
   |  on episode boundary, if full:  ShardWriter::commit --> .json  (LAST)       |
   +-----------------------------------------------------------------------------+
                                          |
                                          v
                    mirage/data.py   load_shards --> np.memmap, no copy
                                     episode_index --> 500 episodes
                                     WindowSampler --> map-style, flips rows, hashed split
                                          |
                                          v
                    mirage/validator.py   measure_with_truth (mode 1, Phase 0)
                                          measure_pixels_only (mode 2, Phases 2-4)
                                          sweep --> F-9 threshold calibration
```

### 3.2 The C++/Python boundary: a standalone binary, no bindings

`mirage_sim` is an executable that writes files. Python only ever reads files.
No pybind11.

The decisive argument is **E-3**: ASan through a Python extension module needs the
ASan runtime preloaded before the interpreter starts, plus a suppression file for
interpreter internals. ASan on a standalone binary is just running the binary. It
also makes "delete the simulator" literal - after Phase 0, **nothing in `mirage/`
imports MuJoCo**, and `sim/` is deletable.

### 3.3 The four C++ files, and what each owns

| File | Lines | Owns | Does not own |
|---|---|---|---|
| `gl_context.{h,cpp}` | 138 | the offscreen render target: hidden GLFW window, `mjrContext`, viewport | the model, the scene, the camera |
| `policy.{h,cpp}` | 851 | action choice; two seeded RNG streams | the simulation - `step` takes a `const mjData*` |
| `truth.{h,cpp}` | 558 | the segmentation pass and per-block pixel counts | files, simulation state |
| `shard_writer.{h,cpp}` | 603 | two blob handles, byte counters, the sidecar | what a frame *contains* - it measures nothing |
| `main.cpp` | 328 | arg parsing, the episode loop, shard rotation | - |

`truth.cpp` stays separate from `shard_writer.cpp` on purpose: one measures
`mjData`, the other serialises. **`truth` is the only C++ that must exist**,
because its inputs vanish the moment the simulator is deleted.

Four invariants those files hold, each of which is a bug that would otherwise be
silent:

1. **RGB readback precedes `Truth::read`.** The segmentation pass leaves the
   framebuffer holding id colours. Reverse the order and the shard stores id
   colours - which look like a plausible flat-shaded image and fail no check
   downstream.
2. **Shards rotate on episode boundaries only.** `Policy` is seeded per shard, so
   a mid-episode rotation would reseed mid-episode.
3. **`data_hash` is passed in, never recomputed in C++.** A second
   canonical-JSON hash would have to match Python's float formatting byte for
   byte, and the day it stopped, two shards with identical contents would carry
   different names.
4. **Every offset and frame counter is 64-bit, with a bounds assert at the write
   site.** 3.6 GB against `2^31` = 2.1e9 means a signed 32-bit offset wraps past
   frame ~175k and silently corrupts the back half of a shard. This is the one
   UBSan class that mattered; it is closed by typing, not by a second toolchain.

### 3.4 The policy: per-episode mixing, and a histogram tuned at a measured knee

Action encoding is base-3, one digit per actuated joint:

```
action = sum_i (sign_i + 1) * 3^i,   sign_i in {-1, 0, +1}
```

With two hinges that is **9 actions**, and `nu <= 5` is enforced because the meta
record stores the index as a `u8`.

**The mix is per-episode, not per-frame.** A scripted reach needs consecutive
steps to complete; coin-flipping per frame destroys the property the mix exists
for. F-5's near-uniform histogram still holds in aggregate because the random
half carries it.

Two config changes fell out of measurement here:

- **`steps_per_episode` is 600, not 200.** At 200 steps the episode is 0.4 s of
  sim time and the arm cannot cross to a block: the scripted half arrives 13% of
  the time instead of 44%. 500 x 600 is the same 300,000 frames, and it cuts
  ctx=15 boundary loss from 7.5% to 2.5%.
- **The histogram knobs are `jacobian_deadband` and `reach_digit_noise_prob`**,
  not whole-action noise. The scripted reach commands `sign(gain)` and a double
  is never exactly zero, so with no dead zone it can *only* emit corner actions -
  measured at 70.6% of frames against a uniform 44.4%.

F-5's 2.5 ceiling is **the knee of a measured 14-configuration frontier**, not a
round number: arrival is flat at 42.5-44% down to ratio 2.44, then falls off
(33.5% at 2.17). And anything below ~1.7 is unreachable by any tuning - a
two-link planar arm reaching outward turns both hinges the same way. That is
kinematics, not a bug.

### 3.5 Provenance: a hash tree, so no invalidation code exists

```
data_hash = sha256(canon(sim) + canon(data) + arm_blocks.xml)
  |-- tokenizer_hash = sha256(data_hash + canon(tokenizer))
  |     '-- dynamics_hash = sha256(tokenizer_hash + canon(dynamics))
  |           '-- engine_hash = sha256(dynamics_hash + canon(engine))
  '-- validator_hash = sha256(data_hash + canon(validator))
```

The scene XML is inside `data_hash` or E-4 has a hole - a bench number from a
different scene is not comparable. `validator_hash` **branches off** rather than
chaining, so re-tuning a threshold and retraining the tokenizer are mutually
non-invalidating.

What it buys: **no invalidation code exists.** A stale artifact is unreachable
because nothing computes its name. "Already computed?" is `os.path.exists`.
Concurrent runs cannot collide. Every number walks back to its frames. ~20 lines.

The honest limit, conceded up front: **the hash covers config, not code.** Change
an implementation without touching its config and the hash is identical while
behaviour differs. The fix is to store the **git SHA inside each artifact** - not
in the hash, which would invalidate everything every commit - and accept that
staleness *detection* stays manual while provenance *diagnosis* is solved.

### 3.6 The Python side

**`mirage/config.py` (254 lines)** - sectioned JSON, key and value checking, the
hash tree, and a `Shapes` named tuple. Enforces the "no hardcoded shapes" rule:
image size, token grid, context length and action count all derive from config.

**`mirage/data.py` (364 lines)** - `load_shards` globs sidecars and memmaps the
blobs; `meta_dtype(joints, blocks)` builds the structured dtype **from the
sidecar's counts**, not from a hardcoded 46. `WindowSampler` is map-style, so
window *i* is the same window under a DataLoader, a shuffle, or a resumed run.

Two gotchas that cost real time and are worth carrying forward:

- **Slicing an `np.memmap` returns a lazy view and touches no pages.** A first
  probe reported 3.4M fps having timed slice arithmetic. `__getitem__` must
  return `np.array(...)` to force the copy.
- **The blob is bottom-up** - `mjr_readPixels`' origin is bottom-left and nothing
  in `sim/` flips it. The flip lives **in the sampler**, not in `Shard.pixels`,
  which stays raw so F-8 has something byte-exact to compare. Read frames any
  other way and `link_angle` comes out mirrored.

The **split is by episode, hashed** - 473 train / 27 val (5.4%) at
`val_fraction` 0.05. A per-frame split anywhere downstream reintroduces the leak
this removed: consecutive frames differ by one 2 ms step.

**`mirage/validator.py` (452 lines)** - emits a measurement vector, never a
verdict. "The validator failed" is a threshold expression over that vector,
defined in config rather than in code.

| Field | Per | Detects |
|---|---|---|
| `px_count[color]` | colour | missing / dissolved objects |
| `bbox[color]` | colour | position, extent |
| `compactness[color]` | colour | fragmentation, smearing |
| `link_extent[2]` | link | Q-5 drift |
| `link_angle[2]` | link | **Q-4 action-following** |
| `offpalette_px` | frame | palette violation (F-9) |
| `n_unique_colors` | frame | F-2's 24-colour bar, mode 1 only |

Three properties this buys: Q-3's coherence horizon becomes recomputable without
re-running rollouts (~3 MB of stored vectors); connected-component labelling never
becomes a commitment; and **Q-4 needs no inverse dynamics model** - actions are
joint deltas in `{-1,0,+1}` and the PCA that produces `link_extent` yields
`link_angle` for free, so action-following is `sign(theta_t+1 - theta_t)` against
the commanded sign. That is *better* than an IDM, which would carry an
unquantified generalization gap from ground-truth frames to generated ones.

**Why colour counting and not connected components** - the argument is
correctness, not effort. F-7 *requires* full occlusion in >= 3% of frames, so
partial occlusion is common. An arm crossing a block splits it into two
disconnected same-colour blobs, and CC reports that as **two blocks** - a phantom
object. Colour counting is immune by construction. `compactness` uses a **PCA
oriented bbox**, not axis-aligned, because both links revolve and a free-joint
block rotates: an axis-aligned box around a 45-degree square has 2x the area,
scores ~0.5, and collides with the occluded case.

---

## 4. What measurement caught that the plan got wrong

This is the strongest material in the project. Every claim carrying a number held
up; every claim without one was eventually wrong.

| Claim in the plan | What measurement returned |
|---|---|
| `mjr_readPixels` costs ~30 ms/call under GLFW; hand-roll a WGL pbuffer | **25.4 us.** Three orders of magnitude off. The real fixed cost was MuJoCo's own defaults - `offsamples=4` forcing an MSAA resolve, `shadowsize=4096`. Zeroing both cut render 316 -> 26 us and readback 71 -> 25 us. A config flag, not a week of code |
| 448 GB/s memory bandwidth | **384 peak, 308.3 measured.** 448 is the *desktop* 5060 at 28 Gbps; the Laptop part runs 24. Every compute floor rose 45%, and CUDA graphs went from optional to required on **both** fork paths |
| Gate every benchmark on `pstate == P0` | **Refuted.** pstate follows the *memory* clock domain; correct compute-bound work reads P4 while the SMs sit at 86% of max drawing 99 W of a 100 W cap. The rule would have rejected every valid compute number this machine can produce |
| `castshadow="false"` disables shadows on geoms | Not a geom attribute at all - MuJoCo's compiler rejects it. The real requirement is an **ambient-only headlight**, which took the colour count from 28 to 6 |
| `sim.steps_per_episode = 200` | 0.4 s of sim time; the arm cannot cross to a block. Scripted arrival 13% vs 44% at 600 steps |
| `reach_noise_prob` is the knob for F-5 | **Wrong knob.** Whole-action replacement is dominated at every setting by the Jacobian deadband, and by per-digit corruption once the deadband is on |
| `rgba * 255` lands on exact byte values | It does not, and not by a modellable rule - 0.65 rounds *up* to 166, 0.90 rounds *down* to 229. With a byte-rounded palette, exact equality calls **4 of 7 entries missing on a flawless frame.** Nearest-palette-by-argmin is load-bearing, not a nicety |
| The palette is exactly the XML's `rgba` attributes | It needs **a seventh entry the XML cannot name**: 14.1% of every frame is the black void past the far table edge, the framebuffer clear colour. Without it `offpalette_px` reads ~578 px on a perfect frame and F-9 can never reach zero false positives |
| Thermal state is visible in the throttle flags | The flags read `Not Active` through a 45 W cap. The evidence was in `nvidia-smi -q -d PERFORMANCE` **counters**. A chassis cooling fix moved the enforced limit 55 -> 100 W and fp16 matmul **3.0 -> 27.6 TFLOP/s - 9x from cooling alone** |
| WSL2 will work for rendering | `dxgkrnl` ioctl fails at every boot, no `/dev/dri` node, Mesa falls back to `llvmpipe` (CPU, ~50x too slow). Moved the whole project to native Windows |

Two more silent-failure classes, both found by running rather than reading:

- **`mjv_updateScene` before any `mj_forward`/`mj_step` renders a black frame**
  while `scene.ngeom` reads a correct 6. Derived `xpos`/`xmat` are zero until
  forward dynamics runs. Checking the geom count does not catch it.
- **`<camera mode="targetbody">` without `target=`** compiles clean,
  `cam_targetbodyid = -1`, and the camera aims nowhere. Renders empty ground with
  no error.

---

## 5. What was deliberately not built

Each has an explicit trigger recorded in the architecture doc. None is a judgement
call - wait for the trigger.

| Not built | Why | Trigger that would build it |
|---|---|---|
| Parallel generation | 11.2x margin over P-6 | P-6 missed. Fallback is N processes, shard *i* seeded `base + i` |
| Hand-rolled WGL pbuffer | readback is 25 us, not 30 ms | readback near ~30 ms/call |
| Single-pass palette render | two passes cost 7.6% of budget | readback above ~0.5 ms/call. Held in reserve, ~10-min switch |
| Connected-component labelling | colour counting is immune to the phantom-object failure | a Phase 2 diagnostic on already-failing frames |
| `--replay` mode | F-4 is tested by generating twice and `cmp`-ing | - |
| pybind11 | E-3, and "delete the simulator" stays literal | - |
| An IDM for Q-4 | geometric measurement has no generalization gap | - |
| scipy | nothing in the shipped design needs it | the CC diagnostic above |
| clang-cl for UBSan | its one relevant class is closed by 64-bit offsets | UB suspected in code ASan does not cover |
| Version integers in config | seven hand-bumped ints across seven sections will rot | - |

> Fallbacks are not built in advance. Building them speculatively is how a
> five-day stage becomes a three-week one.

---

## 6. Open items

**Nothing blocks Phase 1.** These are the honest gaps.

| Item | State |
|---|---|
| **Validator thresholds not written into config** | `python -m mirage.validator` *prints* the set with zero false positives. The documented build order recalibrates against Phase 1 tokenizer reconstructions before anything is written, because ground-truth frames are perfectly rendered while Q-3's inputs carry decoder artifacts. Writing a Phase-0-only number into `validator_hash` now would be a guess |
| **`px_count` ruled out as a per-frame threshold** | The smallest `px_count` on a block that ground truth calls *visible* is **1 px with margin 0**, because F-7 makes partial occlusion common. The viable verdict today is `offpalette_px` alone at tau 8, which has **11x headroom** over render rounding |
| **`sim.action_hold_steps = 20` is a guess** | Estimated from `inertia / damping` (~15 steps) and rounded up. **Q-4's 90% depends on it**: for about one settling time after each sign flip the joint is still moving the old way. Replace with a sweep - log commanded sign against `sign(delta theta)` and find where agreement crosses 90% |
| **E-4's 5% not demonstrated for `mj_step`** | The series had not plateaued after 6 runs. Needs a quiescent-machine protocol |
| **F-7 carries one known bias** | A block knocked off the table reads 0 px forever and counts as occluded. The full-set 16.18% cannot separate that from a genuine occlusion |
| **F-5 compliance cost is partly unmeasured** | The deadband + noise settings cost arrival 44% -> 35.7%. Whether that cost any *contact* is reasoned about but not measured |
| **Phase 1 structural plan not written** | Draftable now - only the shard format gated it, and that format is read as well as written. Phases 2 and 4 are **not** draftable: Phase 2's numbers wait on the tokenizer PSNR, Phase 4's whole plan derives from the Phase 3 profile |

---

## 7. Code inventory

7,548 lines across docs and code. Working tree clean at `14d2fb7`.

| Area | Files | Lines |
|---|---|---|
| C++ simulator (`sim/`) | 8 + CMakeLists | 2,478 |
| Python package (`mirage/`) | 3 + base.json | 1,103 |
| Benchmarks (`bench/`) | 7 probes | 878 |
| Scene | `arm_blocks.xml` | 78 |
| Docs | 8 markdown files | 2,911 |

Each module carries its own runnable check - no test framework, no fixtures:

| Check | Covers |
|---|---|
| `policy_self_check` | all 9 actions decode to distinct `ctrl` and re-encode to the same index |
| `policy_dry_run` | F-4 (aborts on divergence), F-5 (prints a verdict) |
| `truth_dry_run` | three fatal id-colour checks, prints F-6/F-7 |
| `shard_writer_self_check` | layout, the overflow predicate at `INT64_MAX`, sidecar absent before `commit()` |
| `python -m mirage.config` | hash tree, key/value validation |
| `python -m mirage.data` | **F-8** - every meta field decodes twice, once through the structured dtype and once through an independent `struct.unpack` |
| `python -m mirage.validator` | **F-9** sweep, F-2 over the full set |

The `struct.unpack` double-decode in `data.py` is not belt-and-braces. **Every way
of getting a structured dtype wrong still reads back cleanly** - wrong field order,
wrong endianness, wrong padding all produce numbers rather than an error. Two
independent decodes at documented offsets is the only thing that catches it.

---

## 8. Verified environment

| Item | Value |
|---|---|
| OS | Windows 11, native. **WSL2 is out and settled** - broken GPU graphics path |
| GPU | RTX 5060 **Laptop**, 8 GB, sm_120, capability (12,0) |
| Bandwidth | 384 GB/s peak, **308.3 GB/s measured read** |
| Compute | **27.6 TFLOP/s fp16** at 2662 MHz / 99 W, after the cooling fix |
| Python | 3.14.2, numpy 2.4.4, torch 2.9.1+cu130, CUDA 13.0 |
| Compiler | MSVC via CMake generator `Visual Studio 18 2026`, C++20 confirmed by the binary printing `202002` |
| GL backend | **GLFW offscreen.** No EGL on Windows MuJoCo; OSMesa is a CPU rasterizer and an anti-choice |
| Sanitizers | **ASan only.** MSVC ships no UBSan and no leak detection. Separate build type, `/O2` stays the default |
| Flags | `/W4 /WX`, `/Zi` + linker `/DEBUG` (mandatory - `C5072` is fatal under `/WX`). **`/fp:fast` never** |

Two gates to remember when taking any number on this machine:

- **Do not gate on `pstate == P0`.** Gate compute on **SM clock + power draw**,
  bandwidth on **memory clock == max**. No single load clocks both domains.
- **Sample thermal counters, not throttle flags.** The flags lie.

---

## 9. What happens next

1. **Write the Phase 1 structural plan** - ownership table, build order with named
   APIs, done-when per file, gotchas. Same shape as `phase0_structural_plan.md`.
   Draftable now; only the shard format gated it.
2. **Phase 1: the FSQ tokenizer.** 64x64 to an 8x8 grid over 512 codes. Q-1 wants
   >= 30 dB PSNR held out.
3. **Recalibrate the validator** against Phase 1 reconstructions, then write the
   thresholds into config. Phase 1 produces those reconstructions anyway for Q-1.
4. **Phase 1's PSNR decides the 64x64 vs 96x96 fork**, and the decision is already
   mechanical. Worth knowing before spending a week on it: the compression ratio
   is **provably identical on both paths**, exactly `1536/9 = 170.667:1`,
   resolution-independent. 96x96 buys **oversampling relative to feature size**,
   not compression - a block spanning one 8x8 patch at 64x64 spans ~2.25 at
   96x96. **So the fallback only helps if Q-1 fails on edge placement.** If it
   fails on colour drift or lost global structure, the real lever is the FSQ
   levels table.

**Standing practice: draft the structural plan for a phase when you reach it, not
before.** "Profile before changing anything," applied to planning.
