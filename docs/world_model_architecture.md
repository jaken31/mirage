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

Decisive argument is **E-3** (ASan clean on the full data-generation run):
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

`contact_mask` is **two fields in one byte**: bits 0..6 are "block *i* touches the
arm", bit 7 is "this episode is the scripted-reach half of the 50/50 mix". Packed
rather than given its own `u8`, which would take the record to 47 B for one
boolean. `sim/truth.cpp` refuses a scene with more than seven blocks, so the two
cannot collide, and `sim/shard_writer.cpp` aborts if a block bit ever reaches bit
7 anyway. **Every reader must mask** - `contact_mask != 0` on the raw byte reads
every scripted frame as a contact and takes F-6 from 16.6% to over 50% without
failing anything. The constant has exactly two homes, `kScriptedBit` in
`sim/shard_writer.h` and `SCRIPTED_BIT` in `mirage/data.py`, which is one per
language and the same arrangement `meta_dtype` uses.

As written by `sim/shard_writer.cpp`, the sidecar carries `frames`, `height`,
`width`, `channels`, `pixel_dtype`, `meta_record_bytes`, `meta_joints`,
`meta_blocks`, `seed`, `shard_index`, `data_hash`, `git_sha`. The three `meta_*`
fields are what let the reader build its dtype from the file instead of
hardcoding 46 - the "no hardcoded shapes" rule, applied across the language
boundary. The object is flat and every value is an integer or a
character-checked atom, so the writer needs no JSON library and no escaping;
nesting anything in it is the trigger to use `nlohmann/json`, which the config
reader already links.

Field justifications, one each - a field with no named consumer does not ship:

| Field | Consumer |
|---|---|
| `action` | F-5 histogram, training targets, Q-4 commanded direction |
| `qpos[2]` | calibration reference for pixel-measured link angles (Q-4, Q-5) |
| `block_xy` | Q-6 position error |
| `visible_px[3]` | F-7 occlusion rate, Q-6 occlusion events |
| `contact_mask` bits 0..6 | F-6 contact rate |
| `contact_mask` bit 7 (scripted) | which half of the policy mix produced the episode - per-half breakdowns of F-6, Q-4 and window coverage, none of which the dataset could answer before |
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
anything. So `<quality offsamples="0" shadowsize="0"/>` is **required** in
`scene/arm_blocks.xml`, not cosmetic: it is what buys the P-6 margin, and
`offsamples="0"` is independently required for F-2's <=24-colour palette.

`castshadow="false"` was listed here too and is wrong: `castshadow` is an attribute
of `<light>`, and MuJoCo's compiler rejects it on a geom. The scene declares no
`<light>` elements, so shadow casting has no source to disable. The second real
requirement is an **ambient-only headlight** (`diffuse="0 0 0" specular="0 0 0"`):
diffuse shading varies per face, so one `rgba` becomes three palette entries per
box. Measured 2026-08-23 on the first working scene - adding the `<visual>` block
took the colour count from 28 to 6.

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

Either way, visualization decorations (contact points, joint axes) must stay off
- they pollute both the palette and the segmentation. **They already are:
`mjv_defaultOption` leaves every decoration off, so the correct action is to call
it and change nothing.** Do not zero the flag array to "make sure": that also
clears `mjVIS_STATIC`, and every worldbody geom silently stops drawing. Measured
2026-08-23; the row is in the verification log.

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
different scene is not comparable. Its bytes enter **with CRLF normalised to LF**
(`mirage.config.scene_bytes`), which is not cosmetic: it was measured on
2026-08-28 that a CRLF working tree and an LF one hash the same scene to
`219ab0af` and `18a76531`. `.gitattributes` sets `eol=lf` to prevent exactly that
and could not, because git applies `eol` only at checkout and never rewrites a
working tree that already exists. The attribute stays for readable diffs;
provenance no longer depends on it. `validator_hash` branches off `data_hash`
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
steps, 14%. **`steps_per_episode` is 600, not 200**: at 200 steps the episode is
0.4 s of sim time and the arm cannot cross to a block, so the scripted half
completes 13% of the time instead of 44%. 500 x 600 is the same 300k frames.

## F-5's threshold is the knee of a measured curve, not a round number

The scripted reach can only ever emit a **corner** action - both joints driven -
because it commands `sign(gain)` and a double is never exactly zero. Measured over
120k steps: the four corners took 70.6% against a uniform 44.4%, max/min 4.26.

Two knobs move it, measured across 14 configurations at 600 steps:

- **Jacobian deadband.** Command zero when a joint moves the tip less than the
  threshold. State-dependent and physically correct - the joint really cannot
  help. At 0.04 m/rad: ratio 4.26 -> 2.55, arrival 44% -> 42.5%.
- **Per-digit noise** instead of whole-action replacement. Corrupting one joint
  leaves the other steering. Dominates whole-action noise at every setting once
  the deadband is on: 2.09 at 39% arrival, against 2.17 at 33.5%.

The frontier has a knee. Arrival is flat at 42.5-44% down to ratio 2.44, then
falls off - 33.5% at 2.17, 29.5% at 1.87. **2.5 is the flattest point reachable
without paying reach quality**, which is why the threshold sits there rather than
at an aesthetically rounder 1.5.

Below ~1.7 is unreachable by any tuning. At deadband 0.04 the corner-vs-edge
problem is already gone (corners 48.4% against 44.4%); what remains is same-sign
corners at 19,200 against opposite-sign at 9,300, because a two-link planar arm
reaching outward turns both hinges the same way. That is kinematics. A threshold
under 1.7 would be a requirement nothing can satisfy.

**Shipped: `jacobian_deadband = 0.04`, `reach_digit_noise_prob = 0.15`.** Verified
at F-5's own sample size, 2,000 episodes x 600 steps: min share **7.15%**, ratio
**2.06**, both inside the thresholds with margin. The cost is arrival 44% -> 35.7%
and median closest 0.042 -> 0.052 m. Arrival is a diagnostic, not a requirement -
F-6's contact rate is the number that would make this a bad trade, and `truth.cpp`
does not measure it yet. Re-check when it does.

The 5% floor is a different quantity - coverage, not balance. It puts >= 15,000
frames of the rarest action in a 300k set, against roughly 1,000 per class that
Q-4's balanced eval subset needs. **The link from examples-per-action to Q-4
accuracy is unverified**: no inverse dynamics model exists yet. Trigger to raise
it: Q-4 misses 90% and its confusion matrix concentrates on the rare actions.

**Rejected: a compensating prior on the random half.** Solving
`r = (1/9 - s*q)/(1 - s)` for the random half's distribution makes the total
histogram exactly flat at zero cost to the reach, and every weight comes out
positive, so it is feasible. It would also drop actions 0 and 8 to ~1.3% of random
draws and raise action 4 to 20% - random episodes would coast a fifth of the time
and almost never sweep both joints together. That buys a flat action marginal by
distorting state visitation, which is what the pixel model actually learns from.
Revisit only if the residual same-sign bias is measured hurting Q-4.

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

  **Run the same measurement on ground-truth frames and report both numbers.**
  The metric has a ceiling below 100% that has nothing to do with the model: for
  about one joint settling time after each commanded sign flip, the joint is
  still moving the old way, so the sign disagrees with the command in the
  simulator itself. Measured 2026-08-28 at 83.1% under the shipped
  `action_hold_steps`. **Q-4 is therefore scored as a fraction of that ceiling,
  not against an absolute 90%** - an absolute bar above the ceiling fails a model
  that is exactly right. Recompute the ceiling whenever the scene or the hold
  changes; `bench/hold_probe.py` produces it.

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
| 1 | Human-authored run log: config hash, change, number, conclusion | **E-5, Must.** A lab notebook, one line per run. No tool produces it. Lives at `runs.jsonl`; one JSON object per line, keys `date`, `run`, `hash`, `requirement`, `change`, `number`, `conclusion`. `hash` is null for a run that reads no config - the `bench/` probes do not. `number` is an object rather than prose so F-17's jsonl-to-markdown script can read it. Backfilled 2026-08-28 from the dated, numbered rows of the verification log below. |
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

**Recalculated 2026-08-23 against measured bandwidth.** The floors are pure streaming
reads - weights (14.56M params, bf16 = 29.12 MB) plus the KV cache, divided by read
bandwidth. They were computed at an assumed 448 GB/s; the measured figure is **308.3 GB/s**
(`bench/gpu_probe.py`), so every floor rose by 45%. Two consequences:

- **CUDA graphs are now required on both paths, not just the 144 one.** The 64 path's naive
  total was 95% of budget and is now 103% - it no longer fits without them.
- **The 144 path has no margin left without DiagD**: 100% of its per-token-step budget after
  graphs, against 77% before. DiagD was already marked required; it is now the only thing
  between that path and the wall.

With graphs *and* DiagD both paths are comfortable - 8% and 16% of the 33.3 ms frame - so
the fork decision itself is unchanged. What changed is that both mitigations became
mandatory rather than one being held in reserve.

| | 64 tok/frame | 144 tok/frame |
|---|---|---|
| Per-token-step budget @ 30 fps | 520.8 us | 231.5 us |
| Compute floor | 135 us | 181 us |
| Launch overhead (~80 kernels) | 400 us | 400 us |
| **Naive total** | **~535 us (103% of budget)** | **~581 us (251% of budget)** |
| After CUDA graphs (~50 us overhead) | ~185 us (36%) | **~231 us (100%)** |
| DiagD reduction | 15 diagonals vs 64 = 4.27x | 23 vs 144 = **6.26x** |
| With graphs + DiagD, per frame | 2.78 ms (8% of 33.3 ms) | 5.32 ms (16%) |
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

**Measured correction 2026-08-28: the "colour drift" branch is close to dead, and
the fork is therefore likelier than this table assumed.** A k-means codebook of 512
entries over real 8x8 patches carries **99.86% of its squared error in the 37% of
patches that are not a single flat colour** - flat patches contribute 0.14%. Edge
placement is not one of two comparably likely failure modes; it is where essentially
all of the error already lives before any training has happened. So the branch that
sends this project to 96x96 is the branch the data points at, and the consequences
below - DiagD required, F-16 promoted to **M**, graphs as table stakes rather than the
headline - should be treated as more likely than not until Phase 1 says otherwise.
**That is why the 96x96 arm is in Phase 1's ladder rather than held as a fallback**:
one config file, ~45 s of generation and one training run replaces this prior with a
number.

## Phase 1 decisions: the tokenizer

Recorded here so item 5 of `phase1_structural_plan.md` does not re-derive them. Each
carries the trigger that would reverse it.

### Q-1 is the phase's real risk, and Q-2 is at risk for the opposite reason

Both claims are measured, over the 300,000 frames on disk, before any of Phase 1
exists.

**Q-1.** A patch-independent tokenizer cannot pass. k-means with 512 centroids over
real 8x8 patches reaches **26.39 dB** against Q-1's 30 dB; 1,024 centroids reach
**27.60 dB**, so vocabulary is not the lever. The entire 3.6 dB has to come from the
22x22 receptive field, the attention layer and a shared decoder - the 64 codes
describing the frame *jointly* rather than independently, since 26.39 dB *is* the
independent number. Restated physically, because a wrong pixel costs **47,814**
squared error on this palette: the floor gets ~38 of 4,096 pixels wrong and the bar
is ~17.

**Q-2.** The data does not force low entropy. Only **19.96%** of interior cells have
a fully flat 22x22 receptive field (all of them table), which puts the provable
entropy ceiling at **94.4% of uniform** at 512 codes - cells whose receptive fields
hold identical pixels must get identical codes, and that is the only hard constraint
the data imposes. What is at risk is that the scene does not *need* 512 codes: the
same k-means run keeps only **150 of 512** centroids alive, and the exact-8x8-patch
distribution carries just **4.43 bits** of entropy against the 9 available.

**If Q-2 misses, shrink the vocabulary. Do not add an entropy loss.** An auxiliary
loss undoes the reason FSQ was chosen over VQ, and it makes the collapse question
permanently unanswerable. Ladder, in order: `[8,8,8]`=512 -> `[8,6,5]`=240 ->
`[5,5,5]`=125 -> `[4,4,4]`=64. Token count never changes, so inference cost is
untouched and the output head shrinks.

**Trigger to revisit:** if R0 - the continuous-bottleneck control - also misses 30 dB,
then neither of the above is the problem and the encoder is. Fix capacity before
touching levels.

### The straight-through gradient is not 1.0, and it moves with the levels table

Measured by running it: the gradient of the quantizer output with respect to its
input, at zero, is **0.858** for `[8,8,8]`, **1.001** for `[5,5,5]` and **0.668** for
`[4,4,4]`. The straight-through estimator bypasses only the rounding, so the `tanh`
derivative inside `bound` stays in the path, and it depends on `levels` through
`half_l` and `offset`.

Consequence: **walking the shrink ladder rescales the effective bottleneck learning
rate by up to 1.5x**, and a comparison run at one LR reports a levels result that is
partly an LR result. Run each variant at the base LR and at the base LR scaled by
`0.858 / g_new`. This is the one place in the phase where a plausible-looking
experiment silently answers a different question than the one asked.

### Loss is plain MSE, and the F-9 recalibration is its counterweight

PSNR is a monotone function of MSE, so the loss *is* the gate. No perceptual term, no
GAN, and none of FSQ's absent auxiliary losses.

**Per-pixel 7-way cross-entropy was considered and loses.** It would give hard edges,
which is attractive given where the error lives. But a misclassified pixel costs the
full 47,814, so classification needs ~99.6% pixel accuracy to clear 30 dB, while
regression can hedge with a blend and pay far less. CE is worse aligned with the gate,
not better.

**The hole this leaves is real: MSE rewards blurring edges, and edges carry 99.86% of
the error.** A model could clear Q-1 by hedging every boundary. The counterweight is
the F-9 recalibration - `offpalette_px` over reconstructions counts exactly the
pixels a blend produces, so PSNR and `offpalette_px` cannot both be gamed. **Report
them together or neither means much.** This is what turns the leftover Phase 0
calibration item into a load-bearing part of Phase 1 rather than a chore.

**Trigger:** if Q-1 passes while `offpalette_px` on reconstructions is far worse than
the render-rounding baseline, add a small cross-entropy term as an auxiliary and
re-measure both. Not before that evidence exists.

### Decoder upsampling is nearest-plus-conv, never transposed

`ConvTranspose2d` checkerboarding presents as misplaced edges. Misplaced edges are
the exact signal that decides the 64-vs-144 fork, and the fork costs a week and
reshapes Phase 4. A checkerboard artifact would therefore not merely add noise - it
would produce a *false positive on the one diagnostic the phase exists to read*. No
trigger reverses this; the cost of being wrong is asymmetric.

### One self-attention layer on the 8x8 grid

64 positions, so the attention matrix is 64x64 and the cost is negligible against
three conv stages. It is also the only mechanism in the design by which codes
cooperate, which the 26.39 dB floor establishes is the whole game. It is a ladder
rung (R2) rather than an assumption, so its contribution is measured rather than
asserted.

**Trigger to add more:** if R2 still falls short, residual blocks at the bottleneck
come before wider channels, and both come before touching `levels`.

### fp32 for Phase 1, not bf16

`world_model_ingredients.md` specifies bf16 training. That line is about the
15M-parameter dynamics model at context 1024, where it is necessary. The tokenizer is
~1.5M parameters with ~400 MB of activations at batch 128, so fp32 costs nothing
against R-1's 7.5 GB and removes a class of numerical doubt from the single number
the whole phase turns on - an MSE around 1e-3 has little headroom in an 8-bit
mantissa.

**Trigger:** a measured step time that makes the ladder inconvenient. Then autocast
the forward pass only, keep the loss reduction in fp32, and re-run one rung both ways
to confirm the PSNR is unchanged.

### PSNR is computed on uint8-rounded reconstructions

The float decoder output is not what the pipeline delivers, and it is not what the
validator sees. Rounding first costs ~0.01 dB of optimism and makes rows 1 and 6 of
the gate describe the same frames. Cheap, and the alternative is a number that is
quietly about something else.

### The scene XML is frozen, and the offscreen size moves to config

`data_hash` is `sha256(canon(sim) + canon(data) + xml_bytes)`, so **any** byte of
`scene/arm_blocks.xml` - including a comment - changes it and orphans 300,000 frames.
The 96x96 arm needs `offwidth`/`offheight` at 96, which the XML currently states
literally at 64.

**Resolved by writing `model->vis.global.offwidth`/`offheight` from config between
`mj_loadXML` and context creation.** They are plain mutable `int` fields on `mjModel`
(`mjmodel.h`, `struct mjVisual_`), read by `mjr_makeContext`. Both existing guards
keep their teeth: `gl_context.cpp` still compares `mjr_maxViewport` against
`model->vis.global.offwidth`, now the config value, and `main.cpp` still cross-checks
the viewport against `cfg.width`/`cfg.height`. Cost is three lines and both
resolutions become a config-only change; the alternative was regenerating the
existing dataset in order to make a second one possible.

**Consequence to carry: the XML's literal `offwidth="64"` is now decorative, and the
comment explaining that cannot be added to the XML.** It lives in `main.cpp` and
here.

### Frames are preloaded as palette indices, not RGB

The loader reads **6,804 frames/s cold against 109,682 warm** at ctx=0, and training
needs ~13,000. With ~4.9 GB free against a 3.5 GB working set the page cache sits
exactly on the boundary, so the cold case is a real failure mode rather than a
first-epoch transient.

**The union of distinct byte triples over all 300,000 frames is exactly 7**, one per
palette entry, worst distance 0.75 - measured as a union over the whole set, not a
per-frame count, because "at most 7 per frame" would permit a larger union and make a
7-entry LUT lossy. So one byte per pixel is lossless and the train split preloads in
**1.16 GB instead of 3.49 GB**.

Two things this must not get wrong: the blob is bottom-up and `preload` bypasses the
sampler's flip, and nearest-palette-by-argmin is required because `rgba * 255` does
not land on integers. Both are assertions in the function, not comments.

**Trigger:** a scene change that adds a distinctly-coloured object raises the LUT
size. It stays a `uint8` index up to 256 entries, well past F-2's 24-colour ceiling.

### The token cache is per-shard and named by the run

One `.npy` of `uint16` per shard, shape `(frames, 8, 8)`, in a directory named by run
id, plus a manifest carrying `tokenizer_hash`, the checkpoint and per-shard frame
counts. 38.4 MB total.

Per-shard rather than one flat array, because a flat array is addressed through a
cumulative frame offset and that is an off-by-one factory; per-shard makes
`len(tokens) == shard.frames` a loud assert. Named by run rather than by
`tokenizer_hash` for the reason already recorded under provenance: two runs at
identical config and different seeds share a hash and produce different tokens.

## Sanitizer cost, and keeping E-3 cheap

E-3 requires ASan clean on the **full** data-generation run, plus 64-bit shard
offsets carrying a bounds assert. **MSVC provides ASan only.** Microsoft lists
`/fsanitize=undefined` and `/fsanitize=leak` among sanitizers it may ship *later*,
so this toolchain has no UBSan and no leak detection.

**Decided 2026-08-26: E-3 drops the UBSan half rather than adding a clang-cl
configuration.** Of the UBSan classes that apply to this code - MuJoCo calls, a
pixel buffer, segid integer math, appending to a blob - ASan already covers
out-of-bounds and use-after-free, and `/W4 /WX` rejects the sloppy casts
(`/w14242` lossy conversion, `/w14826` sign extension). The one real gap is
**signed overflow on shard byte offsets**: 300k frames at 64x64x3 is 3.6 GB
against `2^31` = 2.1e9, so an offset held in a signed 32-bit int wraps somewhere
past frame ~175k and silently corrupts the back half of a shard. That single class
is closed by typing every offset and frame counter `size_t`/`int64_t` and
asserting the computed offset against the mapped blob size before each append -
a check that runs in the optimised build, which a side-configuration sanitizer
would not. The alternative costs a second set of flag spellings and a second set
of MuJoCo/GLFW link quirks, to instrument ~200 lines of loop arithmetic.

**Trigger to revisit:** UB suspected in code ASan does not cover - alignment or
type-punning on the pixel buffer, or arithmetic outside the offset math above.
Whether clang-cl's UBSan works on Windows in diagnostic mode or only in trap mode
(`-fsanitize-trap=undefined`) is **unverified**; check it if the trigger fires,
not before.

Published overheads, for the ASan half that does exist:

| Sanitizer | Measured |
|---|---|
| ASan | **73% average** on SPEC CPU2006 (original paper), worst case 2.6x; 103% average in a later study |
| UBSan, full set | **up to 228%** on SPEC2006 across all 19 sub-sanitizers; 59% average elsewhere - no longer applicable, kept as the cost the clang-cl route would have carried |

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
contexts under WDDM. **The ingredients doc's 448 GB/s was therefore neither confirmed
nor refuted by this attempt.** It has since been **refuted** by a clocked-up rerun:
the part peaks at 384 GB/s and delivers 308.3 GB/s streaming reads. See the
verification log. Note the table above already recorded `clocks.max.memory` as
12001 MHz - the refutation was sitting in the data nobody divided.

The finding that matters is not the bandwidth number:

1. **E-4 ("rerun matches within 5%") is unachievable without pinning the clock
   state.** A GPU drifting between idle and boost clocks varies by more than an
   order of magnitude, not 5%. This is a statement about *clocks*; the pstate
   label is not the way to check it - see the correction under item 3.
2. **P-4 (p99/p50 <= 1.3) can be blown by a single boost transition mid-run**,
   independent of any kernel. On a laptop with dynamic boost this is the most
   likely cause of a failed P-4, and it would look exactly like a kernel problem.
3. **Therefore the bench harness must record `pstate`, `clocks.current.sm`,
   `clocks.current.memory`, `power.draw`, and `temperature.gpu` on every timing
   row, and refuse to run when the relevant clock domain sits below a configured
   fraction of max.** A timing without its clock state is not reproducible, so by
   E-4 it is not a valid number. This is cheap - one `nvidia-smi` query per run,
   outside the timed region per the observability rules.

   **Corrected 2026-08-23: this item originally said "refuse to run when
   `pstate != P0`", and that gate is refuted.** The reported pstate follows the
   **memory** clock domain, so a correct compute-bound run reads P4 while the SMs
   hold 2662 MHz of 3090 at 99 W of a 100 W cap. P0 appears only under
   memory-bound load, and no single load clocks both domains - so the original
   rule would have rejected every valid compute number this machine can produce.
   Gate **compute** on SM clock (>= 80% of `clocks.max.sm`) plus `power.draw`
   (>= 80% of `enforced.power.limit`), and **bandwidth** on
   `clocks.current.memory == clocks.max.memory`. `bench/gpu_probe.py` runs both
   phases; the refutation is the `pstate == P0` row in the verification log.
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
were tried. Everything now runs natively on Windows. **The C++ toolchain question
is closed**: MSVC via CMake generator `Visual Studio 18 2026`, C++20 confirmed by
`sim/main.cpp` printing `202002`, with `sim/build/` and `sim/build-asan/` both
building and running. MinGW was never needed.

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

- **At 64 tokens, P-1 needs graph capture and nothing beyond it; P-5 is the binding
  requirement after that.** **Corrected 2026-08-28 - this bullet still carried the
  pre-recalculation numbers and, worse, the conclusion they supported.** It read
  "~400-495 us against a 520 us budget, so the Phase 3 baseline lands near 30 fps
  *before* any optimization." Against the measured 308.3 GB/s the naive path is
  **~535 us against a 520.8 us budget, 103%**, so **the Phase 3 baseline does not
  reach 30 fps unoptimized** - that is what the fork table's "CUDA graphs are now
  required on both paths" means in Phase 3 terms. After graphs the 64 path sits at
  ~185 us, 36% of budget, and that is where the headroom for P-5's 3x comes from.
  At 144 tokens naive is **251%** of budget and still 100% *after* graphs, which is
  why DiagD is the only thing between that path and the wall.

  Kept as a worked example of the drift this doc is meant to prevent: the
  2026-08-23 re-measurement updated the fork table and `world_model_ingredients.md`
  where 448 GB/s appeared literally, but missed every place the old figure had
  already been *propagated into a consequence* - this bullet, plus two sites in the
  derived docs. A number is easy to grep for; a conclusion that silently inverted
  is not. **When a verification-log row refutes a prior value, search for the
  consequences by concept, not by digit.**
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
- **The palette has exactly one home: the XML's `rgba` attributes**, plus one
  entry no `rgba` can name. The validator reads the attributes with
  `xml.etree.ElementTree` (stdlib, ~5 lines) rather than duplicating the list in
  config JSON. Duplication here is the same class of bug as two validator
  implementations - the copies drift, and the symptom is a validator reporting
  "block missing" for a block that is present.

  **Measured correction 2026-08-27: the palette is the six `rgba` colours plus
  an implicit black void.** 14.1% of a frame is `(0,0,0)` - the framebuffer
  clear colour, showing past the far edge of a finite table with no skybox - and
  it appears in no `rgba` attribute. Without the extra entry `offpalette_px`
  reads ~578 px on a flawless frame and F-9 can never reach zero false
  positives. The entry lives in `mirage/validator.py` (`VOID_RGB`), not in the
  scene: adding a black geom to the XML would change `data_hash` and invalidate
  300k frames to fix a reader's bookkeeping. Also **keep the palette unrounded**
  - comparing against `0.90 * 255 = 229.5` rather than `230` halves every
  distance, and the worst a rendered pixel then sits from its own entry is 0.75
  over 8,000 frames.
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
| **Clean build from scratch** | **PASSED 2026-08-28** from a directory that had never held this project: `cmake -S sim -B sim/build -G "Visual Studio 18 2026" -A x64` then `cmake --build sim/build --config Release`, the same pair again for `sim/build-asan` with `-DMIRAGE_ASAN=ON`, both binaries run, and a 40-frame generation with each. Recipe, prerequisites and the four measured gotchas are in `README.md`, "Build" | E-2 |
| GL context is hardware, not software | binary prints `glGetString(GL_RENDERER)` after context creation; reject `GDI Generic` / `Microsoft Basic Render Driver`. A software fallback is ~50x slower and silently kills P-6. Deny by renderer name, not by vendor: vendor strings do not identify hardware. Deny-list, not an allow-list on "RTX 5060" - the allow-list form fails on any other machine that is perfectly fine, while the two software spellings are what actually indicate the failure. **PASSED 2026-08-23** in `bench/readback_probe.py`, and **again 2026-08-26 in the C++ port** (`sim/gl_context.cpp`): `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2`, offscreen buffer 64x64 | F-3 |
| **Per-call `mjr_readPixels` latency, in isolation** | time N readbacks in a tight loop, report us/call. Not end-to-end fps, which hides which term dominates. **PASSED 2026-08-23**: 25.4 us RGB / 49.6 us RGB+depth / 75.8 us with render, at P2. Two-pass render confirmed, 13x margin on P-6. `bench/readback_probe.py`. Re-run the render arm against the real scene | P-6 |
| **MuJoCo step time for this scene** | time `mj_step` alone, report us/step, with a guard that the measured steps actually contain arm-block contact. **PASSED 2026-08-23**: 10.5-10.8 us median driven, 131-176x under the ~1850 us the frame leaves after render+readback. `bench/step_probe.py`. CPU-only, so the P0 blocker does not apply | P-6 |
| **GPU clocked up, and bandwidth re-measured there** | two phases, because no single load clocks both domains: judge *compute* on SM clock (>=80% of max, drift <5%) and power (>=80% of the enforced limit); judge *bandwidth* on memory clock == `clocks.max.memory`. **Do not gate on `pstate == P0`** - it follows the memory domain and reads P4 during correct compute-bound work. **PASSED 2026-08-23**: 27.6 TFLOP/s fp16 at 2662 MHz / 99 W, 308.3 GB/s read at P0 / 12001 MHz. `bench/gpu_probe.py` | E-4, P-4, and every P-row |
| Determinism | generate twice, same seed, `cmp` the pixel blobs | F-4, E-1 |
| Sanitizers clean | full generation run under ASan, zero reports. No UBSan on MSVC: its one relevant class is covered by 64-bit offsets plus an offset-vs-blob-size assert, verified by deliberately overflowing an offset in a unit check | E-3 |
| Throughput | binary reports frames/sec at exit, assert >= 500 | P-6 |
| Dataset size | `du -sh` the dataset dir, assert <= 20 GB | R-4 |
| Memmap round-trip | write a known buffer in C++, read via `np.memmap`, byte-compare | F-8 |
| Palette adherence | validator mode 1 over all shards, `n_unique_colors` <= 24 | F-2 |
| Action histogram | `np.bincount` over `action`, assert roughly uniform over 9 | F-5 |
| Contact rate | nonzero fraction of `contact_mask` **bits 0..6** - mask the scripted flag off bit 7 first, or this reads over 50% - assert > 5% | F-6 |
| Occlusion rate | fraction of frames with any `visible_px[i] == 0` **where that block is visible again later in the episode**, assert >= 3%. `mirage.data.seen_later` owns the split and both `mirage/validator.py` and `bench/occlusion_probe.py` call it. Counting every zero-pixel frame reads **19.83%** against the restated **5.35%**, and 73% of the difference is blocks that never return | F-7 |
| Validator false positives | threshold sweep of mode 2 against mode 1, zero FP on ground truth; repeat against Phase 1 reconstructions | F-9 |

The occlusion and contact checks are the two most likely to fail on the first scene,
and both are fixed by editing the XML - arm reach versus block placement - not by
changing code. Budget an iteration or two.

## Verification log

What has been checked and how, so future sessions neither re-derive nor over-trust.

**"Asserted at" is not optional on a row that refutes or corrects something.** This
log is append-only and the rest of the tree is not, so recording *what came back
wrong* does nothing to retract the places that still say the wrong thing. Twelve
such places were found on 2026-08-28 and ten were real - the audit is D8 in
`phase0_debt_checklist.md`. The rule that follows from it:

- **A row reporting a refutation lists every site that asserts the refuted claim**,
  cited as `file, "Section name"` and never as a line number, which rots. Mark each
  corrected or still-standing. A row with nothing to retract gets `-`.
- **Fill it when the row is written**, not later. Later means someone greps this
  row's wording, and the assertion is phrased differently - which is exactly how
  the `mjvOption` and `record.cc` sites survived.
- **A finding with no site is a finding nobody will act on.** Two silent-failure
  gotchas here - `mjv_updateScene` before `mj_forward`, and `targetbody` without
  `target` - lived only in this table until they were routed into
  `phase0_structural_plan.md`, "Found by running Phase 0". If a row's sites column
  would be empty, that is the signal to give the finding a home.

| Claim | Method | Result | Asserted at |
|---|---|---|---|
| **F-9, F-2 over the full set, and the validator's two modes** (`mirage/validator.py`) | **run** - `python -m mirage.validator`: F-2 over all 300,000 frames, the F-9 sweep over 8,000 frames drawn from 500 windows | confirmed 2026-08-27. **F-2: max 7 unique colours over every frame** against the 24 ceiling - measured on the whole set, not a sample. **F-9's sweep is executable and mostly passes**, with one field it rules out. Passing: `offpalette_px` reads **0 on every ground-truth frame** at tau 8, and the worst distance any rendered pixel sits from its own palette entry is **0.75**, so tau has 11x headroom - that is the viable verdict expression today. **Ruled out: `px_count` cannot be a per-frame threshold.** The smallest `px_count` on a block ground truth calls *visible* is **1 px with margin 0**, because F-7 makes partial occlusion common - so a "block missing" rule has no headroom at all and the sweep says so rather than picking a number off a cliff. This is why `Sweep` reports `px_count_margin`. **Mode 2 is validated directly against ground truth**: pixel-only `px_count` equals the segmentation `visible_px` **exactly on 100.0% of 6,000 block readings, max \|diff\| 0** - the id-colour decode and the palette agree with each other to the pixel. **Two corrections to this doc, both found by measurement.** (1) `rgba * 255` does not land exactly - link0's `0.90 0.75 0.10` renders (229, 191, 25), and 0.65 rounds *up* to 166 while 0.90 rounds *down* to 229, so there is no rule worth modelling. With a byte-rounded palette, exact RGB equality counts zero pixels for **4 of 7 entries** and calls block0, block2, link1 and table missing on a flawless frame - the measured case for nearest-palette-by-argmin, which the design already specified. Keeping `Palette.rgb` unrounded is what takes the worst distance to 0.75. (2) **The palette needed a seventh entry the XML cannot provide** - see the palette bullet above. F-6 **20.69%** and F-7 **16.18%** re-confirmed from the meta over the full set. **Not yet done**: thresholds are printed, not written into config, because the documented build order recalibrates them against Phase 1 tokenizer reconstructions and writing a Phase-0-only number into `validator_hash` now would be a guess | The `rgba * 255` and seventh-palette-entry corrections land in `world_model_architecture.md`, "Findings that change the scene XML" - **corrected in place 2026-08-27**. `mirage/validator.py` holds `VOID_RGB` and keeps `Palette.rgb` unrounded. |
| **F-8, and the loader against P-7** (`mirage/data.py`, `bench/loader_probe.py`) | **run** over the full 300k set - `python -m mirage.data`, then `python bench/loader_probe.py` | confirmed 2026-08-27. **F-8 holds**: 448 sampled records decode identically through the structured dtype and through an independent `struct.unpack` at the documented offsets, and both blobs are exactly `frames x per-frame` bytes. Index: **500 episodes of 600 steps, ids 0..499 each once, none split across shards**, built in 9 ms. 20,000 sampled windows of 16 all carry **one `episode_id` with contiguous `step_idx`**. Throughput single-threaded, no DataLoader workers: sequential sweep **5.9 s = 50,979 fps** (597 MiB/s), random episode-aware windows **5,642 -> 7,646 win/s = 90,269 -> 122,342 fps** (131-177 us/window) across three passes. Against **P-7's 167 fps floor that is 306x sequential and 734x random**, so the loader costs 0.14-0.33% of the 30-minute epoch budget and **needs no workers, prefetch, or caching layer** - 3.43 GiB of pixels in 31.6 GB of RAM means the page cache holds the whole set, which is why pass 1 to pass 3 moves only 35%. **New gotcha, found by getting it wrong**: slicing an `np.memmap` returns a lazy view and touches no pages, so a first probe reported 3.4M fps having timed slice arithmetic - `__getitem__` returns `np.array(...)` and any probe must force the copy. **Row order settled**: the blob is bottom-up (`mjr_readPixels` origin is bottom-left, nothing in `sim/` flips it), and the flip belongs in the sampler, not in `Shard.pixels`, which stays raw so F-8 has something to compare. Verified against the scene rather than the convention - the camera at y=-0.5 at ~24 deg elevation puts the void past the far table edge at the **top**, which only happens after the flip, matching `bench/preview.png`. Split is by **episode, hashed** - 473 train / 27 val (5.4%) at `val_fraction` 0.05, disjoint; by frame it would leak, since consecutive frames differ by one 2 ms step | Lazy-memmap and row-order gotchas: `phase0_structural_plan.md`, "Found by running Phase 0" - **added 2026-08-28**, having been log-only for a day. Row order is enforced in `mirage/data.py` `WindowSampler`. |
| All dataset / meta / token-cache / KV / param / budget / diagonal arithmetic | computed | confirmed; values in the tables above | - |
| Ingredients doc's budget table | recomputed from first principles | KV 28.1/59.7 us vs its "~28/~60"; weights 65.0 us vs "~67"; params 14.56M vs "15M"; DiagD 4.27x/6.26x vs "4.3x/6.3x". **The doc's arithmetic is sound.** | - |
| Compression ratio identical across the fork | closed form + computed | exactly `1536/9` both, resolution-independent | - |
| `mjRND_SEGMENT` / `mjRND_IDCOLOR` exist, set via `mjvScene.flags` | docs, then **run empirically** on `arm_blocks.xml` | confirmed. Decoding RGB as `r + 256g + 65536b - 1` gives per-geom pixel counts that track the arm's pose, so the F-7 measurement path works end to end | - |
| numpy structured field view / torch behaviour | **run empirically** (numpy + torch 2.9.1) | non-contiguous as expected, but `from_numpy` succeeds. **Corrected an overstated claim** - SoA is a simplicity argument, not a speed one | `world_model_architecture.md`, "Shard format" - the SoA bullet was **corrected in place**, from a speed claim to a simplicity claim. |
| W&B offline mode, background metrics process, `x_disable_stats` | W&B source and docs | confirmed | - |
| ASan / UBSan overhead | published benchmarks | ASan 73% avg, UBSan full set up to 228%. **Corrected a too-optimistic guess** | `world_model_architecture.md`, "Sanitizer cost" - overhead table **corrected in place**. |
| `mjr_readPixels` at ~30 ms per call under GLFW | MuJoCo discussion #2222, then **measured and refuted** | **Does not reproduce.** 25.4 us RGB / 49.6 us RGB+depth at 64x64, GLFW offscreen on Windows - three orders of magnitude below the reported figure. **This was the highest-risk number in the plan; it is now the safest.** No pbuffer, no single-pass collapse | `world_model_requirements.md` P-6 risk row and `world_model_architecture.md`, "Render path and occlusion measurement" - both **corrected**, the latter keeping the retired trigger on the record. `README.md` Environment table said "day-1 blocker" until **2026-08-28**. |
| What the fixed per-call cost actually was | measured by toggling MuJoCo's visual defaults | `shadowsize=4096` and `offsamples=4`. Zeroing both cut `mjr_render` 316 -> 26 us and readback 71 -> 25 us; the readback share was an MSAA resolve. **Promotes `<quality offsamples="0" shadowsize="0"/>` from cosmetic to a hard `arm_blocks.xml` requirement with a measured justification.** `castshadow="false"` was listed alongside it in error - it is a `<light>` attribute and the compiler rejects it on a geom | - |
| Hardware GL context on Windows (F-3) | `bench/readback_probe.py` asserts `GL_RENDERER` contains neither `GDI Generic` nor `Microsoft Basic Render Driver` | confirmed - assert passes, offscreen framebuffer selected and non-empty. Re-run 2026-08-24 after the assert moved from an "RTX 5060" allow-list to this deny-list: still passes, `55.3 us` depth / `30.5 us` nodepth / `87.1 us` depth+render | - |
| C++ offscreen context end to end (`sim/gl_context.cpp`) | built `/W4 /WX` clean, then **run** against `scene/arm_blocks.xml` | confirmed 2026-08-26 - `GL_RENDERER` = `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2`, `currentBuffer` == `mjFB_OFFSCREEN`, `mjr_maxViewport` 64x64 matching the scene's `offwidth`/`offheight`, `mjr_getError` 0. All four are fatal at startup, so a software renderer or a silent 640x480 fallback cannot reach the data run | - |
| Target machine: sm_120, CUDA version, WSL2 availability | queried | **partly refuted** - capability (12,0) and CUDA 13.0 > required 12.8 both confirmed, and it is a 5060 **Laptop** GPU, which the docs do not distinguish. But WSL2 running is not the same as WSL2 rendering: its GL path is broken here, so the project moved to Windows | `world_model_architecture.md`, "Context" (superseded note) and `world_model_ingredients.md` hardware table - **corrected**. `README.md` Environment and `.gitattributes` header held the WSL2 assumption until **2026-08-28**. |
| **448 GB/s memory bandwidth** | re-measured clocked up, `bench/gpu_probe.py` | **REFUTED.** `clocks.max.memory` is 12001 MHz and nvidia-smi reports half the GDDR7 data rate, so the ceiling is 12001 x 2 x 16 B (128-bit bus) = **384 GB/s**. 448 is the *desktop* 5060 at 28 Gbps; the Laptop part runs 24. Measured **262 GB/s copy / 308.3 read / 318.9 write** = 68-83% of the real 384. Entered the project via `world_model_ingredients.md`. **Fork table recalculated against 308.3** - floors +45%, CUDA graphs now required on both paths, 144 path at 100% of budget without DiagD | `world_model_ingredients.md`, "Hardware" - **corrected**. `world_model_architecture.md`, "The 64x64 vs 96x96 fork" - floors recalculated. `bench/gpu_probe.py` still names 448 deliberately, as the assumption it disproves. |
| **`pstate == P0` as a benchmark validity gate** | **run, refuted** | The reported pstate follows the **memory** clock domain. Under compute-bound fp16 matmul the driver correctly drops memory to 9001 MHz and pstate reads **P4** while the SMs hold 2662 MHz of 3090 at 99 W of a 100 W cap - fully clocked up. P0 appears only under memory-bound load. **No single load clocks both domains**, so gate compute on SM clock + power and bandwidth on memory clock. The earlier P0 rule would have rejected every valid compute number this machine can produce | **Three sites, all corrected 2026-08-28 - five days after this row was written.** `README.md`, "Taking a measurement"; `world_model_architecture.md`, "Benchmark validity" items 1 and 3; `phase0_structural_plan.md`, "The two numbers to take before writing file 3". The worst case in this table: the doc instructed the wrong action while this row refuted it. |
| GPU thermal state gates everything above | measured before and after a chassis cooling fix | **Enforced power limit 55 W -> 99.86 W, idle 73 C -> 58 C, fp16 matmul 3.0 -> 27.6 TFLOP/s - 9x from cooling alone.** The instantaneous throttle flags all read `Not Active` throughout; the evidence was in `nvidia-smi -q -d PERFORMANCE` **counters**, which showed SW Thermal Slowdown for essentially the whole uptime. **Sample the counters, not just the flags.** Compute now peaks at 85 C with no measurable decay (-0.3%) | - |
| Full per-frame generation cost, real scene | measured, `bench/frame_probe.py`, 5x1000 calls, XML `main` camera, segmentation pass included | **`mjv_updateScene` 1.1 us** - the last day-1 unknown, and negligible. 1-pass frame 61.6 us, 2-pass 144.0, **step + 2-pass 178.5 us = 5,602 fps, 11.2x over P-6**. **Parallel generation is not needed.** 300k frames in ~1 min single-threaded. p99 2008 us exceeds the whole frame budget but P-6 is throughput not latency, so at 1% weight it costs ~10% of mean. Taken while the GPU was thermally capped, so it is a pessimistic bound | - |
| `mjv_updateScene` before any `mj_forward`/`mj_step` | **run, silently wrong** | renders an entirely black frame while `scene.ngeom` reads a correct 6. mjData's derived `xpos`/`xmat` are zero until forward dynamics runs. Checking the geom count does not catch it - same family as the `cam_targetbodyid = -1` entry | **Nowhere, for five days.** Now `phase0_structural_plan.md`, "Found by running Phase 0" - **added 2026-08-28**. A silent-failure finding that lived only in this log is a finding nobody acts on. |
| MuJoCo step time for this scene | **measured** - `bench/step_probe.py`, 5x1000 steps, reset every 200, idle and driven arms, 6 reruns | **10.5-10.8 us median driven** (floor over 6 runs; the first run after other machine load reads up to 14.1), **p99 32-60 us**, at 12-14 total contacts. **131-176x under the ~1850 us P-6 allowance - P-6 is not at risk from physics.** The guard is now arm-block specific (geom-pair filter excluding the table, which rests under every block): **77.9% driven vs 0.0% idle**, replacing an earlier `ncon > 0` check that scored both arms at 100% and was vacuous. Contact counts and both fractions came back **bit-identical across 6 separate processes** - unseeded determinism, relevant to the Phase 0 replay gate. Max is 11-100x median on both arms: Windows scheduler preemption, not solver work. **E-4's 5% is not demonstrated for this number** - the series had not plateaued after 6 runs and needs a quiescent-machine protocol. CPU-only - GPU pstate does not apply | - |
| Offscreen GL on WSL2 | attempted, **refuted** | `bench/egl_probe.py` reaches a context but `GL_RENDERER` is `llvmpipe` (CPU) under every platform and driver override. Root cause is below Mesa: `dxgkio_query_adapter_info: Ioctl failed: -22`, no `/dev/dri` node. CUDA is unaffected - different ioctl path. **Moved all rendering to Windows** | **Four sites, all corrected 2026-08-28.** `world_model_ingredients.md` "Components to build" and the phase table; `world_model_learning_roadmap.md` "MuJoCo documentation" and "Communities"; `.gitattributes` header. Survived because "EGL" was licensed by a blanket read-as note scoped to one file. |
| ASan-under-CPython specifics | **not verified in detail** | claim softened; the boundary decision does not depend on the magnitude | `world_model_architecture.md`, "C++/Python boundary" - already carries "The magnitude of that gap is unverified". **No stale site.** |
| **Ambient-only headlight is what collapses the palette** | measured on the first working scene | Adding `<visual>` with `offsamples=0`, `shadowsize=0` and `headlight diffuse="0 0 0" specular="0 0 0"` took the colour count 28 -> 6. Plain `rgba` needs no materials: diffuse light shades per face, so one `rgba` becomes three entries. **Replaces the erroneous `castshadow` requirement** | - |
| First working `arm_blocks.xml` meets F-2, F-6, F-7 | measured | `nq=23` (2 hinges + 3 free joints), 6 colours, arm-block contact in 95.8% of constant-drive steps against F-6's 5% floor, and block0 at **0 visible px** at rest rising to 68 px when the arm swings - F-7's full-occlusion mechanic demonstrated. The contact figure is a full-torque drive, not the 50/50 policy | - |
| `<camera mode="targetbody">` without `target` | **run, silently wrong** | compiles clean, `cam_targetbodyid = -1`, camera aims nowhere. Renders empty ground with no error - the failure mode a dataset run would not survive. Give `target=` or set orientation with `xyaxes` | **Nowhere, for five days.** Now `phase0_structural_plan.md`, "Found by running Phase 0" - **added 2026-08-28**. `scene/arm_blocks.xml` uses `xyaxes`, so the scene never carried the bug. |
| "Decorations off in `mjvOption`" by zeroing the flag array | **run, refuted as written** | `mjv_defaultOption` already leaves every decoration off; zeroing all flags also clears `mjVIS_STATIC` and worldbody geoms stop drawing entirely. Do not zero the array | **Two sites, both corrected 2026-08-28.** `world_model_architecture.md`, "Render path and occlusion measurement" final paragraph, and `phase0_structural_plan.md` gotchas table. Both said "disable"; this row says "by zeroing the flag array", so grepping this row's own wording never found them. |
| **Phase 0 gate at the original physics** (`data_hash 0259947e`) - **SUPERSEDED 2026-08-28 by the row below**, which re-ran every one of these against `gear 6 / damping 1.5`. Kept because the comparison is the evidence that the scene change cost nothing. | **run in full** at `mirage/configs/base.json`, `data_hash 0259947e...`, git `ce5c193` | **all three met 2026-08-27.** 300,000 frames / 7 shards / **3.7 GB** in 44.3 s at **6,775 fps**, 13.6x over P-6. Replay: two complete runs at seed 0 gave **bit-identical** `.pixels` and `.meta` for all 7 shards - determinism demonstrated over the real dataset, not a sample. `GL_RENDERER` reads `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2`. Over the whole set: F-5 min share **7.15%** and ratio **2.27** (thresholds 5% and 2.5), F-6 **20.69%**, F-7 **16.18%**, F-2 **7 colours**, and episodes 0..499 each carrying exactly 600 steps with `data_hash` and `git_sha` identical across all 7 sidecars. **Corrects a small-sample figure**: F-6 and F-7 read 62.4% and 40.3% off a single 1,200-frame shard, 3x the full-set values - a 2-episode window does not estimate a 500-episode run | - |
| **Phase 0 gate re-run at `gear 6 / damping 1.5`, `action_hold_steps 15`** | **run in full 2026-08-28** at `mirage/configs/base.json`, `data_hash 219ab0af`, git `29604ae`. Generated, then generated again at the same seed and SHA256-compared all 14 blobs; then `python -m mirage.data`, `python -m mirage.validator`, and a numpy pass over all 7 meta blobs | **All three gate conditions hold, and every M-tier row re-verified.** 300,000 frames / 7 shards / 3.686 GB in 42.7-60.1 s at **4,993-7,033 fps** (P-6 floor 500). Replay **bit-identical across all 14 blobs**. `GL_RENDERER` reads `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2`. Over the whole set: **F-2 7 colours**, **F-5 min share 7.17% / ratio 2.15**, **F-6 16.63%**, **F-7 19.83%**, F-8's 448 records double-decode, F-9's `offpalette_px` **0 on every ground-truth frame** at tau 8 with mode 2 exact on **100.0%** of 6,000 block readings. **The two numbers this change was for:** Q-4's ground-truth ceiling measured on the shipped 50/50 mix over 421,701 driven (frame, joint) pairs is **91.5%**, up from 83.1% - so the relative bar is 82.3% and the ceiling now clears the old absolute 90% outright; and ctx=15 window coverage is **60.9%**, up from 41.3%, a 1.47x increase in windows carrying any evidence of what an action does. **Costs, both small:** F-6 fell 20.69% -> 16.63%, still 3.3x its floor, and throughput fell from 6,775 fps, which is the more responsive arm making more contacts for the solver. Scripted-reach arrival *rose*, 36.4% -> 40.0% | `docs/phase0_report.md` sections 2.1, 2.3 and 6 - **corrected 2026-08-28**. `runs.jsonl` gains an entry rather than an edit. **`AGENDA.md` in the main checkout still quotes the superseded figures** and was rewritten there for Phase 1 while this branch ran, so it is left alone deliberately - reconcile at merge. |
| Shard writer and the whole generation loop (`sim/shard_writer.cpp`, `sim/main.cpp`) | built `/W4 /WX` clean in both configurations, then **run** on a 6-episode config, blobs read back in numpy | confirmed 2026-08-27, **ASan-clean**. 3,600 frames over 3 shards at **6,760 fps** optimised / 5,876 ASan, against P-6's 500. **F-4 holds at the pixel level**: two runs at one seed produced byte-identical `.pixels` and `.meta` blobs, which is stronger than `policy_dry_run`'s action-sequence check. F-8's round trip works as specified - `np.memmap(...).reshape(-1,64,64,3)` needs no stride math, and a structured dtype built from the sidecar's `meta_joints`/`meta_blocks` has `itemsize` **46**, matching the doc. Off one shard's meta blob under the real policy: **F-6 62.4%**, **F-7 40.3%**, **F-2 7 unique colours**. The sidecar is absent until `commit()`, checked at the one moment it is observable. Two decisions came out of it: **`data_hash` is passed in as a required flag, never recomputed in C++** - a second canonical-JSON hash would have to match Python's float formatting byte for byte - and **shards rotate only on episode boundaries**, because `Policy` is seeded per shard and a mid-episode rotation would reseed mid-episode | - |
| C++ ground-truth read end to end (`sim/truth.cpp`) | built `/W4 /WX` clean in both configurations, then **run** against `scene/arm_blocks.xml`, optimised and ASan, `mjr_getError` 0 | confirmed 2026-08-27, identical numbers under both builds and **ASan-clean**. The id-colour decode is now verified from C++ as well as Python: at rest `visible_px = 0 175 44` of 4096, and block 1 reads **175 px open -> exactly 0 parked out of frame -> 175 restored** - the check that fails if the segid channel order is wrong or the histogram is keyed on anything but that block's own segid. Block 0's 0 at rest is scene geometry, the sightline to it crosses `link1`, and it matches the earlier Python measurement. **New gotcha: the segmentation pass leaves the offscreen framebuffer holding id colours**, so the RGB readback must precede `Truth::read`; reversed, the shard stores id colours and nothing downstream fails. Sweep rates (1500 steps, joint 0 at full torque): F-6 60.93%, F-7 79.73%, both **biased high** - the arm never stops, and a block punted off the table reads 0 for every remaining frame - so they show the measurement responds, not that the scene passes | The id-colour framebuffer gotcha is in `sim/truth.h` on `Truth::read`, current since written, and in `phase0_structural_plan.md`, "Found by running Phase 0" - **added 2026-08-28**. |
| MSVC sanitizer capability | Microsoft docs, then **built and run** | ASan only: `/fsanitize=undefined` and `/fsanitize=leak` are listed as possible future work, so **E-3's UBSan half has no implementation and there is no leak detection**. ASan also needs `/Zi` (`C5072` is fatal under `/WX`) and its runtime DLL copied beside the binary - `/MT` does not remove the dynamic import. C++20 builds clean | - |
| `record.cc` "ships with MuJoCo" | **checked the installed wheel** | refuted for pip: `mujoco` 3.12.0 ships headers and two test XMLs, no samples and no model zoo. Read it in the GitHub repo instead | `world_model_learning_roadmap.md`, "MuJoCo documentation" - **corrected inline with its evidence**. `phase0_structural_plan.md`, gl_context section - was already correct. A keyword sweep flags the gotchas-table mention too; that one is fine in context. |
| Action encoding round-trips, and the model's ids resolve (`sim/policy.*`) | built `/W4 /WX` clean in both configurations, then **run** against `scene/arm_blocks.xml` | confirmed 2026-08-26 - all 9 actions decode to distinct `ctrl` pairs and re-encode to the same index; `nu=2`, so `action_count` is `3^2 = 9`. Block bodies discovered by name prefix as ids **3 4 5**, `joint0`/`joint1` `qposadr` **0**/**1**, matching a direct model probe. Sanitizer build runs clean. Note `jnt_qposadr` and `jnt_dofadr` **diverge** for free joints - block1 is 9 vs 8, block2 16 vs 14, because a free joint is 7 wide in `qpos` and 6 in `qvel`. Never compute either address, always read the array | - |
| **`sim.action_hold_steps = 20`** | **measured 2026-08-28** - `bench/hold_probe.py`. Settling time by driving one joint from rest and reading the step count at 63% of terminal velocity, at four link1 angles; then agreement swept over nine holds, 60 episodes x 600 steps each | **The estimate was wrong and the parameter does not settle.** The doc predicted ~15 steps from `inertia / damping` using `dof_armature = 0.01`; **armature is the term *added* to the mass-matrix diagonal, not the diagonal**, so the link inertia was omitted. Measured: **joint0 21-34 steps** depending on link1's angle (M00 runs 0.0203-0.0350), **joint1 12 steps** and constant, since joint1's own inertia does not depend on the link it carries. The first-order `M/b` prediction matches every measurement (34 vs 35.0, 30 vs 29.4, 21 vs 20.3, 12 vs 12.3), so both are trustworthy. **Consequence: at the shipped hold of 20, ground-truth action-following scores 83.1%, against Q-4's 90% bar** - so a model that reproduced the simulator exactly would fail Q-4. Agreement does not reach 90% until **hold 60**, and by then only 19.4% of ctx=15 windows carry an action change, against 62.1% at hold 20. **No hold satisfies both**; see the row below. `20` is unchanged pending that decision - it is now a known trade rather than a guess | `mirage/configs/base.json` `sim.action_hold_steps`, `sim/policy.h` `PolicyParams`, and `world_model_requirements.md` Q-4. **The value is unchanged and the trade is open** - D1 and D2 in `phase0_debt_checklist.md`. |
| **Q-4's 90% bar against the simulator's own score, and whether the hold trade is forced** | **measured 2026-08-28** - `bench/hold_probe.py` sections B, C and D. Agreement is `sign(qpos_t+1 - qpos_t)` against the commanded sign, on driven digits, over the random half of the policy | **Two findings. (1) Q-4's threshold is above its own ceiling.** Ground truth scores **83.1%** at the shipped hold, so a perfect world model fails Q-4 by 7 points. This is a requirement defect, not a dataset defect, and it is independent of whatever the hold is set to. **(2) The trade is not forced.** Agreement wants a long hold and window coverage wants a short one, but the transient is `tau = M/damping` while arm speed is `v_term = gear*ctrl/damping` - **scaling gear and damping together holds the speed and shrinks the transient.** Measured at 40 episodes: `gear 2 -> 6`, `damping 0.5 -> 1.5` takes tau0 **34 -> 12** with terminal velocity flat at 3.92 -> 4.00 rad/s, and hold 15 then reads **90.3% agreement with 83.3% of windows carrying a change** - both bars, which nothing achieves at the shipped physics. **Not applied**: it is a scene edit, so it moves `data_hash`, invalidates the 300k set and requires F-5, F-6 and F-7 re-verified. One caveat that survives it - the window column is the random half; the shipped 50/50 reads 41.3% against 62.1%, so the scripted half roughly halves coverage and no physics change touches that cause | Nothing asserts the refuted claim yet, because Q-4's ceiling was never written down. Now recorded at `world_model_requirements.md`, "Requirements at risk" - **added 2026-08-28** - and as D1/D2 in `phase0_debt_checklist.md`. `scene/arm_blocks.xml` still carries `gear=2` / `damping=0.5`. |
| `sim.reach_noise_prob = 0.15` | **superseded - now measured**, see the frontier row below | Probability that the scripted reach substitutes a uniform random action. Exists so the scripted half does not collapse into three deterministic trajectories. The "tune against the histogram, not by eye" instruction was followed and the answer was that **this is the wrong knob**: whole-action replacement is dominated at every setting by the Jacobian deadband, and by per-digit corruption once the deadband is on | `mirage/configs/base.json` - the key is now `reach_digit_noise_prob`, and `sim/policy.h` `PolicyParams` carries the reasoning for why. **No stale site.** |
| `sim.reach_done_dist = 0.04` m | **measured 2026-08-27** | Fingertip-to-block distance at which the reach re-targets. Sized against the scene: blocks are 0.025 half-size, `link1` is 0.018 half-width, total arm reach 0.33. At 600 steps the median closest approach is **0.042 m** - the threshold sits almost exactly on the median, which is the most sensitive place it could be, so arrival rate is unusually responsive to it. Left at 0.04 because arrival is a diagnostic, not a requirement; F-6's contact rate is the number that would justify moving it, and truth.cpp does not measure that yet | - |
| `begin_episode` / `step`, and F-4 for the action stream (`sim/policy.*`) | built `/W4 /WX` clean in both configurations, then **run** - `policy_dry_run`, 200 episodes x 600 steps, twice | confirmed 2026-08-27 - both passes byte-identical over 120,000 actions. The fingertip offset derives to **(0.15, 0, 0)** body-local, exactly the geom `pos.x + size.x` the XML implies, and `qposadr/dofadr` read **0/0** and **1/1**. ASan build runs clean. `mj_jac` takes a `const mjData*`, so `step` cannot perturb the sim | - |
| `sim.steps_per_episode = 200` | **run, refuted** | 200 steps is 0.4 s of sim time and the arm cannot cross to a block from a random start angle: the scripted half arrives in **13%** of episodes, median closest **0.193 m**. At 600 it is **44%** and **0.042 m**, at 1500 **52.5%**. Changed to 600 x 500 episodes, the same 300k frames, which also cuts ctx=15 boundary loss from 7.5% to 2.5%. The A/B that separates "reach is broken" from "episode too short" is steering-off (`reach_noise_prob = 1.0`): 4% arrival, median 0.258 m, a 6x separation in median | `mirage/configs/base.json` - now 600. `world_model_architecture.md`, "Policy mixing is per-episode" and `AGENDA.md` - both **corrected in place**. |
| The scripted reach emits only corner actions | **measured** | 120k steps at 600: corners (both joints driven) took **70.6%** against a uniform 44.4%, max/min **4.26**. Cause is structural, not a bug - `sign(gain)` on a double is never zero, so the neutral digit is unreachable by the reach. Random draws account for every non-corner count: predicted 7,667 per bin, observed 6,080-7,700 | - |
| `sim.reach_noise_prob = 0.15`, and the flatness/reach frontier | **measured**, 14 configurations at 200 episodes x 600 steps | Arrival is flat at 42.5-44% down to ratio **2.44**, then falls - 33.5% at 2.17, 29.5% at 1.87. Jacobian deadband **0.04 m/rad** buys 4.26 -> 2.55 for 1.5 points of arrival; per-digit noise dominates whole-action noise once it is on (2.09 at 39% against 2.17 at 33.5%). Every no-deadband configuration is strictly dominated. F-5's 2.5 threshold is set at this knee | - |
| A flatness below ~1.7 is reachable | **refuted by measurement** | At deadband 0.04 the corner-vs-edge imbalance is already gone - corners 48.4% against 44.4%. The residual is same-sign corners at ~19,200 against opposite-sign at ~9,300, because a two-link planar arm reaching outward turns both hinges the same way. Kinematic, not tunable | `world_model_architecture.md`, "F-5's threshold is the knee of a measured curve" - carries the refutation in place. **No stale site.** |
| F-5 passes with `jacobian_deadband = 0.04` and `reach_digit_noise_prob = 0.15` | built `/W4 /WX` clean in both configurations, then **run at F-5's own sample size** - 2,000 episodes x 600 steps, twice, 22.4 s | confirmed 2026-08-27 - min share **7.15%** against the 5% floor, ratio **2.06** against the 2.5 ceiling, and both passes identical over 1.2M actions. Costs arrival 44% -> 35.7%, median closest 0.042 -> 0.052 m. ASan build reports nothing. The startup smoke run is 200 episodes and reads 2.30 / 6.55%, so it is marked indicative - at 200 episodes the ratio moves ~0.2 between samples, which is why F-5 names 2,000 | - |
| `sim/truth.*`, and F-6 / F-7 under a single-joint sweep | built `/W4 /WX` clean, then **run** against `scene/arm_blocks.xml` - `truth_dry_run`, 1500 steps, joint 0 at full torque | confirmed 2026-08-27 - **F-6 60.93%** of frames against the 5% floor, 12x margin; F-7 79.73% against 3%. All three fatal checks pass, which is what validates the id-colour decode `r + 256*g + 65536*b`: at rest `visible_px` = **0 175 44** of 4096, block 1 teleported below the table reads exactly **0**, restoring it returns **175**. **Both rates are indicative.** F-7's 79.73% was an artifact - block 0 read 0 px in the resting pose, so every unmoved frame counted as a full occlusion; fixed in the row below. Neither is the verdict: the sweep is not the 50/50 policy, and the enforcing check is the validator over a shard | - |
| `block0` moved to y = **-0.06**, and what it costs | **run** - `truth_dry_run` 1500 steps, then F-5 re-run at its own 2,000 x 600 sample size, 21.5 s | confirmed 2026-08-27 - block 0 goes **0 -> 46 visible px at rest** and still reaches 0 mid-sweep, so F-7 now measures the arm passing in front of a block rather than one static pose. Cause: the camera is at y=-0.5 looking toward +y and the arm rests along +x at y=0, so any block at small positive y is behind it. Costs: **F-6 60.93% -> 50.00%** (still 10x the floor), **F-7 79.73% -> 58.07%**, and **F-5 ratio 2.06 -> 2.18** against the 2.5 ceiling with min share 7.15% -> **7.05%**. Arrival 35.7% -> 36.4%, median closest 0.052 -> 0.053 m. The F-5 row above was measured at y = +0.06 and its exact figures no longer describe this scene; the verdict is unchanged. **F-7 keeps one known bias** - a block knocked off the table reads zero px forever and counts as occluded, which these numbers cannot separate from a genuine occlusion | - |
| **E-2, clean build from scratch, both build types** | **run in full 2026-08-28** from a directory that had never held this project: clone, configure, build, run, generate 40 frames - Release and ASan, four CMake commands, no manual step | **E-2 MET, and it had never once been executed.** MSVC 19.50.35728 / toolset 14.50.35717, CMake 4.3.1, generator `Visual Studio 18 2026`; `__cplusplus` 202002, MuJoCo 3.12.0, GLFW 3.5.1, all three dependencies fetched at configure time and none vendored. From empty: Release **16.5 s configure + 11.6 s build**, ASan **14.7 s + 9.7 s**. Both binaries start from a plain shell - `mujoco.dll` and `clang_rt.asan_dynamic-x86_64.dll` are copied beside them by POST_BUILD steps rather than found on PATH. **Three findings only a from-empty run produces.** (1) **The build cannot live under `%TEMP%`**: MSBuild's FileTracker fails `FTK1011` inside CMake's compiler probe, so it surfaces as a missing C++ compiler rather than as a path problem - it cost the first attempt. (2) `sim/build` and `sim/build-asan` keep **separate `_deps` trees**, so the second configure re-downloads MuJoCo and re-clones GLFW; nothing is shared. (3) The ASan build ran the same 40-frame generation **sanitizer-clean with both C++ self-checks passing, and its blobs are byte-identical to the Release build's** - widening F-4 from same-build-twice to two build configurations, on one toolchain and one driver | `README.md`, "Build" said **"Not yet implemented"** - **replaced 2026-08-28** with the verified recipe, prerequisites, the smoke-test command and the gotchas above. The `Verification` table's E-2 row named `cmake -B build && cmake --build build`, which is not a command that works here - **corrected in place**. `docs/phase0_debt_checklist.md` D4 - **closed**. |
| **`data_hash` forks on the working tree's line endings** | **found by the E-2 clean clone above, then measured directly.** The clean clone and this worktree stand on the same commit; both were asked for `load('mirage/configs/base.json').data_hash` | **REFUTES the assumption that `.gitattributes` closes this.** The worktree read **`219ab0af`** and the fresh clone **`18a76531`**, and `scene/arm_blocks.xml` is byte-identical between them once CR is stripped. `data_hash` hashes the XML's **raw bytes**, so a CRLF tree and an LF tree name different datasets for one scene - **precisely the fork `.gitattributes`' own comment says must never happen**, live in the repo. Cause: `* text=auto eol=lf` landed in `f0cdf09`, and git applies `eol` only at **checkout**; it never rewrites a working tree that already exists, so every file already on disk kept the CRLF that `core.autocrlf=true` had given it. `git status` hid it - `text=auto` normalises before comparing, so the tree read clean. **The whole 300,000-frame set was stamped `219ab0af`, a hash no fresh clone could reproduce**, which is E-1 and E-4 provenance broken rather than merely untidy | Fixed in `mirage/config.py`, `scene_bytes` - CRLF is normalised out **before** hashing, so provenance no longer rests on any checkout honouring an attribute, and `_self_check` now asserts a CRLF scene hashes as its LF twin. Worktree renormalised to LF (12 files). `.gitattributes` kept, for diffs. The hash definition in this doc, "Provenance and hashing", **carries the normalisation now**. `data_hash 219ab0af` is **superseded by `18a76531`** - the gate row below re-ran everything against it. |
| **Phase 0 gate re-run at `data_hash 18a76531`, with D3's scripted flag in the record** | **run in full 2026-08-28.** Regenerated all 300,000 frames after the CRLF fix, generated a second time at the same seed and SHA256-compared all 14 blobs, then `python -m mirage.data`, `python -m mirage.validator`, and the ASan build over the writer path | **All three gate conditions hold, every M-tier row re-verified, and every measured value is unchanged from the `219ab0af` run** - which is the expected result, since `sim/policy.cpp` and `scene/arm_blocks.xml` are untouched and only the `contact_mask` byte's high bit differs. 300,000 frames / 7 shards / 3.686 GB in 45.1-50.1 s at **5,987-6,653 fps** (P-6 floor 500). Replay **bit-identical across all 14 blobs**. Over the whole set: **F-2 7 colours**, **F-6 16.63%**, **F-7 19.83%**, F-8's 448 records double-decode, F-9's `offpalette_px` **0 on every ground-truth frame** at tau 8.0 with mode 2 exact on **100.0%** of 6,000 block readings. F-5 is unchanged by inspection rather than by re-measurement: `git diff sim/policy.cpp scene/arm_blocks.xml` is empty, so it is the same computation that read min share 7.17% / ratio 2.15. **D3's flag reads correctly**: the scripted bit is **constant across every frame of all 500 episodes** - checked per episode, because a per-frame bug would still produce a plausible mix in aggregate - and **53.2% of episodes are scripted**. That F-6 still reads 16.63% rather than jumping past 50% is the evidence that the readers mask properly | `docs/phase0_report.md` sections 1, 2.1, 2.2 and 2.3 - **corrected 2026-08-28**. `docs/phase0_debt_checklist.md` D3 - **closed**, its "only if a regeneration happens anyway" condition having fired a second time. The `219ab0af` gate row above is **superseded**; the `0259947e` row was already. `runs.jsonl` gains an entry rather than an edit. **`AGENDA.md` in the main checkout still quotes the superseded figures**, unchanged from the previous note - reconcile at merge. |
| **F-8 and F-9 are runnable in a clone that has never generated anything**, and `validator.offpalette_tau` | **run 2026-08-28.** A 40-frame fixture (2 episodes x 20 steps) written by the binary from the E-2 row above, committed at `mirage/fixtures/`; both self-checks fall back to it when `data/shards` holds no committed shard. Separately, `tau` moved out of three `validator.py` default arguments and into config | **Both acceptance tests now run in a fresh clone, a linked worktree and CI**; before this they died on `FileNotFoundError` in all three, because `data/` is 3.5 GB and correctly gitignored - which meant F-8's and F-9's acceptance tests were unrunnable by anyone who had not first generated 300,000 frames. The fixture costs **4.9 KB packed** (491,520 bytes raw) at `data_hash d897fc24`. It is **real writer output, not synthesised** from `meta_dtype` plus random pixels - F-8 is a claim about the bytes `shard_writer.cpp` emits, and a synthesised shard would agree with the reader by construction while testing nothing. **Three checks are skipped on the fixture and say so**: F-6, F-7 and the episode-level train/val split, all claims about the *dataset* that 40 frames from 2 episodes can only meet or miss by luck. The split's hash function is instead checked over **4,000 synthetic episode ids, 4.98% to val**, which needs no data at all. **`validator_hash` moved**, which is the entire point of the tau change: with tau in code, editing it left `validator_hash` unchanged, so two Q-3 coherence-horizon rows taken at different taus would have carried the same hash and claimed to be comparable. The **value** stays uncalibrated until the Phase 1 recalibration - this moved where the number lives, not what it is | The F-9 row above says thresholds are "printed, not written into config" - **still true**: `tau` is the measurement's radius, not a verdict threshold. `.gitattributes` said the blobs were "gitignored, but declare intent in case one is ever forced in" - **corrected**, one now is. `docs/phase0_debt_checklist.md` D9 and D10 - **both closed**. |
| **How much of F-7 is recoverable occlusion, and whether a block can leave the table** | **measured 2026-08-28** - `bench/occlusion_probe.py`, over all 300,000 frames. Split `visible_px == 0` into frames where the block is visible again later in the same episode and frames where it never is, then projected every block into the camera frustum to say which | **REFUTES the recorded cause, and the bias is larger than the framing implied.** No block has ever left the table: **0 of 900,000 block-frames**, worst reach 0.446 m against a **1.2 m** half-extent, with an arm that reaches ~0.33 m. So the `is_on_table` field this project had been reserving for the next regeneration would have separated nothing. The bias is real: of F-7's **19.83%**, only **5.35%** is recoverable occlusion and **14.48%** is a block that never returns - **73% of the headline number**. The honest occlusion rate is 5.35%, which still clears the 3% floor but at **1.8x, not 6.6x**; quote that one. The cause is the **camera**: **12.66%** of frames have a block outside the 45-degree frustum, and only **3.56%** are terminal with the block still in frame - parked behind the arm or another block. The projection is validated against the renderer rather than trusted, agreeing on **99.9988%** of frames where a block rendered a pixel, and it needs the cube's circumradius as a margin because blocks reach within **0.196 m** of the camera, close enough that a centre leaves the frustum while a corner still renders. **No new field is needed and no regeneration**: the split falls out of `visible_px` alone via a reversed cumulative maximum, so Q-6 can score object permanence on occlusion events that actually end, over the shards that already exist | `docs/phase0_debt_checklist.md` Tier 5's F-7 row asserted the knocked-off-table cause - **corrected and closed 2026-08-28**, with the measurement written up as D12. `docs/phase0_report.md` section 3's F-7 row now carries the split rather than the bare 19.83%. **F-7 was restated on these numbers the same day** - see the row below. |
| **F-7 restated to count recoverable occlusion only** | **decision 2026-08-28**, taken on the row above's measurement. Requirement reworded, `mirage/validator.py`'s acceptance check rewritten to score per episode, the config key renamed, and the split moved into `mirage.data.seen_later` so the validator and the probe share one implementation | **F-7 now reads 5.35% against an unchanged 3% floor - 1.8x, where the old counter read 19.83% and implied 6.6x.** What changed is the quantity, not the bar: a block that reads zero pixels for the rest of an episode is gone rather than occluded, and Q-6 cannot score object permanence on an event that never ends. **The config key is renamed** `occlusion_rate_min` -> `recoverable_occlusion_rate_min`, which is what moves `validator_hash` `e15f209e` -> `48882ee2`. That rename is load-bearing, not cosmetic: the threshold *value* is unchanged, so without it the hash would have stayed put while the measurement underneath it changed meaning, and two incomparable F-7 rows would have claimed to be comparable - the exact failure D9 moved `tau` into config to prevent. `data_hash` is untouched, so **no regeneration**. **Cross-checked**: `mirage/validator.py` computes the rate per episode over the concatenated shards and `bench/occlusion_probe.py` computes it over a reshaped episode axis, written separately, and they agree at 5.35% to the digit | `docs/world_model_requirements.md` F-7 row and its risk row - **restated 2026-08-28**, with the reasoning and the precedent named. `docs/phase0_report.md` section 2.3's F-7 row now reads 5.35%, and section 6's "F-7 carries one known bias" row is **struck and closed**. The `Verification` table's occlusion row above named the old counter - **corrected**. `docs/phase0_debt_checklist.md` D12 records the decision. **Residual risk, now on the record**: the margin is 1.8x, not 6.6x, so a scene edit has far less room than the old number suggested. |
| **Phase 1's six pre-work measurements describe a dataset that no longer exists** | **checked 2026-08-28 while reconciling `AGENDA.md`.** Read the `data_hash` the main checkout's shards carry, and the commit order of the Phase 1 drafting commits against the regeneration | **They were taken against `0259947e`** - the original physics, `action_hold_steps 20`, before the scene was rescaled to `gear 6 / damping 1.5`. Main's seven sidecars still read `0259947e`, and `3532e4f` "draft phase 1 tokenizer plan from six pre-work measurements" predates the regeneration. They also predate `runs.jsonl`, which did not exist on main, so **none of them has a provenance row** - the first six numbers in the project with no evidence trail, which is what the log exists to prevent. **Affected:** the k-means-512 floor **26.39 dB**, the 1,024 figure **27.60 dB**, **150 of 512** centroids live, flat receptive fields **19.96%**, Q-2's ceiling **94.4%**, and the **99.86% / 37%** edge-error split. Geometry, palette and camera are unchanged so these will be close, but the arm's pose distribution is not - window coverage alone moved 41.3% -> 60.9% - and patch statistics are exactly what that changes. **Not affected:** 300,000 frames, 16,200 val frames, 8.294 GB at 96x96, the 1.16 / 3.49 GB preload sizes, 4,096 pixels, and the **47,814** squared-error cost of a wrong pixel, which is a property of the palette rather than of the trajectories. **The gate half-survives**: its row 2 recomputes the floor on the same val frames and is self-correcting, but its `+3.6 dB` bar was derived as `30.0 - 26.39`, so the two rows stop describing one requirement the moment the floor moves | `AGENDA.md` - **reconciled 2026-08-28**: main's Phase 1 rewrite taken onto this branch, gate status and hashes corrected, and the affected/unaffected split written in above the gate table. **`docs/phase1_structural_plan.md` is NOT corrected and still states all six as fact** - it exists only on `main` and is not in this worktree, so it cannot be edited here. **Correct it at merge, or re-measure first.** |
| **The six pre-work numbers, re-measured - and the k-means floor refuted** | **run 2026-08-28** - `bench/patch_probe.py` at `data_hash 18a76531`, 179,200 patches from 2,800 frames evenly across all 7 shards, Lloyd 25 iterations, seed 0, both seedings | **The regeneration was never the problem; the unrecorded initialisation was.** Random seeding reproduces the superseded figures to within 0.2 dB - 25.47 / 26.59 / 27.65 dB at 240 / 512 / 1024 against the recorded 25.67 / 26.39 / 27.60, and 172 live of 512 against 150. **k-means++ seeding beats them by 2.4-2.9 dB: the k-means-512 floor is 29.02 dB, not 26.39**, 1,024 reaches **30.51 dB** which clears Q-1's bar outright, and **all 512 centroids stay live**. So gate row 2's bar is **+0.98 dB**, not +3.6, and the conv context has to buy about a quarter of what the plan was written around. Every init-free statistic reproduced, which is what pins the cause: the top-512 exact-patch dictionary **26.05 dB** against 26.02, top-2,048 **29.32** against 29.31, exact-patch entropy **4.40 bits** against 4.43, flat 8x8 patches **63.47%** against 63.22%, flat receptive fields **20.28%** against 19.96% with **still zero void**, the Q-2 ceiling **94.3%** against 94.4%, and the edge-error split **99.95% of error in 36.53% non-flat patches** against 99.86% / 37%. **The two conclusions that survive and the two that do not.** Surviving: the 96x96 fork's evidence is untouched, and Q-2's ceiling still permits the 70% bar. Dead: "a patch-independent tokenizer cannot pass Q-1" is now true only at 512 codes and by 0.98 dB, and "the scene does not need 512 codes" was an artifact - the Q-2 shrink ladder keeps its mechanism but loses this evidence for taking it. **29.02 dB is a lower bound on the patch-independent optimum, not the optimum**: 25 Lloyd iterations may not have converged, and a better seeding would only raise it further | `AGENDA.md`, the pre-regeneration caveat box, the gate table row 2, the R1 ladder row and both risk sections - **corrected**; `docs/phase1_structural_plan.md`, "What the pre-work measured" and the two prose sites - **corrected**; `docs/timeline.md`, the reconstruction-quality and vocabulary-collapse risk rows - **corrected**; `docs/decision_notes.md`, "H3"'s measured-facts paragraph and "I2. If the vocabulary collapses, shrink it" - **corrected**; this doc's "Phase 1 pre-work" subsection - **corrected**, its rows now carry the superseded figures beside the current ones |

### Phase 1 pre-work, measured 2026-08-28 before any of Phase 1 was written

**Three of these rows were re-run later the same day and two of them moved, one a lot** - see "The six pre-work numbers, re-measured" in the log above. The k-means figures here were produced by a run whose initialisation nobody recorded, and that choice turned out to be worth 2.6 dB. They are kept as written, with the current figure marked inline, because the superseded number is what the rest of the tree was planned around and a reader who greps for `26.39` has to land somewhere that says so.

| Claim | Method | Result |
|---|---|---|
| **A patch-independent tokenizer can pass Q-1** | **refuted by measurement** - Lloyd's k-means, 25 iterations on GPU, 179,200 real 8x8 patches from 2,800 frames sampled evenly across all 7 shards, PSNR from summed squared error over all pixels and channels | **26.39 dB at 512 centroids**, 25.67 at 240, **27.60 at 1024** - all short of Q-1's 30 dB, so vocabulary is not the lever. **SUPERSEDED 2026-08-28: those are random-init numbers. Seeded with k-means++ the same patches give 27.53 / 29.02 / 30.51 dB at 240 / 512 / 1024, so 1,024 centroids clear the bar outright and the floor at 512 is 29.02.** Only **150 of 512** centroids stay live, which is the Q-2 risk stated directly.
**SUPERSEDED: all 512 stay live under k-means++, so this evidence for the Q-2 risk is
gone.** A frequency-ranked dictionary of the top-512 exact patches does slightly worse, 26.02 dB, and needs **2,048** entries to reach 29.31. **SUPERSEDED: the gap is 0.98 dB, not 3.6.** **The gap is what the 22x22 receptive field, the attention layer and a shared decoder have to buy** - the phase's whole task, and the reason R2 exists |
| **Q-1's failure mode will be edge placement, not colour drift** | **measured** - per-patch squared error from the k-means run, split by whether the patch is a single flat colour | **99.86% of all error sits in the 37% of patches that are not flat** (**re-measured 2026-08-28: 99.95% in 36.53%, confirmed**); flat patches carry 0.14%. At 240 centroids 99.76%, at 1024 99.93%. The fork table's "colour drift" branch is close to dead before training starts, which is what moved the 96x96 arm from fallback into Phase 1's ladder |
| **Q-2's 70% bar is reachable at all** | **measured, as a provable ceiling** - a code is a deterministic function of its cell's 22x22 receptive field, so cells whose receptive fields hold identical pixels must share a code. Counted flat receptive fields over the 36 interior cells of 64, on 3,500 frames across all 7 shards, then maximised entropy subject to that collapsed mass | **19.96%** of interior cells have a fully flat receptive field (**re-measured 20.28%, confirmed**), all of them table - **no black-void cell collapses**, because the void band sits outside every interior cell's field. Ceiling is **8.494 bits = 94.4% of uniform** (**re-measured 94.3%, confirmed**) at 512 codes, 95.1% at 240. **So the data does not force Q-2 to fail.** Context: 63.22% of plain 8x8 patches are one flat colour, and the exact-patch distribution carries only **4.43 bits** against the 9 available, so nothing forces Q-2 to *pass* either - it is a property of the trained model, not of the dataset |
| **The palette-index preload is lossless** | **measured over the whole set**, not a sample - union of packed RGB triples across all 300,000 frames, then nearest-palette by argmin | Union is **exactly 7** byte triples, one per palette entry, worst distance **0.75**: `(0,0,0)` void, `(76,82,97)` table, `(217,25,25)` `(38,191,51)` `(140,64,217)` blocks, `(229,191,25)` `(25,166,229)` links. **The existing F-9 entry's "at most 7 per frame" does not imply this** - it permits a larger union - so the check had to be redone as a union. One byte per pixel is therefore lossless, and the train split preloads in **1.16 GB instead of 3.49 GB**. Scan took 21.3 s at 14,059 frames/s |
| **The ctx=0 loader is fast enough to feed training** | **refuted for the cold case** - `WindowSampler(ctx=0)`, 20,000 random frames read twice from a fresh process, then the same via a vectorised memmap fancy-index | **6,804 frames/s cold vs 109,682 warm, a 16x swing**, against a ~13,000 frames/s need at batch 128. Vectorised reads give 24,207 cold / 207,104 warm, so **it is I/O bound, not Python bound** - grouping a batch by shard is not the fix. With 4.9 GB free against a 3.5 GB working set the page cache sits on the boundary, which is why `preload` exists rather than a DataLoader worker |
| **The FSQ bound/quantize/index math, as written into the plan** | **measured** - 40,001 inputs over `[-25, 25]` per dimension, plus full enumeration of the digit grid, plus one backward pass | `[8,8,8]` `[8,6,5]` `[5,5,5]` `[4,4,4]` each yield **exactly `prod(levels)` distinct values per dimension** and `codes_to_indices` is a bijection onto `0..prod(levels)-1`. **Correction to an assumption: the straight-through gradient is not 1.0.** It reads **0.858 / 1.001 / 0.668** for `[8,8,8]` / `[5,5,5]` / `[4,4,4]` - the STE bypasses only the rounding, so `tanh`'s derivative inside `bound` stays in the path and scales with `levels`. Confirmed analytically: `(1 - (offset/half_l)^2) * half_l / (levels//2)`. **Walking the Q-2 shrink ladder therefore rescales the bottleneck LR by up to 1.5x**, so every levels comparison needs a paired LR run |
| **`offwidth`/`offheight` can move from the XML to config** | **verified by reading the pinned header** - `mjmodel.h` from the MuJoCo 3.12.0 archive CMake fetches | `struct mjVisual_`'s `global` sub-struct declares `int offwidth` and `int offheight` as plain mutable fields, read by `mjr_makeContext`. Setting them between `mj_loadXML` and context creation leaves `scene/arm_blocks.xml` byte-identical, so **the existing 300,000 frames keep their `data_hash`** and the 96x96 arm costs one config file. **Not yet run** - the byte-identical regeneration is Phase 1 item 1 and is the actual proof |
| **Q-1's bars, restated in pixels** | **computed** from the palette: mean squared distance between two distinct entries | **47,814** (min 15,593, max 97,863). So 30 dB is about **17 of 4,096 pixels completely wrong** (0.41%), 35 dB is 5.3, and the k-means floor is 38.4. Phase 1's task is to halve the floor's error count. Also the reason per-pixel cross-entropy loses to MSE: one misclassified pixel spends 1/17th of the entire 30 dB budget, so classification needs ~99.6% pixel accuracy where regression can hedge |

Sources: [ASan paper](https://research.google.com/pubs/archive/37752.pdf),
[Debloating ASan (USENIX Sec '22)](https://www.usenix.org/system/files/sec22summer_zhang-yuchen.pdf),
[Reducing Redundant Sanitizer Checks (OSDI '21)](https://www.usenix.org/system/files/osdi21-zhang.pdf),
[MuJoCo discussion #2222](https://github.com/google-deepmind/mujoco/discussions/2222),
[MuJoCo mjtRndFlag reference](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).
