# Mirage: one-page defense

## The thesis (say this first)

> Fixed overhead is what makes a small model no faster than a big one.
> This project removes the overhead.

MineWorld: 300M params, **499 us/token**. Mirage: 15M params, **521 us/token**.
Same latency, 20x fewer params. Neither is compute-bound. Launch overhead is
~400 us against a ~135 us compute floor.

## The answer shape

Every answer is three parts. Missing one is a tell.

**decision -> the number -> the trigger that reverses it**

No number = opinion. No trigger = dogma.

## Six numbers to know cold

| Number | Meaning |
|---|---|
| **178.5 us** | step + 2-pass render, real scene = 5,602 fps, **11.2x** over the 500 fps bar |
| **25.4 us** | `mjr_readPixels` RGB @ 64x64. The docs said 30 ms. Off by 1000x |
| **308.3 GB/s** | measured read. Plan assumed 448. Every budget floor rose 45% |
| **27.6 TFLOP/s** | fp16, after a cooling fix. Was 3.0. **9x from cooling alone** |
| **10.5 us** | `mj_step`. 131x under budget. Physics is not the problem |
| **28 -> 6** | colours, after the ambient-only headlight. F-2 ceiling is 24 |

## Four things the plan got wrong, and measurement caught

Strongest material. Lead with a refutation, not a success.

| Claim | What happened |
|---|---|
| 30 ms readback, so hand-roll a WGL pbuffer | **25.4 us.** Cause was `offsamples=4` forcing an MSAA resolve. A config flag, not a week of code |
| 448 GB/s bandwidth | **384 peak / 308 real.** That is the *desktop* 5060. Promoted CUDA graphs to required |
| Gate benchmarks on `pstate == P0` | **Refuted.** pstate follows the *memory* domain; correct compute-bound work reads P4 at 99 W. The rule would have rejected every valid number |
| `castshadow="false"` on geoms | Not a geom attribute. MuJoCo rejects it. The real fix was the headlight |

Pattern: **every claim with a number next to it held up. Every claim without one
was eventually wrong.**

## Decisions, one line each

**Scene**
- Occlusion is native to manipulation, so object permanence is measurable, not contrived.
- Flat render: shading gradients do not compress to 64 tokens. Highest-leverage config in the project.
- Two arm links get different colours, so Q-4/Q-5 are pixel counts, not segmentation.
- The palette lives only in the XML. Two copies drift; the symptom is "block missing" for a present block.

**Data**
- SoA for **simplicity, not speed** - the speed claim was measured and retracted (197 KB, noise).
- Store `visible_px` counts, not an `is_occluded` bool. A bool bakes in a threshold; counts re-derive.
- `episode_id` + `step_idx` mandatory: a window straddling a reset produces plausible-looking loss curves.
- Sidecar JSON written last, so its existence is the commit marker. No lockfiles.

**Provenance**
- Hash tree, so **no invalidation code exists.** A stale artifact is unreachable because nothing computes its name.
- Token cache keyed by checkpoint *bytes*, not config: same config + different seed = different tokens.
- Limit, concede it early: **the hash covers config, not code.** Git SHA inside the artifact; detection stays manual.

**Validator**
- Emits measurements, never verdicts. Q-3 recomputes under any threshold without re-running rollouts.
- Colour counting over connected components: an arm crossing a block splits it into two blobs, and CC calls that two blocks.
- PCA oriented bbox: a 45-degree-rotated square in an axis-aligned box scores ~0.5 and collides with the occluded case.
- Q-4 needs no inverse dynamics model. A geometric measurement has no generalization gap.

**Tokenizer**
- Compression ratio is **identical on both paths**, exactly 1536/9. 96x96 buys oversampling, not compression.
- So the fallback only helps if Q-1 fails on **edge placement**. Colour drift means the FSQ levels table instead.

**Engine**
- Seam: model is `(tokens, cache, mask) -> logits`; engine owns loop, cache, capture.
- That one line assigns every rung: cache/graphs/DiagD are engine, INT8/fused block are model.
- Five flags become two submodule swaps + three engine behaviours, not 32 code paths.
- Ladder order follows the budget: graphs first (400 us), INT8 third (94 us). Attack the big term.
- Explicit mask and a static KV cache from commit 1 - both are expensive to retrofit.

**Discipline**
- Nothing logs inside the timed region. One 1 ms hitch is 2.5% of the frame budget and lands in p99.
- Read peak VRAM from the allocator high-water mark. A sampler can miss a transient peak.
- Sample thermal **counters**, not throttle flags. The flags read `Not Active` through a 45 W cap.

## What was deliberately NOT built

Parallel generation (11.2x margin), WGL pbuffer (trigger never fired), single-pass
render (reserve, 10-min switch), connected components (deferred), pybind11, replay
mode, scipy, version integers, an IDM for Q-4.

> Fallbacks are not built in advance. Each one has a trigger written next to it.
> Building fallbacks speculatively is how a five-day stage becomes a three-week one.

## Weak spots - concede these fast

| Soft spot | The line |
|---|---|
| Nothing past Phase 0 is measured | "Phase 0 is verified end to end. Everything later is a budget, not a result." |
| UBSan has no implementation on MSVC | Settled: E-3 is ASan-only. The one class that mattered - signed overflow across 3.6 GB of shard offsets - is 64-bit offsets plus a write-site assert, not a second toolchain |
| E-4's 5% not demonstrated for `mj_step` | Series had not plateaued after 6 runs. Needs a quiescent-machine protocol |
| Object permanence may not emerge at 15M | Tiered **S**, not M, on purpose. Ships without it, reports the negative result |
| C++ toolchain unverified | Closed: MSVC via `Visual Studio 18 2026`, C++20 confirmed by the binary printing `202002`, default and ASan builds both compile and run |
| 26 us render is one sphere, a floor | Re-run against the real scene. Readback is per-pixel, so 25 us holds |

## If you only remember one thing

The verification log at the end of `docs/world_model_architecture.md` is a table of
claims that were checked **and what came back wrong**. That table, not any
methodology answer, is the response to "how do you make technical decisions."
