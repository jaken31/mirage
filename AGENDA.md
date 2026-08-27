# Mirage: Agenda

Design lives in `world_model_architecture.md`; requirements in
`world_model_requirements.md`. This file is only the ordered list of what to do
next. Keep it short - delete items as they land, do not accumulate history.

Plain-English versions for non-engineering readers: `timeline.md` (schedule,
gates, risks, with the study plan woven in) and `decision_notes.md` (every
decision with its trigger and fallbacks). Both are derived from the two docs
above - when a decision changes, change it there first.

Phase 0 is budgeted at 5 days. Its gate: 300k frames on disk, deterministic
replay, hardware render confirmed by the renderer name.

---

## Day 1 measurements - four numbers, each gating a decision

| Measure | Decides | State |
|---|---|---|
| Bandwidth + fp16 matmul, clocked up | whether the fork table's compute floors are real | **done** - 27.6 TFLOP/s fp16, and **308.3 GB/s streaming read**. **The assumed 448 GB/s was wrong**: this part peaks at 384 (12001 MHz x 2 x 16 B), so every compute floor moved. Fork table recalculated |
| Per-call `mjr_readPixels` latency, in isolation | one-pass vs two-pass render, and GLFW vs hand-rolled WGL | **done** - 25.4 us RGB, 49.6 us RGB+depth, 75.8 us with render, at P2. **Two-pass render confirmed, 13x margin.** Neither the single-pass collapse nor the WGL pbuffer is needed |
| `mj_step` time alone | remaining P-6 headroom | **done** - 10.5-10.8 us median driven, 131-176x under the ~1850 us allowance. P-6 is not at risk from physics |
| Frames/sec end to end | whether parallel generation is needed at all | **done** - `mjv_updateScene` is 1.1 us; the full frame (step + 2-pass render + readback) is **178.5 us, 5,602 fps, 11.2x over P-6**. `bench/frame_probe.py`. **Parallel generation is not needed** - the trigger does not fire |

Measure per-call, not end-to-end fps. End-to-end hides which term dominates.

---

## Phase 0 build order

1. `mirage/config.py` - sectioned JSON, hash tree, `Shapes` - **done**
2. `sim/gl_context.*` - GLFW context plus `GL_RENDERER` assert - **done 2026-08-26**.
   Prints `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2` at 64x64; the renderer
   deny-list, the `currentBuffer` check, the viewport-vs-XML check and
   `mjr_getError` are all fatal at startup. No pbuffer was needed
3. `sim/policy.*` - per-episode 50/50 random vs scripted reach - **done 2026-08-27**.
   Two seeded streams (`seed_seq`, not `base + i`), `begin_episode`/`step`, and a
   transpose-Jacobian reach. `policy_dry_run` is the check: F-4 aborts on
   divergence, F-5 prints a verdict. Verified at F-5's own 2,000-episode sample
   size - min share 7.15%, ratio 2.06, both inside threshold. Two config changes
   came out of it: **episodes are 600 steps x 500, not 200 x 1500** (at 200 the arm
   cannot cross to a block inside 0.4 s of sim time), and the histogram knobs are
   `jacobian_deadband` + `reach_digit_noise_prob`, not whole-action noise
4. `sim/truth.*` - segmentation pixel counts, contact mask, poses
5. `sim/shard_writer.*` - blobs first, sidecar JSON last (it is the commit marker)
6. `mirage/data.py` - memmap reader, episode-aware sampler
7. `mirage/validator.py` - measurement vector, both modes, threshold sweep

Toolchain verified: MSVC via CMake generator `Visual Studio 18 2026`, C++20 confirmed
by `sim/main.cpp` printing `202002`, and both `sim/build/` and `sim/build-asan/`
build and run. The sanitizer build type already exists, as a build type and not
the default. `/fp:fast` never.

**Structural plan for these seven: `phase0_structural_plan.md`.** What each file
owns, the calls it needs and the doc page for them, what "working" looks like,
and the gotchas.

**Standing practice: draft the structural plan for a phase when you reach it, not
before.** One file per phase, same shape as the Phase 0 one - ownership table,
build order with named APIs, done-when per file, gotchas. Phase 1's is the next
one to write, and it can be drafted now because only the shard format gates it.
Phases 2 and 4 cannot: Phase 2's numbers wait on the tokenizer PSNR, and Phase 4's
whole plan derives from the Phase 3 profile. Drafting those early is guessing.

---

## Budget extra iterations for these two

F-6 (arm-block contact > 5% of frames) and F-7 (full occlusion >= 3%) are the
checks most likely to fail on the first scene. Both are fixed by editing the XML -
arm reach versus block placement - not by changing code.

---

## Deferred - do not start

Phases 2 through 4. Phase 4's plan derives from the Phase 3 profile, which does not
exist yet.

Connected-component labelling, parallel generation, `--replay` mode, and the
single-pass render each have an explicit trigger recorded in the architecture doc.
None of them is a judgement call - wait for the trigger.
