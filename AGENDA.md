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
4. `sim/truth.*` - segmentation pixel counts, contact mask, poses - **done 2026-08-27**.
   `Truth::read` fills one reusable `TruthFrame`; `truth_dry_run` is the check.
   Three fatal checks - some block visible at rest, a block moved out of frame
   reads exactly 0, restoring it returns the same count - plus printed F-6/F-7
   rates. Two things to carry into item 6: **the segmentation pass leaves the
   framebuffer holding id colours, so the RGB readback has to happen before
   `read()`**, and the dry run's rates are biased high in both directions - a
   block knocked off the table reads 0 forever - so they say the measurement
   responds, not that the scene passes
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

`truth_dry_run` prints both already, but it drives actuator 0 at full torque, a
cruder sweep than the policy produces. **Its numbers are evidence, not the
verdict.** Run 2026-08-27: **F-6 60.93%**, 12x the floor, so the playbook below
is unlikely to be needed. F-7 read 79.73% but mostly because block 0 is hidden
at rest, so that one is passing on a technicality. The verdict needs the real policy driving: count `contact_mask` over
the 2,000 x 600 dry run, beside F-5's histogram. Open question it settles -
F-5 compliance cost arrival 44% -> 35.7%, and whether that cost any contact is
reasoned about but unmeasured (`docs/world_model_architecture.md`, "F-5's
threshold is the knee of a measured curve").

### If F-6 misses, in this order

1. **Move the blocks in `scene/arm_blocks.xml`.** They sit at radius 0.20-0.21
   against a 0.33 total reach. Pulling them into the mid-sweep band raises
   contact for the random half as well, and costs F-5 nothing. This is K4 in
   `docs/decision_notes.md` - contact frequency is tuned by moving objects.
2. **Tighten `sim.reach_done_dist` to ~0.025 m**, the block half-size. At 0.04
   the reach re-targets while the fingertip is still 1.5 cm of air away from a
   head-on block face, so the arm turns away before touching. Arrival will read
   lower; it is a diagnostic.
3. **Spend F-5's margin.** Ratio 2.06 against the 2.5 ceiling, min share 7.15%
   against 5%. `reach_digit_noise_prob` 0.15 -> ~0.10 buys reach quality back,
   and the 14-configuration frontier already has that point measured.

**Do not raise `reach_done_dist`.** It improves the printed arrival rate and
produces no extra contact - the one change here that makes a diagnostic lie.

---

## Deferred - do not start

Phases 2 through 4. Phase 4's plan derives from the Phase 3 profile, which does not
exist yet.

Connected-component labelling, parallel generation, `--replay` mode, and the
single-pass render each have an explicit trigger recorded in the architecture doc.
None of them is a judgement call - wait for the trigger.
