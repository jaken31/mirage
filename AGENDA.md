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

## Blocker - nothing measured downstream is trustworthy until this passes

**Confirm the GPU reaches P0 under sustained load.** Not at idle - idle P4 is
correct behaviour for a laptop GPU and is not the failure. The readback probe
already sampled P2 while rendering, so the card does clock up; what is unverified
is whether a sustained compute load pins it at P0 and holds there. Until that is
shown, the bandwidth and fp16 matmul floors are unusable, because both are
clock-linear, and E-4 ("rerun matches within 5%") cannot be claimed while the
clock is free to drift mid-run.

The check: run a fp16 matmul in a loop long enough to heat the card - 30 s or
more - and sample in a second shell *while it runs*:

```bash
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu --format=csv -l 1
```

Pass means pstate reads P0 for the whole window and the SM clock stays flat. A
clock that starts high and decays is thermal or power throttling, not a pstate
problem, and needs mains power plus the Windows/NVIDIA performance profile before
retrying. Record the pstate next to every number this project ever reports.

---

## Day 1 measurements - four numbers, each gating a decision

| Measure | Decides | State |
|---|---|---|
| Bandwidth + fp16 matmul, at P0 | whether the fork table's compute floors are real. They all derive from an assumed 448 GB/s, currently unverified | blocked on P0 |
| Per-call `mjr_readPixels` latency, in isolation | one-pass vs two-pass render, and GLFW vs hand-rolled WGL | **done** - 25.4 us RGB, 49.6 us RGB+depth, 75.8 us with render, at P2. **Two-pass render confirmed, 13x margin.** Neither the single-pass collapse nor the WGL pbuffer is needed |
| `mj_step` time alone | remaining P-6 headroom | **next.** Render+readback leaves ~1850 of the 2000 us frame, so this is the only day-1 number that can still break P-6 |
| Frames/sec end to end | whether parallel generation is needed at all | after `mj_step` |

Measure per-call, not end-to-end fps. End-to-end hides which term dominates.

The readback result carried two side effects, both recorded in the architecture doc:
the ~30 ms figure from MuJoCo discussion #2222 does not reproduce here, and the fixed
cost that did exist was MuJoCo's own `shadowsize=4096` / `offsamples=4` defaults - so
`<quality offsamples="0" shadowsize="0"/>` is now a hard requirement on file 1
below, not cosmetic. (`castshadow` is a `<light>` attribute, not a geom one - MuJoCo
rejects it on a geom. With no `<light>` elements and an ambient-only headlight there
is nothing to disable.)

---

## Phase 0 build order

1. `scene/arm_blocks.xml` - flat-render config, **two arm links in different
   colours**, palette under 24 unique RGB, decorations off in `mjvOption`.
   `<quality offsamples="0" shadowsize="0"/>` plus an ambient-only headlight
   (`diffuse="0 0 0" specular="0 0 0"`) are required here - the quality settings are
   measured at 12x on `mjr_render`, and the headlight is what collapses the palette.
   `offsamples="0"` alone is not enough: a diffuse-lit box shades per face, so one
   `rgba` becomes three palette entries. No materials needed - plain `rgba` under an
   ambient-only headlight measured 6 colours
2. `mirage/config.py` - sectioned JSON, hash tree, `Shapes`
3. `sim/gl_context.*` - GLFW context plus `GL_RENDERER` assert. **De-risked** - the
   day-1 readback cleared GLFW, so this is a plain port of what
   `bench/readback_probe.py` already does. No pbuffer
4. `sim/policy.*` - per-episode 50/50 random vs scripted reach, episodes >= 200 steps
5. `sim/truth.*` - segmentation pixel counts, contact mask, poses
6. `sim/shard_writer.*` - blobs first, sidecar JSON last (it is the commit marker)
7. `mirage/data.py` - memmap reader, episode-aware sampler
8. `mirage/validator.py` - measurement vector, both modes, threshold sweep

Toolchain verified: MSVC via CMake generator `Visual Studio 18 2026`, C++20 confirmed
by `sim/main.cpp` printing `202002`, and both `sim/build/` and `sim/build-asan/`
build and run. Sanitizer build type exists from file 1, as a build type and not the
default. `/fp:fast` never.

**Structural plan for these eight: `phase0_structural_plan.md`.** What each file
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
