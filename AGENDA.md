# Mirage: Agenda

Design lives in `world_model_architecture.md`; requirements in
`world_model_requirements.md`. This file is only the ordered list of what to do
next. Keep it short - delete items as they land, do not accumulate history.

Phase 0 is budgeted at 5 days. Its gate: 300k frames on disk, deterministic
replay, EGL verified.

---

## Blockers - nothing measured downstream is trustworthy until these pass

**1. Get the GPU to P0.** Currently sitting at P4, 6.16 W, SM clock 1102 of
3090 MHz. Every timing taken in this state is meaningless, and E-4 ("rerun matches
within 5%") is unachievable while the clock drifts. Mains power, NVIDIA/Windows
performance profile, close the ~50 desktop GPU consumers, then confirm:

```bash
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu --format=csv
```

**2. Windows render backend.** Rendering moved off WSL2 - its GPU graphics path is
broken on this machine and no env override reaches hardware (evidence in `CLAUDE.md`,
produced by `bench/egl_probe.py`). Install `mujoco` on Windows, create a context,
assert `GL_RENDERER` names the RTX 5060. `GDI Generic` and `Microsoft Basic Render
Driver` are the software fallbacks - ~50x slower, and they fail P-6 silently: no
error, just a number that never reaches 500 fps.

**3. C++ toolchain on Windows.** MSVC or MinGW, unverified. Phase 0's `sim/` files
cannot start without it. Python covers the day-1 measurements, so this does not
block today.

---

## Day 1 measurements - four numbers, each gating a decision

| Measure | Decides |
|---|---|
| Bandwidth + fp16 matmul, at P0 | whether the fork table's compute floors are real. They all derive from an assumed 448 GB/s, currently unverified |
| **Per-call `mjr_readPixels` latency, in isolation** | **one-pass vs two-pass render**, and now also **GLFW vs hand-rolled WGL**. Above ~0.5 ms/call two passes cannot meet P-6; near ~30 ms/call GLFW itself is the problem |
| `mj_step` time alone | remaining P-6 headroom |
| Frames/sec end to end | whether parallel generation is needed at all |

Measure per-call, not end-to-end fps. End-to-end hides which term dominates, and
the known failure mode here is fixed per-call cost rather than bandwidth.

---

## Phase 0 build order

1. `scene/arm_blocks.xml` - flat-render config, **two arm links in different
   colours**, palette under 24 unique RGB, decorations off in `mjvOption`
2. `mirage/config.py` - sectioned JSON, hash tree, `Shapes`
3. `sim/gl_context.*` - WGL context plus `GL_RENDERER` assert. Riskiest file; do it
   early. Hand-rolled WGL pbuffer only if the day-1 readback says GLFW is too slow
4. `sim/policy.*` - per-episode 50/50 random vs scripted reach, episodes >= 200 steps
5. `sim/truth.*` - segmentation pixel counts, contact mask, poses
6. `sim/shard_writer.*` - blobs first, sidecar JSON last (it is the commit marker)
7. `mirage/data.py` - memmap reader, episode-aware sampler
8. `mirage/validator.py` - measurement vector, both modes, threshold sweep

Sanitizer build type exists from file 1, as a build type and not the default.
`-ffast-math` never.

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
