# mirage

Generate a MuJoCo robot-arm dataset (300k frames at 64x64), tokenize it, train a
world model on it. `AGENDA.md` is the ordered list of what to do next.

## Authoritative files - read these, do not duplicate them

- `AGENDA.md` - what to do next, in order. Items get deleted as they land.
- `docs/world_model_architecture.md` - design decisions, and the explicit trigger
  that would change each one.
- `docs/world_model_requirements.md` - the `P-` / `F-` / `E-` / `Q-` requirement
  IDs referenced everywhere else.
- `docs/canonical_numbers.md` - **the current value of every number quoted in more
  than one place**, each with the `runs.jsonl` row that asserts it. `runs.jsonl`
  stays the evidence and keeps superseded values on purpose; this is the
  current-state view over it, which is the one question an append-only log cannot
  answer. **Cite the `NUM-` id, do not copy the value.** `python check.py` fails
  on a citation to an id that does not exist, which is what makes renaming safe.

Derived explainers restate those three in plain words and are never a citable
source - when one disagrees with the code, the code wins: `decision_notes.md`
(each choice with its trigger and fallbacks), `timeline.md` (schedule and gates),
`mathematics_notes.md` (every formula the project runs, with the measured number
next to it and a provenance table).

## Every requirement claim carries its evidence

A doc line naming a specific flag, attribute, or API as a **requirement** must carry
a measurement or the command that verified it. With no evidence, write it as
unverified rather than as a requirement. Record verifications in the verification log
at the end of `docs/world_model_architecture.md` - that table is the mechanism, use
it instead of inventing another.

Why: every wrong claim found so far was one nobody had executed - `castshadow="false"`
(not a geom attribute at all, MuJoCo rejects it), `egl_context` (never built),
`-fsanitize=address,undefined` (MSVC has no UBSan), `g++ 13+` (wrong compiler
entirely). Every claim with a number next to it held up. A platform change
invalidates spellings and capabilities wholesale while leaving the reasoning intact,
so after one, re-check the flags and expect the stale copies to outnumber the
decision that caused them.

## Environment facts (verified 2026-08-21)

**Everything runs on Windows.** Python 3.14.2, torch 2.9.1+cu130 with working
CUDA, numpy 2.4.4, pyopengl 3.1.10 already installed; the `mujoco` wheel resolves
cleanly for this Python.

- **WSL2 cannot render on this machine, and this is settled.** `dxgkrnl` logs
  `dxgkio_query_adapter_info: Ioctl failed: -22` at every boot, no `/dev/dri`
  node is ever created, so Mesa finds no DRM device and falls back to `llvmpipe`
  (CPU, ~50x too slow). Verified by `bench/egl_probe.py` across two restarts,
  `wsl --update --pre-release`, and every platform/driver env override; Mesa's
  own debug log says `Falling back to surfaceless swrast without DRM`. CUDA in
  WSL still works, because `libcuda` uses a different ioctl path - but there is
  no reason to use it now. Do not re-litigate this; the probe is the evidence.
- **Windows MuJoCo has no EGL.** The backends are GLFW (default) and OSMesa
  (software - never). The ~30 ms `mjr_readPixels` figure from MuJoCo discussion
  #2222 **does not reproduce**: measured 25.4 us RGB, 75.8 us with render, at 64x64
  under GLFW offscreen. Two-pass render confirmed with a 13x margin, so no WGL
  pbuffer and no single-pass collapse. Probe: `bench/readback_probe.py`.
- Hardware check on Windows: `GL_RENDERER` must name the GPU. Reject
  `GDI Generic` and `Microsoft Basic Render Driver` - those are the software
  fallbacks. Read it once, right after MuJoCo creates its context.
- The GPU is an RTX 5060 **Laptop** (8 GB), 384 GB/s peak, 308.3 GB/s measured
  streaming read. **Do not gate benchmarks on `pstate == P0`** - that rule was
  refuted 2026-08-23. The reported pstate follows the *memory* clock domain, so
  a correct compute-bound run reads P4 while the SMs are at 86% of max drawing
  99 W of a 100 W cap. No single load clocks both domains. Gate compute numbers
  on **SM clock + power draw**, bandwidth numbers on **memory clock == max**, and
  record those next to the number. `bench/gpu_probe.py` does both.
- Thermal state dominates everything: a chassis cooling fix moved the enforced
  power limit 55 -> 100 W and fp16 matmul 3.0 -> 27.6 TFLOP/s. The instantaneous
  throttle flags read `Not Active` the whole time it was capped - the evidence was
  in the `nvidia-smi -q -d PERFORMANCE` **counters**. Sample those, not the flags.
