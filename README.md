# mirage

Train an action-conditioned world model on a MuJoCo manipulation scene, delete the
simulator, and control the learned model in real time at 30 fps.

## Docs

| File | What |
|---|---|
| [AGENDA.md](AGENDA.md) | What to do next. Start here. |
| [docs/world_model_architecture.md](docs/world_model_architecture.md) | How the pieces meet: interfaces, shard format, provenance, validator |
| [docs/world_model_requirements.md](docs/world_model_requirements.md) | Tiered requirements and acceptance tests |
| [docs/world_model_ingredients.md](docs/world_model_ingredients.md) | Scope, model config, latency budget, phases |
| [docs/world_model_learning_roadmap.md](docs/world_model_learning_roadmap.md) | Study plan, sequenced against the phases |
| [docs/timeline.md](docs/timeline.md) | **Plain English.** Schedule, gates, branch points, risks, study woven in. For non-engineering readers |
| [docs/decision_notes.md](docs/decision_notes.md) | **Plain English.** Every decision with its reason, its wrong-if signal, and 1-2 fallbacks |
| [docs/phase0_report.md](docs/phase0_report.md) | **Derived.** Phase 0 completion report: output, architecture, measured results, open items |

The last three are derived from the architecture and requirements docs. When a
decision changes, change it there first.

## Environment

| Item | Requirement | Status on this machine |
|---|---|---|
| GPU | RTX 5060, 8 GB, sm_120 | confirmed, capability (12,0) - **laptop variant** |
| OS | **Native Windows** | Windows 11, Python 3.14.2 |
| CUDA | 12.8+ (Blackwell) | 13.0 |
| PyTorch | cu128+ | 2.9.1+cu130 |
| GL backend | **GLFW**, offscreen. EGL is unavailable on Windows MuJoCo; OSMesa is a CPU rasterizer and an anti-choice | unverified - `mjr_readPixels` per-call latency is the day-1 blocker |
| MuJoCo | 3.x, C API | wheel resolves for this Python; not yet exercised |
| Compiler | C++20 | unverified - MSVC or MinGW, remaining Phase 0 prerequisite |

**WSL2 is out, and this is settled.** Its GPU graphics path is broken on this
machine - no `/dev/dri` node, Mesa falls back to a CPU rasterizer. Evidence in
`CLAUDE.md`, produced by `bench/egl_probe.py`. Do not re-litigate it.

**Before taking any timing measurement**, confirm the GPU is at P0. It idles at P4
under ~6 W, where every number is meaningless and E-4's 5% reproducibility bar is
unreachable:

```bash
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw --format=csv
```

## Build

Not yet implemented. E-2 requires a verified clean-build-from-scratch recipe here.
