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

## Environment

| Item | Requirement | Status on this machine |
|---|---|---|
| GPU | RTX 5060, 8 GB, sm_120 | confirmed, capability (12,0) - **laptop variant** |
| OS | Linux or WSL2, not native Windows | WSL2 Ubuntu-24.04 |
| CUDA | 12.8+ (Blackwell) | 13.0 |
| PyTorch | cu128+ | 2.9.1+cu130 |
| GL backend | **EGL**. Not OSMesa (software), not GLFW (~30x slower offscreen) | unverified |
| MuJoCo | 3.x, C API | unverified |
| Compiler | g++ 13+, C++20 | unverified |

**Before taking any timing measurement**, confirm the GPU is at P0. It idles at P4
under ~6 W, where every number is meaningless and E-4's 5% reproducibility bar is
unreachable:

```bash
nvidia-smi --query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw --format=csv
```

## Build

Not yet implemented. E-2 requires a verified clean-build-from-scratch recipe here.
