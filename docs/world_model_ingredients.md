# Robot World Model: Ingredients (v3, robotics domain)

**One line:** train an action-conditioned world model on a MuJoCo manipulation scene, delete the simulator, and control the learned model in real time at 30fps.

**Hardware:** RTX 5060 8GB, sm_120, 448 GB/s peak.
**Timeline:** 12 weeks with 2 weeks slack.
**Scope principle:** ML ambition at the floor, systems ambition intact. The inference engine is the project.

---

## Why robotics

The architecture you already chose is the one this field converged on. Recent work on causal world modeling for robot control unifies video and action in a single autoregressive framework specifically to get persistent memory via KV cache and real-time observation integration. Diffusion video models predict better but are too slow to close a control loop. That gap is where this project sits.

The demo also reads correctly to the audience you care about: a learned manipulation simulator running real-time on consumer hardware is immediately legible to robotics and ML infra people.

## What changed from the game-world version

| Was | Now |
|---|---|
| Self-written C++ 2D sandbox | MuJoCo scene, C++ harness against the C API |
| Avatar + bouncing balls | 2-link planar arm + 3 pushable blocks |
| 5 arrow-key actions | 9 actions: 3 joints-deltas x 3 per joint |
| Memory test cut | **memory test back in** — the arm naturally occludes blocks |
| Flat 2D sprites | 3D scene rendered deliberately flat |
| DiagD held in reserve | **DiagD may be a required rung** (see budget) |

**Unchanged:** batch size 1, 30fps bar, FSQ tokenizer, decoder-only transformer, the full ladder.

## Hardware and environment

| Item | Spec | Note |
|---|---|---|
| GPU | RTX 5060 **Laptop**, 8GB, sm_120 | idles at P4; record the pstate next to every timing |
| OS | Windows 11 | WSL2 cannot render on this machine - no `/dev/dri`, settled |
| CUDA | 13.0 | the version the installed torch wheel carries |
| PyTorch | 2.9.1+cu130, stable | `torch.cuda.is_available()` true on sm_120, verified |
| Triton | verify ptxas accepts sm_120 | **week 1 blocker** |
| MuJoCo | 3.x, C API | link against `mujoco.lib` |
| GL backend | **GLFW** | Windows MuJoCo has no EGL. Never OSMesa, it is software and slow |
| Host compiler | MSVC 14.50, VS 18 2026 / C++20 | `/W4 /WX`; ASan yes, UBSan unavailable |

## The flat-render config (critical)

A default MuJoCo render has shading gradients, shadows, and specular highlights. Those do not compress to 64 tokens. Force the scene flat:

- ambient-only lighting, no directional lights
- `shadow=false`, `reflection=false`
- box geoms only, no capsules or spheres (curved surfaces produce shading ramps)
- no textures, no skybox, solid background plane
- distinct saturated colors per object class
- disable anti-aliasing (`offsamples=0`) so edges stay hard

This is the single highest-leverage config decision in the project. Get it wrong and the tokenizer budget collapses.

## Scene

- Table plane, fixed third-person camera, orthographic-leaning FOV
- 2-link planar arm, revolute joints, box link geoms
- 3 pushable blocks, distinct colors
- Blocks are occluded by the arm at some poses. This is deliberate.

## Action space

9 discrete actions: joint-1 delta and joint-2 delta, each in {-1, 0, +1}. One token per frame.

## Data policy

50/50 mix:
- random joint deltas (broad state coverage)
- scripted noisy reach toward a random block (guarantees contact events)

Pure random flails and rarely touches anything. The mix is what makes contact learnable.

## Components to build

| # | Component | Language |
|---|---|---|
| 1 | MuJoCo scene XML, flat-configured | XML |
| 2 | C++ harness: EGL context, step loop, offscreen render, pixel readback | C++ |
| 3 | Data policy | C++ |
| 4 | Shard writer + frame validator | C++ |
| 5 | FSQ autoencoder | PyTorch |
| 6 | Dynamics transformer | PyTorch |
| 7 | Inference engine: KV cache, graph capture, INT8, fused kernel, DiagD | PyTorch + Triton |
| 8 | Bench harness | Python |

## Model config

```
Render:     64x64 target, 96x96 fallback
Tokenizer:  FSQ, levels [8,8,8] = 512 codes, stride 8
            -> 8x8 = 64 tokens/frame (target)
            -> 12x12 = 144 tokens/frame (fallback)
Dynamics:   d_model 384, 8 layers, 6 heads, plain MHA
            ctx 15 frames: 1024 tokens (target) / 2176 (fallback)
Training:   bf16, AdamW
```

## The budget, both cases

| Term | 64 tok/frame | 144 tok/frame |
|---|---|---|
| Per-step budget @ 30fps | ~520 us | ~231 us |
| Weights (15M bf16) | ~67 us | ~67 us |
| KV cache | ~28 us | ~60 us |
| **Compute floor** | **~95 us** | **~127 us** |
| **Launch overhead (~80 kernels)** | **~400 us** | **~400 us** |

At 64 tokens, overhead is 4x compute and CUDA graphs alone likely close it.

At 144 tokens, **overhead exceeds the entire budget.** Graph capture becomes mandatory rather than the headline win, and DiagD graduates from reserve to required. Note DiagD scales *better* at the larger grid: 23 diagonals for a 12x12 grid versus 144 sequential steps is a 6.3x reduction, against 4.3x on an 8x8 grid.

**Decision rule:** train the tokenizer at 64x64 first. If held-out PSNR clears 30 dB, keep 64 tokens. If not, move to 96x96 and accept the harder budget. Decide with data, not now.

## The Phase 4 ladder

1. KV cache
2. **CUDA graph capture**
3. INT8 weight-only quantization
4. Fused Triton attention + MLP block
5. **Diagonal decoding** (reserve at 64 tokens, required at 144)

## Phases

| Phase | Duration | Gate |
|---|---|---|
| 0 Sim + data | 5 days | 300k frames on disk, deterministic replay, EGL verified |
| 1 Tokenizer | 1 week | reconstruction PSNR >= 30 dB at 64x64 |
| 2 Dynamics | 2 weeks | coherent 200-step rollout, block persists through occlusion |
| 3 Playable | 3 days | keyboard drives the arm, baseline fps + CPU/GPU split captured |
| 4 Engine | 6 weeks | the ladder |
| Slack | 2 weeks | |

**Do not optimize before Phase 3 ships.**

## Prior art

| Work | Why |
|---|---|
| Causal World Modeling for Robot Control (2601.21998) | autoregressive video+action, KV cache, real-time. Your architecture |
| World Model for Robot Learning survey (2605.00080) | positions the whole field |
| MineWorld (2504.08388) | closest open implementation, same family |
| Diagonal Decoding (2503.14070) | the parallel-decode rung |
| FSQ (2309.15505) | tokenizer, levels table |
| gpt-fast (PyTorch repo) | reference for the entire ladder |

## Add-backs if Phase 2 lands early

1. **Third link on the arm** — richer dynamics, same everything else
2. **Block-block collision** — one flag in the XML
3. **Moving camera** — reintroduces the harder memory problem
4. **First-person / wrist camera** — the impressive version, expensive on tokens
