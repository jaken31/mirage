# Robot World Model: Requirements (v3, robotics domain)

Tiers: **M** = must (v1 does not ship without it), **S** = should, **C** = could.

---

## 1. Functional requirements

### Simulation and data

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| F-1 | M | MuJoCo scene: 2-link planar arm, 3 pushable blocks, fixed camera | Scene loads, arm reaches all blocks |
| F-2 | M | Flat-render config enforced: ambient-only light, no shadows, box geoms, no textures, offsamples=0 | Rendered frame has <= 24 unique RGB values |
| F-3 | M | C++ harness renders offscreen via **EGL**, not OSMesa | `eglQueryString` reports a hardware vendor |
| F-4 | M | Deterministic given a seed | Same seed and action sequence give bit-identical frames |
| F-5 | M | Data policy: 50/50 random joint deltas and scripted noisy reach | Action histogram roughly uniform over 9 actions |
| F-6 | M | Arm-block contact events exceed 5% of frames | Contact counter over a full run |
| F-7 | M | Block fully occluded by arm in >= 3% of frames | Occlusion counter via visible-pixel test |
| F-8 | M | Shard writer emits packed frames and actions | Round-trip via numpy memmap matches the C++ buffer byte for byte |
| F-9 | M | Frame validator reports block count, arm pose plausibility, palette adherence | Zero false positives on ground-truth frames |

F-2 and F-7 are new and both load-bearing. F-2 protects the token budget. F-7 is what makes Q-6 measurable.

### Models

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| F-10 | M | FSQ tokenizer encodes a frame to an 8x8 grid over 512 levels and decodes back | Meets Q-1 |
| F-11 | M | Dynamics model consumes interleaved frame and action tokens, predicts next token | Held-out accuracy beats marginal-frequency baseline by 3x |
| F-12 | M | Generates a full next frame from previous frames plus one action, fixed step count | No fallback path |
| F-13 | S | Configurable context length at load time | Rollout runs at 4, 8, 15 frames from one checkpoint |

### Inference and control

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| F-14 | M | Control loop reads keyboard and drives the model with MuJoCo not running | MuJoCo process absent, arm still responds |
| F-15 | M | KV cache, graph capture, INT8 behind independent flags | Each toggles alone, bench reports per configuration |
| F-16 | S | Fused Triton block and diagonal decoding | Same. Promotes to M if the 144-token path is taken |
| F-17 | M | Bench reports p50/p99 frame time and CPU dispatch vs GPU busy split | One command produces the ladder table |

---

## 2. Non-functional requirements

### Performance

| ID | Tier | Metric | Threshold |
|---|---|---|---|
| P-1 | M | Sustained frame rate, 1000-frame run | >= 30 fps |
| P-2 | M | p99 frame time | <= 40 ms |
| P-3 | M | Input-to-display latency | <= 66 ms |
| P-4 | S | p99 / p50 ratio | <= 1.3 |
| P-5 | M | Baseline eager to final engine speedup | >= 3x |
| P-6 | M | Data generation throughput incl. render | >= 500 frames/sec |
| P-7 | S | Full 300k-frame epoch | <= 30 min |

P-6 dropped from 20k/sec. MuJoCo plus offscreen render is far slower than a hand-written 2D sandbox. 500/sec gives 300k frames in about ten minutes, which is fine.

### Resource

| ID | Tier | Metric | Threshold |
|---|---|---|---|
| R-1 | M | Peak training VRAM | <= 7.5 GB |
| R-2 | M | Peak inference VRAM | <= 2 GB |
| R-3 | M | Dynamics model parameters | <= 20M bf16 |
| R-4 | M | Dataset on disk | <= 20 GB |
| R-5 | S | Cold start to first frame | <= 10 s |

### Quality

| ID | Tier | Metric | Threshold |
|---|---|---|---|
| Q-1 | M | Tokenizer reconstruction PSNR, held-out, at 64x64 | >= 30 dB |
| Q-1b | S | Same | >= 35 dB |
| Q-2 | M | Token entropy vs uniform over 512 codes | >= 70% |
| Q-3 | M | Coherence horizon: frames until F-9 validator fails | >= 200 |
| Q-3b | S | Same | >= 500 |
| Q-4 | M | Action-following accuracy via inverse dynamics model | >= 90% |
| Q-5 | M | Arm kinematic plausibility: link lengths stable across a 200-step rollout | drift <= 10% |
| Q-6 | S | **Object permanence**: block reappears in correct position after full occlusion | >= 80% of occlusion events |
| Q-6b | C | Same, with position error <= 2 px | >= 60% |

Q-6 is the memory result, back in scope because occlusion is native to manipulation rather than bolted on. It stays **S**, not M. It is the most likely thing to fail and the project ships without it.

Q-5 replaces velocity-preservation. A model that hallucinates arm geometry is the characteristic failure here.

### Engineering

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| E-1 | M | Deterministic sim given a seed | Same as F-4 |
| E-2 | M | Clean build from scratch, MuJoCo and EGL linked | Documented in README, verified once |
| E-3 | M | ASan and UBSan clean on the full data-generation run | Zero reports |
| E-4 | M | Every bench number reproducible from a config hash | Rerun matches within 5% |
| E-5 | M | Append-only run log: config hash, change, number, conclusion | One entry per run |

---

## 3. Ship criteria

Every **M** row passes and the ladder table is populated end to end.

## 4. Explicit non-requirements

Photorealism. Sim-to-real transfer. Policy learning or planning on top of the model. Generalization to unseen scenes. Multi-arm. Dexterous or grasping manipulation. Stability past 500 rollout steps. Windows-native support.

## 5. Requirements at risk

| ID | Risk | Fallback |
|---|---|---|
| F-2 | Flat config may still leave gradients that hurt Q-1 | Drop to 96x96 / 144 tokens, promote F-16 to M |
| P-1 | 30 fps is tight on the 144-token path | DiagD becomes required. If still short, report the curve and accept 20 fps rather than cutting a ladder rung |
| Q-6 | Object permanence may simply not emerge at 15M params | Stays S. Report the measurement either way, including a negative result |
| R-3 | 20M may be too small for Q-4 | Raise to 25M only if profiling shows headroom. Never above 40M |
| P-6 | EGL misconfigured falls back to software rendering, 50x slower | Verify vendor string in Phase 0, day 1 |
