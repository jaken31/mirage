# Robot World Model: Requirements (v3, robotics domain)

Tiers: **M** = must (v1 does not ship without it), **S** = should, **C** = could.

---

## 1. Functional requirements

### Simulation and data

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| F-1 | M | MuJoCo scene: 2-link planar arm, 3 pushable blocks, fixed camera | Scene loads, arm reaches all blocks |
| F-2 | M | Flat-render config enforced: ambient-only light, no shadows, box geoms, no textures, offsamples=0 | Rendered frame has <= 24 unique RGB values |
| F-3 | M | C++ harness renders offscreen on **GPU hardware**, never a software rasterizer | `glGetString(GL_RENDERER)` names neither `GDI Generic` nor `Microsoft Basic Render Driver`. Deny-list, not an allow-list on "RTX 5060", which would fail on any other machine that is perfectly fine. Asserted at context creation |
| F-4 | M | Deterministic given a seed | Same seed and action sequence give bit-identical frames |
| F-5 | M | Data policy: 50/50 random joint deltas and scripted noisy reach | Over >= 2,000 episodes at the configured length: every one of the 9 actions holds **>= 5% of frames**, and **max bin / min bin <= 2.5**. Both reported by `policy_dry_run` |
| F-6 | M | Arm-block contact events exceed 5% of frames | Contact counter over a full run |
| F-7 | M | Block fully occluded in >= 3% of frames, **counting only occlusion the block recovers from** | Visible-pixel counter, excluding any run where the block never becomes visible again in that episode. `validator.recoverable_occlusion_rate_min` |
| F-8 | M | Shard writer emits packed frames and actions | Round-trip via numpy memmap matches the C++ buffer byte for byte |
| F-9 | M | Frame validator reports block count, arm pose plausibility, palette adherence | Zero false positives on ground-truth frames |

F-2 and F-7 are new and both load-bearing. F-2 protects the token budget. F-7 is what makes Q-6 measurable.

**F-7 was restated 2026-08-28**, after `bench/occlusion_probe.py` measured what its
counter was actually counting. It had been "any frame where a block reads zero
pixels", and **73% of that was blocks that never came back** - 14.48 points of
19.83. A block that is gone for the rest of the episode is not occluded, and
scoring Q-6's object permanence on an occlusion event that can never end asks a
model to recover something that never reappears. The restated requirement counts
recoverable occlusion only: **5.35%**, which still clears the 3% floor at 1.8x
rather than the 6.6x the old number implied. The floor itself is unchanged - this
changed what is counted, not the bar.

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
| Q-3 | M | **Coherence horizon**: frames until the rollout's **frame-to-frame continuity** verdict fires | **>= 200.** The verdict bounds per-step change in the pose features already on the validator's vector - `link_angle`, `link_extent`, and each block's `bbox` centroid - calibrated so that **zero windows of ground-truth frames fire**, the same acceptance shape F-9 uses. **Was "frames until the F-9 validator fails" until 2026-08-30, and that verdict was blind**: `runs.jsonl` r48 offered a rollout the reconstruction of a frame 300 steps out of position and F-9's palette verdict fired on **0.00%** of them, against 0.00% on correct reconstructions and 100% on its sigma-16 noise control. The blindness is **structural, not a threshold** - a drifted frame is a *plausible* frame, and the verdict never references frame `t`, so no value of `offpalette_tau` or `offpalette_frac_max` recovers a comparison the statistic does not make. `bench/q3_blind_probe.py` is the acceptance test for the replacement: it must fire on 100% of 300-step substitutions and 0% of clean reconstructions |
| Q-3b | S | Same | >= 500 |
| Q-4 | M | Action-following accuracy, **scored relative to the simulator's own score** | **>= 90% of the ground-truth agreement measured the same way on the same subset**, and **both numbers reported**. Agreement is `sign(theta_t+1 - theta_t)` against the commanded sign; the subset is **action-balanced** - equal frames per action, drawn from the val split, not the raw split, which F-5's 5% floor is what makes drawable. **Was an absolute 90% until 2026-08-28, which is above what the simulator itself scores**: ground truth reads 83.1% at `action_hold_steps = 20`, because for about one joint settling time after each sign flip the joint is still moving the old way. An absolute bar there fails a model that is exactly right. The ceiling is a property of the physics, so the bar has to be too. `bench/hold_probe.py` measures the ground-truth term |
| Q-5 | M | Arm kinematic plausibility: link-length drift across a 200-step rollout, **scored relative to the simulator's own drift** | **<= 1.1x the ground-truth drift, measured the same way on the same statistic and the same subset, per link, and both numbers reported.** The statistic is the pixel-measured major extent's `(max - min) / median` over non-overlapping 200-frame windows. **Was an absolute 10% until 2026-08-30, which is below what the simulator itself scores**: ground truth reads 23.0% on link0 and 44.2% on link1 (`runs.jsonl` r47), so a perfect model failed 31 and 34 of its 36 windows. Two causes were identified and **removing them was measured not to help** (r50): deprojecting by the pixel-measured angle and excluding occluded or clipped frames still leaves 25.9%-34.8%, while discarding 77%-96% of the frames. The residual is the noise floor of a ~30-pixel PCA extent at 64x64. The ceiling is a property of the measurement, so the bar has to be too. `bench/link_drift_probe.py` measures the ground-truth term |
| Q-6 | S | **Object permanence**: block reappears in correct position after full occlusion | >= 80% of occlusion events |
| Q-6b | C | Same, with position error <= 2 px | >= 60% |

Q-6 is the memory result, back in scope because occlusion is native to manipulation rather than bolted on. It stays **S**, not M. It is the most likely thing to fail and the project ships without it.

Q-5 replaces velocity-preservation. A model that hallucinates arm geometry is the characteristic failure here.

**Q-3 was restated 2026-08-30**, after `bench/q3_blind_probe.py` measured what its
terminator could actually see. **F-9 itself is unchanged and is not at fault** -
it is a per-frame plausibility check accepted at zero false positives on ground
truth, it does that job, and it caught the noise control at 100% in the same run.
What was wrong is the *inference* Q-3 drew from it: that surviving 200 F-9
verdicts means the rollout stayed coherent. **Implementing F-9's block-count and
arm-pose halves would not have fixed this either** - they are per-frame
plausibility too, and the failure mode is precisely that a wrong frame is a
plausible one. Continuity is the weakest property a drifted rollout must actually
violate, because arriving somewhere wrong requires a step no physics allows. The
horizon and the tier are unchanged; this changes what terminates it.

**Q-5 was restated 2026-08-30**, and its restatement went the long way round.
`bench/link_drift_probe.py` first measured the old absolute bar against ground
truth and found a perfect model failing it (r47). Its two controls named
different causes for the two links - link0 is pure camera foreshortening, r 0.664
against the projection model with nothing clipped; link1 is not projection at all,
correlating **negative** at -0.104 while tracking visible pixel count at 0.951,
with 45.9% of frames under a floor projection cannot cross and 32.3% touching the
border. **Both causes were then removed, and the reading barely moved** (r50):
deprojection from the pixel-measured angle is sound where it can be checked
(r 0.928 against the `qpos`-derived factor) yet the residual drift is still
25.9%-34.8%, and the visibility filter that gets there discards up to 96% of the
frames. So the residual is the noise floor of measuring a ~30-pixel blob's PCA
extent at 64x64, which is a property of the resolution and not of any wording.
That is what forces the Q-4 treatment - a relative bar - rather than a better
statistic. **A robust spread would very likely have passed, and was rejected for
that reason**: choosing a statistic because ground truth passes it is circular,
which is also why r50 sweeps its one tolerance instead of picking a value.
Raising the absolute bar was rejected on the same evidence - past 44.2% it would
accept a model that has lost the link entirely, which is exactly what a 182.6%
window is. MuJoCo's links are rigid by construction; none of this was ever a
claim about the simulator's physics.

### Engineering

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| E-1 | M | Deterministic sim given a seed | Same as F-4 |
| E-2 | M | Clean build from scratch, MuJoCo and the offscreen GL context linked | Documented in README, verified once |
| E-3 | M | ASan clean on the full data-generation run; every shard offset and frame counter 64-bit, with a bounds assert at the write site | Zero ASan reports, and the assert fires on a deliberately overflowed offset |
| E-4 | M | Every bench number reproducible from a config hash | Rerun matches within 5% |
| E-5 | M | Append-only run log: config hash, change, number, conclusion | One entry per run |

E-3 was ASan **and** UBSan until 2026-08-26. MSVC has no UBSan, and a second
toolchain was rejected in favour of typing the offsets: reasoning and the trigger
that would reverse it are in `world_model_architecture.md`, "Sanitizer cost".

---

## 3. Ship criteria

Every **M** row passes and the ladder table is populated end to end.

## 4. Explicit non-requirements

Photorealism. Sim-to-real transfer. Policy learning or planning on top of the model. Generalization to unseen scenes. Multi-arm. Dexterous or grasping manipulation. Stability past 500 rollout steps. **Linux support** - the project runs on Windows and nothing needs to be portable off it.

## 5. Requirements at risk

| ID | Risk | Fallback |
|---|---|---|
| F-2 | Flat config may still leave gradients that hurt Q-1 | Drop to 96x96 / 144 tokens, promote F-16 to M |
| P-1 | 30 fps is tight on the 144-token path | DiagD becomes required. If still short, report the curve and accept 20 fps rather than cutting a ladder rung |
| Q-6 | Object permanence may simply not emerge at 15M params | Stays S. Report the measurement either way, including a negative result |
| R-3 | 20M may be too small for Q-4 | Raise to 25M only if profiling shows headroom. Never above 40M |
| **F-7** | **Was restated 2026-08-28 to exclude blocks that never return**, after measurement showed 73% of the old count was exactly that. Residual risk: the margin is now 1.8x, not 6.6x, so a scene edit that reduces genuine occlusion has far less room than the old number suggested. The recorded *cause* of the bias was also wrong - it said blocks were knocked off the table, and none ever has | Re-run `bench/occlusion_probe.py` after any scene change; it reports the split, the per-block breakdown, and whether a block has left the table or merely the camera. If the rate falls under 3%, widen the camera or move a block rather than relaxing the floor - F-7 exists to give Q-6 events to score |
| **Q-4** | **Was measured 2026-08-28 to sit above its own ceiling, and has been restated relative** - see the row above. Residual risk: the ground-truth term must be recomputed whenever the scene or `action_hold_steps` changes, and a Q-4 row quoting only the model's number is unfalsifiable | Report both numbers or the row does not count. `bench/hold_probe.py` produces the ground-truth term |
| **Q-3** | **Was measured 2026-08-30 to terminate on a verdict blind to dynamics failure, and its terminator has been replaced** - see the row above. Residual risk: the continuity bound is calibrated on ground-truth frames, which are perfectly rendered, while Q-3's inputs are decoder output - the same two-regime trap that cost build-order item 6 its obvious recipe | Calibrate on **reconstructions**, not renders. Keep `bench/q3_blind_probe.py` as the regression test: a verdict that stops firing on the 300-step substitution has silently gone blind again |
| **Q-5** | **Was measured 2026-08-30 to sit far below its own ceiling, and has been restated relative** - see the row above. Residual risk: the ground-truth term must be recomputed whenever the scene, the camera or the resolution changes, and a Q-5 row quoting only the model's number is unfalsifiable. The 1.1x factor is a judgement, not a measurement - nothing has established how much worse than the simulator a *bad* model reads on this statistic, so the bar may not discriminate | Report both numbers or the row does not count; `bench/link_drift_probe.py` produces the ground-truth term. Before Phase 3 quotes a Q-5 verdict, measure the statistic on a deliberately broken rollout - if it does not separate from the simulator's own reading, Q-5 has no discriminating power and should be retired rather than re-tuned. **Do not re-attempt the deprojection**: r50 records it as measured and refuted |
| P-6 | Two risks now. A software rasterizer instead of the GPU, ~50x slower; and `mjr_readPixels` fixed per-call cost under GLFW, reported at ~30 ms | Assert the renderer string in Phase 0 day 1 - **not** the vendor string, which does not identify hardware. Then measure per-call readback latency in isolation: above ~0.5 ms collapse to the single-pass render, near ~30 ms hand-roll a WGL pbuffer context |
