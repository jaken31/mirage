# Robot World Model: Requirements (v3, robotics domain)

Each requirement has a tier: **M** = must (v1 does not ship without it),
**S** = should, **C** = could.

The ID in each table's first column (`F-7`, `Q-3` and so on) is how other docs
and code point at a row. `F` is functional, `P` performance, `R` resource, `Q`
quality, `E` engineering.

---

## 1. Functional requirements

### Simulation and data

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| F-1 | M | MuJoCo scene: 2-link planar arm, 3 pushable blocks, fixed camera | Scene loads, arm reaches all blocks |
| F-2 | M | Flat-render config enforced: ambient-only light, no shadows, box geoms, no textures, offsamples=0 | Rendered frame has <= 24 unique RGB values |
| F-3 | M | C++ harness renders offscreen on **GPU hardware**, never a software rasterizer. **On Linux, on the NVIDIA GPU only** | `glGetString(GL_RENDERER)` names neither `GDI Generic` nor `Microsoft Basic Render Driver` (nor, on Linux, `llvmpipe` or `softpipe`). On Linux it must also contain `NVIDIA`: a hybrid laptop's integrated GPU is real hardware and renders different frames under the same `data_hash` (`runs.jsonl` r55), and the Intel path now aborts at context creation (r56). Windows keeps the deny-list alone. Neither check names "RTX 5060": an allow-list on one model would fail on any other machine that is perfectly fine. Asserted at context creation |
| F-4 | M | Deterministic given a seed | Same seed and action sequence give bit-identical frames |
| F-5 | M | Data policy: 50/50 random joint deltas and scripted noisy reach | Over >= 2,000 episodes at the configured length: every one of the 9 actions holds **>= 5% of frames**, and **max bin / min bin <= 2.5**. `policy_dry_run` reports both |
| F-6 | M | Arm-block contact events exceed 5% of frames | Contact counter over a full run |
| F-7 | M | Block fully occluded in >= 3% of frames, **counting only occlusion the block recovers from** | Visible-pixel counter, excluding any stretch where the block never becomes visible again in that episode. `validator.recoverable_occlusion_rate_min` |
| F-8 | M | Shard writer emits packed frames and actions | Round-trip via numpy memmap matches the C++ buffer byte for byte |
| F-9 | M | Frame validator reports block count, arm pose plausibility, palette adherence | Zero false positives on ground-truth frames |

The flat-render config and the occlusion floor are new, and both carry real
weight. The flat render protects the token budget. The occlusion floor is what
gives object permanence something to score.

**The occlusion floor was restated 2026-08-28**, after
`bench/occlusion_probe.py` measured what its counter was actually counting. It
had been "any frame where a block reads zero pixels". **73% of that was blocks
that never came back** - 14.48 points of 19.83.

A block that is gone for the rest of the episode is not occluded. Scoring object
permanence on an occlusion that never ends asks the model to recover something
that never reappears. So the restated requirement counts only occlusion the block
recovers from: **5.35%**. That still clears the 3% floor, but at 1.8x rather than
the 6.6x the old number implied. The floor itself did not change. What changed is
what gets counted.

### Models

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| F-10 | M | FSQ tokenizer encodes a frame to an 8x8 grid over 512 levels and decodes back | Meets the tokenizer reconstruction bar |
| F-11 | M | Dynamics model consumes interleaved frame and action tokens, predicts next token | Held-out accuracy beats the **persistence baseline** - copying the previous frame's token at the same cell - scored like-for-like on the same held-out population as the model, re-measured with `bench/token_stability_probe.py` on that population. On the R1 checkpoint that probe measures it at **85.67%** over 12 val episodes. The marginal-frequency baseline is reported alongside |
| F-12 | M | Generates a full next frame from previous frames plus one action, fixed step count | No fallback path |
| F-13 | S | Configurable context length at load time | Rollout runs at 4, 8, 15 frames from one checkpoint |

**The dynamics model's acceptance test was restated 2026-09-22**, on a
decision taken 2026-09-18 to test it against the persistence baseline instead of
the marginal frequency. It had read "held-out accuracy beats marginal-frequency
baseline by 3x". That text is superseded.

The reason: a test that a zero-parameter baseline passes cannot show the model
learned dynamics. `bench/token_stability_probe.py` measured the zero-parameter
copy of the previous frame at **85.67%** over 460,032 held-out cell-transitions
from 12 val episodes, on the R1 checkpoint Phase 2 inherits. It recorded that as
far above 3x the marginal top-1. Beating the marginal frequency only asks the
model to know which codes are common. Beating persistence asks it to know when a
token changes, and that is the only part of the sequence that carries the
physics.

**The comparison with the old bar is asserted, not measured.** Neither that probe
run nor `bench/token_stability_probe.py` computes the marginal top-1. The first
dynamics-model run that reports that baseline alongside is where the comparison gets a
number. The tier and the description are unchanged. No margin over persistence
is set: the bar is the baseline itself.

**The comparison was measured 2026-09-23, on one population** (`runs.jsonl`
r54). `bench/mask_probe.py` computes the marginal top-1 - the single most common
code, 474, fit on the train split's frames - at 7.00% over every val window's last
frame, where persistence reads 86.69%, 12.4x it. On the token-stability probe's
12-episode population the comparison is still asserted, and the paragraph above
stands as written for it.

**The dynamics model's description was amended 2026-09-23.** This is a second amendment,
separate from the 2026-09-22 restatement above, which stands as written. The
table row keeps its original wording, and **this paragraph supersedes its
"predicts next token"**. It now reads: *predicts the next frame's tokens
together, each from the frames and actions before it.*

The measurement of strictly-causal against block-causal attention, ordered
2026-09-22, selected **block-causal** (`bench/mask_probe.py`). Under
block-causal, the 64 tokens of a frame attend to each other and are predicted in
one pass from the earlier frames and the action that produced the frame. No token
is predicted from the one before it in the same frame, so "predicts next token"
stopped describing the model. Block-causal won on the generated next frame from
ground-truth context, 2.40 points ahead of strictly-causal against a seed spread
of 0.20. Strictly-causal compounds its own early errors through the rest of a
frame.

**The acceptance test is unchanged.** Under block-causal, the teacher-forced and
generated accuracies are the same number by construction, so the test has one
reading. **The mask measurement is not a verdict on this requirement.** It scored two probe
models at a one-epoch endpoint past their held-out minimum, and both sit below
persistence on its population. The tier is unchanged.

**Added 2026-09-24:** the block-causal arm was also below persistence before its
endpoint. Under block-causal its held-out curve's last-frame accuracy is the
generated score, and against persistence on the curve's own 512 windows (86.47%)
neither seed was above it at any of 18 checkpoints, the best-accuracy and
lowest-loss points included (`runs.jsonl` r59). That is still a one-epoch probe
model, not a verdict on this requirement.

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

The generation throughput bar dropped from 20k/sec. MuJoCo plus an
offscreen render is far slower than a hand-written 2D sandbox. 500/sec gives
300k frames in about ten minutes, which is fine.

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
| Q-3 | M | **Coherence horizon**: frames until the rollout's **frame-to-frame continuity** check fires | **>= 200.** The check bounds how much the pose features can change per step. It uses features already on the validator's vector: `link_angle`, `link_extent`, and each block's `bbox` centroid. It is calibrated so that **zero windows of ground-truth frames fire**, the same acceptance shape the frame validator uses. **Until 2026-08-30 this was "frames until the F-9 validator fails", and that check was blind.** `bench/q3_blind_probe.py` fed it the reconstruction of a frame 300 steps out of position. The frame validator's palette check fired on **0.00%** of those, against 0.00% on correct reconstructions and 100% on its sigma-16 noise control. The blindness is **structural, not a threshold problem**: a drifted frame is a *plausible* frame, and the check never looks at frame `t`. No value of `offpalette_tau` or `offpalette_frac_max` can recover a comparison the statistic does not make. `bench/q3_blind_probe.py` is the acceptance test for the replacement: it must fire on 100% of 300-step substitutions and 0% of clean reconstructions |
| Q-3b | S | Same | >= 500 |
| Q-4 | M | Action-following accuracy, **scored relative to the simulator's own score** | **>= 90% of the ground-truth agreement measured the same way on the same subset**, and **both numbers reported**. Agreement is `sign(theta_t+1 - theta_t)` against the commanded sign. The subset is **action-balanced** - equal frames per action, drawn from the val split, not the raw split. The data policy's 5% floor is what makes that subset drawable. **Until 2026-08-28 this was an absolute 90%, which is above what the simulator itself scores.** Ground truth reads 83.1% at `action_hold_steps = 20`, because for about one joint settling time after each sign flip the joint is still moving the old way. An absolute bar there fails a model that is exactly right. The ceiling is a property of the physics, so the bar has to be too. `bench/hold_probe.py` measures the ground-truth term |
| Q-5 | M | Arm kinematic plausibility: link-length drift across a 200-step rollout, **scored relative to the simulator's own drift** | **<= 1.1x the ground-truth drift, measured the same way, on the same statistic and the same subset, per link, and both numbers reported.** The statistic is the pixel-measured major extent's `(max - min) / median` over non-overlapping 200-frame windows. **Until 2026-08-30 this was an absolute 10%, which is below what the simulator itself scores.** `bench/link_drift_probe.py` measured ground truth at 23.0% on link0 and 44.2% on link1, so a perfect model failed 31 and 34 of its 36 windows. Two causes were identified, and **removing them was measured not to help**. Deprojecting by the pixel-measured angle and excluding occluded or clipped frames still leaves 25.9%-34.8%, while discarding 77%-96% of the frames. What remains is the noise floor of a ~30-pixel PCA extent at 64x64. The ceiling is a property of the measurement, so the bar has to be too. `bench/link_drift_probe.py` measures the ground-truth term |
| Q-6 | S | **Object permanence**: block reappears in correct position after full occlusion | >= 80% of occlusion events |
| Q-6b | C | Same, with position error <= 2 px | >= 60% |

Object permanence is the memory result. It is back in scope because
occlusion comes naturally with manipulation rather than being bolted on. It stays
**S**, not M. It is the thing most likely to fail, and the project ships without
it.

Arm plausibility replaces velocity preservation. A model that hallucinates
arm geometry is the typical failure here.

**The coherence horizon was restated 2026-08-30**, after
`bench/q3_blind_probe.py` measured what its stopping check could actually see.

**The frame validator is unchanged and is not at fault.** It is a
per-frame plausibility check, accepted at zero false positives on ground truth.
It does that job, and it caught the noise control at 100% in the same run. What
was wrong is the conclusion the coherence horizon drew from it: that a rollout
surviving 200 validator checks stayed coherent.

**Implementing the validator's block-count and arm-pose halves would not have fixed this
either.** They are per-frame plausibility checks too, and the failure is exactly
that a wrong frame looks plausible. Continuity is the weakest property a drifted
rollout must actually break: arriving somewhere wrong takes a step no physics
allows. The horizon and the tier are unchanged. What changed is what ends it.

**Arm plausibility was restated 2026-08-30**, and it took the long way
round. `bench/link_drift_probe.py` first measured the old absolute bar against
ground truth and found a perfect model failing it. Its two controls pointed at
different causes for the two links:

- **link0 is pure camera foreshortening**: r 0.664 against the projection model,
  with nothing clipped.
- **link1 is not projection at all.** It correlates **negative**, at -0.104,
  while tracking visible pixel count at 0.951. 45.9% of its frames sit under a
  floor projection cannot cross, and 32.3% touch the border.

**Both causes were then removed, and the reading barely moved.** Deprojection
from the pixel-measured angle is sound where it can be checked (r 0.928 against
the `qpos`-derived factor), yet the remaining drift is still 25.9%-34.8%. The
visibility filter that gets there discards up to 96% of the frames. So what
remains is the noise floor of measuring a ~30-pixel blob's PCA extent at 64x64.
That is a property of the resolution, not of any wording. It is what forces the
action-following treatment - a relative bar - rather than a better statistic.

**A robust spread would very likely have passed, and was rejected for that
reason.** Choosing a statistic because ground truth passes it is circular. That
is also why the repair run sweeps its one tolerance instead of picking a value.
Raising the absolute bar was rejected on the same evidence: past 44.2% it would
accept a model that has lost the link entirely, which is exactly what a 182.6%
window is. MuJoCo's links are rigid by construction; none of this was ever a
claim about the simulator's physics.

### Engineering

| ID | Tier | Requirement | Acceptance test |
|---|---|---|---|
| E-1 | M | Deterministic sim given a seed | Same as the functional determinism test |
| E-2 | M | Clean build from scratch, MuJoCo and the offscreen GL context linked | Documented in README, verified once |
| E-3 | M | ASan clean on the full data-generation run; every shard offset and frame counter 64-bit, with a bounds assert at the write site | Zero ASan reports, and the assert fires on a deliberately overflowed offset |
| E-4 | M | Every bench number reproducible from a config hash | Rerun matches within 5% |
| E-5 | M | Append-only run log: config hash, change, number, conclusion | One entry per run |

The sanitizer requirement was ASan **and** UBSan until 2026-08-26. MSVC has
no UBSan. Adding a second toolchain was rejected in favour of typing the offsets
64-bit. The reasoning, and the trigger that would reverse it, are in
`world_model_architecture.md`, "Sanitizer cost".

---

## 3. Ship criteria

Every **M** row passes and the ladder table is populated end to end.

## 4. Explicit non-requirements

Photorealism. Sim-to-real transfer. Policy learning or planning on top of the model. Generalization to unseen scenes. Multi-arm. Dexterous or grasping manipulation. Stability past 500 rollout steps. **Portability beyond Windows and Linux on x86_64** - both run natively (`README.md`, "Build"), and nothing needs to go further, such as macOS or ARM.

## 5. Requirements at risk

| ID | Risk | Fallback |
|---|---|---|
| F-2 | The flat config may still leave gradients that hurt tokenizer PSNR | Drop to 96x96 / 144 tokens, promote fused Triton and diagonal decoding to M |
| P-1 | 30 fps is tight on the 144-token path | Diagonal decoding (DiagD) becomes required. If still short, report the curve and accept 20 fps rather than cutting a ladder rung |
| Q-6 | Object permanence may simply not emerge at 15M params | Stays S. Report the measurement either way, including a negative result |
| R-3 | 20M may be too small for action-following | Raise to 25M only if profiling shows headroom. Never above 40M |
| **F-7** | **Restated 2026-08-28 to exclude blocks that never return**, after measurement showed 73% of the old count was exactly that. Remaining risk: the margin is now 1.8x, not 6.6x, so a scene edit that reduces genuine occlusion has far less room than the old number suggested. The recorded *cause* of the bias was also wrong: it said blocks were knocked off the table, and none ever has been | Re-run `bench/occlusion_probe.py` after any scene change. It reports the split, the per-block breakdown, and whether a block has left the table or only the camera's view. If the rate falls under 3%, widen the camera or move a block rather than relaxing the floor. The floor exists to give object permanence events to score |
| **Q-4** | **Measured 2026-08-28 to sit above its own ceiling, and restated as a relative bar** - see its row above. Remaining risk: the ground-truth term must be recomputed whenever the scene or `action_hold_steps` changes, and an action-following row quoting only the model's number cannot be checked | Report both numbers or the row does not count. `bench/hold_probe.py` produces the ground-truth term |
| **Q-3** | **Measured 2026-08-30 to stop on a check blind to dynamics failure, and its stopping check was replaced** - see its row above. Remaining risk: the continuity bound is calibrated on ground-truth frames, which are perfectly rendered, while the horizon's inputs are decoder output. That is the same two-population trap that cost build-order item 6 its obvious recipe | Calibrate on **reconstructions**, not renders. Keep `bench/q3_blind_probe.py` as the regression test: a check that stops firing on the 300-step substitution has silently gone blind again |
| **Q-5** | **Measured 2026-08-30 to sit far below its own ceiling, and restated as a relative bar** - see its row above. Remaining risk: the ground-truth term must be recomputed whenever the scene, the camera or the resolution changes, and an arm-plausibility row quoting only the model's number cannot be checked. The 1.1x factor is a judgement, not a measurement. Nothing has established how much worse than the simulator a *bad* model reads on this statistic, so the bar may not tell good from bad | Report both numbers or the row does not count; `bench/link_drift_probe.py` produces the ground-truth term. Before Phase 3 quotes an arm-plausibility verdict, measure the statistic on a deliberately broken rollout. If it does not separate from the simulator's own reading, the requirement cannot tell models apart and should be retired rather than re-tuned. **Do not re-attempt the deprojection**: it was measured and refuted |
| **F-11** | **Restated 2026-09-22 against the persistence baseline** - see the first paragraph under the Models table. **Its description was amended 2026-09-23** - see the second paragraph there. The measurement of strictly-causal against block-causal attention at a fixed step budget, ordered 2026-09-22 to run before Phase 2's first build item, ran (`bench/mask_probe.py`) and selected block-causal, so "predicts next token" no longer describes the model. Two remaining risks. First, the mask measurement's block-causal model did not beat persistence at any checkpoint of its one-epoch curve, its held-out optimum included: no curve checkpoint was above persistence on the curve's 512 windows (re-read 2026-09-24, `runs.jsonl` r59). Strictly-causal's generated score at its optimum was not measured, because its curve accuracy is teacher-forced. Second, the baseline belongs to the tokenizer checkpoint, not to F-11: it reads 85.67% on R1, 93.22% on r1c and 77.28% on R2, so it moves whenever the tokenizer does | Block-causal was selected, and the description is restated in the second, separately dated amendment rather than by editing the first. Reopen the mask only on a generated-frame comparison at each arm's held-out optimum, on the mask measurement's population, with strictly-causal ahead by more than the seed spread. Re-run `bench/token_stability_probe.py` on any new tokenizer checkpoint and score against that figure, not against the 85.67% R1 figure |
| P-6 | Two risks. A software rasterizer instead of the GPU, ~50x slower; and a fixed per-call cost for `mjr_readPixels` under GLFW, reported at ~30 ms | Assert the renderer string in Phase 0 day 1 - **not** the vendor string, which does not identify hardware. Then measure per-call readback latency in isolation: above ~0.5 ms, collapse to the single-pass render; near ~30 ms, hand-roll a WGL pbuffer context |
