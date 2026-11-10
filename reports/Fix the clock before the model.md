# Fix the clock before the model

**Recommendation: stop changing the model and change the frame rate.** The dataset records one frame every 2 ms of simulated time. At that rate the arm's tip moves a median of 0.13 pixels per frame, 3 of 4,096 pixels change, and the model's whole 15-frame context covers 1.7 pixels of motion. The model is asked to predict motion smaller than its input can show. The best available answer is then "copy the last frame", and that is what every run since r62 learned. A read-only probe written for this report measured the motion. No model has been trained at a slower clock yet, so the causal claim is an inference with a one-day test attached.

Two more findings change the plan around it. The tokenizer fix for spurious flips already exists and is blocked only by the token-entropy bar. And the speed thesis has lost its bottleneck: today's model runs at 747 frames per second with no engine work.

The framework itself (tokens, a small transformer, measured gates) is sound. Three of its inputs are wrong: the clock, one tokenizer gate, and the premise of the engine phase.

Labels used below: **measured** is from this report's probes or a `runs.jsonl` row. **Published** means the cited source states it. **Inference** is my reading. Research date 2026-11-10.

## Result of the clock test (2026-11-10): passes at 30 ms

Run after this report was written, with `bench/stride_pilot.py`: the plain one-pass 2.0M model, real context only, no rollout training, no flip cleaning, on the retrained `r1c` tokenizer (`20261110-005120-r1c`: 30.88 dB held-out, 0 spurious flips on 460,032 transitions). Greedy rollouts on the 27 val episodes, final checkpoints, one seed.

| | Every frame, 2 ms (control) | Every 5th, 10 ms | Every 15th, 30 ms | Bar |
|---|---|---|---|---|
| Base-link tracking over about 250 ms | 0.19 | 0.73 | **0.90** | at least 0.8 |
| Base-link action gain | 0.09 | 0.84 | **1.00** | at least 0.5 |
| Frozen rollout frames, against the truth's | 87.5% vs 21.6% (4.06x) | 17.0% vs 9.1% (1.86x) | **9.8% vs 8.6% (1.14x)** | at most 1.5x |
| Base-link angle error at 250 ms, against held still | 19.8 vs 20.7 deg | 4.4 vs 20.1 | 4.4 vs 17.4 | reported |
| Forearm tracking over about 250 ms (tokens' ceiling) | 0.07 (0.85) | 0.30 (0.92) | 0.37 (0.84) | reported |
| One-step lift over copying | +0.07 | +0.18 | +0.13 | reported |
| Training time | 23 min | 15 min | 6 min | |

**Every 15th frame passes all three bars. Every 5th misses two.** The 2 ms control uses the same tokenizer, model and code and still freezes, so the clock is the cause and the clean tokenizer alone is not.

What it does not show:

- **The forearm is still poor**: 0.37 against a tokenizer ceiling of 0.84. It beats holding still on angle error (18 against 29 degrees) but does not track.
- **One seed, 18,900 training frames at 30 ms.** Held-out loss turns up after about 3,000 steps. The checkpoint with the best one-step loss rolls out worse than the final one (frozen 16.1% against 9.8%), so do not select on one-step loss.
- With another episode's actions, tracking falls to about zero at every stride, so the motion comes from the actions.

The same runs on R1's noisy tokens, started while `r1c` trained, moved the same way but less: at 30 ms, tracking 0.70, gain 0.88, frozen 0.80x the truth's. So the tokenizer matters too, second.

Logs and numbers: `research_notes/Fix the clock before the model/pilots/`. Next is step 3 of the plan: the new dataset.

## Where the project stands

- It is week 12 of a 12-week plan with 2 weeks of slack (first commit 2026-08-20). The dynamics phase was budgeted at 2 weeks and is unfinished. The playable loop and the engine have not started, and `docs/timeline.md` calls the speed target "the whole thesis".
- About thirty dynamics experiments have run (r62 to r94, plus one pilot not yet logged). One-step accuracy has never beaten copying the last frame by more than 0.26 points, and seeds alone move it by 0.73 (r66). Rollouts froze on 91-96% of frames until training on the model's own frames (r93, r94), which bought movement and cost one-step accuracy.
- The third pilot finished on 2026-11-09 and is not in `runs.jsonl` yet: one-step 1.53 points under copying, base-link tracking 0.84, action gain 0.69 (`dynrs.log`, `pilotscorers.log`). Its own rule says nothing further launches. This line of work is at its planned end.

## Finding 1: a frame is 2 ms, and the arm moves an eighth of a pixel in it

**Measured.** `scene/arm_blocks.xml` sets a 2 ms physics step, and `sim/main.cpp` renders after every step. The joints top out near 4 rad/s, so a joint turns at most 0.46 degrees per frame.

The probe compares frame `t` with frame `t+k` on the stored frames and R1's tokens, and projects the arm's tip through the scene camera. Sample: every 4th episode, 125 episodes, 75,000 frames.

| Record every | Time per frame | Tip moves per frame, median / 90th percentile | Pixels that change, median of 4,096 | Cells that keep their token | Frames with no token change | Tip travel across a 15-frame context, median |
|---|---|---|---|---|---|---|
| 1 step (today) | 2 ms | 0.13 / 0.33 px | 3 | 86.8% | 19.2% | 1.7 px |
| 5 steps | 10 ms | 0.64 / 1.63 px | 15 | 69.4% | 6.4% | 5.5 px |
| 8 steps | 16 ms | 1.01 / 2.59 px | 23 | 62.8% | 5.1% | 7.8 px |
| 15 steps | 30 ms | 1.79 / 4.79 px | 42 | 54.0% | 3.9% | 11.6 px |
| 30 steps | 60 ms | 3.15 / 9.07 px | 74 | 44.6% | 2.8% | 19.4 px |

Today the tip never moves a full pixel in one frame. A token cell is 8 pixels wide, so the model's entire context shows less than a quarter of a cell of travel.

**What this explains.** Each line is the project's own measurement; the link to the clock is inference.

- One step of motion is not in the tokens. The tokens' angle change correlates 0.66 (base link) and 0.17 (forearm) with the render's at 1 step, against 0.98 and 0.62 at 25 steps (r89).
- The model cannot tell which moving cells change: about 0.5 change probability on cells that do, 0.3 on cells that do not (r77). That is what a correct model looks like when the moment a cell flips depends on sub-pixel position it cannot see. Greedy decoding then picks "keep" everywhere and the rollout freezes. Sampling moves cells without knowing where they should go (r72, r81).
- Real context helps because it carries real velocity. Resetting the context every 5 steps lifts base-link tracking from 0.56 to 0.77 (r90, r91).
- The evaluation instruments fail for the same reason. On ground truth, the pixel-measured direction of motion agrees with the command 47% of the time, which is chance, and the continuity check catches 54% of 300-step substitutions (r61). One frame of motion, at most 0.46 degrees, is smaller than the 0.6 to 3 degree error of reading an angle back from a decoded frame (r82-r85).
- Training on self-generated frames works because it is the only remedy that gives the model a multi-step signal. A slower clock gives that signal in every training step, with no trade against one-step accuracy, because one step is then a visible amount of motion.

**What the demo would look like.** Arithmetic from the scene file:

- Played at 30 frames per second, 2 ms frames are 17x slow motion. The fastest joint turns 14 degrees per second on screen.
- The 200-frame coherence target is 0.4 s of simulated time. "Stable past 500 steps", a stated non-requirement, is 1 s.
- One simulated second costs 500 model steps today, and 33 at 30 ms.

**Published practice.** Nobody trains a world model this finely.

- PlaNet trains on 64x64 frames and repeats each action 2 to 8 times "to reduce the planning horizon and provide a clearer learning signal to the model" ([Hafner et al. 2019](https://ar5iv.labs.arxiv.org/html/1811.04551)).
- Gymnasium's MuJoCo Pusher, an arm pushing an object, steps every 50 ms: 5 physics steps of 10 ms ([Gymnasium docs](https://gymnasium.farama.org/environments/mujoco/pusher/)).
- DIAMOND uses frameskip 4 on Atari and 16 Hz data for CS:GO ([Alonso et al. 2024](https://arxiv.org/html/2405.12399v2)). Genie trains at 10 frames per second ([Bruce et al. 2024](https://arxiv.org/html/2402.15391)).
- A learned plasma simulator was trained at strides of 1, 2 and 3 s. The longer strides "accumulate less forecast error" although they had a half and a third of the training snapshots ([arXiv 2601.12614](https://arxiv.org/html/2601.12614)).
- The NeurIPS 2026 paper the project already cites says densely sampled targets make copying score near 1 (2ρ − 1 for lag-1 autocorrelation ρ) and recommends reporting lift over that floor ([abstract](https://neurips.cc/virtual/2026/166122)).

These step every 50 to 100 ms. Mirage's 2 ms is 25 to 50 times finer.

## Finding 2: half of all token changes are tokenizer noise, at every clock

**Measured here.** About half of all token changes land on cells with no pixel change anywhere in the 22x22 pixels around them: 6.8 of 13.2 points at 2 ms, and 23.3 of 46.0 points at 30 ms. A slower clock alone leaves that noise in, and it grows with the stride.

**Measured by the project in August.** Normalising per channel only (the `r1c` rung, still in `mirage/fsq.py`) gives 0 flips on 396,013 quiet transitions, for 0.28 dB of reconstruction quality (30.81 against 31.10 dB, both over the 30 dB bar). It was rejected because it scores 54.6% on the token-entropy bar, which asks for 70%.

**Inference.** That bar ranks the three tokenizers in the same order as their flip rates: R2 at 77.6% entropy and 18.75% flips, R1 at 74.1% and 8.86%, r1c at 54.6% and none. Part of what the bar rewards is the whole-frame coupling that makes tokens flip. The 2026-11-01 report said the same and it was not acted on. The architecture doc's own second trigger, evidence that token instability costs prediction accuracy, fired at r68 and r69.

Recommendation: report the entropy figure, gate on dead codes instead, and take r1c. `--clean-flips` then has nothing to clean, in targets or in inputs.

## Finding 3: the speed thesis has no bottleneck left

**Measured 2026-11-10.** Batch 1, context 15, eager PyTorch, no cache, the 2.0M-parameter K-round model (`20261108-103434-dyn/best.pt`). The figures exclude decoding to pixels and display.

| Rounds per frame | Time per frame | Frames per second |
|---|---|---|
| 1 | 1.34 ms | 747 |
| 4 | 5.11 ms | 196 |
| 8 | 10.15 ms | 99 |

The plan's budget assumed token-by-token decoding: 64 steps per frame at 521 microseconds each. Two later decisions removed that. Block-causal attention (2026-09-23) made a frame one pass, and the size sweep took the model from 14.6M to 2.0M parameters. Both were sound on accuracy. Together they mean the 30 frames-per-second bar is passed about 25 times over before the first engine rung, and diagonal decoding has nothing to parallelise.

This needs a decision, not an experiment. It is in the next section.

## The framework, option by option

1. **Keep tokens and the transformer; fix the clock and the tokenizer.** Recommended first. It reuses all the code and can be tested in a day.
2. **Decode token by token, with sampling.** This is the original design, and how IRIS, GAIA-1 and MineWorld decode ([Micheli et al. 2023](https://arxiv.org/html/2209.00588v2); [Hu et al. 2023](https://arxiv.org/html/2309.17080); [MineWorld](https://arxiv.org/html/2504.08388v1)). It samples cells jointly, which is the published cure for the freeze. It also puts back the bottleneck the engine ladder was built for: about 86 ms per frame without a cache (64 passes at 1.34 ms, an estimate), so the ladder has a real 12 to 30+ frames-per-second story. Reopen the mask decision at the new clock; the September probe compared greedy decoding at 2 ms.
3. **Predict pixels.** A small residual U-Net, then DIAMOND-style diffusion, as the 2026-11-01 notes proposed. Published to work at 64x64 with a 4M-parameter U-Net. It drops the tokens, the cache and diagonal decoding, so the engine story changes with it.
4. **A causal transformer on continuous latents with a few-step flow objective.** Dreamer 4 runs this way ([Hafner et al. 2025](https://arxiv.org/abs/2509.24527)), and by its abstract so does the paper the ingredients doc calls "your architecture": it is "an autoregressive diffusion framework", not a discrete-token model ([arXiv 2601.21998](https://arxiv.org/abs/2601.21998)). It keeps the cache-based engine and has no tokens to flip. Highest ML risk. Only if 1 and 2 fail and time allows.

Changing the model family does not remove the 2 ms problem, so it should come after the clock.

**The thesis decision, which is yours.**

- (a) Keep "30 frames per second on a laptop, through an inference engine". That needs option 2, or a much larger model, or there is nothing to speed up.
- (b) Make the playable simulator and its exact-ground-truth scoring the product. Engine work becomes export: INT8 and a build that runs in a browser tab. Precedents exist: Neural Drive runs a 131M-parameter world model on WebGPU with the KV cache inside the ONNX graph ([model card](https://huggingface.co/codelion/neural-drive-model)), and a 2.4M-parameter 64x64 Breakout model runs on ONNX Runtime Web ([author's note](https://ankurbohra-com.pages.dev/notes/building-a-tiny-browser-playable-world-model)).

I recommend (a) with option 2. It is the original thesis, the study plan is built for it, and it doubles as a modelling fix. The browser build is a good last step if time remains. Decide after the clock test, not before.

## Plan

Costs are estimates, not measurements.

| Step | What | Cost | What it decides |
|---|---|---|---|
| 1 | Retrain the `r1c` tokenizer on today's frames and cache its tokens. | about 1.5 h | Clean tokens at any clock. |
| 2 | Clock sweep on today's data. Train the plain one-pass 2.0M model on every 5th and every 15th frame, windows aligned to the 15-step action hold so each interval has one action. No rollout training. Needs a strided window sampler and stride-aware scoring. | about 1 h a run, half a day of code | Whether the clock is the cause. |
| 3 | If step 2 passes: a new dataset. Keep physics at 2 ms, record every Nth step, one action per frame, an arm slow enough that the tip moves 1 to 4 px per frame, and blocks that stay in view. Re-run the Phase 0 probes. Retrain the tokenizer. | 2 to 3 days | The data the demo is played on. |
| 4 | Dynamics on the new data, two seeds, gated on rollout measures by simulated seconds. | about 1 day | The dynamics phase. |
| 5 | The playable loop and a side-by-side recording. | 3 days, the plan's own budget | The first thing to show. |
| 6 | The engine, after the thesis decision. | 1 to 2 weeks | The ladder table. |

**Rule for step 2, set before the run.** It passes if a model trained on real context only reaches all three at either stride: base-link tracking of at least 0.8 over 250 ms of simulated time, action gain of at least 0.5, and a frozen share within 1.5x the truth's at that stride. For comparison at 2 ms: tracking 0.48-0.59 over 50 ms and gain 0.13-0.19 without rollout training (r89, r90). The thresholds are judgement. Every 15th frame leaves 18,900 training frames, so that run is a direction check and will overfit.

**If step 2 fails at both strides:** the clock is not enough. Try option 2 at the slower clock, then option 3.

Extend the sweep to strides 2, 3 and 8 if the first two differ. Rollout tracking and frozen share plotted against pixels moved per frame is worth having either way.

## Making it presentable

1. **One recording at the top of the README:** simulator and model side by side under the same keys, 10 to 15 seconds. Today the README opens with a file table and has no picture of the scene.
2. **One command to play**, with a key that flips between the model and the real simulator. DIAMOND's README is the model to copy: demo GIFs first, then `python src/play.py --pretrained`, with arrow keys switching between world model and real environment ([repo](https://github.com/eloialonso/diamond)).
3. **Five numbers, each beside the simulator's own score:** tracking at 0.25 s and 1 s, action gain, frozen share, frames per second. The project already measures all of them (`bench/forearm_probe.py` parts 5 and 6, and the gate's frozen-share row).
4. **Writeup part two as one story:** thirty remedies at the wrong clock, the probe, the sweep. The figure is tracking against pixels moved per frame, with the failed runs as its left end. If the sweep shows a threshold, that is a finding on its own.
5. **Housekeeping:** 22 loose logs in the repo root, and an AGENDA of 754 lines against its own "keep it short".

## Going faster

- A full run at the 2 ms clock costs 7 hours (43 minutes an epoch, measured on `20261108-103434-dyn`). A pilot on every 15th frame of today's data is about 740 steps an epoch against 17,294. Run pilots first and full runs last.
- Two runs per hypothesis, then question the level above it. r75 to r94 refined the symptom at a fixed clock.
- Gate on what the demo needs: tracking by simulated seconds, action gain, frozen share, frames per second. Report the rest. Replace "beat copying at one step" with lift over copying.
- During a pilot series, write one `runs.jsonl` row and one AGENDA line per run. Update the architecture doc and the register at decisions. Since 2026-10-30 there are 30 docs commits and 17 feat commits.
- `cluster/launch.sh` already runs one job per GPU. Use it for seeds.

## Not verified

- No model has been trained at a slower clock. Finding 1's causal claim rests on the probe and on r61, r77, r89 and r90.
- The 86 ms token-by-token figure is arithmetic from the one-pass timing. No such model exists here.
- The timing excludes pixel decoding and display.
- The r1c numbers are the project's August measurements. Its zero flips at longer strides follow from its 15x15 pixel field but were not measured.
- Block coverage was only spot-checked: a block sits more than 5 cm from its start in 41% of frames. Every episode starts from the same block layout, and a long play session may leave what the data covers.
- Dreamer 4, the "Causal World Modeling" paper, the NeurIPS 2026 paper and Self Forcing were read from abstracts only.
- All time estimates are judgement.

## Reproduce

Raw output and the probe script are in `research_notes/Fix the clock before the model/`.

```bash
.venv/bin/python "research_notes/Fix the clock before the model/stride_probe.py" . 4
```

## Sources

- [PlaNet, Hafner et al. 2019](https://ar5iv.labs.arxiv.org/html/1811.04551)
- [Gymnasium Pusher](https://gymnasium.farama.org/environments/mujoco/pusher/)
- [DIAMOND, Alonso et al. 2024](https://arxiv.org/html/2405.12399v2) and its [repo](https://github.com/eloialonso/diamond)
- [Genie, Bruce et al. 2024](https://arxiv.org/html/2402.15391)
- [Hybrid-Vlasov surrogates, arXiv 2601.12614](https://arxiv.org/html/2601.12614)
- [Your world model may be worse than copying the last frame, NeurIPS 2026](https://neurips.cc/virtual/2026/166122)
- [MineWorld, arXiv 2504.08388](https://arxiv.org/html/2504.08388v1)
- [IRIS, Micheli et al. 2023](https://arxiv.org/html/2209.00588v2) and [GAIA-1, Hu et al. 2023](https://arxiv.org/html/2309.17080), as read for the 2026-11-06 report
- [Dreamer 4, arXiv 2509.24527](https://arxiv.org/abs/2509.24527)
- [Causal World Modeling for Robot Control, arXiv 2601.21998](https://arxiv.org/abs/2601.21998)
- [Self Forcing, arXiv 2506.08009](https://arxiv.org/abs/2506.08009)
- [Neural Drive model card](https://huggingface.co/codelion/neural-drive-model)
- [Building a tiny browser-playable world model](https://ankurbohra-com.pages.dev/notes/building-a-tiny-browser-playable-world-model)
