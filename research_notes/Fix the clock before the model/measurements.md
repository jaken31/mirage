# Measurements behind "Fix the clock before the model"

Taken 2026-11-10 on Linux, RTX 5060 Laptop, at git b472f40. Nothing here changed a tracked file.

## Frame-stride probe

Read-only, numpy, no GPU. `stride_probe.py` in this folder.

```bash
.venv/bin/python "research_notes/Fix the clock before the model/stride_probe.py" . 4
```

125 episodes (every 4th), tokenizer `20260829-005439-r1`, focal length 77.3 px. Each row compares frame `t` with frame `t+k`. "Moved (22)" means at least one pixel changed inside the cell's 8x8 patch grown by 7 px on every side. Tip positions are projected from `qpos` through the scene camera.

| stride | ms/frame | frames pixel-identical | median px changed | tokens kept | frames with no token change | cells moved (22) | token change on moved cells | token change on quiet cells | quiet share of changes | tip px median | tip px p90 | tip < 0.5 px | tip < 1 px | 15-frame context tip travel, median | same, p90 | pixel-identical while the arm moves | token-frozen while the arm moves |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 17.8% | 3 | 86.83% | 19.2% | 18.8% | 6.37% | 6.79% | 52% | 0.13 | 0.33 | 96% | 100% | 1.7 | 4.5 | 11.7% | 13.2% |
| 2 | 4 | 10.6% | 6 | 80.41% | 11.4% | 23.6% | 9.82% | 9.77% | 50% | 0.26 | 0.65 | 82% | 96% | 3.0 | 8.5 | 4.6% | 5.3% |
| 3 | 6 | 8.2% | 9 | 75.83% | 8.6% | 26.1% | 12.23% | 11.95% | 49% | 0.38 | 0.98 | 63% | 91% | 4.0 | 12.1 | 2.4% | 2.8% |
| 5 | 10 | 6.2% | 15 | 69.36% | 6.4% | 28.9% | 15.52% | 15.12% | 49% | 0.64 | 1.63 | 40% | 73% | 5.5 | 18.5 | 1.0% | 1.2% |
| 8 | 16 | 5.0% | 23 | 62.82% | 5.1% | 31.0% | 18.61% | 18.56% | 50% | 1.01 | 2.59 | 26% | 50% | 7.8 | 26.4 | 0.4% | 0.5% |
| 15 | 30 | 3.8% | 42 | 53.99% | 3.9% | 33.5% | 22.68% | 23.33% | 51% | 1.79 | 4.79 | 15% | 29% | 11.6 | 42.8 | 0.1% | 0.2% |
| 30 | 60 | 2.8% | 74 | 44.61% | 2.8% | 36.2% | 26.95% | 28.44% | 51% | 3.15 | 9.07 | 10% | 17% | 19.4 | 59.9 | 0.1% | 0.1% |

Checks against the project's own numbers: tokens kept at stride 1 is 86.83% here against 86.69% on the val windows (r54), and token-frozen frames 19.2% against 20.1% (r75). The populations differ (this is a sample of all episodes), so they should be close, not equal.

"Arm moves" means the two joint angles changed by more than 0.001 rad per step in total.

## Rollout speed at batch 1

`dynamics_eval.rollout` on `runs/20261108-103434-dyn/best.pt` (`dyn192x4_maskact`, 1,980,480 parameters), random seed clip, context 15, 120 frames after a 10-frame warm-up, bf16 autocast, CUDA synchronised before and after. Eager, no KV cache, full recompute each frame. Excludes decoding tokens to pixels.

| Rounds | ms per frame | frames per second |
|---|---|---|
| 1 | 1.34 | 747 |
| 4 | 5.11 | 196 |
| 8 | 10.15 | 99 |

One run, GPU otherwise idle, no clock or power gate recorded. Treat as an order of magnitude.

## Other facts read from the tree

- `scene/arm_blocks.xml`: `timestep="0.002"`; gear 6 and damping 1.5, so terminal joint speed is about 4 rad/s.
- `sim/main.cpp`: one `mj_step` per rendered frame.
- `mirage/configs/base.json`: `action_hold_steps` 15. All 12,742 action changes in the dataset land on `step_idx % 15 == 0`, so windows of every 5th or 15th frame can be aligned to carry one action per interval.
- `sim/policy.cpp`, `begin_episode`: joint angles are randomised, blocks start where the XML puts them every episode.
- Block displacement from its start over all 300,000 frames: median 0, 90th percentile 12.5 cm, 99th 41.3 cm. A block is more than 5 cm away in 41.3% of frames.
- Training speed, `runs/20261108-103434-dyn/metrics.jsonl`: 137-150 ms per step at batch 16, 2,411-2,653 s per epoch, 25,692 s for 10 epochs, median 78 W of a 100 W cap.
- Third warm-start pilot (`dynrs.log`, `pilotscorers.log`), not in `runs.jsonl`: final checkpoint 85.16% against persistence 86.69% (1.53 points under), base-link 25-step tracking 0.842, action gain 0.69.
- Commits by prefix: 73 docs, 50 feat, 9 fix overall; 30 docs and 17 feat since 2026-10-30.

## Sources read, and how far

Read in full or in the relevant section: PlaNet (ar5iv 1811.04551, action repeat), Gymnasium Pusher docs, DIAMOND (arXiv 2405.12399v2, first 100k characters) and its GitHub README, Genie (2402.15391), MineWorld (2504.08388v1), the hybrid-Vlasov surrogate paper (2601.12614, first 100k characters), the Neural Drive model card, the tiny browser world model note.

Abstract only: the NeurIPS 2026 copy-baseline paper, Dreamer 4 (2509.24527), Causal World Modeling for Robot Control (2601.21998), Self Forcing (2506.08009), Tallec et al. on time discretisation (1901.09732, not used in the report).

Could not be read: RigidFormer (2605.09196), PDF did not extract. Not used.

MineWorld's abstract and its Table 1 disagree on frame rate; the table's 5.91 frames per second for the 300M model matches the figure already in `docs/world_model_architecture.md`.
