# Stride curve, r1c tokens (2026-11-10)

`bench/stride_pilot.py --tokenizer 20261110-005120-r1c`, plain one-pass model, final checkpoints, greedy rollouts on the 27 val episodes. Strides 1, 5 and 15 at seed 0 are the earlier runs (`r1c_s1`, `r1c_s5`, `r1c_s15`); the rest ran today, one at a time, in this order: seed 1 at 15 and 5, then seed 0 at 30, 10, 8, 3, 2. Figure: `../stride_curve.png`.

Pass rule: base-link tracking over about 250 ms at least 0.8, action gain at least 0.5, frozen share at most 1.5x the truth's.

| Stride | ms/frame | Seed | Tip px/frame (median) | Steps (epochs) | Base tracking (horizon ms) | Action gain | Frozen ratio | Forearm tracking (tokens' ceiling) | One-step lift | n | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 0 | 0.13 | 30,000 (1.7) | 0.19 (250) | 0.09 | 4.06x | 0.06 (0.85) | +0.072 | 513 | miss |
| 2 * | 4 | 0 | 0.26 | 30,000 (3.6) | 0.44 (248) | 0.69 | 3.29x | 0.16 (0.88) | +0.118 | 513 | miss |
| 3 | 6 | 0 | 0.38 | 25,000 (4.6) | 0.57 (252) | 0.92 | 2.60x | 0.18 (0.91) | +0.165 | 486 | miss |
| 5 | 10 | 0 | 0.64 | 20,000 (6.4) | 0.73 (250) | 0.84 | 1.86x | 0.30 (0.92) | +0.184 | 459 | miss |
| 5 | 10 | 1 | 0.64 | 20,000 (6.4) | 0.68 (250) | 0.92 | 1.98x | 0.34 (0.89) | +0.180 | 459 | miss |
| 8 * | 16 | 0 | 1.01 | 15,000 (8.5) | 0.74 (256) | 0.95 | 1.46x | 0.35 (0.87) | +0.162 | 405 | miss |
| 10 * | 20 | 0 | 1.24 | 12,000 (9.0) | 0.68 (240) | 0.87 | 1.23x | 0.39 (0.84) | +0.147 | 459 | miss |
| 15 | 30 | 0 | 1.79 | 8,000 (10.8) | 0.90 (240) | 1.00 | 1.14x | 0.37 (0.84) | +0.130 | 486 | pass |
| 15 | 30 | 1 | 1.79 | 8,000 (10.8) | 0.77 (240) | 0.90 | 1.05x | 0.45 (0.84) | +0.142 | 486 | miss (tracking) |
| 30 † | 60 | 0 | 3.15 | 3,000 (20.4) | 0.59 (240) | 1.04 | 0.83x | 0.15 (0.97) | -0.079 | 54 | miss |

\* Stride does not divide the 15-step action hold, so some intervals between two frames span two actions; the script reads the action at the later frame and asserts nothing. The model sees only one action per interval.

† Stride 30 is a multiple of the hold: every interval spans exactly two actions, and the model sees only the second. It also leaves 20 frames an episode: 5 training windows each (2,365 in all, 20 epochs at 3,000 steps), and 2 rollout starts per val episode, so n = 54. One-step accuracy is under copying; the best-by-loss checkpoint (step 2,000) is under too (lift -0.053). Too little data to separate the clock from the data shortage.

Tip motion: `measurements.md`, except stride 10, measured today with a copy of `stride_probe.py` set to stride 10 (same 125 episodes): median 1.24 px, p90 3.22 px, 15-frame context travel median 9.1 px.

Step counts for the new strides were chosen between the earlier runs', not held at equal epochs; held-out loss turns up after about 4 epochs at stride 15, so the stride 8 to 30 rows are at or past that point.
