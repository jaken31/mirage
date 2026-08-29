"""Calibrate F-9's palette verdict as a *quantile of distance*, not a pixel count.

    python bench/palette_pctl_probe.py --run 20260829-005439-r1
    python bench/palette_pctl_probe.py --run <96x96 run> --config mirage/configs/base96.json

Reads only: the run's `model.pt` and the shards. Writes nothing - the numbers go
into `validator.offpalette_pctl` / `validator.offpalette_dist_max` and a
`runs.jsonl` row by hand, the same way every other probe in `bench/` reports.

**Why this probe exists.** Build order item 6 calibrated the palette verdict as
`offpalette_px > N` - a count of pixels further than tau from every palette
entry - and landed N = 350 at 64x64. A count is a *resolution-dependent*
statistic, so the 96x96 fork needed its own N, and the obvious rescale by frame
area (350 * 2.25 = 788) is wrong on this project's own evidence: 96% of squared
error is edge geometry, and edge length scales 1.5x where area scales 2.25x. A
quantile of the per-pixel distance is invariant to frame size by construction,
so one threshold serves every resolution and the rescale question stops existing.

**The methodology is item 6's, deliberately unchanged**, so the two calibrations
are comparable:

  1. fix the threshold at the maximum over *clean* held-out reconstructions, which
     is zero false positives by construction;
  2. measure what fraction of a decayed frame each setting then catches;
  3. take the interior optimum.

The four decay proxies are item 6's four, for the same reason - a diverging world
model produces mush, and these are the four shapes mush takes:

  * `blend`  - a 50/50 average of two futures, the hedge a model makes when it
               cannot commit. This is the realistic one and the hardest to catch.
  * `blur3`  - a 3x3 box blur iterated 4x, detail loss without colour drift.
  * `noise16`- gaussian noise sigma 16, the high-frequency failure.
  * `grey`   - collapse to uniform grey, the degenerate endpoint.

**What a quantile buys and what it costs.** It buys resolution invariance, which
is the whole point. It costs the tail: at p99.9 on a 64x64 frame the statistic is
the 4th-worst pixel of 4,096, so it is nearly the maximum and inherits the
maximum's variance. At p99 it is the 41st-worst, much steadier but blind to a
small bright fault. The ladder below is what decides, and the answer is expected
to be interior for exactly that reason.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mirage import config, data, validator  # noqa: E402
from mirage.fsq_eval import load_run, reconstruct  # noqa: E402

# The ladder. Spans "41st-worst pixel of 4,096" to "2nd-worst", which is the
# range over which the statistic stops being a max and starts being a summary.
PCTLS = (0.99, 0.995, 0.999, 0.9995)
PROXIES = ("blend", "blur3", "noise16", "grey")
# `grey` is measured but **excluded from the pick**, and that is a finding rather
# than a convenience. A frame collapsed to its own mean lands ~22 RGB units from
# the table colour - inside any usable tau - so a uniform grey is *palette
# plausible* and no palette statistic can see it, the count included. It is
# caught by F-9's other two checks: every block's `px_count` goes to zero. Left
# in the table so nobody re-derives that the hard way.
PALETTE_PROXIES = ("blend", "blur3", "noise16")
# The candidate verdict statistics, in one table so the comparison is like for
# like: the quantile ladder, then the off-palette *fraction* at tau.
COLS = tuple(f"p{q:.4%}" for q in PCTLS) + ("offpal_frac",)
SAMPLE = 4000  # frames per proxy - the clean side always uses the whole split


def box_blur(x: np.ndarray, iters: int) -> np.ndarray:
    """3x3 box blur, `iters` times, edge-replicated. float in, float out."""
    for _ in range(iters):
        p = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
        acc = np.zeros_like(x)
        for i in range(3):
            for j in range(3):
                acc += p[:, i:i + x.shape[1], j:j + x.shape[2], :]
        x = acc / 9.0
    return x


def decay(clean: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    """One decay proxy over a (n, h, w, 3) uint8 batch. Returns uint8."""
    x = clean.astype(np.float64)
    if kind == "blend":
        # Pair each frame with a *different* frame, never itself: a 50/50 blend
        # of a frame with itself is the identity and would measure nothing.
        x = 0.5 * x + 0.5 * x[rng.permutation(len(x))]
    elif kind == "blur3":
        x = box_blur(x, 4)
    elif kind == "noise16":
        x = x + rng.normal(0.0, 16.0, x.shape)
    elif kind == "grey":
        x = np.broadcast_to(x.mean(axis=(1, 2, 3), keepdims=True), x.shape).copy()
    else:
        raise ValueError(f"unknown decay proxy {kind!r}")
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def frame_stats(frames: np.ndarray, palette: validator.Palette,
                pctls: tuple[float, ...], tau: float) -> np.ndarray:
    """(n, len(pctls) + 1) per frame: each distance quantile, then the off-palette
    *fraction* at `tau` - the count statistic made resolution-free by dividing by
    the frame's pixels.

    Both candidates in one pass, because `_label` re-derives the nearest-palette
    mapping on each call and that is the expensive part. Measuring them together
    is also the only way the comparison is fair: same frames, same decode.

    The two are different questions, which is the whole finding here. A quantile
    asks *how far off* the frame's worst pixels are - a tail statistic. A fraction
    asks *how much* of the frame is off - a bulk statistic. Uniform decay (grey
    collapse, additive noise) is a bulk failure with a modest tail, so a tail
    statistic cannot see it.
    """
    out = np.empty((len(frames), len(pctls) + 1), dtype=np.float64)
    n_px = float(frames.shape[1] * frames.shape[2])
    for i, f in enumerate(frames):
        _, _, dist, counts = validator._label(f, palette)
        for j, q in enumerate(pctls):
            out[i, j] = validator._weighted_pctl(dist, counts, q)
        out[i, -1] = counts[dist > tau].sum() / n_px
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, metavar="RUN_ID")
    ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "base.json"))
    ap.add_argument("--sample", type=int, default=SAMPLE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = config.load(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, knobs = load_run(args.run, cfg, dev)

    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    val_idx, lut_np = data.preload(shards, index, "val", cfg.data["val_fraction"], palette.rgb)
    lut = torch.from_numpy(lut_np).to(dev).float()
    tau = cfg.validator["offpalette_tau"]
    h, w = cfg.shapes.image_size

    print(f"run {args.run} ({knobs['levels']}, attention={knobs['attention']}), "
          f"{h}x{w}, {len(val_idx):,} held-out frames, device {dev}")
    print(f"quantile ladder {PCTLS} - on {h * w:,} px those are ranks "
          f"{[int(round((1 - q) * h * w)) for q in PCTLS]} from the worst\n")

    # --- the clean side: the whole val split, because the threshold is a MAX ---
    # A max over a subsample is systematically smaller than a max over the whole
    # split, so calibrating on a sample would set a threshold the gate then fails.
    t0 = time.time()
    clean = reconstruct(model, val_idx, lut, np.arange(len(val_idx)))
    clean_q = frame_stats(clean, palette, PCTLS, tau)
    thresholds = clean_q.max(axis=0)
    print(f"clean reconstructions, all {len(clean):,} held-out frames ({time.time() - t0:.0f}s)")
    for j, name in enumerate(COLS):
        col = clean_q[:, j]
        print(f"  {name:<12} max {thresholds[j]:8.4f}   median {np.median(col):8.4f}   "
              f"frame-p99 {np.quantile(col, 0.99):8.4f}   "
              f"max/median {thresholds[j] / max(np.median(col), 1e-9):.2f}x")

    # --- the decay side, at exactly those thresholds ---
    rng = np.random.default_rng(args.seed)
    take = rng.choice(len(val_idx), size=min(args.sample, len(val_idx)), replace=False)
    take.sort()
    base = clean[take]
    print(f"\ndetection at zero false positives, {len(base):,} frames per proxy")
    print(f"  {'proxy':<9}" + "".join(f"  {n:>12}" for n in COLS))
    detection: dict[str, dict[str, float]] = {}
    for kind in PROXIES:
        bad = decay(base, kind, np.random.default_rng(args.seed))
        bad_q = frame_stats(bad, palette, PCTLS, tau)
        hits = (bad_q > thresholds[None, :]).mean(axis=0)
        detection[kind] = {n: float(r) for n, r in zip(COLS, hits)}
        print(f"  {kind:<9}" + "".join(f"  {r:>12.1%}" for r in hits))

    # The pick is by the WORST proxy, not the mean. A verdict that catches three
    # decay modes and misses the fourth is a verdict a diverging model walks
    # straight through, and averaging hides exactly that.
    worst = [min(detection[k][n] for k in PALETTE_PROXIES) for n in COLS]
    best = int(np.argmax(worst))
    print(f"\n  {'WORST*':<9}" + "".join(f"  {r:>12.1%}" for r in worst))
    print(f"  * over {PALETTE_PROXIES}; grey is excluded - see PALETTE_PROXIES")
    print(f"\npick: {COLS[best]}, threshold {thresholds[best]:.4f}, "
          f"worst-proxy detection {worst[best]:.1%}")
    if best < len(PCTLS) and best in (0, len(PCTLS) - 1):
        print("  ** the optimum sits at a LADDER EDGE among the quantiles - extend the "
              "ladder before trusting it")

    print("\n--- as JSON for runs.jsonl ---")
    print(json.dumps({
        "run": args.run,
        "resolution": [h, w],
        "val_frames": int(len(val_idx)),
        "decay_frames": int(len(base)),
        "pctl_ladder": list(PCTLS),
        "statistics": list(COLS),
        "clean_max": {n: float(t) for n, t in zip(COLS, thresholds)},
        "clean_median": {n: float(np.median(clean_q[:, j])) for j, n in enumerate(COLS)},
        "detection_at_zero_fp": detection,
        "worst_proxy_detection": {n: float(r) for n, r in zip(COLS, worst)},
        "worst_over": list(PALETTE_PROXIES),
        "pick": COLS[best],
        "pick_threshold": float(thresholds[best]),
    }, indent=1))
    print("\npalette pctl probe ok")


if __name__ == "__main__":
    main()
