"""Try calibrating the validator's palette check as a *distance quantile* instead of a pixel count.

    python bench/palette_pctl_probe.py --run 20260829-005439-r1
    python bench/palette_pctl_probe.py --run <96x96 run> --config mirage/configs/base96.json

Reads only the run's `model.pt` and the shards, and writes nothing. Results are
copied into a `runs.jsonl` row by hand, like every other probe in `bench/`.

Outcome: the quantile lost. The off-palette *fraction* (also measured here)
became the validator's pass/fail number; see
`validator.measure_pixels_only`.

**Why this probe exists.** The palette check was first calibrated as
`offpalette_px > N` (a count of pixels further than tau from every palette
colour), with N = 350 at 64x64. A count depends on resolution, so 96x96 needed
its own N, and simply scaling by frame area (350 * 2.25 = 788) is wrong by the
project's own evidence: 96% of squared error is at edges, and edge length
grows 1.5x where area grows 2.25x. A quantile of per-pixel distance does not
depend on frame size, so one threshold would serve every resolution.

**The method is the original calibration's, unchanged**, so the two are
comparable:

  1. set the threshold at the maximum over *clean* held-out reconstructions,
     which gives zero false alarms by construction;
  2. measure what fraction of damaged frames each setting then catches;
  3. take the best setting that is not at either end of the range.

The four kinds of damage are also the original four, because a failing world
model produces mush, and these are the shapes mush takes:

  * `blend`  - a 50/50 average of two possible futures, what a model does when
               it cannot commit. The most realistic and hardest to catch.
  * `blur3`  - a 3x3 box blur applied 4 times: lost detail, no colour drift.
  * `noise16`- gaussian noise with sigma 16, the high-frequency failure.
  * `grey`   - collapse to uniform grey, the extreme case.

**What a quantile gains and loses.** It gains resolution independence, the
whole point. It loses on the extreme pixels: at p99.9 on a 64x64 frame it is
the 4th-worst pixel of 4,096, nearly the maximum, and just as jumpy. At p99 it
is the 41st-worst, much steadier but blind to a small bright fault. The range
of quantiles tried below decides, and the best is expected to be in the middle
for that reason.
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

# The quantiles tried. From "41st-worst pixel of 4,096" to "2nd-worst": the
# range where the statistic goes from almost a maximum to a summary.
PCTLS = (0.99, 0.995, 0.999, 0.9995)
PROXIES = ("blend", "blur3", "noise16", "grey")
# `grey` is measured but **excluded from the choice**, and that is a finding,
# not a shortcut. A frame collapsed to its mean colour lands about 22 RGB units
# from the table colour, inside any usable tau, so uniform grey *looks
# on-palette* and no palette statistic can see it, the count included. The
# validator's object checks catch it instead: every block's `px_count` drops to
# zero. Kept in the table so nobody has to rediscover this.
PALETTE_PROXIES = ("blend", "blur3", "noise16")
# The candidate pass/fail numbers, in one table so they are compared fairly:
# the quantiles, then the off-palette *fraction* at tau.
COLS = tuple(f"p{q:.4%}" for q in PCTLS) + ("offpal_frac",)
SAMPLE = 4000  # frames per kind of damage; the clean side always uses the whole split


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
    """Apply one kind of damage to a (n, h, w, 3) uint8 batch. Returns uint8."""
    x = clean.astype(np.float64)
    if kind == "blend":
        # Pair each frame with a *different* frame, never itself: blending a
        # frame with itself changes nothing and would measure nothing.
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
    *fraction* at `tau` (the pixel count divided by the frame's pixels, so it no
    longer depends on resolution).

    Both in one pass, because `_label`'s nearest-colour matching is the expensive
    part. Measuring them together also keeps the comparison fair: same frames,
    same decode.

    They answer different questions, which is the main finding. A quantile asks
    *how far off* the worst pixels are. A fraction asks *how much* of the frame is
    off. Damage spread over the whole frame (grey collapse, added noise) changes
    many pixels a little, so the worst-pixel view misses it.
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

    # --- clean frames: the whole validation split, because the threshold is a MAX ---
    # A max over a sample is usually smaller than over the whole split, so
    # calibrating on a sample would set a threshold the gate then fails.
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

    # --- damaged frames, at exactly those thresholds ---
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

    # Choose by the WORST kind of damage, not the average. A check that catches
    # three kinds and misses the fourth lets a failing model walk straight
    # through, and averaging hides exactly that.
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
