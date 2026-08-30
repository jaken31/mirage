"""Can Q-3's coherence horizon see a dynamics failure at all?

Q-3 is "frames until the F-9 validator fails", accepted at >= 200.  F-9's
decoder-output verdict is a **per-frame palette-adherence** test - off-palette
share above `validator.offpalette_frac_max` at `validator.offpalette_tau` - and
it never compares a frame to a reference.  So the question this probe asks is:
if a rollout drifted to a completely wrong but perfectly *plausible* frame,
would F-9 notice?

The substitution makes that concrete without training a dynamics model.  Take
the reconstruction of frame `t + lag` and offer it as the prediction for frame
`t`.  That is the worst dynamics failure available - the arm is somewhere else
entirely - while remaining a real decoder output, exactly what a well-trained
model would emit after drifting.  If the fire rate is unchanged, Q-3 terminates
on decoder artifacts and nothing else, and its 200-frame horizon is not a
statement about dynamics.

Decoding by round-tripping frame `t + lag`'s pixels rather than by decoding its
cached token row: encoding is deterministic at a pinned batch (gate row 5), so
the two are the same frame, and this path reuses `fsq_eval.reconstruct` instead
of adding a second decode implementation.

**Two controls, because a bare 0.0% is indistinguishable from a broken verdict.**

1. *The verdict fires on something.*  The same expression is run on a noised
   reconstruction (sigma 16), which `bench/palette_pctl_probe.py` measured F-9
   catching at 100%.  If that does not fire here, the probe is wrong, not Q-3.
2. *The substituted frames really are different.*  Reports the share of pixels
   that differ between the ground truth at `t` and at `t + lag`.  A near-zero
   fire rate on near-identical frames would prove nothing.

    python bench/q3_blind_probe.py 20260829-005439-r1 [--lag 300]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import config, data, fsq_eval, validator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAG = 300         # "300 steps later" - half an episode at steps_per_episode 600
STRIDE = 10       # every 10th start, so the pairs are not 600 near-copies
NOISE_SIGMA = 16  # the corruption F-9 was measured catching at 100%


def _fires(frames: np.ndarray, palette: validator.Palette, tau: float,
           frac_max: float) -> np.ndarray:
    """F-9's decoder-output verdict, per frame. One expression, used three times."""
    return np.array([
        validator.measure_pixels_only(f, palette, tau).offpalette_frac > frac_max
        for f in frames
    ])


def probe(run_id: str, cfg: config.Config, lag: int = LAG,
          device: str | None = None) -> dict:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = fsq_eval.load_run(run_id, cfg, dev)
    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    val_idx, lut_np = data.preload(shards, index, "val", cfg.data["val_fraction"], palette.rgb)
    lut = torch.from_numpy(lut_np).to(dev).float()
    tau = cfg.validator["offpalette_tau"]
    frac_max = cfg.validator["offpalette_frac_max"]

    # Row offsets of each val episode inside val_idx - preload concatenates
    # split_episodes in index order, so the offsets are a running sum of lengths.
    eps = data.split_episodes(index, "val", cfg.data["val_fraction"])
    offsets = np.concatenate(([0], np.cumsum([e.length for e in eps])))
    assert offsets[-1] == len(val_idx), "episode lengths do not sum to the val split"

    now, later = [], []
    for e, off in zip(eps, offsets[:-1]):
        for t in range(0, e.length - lag, STRIDE):
            now.append(off + t)
            later.append(off + t + lag)
    now, later = np.array(now), np.array(later)

    recon_now = fsq_eval.reconstruct(model, val_idx, lut, now)
    recon_later = fsq_eval.reconstruct(model, val_idx, lut, later)
    lut8 = lut.round().clamp(0, fsq_eval.PEAK).byte().cpu().numpy()
    truth_now, truth_later = lut8[val_idx[now]], lut8[val_idx[later]]

    rng = np.random.default_rng(0)
    noised = np.clip(
        recon_now.astype(np.int16) + rng.normal(0, NOISE_SIGMA, recon_now.shape),
        0, 255).astype(np.uint8)

    fire_now = _fires(recon_now, palette, tau, frac_max)
    fire_later = _fires(recon_later, palette, tau, frac_max)
    fire_noise = _fires(noised, palette, tau, frac_max)

    return {
        "run_id": run_id,
        "lag": lag,
        "pairs": int(len(now)),
        "episodes": len(eps),
        "tau": tau,
        "offpalette_frac_max": frac_max,
        "fire_rate_correct_frame": float(fire_now.mean()),
        "fire_rate_substituted_frame": float(fire_later.mean()),
        "fire_rate_noise_control": float(fire_noise.mean()),
        "pixels_differing_share": float(
            (truth_now != truth_later).any(-1).mean()),
        "psnr_db_substituted_vs_truth": float(fsq_eval.psnr_db(
            float(((truth_later.astype(np.float64) - truth_now) ** 2).sum()),
            truth_now.size)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_id")
    ap.add_argument("--lag", type=int, default=LAG)
    a = ap.parse_args()
    cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
    d = probe(a.run_id, cfg, lag=a.lag)
    print(f"{d['run_id']}: {d['pairs']:,} pairs over {d['episodes']} held-out episodes, "
          f"lag {d['lag']} frames")
    print(f"verdict: offpalette share > {d['offpalette_frac_max']:.5%} at tau {d['tau']}\n")
    print(f"  F-9 fires on the correct reconstruction      {d['fire_rate_correct_frame']:8.2%}")
    print(f"  F-9 fires on the frame {d['lag']} steps later        "
          f"{d['fire_rate_substituted_frame']:8.2%}   <- the dynamics failure")
    print(f"  F-9 fires on a sigma-{NOISE_SIGMA} noised reconstruction "
          f"{d['fire_rate_noise_control']:8.2%}   <- control: the verdict works")
    print(f"\n  the substituted frames differ from the truth on "
          f"{d['pixels_differing_share']:.1%} of pixels, "
          f"{d['psnr_db_substituted_vs_truth']:.2f} dB")


if __name__ == "__main__":
    main()
