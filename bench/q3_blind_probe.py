"""Can a validator-based rollout horizon detect a dynamics failure at all?

The original rollout quality measure was "frames until the validator fails",
with a target of at least 200. On decoder output the validator only checks
**each frame's colours against the palette** (off-palette share above
`validator.offpalette_frac_max` at `validator.offpalette_tau`) and never
compares a frame with the right answer. So: if a rollout drifted to a
completely wrong but perfectly *plausible* frame, would the validator notice?

A substitution tests that without training a dynamics model. Take the
reconstruction of frame `t + lag` and offer it as the prediction for frame
`t`. That is the worst possible dynamics failure (the arm is somewhere else
entirely) while still being real decoder output, exactly what a good model
would produce after drifting. If the failure rate does not change, the horizon
only ever stops on decoder artifacts and says nothing about dynamics.

Frame `t + lag` is re-encoded and decoded from its pixels rather than decoded
from the cached tokens. Encoding is deterministic at a fixed batch size, so
both give the same frame, and this reuses `fsq_eval.reconstruct` instead of a
second decode path.

**Two controls, because a plain 0.0% looks the same as a broken check.**

1. *The check fires on something.* The same rule runs on a noised
   reconstruction (sigma 16), which `bench/palette_pctl_probe.py` found the
   validator catches 100% of the time. If that does not fire here, the probe
   is broken, not the measure.
2. *The substituted frames really differ.* Reports the share of pixels that
   differ between the ground truth at `t` and at `t + lag`. A near-zero rate
   on near-identical frames would prove nothing.

**It is also the regression test for the check that replaced the validator**
(Phase 2 item 6): the frame-to-frame continuity check on `link_angle`,
`link_extent` and each block's bbox centre, with the bounds in the `validator`
config section. The requirement asks it to fire on 100% of the substitutions and
0% of clean frames. That half scores the val split's **cached** token rows
decoded through `Tokenizer.decode`, not `reconstruct`'s output, because the two
encoder input layouts disagree on 0.33% of R1's tokens and a rollout produces
cache-like tokens (Phase 2 item 5). **It reads 53.7%, not 100%**
(`runs.jsonl`, item 6's row): the per-step noise of those features on decoded
frames is larger than most 300-step jumps, so gate row 4 is reported, not
pass/fail, until the coherence-horizon requirement is restated.

    python bench/q3_blind_probe.py 20260829-005439-r1 [--lag 300]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import config, data, dynamics, dynamics_eval, fsq_eval, validator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAG = 300         # "300 steps later": half an episode at steps_per_episode 600
STRIDE = 10       # every 10th start, so the pairs are not 600 near-copies
NOISE_SIGMA = 16  # noise the validator was measured catching 100% of the time


def _fires(frames: np.ndarray, palette: validator.Palette, tau: float,
           frac_max: float) -> np.ndarray:
    """The validator's decoder-output pass/fail rule, per frame. One rule, used three times."""
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

    # Where each validation episode starts in val_idx. preload joins
    # split_episodes in order, so the offsets are a running sum of lengths.
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


def continuity(cfg: config.Config, lag: int = LAG, device: str | None = None) -> dict:
    """The continuity check's fire rates on substituted and clean decoded cached rows."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    splits = dynamics.load_splits(cfg)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    eps = dynamics_eval.val_episodes(splits, len(palette.blocks))
    truth = dynamics_eval.truth_features(cfg, splits, eps, palette, dev)
    return dynamics_eval.substitution_test(
        truth, dynamics_eval.continuity_bounds(cfg, palette), lag, STRIDE)


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

    c = continuity(cfg, lag=a.lag)
    print(f"\ncontinuity check (validator.continuity_*), decoded cached rows, {c['pairs']:,} pairs:")
    print(f"  fires on the frame {c['lag']} steps later   {c['fire_substituted']:8.2%}   <- needs 100%")
    print(f"  fires on the correct frame             {c['fire_clean']:8.2%}   <- needs 0%")


if __name__ == "__main__":
    main()
