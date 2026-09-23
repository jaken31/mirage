"""Are a run's tokens stable in time, and do they flip for no local reason?

Two numbers per run, both over held-out episodes and both read off artifacts
already on disk - the token cache and the shard pixels. No GPU, no training.

**persistence** - the share of cell-transitions where a token equals the token
at the same cell in the previous frame. This is the baseline F-11's "3x the
marginal top-1" bar has to beat and does not: a zero-parameter copy of the
previous frame scores far above it.

**spurious flip rate** - P(a token flips | not one pixel of its own 15x15
convolutional receptive field changed). Under `GroupNorm` the encoder's
statistics span the whole feature map, so a cell's output depends on pixels it
never convolved with, and this reads well above zero. Rung `r1c` replaces that
normalisation with a per-pixel one; if the mechanism is what this measures, r1c
must read ~0 here. That is the falsifier.

The 15x15 window: three stride-2, pad-1, 3x3 convs compose to input rows
[8*i - 7, 8*i + 7] for latent row i, clipped at the frame edge.

**The population is an option, and the default is r46's.** F-11 scores the
baseline like-for-like on the population the model is scored on, so a model
scored elsewhere needs the baseline re-measured here, not r46's figure quoted.
`--episodes` takes the first N val episodes in index order, or `all`;
`--first-target` drops transitions whose *target* frame sits before that step of
its episode - 15 keeps only frames with a full `ctx` 15 of history before them.
The defaults, 12 and 1, are the population r46 measured, and must keep
reproducing its figures exactly.

    python bench/token_stability_probe.py 20260829-005439-r1 [more run ids...]
    python bench/token_stability_probe.py 20260829-005439-r1 --episodes all --first-target 15
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import config, data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPISODES = 12  # what the finding this reproduces used
RF = 15        # the true conv field. bench/patch_probe.py's RF = 22 is wrong.


def _field_changed(px: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """(T, h, w, 3) uint8 -> (T-1, gh, gw) bool: did this cell's 15x15 field move?

    Integral image over the changed-pixel mask, so the 64 windows per frame cost
    one cumsum rather than 64 slices.
    """
    changed = (px[1:] != px[:-1]).any(-1).astype(np.int32)
    s = np.zeros((len(changed), px.shape[1] + 1, px.shape[2] + 1), np.int32)
    s[:, 1:, 1:] = changed.cumsum(1).cumsum(2)
    h, w = px.shape[1], px.shape[2]
    r = np.array([(max(0, 8 * i - RF // 2), min(h, 8 * i + RF // 2 + 1))
                  for i in range(grid[0])])
    c = np.array([(max(0, 8 * j - RF // 2), min(w, 8 * j + RF // 2 + 1))
                  for j in range(grid[1])])
    r0, r1 = r[:, 0][:, None], r[:, 1][:, None]
    c0, c1 = c[None, :, 0], c[None, :, 1]
    tot = (s[:, r1, c1] - s[:, r0, c1] - s[:, r1, c0] + s[:, r0, c0])
    return tot > 0


def probe(run_id: str, cfg: config.Config, episodes: int | None = EPISODES,
          first_target: int = 1) -> dict:
    """`episodes=None` is every val episode. `first_target` is the first frame
    of an episode scored as a transition's target, so 1 is every transition."""
    if first_target < 1:
        raise ValueError(f"first_target is {first_target}; frame 0 has no previous frame")
    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    val = data.split_episodes(index, "val", cfg.data["val_fraction"])
    if episodes is not None:
        assert len(val) >= episodes, f"only {len(val)} val episodes available"
        val = val[:episodes]
    grid = tuple(cfg.shapes.token_grid)
    tok_dir = ROOT / "runs" / run_id / "tokens"

    flips = still = spurious = trans = 0
    for ep in val:
        sh = shards[ep.shard]
        toks = np.load(tok_dir / f"shard_{sh.index:03d}.npy")
        assert len(toks) == sh.frames, f"shard {sh.index}: token rows != frames"
        t = toks[ep.start:ep.start + ep.length].astype(np.int32)
        # ::-1 to match write_token_cache: the blob holds rows bottom-up.
        px = np.ascontiguousarray(sh.pixels[ep.start:ep.start + ep.length, ::-1])
        # Transition k has frame k + 1 as its target.
        f = (t[1:] != t[:-1])[first_target - 1:]
        quiet = ~_field_changed(px, grid)[first_target - 1:]
        flips += int(f.sum())
        trans += f.size
        still += int(quiet.sum())
        spurious += int((f & quiet).sum())

    return {
        "run_id": run_id,
        "episodes": len(val),
        "first_target": first_target,
        "transitions": trans,
        "persistence": 1 - flips / trans,
        "p_flip_given_quiet_field": spurious / still,
        "spurious_share_of_flips": spurious / flips,
        "quiet_field_share": still / trans,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="RUN_ID")
    ap.add_argument("--episodes", default=str(EPISODES),
                    help=f"first N val episodes, or 'all' (default {EPISODES}, r46's)")
    ap.add_argument("--first-target", type=int, default=1,
                    help="first frame of an episode scored as a target (default 1, r46's)")
    a = ap.parse_args()
    episodes = None if a.episodes == "all" else int(a.episodes)
    cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
    print(f"population: {a.episodes} val episodes, targets from frame {a.first_target}")
    print(f"{'run':<24} {'transitions':>12} {'persistence':>12} "
          f"{'P(flip|quiet)':>14} {'spurious share':>15}")
    for r in a.runs:
        d = probe(r, cfg, episodes, a.first_target)
        print(f"{d['run_id']:<24} {d['transitions']:>12,} "
              f"{d['persistence']:>11.2%} {d['p_flip_given_quiet_field']:>14.2%} "
              f"{d['spurious_share_of_flips']:>15.2%}")


if __name__ == "__main__":
    main()
