"""Are a run's tokens stable over time, and do they change for no local reason?

Two numbers per run, over held-out episodes, read from files already on disk
(the token cache and the shard pixels). No GPU, no training.

**persistence** - the share of frame-to-frame steps where a cell's token is the
same as in the previous frame. This is the "just copy the last frame" baseline
the dynamics model must beat. The old accuracy target (3x the rate of always
guessing the most common token) is far below it: copying the previous frame,
with zero parameters, already beats that target easily.

**spurious flip rate** - the chance a token changes when not one pixel in its
own 15x15 input window changed. With `GroupNorm`, the encoder's statistics
cover the whole image, so a cell's output depends on pixels outside its window,
and this reads well above zero. Rung `r1c` switches to per-pixel normalisation;
if that is the cause, r1c must read about 0 here. That is the test.

The 15x15 window: three stride-2, pad-1, 3x3 convs map latent row i to input
rows [8*i - 7, 8*i + 7], clipped at the frame edge.

**Which frames are scored is an option.** The baseline must be measured on the
same frames the model is scored on, so a model scored on other frames needs
the baseline re-measured here, not an old figure quoted. `--episodes` takes the
first N validation episodes in order, or `all`. `--first-target` skips steps
whose *target* frame comes before that step of its episode; 15 keeps only
frames with a full 15 frames of history. The defaults, 12 and 1, match the
original measurement recorded in runs.jsonl and must keep reproducing it
exactly.

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
EPISODES = 12  # what the original measurement used
RF = 15        # the real conv window size. bench/patch_probe.py's RF = 22 is wrong.


def _field_changed(px: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """(T, h, w, 3) uint8 -> (T-1, gh, gw) bool: did any pixel in this cell's 15x15 window change?

    Uses a summed-area table over the changed-pixel mask, so the 64 windows per
    frame cost one cumulative sum instead of 64 slices.
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
    """`episodes=None` means every validation episode. `first_target` is the
    first frame of an episode scored as a step's target, so 1 scores every step."""
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
        # ::-1 to match write_token_cache: frames are stored bottom-up.
        px = np.ascontiguousarray(sh.pixels[ep.start:ep.start + ep.length, ::-1])
        # Step k has frame k + 1 as its target.
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
