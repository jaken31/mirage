"""Everything that runs against a finished dynamics checkpoint.

The split from `mirage/dynamics.py` is by when code runs, the rule that split
`fsq_eval.py` out of `fsq.py`: that file builds and trains, this one reads a
trained run. So far it holds item 4's last step - scoring the best checkpoint
on the full population. Rollout and the gate table (item 6) start here too.

**The full population** is every val window's last frame: the 27 val episodes
at frames 15..599, 15,795 windows, 1,010,880 cells. It is the population
`bench/token_stability_probe.py --episodes all --first-target 15` reads 86.69%
persistence on (r54, reproduced in r57), and `score_run` re-measures it there
by calling that probe and asserting it counts the same cells, rather than
quoting the figure. The probe's default 12-episode population reads 85.67%
(r46), which is a different statistic, and never the bar here.

Scored beside gate row 1's measure:

- the marginal-frequency baseline - always guess the most common code in the
  train split's frames - which the dynamics requirement reports alongside;
- row 10's copy overlap, the share of predictions persistence also makes;
- the **false-flip rate on static cells**: of the cells whose token did not
  change and whose own 15x15 pixel field did not change either, the share the
  model predicts as changed. The field is `token_stability_probe`'s, and r59
  found the mask measurement's model losing to copying there.

    python -m mirage.dynamics_eval            # self-check
"""

import math
from pathlib import Path

import numpy as np
import torch

from mirage import config, dynamics

ROOT = Path(__file__).resolve().parent.parent


def population(cfg: config.Config, splits: dynamics.TokenSplits,
               lag: int = 1) -> dict[str, np.ndarray]:
    """Every val window's last frame, per cell: `target`, `prev`, and `quiet`.

    `quiet` is True where no pixel of the cell's own 15x15 field changed between
    the previous frame and the target, `token_stability_probe._field_changed` on
    the episode's frames (flipped right side up, as the token cache is). Window
    `i` at offset `o` targets frame `o + ctx`, which is transition `o + ctx - 1`
    of its episode. `lag` other than 1 reads the fields that many transitions
    early instead, the self-check's deliberate error.
    """
    if splits.shards is None:
        raise ValueError("the full population needs the shards' pixels")
    probe = dynamics._bench_module("token_stability_probe")
    val, ctx = splits.val, cfg.data["ctx"]
    grid = tuple(cfg.shapes.token_grid)
    cells = math.prod(grid)
    changed = {}
    for ep in val.windows.episodes:
        px = np.ascontiguousarray(
            splits.shards[ep.shard].pixels[ep.start:ep.start + ep.length, ::-1])
        changed[ep.episode_id] = probe._field_changed(px, grid).reshape(ep.length - 1, cells)
    n = len(val)
    out = {k: np.empty((n, cells), np.int64) for k in ("target", "prev")}
    out["quiet"] = np.empty((n, cells), bool)
    for i in range(n):
        w = val[i]
        out["target"][i], out["prev"][i] = w.tokens[-1], w.tokens[-2]
        out["quiet"][i] = ~changed[w.episode_id][w.offset + ctx - lag]
    return out


def probe_agrees(pop: dict[str, np.ndarray], probe: dict) -> bool:
    """True if `token_stability_probe.probe` counted exactly `pop`'s cells: the
    same transitions, persistence, quiet-field share, and flip rate on quiet fields."""
    flip, quiet = pop["target"] != pop["prev"], pop["quiet"]
    return (probe["transitions"] == flip.size
            and abs(probe["persistence"] - (1 - float(flip.mean()))) < 1e-12
            and abs(probe["quiet_field_share"] - float(quiet.mean())) < 1e-12
            and abs(probe["p_flip_given_quiet_field"] - float((flip & quiet).sum() / quiet.sum())) < 1e-12)


def marginal_top1(splits: dynamics.TokenSplits, codes: int) -> int:
    """The most common code over the train split's frames."""
    counts = np.zeros(codes, np.int64)
    for ep in splits.train.windows.episodes:
        counts += np.bincount(splits.train.tokens[ep.shard][ep.start:ep.start + ep.length].ravel(),
                              minlength=codes)
    return int(counts.argmax())


def summary(pred: np.ndarray, pop: dict[str, np.ndarray], top1: int) -> dict:
    """Gate row 1's measure against persistence and the marginal baseline, copy
    overlap, and the false-flip rate on static cells."""
    target, prev = pop["target"], pop["prev"]
    static = pop["quiet"] & (target == prev)
    flips = (pred != prev) & static
    return {**dynamics.last_frame_summary(pred, target, prev),
            "marginal_top1_acc": float((target == top1).mean()),
            "static_share": float(static.mean()),
            "false_flip_static": float(flips.sum() / static.sum()),
            "cells_lost_static": int(flips.sum()),
            "frame_exact": float((pred == target).all(1).mean())}


def score_run(run_dir: Path | str, cfg: config.Config, splits: dynamics.TokenSplits,
              dev: torch.device) -> dict:
    """Score a run's `best.pt` and its last `model.pt` on the full population.

    Persistence is re-measured by `bench/token_stability_probe.py` at `--episodes
    all --first-target <ctx>`, and refused unless the probe counted exactly the
    cells scored here (`probe_agrees`) - which also shows the pixel fields line
    up with the right transitions.
    """
    run_dir = Path(run_dir)
    pop = population(cfg, splits)
    ctx, codes = cfg.data["ctx"], cfg.tokenizer["codebook_size"]
    probe = dynamics._bench_module("token_stability_probe").probe(
        dynamics.TOKENIZER_RUN, cfg, None, ctx)
    persistence = float((pop["target"] == pop["prev"]).mean())
    if not probe_agrees(pop, probe):
        raise ValueError(f"population disagrees with token_stability_probe: {probe}")
    top1 = marginal_top1(splits, codes)
    out = {"population": {"windows": len(splits.val), "episodes": len(splits.val.windows.episodes),
                          "first_target": ctx, "cells": int(pop["target"].size),
                          "persistence": persistence, "probe": probe,
                          "probe_cmd": f"bench/token_stability_probe.py --episodes all "
                                       f"--first-target {ctx}",
                          "marginal_top1_code": top1}}

    lay = dynamics.layout(ctx, math.prod(cfg.shapes.token_grid))
    mask = dynamics.attention_mask(lay).to(dev)
    every = np.arange(len(splits.val))
    for name, file in (("best", "best.pt"), ("final", "model.pt")):
        ck = torch.load(run_dir / file, map_location="cpu", weights_only=True)
        for k in ("data_hash", "tokenizer_hash", "dynamics_hash"):
            if ck[k] != getattr(cfg, k):
                raise ValueError(f"{run_dir / file}: {k} is {ck[k]}, the config's is "
                                 f"{getattr(cfg, k)}")
        model = dynamics.build(cfg).to(dev)
        model.load_state_dict(ck["state_dict"])
        got = dynamics.score_windows(model, splits.val, every, lay, mask, dev)
        assert np.array_equal(got["target"], pop["target"]) and \
            np.array_equal(got["prev"], pop["prev"]), "scored other cells than the population"
        out[name] = {"step": ck["step"], "epoch": ck["epoch"], **summary(got["pred"], pop, top1)}
        s = out[name]
        print(f"  {name} (step {ck['step']:,}): acc {s['acc']:.4%}  persistence "
              f"{persistence:.4%}  {s['margin_points']:+.2f} pts ({s['margin_cells']:+,} cells)  "
              f"marginal {s['marginal_top1_acc']:.2%}  copy overlap {s['copy_overlap']:.2%}  "
              f"false flips on static {s['false_flip_static']:.2%}", flush=True)
    return out


def _self_check() -> None:
    """The static-cell arithmetic on a hand-built case, and the population's
    alignment with `token_stability_probe` when the generated set is present."""
    # 1 window, 4 cells: cell 0 static and copied, cell 1 static and flipped,
    # cell 2 changed token, cell 3 same token but its pixels changed.
    pop = {"target": np.array([[5, 5, 7, 9]]), "prev": np.array([[5, 5, 6, 9]]),
           "quiet": np.array([[True, True, True, False]])}
    s = summary(np.array([[5, 4, 7, 9]]), pop, top1=5)
    assert s["static_share"] == 0.5 and s["false_flip_static"] == 0.5 and s["cells_lost_static"] == 1
    assert s["acc"] == 0.75 and s["persistence"] == 0.75 and s["margin_cells"] == 0
    assert s["copy_overlap"] == 0.5 and s["marginal_top1_acc"] == 0.5 and s["frame_exact"] == 0.0
    print("summary: static cells, false flips, copy overlap and the marginal baseline by hand")

    cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
    real = ROOT / "runs" / dynamics.TOKENIZER_RUN / "tokens" / "manifest.json"
    if not (ROOT / cfg.data["shard_dir"]).exists() or not real.exists():
        print("population: skipped - no generated set or no R1 token cache")
        print("dynamics_eval self-check ok (no data)")
        return
    splits = dynamics.load_splits(cfg)
    pop = population(cfg, splits)
    probe = dynamics._bench_module("token_stability_probe").probe(
        dynamics.TOKENIZER_RUN, cfg, None, cfg.data["ctx"])
    persistence = float((pop["target"] == pop["prev"]).mean())
    assert probe_agrees(pop, probe), probe
    # The control: pixel fields read one transition early must disagree.
    early = population(cfg, splits, lag=2)
    assert not probe_agrees(early, probe), "a one-transition shift went unseen"
    flip = early["target"] != early["prev"]
    early_rate = float((flip & early["quiet"]).sum() / early["quiet"].sum())
    print(f"population: {len(splits.val):,} val windows, {pop['target'].size:,} cells; "
          f"persistence {persistence:.4%}, quiet-field share {probe['quiet_field_share']:.4%} "
          f"and flips on quiet fields {probe['p_flip_given_quiet_field']:.4%}, all "
          f"token_stability_probe's exactly; fields one transition early read "
          f"{early_rate:.4%} flips and fail")
    print("dynamics_eval self-check ok")


if __name__ == "__main__":
    _self_check()
