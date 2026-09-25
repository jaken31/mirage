"""Everything that runs against a finished dynamics checkpoint: rollout and the gate.

The split from `mirage/dynamics.py` is by when code runs, the rule that split
`fsq_eval.py` out of `fsq.py`: that file builds and trains, this one reads a
trained run. It holds item 4's last step - scoring the best checkpoint on the
full population - and item 6, the rollout and the gate table.

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

**The rollout** (`rollout`) is greedy (decision 5): a seed clip of `ctx`
frames from a val episode, then one frame per step from the episode's own
action column, every step one forward pass over all 64 cells, a fixed step
count, no early exit and no fallback. The context length is an argument, so
one checkpoint rolls out at 4, 8 and 15. The gate runs one rollout per val
episode, from frame 0 to the episode's end.

**Every pixel row decodes cached-style tokens through `Tokenizer.decode`**,
the ground truth included - never `fsq_eval.reconstruct`'s output, whose
encoder input layout flips 0.33% of R1's tokens (item 5).

**Rows 4 and 5 are reported, not pass/fail** (decided 2026-09-25). The
continuity check behind row 4 is calibrated on decoded ground truth so that
none of it fires, and then fires on only 53.7% of `bench/q3_blind_probe.py`'s
300-step substitutions, against the 100% the coherence-horizon requirement
asks; the per-step noise of its features on decoded frames swamps the motion.
Row 5's pixel-measured per-step angle sign is at chance on the ground truth
itself. Both instruments are built as specified and print that beside them.

    python -m mirage.dynamics --eval RUN_ID                 # the gate table; exits 1 on a miss
    python -m mirage.dynamics_eval                          # self-check
    python -m mirage.dynamics_eval --calibrate-continuity   # the continuity bounds, once
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from mirage import config, data, dynamics, fsq_eval, validator
from mirage.fsq import PEAK

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


def load_checkpoint(path: Path, cfg: config.Config,
                    dev: torch.device) -> tuple[dynamics.Dynamics, dict]:
    """A dynamics checkpoint as an eval-mode model, plus the checkpoint dict.

    Refused unless its `data_hash`, `tokenizer_hash` and `dynamics_hash` are the
    config's, the way `fsq_eval.load_run` refuses a tokenizer checkpoint.
    """
    ck = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("data_hash", "tokenizer_hash", "dynamics_hash"):
        if ck[k] != getattr(cfg, k):
            raise ValueError(f"{path}: {k} is {ck[k]}, the config's is {getattr(cfg, k)}")
    model = dynamics.build(cfg).to(dev)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def score_run(run_dir: Path | str, cfg: config.Config, splits: dynamics.TokenSplits,
              dev: torch.device, keep_pred: bool = False) -> dict:
    """Score a run's `best.pt` and its last `model.pt` on the full population.

    Persistence is re-measured by `bench/token_stability_probe.py` at `--episodes
    all --first-target <ctx>`, and refused unless the probe counted exactly the
    cells scored here (`probe_agrees`) - which also shows the pixel fields line
    up with the right transitions. `keep_pred` also returns each checkpoint's
    predictions, `(windows, cells)`, under `pred`.
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
        model, ck = load_checkpoint(run_dir / file, cfg, dev)
        got = dynamics.score_windows(model, splits.val, every, lay, mask, dev)
        assert np.array_equal(got["target"], pop["target"]) and \
            np.array_equal(got["prev"], pop["prev"]), "scored other cells than the population"
        out[name] = {"step": ck["step"], "epoch": ck["epoch"], **summary(got["pred"], pop, top1)}
        s = out[name]
        print(f"  {name} (step {ck['step']:,}): acc {s['acc']:.4%}  persistence "
              f"{persistence:.4%}  {s['margin_points']:+.2f} pts ({s['margin_cells']:+,} cells)  "
              f"marginal {s['marginal_top1_acc']:.2%}  copy overlap {s['copy_overlap']:.2%}  "
              f"false flips on static {s['false_flip_static']:.2%}", flush=True)
        if keep_pred:
            out.setdefault("pred", {})[name] = got["pred"]
    return out


# -------------------------------------------------------------------- rollout

class Rollout(NamedTuple):
    frames: np.ndarray  # (B, ctx + steps, cells) int64: the seed clip, then every generated frame
    passes: int         # forward passes the model made, one per generated frame


@torch.no_grad()
def rollout(model: dynamics.Dynamics, seed: np.ndarray, actions: np.ndarray, steps: int,
            dev: torch.device) -> Rollout:
    """Greedy rollout (decision 5) from seed clips `(B, ctx, cells)`, `steps` frames long.

    `actions[:, t]` is `action[t]`, read at the record of the clip's frame `t`,
    so it has `ctx + steps` columns. Step `k` builds the `ctx + 1`-frame window
    of frames `k .. ctx + k` - the last of which is the one being predicted, and
    is never in the input (`_self_check` in `mirage.dynamics`) - and takes the
    argmax of the model's logits at that frame's 64 cells. So the frame
    `ctx + k` is predicted from the `ctx` frames before it and `action[ctx + k]`,
    the action that produced it, in one forward pass: all 64 positions every
    step, a fixed step count, no early exit and no other path.

    The context length is the seed's, chosen here rather than in the config
    (item 2): one checkpoint runs at any `ctx` its RoPE tables cover. The whole
    window's read positions are scored, exactly as `dynamics.score_windows`
    scores them, so the first step of a batch equals that pass on the same
    windows at the same batch bit for bit (gate row 2).
    """
    b, ctx, cells = seed.shape
    if actions.shape != (b, ctx + steps):
        raise ValueError(f"actions {actions.shape} for {b} clips of {ctx} frames and {steps} steps")
    codes = model.head.out_features
    lay = dynamics.layout(ctx, cells)
    mask = dynamics.attention_mask(lay).to(dev)
    read = torch.from_numpy(lay.read).to(dev)
    frames = np.zeros((b, ctx + steps, cells), np.int64)
    frames[:, :ctx] = seed
    was = model.training
    model.eval()
    for k in range(steps):
        x, _ = dynamics.assemble(lay, frames[:, k:k + ctx + 1], actions[:, k:k + ctx + 1], codes)
        with dynamics._autocast(dev):
            logits = model(torch.from_numpy(x).to(dev), mask, read)
        frames[:, ctx + k] = logits.float()[:, -cells:].argmax(-1).cpu().numpy()
    model.train(was)
    return Rollout(frames, steps)


class Episodes(NamedTuple):
    """Every val episode whole, from the token cache and the meta records."""
    tokens: np.ndarray    # (E, T, cells) int64, cached rows
    actions: np.ndarray   # (E, T) int64, action[t] at record t
    visible: np.ndarray   # (E, blocks, T) the renderer's visible pixel count per block
    qpos: np.ndarray      # (E, T, joints) the simulator's joint angles
    ids: list[int]        # episode ids, in `data.split_episodes` order
    first: np.ndarray     # (E,) the val window index of each episode's offset-0 window


def val_episodes(splits: dynamics.TokenSplits, blocks: int) -> Episodes:
    """The val split's episodes, all the same length, in the val sampler's order."""
    val = splits.val
    eps = val.windows.episodes
    if len({ep.length for ep in eps}) != 1:
        raise ValueError(f"val episodes of lengths {sorted({ep.length for ep in eps})}; "
                         f"a rollout batch needs one length")
    first = np.concatenate(([0], np.cumsum([ep.length - val.window + 1 for ep in eps])[:-1]))
    for ep, i in zip(eps, first):   # the arithmetic above, checked against the sampler's own
        assert val.windows.locate(int(i)) == (ep, 0), f"window {i} is not episode {ep.episode_id}'s first"
    sl = [slice(ep.start, ep.start + ep.length) for ep in eps]
    meta = [splits.shards[ep.shard].meta[s] for ep, s in zip(eps, sl)]
    return Episodes(np.stack([val.tokens[ep.shard][s] for ep, s in zip(eps, sl)]).astype(np.int64),
                    np.stack([val.actions[ep.shard][s] for ep, s in zip(eps, sl)]).astype(np.int64),
                    np.stack([[m[f"visible_px{b}"] for b in range(blocks)] for m in meta]).astype(np.int64),
                    np.stack([np.stack([m[f"qpos{j}"] for j in range(JOINTS)], -1)
                              for m in meta]).astype(np.float64),
                    [ep.episode_id for ep in eps], first)


# --------------------------------------------------------------------- pixels

# One decode batch for every population the gate measures, so ground truth and
# rollouts go through the same kernels. The token cache's own batch.
DECODE_BATCH = 128


@torch.no_grad()
def decode_u8(tok: torch.nn.Module, ids: np.ndarray, grid: tuple[int, int], dev: torch.device,
              batch: int = DECODE_BATCH) -> np.ndarray:
    """Token rows `(N, cells)` -> `(N, h, w, 3)` uint8 frames, through `Tokenizer.decode`.

    Converted to uint8 exactly as `fsq_eval.reconstruct` converts. **Ground
    truth here is always decoded cached rows**, never `reconstruct`'s output:
    the two encoder input layouts disagree on 0.33% of R1's tokens (item 5), and
    a rollout produces cache-like tokens.
    """
    out = []
    for i in range(0, len(ids), batch):
        x = torch.from_numpy(np.ascontiguousarray(ids[i:i + batch]).reshape(-1, *grid)).to(dev)
        y = tok.decode(x)
        out.append((y * PEAK).round().clamp(0, PEAK).byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


class Features(NamedTuple):
    """The validator's per-frame measurements the pixel rows read, stacked over frames."""
    link_major: np.ndarray    # (..., links) the long side of each link's colour blob, px
    link_minor: np.ndarray    # (..., links) its short side, px
    link_angle: np.ndarray    # (..., links) radians in [0, pi)
    block_px: np.ndarray      # (..., blocks) pixels of each block's colour
    block_centre: np.ndarray  # (..., blocks, 2) centre of each block's bbox, (x, y) px


def features(frames: np.ndarray, palette: validator.Palette, tau: float) -> Features:
    """`validator.measure_pixels_only` over frames `(..., h, w, 3)`."""
    lead = frames.shape[:-3]
    ms = [validator.measure_pixels_only(f, palette, tau)
          for f in frames.reshape(-1, *frames.shape[-3:])]
    blocks = list(palette.blocks)
    box = np.stack([m.bbox[blocks] for m in ms]).astype(np.float64)
    return Features(np.stack([m.link_extent[:, 0] for m in ms]).reshape(*lead, -1),
                    np.stack([m.link_extent[:, 1] for m in ms]).reshape(*lead, -1),
                    np.stack([m.link_angle for m in ms]).reshape(*lead, -1),
                    np.stack([m.px_count[blocks] for m in ms]).reshape(*lead, -1),
                    np.stack(((box[:, :, 0] + box[:, :, 2]) / 2,
                              (box[:, :, 1] + box[:, :, 3]) / 2), -1).reshape(*lead, len(blocks), 2))


# Gate row 6: the link-drift statistic's window, the requirement's 200 steps.
DRIFT_WINDOW = 200
DRIFT_RATIO_MAX = 1.1


def link_drift(major: np.ndarray, start: int, window: int = DRIFT_WINDOW) -> np.ndarray:
    """`(E, T, links)` major extents -> `(E * n, links)` drift, `(max - min) / median`.

    Over the `n` non-overlapping `window`-frame windows from frame `start`, the
    statistic `bench/link_drift_probe.py` computes, on the raw pixel extent: the
    deprojection it tried was measured and refuted, and is not re-attempted. A
    link missing from a whole window has median 0 and reads infinite drift.
    """
    n = (major.shape[1] - start) // window
    if n < 1:
        raise ValueError(f"{major.shape[1]} frames hold no {window}-frame window from {start}")
    seg = major[:, start:start + n * window].reshape(major.shape[0], n, window, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (seg.max(2) - seg.min(2)) / np.median(seg, 2)
    return np.where(np.isnan(d), np.inf, d).reshape(-1, major.shape[-1])


# Gate row 7 (S, reported): where a reappearing block must be. Q-6 names no
# tolerance; Q-6b (C) names 2 px, which is reported beside it.
PERMANENCE_TOL_PX = 4.0
PERMANENCE_TOL_C_PX = 2.0


def reappearances(visible: np.ndarray, start: int) -> list[tuple[int, int, int]]:
    """`(episode, block, frame)` for every reappearance at a frame `>= start`.

    `visible` is `(E, blocks, T)`, the renderer's visible pixel count. A
    reappearance is a frame where a block is visible again after one or more
    frames of full occlusion that `data.seen_later` counts as recoverable - the
    split it owns - and that began inside the episode, after a visible frame.
    """
    hidden = visible == 0
    rec = hidden & data.seen_later(visible)
    out = []
    for e, b, t in zip(*np.nonzero(~hidden[:, :, 1:] & rec[:, :, :-1])):
        t = int(t) + 1
        run = t - 1
        while run >= 0 and hidden[e, b, run]:
            run -= 1
        if t >= start and run >= 0:
            out.append((int(e), int(b), t))
    return out


# Gate row 4: the frame-to-frame continuity check. Each feature's bound is a
# `validator` config key, one value per object in palette name order - the
# validator emits measurements, and the verdict is a threshold in config
# (Phase 0's rule). The bounds are the largest per-step change over every
# transition of the val split's cached rows, decoded (`calibrate_continuity`),
# so zero ground-truth windows fire by construction.
CONTINUITY_KEYS = {"link_angle": "continuity_link_angle_max",
                   "link_major": "continuity_link_major_max",
                   "link_minor": "continuity_link_minor_max",
                   "block_centre": "continuity_block_centre_max"}
COHERENCE_BAR = 200   # Q-3: frames before the check fires
EXPOSURE_TRIGGER = 100   # decision 7: a horizon under this while row 1 passes


def step_change(a: Features, b: Features) -> dict[str, np.ndarray]:
    """Each continuity feature's change from frames `a` to frames `b`, per object.

    A link's angle only means something modulo pi (`validator._oriented`), so
    its change is wrapped into [-pi/2, pi/2) before the absolute value. A
    block's bbox centre does not exist while the block shows no pixels, so its
    change is 0 unless the block shows in both frames.
    """
    shown = (a.block_px > 0) & (b.block_px > 0)
    return {"link_angle": np.abs((b.link_angle - a.link_angle + np.pi / 2) % np.pi - np.pi / 2),
            "link_major": np.abs(b.link_major - a.link_major),
            "link_minor": np.abs(b.link_minor - a.link_minor),
            "block_centre": np.where(shown, np.linalg.norm(b.block_centre - a.block_centre, axis=-1),
                                     0.0)}


def consecutive(f: Features) -> tuple[Features, Features]:
    """Features `(E, T, ...)` -> the `(E, T - 1, ...)` frames before and after each transition."""
    return Features(*(x[:, :-1] for x in f)), Features(*(x[:, 1:] for x in f))


def fires(change: dict[str, np.ndarray], bounds: dict[str, np.ndarray]) -> np.ndarray:
    """True where any feature of any object changed by more than its bound."""
    return np.any([(change[k] > bounds[k]).any(-1) for k in CONTINUITY_KEYS], axis=0)


def continuity_bounds(cfg: config.Config, palette: validator.Palette) -> dict[str, np.ndarray]:
    """The calibrated bounds from `cfg.validator`, checked against the palette's object counts."""
    out = {}
    for k, key in CONTINUITY_KEYS.items():
        want = len(palette.blocks if k == "block_centre" else palette.links)
        out[k] = np.asarray(cfg.validator[key], np.float64)
        if out[k].shape != (want,):
            raise ValueError(f"validator.{key} holds {len(out[k])} values, the palette has {want} "
                             f"{'blocks' if k == 'block_centre' else 'links'}")
    return out


def horizon(fire: np.ndarray, ctx: int) -> np.ndarray:
    """`(E, T - 1)` transition verdicts of rollouts seeded with `ctx` frames -> frames until the check fires.

    Transition `j` is frame `j` to `j + 1`, so the first generated frame is
    judged at transition `ctx - 1`, against the seed's last frame. A rollout
    whose first generated frame fires has horizon 0; one that never fires reads
    its whole length, `T - ctx`, a lower bound.
    """
    gen = fire[:, ctx - 1:]
    return np.where(gen.any(1), gen.argmax(1), gen.shape[1])


def substitution_test(truth: Features, bounds: dict[str, np.ndarray], lag: int,
                      stride: int) -> dict:
    """`bench/q3_blind_probe.py`'s shape, on the continuity check: the share of
    300-step substitutions it fires on, and of the clean transitions at the same frames.

    For every `stride`-th frame `t` of each episode, frame `t - 1` is followed
    by frame `t + lag` offered as frame `t`, the worst dynamics failure that is
    still real decoder output. The clean control is frame `t` itself.
    """
    t = np.arange(1, truth.link_angle.shape[1] - lag, stride)
    before = Features(*(x[:, t - 1] for x in truth))
    sub = fires(step_change(before, Features(*(x[:, t + lag] for x in truth))), bounds)
    clean = fires(step_change(before, Features(*(x[:, t] for x in truth))), bounds)
    return {"lag": lag, "stride": stride, "pairs": int(sub.size),
            "fire_substituted": float(sub.mean()), "fire_clean": float(clean.mean())}


# Gate row 5 (reported this phase): action `a` drives joint `j` by digit
# `(a // 3**j) % 3 - 1`, least significant first (`sim/policy.h`).
JOINTS = 2
COMMAND = np.array([[(a // 3 ** j) % 3 - 1 for j in range(JOINTS)] for a in range(dynamics.N_ACTIONS)])
FOLLOW_RATIO = 0.9    # Q-4: >= 90% of the simulator's own agreement
FOLLOW_SEED = 0


def pixel_joints(link_angle: np.ndarray) -> np.ndarray:
    """Link angles `(..., links)` on screen -> joint angles `(..., joints)`, as the pixels give them.

    Joint 0 turns link 0; joint 1 turns link 1 relative to link 0. On screen
    each is a perspective-squashed angle modulo pi, which is all a rollout has.
    """
    return np.stack((link_angle[..., 0], link_angle[..., 1] - link_angle[..., 0]), -1)


def wrapped(d: np.ndarray) -> np.ndarray:
    return (d + np.pi / 2) % np.pi - np.pi / 2


def action_balanced(actions: np.ndarray, seed: int = FOLLOW_SEED) -> np.ndarray:
    """Indices into `actions`, the same number for every action that drives a joint, sorted.

    The neutral action drives none, so it has no commanded sign and is left
    out. The count is the rarest driving action's, so none is resampled.
    """
    driving = [a for a in range(dynamics.N_ACTIONS) if COMMAND[a].any()]
    n = min(int((actions == a).sum()) for a in driving)
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(actions == a), n, replace=False)
                                   for a in driving]))


def agreement(delta: np.ndarray, command: np.ndarray) -> float:
    """Share of driven (frame, joint) pairs whose change has the commanded sign."""
    driven = command != 0
    return float((np.sign(delta[driven]) == command[driven]).mean())


def truth_features(tok: torch.nn.Module, cfg: config.Config, eps: Episodes,
                   palette: validator.Palette, dev: torch.device) -> Features:
    """Every val frame's features, measured on its cached row decoded through `tok`, R1."""
    grid, cells = tuple(cfg.shapes.token_grid), math.prod(cfg.shapes.token_grid)
    px = decode_u8(tok, eps.tokens.reshape(-1, cells), grid, dev)
    return features(px.reshape(*eps.actions.shape, *px.shape[1:]), palette,
                    cfg.validator["offpalette_tau"])


def calibrate_continuity(cfg: config.Config, device: str | None = None) -> dict:
    """The continuity bounds: each feature's largest per-step change over every val transition.

    Measured on the val split's cached rows decoded through R1 - reconstructions,
    not renders, and the population a rollout produces (item 5's finding) - so
    no ground-truth transition, and so no window of ground-truth frames, fires.
    Returns the `validator` keys to write into the config, and prints them with
    the regression test the bounds give. Run once; changing them moves
    `validator_hash`, as moving the palette thresholds did.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    splits = dynamics.load_splits(cfg)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    eps = val_episodes(splits, len(palette.blocks))
    tok, _ = fsq_eval.load_run(dynamics.TOKENIZER_RUN, cfg, dev)
    truth = truth_features(tok, cfg, eps, palette, dev)
    change = step_change(*consecutive(truth))
    bounds = {k: change[k].reshape(-1, change[k].shape[-1]).max(0) for k in CONTINUITY_KEYS}
    assert not fires(change, bounds).any()
    q3 = dynamics._bench_module("q3_blind_probe")
    sub = substitution_test(truth, bounds, q3.LAG, q3.STRIDE)
    keys = {CONTINUITY_KEYS[k]: [float(v) for v in bounds[k]] for k in CONTINUITY_KEYS}
    print(f"{len(eps.ids)} val episodes, {change['link_angle'].shape[0] * change['link_angle'].shape[1]:,} "
          f"transitions of decoded cached rows")
    print(json.dumps(keys, indent=1))
    print(f"regression (bench/q3_blind_probe.py's shape, {sub['pairs']} pairs): fires on "
          f"{sub['fire_substituted']:.1%} of {q3.LAG}-step substitutions and "
          f"{sub['fire_clean']:.1%} of clean transitions")
    return {"bounds": keys, "regression": sub}


# ----------------------------------------------------------------- the gate

# The configurable-context requirement's three lengths, from one checkpoint.
ROLLOUT_CTXS = (4, 8, 15)
PARAM_BAR = 20_000_000   # "dynamics model parameters <= 20M bf16"
VRAM_BAR_GB = 7.5        # "peak training VRAM <= 7.5 GB", read as GiB like the run's records


def evaluate(run_id: str, cfg: config.Config, checkpoint: str = "best",
             device: str | None = None, runs_dir: Path | str = ROOT / "runs") -> dict:
    """The dynamics gate table for one run: rows 1-6 and 8-9 pass/fail, 7 and 10 reported.

    `checkpoint` is `best` (item 4's `best.pt`, kept by gate row 1's measure)
    or `final` (the last per-epoch `model.pt`). Every rollout starts at frame
    0 of a val episode, one per episode, with that episode's own action column,
    and runs to the episode's end.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = Path(runs_dir) / run_id
    file = {"best": "best.pt", "final": "model.pt"}[checkpoint]
    ctx, cells, grid = cfg.data["ctx"], math.prod(cfg.shapes.token_grid), tuple(cfg.shapes.token_grid)
    if max(ROLLOUT_CTXS) != ctx:
        raise ValueError(f"the longest rollout is ctx {max(ROLLOUT_CTXS)}, the model trained at {ctx}")
    splits = dynamics.load_splits(cfg)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    tau = cfg.validator["offpalette_tau"]
    eps = val_episodes(splits, len(palette.blocks))
    n_eps, length = eps.actions.shape

    bounds = continuity_bounds(cfg, palette)

    # Row 1 and row 10's copy and false-flip figures, on the full population.
    pop = score_run(run_dir, cfg, splits, dev, keep_pred=True)
    pred = pop.pop("pred")[checkpoint]
    row1 = pop[checkpoint]
    history = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)["history"]

    model, ck = load_checkpoint(run_dir / file, cfg, dev)
    params = sum(p.numel() for p in model.parameters())

    # Rows 3, 2 and 9: one rollout per val episode at each context length.
    rolls, secs = {}, {}
    for c in ROLLOUT_CTXS:
        t0 = time.perf_counter()
        rolls[c] = rollout(model, eps.tokens[:, :c], eps.actions, length - c, dev)
        secs[c] = time.perf_counter() - t0
    ran = {c: r.passes == length - c and r.frames.shape == (n_eps, length, cells)
           and np.array_equal(r.frames[:, :c], eps.tokens[:, :c])
           and 0 <= r.frames.min() and r.frames.max() < model.head.out_features
           for c, r in rolls.items()}
    lay = dynamics.layout(ctx, cells)
    one = dynamics.score_windows(model, splits.val, eps.first, lay,
                                 dynamics.attention_mask(lay).to(dev), dev, batch=n_eps)
    assert np.array_equal(one["target"], eps.tokens[:, ctx]), "row 2 scored other windows"
    row2 = np.array_equal(rolls[ctx].frames[:, ctx], one["pred"])
    again, _ = load_checkpoint(run_dir / file, cfg, dev)
    row9 = np.array_equal(rollout(again, eps.tokens[:, :ctx], eps.actions, length - ctx, dev).frames,
                          rolls[ctx].frames)
    del again

    # The pixel rows: ground truth is the cache decoded, the rollout its own tokens decoded.
    tok, _ = fsq_eval.load_run(dynamics.TOKENIZER_RUN, cfg, dev)
    truth = truth_features(tok, cfg, eps, palette, dev)
    px = decode_u8(tok, rolls[ctx].frames.reshape(-1, cells), grid, dev)
    gen = features(px.reshape(n_eps, length, *px.shape[1:]), palette, tau)

    # Row 4 (reported until Q-3 is restated): frames until the continuity check
    # fires, beside the check on the truth itself and the q3 regression test.
    truth_fire = fires(step_change(*consecutive(truth)), bounds)
    hz = horizon(fires(step_change(*consecutive(gen)), bounds), ctx)
    q3 = dynamics._bench_module("q3_blind_probe")
    sub = substitution_test(truth, bounds, q3.LAG, q3.STRIDE)

    # Row 5 (reported this phase): one-step predictions from true context, on an
    # action-balanced subset of the row-1 population, and the truth on the same frames.
    e_of = np.searchsorted(eps.first, np.arange(len(pred)), side="right") - 1
    t_of = np.arange(len(pred)) - eps.first[e_of] + ctx
    sub5 = action_balanced(eps.actions[e_of, t_of])
    e5, t5 = e_of[sub5], t_of[sub5]
    for i in sub5:   # window i's target is frame t_of[i] of episode e_of[i], by the sampler's own addressing
        ep, off = splits.val.windows.locate(int(i))
        assert ep.episode_id == eps.ids[e_of[i]] and off + ctx == t_of[i], i
    px5 = decode_u8(tok, pred[sub5], grid, dev)
    pred_joint = pixel_joints(features(px5, palette, tau).link_angle)
    prev_joint = pixel_joints(truth.link_angle[e5, t5 - 1])
    d_truth = wrapped(pixel_joints(truth.link_angle[e5, t5]) - prev_joint)
    d_model = wrapped(pred_joint - prev_joint)
    d_qpos = eps.qpos[e5, t5] - eps.qpos[e5, t5 - 1]
    # Image y points down, so the pixel angle's sign against the joint's is
    # calibrated from data, as `validator._oriented` asks, on the truth's frames.
    sign = np.array([np.sign(np.corrcoef(d_truth[:, j], d_qpos[:, j])[0, 1]) for j in range(JOINTS)])
    cmd = COMMAND[eps.actions[e5, t5]]
    follow = {"model": agreement(sign * d_model, cmd), "truth": agreement(sign * d_truth, cmd),
              "truth_qpos": agreement(d_qpos, cmd), "frames": len(sub5), "sign": sign.tolist()}

    # Row 6: drift over the same windows of the rollout's generated frames and of the truth.
    drift_gen = link_drift(gen.link_major, ctx).mean(0)
    drift_truth = link_drift(truth.link_major, ctx).mean(0)
    row6 = bool((drift_gen <= DRIFT_RATIO_MAX * drift_truth).all())

    # Row 7: at each reappearance inside the rollout, is the block where the truth shows it?
    events = reappearances(eps.visible, ctx)
    near = {PERMANENCE_TOL_PX: 0, PERMANENCE_TOL_C_PX: 0}
    truth_absent = 0
    for e, b, r in events:
        # Scored at frame `r` itself. A block coming back shows a pixel or two,
        # which the decoder often drops; with nothing to compare against, the
        # event is a miss, and how many were is printed beside the row.
        if truth.block_px[e, r, b] == 0:
            truth_absent += 1
            continue
        dist = np.linalg.norm(gen.block_centre[e, r, b] - truth.block_centre[e, r, b])
        for tol in near:
            near[tol] += int(gen.block_px[e, r, b] > 0 and dist <= tol)

    vram = max(h["peak_vram_reserved_gb"] or 0.0 for h in history)
    vram_alloc = max(h["peak_vram_alloc_gb"] or 0.0 for h in history)
    last = history[-1]
    names = ", ".join(palette.names[i] for i in palette.links)
    rows = [
        (1, "Held-out next-token accuracy vs persistence",
         f"{row1['acc']:.4%} vs {row1['persistence']:.4%} ({row1['margin_points']:+.2f} pts); "
         f"marginal {row1['marginal_top1_acc']:.2%}", "> persistence", row1["margin_cells"] > 0),
        (2, f"One full next frame from {ctx} frames + one action",
         f"{'identical' if row2 else 'DIFFERS'} to one pass on {n_eps} windows, "
         f"{rolls[ctx].passes} passes for {length - ctx} frames", "exact", row2),
        (3, "Rollout at ctx " + ", ".join(map(str, ROLLOUT_CTXS)) + " from one checkpoint",
         "; ".join(f"{c}: {'ran' if ran[c] else 'FAILED'} {length - c} steps {secs[c]:.0f}s"
                   for c in ROLLOUT_CTXS), "runs", all(ran.values())),
        (4, "Frames until the continuity check fires (median)",
         f"{np.median(hz):.0f} (min {hz.min()}, max {hz.max()} of {length - ctx})",
         f">= {COHERENCE_BAR} (reported)", None),
        (5, "Action-following, model vs simulator (pixels)",
         f"{follow['model']:.1%} vs {follow['truth']:.1%} (ratio "
         f"{follow['model'] / follow['truth']:.2f}); qpos {follow['truth_qpos']:.1%}",
         f">= {FOLLOW_RATIO:.0%} of truth (reported)", None),
        (6, f"Link drift over {DRIFT_WINDOW}-step windows ({names})",
         " / ".join(f"{g:.1%} vs {t:.1%}" for g, t in zip(drift_gen, drift_truth)),
         f"<= {DRIFT_RATIO_MAX}x truth", row6),
        (7, "Block reappears in place after full occlusion",
         f"{near[PERMANENCE_TOL_PX] / max(len(events), 1):.1%} within {PERMANENCE_TOL_PX:g} px, "
         f"{near[PERMANENCE_TOL_C_PX] / max(len(events), 1):.1%} within {PERMANENCE_TOL_C_PX:g} px "
         f"of {len(events)} events, {truth_absent} unseen in the truth", ">= 80% (S)", None),
        (8, "Parameters; peak training VRAM (reserved)",
         f"{params:,}; {vram:.2f} GB ({vram_alloc:.2f} allocated)",
         f"<= 20M; <= {VRAM_BAR_GB} GB", params <= PARAM_BAR and vram <= VRAM_BAR_GB),
        (9, "Rollout reproduced from checkpoint + seed clips",
         "identical" if row9 else "DIFFERS", "identical", row9),
        (10, "Train-val gap; copy overlap; false flips on static cells",
         f"{last['gap_ce']:+.4f} ce (epoch {last['epoch']}); {row1['copy_overlap']:.2%}; "
         f"{row1['false_flip_static']:.2%}", "reported", None),
    ]
    rows.sort(key=lambda r: r[0])

    print(f"\ngate table - {run_id} {file} (step {ck['step']:,}, epoch {ck['epoch']}), "
          f"dynamics_hash {cfg.dynamics_hash[:8]}, {n_eps} val rollouts of {length - ctx} steps")
    print(f"{'#':>2}  {'measure':<52} {'value':<64} {'bar':<18} verdict")
    for n, name, value, bar, ok in rows:
        verdict = "-" if ok is None else ("PASS" if ok else "FAIL")
        print(f"{n:>2}  {name:<52} {value:<64} {bar:<18} {verdict}")

    # Rows 4 and 5 are reported, not pass/fail, and the reason is printed with
    # them: neither instrument can do its job on decoded 64x64 frames (item 6).
    print()
    print(f"    row 4's check fires on {int(truth_fire.sum())} of {truth_fire.size:,} ground-truth "
          f"transitions; bench/q3_blind_probe.py's regression: {sub['fire_substituted']:.1%} of "
          f"{sub['pairs']} {sub['lag']}-step substitutions (needs 100%), "
          f"{sub['fire_clean']:.1%} of clean transitions (needs 0%)")
    # Decision 5's revisit trigger is a rollout that freezes or loops, and a
    # frozen rollout reads the longest horizon and no link drift at all.
    frozen = float((rolls[ctx].frames[:, ctx:] == rolls[ctx].frames[:, ctx - 1:-1]).all(-1).mean())
    print(f"    {frozen:.1%} of generated frames repeat the frame before them token for token "
          f"(decision 5's trigger is a rollout that freezes); a frozen rollout reads the whole "
          f"horizon in row 4 and no drift in row 6")
    if row1["margin_cells"] > 0 and np.median(hz) < EXPOSURE_TRIGGER:
        print(f"    decision 7's trigger fires: horizon under {EXPOSURE_TRIGGER} while row 1 passes")
    print(f"    row 5 on {follow['frames']:,} action-balanced one-step predictions: the pixel "
          f"angle's per-step sign is at chance on the truth, so the ratio cannot pass or fail")
    failed = [n for n, _, _, _, ok in rows if ok is False]
    print("\nall pass/fail rows pass" if not failed else f"\nFAILED rows: {failed}")
    return {"run_id": run_id, "checkpoint": file, "step": ck["step"], "row1": row1,
            "population": pop["population"], "rollout_s": secs, "row2_exact": row2,
            "row3_ran": ran, "row9_identical": row9,
            "horizon": hz.tolist(), "frozen_share": frozen, "truth_fires": int(truth_fire.sum()), "q3_regression": sub,
            "follow": follow,
            "drift_rollout": drift_gen.tolist(), "drift_truth": drift_truth.tolist(),
            "permanence_events": len(events), "permanence_truth_absent": truth_absent,
            "permanence_near": {str(k): v for k, v in near.items()},
            "params": params, "peak_vram_reserved_gb": vram, "peak_vram_alloc_gb": vram_alloc,
            "gap_ce": last["gap_ce"], "failed_rows": failed}


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
    _check_rollout(cfg)
    _check_rows()
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
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    if not torch.cuda.is_available():
        # The cache was encoded, and the bounds calibrated, on CUDA.
        print("continuity bounds: skipped - no CUDA device")
    else:
        eps = val_episodes(splits, len(palette.blocks))
        cuda = torch.device("cuda")
        truth = truth_features(fsq_eval.load_run(dynamics.TOKENIZER_RUN, cfg, cuda)[0], cfg, eps,
                               palette, cuda)
        bounds = continuity_bounds(cfg, palette)
        fire = fires(step_change(*consecutive(truth)), bounds)
        assert not fire.any(), f"{int(fire.sum())} ground-truth transitions fire the continuity check"
        # The control: every bound shrunk by a tenth must fire somewhere.
        assert fires(step_change(*consecutive(truth)), {k: 0.9 * v for k, v in bounds.items()}).any()
        q3 = dynamics._bench_module("q3_blind_probe")
        sub = substitution_test(truth, bounds, q3.LAG, q3.STRIDE)
        print(f"continuity bounds: 0 of {fire.size:,} decoded ground-truth transitions fire, and "
              f"bounds 10% tighter do; the q3 regression reads {sub['fire_substituted']:.1%} of "
              f"{sub['pairs']} substitutions (the requirement asks 100%)")
    print("dynamics_eval self-check ok")


def _check_rollout(cfg: config.Config) -> None:
    """The rollout on a tiny random model, on CPU: shape, step count, gate row 2's
    exactness, the action each step reads, other context lengths, and reproduction."""
    tiny = dataclasses.replace(cfg, dynamics={**cfg.dynamics, "d_model": 16, "n_layers": 1,
                                              "n_heads": 2})
    torch.manual_seed(0)
    model = dynamics.build(tiny).double().eval()
    ctx, cells, codes, steps, dev = cfg.data["ctx"], math.prod(cfg.shapes.token_grid), 512, 6, \
        torch.device("cpu")
    rng = np.random.default_rng(0)
    tok = rng.integers(0, codes, (3, ctx + steps, cells))
    act = rng.integers(0, dynamics.N_ACTIONS, (3, ctx + steps))
    r = rollout(model, tok[:, :ctx], act, steps, dev)
    assert r.passes == steps and r.frames.shape == (3, ctx + steps, cells)
    assert np.array_equal(r.frames[:, :ctx], tok[:, :ctx]) and r.frames.max() < codes

    # Row 2: the first generated frame is the one-pass argmax over the true
    # window, whose last frame never reached the input.
    lay = dynamics.layout(ctx, cells)
    x, _ = dynamics.assemble(lay, tok[:, :ctx + 1], act[:, :ctx + 1], codes)
    with torch.no_grad():
        one = model(torch.from_numpy(x), dynamics.attention_mask(lay),
                    torch.from_numpy(lay.read))[:, -cells:].argmax(-1).numpy()
    assert np.array_equal(r.frames[:, ctx], one), "the rollout's first frame is not one pass"

    # What each step reads, with stand-ins whose answer names its input. One
    # predicts every cell as the id of the newest action in its input: frame
    # ctx + k must read action[ctx + k], the action that produced it. One copies
    # the newest frame in its input: every generated frame must then be the
    # seed's last, which shows each step's output is fed back as the next input.
    class StandIn(torch.nn.Module):
        """Scores the last frame's cells as `pick(input)`, `(B, cells)` codes."""
        def __init__(self, pick) -> None:
            super().__init__()
            self.pick, self.head = pick, torch.nn.Linear(1, codes)

        def forward(self, x: torch.Tensor, mask: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(x.shape[0], len(read), codes)
            out[:, -cells:] = F.one_hot(self.pick(x), codes).float()
            return out
    newest_action = StandIn(lambda x: (x[:, -1:] - codes).expand(-1, cells))
    newest_frame = StandIn(lambda x: x[:, -cells - 1:-1])
    got = rollout(newest_action, tok[:, :ctx], act, steps, dev).frames
    assert np.array_equal(got[:, ctx:], np.repeat(act[:, ctx:, None], cells, 2)), \
        "a step read another record's action"
    got = rollout(newest_frame, tok[:, :ctx], act, steps, dev).frames
    assert np.array_equal(got[:, ctx:], np.repeat(tok[:, ctx - 1:ctx], steps, 1)), \
        "a generated frame did not feed the next step"
    for c in ROLLOUT_CTXS[:-1]:
        rc = rollout(model, tok[:, :c], act[:, :c + steps], steps, dev)
        assert rc.passes == steps and rc.frames.shape == (3, c + steps, cells)
    assert np.array_equal(rollout(model, tok[:, :ctx], act, steps, dev).frames, r.frames)
    assert _refused(rollout, model, tok[:, :ctx], act[:, 1:], steps, dev)
    print(f"rollout: {steps} steps in {steps} passes; the first frame is the one-pass argmax; "
          f"frame ctx + k reads action[ctx + k] and the frames before it, its own output "
          f"feeding the next step; runs at ctx "
          f"{', '.join(map(str, ROLLOUT_CTXS))}; reproduces exactly")


def _check_rows() -> None:
    """Rows 4-7's arithmetic on hand-built cases."""
    # Row 6: one link, one 200-frame window from frame 2, extent 10 then 12.
    major = np.full((1, 202, 1), 10.0)
    major[0, 102:] = 12.0
    assert np.allclose(link_drift(major, 2), [[2.0 / 11.0]])
    major[0, 2:] = 0.0
    assert np.isinf(link_drift(major, 2)).all(), "a vanished link read finite drift"
    assert _refused(link_drift, major[:, :150], 2)

    # Row 7: block 0 hides at 3..4 and returns at 5; block 1 hides at 0..1
    # (never seen before, not an event) and hides for good at 6.
    vis = np.array([[[9, 9, 9, 0, 0, 9, 9, 9], [0, 0, 9, 9, 9, 9, 0, 0]]])
    assert reappearances(vis, 0) == [(0, 0, 5)] and reappearances(vis, 6) == []

    # Row 4: two frames, one link and one block; the angle wraps across pi.
    def f(angle: float, centre: tuple, px: int) -> Features:
        return Features(np.array([[10.0]]), np.array([[3.0]]), np.array([[angle]]),
                        np.array([[px]]), np.array([[centre]], dtype=np.float64))
    ch = step_change(f(0.05, (5, 5), 9), f(np.pi - 0.05, (8, 9), 9))
    assert np.isclose(ch["link_angle"], 0.1).all() and np.isclose(ch["block_centre"], 5.0).all()
    assert step_change(f(0.0, (5, 5), 0), f(0.0, (60, 60), 9))["block_centre"].item() == 0.0
    bounds = {"link_angle": np.array([0.2]), "link_major": np.array([1.0]),
              "link_minor": np.array([1.0]), "block_centre": np.array([4.0])}
    assert fires(ch, bounds).all() and not fires(ch, {**bounds, "block_centre": np.array([5.0])}).any()
    fire = np.zeros((3, 9), bool)
    fire[0, 3], fire[1, 6] = True, True     # ctx 4: generated frames start at transition 3
    assert horizon(fire, 4).tolist() == [0, 3, 6]

    # Row 5: every driving action equally often; the neutral action 4 left out.
    acts = np.array([4] * 50 + [a for a in range(9) for _ in range(3 + a)])
    sub = action_balanced(acts)
    assert 4 not in acts[sub] and np.bincount(acts[sub], minlength=9).tolist() == [3] * 4 + [0] + [3] * 4
    assert COMMAND[4].tolist() == [0, 0] and COMMAND[0].tolist() == [-1, -1] and COMMAND[5].tolist() == [1, 0]
    assert agreement(np.array([[0.1, -0.2], [0.0, 0.3]]), np.array([[1, 1], [-1, 0]])) == 1 / 3
    print("rows 4-7: continuity change, wrap, absent blocks, horizon; drift and a vanished link; "
          "reappearances; the action-balanced subset and agreement, by hand")


def _refused(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="the dynamics eval: self-check, or calibrate the "
                                             "continuity check (the gate is python -m mirage.dynamics --eval)")
    ap.add_argument("--calibrate-continuity", action="store_true",
                    help="print the continuity bounds for the validator config section")
    ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "base.json"))
    args = ap.parse_args()
    if args.calibrate_continuity:
        calibrate_continuity(config.load(args.config))
        return
    _self_check()


if __name__ == "__main__":
    main()
