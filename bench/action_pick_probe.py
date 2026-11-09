"""Does the action change what the masked dynamics model predicts, or only barely?

No training. Every 5th step of the gate's val episodes, each frame is predicted in
one round from its real context under all 9 actions (and the null action, on a
model that has one), and per kind of cell this prints:

- `copy pick`: how often the top pick under the real action is last frame's code;
- `acc`, `P(copy)`, `P(true)`: the top pick's accuracy and the mean probabilities;
- `flip any / opp / null`: how often the top pick differs when the real action is
  swapped for any other, for its opposite (every joint command negated), or for null;
- `TV opp / null`: total variation between the two predicted distributions;
- `LL real-opp`: nats by which the real action beats its opposite on the true code;
- `margin` and `shift`: nats by which the copy code beats the best other code, and
  the largest change of that margin across the 9 actions.

How to read it on the moving cells (these thresholds are judgement, set before the run):

- TV opp under 0.05: the distribution barely depends on the action, so how the
  action reaches the model, or what the loss rewards, is the limit.
- TV opp at least 0.05, flips rare, and P(copy) well under 1: the distribution
  responds, but the argmax still lands on copy. Greedy decoding is the limit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirage import config, dynamics, dynamics_eval as de, validator  # noqa: E402

CONFIG = ROOT / "mirage" / "configs" / "dyn192x4_mask.json"
EVERY = 5
WEIGHTS = (1.5, 2.0, 3.0, 5.0)
SUMS = ("n", "copy", "acc", "p_copy", "p_true", "flip_any", "flip_opp", "tv_opp", "ll_opp",
        "margin", "shift", "flip_null", "tv_null")


def tv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a.exp() - b.exp()).abs().sum(-1)


def probe(model, eps, cover: np.ndarray, ctx: int, dev: torch.device) -> dict:
    tokens, actions = eps.tokens, eps.actions
    e, length, cells = tokens.shape
    codes = model.head.out_features
    null = getattr(model, "null_id", None)
    lay = dynamics.scoring_layout(model, ctx, cells)
    mask = dynamics.attention_mask(lay).to(dev)
    read = torch.from_numpy(lay.read).to(dev)
    pos = torch.from_numpy(lay.pos).to(dev)
    cats = ("moving", "moving base", "moving forearm", "static")
    tot = {c: dict.fromkeys(SUMS, 0.0) for c in cats}
    wtot = {c: {w: {"copy": 0.0, "acc": 0.0} for w in WEIGHTS} for c in cats}

    def margin(lp: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        c = lp.gather(-1, prev[..., None])[..., 0]
        rest = lp.scatter(-1, prev[..., None], -math.inf).max(-1).values
        return c - rest

    for k in range(0, length - ctx, EVERY):
        t = ctx + k
        win = tokens[:, k:t + 1].copy()
        win[:, -1] = 0
        x, _ = dynamics.assemble(lay, win, actions[:, k:t + 1], codes)
        x = torch.from_numpy(x).to(dev)
        q = torch.full((e, cells), model.mask_id, dtype=x.dtype, device=dev)
        real = torch.from_numpy(actions[:, t]).to(dev)
        if not (x[:, -1] - codes == real).all():
            raise RuntimeError("the stream's last slot is not the predicted frame's action")

        def run(last: torch.Tensor) -> torch.Tensor:
            y = x.clone()
            y[:, -1] = last
            with dynamics._autocast(dev):
                return model(torch.cat((y, q), 1), mask, read, pos).float()

        lg = torch.stack([run(torch.full((e,), codes + a, device=dev)) for a in range(dynamics.N_ACTIONS)], 1)
        lp = torch.log_softmax(lg, -1)                              # (E, actions, cells, codes)
        idx = torch.arange(e, device=dev)
        lr, lo = lp[idx, real], lp[idx, dynamics.N_ACTIONS - 1 - real]
        truth = torch.from_numpy(tokens[:, t]).to(dev)
        prev = torch.from_numpy(tokens[:, t - 1]).to(dev)
        top, tops = lr.argmax(-1), lp.argmax(-1)
        mrg = margin(lr, prev)
        mrgs = margin(lp, prev[:, None].expand(-1, dynamics.N_ACTIONS, -1))
        ll = lr.gather(-1, truth[..., None])[..., 0]
        vals = {"n": torch.ones_like(mrg),
                "copy": (top == prev).float(),
                "acc": (top == truth).float(),
                "p_copy": lr.gather(-1, prev[..., None])[..., 0].exp(),
                "p_true": ll.exp(),
                "flip_any": (tops != top[:, None]).any(1).float(),
                "flip_opp": (lo.argmax(-1) != top).float(),
                "tv_opp": tv(lr, lo),
                "ll_opp": ll - lo.gather(-1, truth[..., None])[..., 0],
                "margin": mrg,
                "shift": (mrgs - mrg[:, None]).abs().max(1).values}
        lc = None
        if null is not None:
            lc = run(torch.full((e,), null, device=dev))
            ln = torch.log_softmax(lc, -1)
            vals["flip_null"] = (ln.argmax(-1) != top).float()
            vals["tv_null"] = tv(lr, ln)
        moving = truth != prev
        arm = torch.from_numpy(cover[:, t] | cover[:, t - 1]).to(dev)    # (E, links, cells)
        groups = {"moving": moving, "moving base": moving & arm[:, 0], "moving forearm": moving & arm[:, 1],
                  "static": ~moving}
        for c, g in groups.items():
            for name, v in vals.items():
                tot[c][name] += float(v[g].sum())
            if lc is not None:
                lreal = lg[idx, real]
                for w in WEIGHTS:
                    pick = (lc + w * (lreal - lc)).argmax(-1)
                    wtot[c][w]["copy"] += float((pick == prev)[g].sum())
                    wtot[c][w]["acc"] += float((pick == truth)[g].sum())
    out = {}
    for c in cats:
        n = tot[c]["n"]
        kept = [s for s in SUMS if s != "n" and (null is not None or not s.endswith("_null"))]
        out[c] = {"cells": int(n), **{s: tot[c][s] / n for s in kept}}
        if null is not None:
            out[c]["guided"] = {str(w): {m: wtot[c][w][m] / n for m in ("copy", "acc")} for w in WEIGHTS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="RUN_ID/best.pt under runs/")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--json", type=Path, help="write every number printed here")
    args = ap.parse_args()
    cfg = config.load(str(args.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = dynamics.load_splits(cfg, dynamics.TOKENIZER_RUN)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    eps = de.val_episodes(splits, len(palette.blocks))
    cover = de.link_cells(de.val_pixels(splits), palette, tuple(cfg.shapes.token_grid))
    ctx = cfg.data["ctx"]
    res = {}
    for ck in args.checkpoints:
        model, _ = de.load_checkpoint(ROOT / "runs" / ck, cfg, dev)
        with torch.no_grad():
            res[ck] = r = probe(model, eps, cover, ctx, dev)
        print(f"=== {ck}, one round from real context, every {EVERY}th step")
        for c, v in r.items():
            null = f"  flip null {v['flip_null']:.2%}  TV null {v['tv_null']:.3f}" if "flip_null" in v else ""
            print(f"  {c:15s} {v['cells']:>8,} cells  copy pick {v['copy']:.1%}  acc {v['acc']:.1%}  "
                  f"P(copy) {v['p_copy']:.3f}  P(true) {v['p_true']:.3f}  flip any {v['flip_any']:.2%}  "
                  f"flip opp {v['flip_opp']:.2%}  TV opp {v['tv_opp']:.3f}  LL real-opp {v['ll_opp']:+.3f}  "
                  f"margin {v['margin']:+.2f}  shift {v['shift']:.2f}{null}")
            for w, g in v.get("guided", {}).items():
                print(f"      w={w}: copy pick {g['copy']:.1%}  acc {g['acc']:.1%}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
