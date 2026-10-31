"""Why does lower held-out cross-entropy not become higher last-frame top-1?

r67 found input corruption lowers the best held-out loss (0.475 -> 0.459) while
no best checkpoint beats persistence. This probe splits gate row 1's population
(every val window's last frame, `dynamics_eval.population`) into four cell
classes by two facts: did the cell's 15x15 pixel field change (`quiet` is no
pixel change), and did its token change. A token that changes on a quiet field
is R1's spurious flip - a change with no visible cause.

Per checkpoint, on the full population:

- cross-entropy, accuracy and copy rate per class, and CE per target frame of
  the window, so the training loss (all 15 frames) and row 1's measure (the
  last frame) can be told apart;
- the model's "bets": cells whose greedy prediction differs from the previous
  frame - their mean confidence, hit rate, and the share that in truth stayed;
- calibration of the argmax confidence in ten bins;
- **a copy-unless-confident rule, held out**: predict the argmax only where its
  probability beats the previous frame's token by more than `delta`, else
  copy. `delta` is tuned on one half of the val episodes (even or odd episode
  id) and scored on the other, both ways, so the reported margin never saw the
  cells it is scored on. Persistence is the score at a `delta` large enough
  that the rule always copies.

The rule is a decoding change, not part of the gate: `dynamics_eval.rollout`
and row 1 stay greedy (decision 5).

    python bench/ce_top1_probe.py [RUN_ID/CHECKPOINT ...] [--config mirage/configs/dyn192x4.json]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import config, dynamics, dynamics_eval as de  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ("20261031-180246-dyn/best.pt", "20261031-181748-dyn/best.pt",
               "20261031-200739-dyn/best.pt", "20261031-200741-dyn/best.pt",
               "20261031-200751-dyn/best.pt", "20261031-200751-dyn/model.pt")
DELTAS = np.round(np.arange(0, 0.951, 0.05), 2)


@torch.no_grad()
def cell_stats(model: dynamics.Dynamics, splits: dynamics.TokenSplits, lay: dynamics.Layout,
               mask: torch.Tensor, dev: torch.device, batch: int = 64) -> dict[str, np.ndarray]:
    """Per last-frame cell: CE, greedy prediction, p(argmax), p(previous token); and CE per target frame."""
    ctx, cells, codes = lay.ctx, lay.cells, model.head.out_features
    read = torch.from_numpy(lay.read).to(dev)
    n = len(splits.val)
    out = {"ce": np.empty(n * cells, np.float32), "p_max": np.empty(n * cells, np.float32),
           "p_prev": np.empty(n * cells, np.float32), "pred": np.empty(n * cells, np.int64),
           "ce_by_frame": np.zeros(ctx)}
    for j in range(0, n, batch):
        idx = np.arange(j, min(n, j + batch))
        tok, act = dynamics._stack(splits.val, idx)
        x, y = (torch.from_numpy(a).to(dev) for a in dynamics.assemble(lay, tok, act, codes))
        with dynamics._autocast(dev):
            logits = model(x, mask, read).float()
        ce = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        out["ce_by_frame"] += ce.view(len(idx), ctx, cells).mean(2).sum(0).cpu().numpy()
        lp = logits[:, -cells:].log_softmax(-1)
        prev = torch.from_numpy(tok[:, -2].astype(np.int64)).to(dev)
        mx, am = lp.max(-1)
        s = slice(j * cells, (j + len(idx)) * cells)
        out["ce"][s] = ce[:, -cells:].flatten().cpu().numpy()
        out["p_max"][s] = mx.exp().flatten().cpu().numpy()
        out["p_prev"][s] = lp.gather(-1, prev[..., None]).exp().flatten().cpu().numpy()
        out["pred"][s] = am.flatten().cpu().numpy()
    out["ce_by_frame"] /= n
    return out


def gated(st: dict, prev: np.ndarray, delta: float) -> np.ndarray:
    """Copy unless the argmax beats the previous frame's token by more than `delta`."""
    return np.where(st["p_max"] - st["p_prev"] > delta, st["pred"], prev)


def margin(pred: np.ndarray, tgt: np.ndarray, prev: np.ndarray, m: np.ndarray) -> float:
    return 100 * float((pred[m] == tgt[m]).mean() - (prev[m] == tgt[m]).mean())


def held_out(st: dict, tgt: np.ndarray, prev: np.ndarray, fold: np.ndarray) -> dict:
    """Tune `delta` on one fold, score on the other, both ways; pooled over all cells."""
    out, hits = {}, np.zeros(len(tgt), bool)
    for tune in (0, 1):
        t, s = fold == tune, fold != tune
        scores = [margin(gated(st, prev, d), tgt, prev, t) for d in DELTAS]
        d = float(DELTAS[int(np.argmax(scores))])
        p = gated(st, prev, d)
        hits[s] = p[s] == tgt[s]
        out[f"tune_{tune}"] = {"delta": d, "tune_margin": round(max(scores), 3),
                               "test_margin": round(margin(p, tgt, prev, s), 3)}
    out["held_out_margin"] = round(100 * float(hits.mean() - (prev == tgt).mean()), 3)
    out["held_out_cells_won"] = int(hits.sum() - (prev == tgt).sum())
    return out


def probe(st: dict, pop: dict, fold: np.ndarray) -> dict:
    tgt, prev, quiet = pop["target"].ravel(), pop["prev"].ravel(), pop["quiet"].ravel()
    stay, pred = tgt == prev, st["pred"]
    hit = pred == tgt
    classes = {"quiet_stays": quiet & stay, "quiet_changes": quiet & ~stay,
               "moving_stays": ~quiet & stay, "moving_changes": ~quiet & ~stay}
    bet = pred != prev
    bins = np.minimum((st["p_max"] * 10).astype(int), 9)
    return {"val_ce_all_frames": round(float(st["ce_by_frame"].mean()), 4),
            "ce_by_frame": [round(float(v), 4) for v in st["ce_by_frame"]],
            "last_frame_ce": round(float(st["ce"].mean()), 4),
            "greedy_margin": round(margin(pred, tgt, prev, np.ones_like(stay)), 3),
            "gain_points": round(100 * float((hit & ~stay).mean()), 3),
            "loss_points": round(100 * float((~hit & stay).mean()), 3),
            "classes": {k: {"share": round(float(m.mean()), 4), "ce": round(float(st["ce"][m].mean()), 3),
                            "acc": round(float(hit[m].mean()), 3),
                            "copy": round(float((pred == prev)[m].mean()), 3)} for k, m in classes.items()},
            "bets": {"share": round(float(bet.mean()), 4), "mean_conf": round(float(st["p_max"][bet].mean()), 3),
                     "hit_rate": round(float(hit[bet].mean()), 3),
                     "truly_stayed": round(float(stay[bet].mean()), 3)},
            "calibration": [[b / 10, round(float(st["p_max"][bins == b].mean()), 3),
                             round(float(hit[bins == b].mean()), 3), int((bins == b).sum())]
                            for b in range(10) if (bins == b).any()],
            "copy_unless_confident": held_out(st, tgt, prev, fold)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoints", nargs="*", default=CHECKPOINTS)
    ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "dyn192x4.json"))
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    args = ap.parse_args()
    cfg = config.load(args.config)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = dynamics.load_splits(cfg)
    pop = de.population(cfg, splits)
    lay = dynamics.layout(cfg.data["ctx"], math.prod(cfg.shapes.token_grid))
    mask = dynamics.attention_mask(lay).to(dev)
    episode = np.array([splits.val[i].episode_id for i in range(len(splits.val))])
    fold = np.repeat(episode % 2, lay.cells)
    res = {"population": {"windows": len(splits.val), "cells": int(pop["target"].size),
                          "persistence": float((pop["target"] == pop["prev"]).mean()),
                          "fold_cells": [int((fold == f).sum()) for f in (0, 1)]},
           "deltas": DELTAS.tolist(), "runs": {}}
    for ck in args.checkpoints:
        model, meta = de.load_checkpoint(ROOT / "runs" / ck, cfg, dev)
        r = {"step": meta["step"], **probe(cell_stats(model, splits, lay, mask, dev), pop, fold)}
        res["runs"][ck] = r
        c = r["copy_unless_confident"]
        print(f"{ck} (step {r['step']:,}): last-frame CE {r['last_frame_ce']:.4f}  greedy "
              f"{r['greedy_margin']:+.2f} pts  bets {r['bets']['share']:.1%} conf "
              f"{r['bets']['mean_conf']:.2f} hit {r['bets']['hit_rate']:.2f}  held-out gated "
              f"{c['held_out_margin']:+.3f} pts (delta {c['tune_0']['delta']}/{c['tune_1']['delta']})",
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
