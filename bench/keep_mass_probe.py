"""Why does greedy keep the old token on cells that truly change? (r77)

One forward pass over every val window from real context (gate row 1's population).
Per cell: p_keep = P(previous token), p_change = 1 - p_keep, the top non-keep code's
probability, and how many codes hold the change mass (exp of its entropy).
Reported on moving cells (pixel field changed) whose token truly changes, on moving
cells that truly keep, and on quiet cells that change (tokenizer flips). Separates two
causes of a kept token: the model rating keep likelier (p_change < 0.5), and change
likelier but split over several codes so keep still wins the argmax.

    python bench/keep_mass_probe.py RUN_ID/CHECKPOINT [...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from mirage import dynamics, dynamics_eval as de  # noqa: E402
import freeze_probe as fp  # noqa: E402



@torch.no_grad()
def stats(c, model, batch=64):
    codes = model.head.out_features
    tgt, prev = c.pop["target"], c.pop["prev"]
    keys = ("p_keep", "p_top_other", "p_true", "n_eff", "argmax_keep", "top_other_true")
    out = {k: np.empty((c.n, c.cells), np.float64) for k in keys}
    for j in range(0, c.n, batch):
        idx = range(j, min(c.n, j + batch))
        toks = np.stack([c.splits.val[i].tokens for i in idx])
        acts = np.stack([c.splits.val[i].actions for i in idx])
        x, _ = (torch.from_numpy(a).to(c.dev) for a in dynamics.assemble(c.lay, toks, acts, codes))
        with dynamics._autocast(c.dev):
            logits = model(x, c.mask, c.read).float()[:, -c.cells:]
        p = torch.softmax(logits, -1)
        pv = torch.from_numpy(prev[j:j + len(idx)]).to(c.dev)[..., None]
        tv = torch.from_numpy(tgt[j:j + len(idx)]).to(c.dev)[..., None]
        pk = p.gather(-1, pv)[..., 0]
        other = p.scatter(-1, pv, 0.0)
        q = other / other.sum(-1, keepdim=True).clamp_min(1e-12)
        ent = -(q * q.clamp_min(1e-12).log()).sum(-1)
        top = other.max(-1)
        s = slice(j, j + len(idx))
        out["p_keep"][s] = pk.cpu().numpy()
        out["p_top_other"][s] = top.values.cpu().numpy()
        out["p_true"][s] = p.gather(-1, tv)[..., 0].cpu().numpy()
        out["n_eff"][s] = ent.exp().cpu().numpy()
        out["argmax_keep"][s] = (p.argmax(-1) == pv[..., 0]).cpu().numpy()
        out["top_other_true"][s] = (top.indices == tv[..., 0]).cpu().numpy()
    return out


def report(label, m, sel):
    pk = m["p_keep"][sel]
    pc = 1 - pk
    print(f"  {label}: {sel.sum():,} cells")
    print(f"    mean p_keep {pk.mean():.3f}, mean p_change {pc.mean():.3f}, mean top other {m['p_top_other'][sel].mean():.3f}, "
          f"mean p_true {m['p_true'][sel].mean():.3f}")
    print(f"    argmax keeps {m['argmax_keep'][sel].mean():.1%}; p_change > p_keep {(pc > pk).mean():.1%}; "
          f"of those, argmax still keeps {m['argmax_keep'][sel][pc > pk].mean():.1%}")
    print(f"    codes holding the change mass (exp entropy): median {np.median(m['n_eff'][sel]):.1f}, "
          f"p90 {np.percentile(m['n_eff'][sel], 90):.1f}; top other code is the true one {m['top_other_true'][sel].mean():.1%}")
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        b = (pc >= lo) & (pc < hi)
        if b.any():
            print(f"    p_change {lo:.2f}-{min(hi, 1):.2f}: {b.mean():5.1%} of cells, argmax keeps {m['argmax_keep'][sel][b].mean():5.1%}, "
                  f"top other is true {m['top_other_true'][sel][b].mean():5.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="RUN_ID/best.pt or RUN_ID/model.pt under runs/")
    ap.add_argument("--config", type=Path, default=fp.CONFIG)
    args = ap.parse_args()
    c = fp.Ctx(args.config)
    tgt, prev, quiet = c.pop["target"], c.pop["prev"], c.pop["quiet"]
    changed = tgt != prev
    print(f"windows {c.n:,}; changing cells {changed.mean():.2%} (on moving cells {changed[~quiet].mean():.2%}, "
          f"on quiet cells {changed[quiet].mean():.2%})")
    for ck in args.checkpoints:
        model, meta = de.load_checkpoint(ROOT / "runs" / ck, c.cfg, c.dev)
        model.eval()
        m = stats(c, model)
        print(f"\n=== {ck} (step {meta['step']:,})")
        report("moving cells that truly change", m, changed & ~quiet)
        report("moving cells that truly keep", m, ~changed & ~quiet)
        report("quiet cells that change (tokenizer flips)", m, changed & quiet)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
