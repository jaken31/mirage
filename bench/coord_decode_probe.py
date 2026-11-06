"""Do rollouts unfreeze when a frame's cells move together, and does the arm then move right?

r75 found greedy decoding moves only part of the arm, because each of a frame's 64
cells takes its own argmax. This probe changes decoding only, never the model:
each step, per cell, p_keep is the probability of the previous frame's token. The
frame moves when sum(1 - p_keep) >= 1 (the model expects at least one changed cell);
on a moving frame every cell with p_keep < tau takes its best token other than the
previous one, and the rest take the argmax. tau 0 is the gate's greedy decoding
(decision 5), which `dynamics_eval.rollout` keeps.

On the gate's rollout population (`dynamics_eval.val_episodes`), per tau:

- the token table of `freeze_probe.Steps` - frozen share, changed cells, accuracy
  against the truth and the seed frame held still, round-trip mismatch;
- the pose table - each link's on-screen angle error against the truth, in degrees
  mod 180 (`dynamics_eval.wrapped`), beside the arm held at the seed's last frame,
  and gate row 6's link drift against the truth's.

    python bench/coord_decode_probe.py RUN_ID/CHECKPOINT [...] [--taus 0 0.5 0.7 0.9]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from mirage import dynamics, dynamics_eval as de, validator  # noqa: E402
import freeze_probe as fp  # noqa: E402

MIN_MOVE = 1.0
POSE_BUCKETS = [(0, 5), (5, 25), (25, 100), (100, 200), (200, None)]


@torch.no_grad()
def rollout(model, seed: np.ndarray, actions: np.ndarray, steps: int, dev: torch.device,
            tau: float) -> np.ndarray:
    """`dynamics_eval.rollout`'s loop with the coordinated pick; tau 0 is its greedy pick."""
    b, ctx, cells = seed.shape
    codes = model.head.out_features
    lay = dynamics.layout(ctx, cells)
    mask = dynamics.attention_mask(lay).to(dev)
    read = torch.from_numpy(lay.read).to(dev)
    frames = np.zeros((b, ctx + steps, cells), np.int64)
    frames[:, :ctx] = seed
    model.eval()
    for k in range(steps):
        x, _ = dynamics.assemble(lay, frames[:, k:k + ctx + 1], actions[:, k:k + ctx + 1], codes)
        with dynamics._autocast(dev):
            logits = model(torch.from_numpy(x).to(dev), mask, read)
        last = logits.float()[:, -cells:]
        pick = last.argmax(-1)
        if tau > 0:
            prev = torch.from_numpy(frames[:, ctx + k - 1]).to(dev)
            p_keep = torch.softmax(last, -1).gather(-1, prev[..., None])[..., 0]
            moves = (1 - p_keep).sum(-1, keepdim=True) >= MIN_MOVE
            alt = last.scatter(-1, prev[..., None], float("-inf")).argmax(-1)
            pick = torch.where(moves & (p_keep < tau), alt, pick)
        frames[:, ctx + k] = pick.cpu().numpy()
    return frames


def pose_table(ang: np.ndarray, truth: np.ndarray, ctx: int, drift: np.ndarray,
               drift_truth: np.ndarray) -> None:
    err = np.degrees(np.abs(de.wrapped(ang[:, ctx:] - truth[:, ctx:])))
    still = np.degrees(np.abs(de.wrapped(truth[:, ctx - 1:ctx] - truth[:, ctx:])))
    links = err.shape[-1]
    print(f"  {'steps':<9}" + "".join(f"{f'link{l} err / still':>20}" for l in range(links)))
    for a, b in POSE_BUCKETS:
        s = slice(a, b)
        print(f"  {f'{a}-{(b or err.shape[1]) - 1}':<9}"
              + "".join(f"{np.nanmean(err[:, s, l]):>11.1f} / {np.nanmean(still[:, s, l]):>5.1f}"
                        for l in range(links)))
    print(f"  missing link angle {np.isnan(ang[:, ctx:]).mean():.1%}; row 6 drift "
          + " / ".join(f"{g:.1%} vs {t:.1%}" for g, t in zip(drift, drift_truth)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="RUN_ID/best.pt or RUN_ID/model.pt under runs/")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.5, 0.7, 0.9])
    ap.add_argument("--config", type=Path, default=fp.CONFIG)
    args = ap.parse_args()
    c = fp.Ctx(args.config)
    st = fp.Steps(c)
    palette = validator.load_palette(ROOT / c.cfg.sim["scene_xml"])
    tau_px = c.cfg.validator["offpalette_tau"]
    eps, ctx = st.eps, st.ctx
    n, length = eps.actions.shape
    truth = de.truth_features(st.tok, c.cfg, eps, palette, c.dev)
    drift_truth = de.link_drift(truth.link_major, ctx).mean(0)
    print(f"truth ({len(eps.ids)} val episodes, {st.steps} steps)")
    st.table(st.truth)
    pose_table(truth.link_angle, truth.link_angle, ctx, drift_truth, drift_truth)
    for ck in args.checkpoints:
        model, meta = de.load_checkpoint(ROOT / "runs" / ck, c.cfg, c.dev)
        for tau in args.taus:
            gen = rollout(model, eps.tokens[:, :ctx], eps.actions, st.steps, c.dev, tau)
            print(f"\n=== {ck} (step {meta['step']:,}) tau {tau:g}")
            st.table(st.per_step(c, gen))
            px = de.decode_u8(st.tok, gen.reshape(-1, c.cells), st.grid, c.dev)
            f = de.features(px.reshape(n, length, *px.shape[1:]), palette, tau_px)
            pose_table(f.link_angle, truth.link_angle, ctx, de.link_drift(f.link_major, ctx).mean(0), drift_truth)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
