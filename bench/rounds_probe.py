"""Does decoding a frame in K masked rounds unfreeze the rollout, and does the arm then move right?

The K-round model (`dynamics.frame_decoding` "maskgit", `world_model_architecture.md`,
"Phase 2 decision: a frame is decoded in K masked rounds, not one pass") is rolled
out on the gate's population (`dynamics_eval.val_episodes`: every val episode from
frame 0, its own actions, to its end) at each K, and per K this prints:

- `freeze_probe.Steps`' token table - frozen share, changed cells, accuracy against
  the truth and against the seed frame held still, round-trip mismatch - with
  P(frozen next | frozen);
- gate row 6 - the frozen share of generated frames and each link's drift, against
  the truth's on the same frames, and its verdict;
- the pose table of `coord_decode_probe` - each link's on-screen angle error against
  the truth, degrees mod 180, beside the arm held at the seed's last frame;
- the link table - token accuracy on the cells each link covers in the truth's
  render (`dynamics_eval.link_cells`), beside holding the seed frame still on the
  same cells.

Gate row 1 is not here: it is K=1 by definition, `python -m mirage.dynamics --eval`.

    python bench/rounds_probe.py RUN_ID/best.pt [...] --rounds 1 2 4 8 16 [--temperature 0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from mirage import dynamics_eval as de, validator  # noqa: E402
import coord_decode_probe as cd  # noqa: E402
import freeze_probe as fp  # noqa: E402

CONFIG = ROOT / "mirage" / "configs" / "dyn192x4_mask.json"
LINK_BUCKETS = [(0, 25), (25, 100), (100, 200), (200, None)]


def link_table(gen: np.ndarray, truth: np.ndarray, cover: np.ndarray, ctx: int, names: list[str]) -> dict:
    """Accuracy on each link's truth cells, and the seed frame held still on the same cells, by step."""
    g, t, c = gen[:, ctx:], truth[:, ctx:], cover[:, ctx:]          # (E, S, cells), (E, S, links, cells)
    hit, still = (g == t)[:, :, None], (truth[:, ctx - 1:ctx] == t)[:, :, None]
    out = {}
    print(f"  {'steps':<9}" + "".join(f"{f'{n} acc / still':>22}" for n in names))
    for a, b in LINK_BUCKETS:
        s = slice(a, b)
        cells = c[:, s]
        acc = (hit[:, s] & cells).sum((0, 1, 3)) / cells.sum((0, 1, 3))
        held = (still[:, s] & cells).sum((0, 1, 3)) / cells.sum((0, 1, 3))
        lab = f"{a}-{(b or g.shape[1]) - 1}"
        out[lab] = {n: [float(acc[l]), float(held[l])] for l, n in enumerate(names)}
        print(f"  {lab:<9}" + "".join(f"{acc[l]:>13.2%} / {held[l]:>6.2%}" for l in range(len(names))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="RUN_ID/best.pt or RUN_ID/model.pt under runs/")
    ap.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=de.ROLLOUT_SEED, help="sampling seed, with --temperature")
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()
    c = fp.Ctx(args.config)
    st = fp.Steps(c)
    palette = validator.load_palette(ROOT / c.cfg.sim["scene_xml"])
    names = [palette.names[i] for i in palette.links]
    tau_px = c.cfg.validator["offpalette_tau"]
    eps, ctx = st.eps, st.ctx
    n, length = eps.actions.shape
    truth = de.truth_features(st.tok, c.cfg, eps, palette, c.dev)
    drift_truth = de.link_drift(truth.link_major, ctx).mean(0)
    frozen_truth = de.frozen_share(eps.tokens, ctx)
    cover = de.link_cells(de.val_pixels(c.splits), palette, st.grid)
    ratio = c.cfg.validator["rollout_frozen_ratio_max"]
    print(f"truth ({len(eps.ids)} val episodes, {st.steps} steps): frozen {frozen_truth:.1%}; "
          f"cells per frame covered by " + ", ".join(f"{nm} {cover[:, ctx:, l].sum(-1).mean():.1f}"
                                                      for l, nm in enumerate(names)))
    st.table(st.truth)
    cd.pose_table(truth.link_angle, truth.link_angle, ctx, drift_truth, drift_truth)
    for ck in args.checkpoints:
        model, meta = de.load_checkpoint(ROOT / "runs" / ck, c.cfg, c.dev)
        if model.mask_id is None:
            raise SystemExit(f"{ck} is a one-pass model; this probe sweeps K-round decoding")
        for k in args.rounds:
            g = torch.Generator(c.dev).manual_seed(args.seed) if args.temperature > 0 else None
            t0 = time.perf_counter()
            gen = de.rollout(model, eps.tokens[:, :ctx], eps.actions, st.steps, c.dev,
                             args.temperature, g, k).frames
            secs = time.perf_counter() - t0
            print(f"\n=== {ck} (step {meta['step']:,}) K {k} temperature {args.temperature:g} ({secs:.0f}s)")
            st.table(st.per_step(c, gen))
            px = de.decode_u8(st.tok, gen.reshape(-1, c.cells), st.grid, c.dev)
            f = de.features(px.reshape(n, length, *px.shape[1:]), palette, tau_px)
            drift = de.link_drift(f.link_major, ctx).mean(0)
            frozen = de.frozen_share(gen, ctx)
            ok = de.drift_verdict(drift, drift_truth, frozen, frozen_truth, ratio)
            print(f"  row 6: frozen {frozen:.1%} vs {frozen_truth:.1%} (bar {ratio:g}x); drift "
                  + " / ".join(f"{a:.1%} vs {b:.1%}" for a, b in zip(drift, drift_truth))
                  + f" -> {'PASS' if ok else 'FAIL'}")
            cd.pose_table(f.link_angle, truth.link_angle, ctx, drift, drift_truth)
            link_table(gen, eps.tokens, cover, ctx, names)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
