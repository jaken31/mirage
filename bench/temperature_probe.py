"""Does temperature sampling stop a dynamics rollout from freezing?

Decision 5 (`docs/phase2_structural_plan.md`) took greedy rollout decoding, with
one revisit trigger: evidence that greedy decoding collapses - a rollout that
freezes or falls into a short loop, not one that merely drifts. The trigger
fired on the first 10-epoch run (r63): `best.pt` freezes on 100% of generated
frames and `model.pt` on 44.2%, against the ground truth's own share on the
same frames. This probe measures whether sampling, rather than training, is
the lever. **It changes nothing in the gate**: `mirage.dynamics_eval.rollout`
stays greedy unless a caller passes a temperature, and only this file does.

Terms: a "frozen" frame repeats the frame before it token for token. A
"looping" frame is not frozen but repeats a frame 2..`LOOP_MAX` steps back. The
"changed-cell share" is the fraction of a frame's 64 cells whose token differs
from the frame before it - the truth's is the motion there really is, and a
rollout far above it is adding noise, not motion. "Truth" is the val episodes'
cached tokens (R1), and every pixel measure decodes tokens through
`Tokenizer.decode`, exactly as the gate does.

Per checkpoint and temperature, over the gate's own population (one rollout
per val episode from frame 0 at ctx 15, the episode's own actions, to its
end):

- frozen and looping shares, beside the truth's;
- the changed-cell share, beside the truth's;
- token accuracy against the truth on every generated frame, beside the seed's
  last frame held still (the score of a rollout that froze at once);
- gate row 6 itself: link drift over 200-step windows and the frozen-share
  clause, judged by `dynamics_eval.drift_verdict` at the config's bars.

Temperature 0 is greedy, and reproduces the gate's own row 6.

**The decision rule, fixed before the run.** Sampling clears the trigger at a
temperature when row 6 passes there and the looping share is no higher than
the truth's. If no temperature clears it on either checkpoint, greedy stays,
and the remedy is training (decision 4). If one does, reopening decision 5 is
the captain's call - it would turn gate row 9 into "identical at a fixed seed",
which this probe checks holds on CUDA. Row 1 is untouched either way: it
scores one-step predictions from true context, which decoding never reaches.

    python bench/temperature_probe.py --self-check      # no dataset, no GPU needed
    python bench/temperature_probe.py RUN_ID [--temperatures 0 0.5 0.7 1.0] [--seed 0]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import config, dynamics, dynamics_eval as de, fsq_eval, validator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mirage" / "configs" / "base.json"
TEMPERATURES = (0.0, 0.5, 0.7, 1.0)
LOOP_MAX = 8   # a "short loop" is a period of at most this many frames


def loop_share(frames: np.ndarray, ctx: int, longest: int = LOOP_MAX) -> float:
    """`(E, T, cells)` -> share of frames from `ctx` on that are not frozen but
    repeat a frame 2..`longest` steps back, token for token."""
    gen, n = frames[:, ctx:], frames.shape[1]
    hit = np.zeros(gen.shape[:2], bool)
    for p in range(2, longest + 1):
        s = max(ctx, p)   # the first frame with one p steps back
        if s >= n:
            break
        hit[:, s - ctx:] |= (frames[:, s:] == frames[:, s - p:n - p]).all(-1)
    frozen = (gen == frames[:, ctx - 1:-1]).all(-1)
    return float((hit & ~frozen).mean())


def changed_share(frames: np.ndarray, ctx: int) -> float:
    """`(E, T, cells)` -> mean share of cells whose token differs from the frame before, from `ctx` on."""
    return float((frames[:, ctx:] != frames[:, ctx - 1:-1]).mean())


def measure(frames: np.ndarray, truth: np.ndarray, ctx: int) -> dict:
    """The token measures of one rollout against the truth on the same episodes and frames."""
    return {"frozen": de.frozen_share(frames, ctx), "loop": loop_share(frames, ctx),
            "changed": changed_share(frames, ctx),
            "acc": float((frames[:, ctx:] == truth[:, ctx:]).mean())}


def run(run_id: str, temperatures: tuple[float, ...], seed: int,
        runs_dir: Path = ROOT / "runs") -> dict:
    cfg = config.load(CONFIG)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx, cells = cfg.data["ctx"], math.prod(cfg.shapes.token_grid)
    grid = tuple(cfg.shapes.token_grid)
    splits = dynamics.load_splits(cfg)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    tau, ratio = cfg.validator["offpalette_tau"], cfg.validator["rollout_frozen_ratio_max"]
    eps = de.val_episodes(splits, len(palette.blocks))
    n_eps, length = eps.actions.shape

    tok, _ = fsq_eval.load_run(dynamics.TOKENIZER_RUN, cfg, dev)
    truth_f = de.truth_features(tok, cfg, eps, palette, dev)
    drift_truth = de.link_drift(truth_f.link_major, ctx).mean(0)
    t_tok = measure(eps.tokens, eps.tokens, ctx)
    still = np.concatenate([eps.tokens[:, :ctx],
                            np.repeat(eps.tokens[:, ctx - 1:ctx], length - ctx, 1)], 1)
    still_acc = measure(still, eps.tokens, ctx)["acc"]

    rows = []
    for name in ("best", "final"):
        file = {"best": "best.pt", "final": "model.pt"}[name]
        model, ck = de.load_checkpoint(Path(runs_dir) / run_id / file, cfg, dev)
        for t in temperatures:
            gen = lambda: torch.Generator(dev).manual_seed(seed)  # noqa: E731
            t0 = time.perf_counter()
            frames = de.rollout(model, eps.tokens[:, :ctx], eps.actions, length - ctx, dev,
                                temperature=t, generator=gen() if t > 0 else None).frames
            secs = time.perf_counter() - t0
            again = de.rollout(model, eps.tokens[:, :ctx], eps.actions, length - ctx, dev,
                               temperature=t, generator=gen() if t > 0 else None).frames
            px = de.decode_u8(tok, frames.reshape(-1, cells), grid, dev)
            f = de.features(px.reshape(n_eps, length, *px.shape[1:]), palette, tau)
            drift = de.link_drift(f.link_major, ctx).mean(0)
            m = measure(frames, eps.tokens, ctx)
            row6 = de.drift_verdict(drift, drift_truth, m["frozen"], t_tok["frozen"], ratio)
            clears = row6 and m["loop"] <= t_tok["loop"]
            rows.append({"checkpoint": file, "step": ck["step"], "temperature": t, **m,
                         "drift": drift.tolist(), "row6": row6, "clears": clears,
                         "reproduced": bool(np.array_equal(frames, again)), "rollout_s": secs})
        del model

    print(f"\ntemperature probe - {run_id}, seed {seed}, {n_eps} val rollouts of {length - ctx} "
          f"steps at ctx {ctx}, dynamics_hash {cfg.dynamics_hash[:8]}")
    print(f"truth: frozen {t_tok['frozen']:.1%}, loop {t_tok['loop']:.2%}, changed "
          f"{t_tok['changed']:.2%}, drift " + " / ".join(f"{d:.1%}" for d in drift_truth)
          + f"; the seed held still scores {still_acc:.2%} token accuracy")
    print(f"{'ckpt':<9}{'T':>5} {'frozen':>8} {'loop':>7} {'changed':>8} {'acc':>8}  "
          f"{'drift':<26} {'row 6':<6} {'repro':<6} clears")
    for r in rows:
        print(f"{r['checkpoint']:<9}{r['temperature']:>5.2f} {r['frozen']:>8.1%} {r['loop']:>7.2%} "
              f"{r['changed']:>8.2%} {r['acc']:>8.2%}  "
              f"{' / '.join(f'{d:.1%}' for d in r['drift']):<26} "
              f"{'PASS' if r['row6'] else 'FAIL':<6} {'yes' if r['reproduced'] else 'NO':<6} "
              f"{'yes' if r['clears'] else 'no'}")
    cleared = [(r["checkpoint"], r["temperature"]) for r in rows if r["clears"]]
    print("\nsampling clears decision 5's trigger at: "
          + (", ".join(f"{c} T={t:g}" for c, t in cleared) if cleared else "no temperature")
          + f" (row 6 passes at {ratio:g}x and loops no more often than the truth)")
    return {"run_id": run_id, "seed": seed, "truth": {**t_tok, "drift": drift_truth.tolist()},
            "still_acc": still_acc, "rows": rows, "cleared": cleared}


def _self_check() -> None:
    """The token measures on synthetic rollouts: no dataset, no GPU."""
    ctx, cells = 3, 4
    base = np.arange(ctx * cells).reshape(1, ctx, cells)
    a, b, c = (np.full((1, 1, cells), v) for v in (100, 101, 102))
    frozen = np.concatenate([base, base[:, -1:], base[:, -1:]], 1)
    assert de.frozen_share(frozen, ctx) == 1.0 and loop_share(frozen, ctx) == 0.0
    assert changed_share(frozen, ctx) == 0.0
    loop = np.concatenate([base, a, b, a, b], 1)    # period 2 from the third generated frame
    assert loop_share(loop, ctx) == 0.5 and de.frozen_share(loop, ctx) == 0.0
    assert changed_share(loop, ctx) == 1.0
    long = np.concatenate([base] + [a, b, c] * 3 + [a], 1)   # period 3, from frame 4 of 10
    assert loop_share(long, ctx) == 0.7 and loop_share(long, ctx, longest=2) == 0.0
    half = np.concatenate([base, base[:, -1:].copy()], 1)
    half[0, ctx, :2] += 50
    m = measure(half, half, ctx)
    assert m["changed"] == 0.5 and m["acc"] == 1.0 and m["frozen"] == 0.0
    print(f"temperature probe: frozen, loop (period 2..{LOOP_MAX}, never a frozen frame), "
          f"changed-cell share and accuracy check out on synthetic rollouts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--temperatures", type=float, nargs="+", default=list(TEMPERATURES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", metavar="PATH", help="also write the result here")
    args = ap.parse_args()
    if args.self_check or not args.run_id:
        _self_check()
        return
    out = run(args.run_id, tuple(args.temperatures), args.seed)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
