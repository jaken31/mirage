"""Clock test: does the one-pass dynamics model learn to move when a frame is `k` physics steps?

`reports/Fix the clock before the model.md` found the arm's tip moves a median
0.13 px per stored frame. This trains the plain one-pass model (`dyn192x4`,
real context only, no rollout training, no flip cleaning) on every `k`-th frame
of a tokenizer's cache and scores greedy rollouts over the same simulated time
at every `k`.

Frames are taken at `step_idx % k == k - 1`, so with `k` dividing the action
hold every interval between two frames lies inside one hold, and the action
read at a frame's record is the one applied over the whole interval before it
(asserted).

Scored on the val episodes, per link, at about 50 and 250 ms of simulated time:
the correlation of the rollout's decoded angle change with the render's (beside
the decoded truth tokens', the tokenizer's ceiling), the angle error beside the
last context frame held still, the action gain (`bench/forearm_probe.py` part
6's), the same correlation under another episode's actions, and the share of
generated frames that repeat the frame before, beside the truth's. One-step
accuracy is reported as lift over copying the previous frame.

    python bench/stride_pilot.py --tokenizer RUN_ID --stride 5 --steps 30000 --out DIR
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirage import config, dynamics, dynamics_eval as de, fsq_eval, validator  # noqa: E402

CONFIG = ROOT / "mirage" / "configs" / "dyn192x4.json"
MIN_MAJOR = 3.0            # px: shortest blob whose angle is compared (forearm_probe's)
HORIZONS_MS = (50, 250)
STEP_MS = 2.0              # scene/arm_blocks.xml timestep


def strided(sampler: dynamics.TokenWindowSampler, k: int, hold: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per episode: (tokens (n, cells), actions (n,), frame indices (n,)) at every k-th frame."""
    out = []
    for ep in sampler.windows.episodes:
        sl = slice(ep.start, ep.start + ep.length)
        tok, act, step = sampler.tokens[ep.shard][sl], sampler.actions[ep.shard][sl], sampler.steps[ep.shard][sl]
        assert np.array_equal(step, np.arange(ep.length)), "an episode's step_idx is not 0..length-1"
        idx = np.arange(k - 1, ep.length, k)
        if hold % k == 0:   # one action per interval (idx[j-1], idx[j]]
            for j in range(1, len(idx)):
                assert (act[idx[j - 1] + 1:idx[j] + 1] == act[idx[j]]).all(), "an interval spans two actions"
        out.append((np.asarray(tok[idx]).astype(np.int64), np.asarray(act[idx]).astype(np.int64), idx))
    return out


def deg_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(de.wrapped(a - b))


@torch.no_grad()
def one_step(model, lay, mask, read, toks, acts, codes, dev, batch=256) -> dict:
    """Teacher-forced last-frame accuracy over windows, beside copying the frame before."""
    hit = keep = n = 0
    ce = 0.0
    for i in range(0, len(toks), batch):
        x, y = dynamics.assemble(lay, toks[i:i + batch], acts[i:i + batch], codes)
        with dynamics._autocast(dev):
            logits = model(torch.from_numpy(x).to(dev), mask, read).float()
        cells = lay.cells
        last, tgt = logits[:, -cells:], torch.from_numpy(y[:, -cells:]).to(dev)
        ce += float(torch.nn.functional.cross_entropy(last.flatten(0, 1), tgt.flatten(), reduction="sum"))
        hit += int((last.argmax(-1) == tgt).sum())
        keep += int((toks[i:i + batch, -1] == toks[i:i + batch, -2]).sum())
        n += tgt.numel()
    acc, pers = hit / n, keep / n
    return {"acc": acc, "persistence": pers, "lift": (acc - pers) / (1 - pers), "ce": ce / n, "cells": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="tokenizer run whose token cache is read")
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--steps", type=int, default=30000, help="optimizer steps in total")
    ap.add_argument("--batch", type=int, default=dynamics.BATCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="directory for result.json and the weights")
    args = ap.parse_args()
    k = args.stride
    cfg = config.load(str(CONFIG))
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ctx, hold = cfg.data["ctx"], cfg.sim["action_hold_steps"]
    grid, cells = tuple(cfg.shapes.token_grid), math.prod(cfg.shapes.token_grid)
    codes = cfg.tokenizer["codebook_size"]
    splits = dynamics.load_splits(cfg, args.tokenizer)
    train, val = strided(splits.train, k, hold), strided(splits.val, k, hold)
    n = len(val[0][0])
    if n < ctx + 2:
        raise SystemExit(f"stride {k} leaves {n} frames an episode, under a {ctx + 1}-frame window")

    # every window start of every train episode
    starts = np.array([(e, j) for e, (t, _, _) in enumerate(train) for j in range(len(t) - ctx)])
    ttok = np.stack([t for t, _, _ in train])            # (E, n, cells): episodes share one length
    tact = np.stack([a for _, a, _ in train])
    win = np.arange(ctx + 1)

    def batch_of(rows):
        e, j = rows[:, 0], rows[:, 1]
        return ttok[e[:, None], j[:, None] + win], tact[e[:, None], j[:, None] + win]

    model = dynamics.build(cfg, dropout=dynamics.DROPOUT).to(dev)
    lay = dynamics.layout(ctx, cells)
    mask = dynamics.attention_mask(lay).to(dev)
    read = torch.from_numpy(lay.read).to(dev)
    groups = [[p for p in model.parameters() if p.dim() >= 2], [p for p in model.parameters() if p.dim() < 2]]
    opt = torch.optim.AdamW([{"params": groups[0], "weight_decay": dynamics.WEIGHT_DECAY},
                             {"params": groups[1], "weight_decay": 0.0}], lr=dynamics.LR)
    total, warm = args.steps, max(1, int(dynamics.WARMUP * args.steps))

    def lr_at(s):
        if s < warm:
            return dynamics.LR * (s + 1) / warm
        p = (s - warm) / max(1, total - warm)
        return dynamics.LR_FLOOR + 0.5 * (dynamics.LR - dynamics.LR_FLOOR) * (1 + math.cos(math.pi * p))

    vtok = np.stack([t for t, _, _ in val])
    vact = np.stack([a for _, a, _ in val])
    vw = np.array([(e, j) for e in range(len(val)) for j in range(n - ctx)])
    vwt, vwa = vtok[vw[:, 0][:, None], vw[:, 1][:, None] + win], vact[vw[:, 0][:, None], vw[:, 1][:, None] + win]

    per_epoch = len(starts) // args.batch
    print(f"stride {k} ({k * STEP_MS:g} ms/frame), tokenizer {args.tokenizer}: {len(train)} train episodes x {n} "
          f"frames, {len(starts):,} windows, {per_epoch} steps/epoch, {total:,} steps "
          f"({total / per_epoch:.1f} epochs); {len(vw):,} val windows", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    best, curve, t0, step = None, [], time.perf_counter(), 0
    every = max(per_epoch, 1000) if per_epoch < 5000 else 5000
    model.train()
    while step < total:
        order = rng.permutation(len(starts))
        for b in range(per_epoch):
            if step >= total:
                break
            tk, ac = batch_of(starts[order[b * args.batch:(b + 1) * args.batch]])
            x, y = dynamics.assemble(lay, tk, ac, codes)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with dynamics._autocast(dev):
                loss = dynamics.frame_loss(model, torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), mask, read)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % every == 0 or step == total:
                model.eval()
                s = one_step(model, lay, mask, read, vwt, vwa, codes, dev)
                model.train()
                tl = loss.item()
                curve.append({"step": step, "train_loss": tl, **s})
                print(f"  step {step:>6,}  train {tl:.4f}  val ce(last) {s['ce']:.4f}  acc {s['acc']:.2%}  "
                      f"copy {s['persistence']:.2%}  lift {s['lift']:+.3f}  {time.perf_counter() - t0:.0f}s", flush=True)
                if best is None or s["ce"] < best["ce"]:
                    best = {**s, "step": step}
                    torch.save(model.state_dict(), args.out / "best.pt")
    torch.save(model.state_dict(), args.out / "final.pt")
    train_s = time.perf_counter() - t0

    # ---------------------------------------------------------------- scoring
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    names = [palette.names[i] for i in palette.links]
    tau = cfg.validator["offpalette_tau"]
    tok, _ = fsq_eval.load_run(args.tokenizer, cfg, dev)
    idx = val[0][2]
    renders = de.val_pixels(splits)[:, idx]
    E = len(val)
    rf = de.features(renders, palette, tau)
    px = de.decode_u8(tok, vtok.reshape(-1, cells), grid, dev)
    df = de.features(px.reshape(E, n, *px.shape[1:]), palette, tau)
    cmd = de.COMMAND[vact]                                  # (E, n, joints)
    truth_frozen = float((vtok[:, ctx:] == vtok[:, ctx - 1:-1]).all(-1).mean())
    hs = sorted({max(1, round(ms / STEP_MS / k)) for ms in HORIZONS_MS})
    H = hs[-1]
    rstarts = np.arange(ctx, n - H + 1, max(1, H // 5))
    perm = np.roll(np.arange(E), 1)

    def feats(ids):
        p = de.decode_u8(tok, ids.reshape(-1, cells), grid, dev)
        return de.features(p.reshape(*ids.shape[:-1], *p.shape[1:]), palette, tau)

    result = {"stride": k, "ms_per_frame": k * STEP_MS, "tokenizer": args.tokenizer, "steps": total,
              "epochs": total / per_epoch, "train_windows": int(len(starts)), "train_s": train_s,
              "truth_frozen": truth_frozen, "best_by_val_ce": best, "curve": curve, "checkpoints": {}}
    for name in ("best", "final"):
        model.load_state_dict(torch.load(args.out / f"{name}.pt", weights_only=True))
        model.eval()
        roll = np.empty((E, len(rstarts), H, cells), np.int64)
        shuf = np.empty((E, len(rstarts), cells), np.int64)
        for i, s in enumerate(rstarts):
            roll[:, i] = de.rollout(model, vtok[:, s - ctx:s], vact[:, s - ctx:s + H], H, dev).frames[:, ctx:]
            acts = np.concatenate((vact[:, s - ctx:s], vact[perm, s:s + H]), 1)
            shuf[:, i] = de.rollout(model, vtok[:, s - ctx:s], acts, H, dev).frames[:, -1]
        prev = np.concatenate((np.stack([vtok[:, s - 1] for s in rstarts], 1)[:, :, None], roll[:, :, :-1]), 2)
        frozen = float((roll == prev).all(-1).mean())
        tr = np.stack([vtok[:, s:s + H] for s in rstarts], 1)
        tprev = np.stack([vtok[:, s - 1:s + H - 1] for s in rstarts], 1)
        frozen_truth_same = float((tr == tprev).all(-1).mean())
        r = {"one_step": one_step(model, lay, mask, read, vwt, vwa, codes, dev), "frozen": frozen,
             "frozen_truth_same_frames": frozen_truth_same, "horizons": {}}
        print(f"\n=== stride {k}, {name}: one-step acc {r['one_step']['acc']:.2%} vs copy "
              f"{r['one_step']['persistence']:.2%} (lift {r['one_step']['lift']:+.3f}); frozen rollout frames "
              f"{frozen:.1%} vs truth {frozen_truth_same:.1%} ({frozen / max(frozen_truth_same, 1e-9):.2f}x)")
        print(f"    {'ms':>5}{'h':>4}{'link':>7}{'ang err':>9}{'still':>8}{'model r':>9}{'tok r':>8}{'shuffled':>10}"
              f"{'gain':>7}{'n':>6}")
        last = rstarts - 1
        sf = feats(shuf)
        for h in hs:
            rl = feats(roll[:, :, h - 1])
            end = rstarts + h - 1
            csum = np.stack([cmd[:, s:s + h].sum(1) for s in rstarts], 1).astype(np.float64)
            r["horizons"][h] = {"ms": h * k * STEP_MS}
            for l, nm in enumerate(names):
                pa, pm = rl.link_angle[..., l], rl.link_major[..., l]
                ta, tm = df.link_angle[:, end, l], df.link_major[:, end, l]
                ha, hm = df.link_angle[:, last, l], df.link_major[:, last, l]
                ra, rm = rf.link_angle[:, end, l], rf.link_major[:, end, l]
                rh, rhm = rf.link_angle[:, last, l], rf.link_major[:, last, l]
                ok = (pm >= MIN_MAJOR) & (tm >= MIN_MAJOR) & (hm >= MIN_MAJOR) & (rm >= MIN_MAJOR) & (rhm >= MIN_MAJOR)
                if h == H:
                    ok &= sf.link_major[..., l] >= MIN_MAJOR
                dr, dm, dt = deg_diff(ra, rh)[ok], deg_diff(pa, ha)[ok], deg_diff(ta, ha)[ok]
                x = np.column_stack((csum[ok], np.ones(int(ok.sum()))))
                br = np.linalg.lstsq(x, dr, rcond=None)[0]
                bm = np.linalg.lstsq(x, dm, rcond=None)[0]
                row = {"ang_err_deg": float(np.abs(deg_diff(pa, ta))[ok].mean()),
                       "still_ang_err_deg": float(np.abs(deg_diff(ha, ta))[ok].mean()),
                       "model_corr": float(np.corrcoef(dm, dr)[0, 1]), "tok_corr": float(np.corrcoef(dt, dr)[0, 1]),
                       "action_gain": float(bm[:-1] @ br[:-1] / (br[:-1] @ br[:-1])), "n": int(ok.sum())}
                if h == H:
                    row["shuffled_corr"] = float(np.corrcoef(deg_diff(sf.link_angle[..., l], ha)[ok], dr)[0, 1])
                r["horizons"][h][nm] = row
                sh = f"{row['shuffled_corr']:>10.3f}" if h == H else f"{'-':>10}"
                print(f"    {h * k * STEP_MS:>5g}{h:>4}{nm:>7}{row['ang_err_deg']:>9.2f}{row['still_ang_err_deg']:>8.2f}"
                      f"{row['model_corr']:>9.3f}{row['tok_corr']:>8.3f}{sh}{row['action_gain']:>7.2f}{row['n']:>6}")
        result["checkpoints"][name] = r
        sys.stdout.flush()
    (args.out / "result.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
