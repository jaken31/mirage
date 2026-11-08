"""Is the forearm's missing motion the tokenizer's limit or the model's? Four parts, no training.

r81 left two branches of the decision rule matching: the model's prediction is
the limit (Copilot4D-style additions), or the tokenizer is (the forearm covers
about 6 of a frame's 64 cells, maybe too few tokens to show it move). Each part
runs on the gate's rollout population (`dynamics_eval.val_episodes`: 27 val
episodes, 585 steps) and the forearm is link 1.

1. **round trip** - each link's on-screen angle (`validator.measure_pixels_only`)
   on the renders against the same measurement on the cached truth tokens
   decoded: the tokenizer's own pose error, beside the per-step angle change it
   must show, and how often a link in the render is missing once decoded.
2. **resolvability** - consecutive truth steps binned by the render's forearm
   angle change: per bin, the share of steps where any forearm cell's token
   changes, and how often the decoded angle change has the render's sign. The
   noise floor is the render angle change on steps where neither joint moved.
3. **one step** - every frame from step ctx on predicted from its real context
   (teacher-forced, one round), greedy and at temperature 1.0: token accuracy
   on each link's truth cells against the frame before held still, and the
   decoded angle error against held still.
4. **action** - the same step with the target frame's action flipped on the
   forearm joint (only steps whose action drives it): the share of each link's
   cells whose argmax moves, and the mean KL between the two predictions.

5. **rollout** - one-step forearm motion is under a pixel (part 2), so the target
   is also read where the tokens do carry it: greedy open-loop rollouts of
   `ROLL_H` frames from real context every `ROLL_EVERY` frames, with the true
   actions. At each horizon h, each link's decoded angle against the decoded
   truth's, beside the last context frame held still, and the correlation of
   the model's angle change since that frame with the render's - next to the
   same correlation for the decoded truth tokens on the same samples, the
   tokenizer's ceiling.

An angle is only compared where the link's blob is at least `MIN_MAJOR` px long
in every frame involved; a 1-2 px blob has no meaningful orientation.

    python bench/forearm_probe.py RUN_ID/best.pt [...]
    python bench/forearm_probe.py --skip-model --tokenizer TOK_RUN --config mirage/configs/base96.json
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

CONFIG = ROOT / "mirage" / "configs" / "dyn192x4_mask.json"
MIN_MAJOR = 3.0       # px: shortest blob whose angle is compared
STILL_QPOS = 1e-4     # rad: a joint that moved less than this in a step is still
BINS_DEG = [0, 0.5, 1, 2, 4, 8, 90]
HORIZONS = [1, 5, 25, 100]
SAMPLE_SEED = 0
ROLL_H = 100          # frames per rollout (part 5)
ROLL_EVERY = 25       # frames between rollout starts
FORE, BASE = 1, 0


def deg_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(de.wrapped(a - b))


def flip_forearm(actions: np.ndarray) -> np.ndarray:
    """Each action with its forearm-joint command negated; the base command kept."""
    cmd = de.COMMAND[actions]                    # (..., joints) in -1/0/1
    cmd[..., FORE] *= -1
    return ((cmd[..., 0] + 1) + 3 * (cmd[..., 1] + 1)).astype(np.int64)


@torch.no_grad()
def onestep(model, tokens: np.ndarray, actions: np.ndarray, ctx: int, dev: torch.device) -> dict:
    """Every frame ctx.. from its real context in one round: greedy, T=1 sample, and the flip's effect."""
    e, length, cells = tokens.shape
    codes = model.head.out_features
    lay = dynamics.scoring_layout(model, ctx, cells)
    mask = dynamics.attention_mask(lay).to(dev)
    read = torch.from_numpy(lay.read).to(dev)
    pos = torch.from_numpy(lay.pos).to(dev)
    flipped = flip_forearm(actions)
    gen = torch.Generator(dev).manual_seed(SAMPLE_SEED)
    steps = length - ctx
    greedy, sample, flip_greedy = (np.empty((e, steps, cells), np.int64) for _ in range(3))
    kl = np.empty((e, steps, cells), np.float32)

    def logits_for(acts, k):
        win = tokens[:, k:k + ctx + 1].copy()
        win[:, -1] = 0                               # the target is never input; zero it as rollout does
        x, _ = dynamics.assemble(lay, win, acts[:, k:k + ctx + 1], codes)
        x = torch.from_numpy(x).to(dev)
        q = torch.full((e, cells), model.mask_id, dtype=x.dtype, device=dev)
        with dynamics._autocast(dev):
            return model(torch.cat((x, q), 1), mask, read, pos).float()

    for k in range(steps):
        real = logits_for(actions, k)
        acts = actions.copy()
        acts[:, ctx + k] = flipped[:, ctx + k]
        flip = logits_for(acts, k)
        greedy[:, k] = real.argmax(-1).cpu().numpy()
        flip_greedy[:, k] = flip.argmax(-1).cpu().numpy()
        probs = torch.softmax(real.reshape(-1, codes), -1)
        sample[:, k] = torch.multinomial(probs, 1, generator=gen).reshape(e, cells).cpu().numpy()
        lp, lq = torch.log_softmax(real, -1), torch.log_softmax(flip, -1)
        kl[:, k] = (lp.exp() * (lp - lq)).sum(-1).cpu().numpy()
    return {"greedy": greedy, "sample": sample, "flip_greedy": flip_greedy, "kl": kl}


def rollouts(model, tokens: np.ndarray, actions: np.ndarray, ctx: int,
             dev: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Greedy `ROLL_H`-frame rollouts from each start: (starts, frames (E, starts, ROLL_H, cells))."""
    e, length, cells = tokens.shape
    starts = np.arange(ctx, length - ROLL_H + 1, ROLL_EVERY)
    out = np.empty((e, len(starts), ROLL_H, cells), np.int64)
    for i, s in enumerate(starts):
        r = de.rollout(model, tokens[:, s - ctx:s], actions[:, s - ctx:s + ROLL_H], ROLL_H, dev)
        out[:, i] = r.frames[:, ctx:]
    return starts, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="*", help="RUN_ID/best.pt under runs/ (none with --skip-model)")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--json", type=Path, help="write every number printed here")
    ap.add_argument("--skip-model", action="store_true", help="parts 1-2 only")
    ap.add_argument("--tokenizer", default=dynamics.TOKENIZER_RUN,
                    help="tokenizer run whose token cache parts 1-2 read (another resolution: pass its "
                         "base config as --config, with --skip-model)")
    args = ap.parse_args()
    out: dict = {}
    cfg = config.load(str(args.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = dynamics.load_splits(cfg, args.tokenizer)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    names = [palette.names[i] for i in palette.links]
    tau = cfg.validator["offpalette_tau"]
    grid, cells = tuple(cfg.shapes.token_grid), math.prod(cfg.shapes.token_grid)
    ctx = cfg.data["ctx"]
    eps = de.val_episodes(splits, len(palette.blocks))
    tok, _ = fsq_eval.load_run(args.tokenizer, cfg, dev)
    e, length = eps.actions.shape

    renders = de.val_pixels(splits)
    rf = de.features(renders, palette, tau)
    df = de.truth_features(tok, cfg, eps, palette, dev)
    cover = de.link_cells(renders, palette, grid)                   # (E, T, links, cells)

    # ---------------------------------------------------------------- part 1
    print(f"=== part 1: round trip ({e} episodes x {length} frames, angles where the blob is >= {MIN_MAJOR:g} px)")
    p1 = {}
    for l, nm in enumerate(names):
        ok = (rf.link_major[..., l] >= MIN_MAJOR) & (df.link_major[..., l] >= MIN_MAJOR)
        err = np.abs(deg_diff(df.link_angle[..., l], rf.link_angle[..., l]))[ok]
        step_ok = (rf.link_major[:, 1:, l] >= MIN_MAJOR) & (rf.link_major[:, :-1, l] >= MIN_MAJOR)
        dstep = np.abs(deg_diff(rf.link_angle[:, 1:, l], rf.link_angle[:, :-1, l]))[step_ok]
        present = rf.link_major[..., l] > 0
        lost = (present & (df.link_major[..., l] == 0)).sum() / present.sum()
        p1[nm] = {"rt_err_deg": {"median": float(np.median(err)), "mean": float(err.mean()),
                                 "p90": float(np.percentile(err, 90))},
                  "step_deg": {"median": float(np.median(dstep)), "mean": float(dstep.mean()),
                               "p90": float(np.percentile(dstep, 90))},
                  "lost_share": float(lost), "present_share": float(present.mean()),
                  "cells_per_frame": float(cover[:, :, l].sum(-1).mean())}
        r = p1[nm]
        print(f"  {nm}: round-trip angle error median {r['rt_err_deg']['median']:.2f} deg, mean "
              f"{r['rt_err_deg']['mean']:.2f}, p90 {r['rt_err_deg']['p90']:.2f}; per-step render change median "
              f"{r['step_deg']['median']:.2f}, mean {r['step_deg']['mean']:.2f}, p90 {r['step_deg']['p90']:.2f}; "
              f"in render {r['present_share']:.1%} of frames, lost once decoded {r['lost_share']:.2%}; "
              f"{r['cells_per_frame']:.1f} cells/frame")
    out["part1"] = p1

    # ---------------------------------------------------------------- part 2
    print("\n=== part 2: do forearm motions reach the tokens? (truth, consecutive steps)")
    dq = np.abs(np.diff(eps.qpos, axis=1))                           # (E, T-1, joints)
    still = (dq < STILL_QPOS).all(-1)
    fr = rf.link_major[..., FORE]
    vis = (fr[:, 1:] >= MIN_MAJOR) & (fr[:, :-1] >= MIN_MAJOR)
    dr = deg_diff(rf.link_angle[:, 1:, FORE], rf.link_angle[:, :-1, FORE])
    dd = deg_diff(df.link_angle[:, 1:, FORE], df.link_angle[:, :-1, FORE])
    dvis = vis & (df.link_major[:, 1:, FORE] >= MIN_MAJOR) & (df.link_major[:, :-1, FORE] >= MIN_MAJOR)
    fcells = cover[:, 1:, FORE] | cover[:, :-1, FORE]                # forearm cells in either frame
    tchg = eps.tokens[:, 1:] != eps.tokens[:, :-1]
    any_fore = (tchg & fcells).any(-1)
    has_fore = fcells.any(-1)
    noise = float(np.percentile(np.abs(dr[still & vis]), 95)) if (still & vis).any() else float("nan")
    still_any = float(any_fore[still & has_fore].mean())
    print(f"  still-arm steps (both joints < {STILL_QPOS:g} rad): {int((still & vis).sum())}; render angle "
          f"change p95 {noise:.2f} deg (the noise floor); any forearm token changes on {still_any:.1%} of them")
    rows = []
    print(f"  {'render |d angle|':<18}{'steps':>7}{'any fore tok chg':>18}{'cells chg / fore':>18}"
          f"{'sign agrees':>13}{'|d decoded| / |d render|':>26}")
    for a, b in zip(BINS_DEG[:-1], BINS_DEG[1:]):
        sel = vis & has_fore & (np.abs(dr) >= a) & (np.abs(dr) < b) & ~still
        n = int(sel.sum())
        if not n:
            continue
        frac_cells = float(((tchg & fcells).sum(-1)[sel] / fcells.sum(-1)[sel]).mean())
        sg = sel & dvis & (np.abs(dr) > noise)
        agree = float((np.sign(dd[sg]) == np.sign(dr[sg])).mean()) if sg.any() else float("nan")
        ratio = float(np.median(np.abs(dd[sel & dvis]) / np.maximum(np.abs(dr[sel & dvis]), 1e-9)))
        rows.append({"bin_deg": [a, b], "steps": n, "any_fore_change": float(any_fore[sel].mean()),
                     "cell_change_share": frac_cells, "sign_agree": agree, "mag_ratio_median": ratio})
        print(f"  {f'{a:g}-{b:g}':<18}{n:>7}{any_fore[sel].mean():>18.1%}{frac_cells:>18.1%}"
              f"{agree:>13.1%}{ratio:>26.2f}")
    above = vis & has_fore & ~still & (np.abs(dr) > noise)
    no_tok = float(1 - any_fore[above].mean())
    print(f"  steps with forearm motion above the noise floor: {int(above.sum())}; "
          f"no forearm token changes on {no_tok:.1%} of them")
    out["part2"] = {"noise_floor_deg_p95": noise, "still_steps": int((still & vis).sum()),
                    "still_any_fore_change": still_any, "bins": rows,
                    "above_noise_steps": int(above.sum()), "above_noise_no_token_change": no_tok}

    # Per-step motion is under a pixel (0.4 deg turns a ~20 px link's tip ~0.14 px), so the
    # tokenizer is also read over the horizons the rollout is scored on: does the decoded
    # angle change over h steps follow the render's, for each link?
    print(f"\n  angle change over h steps, decoded truth against render (both links, blobs >= {MIN_MAJOR:g} px):")
    print(f"  {'h':>5}" + "".join(f"{f'{nm} |d| med / sign / r':>30}" for nm in names))
    hz = {}
    for h in HORIZONS:
        cols, hz[h] = [], {}
        for l, nm in enumerate(names):
            ok = ((rf.link_major[:, h:, l] >= MIN_MAJOR) & (rf.link_major[:, :-h, l] >= MIN_MAJOR)
                  & (df.link_major[:, h:, l] >= MIN_MAJOR) & (df.link_major[:, :-h, l] >= MIN_MAJOR))
            a = deg_diff(rf.link_angle[:, h:, l], rf.link_angle[:, :-h, l])[ok]
            b = deg_diff(df.link_angle[:, h:, l], df.link_angle[:, :-h, l])[ok]
            mv = np.abs(a) > 0
            sign = float((np.sign(a[mv]) == np.sign(b[mv])).mean())
            r = float(np.corrcoef(a, b)[0, 1])
            hz[h][nm] = {"render_abs_median_deg": float(np.median(np.abs(a))), "sign_agree": sign, "corr": r}
            cols.append(f"{np.median(np.abs(a)):>14.2f} / {sign:>5.1%} / {r:>5.3f}")
        print(f"  {h:>5}" + "".join(f"{c:>30}" for c in cols))
    out["part2"]["horizons"] = hz
    if args.skip_model:
        if args.json:
            args.json.write_text(json.dumps(out, indent=1))
        return

    # ------------------------------------------------------------- parts 3, 4
    truth = eps.tokens[:, ctx:]
    held = eps.tokens[:, ctx - 1:-1]
    tc = cover[:, ctx:]                                              # truth cells of each target frame
    chg = truth != held
    t_ang, t_maj = df.link_angle[:, ctx:], df.link_major[:, ctx:]
    h_ang, h_maj = df.link_angle[:, ctx - 1:-1], df.link_major[:, ctx - 1:-1]
    drives = de.COMMAND[eps.actions[:, ctx:]][..., FORE] != 0        # (E, steps)
    out["part3"], out["part4"] = {}, {}
    for ck in args.checkpoints:
        model, meta = de.load_checkpoint(ROOT / "runs" / ck, cfg, dev)
        # Every tokenizer shares one tokenizer_hash, so the hash check cannot catch a
        # model read through another tokenizer's cache; the run id can.
        if meta["knobs"]["tokenizer_run"] != args.tokenizer:
            raise SystemExit(f"{ck} was trained on {meta['knobs']['tokenizer_run']}'s tokens, "
                             f"not --tokenizer {args.tokenizer}")
        if model.mask_id is None:
            raise SystemExit(f"{ck} is a one-pass model; this probe reads the K-round model's one round")
        t0 = time.perf_counter()
        r = onestep(model, eps.tokens, eps.actions, ctx, dev)
        print(f"\n=== {ck} (step {meta['step']:,}), one step from real context ({time.perf_counter() - t0:.0f}s)")
        p3 = {}
        for mode in ("greedy", "sample"):
            pred = r[mode]
            hit = pred == truth
            px = de.decode_u8(tok, pred.reshape(-1, cells), grid, dev)
            pf = de.features(px.reshape(e, -1, *px.shape[1:]), palette, tau)
            p3[mode] = {}
            print(f"  {mode} (T {'0' if mode == 'greedy' else '1.0'}):")
            print(f"    {'link':<8}{'acc all':>9}{'still':>8}{'acc chg':>9}{'acc keep':>10}"
                  f"{'pred chg':>10}{'true chg':>10}{'ang err':>9}{'still':>8}{'lost':>7}")
            for l, nm in enumerate(names):
                c = tc[:, :, l]
                acc, st = hit[c].mean(), (~chg)[c].mean()
                acc_c, acc_k = hit[c & chg].mean(), hit[c & ~chg].mean()
                pch, tch = (pred != held)[c].mean(), chg[c].mean()
                ok = (t_maj[..., l] >= MIN_MAJOR) & (h_maj[..., l] >= MIN_MAJOR) & (pf.link_major[..., l] >= MIN_MAJOR)
                err = np.abs(deg_diff(pf.link_angle[..., l], t_ang[..., l]))[ok].mean()
                serr = np.abs(deg_diff(h_ang[..., l], t_ang[..., l]))[ok].mean()
                lost = ((t_maj[..., l] > 0) & (pf.link_major[..., l] == 0)).sum() / (t_maj[..., l] > 0).sum()
                p3[mode][nm] = {"acc": float(acc), "still_acc": float(st), "acc_changing": float(acc_c),
                                "acc_keeping": float(acc_k), "pred_change": float(pch), "true_change": float(tch),
                                "ang_err_deg": float(err), "still_ang_err_deg": float(serr), "lost": float(lost)}
                print(f"    {nm:<8}{acc:>9.2%}{st:>8.2%}{acc_c:>9.2%}{acc_k:>10.2%}{pch:>10.2%}{tch:>10.2%}"
                      f"{err:>9.2f}{serr:>8.2f}{lost:>7.2%}")
        out["part3"][ck] = p3
        p4 = {}
        print(f"  action flipped on the forearm joint ({int(drives.sum())} steps whose action drives it):")
        for l, nm in enumerate(names):
            c = tc[:, :, l] & drives[..., None]
            moved = (r["greedy"] != r["flip_greedy"])[c].mean()
            p4[nm] = {"argmax_moved": float(moved), "kl_mean": float(r["kl"][c].mean())}
            print(f"    {nm}: argmax moves on {moved:.2%} of its cells, mean KL {r['kl'][c].mean():.4f} nats")
        bg = ~(tc.any(2)) & drives[..., None]
        p4["background"] = {"argmax_moved": float((r["greedy"] != r["flip_greedy"])[bg].mean()),
                            "kl_mean": float(r["kl"][bg].mean())}
        print(f"    cells no link covers: argmax moves on {p4['background']['argmax_moved']:.2%}, "
              f"mean KL {p4['background']['kl_mean']:.4f}")
        out["part4"][ck] = p4

        t0 = time.perf_counter()
        starts, roll = rollouts(model, eps.tokens, eps.actions, ctx, dev)
        n = len(starts)
        px = de.decode_u8(tok, roll.reshape(-1, cells), grid, dev)
        rl = de.features(px.reshape(e, n, ROLL_H, *px.shape[1:]), palette, tau)
        print(f"  greedy rollouts, {ROLL_H} frames from real context, {n} starts x {e} episodes, true actions "
              f"({time.perf_counter() - t0:.0f}s); angle vs decoded truth, change vs render since the last "
              f"context frame:")
        print(f"    {'h':>4}{'link':>7}{'ang err':>9}{'still':>8}{'model r':>9}{'tok r':>8}"
              f"{'model sign':>12}{'tok sign':>10}{'lost':>7}{'n':>6}")
        p5 = {}
        last = starts - 1
        for h in HORIZONS:
            idx = starts + h - 1
            p5[h] = {}
            for l, nm in enumerate(names):
                pa, pm = rl.link_angle[:, :, h - 1, l], rl.link_major[:, :, h - 1, l]
                ta, tm = df.link_angle[:, idx, l], df.link_major[:, idx, l]
                ha, hm = df.link_angle[:, last, l], df.link_major[:, last, l]
                ra, rm = rf.link_angle[:, idx, l], rf.link_major[:, idx, l]
                rh, rhm = rf.link_angle[:, last, l], rf.link_major[:, last, l]
                ok = (pm >= MIN_MAJOR) & (tm >= MIN_MAJOR) & (hm >= MIN_MAJOR) & (rm >= MIN_MAJOR) & (rhm >= MIN_MAJOR)
                err = float(np.abs(deg_diff(pa, ta))[ok].mean())
                serr = float(np.abs(deg_diff(ha, ta))[ok].mean())
                dr = deg_diff(ra, rh)[ok]
                dm, dt = deg_diff(pa, ha)[ok], deg_diff(ta, ha)[ok]
                rm_, rt_ = float(np.corrcoef(dm, dr)[0, 1]), float(np.corrcoef(dt, dr)[0, 1])
                mv = np.abs(dr) > 0
                sm = float((np.sign(dm[mv]) == np.sign(dr[mv])).mean())
                st = float((np.sign(dt[mv]) == np.sign(dr[mv])).mean())
                lost = float(((tm > 0) & (pm == 0)).sum() / max((tm > 0).sum(), 1))
                p5[h][nm] = {"ang_err_deg": err, "still_ang_err_deg": serr, "model_corr": rm_, "tok_corr": rt_,
                             "model_sign": sm, "tok_sign": st, "lost": lost, "n": int(ok.sum())}
                print(f"    {h:>4}{nm:>7}{err:>9.2f}{serr:>8.2f}{rm_:>9.3f}{rt_:>8.3f}{sm:>12.1%}{st:>10.1%}"
                      f"{lost:>7.2%}{int(ok.sum()):>6}")
        out.setdefault("part5", {})[ck] = p5
        sys.stdout.flush()
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
