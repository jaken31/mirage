"""How much does one frame step show? Read-only, numpy only, no GPU.

For frame strides k in STRIDES it compares frame t with frame t+k on the stored
64x64 shards and R1's token cache, and projects the arm's elbow and tip from
qpos through the scene camera to get the motion in pixels.

    python stride_probe.py /home/jake/Projects/mirage [every_nth_episode]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 4
TOK_RUN = "20260829-005439-r1"
STRIDES = [1, 2, 3, 5, 8, 15, 30]
CTX = 15

sys.path.insert(0, str(ROOT))
from mirage import data  # noqa: E402  (meta_dtype only)

# Scene camera, from scene/arm_blocks.xml: pos, xyaxes, default fovy 45 deg.
CAM = np.array([0.0, -0.5, 0.22])
XH = np.array([1.0, 0.0, 0.0])
YH = np.array([0.0, 0.22, 0.5]); YH /= np.linalg.norm(YH)
FH = np.cross(YH, XH); FH /= np.linalg.norm(FH)          # forward = -z_cam
if FH[1] < 0:
    FH = -FH
FOCAL = 32.0 / math.tan(math.radians(22.5))               # px, 64 px tall


def project(p):
    v = p - CAM
    depth = v @ FH
    return np.stack([FOCAL * (v @ XH) / depth, FOCAL * (v @ YH) / depth], axis=-1)


def arm_points(q0, q1):
    z = np.full_like(q0, 0.04)
    elbow = np.stack([0.18 * np.cos(q0), 0.18 * np.sin(q0), z], axis=-1)
    tip = elbow + np.stack([0.15 * np.cos(q0 + q1), 0.15 * np.sin(q0 + q1), 0 * z], axis=-1)
    return project(elbow), project(tip)


def field_any(change, margin):
    """(p, 64, 64) bool -> (p, 8, 8): any change inside each cell's patch grown by `margin` px."""
    p = change.shape[0]
    pad = np.zeros((p, 64 + 2 * margin + 1, 64 + 2 * margin + 1), dtype=np.int32)
    pad[:, margin + 1:margin + 65, margin + 1:margin + 65] = change
    s = pad.cumsum(1).cumsum(2)
    size = 8 + 2 * margin
    r0 = np.arange(8) * 8
    r1 = r0 + size
    tot = (s[:, r1][:, :, r1] - s[:, r0][:, :, r1] - s[:, r1][:, :, r0] + s[:, r0][:, :, r0])
    return tot > 0


shard_dir = ROOT / "data" / "shards"
acc = {k: {n: [] for n in ("px_same", "px_changed", "tok_keep", "tok_frozen", "moved22", "moved8",
                             "chg_moved22", "chg_quiet22", "chg_quiet8", "tip", "elbow",
                             "ctx_tip", "armmove_px_same", "armmove_tok_same")} for k in STRIDES}
episodes = 0
for side_path in sorted(shard_dir.glob("shard_*.json")):
    side = json.loads(side_path.read_text())
    n = side["frames"]
    dt = data.meta_dtype(side["meta_joints"], side["meta_blocks"])
    meta = np.memmap(side_path.with_suffix(".meta"), dtype=dt, mode="r", shape=(n,))
    pix = np.memmap(side_path.with_suffix(".pixels"), dtype=np.uint8, mode="r", shape=(n, 64, 64, 3))
    toks = np.load(ROOT / "runs" / TOK_RUN / "tokens" / (side_path.stem + ".npy"))  # (n, 8, 8), rows upright
    ep = np.asarray(meta["episode_id"])
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        if int(ep[s]) % EVERY:
            continue
        episodes += 1
        px = np.asarray(pix[s:e]).astype(np.uint32)
        px = (px[..., 0] << 16 | px[..., 1] << 8 | px[..., 2])[:, ::-1]      # rows upright, like the tokens
        tk = np.asarray(toks[s:e])
        q0 = np.asarray(meta["qpos0"][s:e], dtype=np.float64)
        q1 = np.asarray(meta["qpos1"][s:e], dtype=np.float64)
        elbow, tip = arm_points(q0, q1)
        for k in STRIDES:
            a = acc[k]
            change = px[k:] != px[:-k]
            npx = change.reshape(len(change), -1).sum(1)
            tchg = tk[k:] != tk[:-k]
            m22 = field_any(change, 7)
            m8 = field_any(change, 0)
            a["px_same"].append(npx == 0)
            a["px_changed"].append(npx)
            a["tok_keep"].append(1.0 - tchg.reshape(len(tchg), -1).mean(1))
            a["tok_frozen"].append(~tchg.reshape(len(tchg), -1).any(1))
            a["moved22"].append(m22.reshape(len(m22), -1).mean(1))
            a["moved8"].append(m8.reshape(len(m8), -1).mean(1))
            a["chg_moved22"].append((tchg & m22).reshape(len(tchg), -1).mean(1))
            a["chg_quiet22"].append((tchg & ~m22).reshape(len(tchg), -1).mean(1))
            a["chg_quiet8"].append((tchg & ~m8).reshape(len(tchg), -1).mean(1))
            dtip = np.linalg.norm(tip[k:] - tip[:-k], axis=1)
            delb = np.linalg.norm(elbow[k:] - elbow[:-k], axis=1)
            a["tip"].append(dtip)
            a["elbow"].append(delb)
            moving = (np.abs(q0[k:] - q0[:-k]) + np.abs(q1[k:] - q1[:-k])) > 1e-3 * k
            a["armmove_px_same"].append((npx == 0)[moving])
            a["armmove_tok_same"].append((~tchg.reshape(len(tchg), -1).any(1))[moving])
            span = (CTX - 1) * k                      # first to last frame of a 15-frame context
            if span < len(tip):
                a["ctx_tip"].append(np.linalg.norm(tip[span:] - tip[:-span], axis=1))

print(f"{episodes} episodes (every {EVERY}th), tokenizer {TOK_RUN}, focal {FOCAL:.1f} px")
hdr = ("stride", "ms/frame", "px-identical", "med px chg", "tok keep", "tok-frozen frames",
       "cells moved(22)", "chg on moved", "chg on quiet(22)", "quiet share of chg",
       "tip px med", "tip px p90", "tip<0.5px", "tip<1px", "ctx15 tip px med", "ctx15 p90",
       "px-identical | arm moving", "tok-frozen | arm moving")
print(" | ".join(hdr))
for k in STRIDES:
    a = {n: np.concatenate(v) if v else np.zeros(0) for n, v in acc[k].items()}
    chg_m, chg_q = a["chg_moved22"].mean(), a["chg_quiet22"].mean()
    row = (k, 2 * k, f"{100 * a['px_same'].mean():.1f}%", f"{np.median(a['px_changed']):.0f}",
           f"{100 * a['tok_keep'].mean():.2f}%", f"{100 * a['tok_frozen'].mean():.1f}%",
           f"{100 * a['moved22'].mean():.1f}%", f"{100 * chg_m:.2f}%", f"{100 * chg_q:.2f}%",
           f"{100 * chg_q / (chg_m + chg_q):.0f}%",
           f"{np.median(a['tip']):.2f}", f"{np.percentile(a['tip'], 90):.2f}",
           f"{100 * (a['tip'] < 0.5).mean():.0f}%", f"{100 * (a['tip'] < 1.0).mean():.0f}%",
           f"{np.median(a['ctx_tip']):.1f}" if len(a["ctx_tip"]) else "-",
           f"{np.percentile(a['ctx_tip'], 90):.1f}" if len(a["ctx_tip"]) else "-",
           f"{100 * a['armmove_px_same'].mean():.1f}%", f"{100 * a['armmove_tok_same'].mean():.1f}%")
    print(" | ".join(str(x) for x in row))
