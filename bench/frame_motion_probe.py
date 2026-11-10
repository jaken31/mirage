"""Motion per recorded frame, blocks lost from view, and the Phase 0 rates, on one dataset.

Adapted from `research_notes/Fix the clock before the model/stride_probe.py`
for the `sim.frame_stride` datasets: it reads the camera and the links from the
config's scene XML instead of hardcoding them, and measures frame t against
frame t+1 of the dataset as recorded (one frame is `frame_stride` physics steps).

    python bench/frame_motion_probe.py mirage/configs/base30.json [--shard-dir DIR]

Numpy only, no GPU. Reads only the meta records, never the pixels.

What it prints, all over every frame of every episode:

- link midpoint and tip motion in pixels per frame, from qpos projected through
  the scene camera. "Driven" means action[t] commands that joint nonzero, so the
  midpoint figure is the motion the action produces, not drift.
- blocks lost from view: a block whose centre projects outside the frame and
  never comes back before the episode ends ("out of frame for good"), and a
  block whose segmentation count is 0 for the rest of the episode ("gone for
  good", which also counts a block parked behind the arm). As a share of frames
  with at least one such block, and as blocks lost by the episode's end.
- F-5 action balance over all frames, F-6 contact, F-7 recoverable occlusion,
  and Q-4 action-following: sign(theta_t - theta_t-1) against the commanded
  sign of action[t], on joint-frames whose command is nonzero.
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mirage import config, data  # noqa: E402


def scene_geometry(root):
    """Camera (pos, right, up, forward, focal in px per half-height) and the two links' layout."""
    cam = next(root.iter("camera"))
    pos = np.array([float(v) for v in cam.get("pos").split()])
    xy = [float(v) for v in cam.get("xyaxes").split()]
    xh = np.array(xy[:3]); xh /= np.linalg.norm(xh)
    yh = np.array(xy[3:]); yh /= np.linalg.norm(yh)
    fwd = -np.cross(xh, yh); fwd /= np.linalg.norm(fwd)   # MuJoCo cameras look along -z
    fovy = float(cam.get("fovy", 45.0))
    bodies = {b.get("name"): b for b in root.iter("body")}
    l0, l1 = bodies["link0_body"], bodies["link1_body"]
    base = np.array([float(v) for v in l0.get("pos").split()])
    g0, g1 = l0.find("geom"), l1.find("geom")
    len0 = float(l1.get("pos").split()[0])
    mid0 = float(g0.get("pos").split()[0])
    mid1 = float(g1.get("pos").split()[0])
    len1 = mid1 + float(g1.get("size").split()[0])
    return (pos, xh, yh, fwd, math.tan(math.radians(fovy / 2))), base, len0, mid0, mid1, len1


def project(cam, p, half):
    pos, xh, yh, fwd, tan_half = cam
    v = p - pos
    depth = v @ fwd
    f = half / tan_half
    return np.stack([f * (v @ xh) / depth, f * (v @ yh) / depth], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard-dir", default=None, help="override data.shard_dir")
    ap.add_argument("--quiet", action="store_true", help="one summary line only")
    args = ap.parse_args()

    cfg = config.load(args.config)
    stride = config.setting(cfg, "sim", "frame_stride")
    shard_dir = Path(args.shard_dir or cfg.data["shard_dir"])
    root = ET.parse(ROOT / cfg.sim["scene_xml"]).getroot()
    ts = float(next(root.iter("option")).get("timestep"))
    ms = 1000.0 * ts * stride
    cam, base, len0, mid0, mid1, len1 = scene_geometry(root)
    half = cfg.sim["height"] / 2

    shards = data.load_shards(shard_dir, data_hash=cfg.data_hash)
    index = data.episode_index(shards)
    n_blocks = int(shards[0].sidecar["meta_blocks"])
    block_z = 0.025  # block centre height on the table; the meta stores x and y only

    acc = {k: [] for k in ("m0", "m1", "m0s", "m1s", "tip", "tip_all", "out_any", "gone_any", "agree")}
    lost_out_end, lost_gone_end, lost_out_at = [], [], []
    hist = np.zeros(9, np.int64)
    contact_frames = occl_frames = frames = 0
    for ep in index:
        m = shards[ep.shard].meta[ep.start:ep.start + ep.length]
        a = np.asarray(m["action"]).astype(np.int64)
        hist += np.bincount(a, minlength=9)
        q0 = np.asarray(m["qpos0"], np.float64); q1 = np.asarray(m["qpos1"], np.float64)
        zero = np.zeros_like(q0)
        ring = lambda r, ang: np.stack([r * np.cos(ang), r * np.sin(ang), zero], -1)  # noqa: E731
        elbow = base + ring(len0, q0)
        pts = {"m0": base + ring(mid0, q0), "m1": elbow + ring(mid1, q0 + q1), "tip": elbow + ring(len1, q0 + q1)}
        mv = {k: np.linalg.norm(np.diff(project(cam, p, half), axis=0), axis=1) for k, p in pts.items()}
        s0, s1 = a % 3 - 1, (a // 3) % 3 - 1           # commanded sign of action[t]
        drv0, drv1 = s0[1:] != 0, s1[1:] != 0           # action[t] moved frame t-1 -> t
        acc["m0"].append(mv["m0"][drv0]); acc["m1"].append(mv["m1"][drv1])
        # Steady: the same nonzero command as the frame before, so the joint
        # starts the frame already turning that way rather than reversing.
        st0 = np.r_[False, (s0[2:] == s0[1:-1])] & drv0
        st1 = np.r_[False, (s1[2:] == s1[1:-1])] & drv1
        acc["m0s"].append(mv["m0"][st0]); acc["m1s"].append(mv["m1"][st1])
        acc["tip"].append(mv["tip"][drv0 | drv1]); acc["tip_all"].append(mv["tip"])
        acc["agree"].append(np.r_[(np.sign(np.diff(q0)) == s0[1:])[drv0],
                                  (np.sign(np.diff(q1)) == s1[1:])[drv1]])

        bxy = np.stack([np.stack([np.asarray(m[f"block_xy{2 * b}"]), np.asarray(m[f"block_xy{2 * b + 1}"])], -1)
                        for b in range(n_blocks)]).astype(np.float64)            # (b, t, 2)
        bp = np.concatenate([bxy, np.full(bxy.shape[:2] + (1,), block_z)], -1)
        inview = (np.abs(project(cam, bp, half)) < half).all(-1)                 # (b, t)
        visn = np.stack([np.asarray(m[f"visible_px{b}"]) for b in range(n_blocks)])
        # "For good": not in view at t, nor at any later frame of the episode.
        later = lambda x: np.flip(np.maximum.accumulate(np.flip(x, 1), 1), 1)   # noqa: E731
        out_good, gone_good = ~later(inview), ~later(visn > 0)
        acc["out_any"].append(out_good.any(0)); acc["gone_any"].append(gone_good.any(0))
        lost_out_end.append(int(out_good[:, -1].sum())); lost_gone_end.append(int(gone_good[:, -1].sum()))
        lost_out_at += [int(np.argmax(out_good[b])) for b in range(n_blocks) if out_good[b, -1]]
        contact_frames += int((data.contact_bits(m) != 0).sum())
        occl_frames += int(((visn == 0) & data.seen_later(visn)).any(axis=0).sum())
        frames += ep.length

    c = {k: np.concatenate(v) for k, v in acc.items()}
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else float("nan")  # noqa: E731
    lo, hi = hist.min(), hist.max()
    sec = cfg.sim["steps_per_episode"] * ms / 1000
    if args.quiet:
        print(f"mid0 {q(c['m0'], 50):.2f} ({q(c['m0s'], 50):.2f}) | mid1 {q(c['m1'], 50):.2f} ({q(c['m1s'], 50):.2f}) | "
              f"tip p50/90/99 {q(c['tip'], 50):.2f}/{q(c['tip'], 90):.2f}/{q(c['tip'], 99):.2f} | "
              f"out4good {100 * c['out_any'].mean():.1f}% lost/ep {np.mean(lost_out_end):.2f} | "
              f"F5 min {100 * lo / hist.sum():.1f}% r {hi / max(lo, 1):.2f} | F6 {100 * contact_frames / frames:.1f}% | "
              f"F7 {100 * occl_frames / frames:.1f}% | Q4 {100 * c['agree'].mean():.1f}%")
        return
    print(f"{shard_dir}: {len(index)} episodes x {cfg.sim['steps_per_episode']} frames, "
          f"stride {stride} = {ms:g} ms per frame, {sec:g} s per episode")
    print("pixels moved per frame (median / p90 / p99 / max):")
    for name, key in [("link0 midpoint, joint0 driven", "m0"), ("  same command as frame before", "m0s"),
                      ("link1 midpoint, joint1 driven", "m1"), ("  same command as frame before", "m1s"),
                      ("tip, either joint driven", "tip"), ("tip, all frames", "tip_all")]:
        x = c[key]
        print(f"  {name:31s} {q(x, 50):5.2f} / {q(x, 90):5.2f} / {q(x, 99):5.2f} / {x.max():5.2f}  "
              f"(<1 px on {100 * (x < 1).mean():.0f}%, >4 px on {100 * (x > 4).mean():.1f}%)")
    print(f"blocks out of frame for good: {100 * c['out_any'].mean():.1f}% of frames have one; "
          f"{np.mean(lost_out_end):.2f} of {n_blocks} lost per episode by its end, "
          f"{100 * np.mean(np.array(lost_out_end) > 0):.1f}% of episodes lose at least one")
    if lost_out_at:
        t = np.array(lost_out_at) * ms / 1000
        print(f"  a lost block leaves for good at median {np.median(t):.2f} s into the episode")
    print(f"blocks gone for good (0 px to the end): {100 * c['gone_any'].mean():.1f}% of frames; "
          f"{np.mean(lost_gone_end):.2f} per episode at the end")
    print(f"F-5 balance: {hist.tolist()}  min share {100 * lo / hist.sum():.2f}% (>= 5%), "
          f"max/min {hi / max(lo, 1):.2f} (<= 2.5)")
    print(f"F-6 contact {100 * contact_frames / frames:.2f}% (> 5%), "
          f"F-7 recoverable occlusion {100 * occl_frames / frames:.2f}% (>= 3%)")
    print(f"Q-4 action-following: {100 * c['agree'].mean():.2f}% of driven joint-frames move the commanded way")


if __name__ == "__main__":
    main()
