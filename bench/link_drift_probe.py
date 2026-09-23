"""Does the 10% link-length drift limit hold on the simulator's own frames?

The arm-plausibility requirement asked a world model's rollout to keep the
arm's link lengths within 10% over 200 steps. From pixels, the only length
available is the **on-screen** long side of a link's colour blob, so this probe
checks what that number does on ground-truth frames, which a perfect model
would reproduce exactly. If ground truth already breaks the limit, no model
can meet it and the requirement itself is wrong (the same situation
`bench/hold_probe.py` found for action-following).

It reproduces an earlier, unrecorded finding: ground truth drifts 35.6% on
average. No GPU, no training; it reads only the shards and `mirage.validator`.

**Two controls, because two effects are at work and the first alone does not
explain the data.** A large reading must be explained, or it looks the same as
a broken measurement.

1. *Perspective.* The camera's `xyaxes="1 0 0 0 0.22 0.5"` means world +x shows
   at full length on screen while world +y is squashed to
   0.22/|(0,0.22,0.5)| = 0.403. A link of fixed length at world angle `theta`
   appears `sqrt(cos^2 theta + 0.403^2 sin^2 theta)` of its full length: a
   2.48x range over one turn, nothing to do with physics. The probe predicts
   each frame's length from `qpos` with this formula and reports the
   correlation.
2. *Visibility.* Perspective never shrinks a link below 0.403, so anything
   shorter is losing pixels, not foreshortening: the far link passes behind the
   near one, or the arm reaches past the frame edge. The probe reports the
   correlation of length with visible pixel count, and the share of frames
   where the blob touches the border.

Together these must explain the reading. If they did not, the probe would be
suspect rather than the requirement.

**A corrected statistic was tried, and its own run disproved it.** Since both
effects looked removable, the obvious fix was to remove them: undo the
perspective using the **pixel-measured** `link_angle` (never `qpos`, which a
model's rollout does not have), and count only frames where the link is fully
visible. It does not work. The angle correction is sound (it correlates 0.928
with the `qpos`-based one on link0), yet drift still reads 25.9% to 34.8%
against the 10% limit, while the visibility filter throws away 77% to 96% of
frames. So what is left is neither perspective nor clipping; it is the
measurement noise of sizing a ~30-pixel blob at 64x64, and no rewording of the
requirement removes it. `tol` is swept so the answer cannot come from tuning
one number until it passes.

    python bench/link_drift_probe.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import config, data, validator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPISODES = 12    # the same held-out count bench/token_stability_probe.py uses
WINDOW = 200     # the requirement's 200-step rollout
BAR = 0.10       # the requirement's 10% drift limit

# The corrected statistic's one tolerance setting, SWEPT rather than chosen,
# because a value tuned until ground truth passes would be circular. A link's
# blob AREA shrinks with perspective by the same factor as its length, so
# `px_count / factor` stays flat while the link is whole and drops when the
# other link covers part of it. `tol` is how much of that corrected area a
# frame must still show to count.
TOLERANCES = (0.70, 0.80, 0.90)
MIN_KEPT = 50    # a window with fewer frames left than this is skipped, not scored


def _camera_yz_compression(scene_xml: Path) -> float:
    """How much the camera squashes a world +y vector on screen.

    Read from the XML rather than hardcoded: the camera's second `xyaxes`
    triple is the screen-up direction in world coordinates, and a world +y unit
    vector projects onto it with weight `y / |v|`. Screen-right is world +x at
    full length, so this one ratio is all the perspective there is for a flat arm.
    """
    import xml.etree.ElementTree as ET

    cam = ET.parse(scene_xml).getroot().find(".//camera")
    assert cam is not None, f"no <camera> in {scene_xml}"
    ax = np.array([float(v) for v in cam.attrib["xyaxes"].split()], dtype=np.float64)
    up = ax[3:]
    return float(up[1] / np.linalg.norm(up))


def probe(cfg: config.Config, episodes: int = EPISODES) -> dict:
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    tau = cfg.validator["offpalette_tau"]
    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    val = [e for e in index
           if data.is_val(e.episode_id, cfg.data["val_fraction"])][:episodes]
    assert len(val) == episodes, f"only {len(val)} val episodes available"

    k = _camera_yz_compression(ROOT / cfg.sim["scene_xml"])
    n_links = len(palette.links)
    per_link_ext: list[np.ndarray] = []
    per_link_pred: list[np.ndarray] = []
    per_link_vis: list[np.ndarray] = []
    per_link_clip: list[np.ndarray] = []
    per_link_fac: list[np.ndarray] = []
    per_link_dep: list[np.ndarray] = []
    windows: list[list[float]] = [[] for _ in range(n_links)]
    first_dev: list[list[float]] = [[] for _ in range(n_links)]

    for ep in val:
        sh = shards[ep.shard]
        px = np.ascontiguousarray(sh.pixels[ep.start:ep.start + ep.length, ::-1])
        meta = sh.meta[ep.start:ep.start + ep.length]
        ext = np.empty((len(px), n_links))
        ang = np.empty((len(px), n_links))
        vis = np.empty((len(px), n_links))
        clip = np.zeros((len(px), n_links), bool)
        h, w = px.shape[1], px.shape[2]
        for t in range(len(px)):
            m = validator.measure_pixels_only(px[t], palette, tau)
            ext[t] = m.link_extent[:, 0]
            ang[t] = m.link_angle
            for L, entry in enumerate(palette.links):
                vis[t, L] = m.px_count[entry]
                x0, y0, x1, y1 = m.bbox[entry]
                clip[t, L] = m.px_count[entry] > 0 and (
                    x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1)

        # World angle of each link: link0 is joint0, link1 is joint0 + joint1,
        # since both hinges turn about z and the second rides on the first. Read
        # qpos from the meta record rather than re-simulating.
        q = np.stack([meta[f"qpos{i}"].astype(np.float64) for i in range(n_links)])
        theta = np.cumsum(q, axis=0)
        pred = np.sqrt(np.cos(theta) ** 2 + (k * np.sin(theta)) ** 2).T  # (T, links)

        # The SAME perspective factor, rebuilt from the PIXEL-measured angle
        # instead of `qpos`, which is the point: a model's rollout has no ground
        # truth, so a correction that needs `qpos` cannot be applied to what is
        # actually scored. An on-screen direction `(cos phi, sin phi)` comes from
        # a world direction proportional to `(cos phi, sin phi / k)` (undoing the
        # camera's squash of world +y), and the factor is that direction's
        # on-screen length once normalised. No special case at phi = pi/2: the
        # world direction is `(0, +-1/k)`, normalises to `(0, +-1)`, and the
        # factor comes out k, which is right. Signs never matter, so neither image
        # y pointing down nor PCA's arbitrary axis sign affects it.
        wx, wy = np.cos(ang), np.sin(ang) / k
        norm = np.hypot(wx, wy)
        ux, uy = wx / norm, wy / norm
        fac = np.sqrt(ux ** 2 + (k * uy) ** 2)          # (T, links), in [k, 1]

        per_link_ext.append(ext)
        per_link_pred.append(pred)
        per_link_vis.append(vis)
        per_link_clip.append(clip)
        per_link_fac.append(fac)
        per_link_dep.append(ext / fac)

        # w0, not w: `w` is the frame width above, and reusing the name only
        # worked because nothing read it afterwards.
        for w0 in range(0, len(px) - WINDOW + 1, WINDOW):   # non-overlapping
            seg = ext[w0:w0 + WINDOW]
            for L in range(n_links):
                s = seg[:, L]
                windows[L].append(float((s.max() - s.min()) / np.median(s)))
                first_dev[L].append(float(np.abs(s - s[0]).max() / s[0]))

    ext_all = np.concatenate(per_link_ext)
    pred_all = np.concatenate(per_link_pred)
    vis_all = np.concatenate(per_link_vis)
    clip_all = np.concatenate(per_link_clip)
    fac_all = np.concatenate(per_link_fac)

    # The perspective-corrected area a whole link shows. The 99th percentile,
    # not the max, so one outlier frame cannot set the reference.
    area_ref = np.percentile(vis_all / fac_all, 99, axis=0)

    restated = []
    for tol in TOLERANCES:
        links = []
        for L in range(n_links):
            kept, dropped, frames_kept, frames_all = [], 0, 0, 0
            for dep, fac, vis, clip in zip(per_link_dep, per_link_fac,
                                           per_link_vis, per_link_clip):
                keep = ~clip[:, L] & (vis[:, L] >= tol * area_ref[L] * fac[:, L])
                frames_kept += int(keep.sum())
                frames_all += len(keep)
                for w0 in range(0, len(dep) - WINDOW + 1, WINDOW):
                    s = dep[w0:w0 + WINDOW, L][keep[w0:w0 + WINDOW]]
                    if len(s) < MIN_KEPT:
                        dropped += 1
                        continue
                    kept.append(float((s.max() - s.min()) / np.median(s)))
            links.append({
                "name": palette.names[palette.links[L]],
                "frames_kept_share": frames_kept / frames_all,
                "windows_scored": len(kept),
                "windows_dropped": dropped,
                "drift_mean": float(np.mean(kept)) if kept else float("nan"),
                "drift_max": float(np.max(kept)) if kept else float("nan"),
                "windows_over_bar": int(np.sum(np.array(kept) > BAR)) if kept else 0,
            })
        restated.append({"tol": tol, "links": links})

    return {
        "restated": restated,
        "area_ref": [float(a) for a in area_ref],
        # Check on the perspective correction itself: the pixel-based factor
        # against the qpos-based one. Low or negative means the angle convention
        # is wrong and every corrected number below is meaningless.
        "corr_factor_pixel_vs_qpos": [
            float(np.corrcoef(fac_all[:, L], pred_all[:, L])[0, 1])
            for L in range(n_links)
        ],
        "episodes": len(val),
        "frames": int(len(ext_all)),
        "window": WINDOW,
        "windows_per_link": len(windows[0]),
        "camera_y_compression": k,
        "projection_range_ratio": float(k),  # smallest/largest predicted factor
        "links": [
            {
                "name": palette.names[palette.links[L]],
                "extent_min_px": float(ext_all[:, L].min()),
                "extent_max_px": float(ext_all[:, L].max()),
                "extent_min_over_max": float(ext_all[:, L].min() / ext_all[:, L].max()),
                "drift_range_over_median_mean": float(np.mean(windows[L])),
                "drift_range_over_median_max": float(np.max(windows[L])),
                "drift_range_over_median_min": float(np.min(windows[L])),
                "windows_over_bar": int(np.sum(np.array(windows[L]) > BAR)),
                "drift_from_first_mean": float(np.mean(first_dev[L])),
                "corr_with_projection": float(
                    np.corrcoef(ext_all[:, L], pred_all[:, L])[0, 1]
                ),
                "corr_with_visible_px": float(
                    np.corrcoef(ext_all[:, L], vis_all[:, L])[0, 1]
                ),
                "below_projection_floor_share": float(
                    np.mean(ext_all[:, L] < ext_all[:, L].max() * k)
                ),
                "border_clipped_share": float(clip_all[:, L].mean()),
            }
            for L in range(n_links)
        ],
    }


def main() -> None:
    cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
    d = probe(cfg)
    print(f"{d['episodes']} held-out episodes, {d['frames']:,} frames, "
          f"{d['windows_per_link']} non-overlapping {d['window']}-frame windows per link")
    print(f"camera compresses world +y to {d['camera_y_compression']:.3f} of world +x, "
          f"so a rigid link's projected length spans {d['camera_y_compression']:.3f}..1.000\n")
    print(f"{'link':<8} {'ext px':>13} {'min/max':>8} {'drift mean':>11} {'max':>8} "
          f"{'>10%':>10} {'r proj':>7} {'r vis':>7} {'<floor':>7} {'clipped':>8}")
    for L in d["links"]:
        print(f"{L['name']:<8} {L['extent_min_px']:5.1f}-{L['extent_max_px']:5.1f} "
              f"{L['extent_min_over_max']:>8.3f} {L['drift_range_over_median_mean']:>10.1%} "
              f"{L['drift_range_over_median_max']:>7.1%} "
              f"{L['windows_over_bar']:>4}/{d['windows_per_link']:<5} "
              f"{L['corr_with_projection']:>7.3f} {L['corr_with_visible_px']:>7.3f} "
              f"{L['below_projection_floor_share']:>6.1%} {L['border_clipped_share']:>7.1%}")
    worst = max(L["drift_range_over_median_mean"] for L in d["links"])
    print(f"\nQ-5 bar is {BAR:.0%}. Ground truth reads {worst:.1%} mean on the worse link.")

    print("\n--- the restated statistic: deprojected from the PIXEL angle, "
          "non-visible frames excluded ---")
    print("control: pixel-derived factor vs the qpos-derived one, r = "
          + ", ".join(f"{r:.3f}" for r in d["corr_factor_pixel_vs_qpos"]))
    print(f"{'tol':>5} {'link':<8} {'frames kept':>12} {'windows':>9} {'dropped':>8} "
          f"{'drift mean':>11} {'max':>8} {'>10%':>6}")
    for block in d["restated"]:
        for L in block["links"]:
            print(f"{block['tol']:>5.2f} {L['name']:<8} {L['frames_kept_share']:>11.1%} "
                  f"{L['windows_scored']:>9} {L['windows_dropped']:>8} "
                  f"{L['drift_mean']:>10.1%} {L['drift_max']:>7.1%} "
                  f"{L['windows_over_bar']:>6}")
    worst_restated = max(L["drift_mean"] for b in d["restated"] for L in b["links"])
    print(f"\nWorst restated reading across the sweep: {worst_restated:.1%} "
          f"against the {BAR:.0%} bar.")


if __name__ == "__main__":
    main()
