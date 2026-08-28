"""Per-frame measurement vector, both modes, and F-9's threshold sweep.

A feature extractor, not a predicate. Per frame it emits a fixed vector, and
"the validator failed" is a threshold expression over that vector. Keeping the
verdict out of here is what makes Q-3's coherence horizon recomputable from
stored vectors under any threshold set, without re-running a single rollout.

Two modes, and F-9's acceptance test is the sweep of one against the other:

    measure_pixels_only(frame, palette)         phases 2, 3, 4 - no ground truth
    measure_with_truth(frame, meta, palette)    phase 0 - shard meta available

Two things measured here that the design doc did not anticipate, both of which
would have made an exact-equality validator report faults on every perfect
frame (see `_self_check`, and the verification log):

  * `rgba * 255` does not land exactly. link0's `0.90 0.75 0.10` renders as
    (229, 191, 25), not (230, 191, 26) - off by one on two channels, and not
    by a rule worth modelling (0.65 rounds up to 166 while 0.90 rounds down to
    229). This is the measured case for nearest-palette-by-argmin over exact
    equality: with the palette rounded to bytes, exact equality counts **zero**
    pixels for 4 of the 7 entries and calls block0, block2, link1 and table
    missing on a flawless frame.

    Which is also why `Palette.rgb` stays **unrounded** float 0..255. Against
    229.5 rather than 230, the worst distance any rendered pixel sits from its
    own palette entry is **0.75** over 8,000 frames - a rounded palette doubles
    that for no gain.
  * The frame is 14.1% black, and black is in no `rgba` attribute. It is the
    void past the far table edge, where MuJoCo's clear colour shows through.
    So the palette is the XML's six colours **plus** an implicit void entry;
    without it `offpalette_px` reads ~578 px on a perfect frame and F-9 can
    never be met.

Run the check from the repo root, after a generation run:

    python -m mirage.validator
"""

# stdlib ElementTree, not defusedxml. The only file this parses is
# `scene/arm_blocks.xml`, a version-controlled repo artifact at the same trust
# level as this source file - there is no XXE boundary to defend, and adding a
# dependency to harden a parse of our own tracked input buys nothing. Revisit if
# a scene ever arrives from outside the repo.
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

# The void: MuJoCo's framebuffer clear colour, visible past the far table edge
# because the table is finite and there is no skybox. A real, stable,
# renderer-produced colour that no `rgba` attribute can name, so it is added
# here rather than by putting a black geom in the scene - a scene edit would
# change `data_hash` and invalidate 300k frames to fix a reader's bookkeeping.
#
# This is the one documented exception to "the palette has exactly one home".
VOID_NAME = "void"
VOID_RGB = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Palette:
    """The colours a correct frame may contain, and which ones are objects.

    Roles come from geom-name prefixes rather than a hardcoded list, the same
    convention `bench/step_probe.py` uses: F-6 and F-7 iterations are expected
    to edit the scene, and a hardcoded list would silently drop a fourth block.
    """

    names: tuple[str, ...]
    rgb: np.ndarray  # (p, 3) float64 in 0..255
    links: tuple[int, ...]  # indices into names, in name order
    blocks: tuple[int, ...]


def load_palette(scene_xml: Path | str) -> Palette:
    """Every named geom's `rgba`, plus the void, in name order.

    The XML is the palette's only home for the colours it names. Nothing copies
    this list into config JSON: two copies drift, and the symptom is a validator
    that reports missing objects on frames that are fine.
    """
    root = ET.parse(scene_xml).getroot()

    found: list[tuple[str, tuple[float, float, float]]] = []
    for geom in root.iter("geom"):
        name, rgba = geom.get("name"), geom.get("rgba")
        if name is None or rgba is None:
            continue
        parts = [float(v) for v in rgba.split()]
        if len(parts) < 3:
            raise ValueError(f"geom {name!r} has rgba {rgba!r}, need at least 3 components")
        found.append((name, (parts[0], parts[1], parts[2])))

    if not found:
        raise ValueError(f"{scene_xml} names no geom with an rgba - the palette would be empty")

    found.sort()
    names = tuple([VOID_NAME] + [n for n, _ in found])
    rgb = np.array([VOID_RGB] + [c for _, c in found], dtype=np.float64) * 255.0

    # Distinct colours per object, checked rather than assumed. Two links
    # sharing an rgba would merge into one palette entry, and `link_angle`
    # would then report the PCA of both links treated as one blob - a number
    # that looks plausible and tracks nothing.
    objects = [i for i, n in enumerate(names) if n.startswith(("link", "block"))]
    if len({tuple(rgb[i]) for i in objects}) != len(objects):
        raise ValueError(f"two object geoms share an rgba in {scene_xml}")

    return Palette(
        names=names,
        rgb=rgb,
        links=tuple(i for i, n in enumerate(names) if n.startswith("link")),
        blocks=tuple(i for i, n in enumerate(names) if n.startswith("block")),
    )


class Measurement(NamedTuple):
    """The mode-2 vector. Generous by design; the verdict expression is not."""

    n_unique_colors: int  # raw frame, before any mapping. F-2, mode 1 only
    offpalette_px: int  # pixels whose nearest palette entry is further than tau
    max_palette_dist: float  # the worst such distance, so tau can be calibrated
    px_count: np.ndarray  # (p,) int64
    bbox: np.ndarray  # (p, 4) int64 - x0, y0, x1, y1 inclusive; zeros if absent
    compactness: np.ndarray  # (p,) float64 - ~1.0 intact, ~0.05 confetti; 0 if absent
    link_extent: np.ndarray  # (n_links, 2) float64 - major, minor
    link_angle: np.ndarray  # (n_links,) float64 - radians in [0, pi)


class Truth(NamedTuple):
    """What mode 1 adds, straight out of the shard meta record."""

    visible_px: np.ndarray  # (b,) segmentation pixel count per block
    block_xy: np.ndarray  # (b, 2) world position
    qpos: np.ndarray  # (j,) joint angles
    contact_mask: int


def _label(frame: np.ndarray, palette: Palette):
    """Nearest palette entry per pixel, plus the raw-frame colour count.

    Order is fixed and load-bearing: `n_unique_colors` comes off the **raw**
    frame, before any mapping. After mapping it cannot exceed the palette size,
    so computing it later silently stops serving F-2.

    Nearest-palette by argmin over squared distances, never exact equality -
    see the module docstring for the measured reason. Mapping happens on the
    frame's *distinct* colours rather than its pixels: ground-truth frames hold
    7 of them, so the argmin is 7x7 instead of 4096x7, and it stays correct on
    a decoder output that emits thousands.
    """
    flat = frame.reshape(-1, 3)
    keys = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    uniq, inverse = np.unique(keys, return_inverse=True)

    colors = np.stack(((uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255), axis=1).astype(np.float64)
    d2 = ((colors[:, None, :] - palette.rgb[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    dist = np.sqrt(d2[np.arange(len(uniq)), nearest])

    labels = nearest[inverse].reshape(frame.shape[:2])
    counts = np.bincount(inverse, minlength=len(uniq))
    return labels, len(uniq), dist, counts


def _oriented(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float, float]:
    """Major extent, minor extent, and major-axis angle, from PCA on the mask.

    Oriented rather than axis-aligned because both arm links revolve and a
    free-joint block rotates when pushed. An axis-aligned box around a square
    turned 45 degrees has 2x the area, so compactness reads ~0.5 and collides
    with the partially-occluded case that F-7 makes common.

    Two things about the angle, both of which matter to Q-4:

      * A PCA eigenvector is defined up to sign, so the angle is only defined
        modulo pi and is canonicalised into [0, pi). A link rotating through
        that boundary shows a jump of nearly pi, so Q-4's `sign(theta_t+1 -
        theta_t)` must unwrap the difference into (-pi/2, pi/2] before taking
        its sign. Skip the unwrap and roughly one step in every half-turn
        reports the opposite direction.
      * y runs *downward* in image coordinates, so the angle increases
        clockwise on screen. Q-4 compares against a commanded joint sign, so it
        must calibrate that sign against the data rather than assume it.
    """
    pts = np.stack((xs, ys)).astype(np.float64)
    pts -= pts.mean(axis=1, keepdims=True)
    _, evecs = np.linalg.eigh(pts @ pts.T / pts.shape[1])  # eigh: ascending
    proj = evecs.T @ pts

    # +1.0 because a single row of pixels spans one pixel, not zero. Without it
    # a 1-px-wide blob divides by zero and compactness comes back inf.
    extent = proj.max(axis=1) - proj.min(axis=1) + 1.0
    major = evecs[:, 1]
    return float(extent[1]), float(extent[0]), float(np.arctan2(major[1], major[0]) % np.pi)


def measure_pixels_only(frame: np.ndarray, palette: Palette, tau: float = 8.0) -> Measurement:
    """The mode-2 vector for one `(h, w, 3)` uint8 frame.

    `tau` is the palette-adherence radius in RGB Euclidean distance.
    `offpalette_px` means "further than tau from every palette entry", which is
    the only definition that survives nearest-palette mapping - afterwards every
    pixel has a nearest entry, so "not in the palette" is otherwise vacuous.
    The default of 8 sits an order of magnitude above the 0.75 that render
    rounding costs on ground truth, and far below any real violation; `sweep`
    is what calibrates it.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"frame must be (h, w, 3) uint8, got {frame.shape} {frame.dtype}")

    labels, n_unique, dist, uniq_counts = _label(frame, palette)
    p = len(palette.names)

    px_count = np.bincount(labels.ravel(), minlength=p).astype(np.int64)
    bbox = np.zeros((p, 4), dtype=np.int64)
    compactness = np.zeros(p, dtype=np.float64)
    extent = np.zeros((p, 2), dtype=np.float64)
    angle = np.zeros(p, dtype=np.float64)

    for i in range(p):
        if px_count[i] == 0:
            continue  # bbox, compactness and angle stay 0 - px_count is the gate
        ys, xs = np.nonzero(labels == i)
        bbox[i] = (xs.min(), ys.min(), xs.max(), ys.max())
        major, minor, ang = _oriented(ys, xs)
        extent[i] = (major, minor)
        angle[i] = ang
        compactness[i] = px_count[i] / (major * minor)

    links = np.array(palette.links, dtype=np.intp)
    return Measurement(
        n_unique_colors=n_unique,
        offpalette_px=int(uniq_counts[dist > tau].sum()),
        max_palette_dist=float(dist.max()),
        px_count=px_count,
        bbox=bbox,
        compactness=compactness,
        link_extent=extent[links],
        link_angle=angle[links],
    )


def measure_with_truth(
    frame: np.ndarray, meta: np.void, palette: Palette, tau: float = 8.0
) -> tuple[Measurement, Truth]:
    """Mode 1: the same vector, plus the ground truth the shard already carries.

    The vector is identical on purpose. Mode 1 is not a better measurement, it
    is the same measurement next to the answer, which is what lets `sweep`
    decide whether a mode-2 reading is a fault or a fact.
    """
    n_blocks = len(palette.blocks)
    field_names = meta.dtype.names
    if field_names is None:
        raise ValueError("meta must be one record of a structured dtype, not a plain array element")
    n_joints = sum(1 for name in field_names if name.startswith("qpos"))
    truth = Truth(
        visible_px=np.array([meta[f"visible_px{i}"] for i in range(n_blocks)], dtype=np.int64),
        block_xy=np.array(
            [meta[f"block_xy{i}"] for i in range(2 * n_blocks)], dtype=np.float64
        ).reshape(n_blocks, 2),
        qpos=np.array([meta[f"qpos{i}"] for i in range(n_joints)], dtype=np.float64),
        contact_mask=int(meta["contact_mask"]),
    )
    return measure_pixels_only(frame, palette, tau), truth


class Sweep(NamedTuple):
    """A threshold set with zero mode-2 false positives, and its margin."""

    frames: int
    tau: float  # the offpalette radius the sweep was run at
    max_palette_dist: float  # worst distance seen; tau must exceed this
    offpalette_px_max: int  # worst offpalette count under that tau
    min_visible_px: int  # smallest px_count on a block truth says is visible
    px_count_margin: int  # gap to the next-smallest, i.e. how tight min_px is
    n_unique_max: int  # F-2's bar, mode 1 only
    worst_compactness: float  # over visible blocks, for reference not a threshold


def sweep(frames: np.ndarray, metas: np.ndarray, palette: Palette, tau: float = 8.0) -> Sweep:
    """F-9's acceptance test: the mode-2 thresholds that fire on no clean frame.

    A false positive is mode 2 declaring a fault where mode 1 says the frame is
    fine. So for each measurement the sweep takes the *extreme value over
    ground-truth frames*, and any threshold beyond it has zero false positives
    by construction.

    Occlusion-aware, which is legitimate because mode 1 knows: a block at
    `visible_px == 0` is genuinely invisible, so a mode-2 `px_count` of 0 there
    is correct rather than a false positive, and including it would drive
    `min_px` to zero and make the threshold useless. Mode 2 inherits the
    calibrated number and needs no gate of its own.

    `px_count_margin` is the number to read before trusting the result. A margin
    of 1 px means the threshold sits on a cliff and the next unseen frame will
    cross it.
    """
    if len(frames) != len(metas):
        raise ValueError(f"{len(frames)} frames against {len(metas)} meta records")

    visible_counts: list[int] = []
    max_dist = 0.0
    off_max = 0
    uniq_max = 0
    worst_compact = np.inf

    for frame, meta in zip(frames, metas):
        m, truth = measure_with_truth(frame, meta, palette, tau)
        max_dist = max(max_dist, m.max_palette_dist)
        off_max = max(off_max, m.offpalette_px)
        uniq_max = max(uniq_max, m.n_unique_colors)
        for b, entry in enumerate(palette.blocks):
            if truth.visible_px[b] > 0:
                visible_counts.append(int(m.px_count[entry]))
                if m.px_count[entry] > 0:
                    worst_compact = min(worst_compact, float(m.compactness[entry]))

    if not visible_counts:
        raise ValueError("no frame had a visible block - nothing to calibrate against")

    ordered = sorted(visible_counts)
    return Sweep(
        frames=len(frames),
        tau=tau,
        max_palette_dist=max_dist,
        offpalette_px_max=off_max,
        min_visible_px=ordered[0],
        px_count_margin=(ordered[1] - ordered[0]) if len(ordered) > 1 else 0,
        n_unique_max=uniq_max,
        worst_compactness=float(worst_compact),
    )


def _self_check() -> None:
    """F-2 over the whole set, F-6/F-7 against config, and F-9's sweep."""
    from mirage import config, data

    root = Path(__file__).resolve().parent.parent
    cfg = config.load(root / "mirage" / "configs" / "base.json")
    palette = load_palette(root / cfg.sim["scene_xml"])
    print(f"palette: {len(palette.names)} entries {palette.names}, "
          f"{len(palette.links)} links, {len(palette.blocks)} blocks")

    shards = data.load_shards(root / cfg.data["shard_dir"], data_hash=cfg.data_hash)
    index = data.episode_index(shards)
    sampler = data.WindowSampler(shards, index, cfg.data["ctx"])

    # The exact-equality trap, asserted rather than described. If this ever
    # starts failing, `rgba * 255` has become exact and the note in the module
    # docstring is stale - but nearest-palette is still the right default.
    frame0 = sampler[0].frames[0]
    exact = (frame0.reshape(-1, 3)[:, None, :] == palette.rgb.astype(np.uint8)[None, :, :]).all(2)
    missed = [palette.names[i] for i in range(len(palette.names)) if not exact[:, i].any()]
    assert missed, "rgba * 255 now lands exactly - the docstring's measurement is stale"
    print(f"exact RGB equality would call {len(missed)} objects missing on a perfect frame: {missed}")

    # F-2 over every frame, not a sample. n_unique is the cheap field, so the
    # whole 300k set is affordable and F-2 is a claim about the renderer.
    worst, worst_at = 0, (-1, -1)
    for shard in shards:
        flat = np.asarray(shard.pixels).reshape(shard.frames, -1, 3)
        for f in range(shard.frames):
            px = flat[f]
            keys = (px[:, 0].astype(np.uint32) << 16) | (px[:, 1].astype(np.uint32) << 8) | px[:, 2]
            n = len(np.unique(keys))
            if n > worst:
                worst, worst_at = n, (shard.index, f)
    assert worst <= 24, f"F-2: {worst} unique colours at shard {worst_at[0]} frame {worst_at[1]}"
    print(f"F-2: max {worst} unique colours over all {sum(s.frames for s in shards):,} frames "
          f"(worst at shard {worst_at[0]} frame {worst_at[1]}), ceiling 24")

    # F-9's sweep, on a sample spread across every shard. The full vector costs
    # a per-colour PCA, so this is thousands of frames rather than 300k - and a
    # threshold that holds on 8,000 frames drawn from 500 episodes is what F-9
    # asks for.
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(sampler), size=500)
    frames = np.concatenate([sampler[int(i)].frames for i in picks])
    metas = np.concatenate([sampler[int(i)].meta for i in picks])
    result = sweep(frames, metas, palette)
    print(f"F-9 sweep over {result.frames:,} frames at tau {result.tau}:")
    print(f"  worst palette distance {result.max_palette_dist:6.2f} - tau must exceed this")
    print(f"  offpalette_px max      {result.offpalette_px_max:6d} - any threshold above is 0 FP")
    print(f"  min px_count on a block truth calls visible {result.min_visible_px:4d} px, "
          f"margin {result.px_count_margin} px to the next")
    print(f"  worst compactness on a visible block        {result.worst_compactness:.3f}")
    assert result.max_palette_dist < result.tau, (
        f"tau {result.tau} is below the {result.max_palette_dist:.2f} that render rounding alone costs"
    )
    assert result.offpalette_px_max == 0, (
        f"{result.offpalette_px_max} off-palette px on ground truth - F-9 cannot reach zero FP"
    )
    assert result.min_visible_px > 0, "a block truth calls visible reads 0 px - the mapping is wrong"

    # What the sweep actually licenses, stated rather than implied. F-7 makes
    # partial occlusion common, so a visible block's px_count reaches all the
    # way down to 1 - a "block missing if px_count < min_px" rule therefore has
    # no headroom at all and cannot be part of the verdict expression on its
    # own. `offpalette_px` does have headroom, which is what the architecture
    # doc predicted when it said the verdict is minimal and built from
    # uncorrelated fields. Recorded here so a future session does not read a
    # green check as "any threshold set works".
    if result.px_count_margin == 0:
        print(f"  -> px_count is NOT usable as a per-frame threshold: a visible block reaches "
              f"{result.min_visible_px} px, margin {result.px_count_margin}. Occlusion, not a bug")
    print(f"  -> viable verdict today: offpalette_px > {result.offpalette_px_max} at tau "
          f"{result.tau} ({result.tau / max(result.max_palette_dist, 1e-9):.0f}x the render-rounding floor)")

    # The claim underneath mode 2: a pixel-only count tracks the segmentation
    # count. Under offsamples=0 there is no anti-aliasing, so these should agree
    # closely; a large gap means the palette or the id-colour decode is wrong,
    # and no threshold sweep would reveal it.
    diffs = []
    for frame, meta in zip(frames[:2000], metas[:2000]):
        m, truth = measure_with_truth(frame, meta, palette)
        for b, entry in enumerate(palette.blocks):
            diffs.append(int(m.px_count[entry]) - int(truth.visible_px[b]))
    diffs = np.array(diffs)
    print(f"mode 2 px_count vs truth visible_px over {len(diffs):,} readings: "
          f"mean {diffs.mean():+.2f}, max |diff| {np.abs(diffs).max()}, "
          f"exact on {(diffs == 0).mean():.1%}")
    assert np.abs(diffs).max() <= 4, f"pixel-only count is off truth by {np.abs(diffs).max()} px"

    # F-6 and F-7, over the full set, against the thresholds config carries.
    contact = np.concatenate([np.asarray(s.meta["contact_mask"]) for s in shards])
    visible = np.stack([
        np.concatenate([np.asarray(s.meta[f"visible_px{b}"]) for s in shards])
        for b in range(len(palette.blocks))
    ])
    f6 = float((contact != 0).mean())
    f7 = float((visible == 0).any(axis=0).mean())
    assert f6 >= cfg.validator["contact_rate_min"], f"F-6: {f6:.2%}"
    assert f7 >= cfg.validator["occlusion_rate_min"], f"F-7: {f7:.2%}"
    print(f"F-6 contact {f6:.2%} (floor {cfg.validator['contact_rate_min']:.0%}), "
          f"F-7 occlusion {f7:.2%} (floor {cfg.validator['occlusion_rate_min']:.0%})")

    # Mode 2 must not need meta. Called with a frame alone, on purpose.
    only = measure_pixels_only(frames[0], palette)
    assert only.px_count.sum() == frames[0].shape[0] * frames[0].shape[1]
    assert np.all((only.link_angle >= 0) & (only.link_angle < np.pi))
    print("mode 2 runs on a frame alone; px_count partitions the frame; angles in [0, pi)")

    print("validator self-check ok")


if __name__ == "__main__":
    _self_check()
