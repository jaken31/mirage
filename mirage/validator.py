"""Per-frame measurements of what is in a frame, and the threshold sweep.

This measures; it does not judge. For each frame it returns a fixed set of
numbers, and "the validator failed" is a threshold rule applied to them later.
Keeping the pass/fail rule out of here means stored measurements can be
re-judged under any thresholds without re-running a single model rollout.

Two modes:

    measure_pixels_only(frame, palette, tau)         mode 2: pixels alone, for model output
    measure_with_truth(frame, meta, palette, tau)    mode 1: plus the simulator's ground truth

`sweep` runs both on real frames to find thresholds that never flag a
correct frame.

Two measured surprises, each of which would make an exact-colour-match
validator report faults on every perfect frame (see `_self_check` and the
verification log):

  * `rgba * 255` does not land exactly. link0's `0.90 0.75 0.10` renders as
    (229, 191, 25), not (230, 191, 26): off by one on two channels, with no
    simple rounding rule (0.65 rounds up to 166, 0.90 rounds down to 229). So
    pixels are matched to the *nearest* palette colour. With a byte-rounded
    palette, exact matching finds **zero** pixels for 4 of the 7 colours and
    calls block0, block2, link1 and the table missing on a perfect frame.

    It is also why `Palette.rgb` stays **unrounded** floats in 0..255.
    Measured against 229.5 instead of 230, no rendered pixel is more than
    **0.75** from its own palette colour over 8,000 frames; rounding the
    palette would double that for nothing.
  * About 14% of each frame is black, and no `rgba` attribute is black. It is
    the empty space past the far table edge, where MuJoCo's background colour
    shows. So the palette is the XML's six colours **plus** an implicit "void"
    entry; without it about 578 pixels of a perfect frame read as off-palette.

Run the check from the repo root:

    python -m mirage.validator

With no generated dataset it falls back to the committed 40-frame fixture, so
the sweep runs in a fresh clone (see `mirage.data.self_check_config`). The two
dataset-wide rates (contact and occlusion) are skipped there: forty frames
say nothing about 300,000.
"""

# Standard-library ElementTree, not defusedxml. The only file parsed is
# `scene/arm_blocks.xml`, which is in the repo and as trusted as this source
# file, so there is no malicious-XML risk to guard against. Revisit if a scene
# ever comes from outside the repo.
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

# mirage.data owns the record layout, so `SCRIPTED_BIT` lives there; importing
# it beats repeating 0x80 here. No import cycle: data imports only config.
from mirage.data import SCRIPTED_BIT

# The void: MuJoCo's background colour, visible past the far table edge because
# the table is finite and there is no sky. It is a real, stable colour that no
# `rgba` attribute names, so it is added here instead of putting a black geom in
# the scene. A scene edit would change `data_hash` and invalidate 300k frames
# just to fix a reader's bookkeeping.
#
# This is the one exception to "palette colours live only in the XML".
VOID_NAME = "void"
VOID_RGB = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Palette:
    """The colours a correct frame may contain, and which ones are objects.

    Roles come from geom-name prefixes, not a hardcoded list, the same rule
    `bench/step_probe.py` uses: the scene is expected to be edited, and a
    hardcoded list would silently miss a fourth block.
    """

    names: tuple[str, ...]
    rgb: np.ndarray  # (p, 3) float64 in 0..255
    links: tuple[int, ...]  # indices into names, in name order
    blocks: tuple[int, ...]


def load_palette(scene_xml: Path | str) -> Palette:
    """Every named geom's `rgba`, plus the void, in name order.

    The XML is the only place these colours are defined. Nothing copies them
    into config JSON: two copies drift apart, and the symptom is a validator
    reporting missing objects on frames that are fine.
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

    # Each object must have its own colour. Checked, not assumed: two links
    # sharing an rgba would merge into one palette entry, and `link_angle`
    # would measure both links as one shape, a plausible-looking number that
    # tracks nothing.
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
    """The per-frame measurements. Deliberately broad; the pass/fail rule uses few of them."""

    n_unique_colors: int  # distinct colours in the raw frame, before mapping; checks flat rendering
    offpalette_px: int  # pixels further than tau from every palette colour
    max_palette_dist: float  # the largest pixel-to-palette distance, for calibrating tau
    offpalette_frac: float  # offpalette_px as a fraction of the frame - THE pass/fail number
    px_count: np.ndarray  # (p,) int64
    bbox: np.ndarray  # (p, 4) int64 - x0, y0, x1, y1 inclusive; zeros if absent
    compactness: np.ndarray  # (p,) float64 - ~1.0 solid shape, ~0.05 scattered pixels; 0 if absent
    link_extent: np.ndarray  # (n_links, 2) float64 - long side, short side
    link_angle: np.ndarray  # (n_links,) float64 - radians in [0, pi)


class Truth(NamedTuple):
    """What mode 1 adds: ground truth straight from the shard meta record."""

    visible_px: np.ndarray  # (b,) segmentation pixel count per block
    block_xy: np.ndarray  # (b, 2) world position
    qpos: np.ndarray  # (j,) joint angles
    contact_mask: int  # block bits only - the scripted flag is masked off
    is_scripted: bool  # whether this episode is scripted or random


def _label(frame: np.ndarray, palette: Palette):
    """Nearest palette colour per pixel, plus the raw frame's colour count.

    The order matters: `n_unique_colors` is counted on the **raw** frame, before
    mapping. After mapping it can never exceed the palette size, so counting
    later would silently make the flat-rendering check meaningless.

    Nearest colour, never exact match (the module docstring gives the measured
    reason). Matching runs on the frame's *distinct* colours, not its pixels:
    real frames have 7, so it is 7x7 work instead of 4096x7, and it still works
    on model output with thousands of colours.
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


def _weighted_pctl(dist: np.ndarray, counts: np.ndarray, q: float) -> float:
    """The `q`-quantile of the per-pixel palette distance, q in (0, 1).

    Nearest-rank, not interpolated: the smallest distance `d` such that at
    least a `q` fraction of pixels are at distance <= d. Interpolating would
    invent a value no pixel has, and the point is that it names a real pixel.

    Weighted because `_label` works on *distinct colours*, not pixels:
    `dist[i]` is one colour's distance and `counts[i]` is how many pixels have
    it. Expanding to per-pixel first would undo that saving.

    **Not part of the measurements and not the pass/fail rule.** It was tried
    as a resolution-independent replacement for the off-palette pixel count
    and **failed when measured** (`bench/palette_pctl_probe.py`). A distance
    quantile asks "how far off are the worst pixels", but failures like the
    whole frame going grey or noisy change *most* pixels a little, which a
    quantile misses. Detection with zero false alarms, best quantile vs the
    fraction that replaced it: blur 98.8% / 100%, blend 97.2% / 87.4%,
    **noise sigma 16 0.1% / 100%**. Kept because the probe uses it, and so the
    next person with this idea finds the result instead of repeating it.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must be in (0, 1), got {q}")
    order = np.argsort(dist, kind="stable")
    d, c = dist[order], counts[order]
    cum = np.cumsum(c)
    # Search the running pixel count, so a colour is never split across the
    # rank boundary; the rank lands on whichever colour contains it.
    need = q * cum[-1]
    return float(d[min(int(np.searchsorted(cum, need, side="left")), len(d) - 1)])


def _oriented(ys: np.ndarray, xs: np.ndarray) -> tuple[float, float, float]:
    """Long side, short side and long-axis angle of a shape, from its pixels.

    Uses the shape's own axes (principal component analysis, PCA), not an
    upright box, because the arm links rotate and a pushed block turns. An
    upright box around a square turned 45 degrees has twice the area, so
    compactness reads about 0.5 and looks like a partly hidden block, which is
    common.

    Two things about the angle matter for the action-following check:

      * A PCA axis has no preferred direction, so the angle only means
        something modulo pi and is reported in [0, pi). A link rotating past
        that boundary jumps by nearly pi, so `sign(theta_t+1 - theta_t)` must
        first wrap the difference into (-pi/2, pi/2]. Without that, about one
        step per half-turn reports the wrong direction.
      * Image y points *down*, so the angle increases clockwise on screen. The
        check compares with a commanded joint sign, so it must calibrate that
        sign from data rather than assume it.
    """
    pts = np.stack((xs, ys)).astype(np.float64)
    pts -= pts.mean(axis=1, keepdims=True)
    _, evecs = np.linalg.eigh(pts @ pts.T / pts.shape[1])  # eigh: ascending
    proj = evecs.T @ pts

    # +1.0 because one row of pixels is one pixel wide, not zero. Without it a
    # 1-px-wide shape divides by zero and compactness comes back infinite.
    extent = proj.max(axis=1) - proj.min(axis=1) + 1.0
    major = evecs[:, 1]
    return float(extent[1]), float(extent[0]), float(np.arctan2(major[1], major[0]) % np.pi)


def measure_pixels_only(frame: np.ndarray, palette: Palette, tau: float) -> Measurement:
    """The pixel-only (mode 2) measurements for one `(h, w, 3)` uint8 frame.

    `tau` is how far (straight-line RGB distance) a pixel may be from a palette
    colour and still count as on-palette. `offpalette_px` counts pixels further
    than tau from *every* palette colour. That is the only useful definition
    once pixels are matched to their nearest colour, since every pixel has one.

    `tau` is required, with no default, and lives in `validator.offpalette_tau`.
    A default here would be a second copy of the number: the code's copy would
    silently win while `validator_hash` (a hash of the `validator` config
    section) kept reflecting the config's value, so results taken at different
    taus would carry the same hash and look comparable. `sweep` calibrates it,
    and the current value was calibrated **on tokenizer output**, which behaves
    very differently from renders:

      * a rendered pixel is at most 0.75 from its palette colour; the worst
        pixel of a tokenizer reconstruction is about 155 away, and 115 on a
        typical frame. No tau is both tight and gives zero off-palette pixels
        on reconstructions.
      * so on reconstructions the rule cannot be `> 0`. **Every clean
        reconstruction has off-palette pixels** at any tau below 96, so a `> 0`
        rule would fail every frame and the rollout quality measure would
        always read zero.

    The pass/fail number is `offpalette_frac` (the same count as a fraction of
    the frame) against `validator.offpalette_frac_max`. A fraction replaced a
    raw pixel count because a count grows with frame size, so 64x64 and 96x96
    each needed their own calibrated limit. A fraction needs one number.

    **A distance quantile was tried first and failed when measured**
    (`bench/palette_pctl_probe.py`). It is also resolution-independent, but it
    only looks at the worst pixels, while the failures that matter change most
    pixels a little: gaussian noise at sigma 16 was caught 0.1% of the time
    versus 100% for the fraction. `_weighted_pctl` remains as the probe's
    helper and is deliberately not measured here.

      * The calibrated tau is the measured best, not a compromise. With the
        threshold set at the clean maximum (so zero false alarms by
        construction), a blend of two possible futures is caught 23% of the
        time at tau 8 and 87% at tau 32, and the noise case falls to 0.3% by
        tau 64.

    Real renders are unaffected: at 0.75 they pass any of these taus, so
    `_self_check` still asserts `offpalette_px_max == 0` on them. The full
    table is in the verification log.
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
            continue  # bbox, compactness and angle stay 0; check px_count first
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
        offpalette_frac=float(uniq_counts[dist > tau].sum()) / float(frame.shape[0] * frame.shape[1]),
        px_count=px_count,
        bbox=bbox,
        compactness=compactness,
        link_extent=extent[links],
        link_angle=angle[links],
    )


def measure_with_truth(
    frame: np.ndarray, meta: np.void, palette: Palette, tau: float
) -> tuple[Measurement, Truth]:
    """Mode 1: the same measurements, plus the ground truth the shard carries.

    Identical measurements on purpose. Mode 1 is not a better measurement; it
    is the same one next to the right answer, which lets `sweep` tell whether
    a pixel-only reading is a real fault or a correct frame.
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
        # Split, never read raw. The record keeps the scripted-episode flag in
        # this byte's high bit, so `contact_mask != 0` on the raw byte is true
        # on every scripted frame. See `mirage.data.SCRIPTED_BIT`.
        contact_mask=int(meta["contact_mask"]) & ~int(SCRIPTED_BIT),
        is_scripted=bool(int(meta["contact_mask"]) & int(SCRIPTED_BIT)),
    )
    return measure_pixels_only(frame, palette, tau), truth


class Sweep(NamedTuple):
    """Thresholds that never flag a correct frame, and how much margin they have."""

    frames: int
    tau: float  # the off-palette distance the sweep used
    max_palette_dist: float  # worst distance seen; tau must be above this
    offpalette_px_max: int  # worst off-palette count at that tau, for reference
    offpalette_frac_max: float  # worst per-frame off-palette fraction - THE limit
    min_visible_px: int  # smallest px_count on a block the truth says is visible
    px_count_margin: int  # gap to the next-smallest, i.e. how tight min_px is
    n_unique_max: int  # most distinct colours in a frame; flat-render check
    worst_compactness: float  # over visible blocks, for reference, not a threshold


def sweep(frames: np.ndarray, metas: np.ndarray, palette: Palette, tau: float) -> Sweep:
    """Finds pixel-only thresholds that flag no correct frame.

    A false alarm is the pixel-only mode reporting a fault on a frame the ground
    truth says is fine. So for each measurement the sweep takes the *most
    extreme value over real frames*; any threshold beyond that has zero false
    alarms by construction.

    It skips hidden blocks, which is fair because the ground truth knows: a
    block with `visible_px == 0` really is invisible, so a `px_count` of 0 there
    is correct, and counting it would push `min_px` to zero and make the
    threshold useless. The pixel-only mode just uses the calibrated number.

    Read `px_count_margin` before trusting the result. A margin of 1 px means
    the threshold is on a knife edge and the next new frame will cross it.
    """
    if len(frames) != len(metas):
        raise ValueError(f"{len(frames)} frames against {len(metas)} meta records")

    visible_counts: list[int] = []
    max_dist = 0.0
    off_max = 0
    frac_max = 0.0
    uniq_max = 0
    worst_compact = np.inf

    for frame, meta in zip(frames, metas):
        m, truth = measure_with_truth(frame, meta, palette, tau)
        max_dist = max(max_dist, m.max_palette_dist)
        off_max = max(off_max, m.offpalette_px)
        frac_max = max(frac_max, m.offpalette_frac)
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
        offpalette_frac_max=frac_max,
        min_visible_px=ordered[0],
        px_count_margin=(ordered[1] - ordered[0]) if len(ordered) > 1 else 0,
        n_unique_max=uniq_max,
        worst_compactness=float(worst_compact),
    )


def _self_check(config_path: Path | str | None = None) -> None:
    """Flat-render colour count over every frame, contact and occlusion rates, and the sweep."""
    from mirage import data

    root = Path(__file__).resolve().parent.parent
    cfg, shard_dir, fixture = data.self_check_config(config_path)
    if fixture:
        print(f"no generated shards - running against the committed fixture, {shard_dir}")
    palette = load_palette(root / cfg.sim["scene_xml"])
    print(f"palette: {len(palette.names)} entries {palette.names}, "
          f"{len(palette.links)} links, {len(palette.blocks)} blocks")

    shards = data.load_shards(shard_dir, data_hash=cfg.data_hash)
    index = data.episode_index(shards)
    sampler = data.WindowSampler(shards, index, cfg.data["ctx"])

    # The exact-match trap, asserted rather than described. If this ever fails,
    # `rgba * 255` has become exact and the module docstring's note is out of
    # date, though nearest-colour matching is still the right default.
    frame0 = sampler[0].frames[0]
    exact = (frame0.reshape(-1, 3)[:, None, :] == palette.rgb.astype(np.uint8)[None, :, :]).all(2)
    missed = [palette.names[i] for i in range(len(palette.names)) if not exact[:, i].any()]
    assert missed, "rgba * 255 now lands exactly - the docstring's measurement is stale"
    print(f"exact RGB equality would call {len(missed)} objects missing on a perfect frame: {missed}")

    # Flat rendering (at most 24 distinct colours per frame), over every frame,
    # not a sample. The colour count is cheap, so all 300k frames are
    # affordable, and the claim is about the renderer.
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

    # The sweep, on a sample spread across every shard. Full measurements need a
    # PCA per colour, so this is thousands of frames rather than 300k; a
    # threshold that holds on 8,000 frames from 500 episodes is what is required.
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(sampler), size=min(500, len(sampler)))
    frames = np.concatenate([sampler[int(i)].frames for i in picks])
    metas = np.concatenate([sampler[int(i)].meta for i in picks])
    tau = cfg.validator["offpalette_tau"]
    result = sweep(frames, metas, palette, tau)
    print(f"F-9 sweep over {result.frames:,} frames at tau {result.tau}:")
    print(f"  worst palette distance {result.max_palette_dist:6.2f} - tau must exceed this")
    print(f"  offpalette_px max      {result.offpalette_px_max:6d} - any threshold above is 0 FP")
    print(f"  offpalette share  {result.offpalette_frac_max:9.5%} - the verdict "
          f"statistic, resolution-free; on renders it is exactly 0")
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

    # What the sweep does and does not allow, stated outright. Partial
    # occlusion is common, so a visible block's px_count goes all the way down
    # to 1. A "block missing if px_count < min_px" rule therefore has no margin
    # and cannot be a pass/fail rule on its own. `offpalette_px` does have
    # margin, as the design predicted. Printed so a passing check is not read as
    # "any thresholds work".
    if result.px_count_margin == 0:
        print(f"  -> px_count is NOT usable as a per-frame threshold: a visible block reaches "
              f"{result.min_visible_px} px, margin {result.px_count_margin}. Occlusion, not a bug")
    print(f"  -> viable verdict on renders: offpalette_px > {result.offpalette_px_max} at tau "
          f"{result.tau} ({result.tau / max(result.max_palette_dist, 1e-9):.0f}x the render-rounding floor)")
    print(f"  -> on decoder output the verdict is the same statistic with a "
          f"non-zero bar: offpalette share > "
          f"{cfg.validator['offpalette_frac_max']:.5%}, which is gate row 6 and "
          f"not this check - see fsq_eval.reconstruction_sweep")

    # The assumption behind pixel-only mode: counting a block's colour matches
    # the simulator's segmentation count. With anti-aliasing off (offsamples=0)
    # they should agree closely. A large gap means the palette or the id-colour
    # decode is wrong, which no threshold sweep would reveal.
    diffs = []
    for frame, meta in zip(frames[:2000], metas[:2000]):
        m, truth = measure_with_truth(frame, meta, palette, tau)
        for b, entry in enumerate(palette.blocks):
            diffs.append(int(m.px_count[entry]) - int(truth.visible_px[b]))
    diffs = np.array(diffs)
    print(f"mode 2 px_count vs truth visible_px over {len(diffs):,} readings: "
          f"mean {diffs.mean():+.2f}, max |diff| {np.abs(diffs).max()}, "
          f"exact on {(diffs == 0).mean():.1%}")
    assert np.abs(diffs).max() <= 4, f"pixel-only count is off truth by {np.abs(diffs).max()} px"

    # Contact and occlusion rates over the whole dataset, against the config's
    # minimums. Skipped on the fixture: these are claims about the *dataset*,
    # not this module, and 40 frames from 2 episodes would pass or fail by luck.
    if fixture:
        print("F-6 and F-7: skipped - dataset-scale rates, and the fixture is 40 frames")
    else:
        contact = np.concatenate([data.contact_bits(s.meta) for s in shards])
        f6 = float((contact != 0).mean())

        # A frame counts only when a block is hidden *and comes back*. A block
        # that reads 0 px for the rest of the episode is gone, not hidden.
        # Counting those once inflated this rate from 5.35% to 19.83%, and would
        # make the object-permanence test score events that can never end.
        # Computed per episode, never across episodes: a block hidden at one
        # episode's end would otherwise be "seen again" at the next one's start,
        # which is a reset.
        occluded = 0
        for ep in index:
            block = shards[ep.shard].meta[ep.start:ep.start + ep.length]
            vis = np.stack([np.asarray(block[f"visible_px{b}"])
                            for b in range(len(palette.blocks))])
            occluded += int(((vis == 0) & data.seen_later(vis)).any(axis=0).sum())
        f7 = occluded / sum(s.frames for s in shards)

        floor = cfg.validator["recoverable_occlusion_rate_min"]
        assert f6 >= cfg.validator["contact_rate_min"], f"F-6: {f6:.2%}"
        assert f7 >= floor, f"F-7: {f7:.2%}"
        print(f"F-6 contact {f6:.2%} (floor {cfg.validator['contact_rate_min']:.0%}), "
              f"F-7 recoverable occlusion {f7:.2%} (floor {floor:.0%}) - "
              f"blocks that never return are excluded, see bench/occlusion_probe.py")

    # Pixel-only mode must not need meta. Called with a frame alone, on purpose.
    only = measure_pixels_only(frames[0], palette, tau)
    assert only.px_count.sum() == frames[0].shape[0] * frames[0].shape[1]
    assert np.all((only.link_angle >= 0) & (only.link_angle < np.pi))
    print("mode 2 runs on a frame alone; px_count partitions the frame; angles in [0, pi)")

    print("validator self-check ok" + (" (fixture)" if fixture else ""))


if __name__ == "__main__":
    import sys

    _self_check(sys.argv[1] if len(sys.argv) > 1 else None)
