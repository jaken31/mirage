"""Memory-mapped reading of shard files, and a sampler that respects episodes.

The Python side of the format `sim/shard_writer.cpp` writes. This file reads
those bytes back and slices them into training windows; what a pixel *means*
is `mirage/validator.py`'s job.

The key test is that Python reads exactly the bytes C++ wrote. `_self_check`
decodes every meta field twice: once through the numpy record type, once by
hand with `struct.unpack` at the offsets `shard_writer.cpp` uses. Two
independent decoders must agree, because every way of getting the record type
wrong still loads without error.

Run the check from the repo root:

    python -m mirage.data

It uses the generated 300k-frame dataset if present and the committed 40-frame
fixture otherwise, so it runs in a fresh clone. See `self_check_config`.
"""

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from mirage import config

# How actions line up with joint angles. This is the one fact about the record
# that cannot be recovered from the record itself. `sim/main.cpp` picks the
# action, writes it to `ctrl`, calls `mj_step`, and only then reads the truth,
# so within one record the action is the one applied *during* the step that
# produced that record's `qpos`:
#
#     qpos[t] - qpos[t-1]  is the result of  action[t]      <- same record
#
# The action-following check compares `sign(theta_t+1 - theta_t)` with the
# commanded sign, so it must pair each change with the action in the *later*
# of the two records.
#
# **The data cannot decide this, and tuning for the best score picks the wrong
# answer.** Each action is held for `sim.action_hold_steps` frames, so a
# one-step shift changes only 1 frame in 15, and both alignments pass the 90%
# bar a couple of points apart. The wrong one scores *higher*: shifting pairs
# each change with the previous, already-settled action instead of the new one
# the joint has not caught up with yet. Both measured scores are in the
# verification log at the end of `docs/world_model_architecture.md`.
#
# What does fix it is the *timing* of action changes against `step_idx`, which
# `_self_check` asserts: `Policy::step` picks a new action only when the hold
# runs out, so an action can only change where
# `step_idx % action_hold_steps == 0`. A writer off by one step would move
# every change off that beat.

# Row order. `mjr_readPixels` returns rows bottom-up (OpenGL's origin is the
# bottom-left) and nothing in `sim/` flips them, so frames are upside-down on
# disk. Checked against the scene, not just assumed: the camera sits in front
# of the table, looking slightly down, so the empty space past the far table
# edge belongs at the *top* of the image, and it only lands there after
# flipping. `bench/preview.py` does the same `np.flipud`, and
# `bench/preview.png` shows the right way up.
#
# Flip in the sampler, never in `Shard.pixels`. `Shard.pixels` stays exactly
# the bytes the writer wrote so the byte-for-byte check has something to
# compare, and exactly one place (`WindowSampler.__getitem__`) fixes the
# orientation. Two places would eventually disagree, and the result would be
# a mirrored image, which crashes nothing.


def meta_dtype(joints: int, blocks: int) -> np.dtype:
    """The record layout `ShardWriter::append` writes, in its field order.

    Every multi-byte field is explicitly little-endian, like the writer. A
    native-order type would happen to match on this machine and break on a
    machine with a different byte order.
    """
    fields: list[tuple[str, str]] = [("action", "u1")]
    fields += [(f"qpos{i}", "<f4") for i in range(joints)]
    fields += [(f"block_xy{i}", "<f4") for i in range(2 * blocks)]
    fields += [(f"visible_px{i}", "<u2") for i in range(blocks)]
    fields += [("contact_mask", "u1"), ("episode_id", "<u4"), ("step_idx", "<u2")]

    # np.dtype(list) has no padding. align=True would pad the record from 46 to
    # 48 bytes, and every frame after the first would decode from the wrong
    # offset: it would load without complaint and be garbage from frame 1 on.
    dt = np.dtype(fields)
    expect = 8 + 4 * joints + 10 * blocks
    if dt.itemsize != expect:
        raise ValueError(f"meta dtype is {dt.itemsize} bytes, the writer's formula says {expect}")
    return dt


# The record's `contact_mask` byte holds two fields: bits 0..6 mean "block i
# touches the arm", bit 7 means "this episode is a scripted reach". Packed by
# `ShardWriter::append`; `sim/shard_writer.h` has the C++ copy of this
# constant, and `sim/truth.cpp` allows at most seven blocks so they cannot
# overlap.
#
# A spare bit instead of a new byte, which would grow the record from 46 to 47
# bytes (13.8 -> 14.1 MB of meta) for one boolean.
#
# **Every reader must mask it off.** `meta["contact_mask"] != 0` on the raw
# byte counts every scripted frame as a contact, reading the contact rate as
# over 50% instead of about 17%, with no error.
SCRIPTED_BIT = np.uint8(0x80)


def contact_bits(meta) -> np.ndarray:
    """Block-contact bits, with the scripted flag masked off. Use this for contact rates."""
    return np.asarray(meta["contact_mask"]) & ~SCRIPTED_BIT


def scripted(meta) -> np.ndarray:
    """True where the frame comes from a scripted-reach episode (half of them).

    The same for every frame of one episode, since the coin is flipped in
    `Policy::begin_episode`; `_self_check` asserts that.
    """
    return (np.asarray(meta["contact_mask"]) & SCRIPTED_BIT) != 0


def seen_later(visible: np.ndarray) -> np.ndarray:
    """True where this block is visible again at some *strictly later* step.

    Works on the last axis, which must be one episode's steps in order:
    `(steps,)`, `(blocks, steps)` and `(blocks, episodes, steps)` all work. Never
    run it across episodes: a block hidden at the end of one episode would be
    "seen again" at the start of the next, which is a reset, not a reappearance.

    This is what separates occlusion from loss. A frame with `visible_px == 0`
    is *occlusion* if the block comes back later, and a block that is simply gone
    (say, knocked off the table) if it does not. Only occlusion counts. It
    matters: `bench/occlusion_probe.py` found that 73% of the old occlusion
    count was blocks that never came back.

    Computed as a reversed running maximum, shifted one step so "later" excludes
    the current frame. Without the shift a block visible *now* would count as
    visible later, and every occlusion run would read one frame short.
    """
    later = np.flip(np.maximum.accumulate(np.flip(visible > 0, axis=-1), axis=-1), axis=-1)
    return np.concatenate([later[..., 1:], np.zeros_like(later[..., :1])], axis=-1)


def meta_struct_format(joints: int, blocks: int) -> str:
    """The same record as a `struct` format string, for the byte-level cross-check.

    Deliberately written out separately from the numpy type. If it were derived
    from it, the two could never disagree, and catching disagreement is the
    point.
    """
    return f"<B{joints}f{2 * blocks}f{blocks}HBIH"


@dataclass(frozen=True)
class Shard:
    """One complete shard: its sidecar, and memory maps over both data files."""

    index: int
    sidecar: dict
    pixels: np.memmap  # (frames, h, w, 3) uint8, rows bottom-up as written
    meta: np.memmap  # (frames,) meta_dtype

    @property
    def frames(self) -> int:
        return int(self.sidecar["frames"])


def load_shards(shard_dir: Path | str, data_hash: str | None = None) -> list[Shard]:
    """Every complete shard in `shard_dir`, in index order.

    The `.json` sidecar marks a shard complete: `ShardWriter::commit` writes it
    after both data files close, and the destructor deliberately does not. So
    data files with no `.json` are a crashed run, and listing sidecars instead of
    data files is all the read side needs for crash safety.

    Pass `data_hash` (from `mirage.config`) to refuse shards made from a
    different config. Otherwise two runs writing into one directory go
    unnoticed: the frames load, the episode ids collide, and the only symptom is
    a model that will not converge.
    """
    shard_dir = Path(shard_dir)
    sidecars = sorted(shard_dir.glob("shard_*.json"))
    if not sidecars:
        raise FileNotFoundError(
            f"no committed shards in {shard_dir} (blobs without a .json are incomplete)"
        )

    shards: list[Shard] = []
    for path in sidecars:
        side = json.loads(path.read_text())

        if side["pixel_dtype"] != "uint8":
            raise ValueError(
                f"{path.name}: pixel_dtype is {side['pixel_dtype']!r}, this reader only knows uint8"
            )

        dt = meta_dtype(side["meta_joints"], side["meta_blocks"])
        if dt.itemsize != side["meta_record_bytes"]:
            raise ValueError(
                f"{path.name}: record is {side['meta_record_bytes']} bytes on disk, "
                f"this reader builds {dt.itemsize}"
            )

        n, h, w, c = side["frames"], side["height"], side["width"], side["channels"]
        pixels_path = path.with_suffix(".pixels")
        meta_path = path.with_suffix(".meta")

        # Exact sizes, not minimums. A file longer than frames * per-frame bytes
        # means the sidecar undercounts and the tail is unreachable; shorter
        # means it overcounts and the last read runs off the end. np.memmap
        # would accept either without complaint.
        for blob, want in ((pixels_path, n * h * w * c), (meta_path, n * dt.itemsize)):
            got = blob.stat().st_size
            if got != want:
                raise ValueError(f"{blob.name}: {got} bytes on disk, sidecar implies {want}")

        shards.append(
            Shard(
                index=int(side["shard_index"]),
                sidecar=side,
                pixels=np.memmap(pixels_path, dtype=np.uint8, mode="r", shape=(n, h, w, c)),
                meta=np.memmap(meta_path, dtype=dt, mode="r", shape=(n,)),
            )
        )

    hashes = {s.sidecar["data_hash"] for s in shards}
    if len(hashes) != 1:
        raise ValueError(f"{shard_dir} mixes {len(hashes)} data_hashes: {sorted(hashes)}")
    if data_hash is not None and hashes != {data_hash}:
        raise ValueError(f"shards carry data_hash {hashes.pop()}, the config says {data_hash}")

    shards.sort(key=lambda s: s.index)
    return shards


class Episode(NamedTuple):
    shard: int  # position in the list load_shards returned, not shard_index
    start: int  # first frame, within that shard
    length: int
    episode_id: int


def episode_index(shards: Sequence[Shard]) -> list[Episode]:
    """Runs of consecutive frames with the same `episode_id`, per shard.

    Scans for runs instead of grouping by id, because an episode should always be
    one unbroken block (shards only switch between episodes, and steps are written
    in order). If that ever broke, grouping would silently join two separate
    pieces into one "episode" and the sampler would draw windows across the gap.
    A run scan reports the pieces separately, so `_self_check`'s "every
    episode_id appears exactly once" fails instead.
    """
    index: list[Episode] = []
    for si, shard in enumerate(shards):
        ep = np.asarray(shard.meta["episode_id"])
        cuts = np.flatnonzero(np.diff(ep)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(ep)]))
        index += [Episode(si, int(a), int(b - a), int(ep[a])) for a, b in zip(starts, ends)]
    return index


def is_val(episode_id: int, val_fraction: float) -> bool:
    """Deterministic train/validation split, by episode, never by frame.

    Splitting by frame leaks. Neighbouring frames of one episode differ by one
    2 ms step, so a frame-level split puts near-duplicates on both sides and the
    validation loss looks better than the model deserves.

    Uses a hash rather than a seeded shuffle so an episode stays on the same side
    when the dataset grows. A shuffle reorders everything when the count changes,
    moving old validation episodes into the new training set.
    """
    digest = hashlib.sha256(f"mirage/val-split:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64 < val_fraction


class Window(NamedTuple):
    frames: np.ndarray  # (ctx+1, h, w, 3) uint8, right-side up, a real copy
    meta: np.ndarray  # (ctx+1,) meta_dtype - action, truth fields, ids


class WindowIndex:
    """Which frames window number `i` holds: an episode and an offset into it.

    The addressing half of `WindowSampler`, split out so `mirage.dynamics` reads
    its token windows through the same arithmetic instead of a copy. Window `i`
    is frames `ep.start + offset` up to `ep.start + offset + window` of shard
    `ep.shard`, where `(ep, offset) = locate(i)`. A second copy of this, off by
    one somewhere, would pair the tokens of one window with the frames of
    another and crash nothing.
    """

    def __init__(
        self,
        index: Iterable[Episode],
        ctx: int,
        split: str = "all",
        val_fraction: float = 0.0,
    ) -> None:
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be all, train or val, got {split!r}")

        self.window = ctx + 1
        self.split = split

        keep = []
        for ep in index:
            if ep.length < self.window:
                continue
            if split != "all" and is_val(ep.episode_id, val_fraction) != (split == "val"):
                continue
            keep.append(ep)
        if not keep:
            raise ValueError(f"no episode of at least {self.window} frames in split {split!r}")
        self.episodes: list[Episode] = keep

        # Running totals of window counts, so `locate` is uniform over windows,
        # not over episodes. The same thing today (every episode is 600 steps),
        # but not once an episode is shorter: uniform over episodes would then
        # oversample it.
        self._cum = np.cumsum([ep.length - self.window + 1 for ep in self.episodes])

    def __len__(self) -> int:
        return int(self._cum[-1])

    def locate(self, i: int) -> tuple[Episode, int]:
        if not 0 <= i < len(self):
            raise IndexError(f"window {i} out of range for {len(self)} windows")
        k = int(np.searchsorted(self._cum, i, side="right"))
        offset = i - (int(self._cum[k - 1]) if k else 0)
        return self.episodes[k], offset


class WindowSampler:
    """Fixed-length windows that never cross an episode boundary.

    Indexable on purpose: `__len__` is the number of distinct windows and
    `__getitem__` depends only on the index, so a torch DataLoader, a shuffle
    and a resumed run all get the same window for the same number. `sample` is
    for one-off use and takes the random generator as an argument, so nothing
    here keeps hidden state between calls.

    ponytail: no worker processes and no prefetch. One thread gives about 7,000
    windows/s, far above the ~167 frames/s a full epoch in 30 minutes needs, so
    `num_workers=0` is enough and Windows' trouble passing memory maps to worker
    processes never comes up. If the loader ever becomes the bottleneck
    (`bench/loader_probe.py` would show it), call `load_shards` per worker in a
    `worker_init_fn`.
    """

    def __init__(
        self,
        shards: Sequence[Shard],
        index: Iterable[Episode],
        ctx: int,
        split: str = "all",
        val_fraction: float = 0.0,
    ) -> None:
        self.shards = list(shards)
        self.windows = WindowIndex(index, ctx, split, val_fraction)
        self.window = self.windows.window
        self.split = split
        self.episodes: list[Episode] = self.windows.episodes

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Window:
        ep, offset = self.windows.locate(i)
        lo = ep.start + offset
        hi = lo + self.window
        shard = self.shards[ep.shard]

        # np.array, not a view: it reads the data from disk now and does the row
        # flip in the one copy the caller needs anyway. A view would push both
        # into the training step, where the cost is hidden, and a loader
        # benchmark would report speed it does not have.
        return Window(
            frames=np.array(shard.pixels[lo:hi, ::-1]),
            meta=np.array(shard.meta[lo:hi]),
        )

    def sample(self, rng: np.random.Generator) -> Window:
        return self[int(rng.integers(len(self)))]


FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "fixture.json"


def split_episodes(index: Sequence[Episode], split: str,
                   val_fraction: float) -> list[Episode]:
    """One split's episodes, in index order; this is the row order used below.

    Shared because `preload` and `split_meta` must agree exactly. They return the
    pixels and the truth for the same rows, and the caller pairs them up. If
    they selected or ordered episodes differently, every truth field would be
    matched to the wrong frame while both arrays still had the right length and
    type, so nothing would raise.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split is {split!r}, expected 'train' or 'val'")
    want_val = split == "val"
    return [e for e in index if is_val(e.episode_id, val_fraction) == want_val]


def split_meta(shards: Sequence[Shard], index: Sequence[Episode], split: str,
               val_fraction: float) -> np.ndarray:
    """The meta records that `preload` leaves out: same split, rows and order.

    `preload` returns only pixels, which is all training needs. The validator
    sweep needs the truth alongside the pixels, so it is rebuilt here instead of
    widening `preload`'s return value: `preload` runs in the training path where
    the meta would be dead weight, and this runs once per calibration.
    """
    episodes = split_episodes(index, split, val_fraction)
    return np.concatenate([np.asarray(shards[e.shard].meta[e.start:e.start + e.length])
                           for e in episodes])


def preload(
    shards: Sequence[Shard],
    index: Sequence[Episode],
    split: str,
    val_fraction: float,
    palette_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One split's frames as palette indices, plus the lookup table to undo it.

    Returns `(n, h, w)` uint8 indices and a `(p, 3)` uint8 lookup table `lut`,
    so `lut[indices]` is the original RGB, rows already flipped right-side up.

    Indices instead of RGB because only **7** distinct colours appear across all
    300,000 frames, so one byte per pixel loses nothing and the training split
    fits in 1.16 GB instead of 3.49 GB at 64x64. At 3.5 GB the OS file cache
    cannot be relied on: the loader reads about 6,800 frames/s from disk versus
    about 110,000 from cache, and training needs about 13,000.

    `palette_rgb` is passed in because `mirage.validator` imports this module,
    so importing `load_palette` here would be circular. Pass
    `validator.load_palette(cfg.sim["scene_xml"]).rgb`.

    Checked to be lossless, not assumed. Each distinct colour maps to its
    nearest palette entry. Exact RGB matching does not work, because
    `rgba * 255` is not a whole number, and it would call four objects missing
    on a perfect frame. Two entries claiming one colour would make the table
    impossible to invert, so that is checked too.
    """
    episodes = split_episodes(index, split, val_fraction)
    n = sum(e.length for e in episodes)
    h = int(shards[0].sidecar["height"])
    w = int(shards[0].sidecar["width"])

    out = np.empty((n, h, w), dtype=np.uint8)
    palette = np.asarray(palette_rgb, dtype=np.float64)
    # Colours are packed into one int as r<<16 | g<<8 | b. With seven distinct
    # colours over 1.2e9 pixels, the nearest-colour search runs once per
    # distinct colour and each pixel is just a sorted lookup, instead of a
    # (pixels, 7) distance matrix that would not fit in memory.
    known_keys = np.empty(0, dtype=np.uint32)
    known_idx = np.empty(0, dtype=np.uint8)
    lut = np.zeros((len(palette), 3), dtype=np.uint8)
    claimed: dict[int, int] = {}
    worst = 0.0

    at = 0
    for ep in episodes:
        px = shards[ep.shard].pixels[ep.start:ep.start + ep.length, ::-1]
        block = np.asarray(px, dtype=np.uint8)
        keys = (block[..., 0].astype(np.uint32) << 16
                | block[..., 1].astype(np.uint32) << 8
                | block[..., 2].astype(np.uint32))

        fresh = np.setdiff1d(np.unique(keys), known_keys)
        if fresh.size:
            rgb = np.stack([fresh >> 16 & 255, fresh >> 8 & 255, fresh & 255], axis=1)
            d2 = ((rgb[:, None, :].astype(np.float64) - palette[None, :, :]) ** 2).sum(2)
            nearest = d2.argmin(1)
            worst = max(worst, float(np.sqrt(d2.min(1)).max()))
            for key, pi, triple in zip(fresh.tolist(), nearest.tolist(), rgb):
                if claimed.setdefault(pi, key) != key:
                    raise ValueError(
                        f"palette entry {pi} claimed by two triples: "
                        f"{claimed[pi]:#08x} and {key:#08x} - the LUT cannot invert"
                    )
                lut[pi] = triple
            known_keys = np.concatenate([known_keys, fresh])
            known_idx = np.concatenate([known_idx, nearest.astype(np.uint8)])
            order = np.argsort(known_keys)
            known_keys, known_idx = known_keys[order], known_idx[order]

        out[at:at + ep.length] = known_idx[np.searchsorted(known_keys, keys)]
        at += ep.length

    assert at == n, f"wrote {at} frames into room for {n}"
    if worst >= 1.0:
        raise ValueError(f"a pixel sits {worst:.3f} from its palette entry, expected < 1.0")
    if len(claimed) != len(palette):
        raise ValueError(
            f"{len(claimed)} of {len(palette)} palette entries appear in the {split} "
            f"split, so the LUT has undefined rows"
        )
    return out, lut



def self_check_config(config_path: Path | str | None = None) -> tuple["config.Config", Path, bool]:
    """The config and shard dir the self-checks use, and which kind it is.

    `mirage/configs/base.json` and its generated dataset if that exists;
    otherwise the committed 40-frame fixture in `mirage/fixtures/`. Without the
    fallback, the reader and validator checks could not run in a fresh clone,
    a linked worktree, or CI: `data/` is 3.5 GB and gitignored, so only someone
    who first generated 300,000 frames could run them.

    The fixture is real simulator output, not fake data built from `meta_dtype`
    and random pixels. The check is about the bytes `sim/shard_writer.cpp`
    actually writes, and a fake shard would agree with the reader by
    construction while testing nothing.

    It has its own `data_hash` over its own config, so editing
    `scene/arm_blocks.xml` or the `sim` section invalidates it, and
    `load_shards` refuses it by name instead of reading stale frames.
    Regenerate it the way it was made, from the repo root:

        <build>/Release/mirage_sim.exe mirage/fixtures/fixture.json \
            --data-hash <config.load(FIXTURE_CONFIG).data_hash> --git-sha <sha>
    """
    root = Path(__file__).resolve().parent.parent

    # An explicit config gets no fixture fallback, so a missing dataset fails
    # by name in load_shards instead of quietly checking 40 other frames and
    # printing ok. This is how another resolution is checked:
    # `python -m mirage.validator mirage/configs/base96.json`.
    if config_path is not None:
        cfg = config.load(config_path)
        return cfg, root / cfg.data["shard_dir"], False

    cfg = config.load(root / "mirage" / "configs" / "base.json")
    shard_dir = root / cfg.data["shard_dir"]
    if any(shard_dir.glob("shard_*.json")):
        return cfg, shard_dir, False

    cfg = config.load(FIXTURE_CONFIG)
    return cfg, root / cfg.data["shard_dir"], True


def _self_check(config_path: Path | str | None = None) -> None:
    """Byte-exact reading of the writer's output, plus the rules the sampler relies on."""
    cfg, shard_dir, fixture = self_check_config(config_path)
    if fixture:
        print(f"no generated shards - running against the committed fixture, {shard_dir}")

    shards = load_shards(shard_dir, data_hash=cfg.data_hash)
    total = sum(s.frames for s in shards)
    committed = {p.stem for p in shard_dir.glob("shard_*.json")}
    orphans = {p.stem for p in shard_dir.glob("shard_*.pixels")} - committed
    print(f"{len(shards)} shards, {total:,} frames, {len(orphans)} incomplete (skipped)")

    # The numpy record type must agree with a hand decode at the documented
    # offsets. A sample, not every frame: a wrong type is wrong on every
    # frame, so 64 per shard proves as much as 43,200 and takes a second.
    rng = np.random.default_rng(0)
    checked = 0
    for shard in shards:
        side = shard.sidecar
        joints, blocks = side["meta_joints"], side["meta_blocks"]
        fmt = meta_struct_format(joints, blocks)
        assert struct.calcsize(fmt) == side["meta_record_bytes"], (fmt, side["meta_record_bytes"])
        raw = np.asarray(shard.meta).view(np.uint8).reshape(shard.frames, -1)

        for f in rng.integers(0, shard.frames, size=64):
            f = int(f)
            hand = struct.unpack(fmt, raw[f].tobytes())
            row = shard.meta[f]
            want = (
                [int(row["action"])]
                + [float(row[f"qpos{i}"]) for i in range(joints)]
                + [float(row[f"block_xy{i}"]) for i in range(2 * blocks)]
                + [int(row[f"visible_px{i}"]) for i in range(blocks)]
                + [int(row["contact_mask"]), int(row["episode_id"]), int(row["step_idx"])]
            )
            assert list(hand) == want, f"shard {shard.index} frame {f}: {hand} != {tuple(want)}"
            checked += 1
    print(f"F-8: {checked} records decode identically through the dtype and through struct.unpack")

    index = episode_index(shards)
    steps = cfg.sim["steps_per_episode"]
    assert len(index) == cfg.sim["episodes"], f"{len(index)} episodes, config says {cfg.sim['episodes']}"
    assert all(ep.length == steps for ep in index), "an episode is not steps_per_episode long"
    ids = sorted(ep.episode_id for ep in index)
    assert ids == list(range(cfg.sim["episodes"])), "episode ids are not 0..episodes-1, each once"
    assert all(a.shard <= b.shard for a, b in zip(index, index[1:])), "episode spans two shards"
    print(f"index: {len(index)} episodes of {steps} steps, ids 0..{ids[-1]}, none split across shards")

    ctx = cfg.data["ctx"]
    sampler = WindowSampler(shards, index, ctx)
    assert len(sampler) == cfg.sim["episodes"] * (steps - ctx)
    ones = np.ones(ctx, np.int32)
    for i in rng.integers(0, len(sampler), size=20000):
        w = sampler[int(i)]
        assert w.frames.shape == (ctx + 1, cfg.sim["height"], cfg.sim["width"], 3), w.frames.shape
        assert len(np.unique(w.meta["episode_id"])) == 1, "window straddles an episode boundary"
        assert np.array_equal(np.diff(w.meta["step_idx"].astype(np.int32)), ones), "step_idx jumps"
    print(f"20,000 windows of {ctx + 1}: one episode_id each, step_idx contiguous")

    # The action-to-angle alignment, checked the only way the data allows (see
    # the note at the top of this file). `Policy::step` picks a new action only
    # when its hold runs out, so every action change must fall where
    # `step_idx % action_hold_steps == 0`. A writer one step off would move
    # every change off that beat. Nothing else catches it: both alignments
    # score around 90% on action-following, and the wrong one scores higher.
    hold = cfg.sim["action_hold_steps"]
    phases: set[int] = set()
    changes = 0
    for shard in shards:
        a = np.asarray(shard.meta["action"])
        st = np.asarray(shard.meta["step_idx"]).astype(np.int64)
        ep = np.asarray(shard.meta["episode_id"])
        # The first record of each episode always counts as a change (the hold
        # is zeroed in `begin_episode`), and it sits at step_idx 0, which is on
        # the beat, so it needs no special case.
        changed = np.concatenate(([True], (a[1:] != a[:-1]) | (ep[1:] != ep[:-1])))
        phases |= set(np.unique(st[changed] % hold).tolist())
        changes += int(changed.sum())
    assert phases == {0}, (
        f"actions change at step_idx phases {sorted(phases)} mod {hold}, expected "
        f"{{0}} - the action is out of step with the truth in the same record"
    )
    print(f"alignment: all {changes:,} action changes sit at step_idx % {hold} == 0, "
          f"so action[t] is the action that produced qpos[t]")

    # The row flip happened, exactly once.
    ep0 = index[0]
    raw0 = np.array(shards[ep0.shard].pixels[ep0.start])
    assert np.array_equal(sampler[0].frames[0], raw0[::-1]), "sampler rows are not the blob's reversed"
    assert not np.array_equal(sampler[0].frames[0], raw0), "flip is a no-op - a symmetric frame?"
    assert np.array_equal(sampler[7].frames, sampler[7].frames), "reads are not repeatable"
    print("orientation: sampler rows are the blob's reversed, and reversed only once")

    # The scripted flag. It is chosen per episode, so it must be the same on
    # every frame of one. Checked, because a per-frame bug would still give a
    # plausible 50/50 split over the whole dataset and fool any overall count.
    flags = []
    for ep in index:
        block = shards[ep.shard].meta[ep.start:ep.start + ep.length]
        seen = np.unique(scripted(block))
        assert len(seen) == 1, f"episode {ep.episode_id} mixes scripted and random frames"
        flags.append(bool(seen[0]))
    share = sum(flags) / len(flags)
    if fixture:
        print(f"scripted flag: constant within both fixture episodes, {sum(flags)} of 2 scripted")
    else:
        assert 0.4 < share < 0.6, f"scripted share {share:.1%} - the 50/50 coin is not fair"
        print(f"scripted flag: constant within all {len(index)} episodes, {share:.1%} scripted")

    # `is_val` depends only on the episode id, so its split ratio can be checked
    # without any dataset. That keeps the split tested on the fixture, whose two
    # episodes are far too few to measure a 5% share.
    synthetic = sum(is_val(i, 0.05) for i in range(4000)) / 4000
    assert abs(synthetic - 0.05) < 0.01, f"is_val sends {synthetic:.1%} of 4,000 ids to val"
    print(f"is_val: {synthetic:.2%} of 4,000 synthetic episode ids land in val")

    if fixture:
        print(f"split: skipped - the episode-level split needs hundreds of episodes to land "
              f"within tolerance and the fixture has {len(ids)}")
        print("data self-check ok (fixture)")
        return

    train = WindowSampler(shards, index, ctx, "train", cfg.data["val_fraction"])
    val = WindowSampler(shards, index, ctx, "val", cfg.data["val_fraction"])
    t_ids = {ep.episode_id for ep in train.episodes}
    v_ids = {ep.episode_id for ep in val.episodes}
    assert not (t_ids & v_ids), "an episode is in both splits"
    assert t_ids | v_ids == set(ids), "the split lost episodes"
    share = len(v_ids) / len(ids)
    assert abs(share - cfg.data["val_fraction"]) < 0.02, f"val share {share:.3f}"
    print(f"split: {len(t_ids)} train / {len(v_ids)} val episodes ({share:.1%}), disjoint, nothing dropped")

    # preload, on the validation split only. The training split runs the same
    # code over 17x the frames, and building 1.16 GB (2.62 GB at 96x96) on every
    # self-check would cost real time for no extra coverage. runs.jsonl records
    # a full build.
    from mirage.validator import load_palette  # deferred: validator imports this module

    palette = load_palette(Path(cfg.sim["scene_xml"]))
    val_idx, lut = preload(shards, index, "val", cfg.data["val_fraction"], palette.rgb)
    h = int(shards[0].sidecar["height"])
    train_frames = total - len(val_idx)
    print(f"preload: {len(val_idx):,} val frames as {val_idx.nbytes / 1e6:.1f} MB of "
          f"indices, {len(lut)}-entry LUT; the train split is {train_frames:,} frames "
          f"= {train_frames * h * h / 1e9:.2f} GB against {train_frames * h * h * 3 / 1e9:.2f} raw")

    # The main check: the lookup table exactly restores the colours, compared
    # with an independent read of the pixel file. A sample is enough because
    # this is an exact byte match: a mapping error shows on any frame that
    # contains the wrong colour.
    rng = np.random.default_rng(0)
    checked = 0
    at = 0
    want_val = True
    for ep in index:
        if is_val(ep.episode_id, cfg.data["val_fraction"]) != want_val:
            continue
        for _ in range(min(8, ep.length)):
            j = int(rng.integers(ep.length))
            got = lut[val_idx[at + j]]
            expect = np.array(shards[ep.shard].pixels[ep.start + j, ::-1])
            assert np.array_equal(got, expect), f"LUT round-trip differs at episode {ep.episode_id}"
            checked += 1
        at += ep.length
    print(f"preload: LUT[indices] is byte-identical to a direct flipped read on "
          f"{checked:,} random frames, worst palette distance under 1.0")

    print("data self-check ok")


if __name__ == "__main__":
    import sys

    _self_check(sys.argv[1] if len(sys.argv) > 1 else None)
