"""np.memmap over the shard blobs, and an episode-aware window sampler.

The Python half of the format `sim/shard_writer.cpp` writes. This file owns
reading those bytes back and slicing them into windows; what a pixel *means* is
`mirage/validator.py`'s job.

Acceptance is F-8 - the round trip matches the C++ buffer byte for byte - and
that is what `_self_check` does, by decoding every meta field twice: once
through the structured dtype, once by hand with `struct.unpack` at the offsets
`shard_writer.cpp` writes them to. Two independent decoders agreeing is the only
version of this test worth running, because every way of getting the dtype wrong
still reads back cleanly and quietly.

Run the check from the repo root, after a generation run:

    python -m mirage.data
"""

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

# Row order. `mjr_readPixels` returns rows bottom-up - the OpenGL origin is the
# bottom-left - and nothing in `sim/` flips them, so the blob is upside-down on
# disk. Checked against the scene rather than assumed from the convention: the
# camera in `scene/arm_blocks.xml` sits at y=-0.5 at ~24 degrees of elevation,
# so the void past the far table edge belongs at the *top* of the image, and it
# only lands there after the flip. `bench/preview.py` does the same `np.flipud`,
# and `bench/preview.png` is what right-side-up looks like.
#
# The flip lives in the sampler, never in `Shard.pixels`. `Shard.pixels` stays
# the bytes the writer wrote so F-8 has something to compare against, and
# exactly one place - `WindowSampler.__getitem__` - corrects the orientation for
# everything downstream. Two places would eventually disagree, and a
# disagreement here is a mirrored image, which nothing crashes on.


def meta_dtype(joints: int, blocks: int) -> np.dtype:
    """The record `ShardWriter::append` packs, in its field order.

    Little-endian is spelled out on every multi-byte field. The writer's PutF32
    and PutU32 are little-endian by construction, so a native-order dtype agrees
    with them on this machine and stops agreeing the first time this reads a
    shard produced on another.
    """
    fields: list[tuple[str, str]] = [("action", "u1")]
    fields += [(f"qpos{i}", "<f4") for i in range(joints)]
    fields += [(f"block_xy{i}", "<f4") for i in range(2 * blocks)]
    fields += [(f"visible_px{i}", "<u2") for i in range(blocks)]
    fields += [("contact_mask", "u1"), ("episode_id", "<u4"), ("step_idx", "<u2")]

    # np.dtype(list) is packed. np.dtype(..., align=True) would pad this record
    # from 46 bytes to 48, and every frame after the first would then decode
    # from the wrong offset - loading without complaint, garbage from frame 1 on.
    dt = np.dtype(fields)
    expect = 8 + 4 * joints + 10 * blocks
    if dt.itemsize != expect:
        raise ValueError(f"meta dtype is {dt.itemsize} bytes, the writer's formula says {expect}")
    return dt


def meta_struct_format(joints: int, blocks: int) -> str:
    """The same record as a `struct` format string, for the F-8 cross-check.

    Deliberately a second, independent spelling of the layout. If it were
    derived from the dtype it could not disagree with it, and disagreement is
    the whole point.
    """
    return f"<B{joints}f{2 * blocks}f{blocks}HBIH"


@dataclass(frozen=True)
class Shard:
    """One committed shard: its sidecar, and memmaps over both blobs."""

    index: int
    sidecar: dict
    pixels: np.memmap  # (frames, h, w, 3) uint8, rows bottom-up as written
    meta: np.memmap  # (frames,) meta_dtype

    @property
    def frames(self) -> int:
        return int(self.sidecar["frames"])


def load_shards(shard_dir: Path | str, data_hash: str | None = None) -> list[Shard]:
    """Every committed shard in `shard_dir`, in index order.

    The sidecar is the commit marker - `ShardWriter::commit` writes it after
    both blobs close, and the destructor deliberately does not. So a shard with
    blobs and no `.json` is a crashed run, and globbing the sidecars rather than
    the blobs is the entire crash-safety story on the read side.

    Pass `data_hash` (from `mirage.config`) to refuse shards from a different
    config. Two runs writing into one directory is silent otherwise: the frames
    load, the episode ids collide, and the mixture surfaces only as a model that
    will not converge.
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

        # Exact sizes, not lower bounds. A blob longer than frames * per-frame
        # means the sidecar undercounts and the tail is unreachable; shorter
        # means it overcounts and the last read walks off the end. np.memmap
        # accepts either happily by picking its own length.
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
    """Contiguous runs of equal `episode_id`, per shard.

    Runs rather than a group-by, because shards rotate on episode boundaries
    only and steps are appended in order, so an episode is always one contiguous
    block. If either fact stopped holding, a group-by would quietly stitch two
    discontiguous halves into one "episode" and the sampler would draw windows
    across the seam. A run scan cannot: it reports the halves separately, and
    `_self_check`'s "every episode_id appears exactly once" then fails.
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
    """Deterministic train/val split, by episode and never by frame.

    By frame leaks. Frames 4 and 5 of one episode differ by one 2 ms step, so a
    frame-level split puts near-duplicates on both sides and the val loss reads
    lower than the model has earned - the failure that looks like success.

    Hashed rather than a seeded permutation so one episode's side does not move
    when the dataset grows. A permutation reshuffles every episode the moment
    the count changes, which walks the old val set into the new train set.
    """
    digest = hashlib.sha256(f"mirage/val-split:{episode_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64 < val_fraction


class Window(NamedTuple):
    frames: np.ndarray  # (ctx+1, h, w, 3) uint8, right-side up, a real copy
    meta: np.ndarray  # (ctx+1,) meta_dtype - action, truth fields, ids


class WindowSampler:
    """Fixed-length windows that never straddle an episode boundary.

    Map-style on purpose: `__len__` is the count of distinct windows and
    `__getitem__` is a pure function of the index, so a torch DataLoader, a
    shuffle, and a resumed run all address the same window by the same number.
    `sample` exists for the ad-hoc case and takes the rng explicitly rather than
    holding one, so nothing here carries hidden state between calls.

    ponytail: no worker sharding and no prefetch. Measured at ~7,000 windows/s
    single-threaded against P-7's 167 frames/s budget, so `num_workers=0` is
    enough and the Windows spawn-vs-fork memmap-pickling problem never arises.
    Add a worker-local `load_shards` in a `worker_init_fn` if the loader ever
    becomes the bottleneck - `bench/loader_probe.py` is the thing that would
    say so.
    """

    def __init__(
        self,
        shards: Sequence[Shard],
        index: Iterable[Episode],
        ctx: int,
        split: str = "all",
        val_fraction: float = 0.0,
    ) -> None:
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be all, train or val, got {split!r}")

        self.shards = list(shards)
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

        # Cumulative start positions, so `__getitem__` picks uniformly over
        # windows rather than uniformly over episodes. Identical today - every
        # episode is 600 steps - and not identical the moment one run is
        # truncated, at which point uniform-over-episodes would oversample it.
        self._cum = np.cumsum([ep.length - self.window + 1 for ep in self.episodes])

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, i: int) -> Window:
        if not 0 <= i < len(self):
            raise IndexError(f"window {i} out of range for {len(self)} windows")
        k = int(np.searchsorted(self._cum, i, side="right"))
        offset = i - (int(self._cum[k - 1]) if k else 0)
        ep = self.episodes[k]
        lo = ep.start + offset
        hi = lo + self.window
        shard = self.shards[ep.shard]

        # np.array, not a view: it forces the page-in and does the row flip in
        # the one copy the caller needs anyway. A view defers both into the
        # training step, where the cost stops being attributable to the loader
        # and a probe of this file reports a throughput it does not have.
        return Window(
            frames=np.array(shard.pixels[lo:hi, ::-1]),
            meta=np.array(shard.meta[lo:hi]),
        )

    def sample(self, rng: np.random.Generator) -> Window:
        return self[int(rng.integers(len(self)))]


def _self_check() -> None:
    """F-8, plus the invariants the sampler's correctness rests on."""
    from mirage import config

    root = Path(__file__).resolve().parent.parent
    cfg = config.load(root / "mirage" / "configs" / "base.json")
    shard_dir = root / cfg.data["shard_dir"]

    shards = load_shards(shard_dir, data_hash=cfg.data_hash)
    total = sum(s.frames for s in shards)
    committed = {p.stem for p in shard_dir.glob("shard_*.json")}
    orphans = {p.stem for p in shard_dir.glob("shard_*.pixels")} - committed
    print(f"{len(shards)} shards, {total:,} frames, {len(orphans)} incomplete (skipped)")

    # F-8: the structured dtype agrees with a hand decode at the documented
    # offsets. Sampled rather than exhaustive - a wrong dtype is wrong on every
    # frame, so 64 per shard is as conclusive as 43,200 and runs in a second.
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

    # The flip happened, exactly once.
    ep0 = index[0]
    raw0 = np.array(shards[ep0.shard].pixels[ep0.start])
    assert np.array_equal(sampler[0].frames[0], raw0[::-1]), "sampler rows are not the blob's reversed"
    assert not np.array_equal(sampler[0].frames[0], raw0), "flip is a no-op - a symmetric frame?"
    assert np.array_equal(sampler[7].frames, sampler[7].frames), "reads are not repeatable"
    print("orientation: sampler rows are the blob's reversed, and reversed only once")

    train = WindowSampler(shards, index, ctx, "train", cfg.data["val_fraction"])
    val = WindowSampler(shards, index, ctx, "val", cfg.data["val_fraction"])
    t_ids = {ep.episode_id for ep in train.episodes}
    v_ids = {ep.episode_id for ep in val.episodes}
    assert not (t_ids & v_ids), "an episode is in both splits"
    assert t_ids | v_ids == set(ids), "the split lost episodes"
    share = len(v_ids) / len(ids)
    assert abs(share - cfg.data["val_fraction"]) < 0.02, f"val share {share:.3f}"
    print(f"split: {len(t_ids)} train / {len(v_ids)} val episodes ({share:.1%}), disjoint, nothing dropped")

    print("data self-check ok")


if __name__ == "__main__":
    _self_check()
