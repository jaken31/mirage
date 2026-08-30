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

Run the check from the repo root:

    python -m mirage.data

It uses the generated 300k-frame set when there is one and the committed
40-frame fixture when there is not, so F-8 is runnable in a fresh clone. See
`self_check_config`.
"""

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from mirage import config

# Action-to-qpos alignment, which is the one thing about this record that a
# reader cannot recover from the record. `sim/main.cpp` picks the action, writes
# it to `ctrl`, calls `mj_step`, and only then reads the truth - so within one
# record the action is the one applied *during* the step that produced that
# record's `qpos`:
#
#     qpos[t] - qpos[t-1]  is the result of  action[t]      <- same record
#
# Q-4 scores `sign(theta_t+1 - theta_t)` against the commanded sign, so it must
# pair a delta with the action in the *later* of the two records.
#
# **The data cannot arbitrate this and a sweep will pick the wrong one.** An
# action is held for `sim.action_hold_steps` frames, so a one-step shift leaves
# 14 of every 15 frames unchanged and both readings land a couple of points
# apart, on the passing side of Q-4's 90% bar. The wrong one scores *higher*,
# because shifting hands each delta the previous, already settled action
# instead of the fresh one whose transient Q-4 cannot win - so calibrating the
# alignment by maximising agreement inverts it. Agreement can never settle
# this: a shifted held sequence is still a valid held sequence. Both measured
# rates are in the verification log at the end of
# `docs/world_model_architecture.md`, in the alignment row.
#
# What does pin it is the *phase* of the action stream against `step_idx`, which
# `_self_check` asserts: `Policy::step` redraws only when its hold expires, so an
# action can change only where `step_idx % action_hold_steps == 0`. A writer that
# shifted the record by one step would move every change off that phase.

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


# The record's `contact_mask` byte is two fields sharing one byte: bits 0..6 are
# "block i touches the arm", bit 7 is "this episode is the scripted-reach half".
# Packed by `ShardWriter::append`; `sim/shard_writer.h` holds the C++ half of
# this constant, and `sim/truth.cpp` caps a scene at seven blocks so the two
# cannot collide.
#
# Packed into a spare bit rather than added as a u8, which would take the record
# 46 -> 47 bytes and the meta blobs 13.8 -> 14.1 MB for one boolean.
#
# **Every reader must mask.** `meta["contact_mask"] != 0` on the raw byte counts
# every scripted frame as a contact - that reads F-6 as over 50% instead of
# 16.6%, and nothing fails.
SCRIPTED_BIT = np.uint8(0x80)


def contact_bits(meta) -> np.ndarray:
    """Block-contact bits, with the scripted flag masked off. F-6 reads this."""
    return np.asarray(meta["contact_mask"]) & ~SCRIPTED_BIT


def scripted(meta) -> np.ndarray:
    """True where the frame came from the scripted-reach half of the 50/50 mix.

    Constant across every frame of one episode - the coin is drawn in
    `Policy::begin_episode` - and `_self_check` asserts exactly that.
    """
    return (np.asarray(meta["contact_mask"]) & SCRIPTED_BIT) != 0


def seen_later(visible: np.ndarray) -> np.ndarray:
    """True where this block is visible again at some *strictly later* step.

    Operates on the last axis, which must be one episode's steps in order -
    `(steps,)`, `(blocks, steps)`, `(blocks, episodes, steps)` all work. Never
    run it across an episode boundary: a block hidden at the end of one episode
    would be "seen again" at the start of the next, which is a reset, not a
    reappearance.

    This is the whole of F-7's recoverable/terminal split. A frame with
    `visible_px == 0` is *occlusion* if the block comes back and a block that is
    simply gone if it does not, and the requirement counts only the first -
    measured 2026-08-28, `bench/occlusion_probe.py`: 14.48 of F-7's 19.83 points
    were blocks that never returned, so 73% of the old number was not occlusion.

    A reversed cumulative maximum, then shifted one step so "later" excludes the
    current frame. Without the shift a block visible *now* counts as visible
    later and every occlusion run reads one frame short.
    """
    later = np.flip(np.maximum.accumulate(np.flip(visible > 0, axis=-1), axis=-1), axis=-1)
    return np.concatenate([later[..., 1:], np.zeros_like(later[..., :1])], axis=-1)


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


FIXTURE_CONFIG = Path(__file__).resolve().parent / "fixtures" / "fixture.json"


def split_episodes(index: Sequence[Episode], split: str,
                   val_fraction: float) -> list[Episode]:
    """One split's episodes, in index order - the row order of everything below.

    Factored out because `preload` and `split_meta` have to agree on it exactly.
    They return two halves of the same rows, pixels and truth, and the caller
    zips them; selecting or ordering differently would attribute every truth
    field to the wrong frame while both arrays kept the right length and the
    right dtype, so nothing would raise. One predicate, one iteration order,
    one function.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split is {split!r}, expected 'train' or 'val'")
    want_val = split == "val"
    return [e for e in index if is_val(e.episode_id, val_fraction) == want_val]


def split_meta(shards: Sequence[Shard], index: Sequence[Episode], split: str,
               val_fraction: float) -> np.ndarray:
    """The meta records `preload` drops, same split, same rows, same order.

    `preload` returns pixels alone, which is all training needs. The F-9 sweep
    needs the truth beside the pixels, and rebuilding that half here beats
    widening `preload`'s return: that call sits in the training loop where the
    meta is dead weight, and this one runs once per calibration.
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
    """One split's frames as palette indices, plus the LUT that inverts them.

    Returns `(n, h, w)` uint8 indices and a `(p, 3)` uint8 LUT, so that
    `lut[indices]` is the original RGB, rows already flipped top-down.

    Indices rather than RGB because there are exactly **7** distinct byte
    triples across all 300,000 frames - a union over the whole set, not a
    per-frame count - so one byte per pixel is lossless and the train split
    holds in 1.16 GB instead of 3.49 at 64x64. The page cache cannot be relied
    on at a 3.5 GB working set: the loader reads 6,804 frames/s cold against
    109,682 warm, and training needs ~13,000.

    `palette_rgb` is passed in rather than loaded here because
    `mirage.validator` imports this module, so importing `load_palette` back
    would be a cycle. Pass `validator.load_palette(cfg.sim["scene_xml"]).rgb`.

    Losslessness is asserted, not assumed. Every distinct triple is mapped to
    its nearest palette entry by argmin over squared distance - exact RGB
    equality does not work, because `rgba * 255` does not land on integers, and
    the architecture doc records the four objects it would call missing on a
    flawless frame. Two entries claiming one triple would make the LUT
    non-invertible, so that is checked too.
    """
    episodes = split_episodes(index, split, val_fraction)
    n = sum(e.length for e in episodes)
    h = int(shards[0].sidecar["height"])
    w = int(shards[0].sidecar["width"])

    out = np.empty((n, h, w), dtype=np.uint8)
    palette = np.asarray(palette_rgb, dtype=np.float64)
    # Keys are packed r<<16 | g<<8 | b. Seven distinct values over 1.2e9 pixels,
    # so the argmin runs once per distinct triple and the per-pixel work is a
    # searchsorted, not a (pixels, 7) distance matrix that would not fit.
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
    """The config and shard dir the self-checks run against, and which one it is.

    `mirage/configs/base.json` and its generated set when that set exists; the
    committed 40-frame fixture under `mirage/fixtures/` when it does not.
    Without the fallback, the two checks that *are* F-8's and F-9's acceptance
    tests cannot run in a fresh clone, in a linked worktree, or in CI - `data/`
    is 3.5 GB and correctly gitignored, so nobody who has not first generated
    300,000 frames can run either one.

    The fixture is real writer output, not synthesised in temp from
    `meta_dtype` plus random pixels. F-8 is a claim about the bytes
    `sim/shard_writer.cpp` actually emits, and a synthesised shard would agree
    with the reader by construction while testing nothing.

    It carries its own `data_hash` over its own config, so editing
    `scene/arm_blocks.xml` or the `sim` section invalidates it and
    `load_shards` refuses it by name rather than reading stale frames.
    Regenerate it the way it was made, from the repo root:

        <build>/Release/mirage_sim.exe mirage/fixtures/fixture.json \
            --data-hash <config.load(FIXTURE_CONFIG).data_hash> --git-sha <sha>
    """
    root = Path(__file__).resolve().parent.parent

    # An explicit config is explicit: no fixture fallback, so a missing set
    # fails by name in load_shards rather than quietly checking 40 other frames
    # and printing ok. This is how a second resolution gets verified -
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
    """F-8, plus the invariants the sampler's correctness rests on."""
    cfg, shard_dir, fixture = self_check_config(config_path)
    if fixture:
        print(f"no generated shards - running against the committed fixture, {shard_dir}")

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

    # The action-to-qpos alignment, pinned the one way the data can pin it - see
    # the module-level note. `Policy::step` redraws only when its hold expires,
    # so every action change must sit at `step_idx % action_hold_steps == 0`. A
    # writer that appended the action one step out of step with the truth would
    # move every change to a different phase, which is the failure that is
    # otherwise invisible: both alignments score ~90% on Q-4 and the wrong one
    # scores higher.
    hold = cfg.sim["action_hold_steps"]
    phases: set[int] = set()
    changes = 0
    for shard in shards:
        a = np.asarray(shard.meta["action"])
        st = np.asarray(shard.meta["step_idx"]).astype(np.int64)
        ep = np.asarray(shard.meta["episode_id"])
        # The first record of each episode is a change by definition - the hold
        # is zeroed in `begin_episode` - and sits at step_idx 0, so it is phase 0
        # and needs no special case.
        changed = np.concatenate(([True], (a[1:] != a[:-1]) | (ep[1:] != ep[:-1])))
        phases |= set(np.unique(st[changed] % hold).tolist())
        changes += int(changed.sum())
    assert phases == {0}, (
        f"actions change at step_idx phases {sorted(phases)} mod {hold}, expected "
        f"{{0}} - the action is out of step with the truth in the same record"
    )
    print(f"alignment: all {changes:,} action changes sit at step_idx % {hold} == 0, "
          f"so action[t] is the action that produced qpos[t]")

    # The flip happened, exactly once.
    ep0 = index[0]
    raw0 = np.array(shards[ep0.shard].pixels[ep0.start])
    assert np.array_equal(sampler[0].frames[0], raw0[::-1]), "sampler rows are not the blob's reversed"
    assert not np.array_equal(sampler[0].frames[0], raw0), "flip is a no-op - a symmetric frame?"
    assert np.array_equal(sampler[7].frames, sampler[7].frames), "reads are not repeatable"
    print("orientation: sampler rows are the blob's reversed, and reversed only once")

    # D3's flag. Per-episode by construction, so it must be constant across every
    # frame of one - checked, because a per-frame bug here still produces a
    # plausible ~50/50 mix at the dataset level and would fool any share check.
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

    # `is_val` is a pure function of the episode id, so its distribution is
    # checkable with no dataset at all - which is what keeps the split covered
    # on the fixture, where two episodes cannot land 5% either side of 5%.
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

    # preload, on the val split. The train split is the same code over 17x the
    # frames, and materialising 1.16 GB (2.62 at 96x96) on every self-check run
    # would trade a real cost for no extra coverage - the size is arithmetic,
    # and runs.jsonl carries the measurement of the full build.
    from mirage.validator import load_palette  # deferred: validator imports this module

    palette = load_palette(Path(cfg.sim["scene_xml"]))
    val_idx, lut = preload(shards, index, "val", cfg.data["val_fraction"], palette.rgb)
    h = int(shards[0].sidecar["height"])
    train_frames = total - len(val_idx)
    print(f"preload: {len(val_idx):,} val frames as {val_idx.nbytes / 1e6:.1f} MB of "
          f"indices, {len(lut)}-entry LUT; the train split is {train_frames:,} frames "
          f"= {train_frames * h * h / 1e9:.2f} GB against {train_frames * h * h * 3 / 1e9:.2f} raw")

    # The whole claim: the LUT inverts, exactly, against an independent read of
    # the blob. Sampled rather than exhaustive because this is a byte identity -
    # a mapping error shows up on any frame that contains the mismatched colour.
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
