"""Are two shard sets the same bytes, and if not, where do they differ?

Built to answer one question: does a simulator build on another OS, compiler or
GPU reproduce the canonical dataset? A matching `data_hash` does not say so. It
covers the config and the scene, not the renderer, so two sets with the same
hash can hold different frames.

Compares shard by shard: SHA256 of every `.pixels` and `.meta` blob, and when a
blob differs, how. For pixels, how many frames differ, how many pixels in each,
and by how much. For meta, which record fields differ, and by how much for
`visible_px`. The sidecars are not compared: their `git_sha` differs whenever
the two runs were built from different commits, and every other field is
checked by `data.load_shards`.

Both directories must be complete shard sets with the same `data_hash`, which
`load_shards` enforces. Run from the repo root:

    python bench/shard_diff.py data/shards /path/to/other/data/shards
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import data  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def diff_pixels(a: np.memmap, b: np.memmap) -> dict:
    """Per-frame pixel differences, a chunk of frames at a time to bound memory."""
    frames = 0
    pixels = 0
    max_px_per_frame = 0
    max_abs = 0
    for start in range(0, len(a), 4096):
        d = np.abs(a[start:start + 4096].astype(np.int16) - b[start:start + 4096].astype(np.int16))
        per_pixel = d.max(axis=-1) > 0
        per_frame = per_pixel.reshape(len(d), -1).sum(axis=1)
        frames += int((per_frame > 0).sum())
        pixels += int(per_frame.sum())
        max_px_per_frame = max(max_px_per_frame, int(per_frame.max()))
        max_abs = max(max_abs, int(d.max()))
    return {"frames_differing": frames, "pixels_differing": pixels,
            "max_pixels_in_one_frame": max_px_per_frame, "max_channel_abs_diff": max_abs}


def diff_meta(a: np.memmap, b: np.memmap) -> dict:
    fields = {}
    records = np.zeros(len(a), dtype=bool)
    for name in a.dtype.names:
        ne = a[name] != b[name]
        records |= ne
        if ne.any():
            entry = {"records": int(ne.sum())}
            if name.startswith("visible_px"):
                gap = np.abs(a[name].astype(np.int32) - b[name].astype(np.int32))
                entry["max_abs_diff"] = int(gap.max())
            fields[name] = entry
    return {"records_differing": int(records.sum()), "fields": fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()

    shards_a = data.load_shards(args.a)
    shards_b = data.load_shards(args.b, data_hash=shards_a[0].sidecar["data_hash"])
    if [s.index for s in shards_a] != [s.index for s in shards_b]:
        raise SystemExit(f"shard indices differ: {[s.index for s in shards_a]} vs "
                         f"{[s.index for s in shards_b]}")

    identical = 0
    report = []
    for sa, sb in zip(shards_a, shards_b):
        if sa.frames != sb.frames:
            raise SystemExit(f"shard {sa.index}: {sa.frames} frames vs {sb.frames}")
        name = f"shard_{sa.index:03d}"
        row: dict = {"shard": sa.index, "frames": sa.frames}
        for ext, ma, mb, how in ((".pixels", sa.pixels, sb.pixels, diff_pixels),
                                 (".meta", sa.meta, sb.meta, diff_meta)):
            ha, hb = sha256(args.a / (name + ext)), sha256(args.b / (name + ext))
            if ha == hb:
                identical += 1
                row[ext[1:]] = {"sha256": ha, "identical": True}
            else:
                row[ext[1:]] = {"sha256_a": ha, "sha256_b": hb, "identical": False, **how(ma, mb)}
        report.append(row)

    print(json.dumps(report, indent=1))
    total = 2 * len(shards_a)
    frames = sum(r["pixels"].get("frames_differing", 0) for r in report)
    records = sum(r["meta"].get("records_differing", 0) for r in report)
    print(f"{identical} of {total} blobs identical; "
          f"{frames} frames with differing pixels, {records} meta records differing, "
          f"of {sum(s.frames for s in shards_a)} frames")


if __name__ == "__main__":
    main()
