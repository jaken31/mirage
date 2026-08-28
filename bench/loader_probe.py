"""Is the memmap loader anywhere near P-7's budget? Measure, do not assume.

P-7 is the only requirement that constrains `mirage/data.py`: a full 300k-frame
epoch in <= 30 min, which is 167 frames/sec end to end including the model. This
probe times the loader alone against that, in the two shapes that will actually
run - a sequential sweep (`validator.py`, and tokenizer training) and random
episode-aware windows (dynamics training).

Three passes over the random case on purpose. The dataset is 3.7 GB against
31.6 GB of RAM, so the OS page cache holds all of it after one pass and the
pass-1-to-pass-3 spread is the only warm-up cost there is. A small spread is the
evidence that no caching layer is worth writing.

The footgun this probe exists to avoid tripping over: slicing an `np.memmap`
returns a lazy view and touches no pages. A first version of this measured
3.4M fps because it had timed the slice arithmetic and nothing else.
`WindowSampler.__getitem__` returns `np.array(...)`, which forces the page-in,
so timing it is honest - but any hand-rolled variant here must do the same.

    python bench/loader_probe.py
"""

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mirage import config, data  # noqa: E402  - after the path insert, deliberately

EPOCH_BUDGET_S = 30 * 60  # P-7
SEQ_SHARE = 0.10  # the loader may spend at most this much of the epoch budget
CHUNK = 4096  # frames per sequential read, ~50 MB - amortised, still streaming

cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
shards = data.load_shards(ROOT / cfg.data["shard_dir"], data_hash=cfg.data_hash)
frames = sum(s.frames for s in shards)
bytes_per_frame = cfg.sim["height"] * cfg.sim["width"] * 3
print(f"{len(shards)} shards, {frames:,} frames, "
      f"{frames * bytes_per_frame / 2**30:.2f} GiB of pixels, ctx {cfg.data['ctx']}")

t0 = time.perf_counter()
index = data.episode_index(shards)
t_index = time.perf_counter() - t0
print(f"episode index      {t_index * 1000:7.0f} ms   {len(index)} episodes")

# Sequential sweep. `.sum()` rather than a bare slice: it is the cheapest thing
# that provably touches every byte.
t0 = time.perf_counter()
acc = 0
for shard in shards:
    for lo in range(0, shard.frames, CHUNK):
        acc += int(shard.pixels[lo:lo + CHUNK].sum(dtype=np.int64) & 1)
t_seq = time.perf_counter() - t0
print(f"sequential sweep   {t_seq:7.1f} s    {frames / t_seq:10,.0f} fps   "
      f"{frames * bytes_per_frame / t_seq / 2**20:6,.0f} MiB/s")

# Random episode-aware windows, through the real sampler.
sampler = data.WindowSampler(shards, index, cfg.data["ctx"])
rng = np.random.default_rng(0)
N = 3000
win_fps = 0.0
for p in range(1, 4):
    t0 = time.perf_counter()
    for _ in range(N):
        w = sampler.sample(rng)
        acc += int(w.frames[0, 0, 0, 0]) + int(w.meta["action"][0])
    dt = time.perf_counter() - t0
    win_fps = N * sampler.window / dt
    print(f"random windows p{p}  {N / dt:7,.0f} win/s {win_fps:10,.0f} fps   "
          f"{win_fps * bytes_per_frame / 2**20:6,.0f} MiB/s   "
          f"{dt / N * 1e6:5.0f} us/window")

floor_fps = frames / EPOCH_BUDGET_S
print(f"\nP-7 floor {floor_fps:.0f} fps ({frames:,} frames in {EPOCH_BUDGET_S // 60} min)"
      f"  |  sequential {frames / t_seq / floor_fps:,.0f}x  |  random {win_fps / floor_fps:,.0f}x")
print(f"loader share of the epoch budget: {t_seq / EPOCH_BUDGET_S:.2%} sequential, "
      f"{frames / win_fps / EPOCH_BUDGET_S:.2%} if every frame arrives via a random window")
print("checksum", acc & 0xFF)

assert t_seq < SEQ_SHARE * EPOCH_BUDGET_S, (
    f"sequential sweep is {t_seq:.0f} s, over {SEQ_SHARE:.0%} of P-7's {EPOCH_BUDGET_S} s budget"
)
assert win_fps > 10 * floor_fps, (
    f"random windows at {win_fps:,.0f} fps leave under 10x over P-7's {floor_fps:.0f} fps floor - "
    "the loader is a candidate bottleneck now, add DataLoader workers"
)
