"""Re-measure the six tokenizer planning numbers on the dataset now on disk.

The six were first measured on an earlier version of the dataset (the original
physics, `action_hold_steps 20`); the scene has since changed to
`gear 6 / damping 1.5`. Shapes, colours and camera did not change, so frames
look the same, but the arm's *spread of poses* did, and patch statistics
measure exactly that. The originals also came before `runs.jsonl` existed, so
none had a recorded source. This probe gives them one.

In one run it reproduces:

  * the k-means baseline at 240 / 512 / 1024 centroids, and how many get used;
  * **the same baseline on the held-out split**, which is what gate row 2
    compares a tokenizer against (the whole-set number is not; see the
    split-aware section);
  * how that error splits between flat and non-flat patches, the evidence for
    choosing 64x64 or 96x96;
  * a dictionary of the most frequent exact patches, as a cheaper alternative;
  * the share of interior cells whose 22x22 input window is one flat colour,
    and the token entropy ceiling that follows.

Two things the original run did not record are fixed here, so a re-run is a
real comparison: the **initialisation** (k-means++ with a fixed seed, checked
by running k=512 twice and requiring identical inertia), and the
**reconstruction type** (centroids stay float; the uint8-rounded PSNR is
printed next to it, since the 30 dB target is on uint8 frames).

Frame orientation is not corrected. Stored rows are bottom-up, but a vertical
flip just maps every patch to its own mirror image, so no statistic here
changes.

    python bench/patch_probe.py                                  # 64x64
    python bench/patch_probe.py --config mirage/configs/base96.json

`--config` makes 96x96 measurable: gate row 2 compares a rung with the k-means
baseline at *its own resolution*, and the 64x64 baseline is the wrong number
for a 96x96 rung. **The number of patches is held fixed across resolutions,
not the number of frames**: a 96x96 frame has 144 patches and a 64x64 frame
has 64, so the same 2,800 frames would give the 96x96 codebook 2.25x the data
and a baseline partly caused by sample size. `PATCH_BUDGET` below is fixed;
the frame count follows from it.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mirage import config, data  # noqa: E402

PATCH = 8
PATCH_BUDGET = 179_200  # 2800 * 64 patches, the count the original 64x64 run used
RF_FRAMES = 3500
RF = 22  # assumed input window of one 8x8 cell (the true conv window is 15; see mirage.fsq)
KS = (240, 512, 1024)
DICT_KS = (512, 2048)
ITERS = 25
SEED = 0
PEAK = 255.0

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "base.json"))
_args = _ap.parse_args()

cfg = config.load(_args.config)
shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
dev = "cuda" if torch.cuda.is_available() else "cpu"

_h, _w = cfg.shapes.image_size
PER_FRAME = (_h // PATCH) * (_w // PATCH)
KMEANS_FRAMES = PATCH_BUDGET // PER_FRAME

print(f"config {pathlib.Path(_args.config).name}, {_h}x{_w}, "
      f"{PER_FRAME} patches/frame, sampling {KMEANS_FRAMES:,} frames "
      f"for a {KMEANS_FRAMES * PER_FRAME:,}-patch budget")
print(f"data_hash {cfg.data_hash[:16]}, {len(shards)} shards, "
      f"{sum(s.frames for s in shards):,} frames, device {dev}")


def sample_frames(total):
    """`total` frames spread evenly within every shard, so no shard dominates.

    Uses the whole dataset on purpose: the six numbers describe the data, not a
    model, so splitting would only shrink the sample. It does mean the sample
    **mixes training and validation episodes**; see `sample_split_frames` for
    the one number where that matters.
    """
    per = total // len(shards)
    out = [np.asarray(s.pixels[np.linspace(0, s.frames - 1, per, dtype=np.int64)])
           for s in shards]
    return np.concatenate(out)


def sample_split_frames(total, split):
    """`total` frames spread evenly across one side of the train/validation split.

    The split comes from `data.is_val` over `data.episode_index`, never from a
    fraction recomputed here. A probe with its own split could silently
    disagree with training about which frames are held out, and that would look
    like a baseline that moved for no reason.

    Spread evenly over the split's frames rather than per episode, because the
    two sides have 473 and 27 episodes: a fixed count per episode would sample
    validation episodes far more densely, making the two samples measure
    different things.
    """
    want_val = split == "val"
    eps = [e for e in data.episode_index(shards)
           if data.is_val(e.episode_id, cfg.data["val_fraction"]) == want_val]
    sh = np.concatenate([np.full(e.length, e.shard, np.int64) for e in eps])
    fr = np.concatenate([e.start + np.arange(e.length, dtype=np.int64) for e in eps])
    pool = len(sh)
    take = np.linspace(0, pool - 1, total, dtype=np.int64)
    sh, fr = sh[take], fr[take]

    side = shards[0].pixels.shape[1:]
    out = np.empty((total, *side), dtype=np.uint8)
    for s in np.unique(sh):
        m = sh == s
        out[m] = shards[s].pixels[fr[m]]  # fr is increasing within a shard
    return out, len(eps), pool


def to_patches(frames):
    """(n, 64, 64, 3) uint8 -> (n * 64, 192) uint8, row-major within the patch."""
    n, h, w, c = frames.shape
    gh, gw = h // PATCH, w // PATCH
    return (frames.reshape(n, gh, PATCH, gw, PATCH, c)
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(n * gh * gw, PATCH * PATCH * c))


def assign(x, cen, chunk=16384):
    """Nearest centroid per row, in chunks, because the full n x k matrix does not fit in memory."""
    idx = torch.empty(len(x), dtype=torch.long, device=x.device)
    sse = torch.empty(len(x), dtype=torch.float64, device=x.device)
    cn = (cen * cen).sum(1)
    for i in range(0, len(x), chunk):
        b = x[i:i + chunk]
        d = (b * b).sum(1, keepdim=True) - 2.0 * (b @ cen.T) + cn
        best, arg = d.min(1)
        idx[i:i + chunk] = arg
        sse[i:i + chunk] = best.clamp_min(0).double()
    return idx, sse


def kmeans(x, k, seed, init="kmeans++"):
    """Standard k-means (Lloyd's algorithm) from `init` seeding. Empty clusters
    are left empty on purpose: how many stay in use is the codebook-usage
    measurement, so re-seeding them would destroy the number this reports.

    Both seedings are run because the original measurement did not record which
    it used, and the choice turns out to matter more than the dataset change
    this probe was written to check."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    cen = torch.empty(k, x.shape[1], device=x.device, dtype=x.dtype)
    if init == "random":
        cen[:] = x[torch.randperm(len(x), generator=g, device=x.device)[:k]]
    else:
        cen[0] = x[torch.randint(len(x), (1,), generator=g, device=x.device)]
        d2 = ((x - cen[0]) ** 2).sum(1).clamp_min(0)
        for j in range(1, k):
            pick = torch.multinomial(d2 / d2.sum(), 1, generator=g)
            cen[j] = x[pick]
            d2 = torch.minimum(d2, ((x - cen[j]) ** 2).sum(1).clamp_min(0))

    for _ in range(ITERS):
        idx, _ = assign(x, cen)
        tot = torch.zeros_like(cen).index_add_(0, idx, x)
        cnt = torch.zeros(k, device=x.device, dtype=x.dtype).index_add_(
            0, idx, torch.ones(len(x), device=x.device, dtype=x.dtype))
        live = cnt > 0
        cen[live] = tot[live] / cnt[live, None]

    idx, sse = assign(x, cen)
    return cen, idx, sse


def psnr(total_sse, n_values):
    return 10.0 * np.log10(PEAK * PEAK / (total_sse / n_values))


# ---------------------------------------------------------------- the patches
t0 = time.perf_counter()
frames = sample_frames(KMEANS_FRAMES)
patches_u8 = to_patches(frames)
n, dim = patches_u8.shape
x = torch.from_numpy(patches_u8).to(dev, torch.float32)
print(f"{n:,} patches of {dim} values from {len(frames):,} frames "
      f"({time.perf_counter() - t0:.1f} s)")

flat = torch.from_numpy(
    (patches_u8 == np.tile(patches_u8[:, :3], PATCH * PATCH)).all(1)
).to(dev)
flat_share = float(flat.float().mean())

# ------------------------------------------------------------------- k-means
print(f"\nLloyd's k-means, {ITERS} iterations, seed {SEED}")
print(f"{'init':>9} {'k':>6} {'PSNR dB':>9} {'uint8 dB':>9} {'live':>10} {'err non-flat':>14}")
runs = {}
for init in ("kmeans++", "random"):
    for k in KS:
        t = time.perf_counter()
        cen, idx, sse = kmeans(x, k, SEED, init)
        db = psnr(float(sse.sum()), n * dim)
        rounded = float(((x - cen[idx].round().clamp(0, 255)) ** 2).sum(dtype=torch.float64))
        db_u8 = psnr(rounded, n * dim)
        live = int(torch.bincount(idx, minlength=k).gt(0).sum())
        edge_share = float(sse[~flat].sum() / sse.sum())
        runs[init, k] = dict(psnr_db=round(db, 2), psnr_uint8_db=round(db_u8, 2),
                             live=live, edge_error_share=round(edge_share, 4))
        print(f"{init:>9} {k:>6} {db:>9.2f} {db_u8:>9.2f} {live:>6}/{k:<3} "
              f"{edge_share:>13.2%}  ({time.perf_counter() - t:.1f} s)")

# The baseline must be the BEST patch-by-patch codebook, not an arbitrary one:
# a baseline weakened by bad initialisation would understate how much the
# network's wider context still has to add, which the tokenizer plan depends on.
kmeans_out = {k: runs["kmeans++", k] for k in KS}

# A baseline nobody can reproduce is not a baseline.
assert float(kmeans(x, 512, SEED)[2].sum()) == float(kmeans(x, 512, SEED)[2].sum()), \
    "k-means is not deterministic at a fixed seed"

# ------------------------------------------ the same baseline, on held-out data
#
# The baseline above is fit and scored on `sample_frames`, spread evenly
# *within every shard*. `data.is_val` splits by **episode**, so that sample
# mixes both sides, and a tokenizer's held-out PSNR cannot be compared with it:
# the tokenizer is trained on one side and scored on the other, while the
# original baseline was fit and scored on a mix of both.
#
# Two numbers, because two things could be inflating the original, and only
# measuring both tells them apart:
#
#   train-fit -> val-score   the fair baseline: the same treatment a tokenizer
#                            gets, so this is what gate row 2 compares against.
#   val-fit   -> val-score   the original fit-and-score-on-the-same-data method,
#                            on validation only. The gap to the row above is
#                            the advantage that method was giving.
#
# Both are reported on **uint8-rounded** centroids as well as float, because
# the 30 dB target is on uint8 frames. Rounding can only hurt, so the float
# number is the optimistic one.
train_frames, train_eps, train_pool = sample_split_frames(KMEANS_FRAMES, "train")
val_frames, val_eps, val_pool = sample_split_frames(KMEANS_FRAMES, "val")
train_u8, val_u8 = to_patches(train_frames), to_patches(val_frames)
xt = torch.from_numpy(train_u8).to(dev, torch.float32)
xv = torch.from_numpy(val_u8).to(dev, torch.float32)
nv = xv.shape[0] * dim

print(f"\nheld-out floor: {train_eps} train / {val_eps} val episodes "
      f"({train_pool:,} / {val_pool:,} frames), sampling {KMEANS_FRAMES:,} from each, "
      f"{nv // dim:,} val patches")
print(f"{'fit on':>9} {'k':>6} {'val dB':>9} {'uint8 dB':>9} {'live':>10} {'err non-flat':>14}")

val_flat = torch.from_numpy(
    (val_u8 == np.tile(val_u8[:, :3], PATCH * PATCH)).all(1)
).to(dev)
held = {}
for fit_on, xf in (("train", xt), ("val", xv)):
    for k in KS:
        t = time.perf_counter()
        cen, _, _ = kmeans(xf, k, SEED)
        idx, sse = assign(xv, cen)
        db = psnr(float(sse.sum()), nv)
        rounded = float(((xv - cen[idx].round().clamp(0, 255)) ** 2).sum(dtype=torch.float64))
        db_u8 = psnr(rounded, nv)
        live = int(torch.bincount(idx, minlength=k).gt(0).sum())
        edge_share = float(sse[~val_flat].sum() / sse.sum())
        held[fit_on, k] = dict(val_psnr_db=round(db, 2), val_psnr_uint8_db=round(db_u8, 2),
                               live_on_val=live, edge_error_share=round(edge_share, 4))
        print(f"{fit_on:>9} {k:>6} {db:>9.2f} {db_u8:>9.2f} {live:>6}/{k:<3} "
              f"{edge_share:>13.2%}  ({time.perf_counter() - t:.1f} s)")

# A codebook fit on the very patches it scores should not lose to one fit
# elsewhere. k-means++ is not perfect, so this is a loose sanity check, not an
# exact rule; a real reversal would mean the two samples come from different
# distributions, which is a data bug, not a fitting quirk.
for k in KS:
    slack = held["val", k]["val_psnr_db"] - held["train", k]["val_psnr_db"]
    assert slack > -0.15, (
        f"k={k}: fitting on val scored {slack:+.2f} dB against fitting on train, "
        f"so the two splits are not the same distribution"
    )
assert held["train", 1024]["val_psnr_db"] > held["train", 512]["val_psnr_db"] \
    > held["train", 240]["val_psnr_db"], \
    "more centroids scored worse on val - the k-means run is broken, not the data"

# ------------------------------------------- the frequency-ranked alternative
uniq, cnt = np.unique(patches_u8, axis=0, return_counts=True)
p = cnt / cnt.sum()
exact_bits = float(-(p * np.log2(p)).sum())
order = np.argsort(-cnt)
print(f"\n{len(uniq):,} distinct exact patches, entropy {exact_bits:.2f} bits "
      f"against the {np.log2(512):.0f} available at 512 codes; "
      f"{flat_share:.2%} of patches are one flat colour")
dict_out = {}
for k in DICT_KS:
    cen = torch.from_numpy(uniq[order[:k]]).to(dev, torch.float32)
    _, sse = assign(x, cen)
    db = psnr(float(sse.sum()), n * dim)
    dict_out[k] = round(db, 2)
    print(f"  top-{k:<5} exact patches as a dictionary: {db:.2f} dB")

# ------------------------------------ the token entropy ceiling, from the data
rf_frames = sample_frames(RF_FRAMES)
# Image size comes from the frames, not a hardcoded 64. Hardcoding is right at
# 64x64 but silently wrong at 96x96: it would scan an 8x8 grid's interior out
# of a 12x12 one and report a ceiling for the wrong frame size.
size = rf_frames.shape[1]
assert rf_frames.shape[1] == rf_frames.shape[2], \
    f"frames are {rf_frames.shape[1]}x{rf_frames.shape[2]} - this sweep assumes square"
grid = size // PATCH
pad = (RF - PATCH) // 2
interior = [r for r in range(grid) if PATCH * r - pad >= 0 and PATCH * r - pad + RF <= size]
flat_cells = void_cells = total_cells = 0
for r in interior:
    for c in interior:
        r0, c0 = PATCH * r - pad, PATCH * c - pad
        f = rf_frames[:, r0:r0 + RF, c0:c0 + RF, :].reshape(len(rf_frames), -1, 3)
        is_flat = (f.max(1) == f.min(1)).all(1)
        flat_cells += int(is_flat.sum())
        void_cells += int((is_flat & (f[:, 0, :] == 0).all(1)).sum())
        total_cells += len(rf_frames)

flat_rf = flat_cells / total_cells
# Only interior cells are affected, but token entropy is measured over every
# token in a frame, so the effect is diluted by the interior's share of the
# grid: 36 of 64 cells at 64x64, a different share at 96x96.
collapsed = flat_rf * len(interior) ** 2 / grid ** 2
print(f"\n{len(interior) ** 2} interior cells of {grid ** 2}, {RF}x{RF} receptive "
      f"field, {len(rf_frames):,} frames")
print(f"  flat receptive fields  {flat_rf:.2%} of interior cells, {void_cells} of them void")
print(f"  collapsed token mass   {collapsed:.2%} over all {grid ** 2} cells")

ceiling = {}
for k in (512, 240):
    h = -collapsed * np.log2(collapsed) - (1 - collapsed) * np.log2((1 - collapsed) / (k - 1))
    ceiling[k] = round(float(h / np.log2(k)), 4)
    print(f"  ceiling at {k:>4} codes   {h:.3f} bits = {h / np.log2(k):.1%} of uniform")

assert void_cells == 0, "a void receptive field collapses - the Q-2 ceiling argument changes"
assert ceiling[512] > 0.70, "the data itself forbids Q-2, which no training run can fix"
assert kmeans_out[1024]["psnr_db"] > kmeans_out[512]["psnr_db"] > kmeans_out[240]["psnr_db"], \
    "more centroids scored worse - the k-means run is broken, not the data"

print("\n--- the six, as JSON for runs.jsonl ---")
print(json.dumps(dict(
    kmeans_floor_512_db=kmeans_out[512]["psnr_db"],
    kmeans_1024_db=kmeans_out[1024]["psnr_db"],
    kmeans_240_db=kmeans_out[240]["psnr_db"],
    live_of_512=kmeans_out[512]["live"],
    flat_receptive_fields=round(flat_rf, 4),
    q2_ceiling_512=ceiling[512],
    q2_ceiling_240=ceiling[240],
    edge_error_share_512=kmeans_out[512]["edge_error_share"],
    non_flat_patch_share=round(1 - flat_share, 4),
    exact_patch_bits=round(exact_bits, 2),
    dict_512_db=dict_out[512],
    dict_2048_db=dict_out[2048],
    gate_row2_bar_db=round(30.0 - kmeans_out[512]["psnr_db"], 2),
    # The held-out baseline and the margin a tokenizer needs over it. These,
    # not the whole-set pair above, are what a tokenizer's val PSNR is compared to.
    heldout_floor_512_db=held["train", 512]["val_psnr_db"],
    heldout_floor_512_uint8_db=held["train", 512]["val_psnr_uint8_db"],
    heldout_floor_1024_db=held["train", 1024]["val_psnr_db"],
    heldout_floor_240_db=held["train", 240]["val_psnr_db"],
    heldout_live_of_512=held["train", 512]["live_on_val"],
    heldout_edge_error_share_512=held["train", 512]["edge_error_share"],
    val_insample_512_db=held["val", 512]["val_psnr_db"],
    val_insample_512_uint8_db=held["val", 512]["val_psnr_uint8_db"],
    insample_advantage_512_db=round(held["val", 512]["val_psnr_db"]
                                    - held["train", 512]["val_psnr_db"], 2),
    wholeset_minus_heldout_512_db=round(kmeans_out[512]["psnr_db"]
                                        - held["train", 512]["val_psnr_db"], 2),
    gate_row2_bar_heldout_db=round(30.0 - held["train", 512]["val_psnr_db"], 2),
    train_episodes=train_eps,
    val_episodes=val_eps,
    val_pool_frames=val_pool,
    random_init_512_db=runs["random", 512]["psnr_db"],
    random_init_512_live=runs["random", 512]["live"],
    random_init_1024_db=runs["random", 1024]["psnr_db"],
    random_init_240_db=runs["random", 240]["psnr_db"],
), indent=1))
print("\npatch probe ok")
