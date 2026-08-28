"""Re-measure Phase 1's six pre-work numbers against the dataset on disk.

The six were measured on 2026-08-28 against `data_hash 0259947e` - the original
physics, `action_hold_steps 20` - and the scene has since been rescaled to
`gear 6 / damping 1.5`. The geometry, palette and camera never moved, so the
frames look the same; the arm's *pose distribution* moved, and patch statistics
are a measurement of exactly that. They also predate `runs.jsonl`, so none of
them carries a provenance row. This probe is what gives them one.

It reproduces, in one run:

  * the k-means floor at 240 / 512 / 1024 centroids, and how many stay live;
  * **the same floor on the held-out split**, which is the number gate row 2
    compares a tokenizer against - the whole-set floor above is not that
    number, and the split-aware section says why;
  * the flat / non-flat split of that error - the 96x96 fork's evidence;
  * the frequency-ranked exact-patch dictionary, as the cheaper alternative;
  * the share of interior cells whose 22x22 receptive field is one flat colour,
    and the Q-2 entropy ceiling that follows from it.

Two things the original run did not record, fixed here so a re-run is a
comparison rather than a coin flip: the **initialisation** (k-means++ under a
fixed seed, checked by running k=512 twice and demanding identical inertia) and
the **reconstruction dtype** (centroids stay float; the uint8-rounded PSNR is
printed beside it, since Q-1's bar is stated on uint8 frames).

Frame orientation is not corrected. The blob's rows are bottom-up, and a
vertical flip is a bijection on the patch set - it maps every patch to its own
flip - so every statistic here is invariant under it.

    python bench/patch_probe.py
"""

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
KMEANS_FRAMES = 2800  # 2800 * 64 patches = the 179,200 the original used
RF_FRAMES = 3500
RF = 22  # the encoder's receptive field at one 8x8 cell
KS = (240, 512, 1024)
DICT_KS = (512, 2048)
ITERS = 25
SEED = 0
PEAK = 255.0

cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
dev = "cuda" if torch.cuda.is_available() else "cpu"

print(f"data_hash {cfg.data_hash[:16]}, {len(shards)} shards, "
      f"{sum(s.frames for s in shards):,} frames, device {dev}")


def sample_frames(total):
    """`total` frames spread evenly within every shard, so no shard dominates.

    Whole-set on purpose: the six numbers this reproduces are statements about
    the dataset, not about a model, so a split would only shrink the sample.
    It does mean this sample **straddles the train/val split** - see
    `sample_split_frames` for the one number where that matters.
    """
    per = total // len(shards)
    out = [np.asarray(s.pixels[np.linspace(0, s.frames - 1, per, dtype=np.int64)])
           for s in shards]
    return np.concatenate(out)


def sample_split_frames(total, split):
    """`total` frames spread evenly across one side of the train/val split.

    The split comes from `data.is_val` over `data.episode_index`, and never from
    a fraction recomputed here. A probe that re-derives the split can disagree
    with the training run about which frames are held out, and that disagreement
    is silent - it reads as a floor that moved for no reason.

    Even over the split's frames rather than per episode, because the two sides
    hold 473 and 27 episodes: a fixed count per episode would give the val side
    four times the frames per episode and make the two samples different
    statistics. Uniform over the address list keeps them the same one.
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
    """Nearest centroid per row, chunked - the full n x k matrix does not fit."""
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
    """Lloyd from `init` seeding. Empty clusters are left empty on purpose:
    counting how many stay live IS the Q-2 risk measurement, so reseeding them
    would destroy the number this probe exists to report.

    Both seedings are run because the original measurement did not record which
    it used, and the choice turns out to be worth more than the regeneration
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

# The floor is the BEST patch-independent tokenizer, not an arbitrary one: a
# floor that is really an initialisation artifact understates how much work the
# conv context still has to do, which is the one thing Phase 1 is planned around.
kmeans_out = {k: runs["kmeans++", k] for k in KS}

# E-1's spirit: a floor nobody can reproduce is not a floor.
assert float(kmeans(x, 512, SEED)[2].sum()) == float(kmeans(x, 512, SEED)[2].sum()), \
    "k-means is not deterministic at a fixed seed"

# ------------------------------------------ the same floor, on the held-out set
#
# The floor above is fit and scored on `sample_frames`, which spreads evenly
# *within every shard*. `data.is_val` splits by **episode**, so that sample
# straddles the split, and a tokenizer's held-out PSNR is therefore not
# comparable to it: the tokenizer is fit on train and scored on val, while
# 29.02 dB was fit and scored on a mixture of both.
#
# Two numbers, because two different things could be inflating 29.02 dB and only
# measuring both separates them:
#
#   train-fit -> val-score   the honest floor. The same treatment a tokenizer
#                            gets, so this is what gate row 2 compares against.
#   val-fit   -> val-score   the original in-sample methodology, restricted to
#                            val. The gap to the row above is the in-sample
#                            advantage 29.02 dB was carrying.
#
# Both are reported on **uint8-rounded** centroids as well as float, because
# Q-1's bar is stated on uint8 frames and a tokenizer's PSNR is measured there.
# Rounding can only cost, so a float floor is the optimistic one.
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
# elsewhere. k-means++ is not globally optimal, so this is a sanity bound with
# slack and not an identity - a real inversion would mean the two samples are
# not drawn from the same distribution, which is a data bug, not a fit artifact.
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

# ------------------------------------------------ Q-2's ceiling, from the data
rf_frames = sample_frames(RF_FRAMES)
grid = 64 // PATCH
pad = (RF - PATCH) // 2
interior = [r for r in range(grid) if PATCH * r - pad >= 0 and PATCH * r - pad + RF <= 64]
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
# The collapse constrains interior cells only, but Q-2 scores the entropy over
# all 64 tokens of a frame, so the constrained mass dilutes by 36/64.
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
    # The held-out floor, and the bar that follows from it. These, not the
    # whole-set pair above, are what a tokenizer's val PSNR is compared to.
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
