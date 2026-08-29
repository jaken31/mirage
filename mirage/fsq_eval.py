"""Everything downstream of a trained checkpoint: the token cache and the gate.

Split out of `fsq.py` when that file passed 500 lines, which is the trigger
`docs/phase1_structural_plan.md` section 5 names and the only reason this file
exists. The division is by *when the code runs*: `fsq.py` builds and trains, this
runs against an artifact that already exists.

- **5e** `write_token_cache` - one uint16 `.npy` per shard, plus a manifest.
- `load_run` - a checkpoint back into an eval-mode `Tokenizer`.
- `evaluate` - the eight-row gate table.

Reached through `fsq.py`'s CLI rather than its own, because `AGENDA.md` documents
`python -m mirage.fsq --eval` and a second entry point would be a second thing to
keep true:

    python -m mirage.fsq --tokens RUN_ID
    python -m mirage.fsq --eval RUN_ID
"""

import hashlib
import json
import math
import time

import numpy as np
import torch
from torch import nn

from mirage import config, data, validator
from mirage.fsq import (PEAK, PSNR_BAR_DB, ROOT, Tokenizer, kmeans_floor_db,
                        _batch, psnr_db, reconstruction_psnr)


# ------------------------------------------------------------ 5e. token cache

def load_run(run_id: str, cfg: config.Config,
             device: torch.device | None = None) -> tuple[Tokenizer, dict]:
    """`runs/<run_id>/model.pt` -> an eval-mode `Tokenizer`, plus its knobs.

    Rebuilds the architecture from the knobs the checkpoint carries rather than
    from arguments, so a caller cannot quietly load R1's weights into R2's shape.
    Refuses a checkpoint whose `data_hash` disagrees with `cfg`: the tokenizer is
    only meaningful over the frames it was trained on, and the two hashes drifting
    apart is exactly how a stale checkpoint gets used on regenerated data.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=True: the checkpoint holds tensors, two hash strings and a
    # knobs dict of primitives, so nothing here needs the arbitrary-unpickling
    # default, and a checkpoint is exactly the kind of file that gets copied
    # between machines.
    ckpt = torch.load(ROOT / "runs" / run_id / "model.pt", map_location=dev,
                      weights_only=True)
    knobs = ckpt["knobs"]
    if ckpt["data_hash"] != cfg.data_hash:
        raise ValueError(
            f"{run_id} was trained on data_hash {ckpt['data_hash'][:8]}, "
            f"this config is {cfg.data_hash[:8]}"
        )
    model = Tokenizer(tuple(knobs["levels"]), attention=knobs["attention"],
                      quantize=knobs["quantize"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, knobs


@torch.no_grad()
def write_token_cache(run_id: str, cfg: config.Config, batch: int = 256,
                      device: str | None = None) -> dict:
    """Encode every frame once: one uint16 `.npy` per shard under the run's dir.

    Per shard and not one flat array, because a flat array needs a cumulative
    frame offset to address and that is an off-by-one factory. Per-shard makes
    `len(tokens) == shard.frames` a loud assert, which is gate row 4.

    Named by run id and not by `tokenizer_hash`, because two runs at identical
    config and different seeds share a hash and produce different tokens. The
    checkpoint carries the hash inside it for provenance, and the manifest
    repeats it.

    **The rows are flipped on the way in.** `Shard.pixels` holds them bottom-up
    as the GL readback wrote them, and `data.preload` flips them for training. A
    cache written without the flip would produce well-formed tokens for
    upside-down frames and fail nowhere until Phase 2 trained on them.

    The pass is over 3.5 GB of memmap and is mandatory, so two gate rows ride
    along rather than paying for their own pass: a 512-bin histogram gives row 3
    (token entropy) and row 8 (live codes), and a sha256 per shard gives row 5 -
    re-running this writer and diffing manifests *is* the bit-identical check.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, knobs = load_run(run_id, cfg, dev)
    codebook = math.prod(knobs["levels"])
    assert codebook <= np.iinfo(np.uint16).max + 1, \
        f"{codebook} codes will not fit in the uint16 the Phase 2 handoff asks for"

    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    out_dir = ROOT / "runs" / run_id / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = np.zeros(codebook, dtype=np.int64)
    per_shard = []
    t0 = time.perf_counter()

    for sh in shards:
        toks = np.empty((sh.frames, *cfg.shapes.token_grid), dtype=np.uint16)
        for i in range(0, sh.frames, batch):
            j = min(i + batch, sh.frames)
            px = np.ascontiguousarray(sh.pixels[i:j, ::-1])  # rows bottom-up -> top-down
            x = torch.from_numpy(px).to(dev).permute(0, 3, 1, 2).float() / PEAK
            toks[i:j] = model.encode(x).cpu().numpy().astype(np.uint16)
        assert len(toks) == sh.frames, \
            f"shard {sh.index}: wrote {len(toks)} token rows for {sh.frames} frames"
        counts += np.bincount(toks.ravel(), minlength=codebook)
        raw = toks.tobytes()
        path = out_dir / f"shard_{sh.index:03d}.npy"
        np.save(path, toks)
        per_shard.append({"shard": sh.index, "frames": sh.frames, "file": path.name,
                          "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        print(f"  shard {sh.index}: {sh.frames:,} frames -> {path.name} "
              f"({len(raw) / 1e6:.1f} MB)")

    total = int(counts.sum())
    p = counts / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    manifest = {
        "run_id": run_id,
        "checkpoint": f"runs/{run_id}/model.pt",
        "tokenizer_hash": cfg.tokenizer_hash,
        "data_hash": cfg.data_hash,
        "levels": knobs["levels"],
        "codebook_size": codebook,
        "token_grid": list(cfg.shapes.token_grid),
        "dtype": "uint16",
        # Recorded because the tokens depend on it. With attention in the
        # encoder, `F.scaled_dot_product_attention` returns batch-size-dependent
        # floating point, and ~2 latent values in 100,000 sit close enough to a
        # quantization boundary to flip: shard 0 of R2 encoded at batch 128 and
        # at 256 differs in 10 of 512,000 tokens, where R1 with attention off
        # differs in 0. A re-encode is therefore only bit-identical at the same
        # batch, so E-1 needs this pinned rather than assumed.
        "batch": batch,
        "frames": sum(s["frames"] for s in per_shard),
        "tokens": total,
        "shards": per_shard,
        # Rows 3 and 8, computed on the pass that had to happen anyway.
        "entropy_bits": entropy,
        "entropy_ratio": entropy / math.log2(codebook),
        "live_codes": int((p > 1e-4).sum()),
        "counts": counts.tolist(),
        "encode_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                           encoding="utf-8", newline="\n")
    print(f"{manifest['frames']:,} frames -> {total:,} tokens, entropy {entropy:.3f} of "
          f"{math.log2(codebook):.0f} bits ({manifest['entropy_ratio']:.1%}), "
          f"{manifest['live_codes']}/{codebook} live, {manifest['encode_s']}s")
    return manifest


# ------------------------------------------------------------- the gate table

def _flat_mask(idx_batch: np.ndarray, patch: int) -> np.ndarray:
    """(B, H, W) palette indices -> (B, H, W) bool, True where the pixel's
    `patch`x`patch` block is a single flat colour.

    Taken from the **ground truth**, never from the reconstruction. Deriving
    flatness from the model's own output would let a blurry decoder reclassify
    its mistakes as edges and flatter the flat-pixel number, so the metric would
    be measuring the model twice.
    """
    b, h, w = idx_batch.shape
    p = idx_batch.reshape(b, h // patch, patch, w // patch, patch)
    flat = (p == p[:, :, :1, :, :1]).all(2).all(-1)  # (b, gh, gw)
    return np.repeat(np.repeat(flat, patch, 1), patch, 2)


@torch.no_grad()
def edge_flat_psnr(model: nn.Module, idx: np.ndarray, lut: torch.Tensor, patch: int,
                   batch: int = 256) -> tuple[float, float, float]:
    """(flat-pixel dB, edge-pixel dB, edge share of squared error) over `idx`.

    Gate row 7, and the 64-vs-144 fork's diagnostic. The k-means floor puts
    **99.95%** of its error in the 36.53% of patches that are not flat; if a
    trained tokenizer does the same, more pixels is the lever and 96x96 is
    indicated.
    """
    was_training = model.training
    model.eval()
    sse = {True: 0.0, False: 0.0}
    values = {True: 0, False: 0}
    for i in range(0, len(idx), batch):
        rows = np.arange(i, min(i + batch, len(idx)))
        x8 = _batch(idx, lut, rows)
        y8 = ((model(x8 / PEAK)) * PEAK).round().clamp(0, PEAK)
        err = (y8 - x8).pow(2).sum(1)  # (b, h, w), summed over the 3 channels
        flat = torch.from_numpy(_flat_mask(np.ascontiguousarray(idx[rows]), patch)).to(err.device)
        for is_flat in (True, False):
            m = flat if is_flat else ~flat
            sse[is_flat] += float(err[m].sum())
            values[is_flat] += int(m.sum()) * 3
    model.train(was_training)
    total = sse[True] + sse[False]
    return (psnr_db(sse[True], values[True]), psnr_db(sse[False], values[False]),
            sse[False] / total if total else 0.0)


def entropy_split(counts: list[int], levels: list[int]) -> dict:
    """Row 3, taken apart: where the missing bits are.

    A token id is the mixed-radix number `d0 + levels[0]*d1 + ...`, so the
    per-channel digit distributions fall out of the same counts vector row 3
    already sums - no GPU, no re-encode. The joint entropy is then
    `sum(marginals) - redundancy`, and the two terms fail for different reasons
    and have different fixes:

    - **marginal skew** - one channel's latent sits off centre in the `tanh`
      bound and never reaches most of its levels. R2's channel 2 puts 81% of its
      mass on digits 0 and 1 and returns 1.964 of 3 bits.
    - **redundancy** - the channels encode copies of each other. This is what
      `GridAttention` fixes: R1 -> R2 it falls 1.339 -> 0.781 bits, which is 76%
      of attention's entire entropy gain and the mechanism behind a result no
      document predicted.

    Split them because the plan's remedy for a Q-2 miss - the shrink ladder - is
    a *collapse* fix, and neither term is collapse: zero of 512 codes have zero
    count in either rung.
    """
    c = np.asarray(counts, dtype=float)
    p = c / c.sum()

    def h(pr: np.ndarray) -> float:
        nz = pr[pr > 0]
        return float(-(nz * np.log2(nz)).sum())

    ids = np.arange(len(c))
    place = 1
    per = []
    for n in levels:
        per.append(h(np.bincount((ids // place) % n, weights=p, minlength=n)))
        place *= n
    joint = h(p)
    return {"joint_bits": joint, "channel_bits": per, "marginal_sum_bits": sum(per),
            "redundancy_bits": sum(per) - joint, "zero_count_codes": int((c == 0).sum())}


# ------------------------------------- row 6. F-9 against reconstructions

@torch.no_grad()
def reconstruct(model: nn.Module, idx: np.ndarray, lut: torch.Tensor,
                rows: np.ndarray, batch: int = 256) -> np.ndarray:
    """`rows` of `idx`, round-tripped through the tokenizer, as the validator
    wants them: `(n, h, w, 3)` uint8, channel axis last.

    Rounded and clamped before it leaves, because uint8 is what the pipeline
    actually delivers and `measure_pixels_only` refuses anything else. Measuring
    the float output would calibrate a validator that never runs - and it would
    calibrate it optimistically, since rounding is itself a source of colours
    that sit off the palette.
    """
    out = []
    for i in range(0, len(rows), batch):
        x8 = _batch(idx, lut, rows[i:i + batch])
        y8 = (model(x8 / PEAK) * PEAK).round().clamp(0, PEAK)
        out.append(y8.byte().permute(0, 2, 3, 1).cpu().numpy())
    return np.concatenate(out)


def reconstruction_sweep(model: nn.Module, cfg: config.Config, shards, index,
                         palette: validator.Palette, val_idx: np.ndarray,
                         lut: torch.Tensor, tau: float, sample: int | None = None,
                         seed: int = 0, batch: int = 256
                         ) -> tuple[validator.Sweep, validator.Sweep]:
    """(reconstruction sweep, ground-truth sweep) over the same val rows.

    Gate row 6, and the measurement build order item 6 calibrates against. F-9
    was accepted at zero false positives on *renders*; every threshold it fixed
    therefore describes a frame with exactly seven colours in it. What Q-3
    actually counts is frames out of the decoder, which softens every edge into
    colours no palette entry names, so the thresholds have to be re-derived
    against those or the coherence horizon reads zero on frame one forever.

    Both sweeps run on the same rows and the same truth. The ground-truth half
    is not decoration:

      * it is the **alignment check**. `val_idx` and the meta come from two
        different functions, and a misalignment would silently pair frame i's
        pixels with frame j's truth - both arrays would still have the right
        length and dtype, and every number below would be wrong and plausible.
        Ground truth run through the same path has to reproduce F-9's known
        result; when it does, the rows line up.
      * it is the **baseline**. A reconstruction number means nothing on its
        own - the question item 6 asks is how much worse than a render the
        decoder is, and that is a difference, not a value.

    Held-out rows on purpose: the val split, the same frames row 1's PSNR is
    quoted over. A threshold calibrated on frames the tokenizer trained on
    would be tuned to reconstructions that are better than any it will meet.
    """
    metas = data.split_meta(shards, index, "val", cfg.data["val_fraction"])
    assert len(metas) == len(val_idx), (
        f"{len(val_idx)} val frames against {len(metas)} val meta records - "
        f"preload and split_meta disagree about the split"
    )

    # The whole val split by default, and not a sample, because the threshold
    # this feeds is a **maximum**. A max over 2,000 frames is systematically
    # smaller than one over 16,200, so a subsampled gate is a strictly easier
    # gate than the calibration that set the number - it would pass here and
    # fail on the full split, which is the worst way for a threshold to be wrong.
    if sample is None:
        rows = np.arange(len(val_idx))
    else:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(val_idx), size=min(sample, len(val_idx)),
                                  replace=False))

    lut_np = lut.round().clamp(0, PEAK).byte().cpu().numpy()
    truth = lut_np[val_idx[rows]]
    recon = reconstruct(model, val_idx, lut, rows, batch=batch)
    assert recon.shape == truth.shape, f"{recon.shape} decoded against {truth.shape} truth"

    r = validator.sweep(recon, metas[rows], palette, tau)
    g = validator.sweep(truth, metas[rows], palette, tau)

    # The alignment check, and the only one that can catch a silent row shift.
    # These two are F-9's *known* ground-truth results - zero off-palette pixels,
    # and a worst distance of 0.75 that is render rounding and nothing else. Truth
    # pushed through this exact path has to reproduce them. If it does not, the
    # frames and the meta are not the same rows, and every reconstruction number
    # above is attributing one frame's pixels to another frame's truth.
    assert g.offpalette_px_max == 0, (
        f"{g.offpalette_px_max} off-palette px on ground truth at tau {tau} - "
        f"F-9 says this is 0, so the frames and meta are misaligned or the palette moved"
    )
    assert g.max_palette_dist < 1.0, (
        f"ground truth sits {g.max_palette_dist:.2f} from the palette, expected 0.75"
    )
    return r, g


def evaluate(run_id: str, cfg: config.Config, device: str | None = None) -> dict:
    """The eight-row gate table for one run. Rows 1-5 are pass/fail here.

    Row 1 is **recomputed from `model.pt`**, not read from `result.json`. The
    gate exists to check the artifact that would ship; reading the training log
    would pass a checkpoint that failed to save correctly.

    Row 2 charges against the recorded floor for *this config's resolution*
    (`fsq.kmeans_floor_db`) rather than refitting
    k-means. The plan asked for a refit so rows 1 and 2 could disagree, but the
    val split is fixed by `data.is_val` over a checked `data_hash`, so a refit at
    seed 0 returns 28.27 dB every time - it is a constant dressed as a
    measurement. What the refit was protecting against is the data moving
    underneath the floor, and `load_run` already refuses a `data_hash` mismatch
    loudly. `bench/patch_probe.py` remains the one place k-means lives.

    Row 6 runs the F-9 sweep against decoder output, using the thresholds build
    order item 6 calibrated into `validator.offpalette_tau`, and the
    `validator.offpalette_frac_max` share that restated item 6's pixel count in
    a resolution-free form. It is a **regression check, not a
    calibration**: the numbers were fixed once, against the R2 rung on the whole
    held-out split, and re-deriving them per run would move `validator_hash` and
    make two runs' Q-3 horizons incomparable. A later rung that cannot meet them
    is telling you its decoder is worse, which is the point.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, knobs = load_run(run_id, cfg, dev)
    run_dir = ROOT / "runs" / run_id
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))

    man_path = run_dir / "tokens" / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(
            f"no token cache for {run_id} - run `python -m mirage.fsq --tokens {run_id}` first"
        )
    man = json.loads(man_path.read_text(encoding="utf-8"))

    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    val_idx, lut_np = data.preload(shards, index, "val", cfg.data["val_fraction"], palette.rgb)
    lut = torch.from_numpy(lut_np).to(dev).float()
    patch = cfg.shapes.image_size[0] // cfg.shapes.token_grid[0]

    db, _ = reconstruction_psnr(model, val_idx, lut)
    flat_db, edge_db, edge_share = edge_flat_psnr(model, val_idx, lut, patch)

    tau = cfg.validator["offpalette_tau"]
    frac_max = cfg.validator["offpalette_frac_max"]
    r6, g6 = reconstruction_sweep(model, cfg, shards, index, palette, val_idx, lut, tau)

    # Row 5: re-encode shard 0 and compare its sha256 to the manifest. One shard
    # is the same property as seven and costs a seventh of the time.
    #
    # **At the manifest's batch, not a hardcoded one.** This originally re-encoded
    # at 256 while the cache had been written at `--batch` (default 128), so the
    # row was testing "re-encode at a different batch size" and R2 failed it while
    # R1 passed. That is a real property - see the `batch` key above - but it is
    # not the property E-1 asks about, and conflating the two would have retired
    # a working determinism check on a false alarm.
    enc_batch = man.get("batch", 256)
    sh = shards[0]
    again = np.empty((sh.frames, *cfg.shapes.token_grid), dtype=np.uint16)
    for i in range(0, sh.frames, enc_batch):
        j = min(i + enc_batch, sh.frames)
        px = np.ascontiguousarray(sh.pixels[i:j, ::-1])
        x = torch.from_numpy(px).to(dev).permute(0, 3, 1, 2).float() / PEAK
        again[i:j] = model.encode(x).cpu().numpy().astype(np.uint16)
    redo = hashlib.sha256(again.tobytes()).hexdigest()

    counts_ok = all(s["frames"] == sh_.frames for s, sh_ in zip(man["shards"], shards))
    floor = kmeans_floor_db(cfg)  # keyed by resolution - see fsq.KMEANS_FLOOR_DB
    rows = [
        (1, "Held-out PSNR, uint8, over the val frames", f"{db:.3f} dB",
         f">= {PSNR_BAR_DB}", db >= PSNR_BAR_DB),
        (2, f"That minus the {floor} dB held-out k-means floor",
         f"{db - floor:+.3f} dB",
         f">= {PSNR_BAR_DB - floor:+.2f}", db - floor >= PSNR_BAR_DB - floor),
        (3, "Token entropy / log2(codebook), all 300,000 frames",
         f"{man['entropy_ratio']:.1%} ({man['entropy_bits']:.3f} bits)",
         ">= 70%", man["entropy_ratio"] >= 0.70),
        (4, "Token cache rows == shard.frames, every shard",
         f"{len(man['shards'])} shards, {man['frames']:,} frames", "exact", counts_ok),
        (5, "Re-encode shard 0 from the checkpoint twice",
         "identical" if redo == man["shards"][0]["sha256"] else "DIFFERS",
         "bit-identical", redo == man["shards"][0]["sha256"]),
        (6, f"F-9 off-palette share of the worst frame, tau {tau}",
         f"{r6.offpalette_frac_max:.4%} "
         f"({r6.offpalette_px_max} px, worst dist {r6.max_palette_dist:.1f})",
         f"<= {frac_max:.4%}", r6.offpalette_frac_max <= frac_max),
        (7, "Edge-pixel PSNR vs flat-pixel PSNR",
         f"edge {edge_db:.3f} dB, flat {flat_db:.3f} dB, {edge_share:.2%} of error at edges",
         "reported", None),
        (8, "Train-val PSNR gap; live codes at mass > 1e-4",
         f"gap {result['gap_db']:+.3f} dB; {man['live_codes']}/{man['codebook_size']} live",
         "reported", None),
    ]

    print(f"\ngate table - {run_id}, levels {knobs['levels']}, "
          f"attention={knobs['attention']}, data_hash {cfg.data_hash[:8]}")
    print(f"{'#':>2}  {'measure':<52} {'value':<44} {'bar':<16} verdict")
    for n, name, value, bar, ok in rows:
        verdict = "-" if ok is None else ("PASS" if ok else "FAIL")
        print(f"{n:>2}  {name:<52} {value:<44} {bar:<16} {verdict}")

    # Row 6 next to the renders it was originally calibrated on. Printed rather
    # than left in the return value because the gap *is* the result: F-9's
    # thresholds were set where the right-hand column sits, and item 6 exists
    # because the left-hand column is where the validator actually has to work.
    print()
    print(f"    row 6 against ground truth on the same {r6.frames:,} rows: "
          f"{g6.offpalette_px_max} off-palette px, worst dist {g6.max_palette_dist:.2f}, "
          f"{g6.n_unique_max} unique colours")
    print(f"    the decoder costs {r6.max_palette_dist / g6.max_palette_dist:.0f}x the palette "
          f"distance and {r6.n_unique_max} unique colours, which is why tau moved")
    h_w = cfg.shapes.image_size[0] * cfg.shapes.image_size[1]
    print(f"    row 6's bar {frac_max:.4%} is {frac_max * h_w:.0f} px at this "
          f"resolution; the same share was {frac_max * 4096:.0f} px at 64x64, "
          f"which is item 6's calibrated 350")

    es = entropy_split(man["counts"], knobs["levels"])
    uniform = math.log2(man["codebook_size"])
    chans = " / ".join(f"{b:.3f}" for b in es["channel_bits"])
    print()
    print(f"    row 3 taken apart: {es['joint_bits']:.3f} of {uniform:.0f} bits, "
          f"channels {chans}, marginal sum {es['marginal_sum_bits']:.3f}")
    print(f"    short by {uniform - es['joint_bits']:.3f} = "
          f"{uniform - es['marginal_sum_bits']:.3f} marginal skew + "
          f"{es['redundancy_bits']:.3f} redundancy; "
          f"{es['zero_count_codes']}/{man['codebook_size']} codes never used")

    failed = [n for n, _, _, _, ok in rows if ok is False]
    out = {"run_id": run_id, "val_psnr_db": db, "edge_psnr_db": edge_db,
           "flat_psnr_db": flat_db, "edge_error_share": edge_share,
           "entropy_ratio": man["entropy_ratio"], "live_codes": man["live_codes"],
           "offpalette_px_max": r6.offpalette_px_max, "offpalette_tau": tau,
           "offpalette_frac_max": r6.offpalette_frac_max,
           "recon_palette_dist": r6.max_palette_dist,
           "gap_db": result["gap_db"], "entropy_split": es, "failed_rows": failed}
    print("\nall pass/fail rows pass" if not failed else f"\nFAILED rows: {failed}")
    return out
