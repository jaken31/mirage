"""Everything after a tokenizer is trained: the token cache and the gate table.

Split out of `fsq.py` when that file grew past 500 lines, the size limit the
plan set. The split is by *when the code runs*: `fsq.py` builds and trains, and
this file works on a run that already exists.

- `write_token_cache` - one uint16 `.npy` of token ids per shard, plus a manifest.
- `load_run` - loads a checkpoint back into an eval-mode `Tokenizer`.
- `evaluate` - the eight-row pass/fail "gate table" for a run.
- `_self_check` - `Tokenizer.decode` on R1's cached rows against `reconstruct`.

Run through `fsq.py`'s command line rather than its own, because the docs use
`python -m mirage.fsq --eval` and a second entry point would be one more thing
to keep in sync:

    python -m mirage.fsq --tokens RUN_ID
    python -m mirage.fsq --eval RUN_ID

Its own entry point runs only the self-check, which decodes R1's cached token
rows through `Tokenizer.decode` and compares them with `reconstruct`:

    python -m mirage.fsq_eval
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


# ---------------------------------------------------------------- token cache

def load_run(run_id: str, cfg: config.Config,
             device: torch.device | None = None) -> tuple[Tokenizer, dict]:
    """`runs/<run_id>/model.pt` -> an eval-mode `Tokenizer`, plus its settings.

    Rebuilds the network from the settings saved in the checkpoint, not from
    arguments, so a caller cannot load R1's weights into R2's shape by mistake.
    Refuses a checkpoint whose `data_hash` differs from `cfg`'s: a tokenizer only
    means something on the frames it was trained on, and mismatched hashes are
    how a stale checkpoint ends up used on regenerated data.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=True: the checkpoint holds only tensors, two hash strings and
    # a dict of plain settings, so full unpickling (which can run arbitrary
    # code) is not needed, and checkpoints are exactly the files that get
    # copied between machines.
    ckpt = torch.load(ROOT / "runs" / run_id / "model.pt", map_location=dev,
                      weights_only=True)
    knobs = ckpt["knobs"]
    if ckpt["data_hash"] != cfg.data_hash:
        raise ValueError(
            f"{run_id} was trained on data_hash {ckpt['data_hash'][:8]}, "
            f"this config is {cfg.data_hash[:8]}"
        )
    # `.get`, not `[...]`: checkpoints from before the `r1c` rung have no
    # `encoder_norm`, and all of them use GroupNorm.
    model = Tokenizer(tuple(knobs["levels"]), attention=knobs["attention"],
                      quantize=knobs["quantize"],
                      encoder_norm=knobs.get("encoder_norm", "group")).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, knobs


@torch.no_grad()
def write_token_cache(run_id: str, cfg: config.Config, batch: int = 256,
                      device: str | None = None) -> dict:
    """Encode every frame once: one uint16 `.npy` per shard in the run's directory.

    Per shard, not one flat array, because a flat array needs running frame
    offsets, which invite off-by-one errors. Per shard also makes
    `len(tokens) == shard.frames` a simple assert (gate row 4).

    Stored by run id, not `tokenizer_hash`, because two runs with the same config
    but different seeds share a hash and give different tokens. The hash is
    recorded inside the checkpoint and repeated in the manifest.

    **Rows are flipped on the way in.** `Shard.pixels` stores them bottom-up, as
    the GL readback wrote them, and `data.preload` flips them for training. A
    cache written without the flip would hold valid-looking tokens for
    upside-down frames and fail nowhere until the next model trained on them.

    This pass over 3.5 GB of frames has to happen anyway, so two gate rows are
    computed along the way: a 512-bin histogram gives row 3 (token entropy) and
    row 8 (live codes), and a sha256 per shard gives row 5, since re-running this
    writer and comparing manifests *is* the bit-identical check.
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
            px = np.ascontiguousarray(sh.pixels[i:j, ::-1])  # flip rows right-side up
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
        # encoder, `F.scaled_dot_product_attention` gives slightly different
        # floating-point results at different batch sizes, and about 2 values in
        # 100,000 sit close enough to a rounding boundary to flip. R2's shard 0
        # encoded at batch 128 vs 256 differs in 10 of 512,000 tokens; R1 (no
        # attention) differs in 0. So a re-encode is only bit-identical at the
        # same batch, and the batch must be recorded, not assumed.
        "batch": batch,
        "frames": sum(s["frames"] for s in per_shard),
        "tokens": total,
        "shards": per_shard,
        # Gate rows 3 and 8, computed on the pass that had to happen anyway.
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

    Computed from the **ground truth**, never from the reconstruction. Using the
    model's own output would let a blurry decoder relabel its mistakes as edges,
    making the flat-pixel score look better than it is.
    """
    b, h, w = idx_batch.shape
    p = idx_batch.reshape(b, h // patch, patch, w // patch, patch)
    flat = (p == p[:, :, :1, :, :1]).all(2).all(-1)  # (b, gh, gw)
    return np.repeat(np.repeat(flat, patch, 1), patch, 2)


@torch.no_grad()
def edge_flat_psnr(model: nn.Module, idx: np.ndarray, lut: torch.Tensor, patch: int,
                   batch: int = 256) -> tuple[float, float, float]:
    """(flat-pixel dB, edge-pixel dB, edge share of squared error) over `idx`.

    Gate row 7, and the evidence for choosing 64x64 (64 tokens) or 96x96 (144
    tokens). The k-means baseline puts **99.95%** of its error in the 37% of
    patches that are not flat. If a trained tokenizer does the same, more pixels
    is what would help, which points to 96x96.
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
    """Gate row 3 (token entropy), broken down to show where the missing bits are.

    A token id is the mixed-base number `d0 + levels[0]*d1 + ...`, so each
    channel's digit distribution can be read off the same counts row 3 uses: no
    GPU, no re-encode. The joint entropy is `sum(per-channel) - redundancy`, and
    the two parts fail for different reasons, with different fixes:

    - **per-channel skew** - one channel's values sit off centre in the `tanh`
      range and never reach most of its levels. R2's channel 2 puts 81% of its
      mass on digits 0 and 1 and gives 1.964 of 3 bits.
    - **redundancy** - the channels carry copies of the same information. This
      is what `GridAttention` fixes: R1 -> R2 it falls 1.339 -> 0.781 bits,
      which is 76% of attention's whole entropy gain.

    Worth splitting because the plan's fix for low entropy (shrinking the
    vocabulary) targets *collapse*, where codes go unused, and neither part is
    collapse: no code has zero count in either rung.
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


# ---------------------------------- row 6: the validator on reconstructions

@torch.no_grad()
def reconstruct(model: nn.Module, idx: np.ndarray, lut: torch.Tensor,
                rows: np.ndarray, batch: int = 256) -> np.ndarray:
    """`rows` of `idx`, passed through the tokenizer and back, in the form the
    validator wants: `(n, h, w, 3)` uint8, channels last.

    Rounded and clamped to uint8, because that is what the pipeline delivers
    and `measure_pixels_only` accepts nothing else. Measuring the float output
    would calibrate for a case that never happens, and too kindly, since
    rounding itself creates off-palette colours.
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
    """(reconstruction sweep, ground-truth sweep) over the same validation rows.

    Gate row 6, and the measurement the validator's thresholds were calibrated
    on. The validator was first tuned on *renders*, frames with exactly seven
    colours. But the rollout quality measure runs on decoder output, which
    softens every edge into colours not in the palette, so the thresholds had to
    be re-derived on that; otherwise every rollout would fail on its first frame.

    Both sweeps use the same rows and the same truth. The ground-truth one
    matters:

      * it is the **alignment check**. `val_idx` and the meta come from two
        different functions, and a mismatch would silently pair frame i's
        pixels with frame j's truth, with both arrays still the right length
        and type and every number wrong but plausible. Ground truth through the
        same path must reproduce the known render result; when it does, the
        rows line up.
      * it is the **baseline**. A reconstruction number means little alone; the
        question is how much worse than a render the decoder is.

    Held-out rows on purpose: the validation split, the same frames row 1's
    PSNR uses. Thresholds tuned on frames the tokenizer trained on would expect
    better reconstructions than it will ever produce on new data.
    """
    metas = data.split_meta(shards, index, "val", cfg.data["val_fraction"])
    assert len(metas) == len(val_idx), (
        f"{len(val_idx)} val frames against {len(metas)} val meta records - "
        f"preload and split_meta disagree about the split"
    )

    # The whole validation split by default, not a sample, because this feeds a
    # **maximum**. A max over 2,000 frames is usually smaller than over 16,200,
    # so a sampled gate is easier than the calibration that set the limit: it
    # would pass here and fail on the full split, the worst way to be wrong.
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

    # The alignment check, and the only thing that catches a silent row shift.
    # These are the *known* ground-truth results: zero off-palette pixels, and a
    # worst distance of 0.75 from render rounding alone. Truth sent through this
    # exact path must reproduce them. If not, frames and meta are not the same
    # rows, and every reconstruction number above pairs one frame's pixels with
    # another frame's truth.
    assert g.offpalette_px_max == 0, (
        f"{g.offpalette_px_max} off-palette px on ground truth at tau {tau} - "
        f"F-9 says this is 0, so the frames and meta are misaligned or the palette moved"
    )
    assert g.max_palette_dist < 1.0, (
        f"ground truth sits {g.max_palette_dist:.2f} from the palette, expected 0.75"
    )
    return r, g


def evaluate(run_id: str, cfg: config.Config, device: str | None = None) -> dict:
    """The eight-row gate table for one run. Rows 1-6 are pass/fail.

    Row 1 is **recomputed from `model.pt`**, not read from `result.json`. The
    gate checks the file that would actually be used; trusting the training log
    would pass a checkpoint that failed to save correctly.

    Row 2 compares with the recorded k-means baseline for *this config's
    resolution* (`fsq.kmeans_floor_db`) instead of refitting k-means. The plan
    asked for a refit, but the validation split is fixed by `data.is_val` and a
    checked `data_hash`, so a refit returns the same number every time: a
    constant pretending to be a measurement. The real risk, the data changing
    under the baseline, is already caught by `load_run`'s `data_hash` check.
    `bench/patch_probe.py` stays the one place k-means is computed.

    Row 6 runs the validator sweep on decoder output, using the calibrated
    `validator.offpalette_tau` and `validator.offpalette_frac_max`. It is a
    **regression check, not a calibration**: the limits were set once, on the R2
    rung over the whole held-out split. Re-deriving them per run would change
    `validator_hash` and make rollout results from different runs incomparable.
    A later rung that fails them has a worse decoder, which is the point.
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

    # Row 5: re-encode shard 0 and compare its sha256 with the manifest. One
    # shard tests the same thing as seven at a seventh of the cost.
    #
    # **Use the manifest's batch size, not a fixed one.** Re-encoding at a
    # different batch than the cache was written with tests batch-size
    # sensitivity (a real effect with attention; see the `batch` key above),
    # not whether encoding is repeatable, and would fail R2 on a false alarm.
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
    floor = kmeans_floor_db(cfg)  # depends on resolution; see fsq.KMEANS_FLOOR_DB
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

    # Row 6 next to the renders the validator was first tuned on. Printed, not
    # just returned, because the gap *is* the result: the original thresholds
    # fit renders, but decoder output is where the validator actually has to
    # work.
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


# ------------------------------------------------------------------ self-check

@torch.no_grad()
def _self_check() -> None:
    """`Tokenizer.decode` on R1's cached token rows, against `reconstruct`.

    `fsq._self_check` proves the ids round-trip without data. This proves the
    token-to-pixel path on the checkpoint Phase 2 inherits: one full batch of a
    held-out episode, read from the token cache and decoded at the batch the
    cache was written at (the manifest's `batch`, the confound gate row 5 once
    failed on), compared with `reconstruct` on the same frames at that batch.

    **The two do not encode the same way, and this check shows it rather than
    hiding it.** `write_token_cache` feeds the encoder a channels-last view
    (`permute` without `contiguous`), while `reconstruct`'s `_batch` makes it
    contiguous. Under cuDNN's default TF32 convolutions the two layouts pick
    different kernels, and about 0.33% of tokens land on a different code. So
    three things are asserted, each exact:

    1. the cache-layout re-encode reproduces the cached rows, which pins every
       difference below on the layout;
    2. decoding the ids `reconstruct` itself produced reproduces its uint8
       frames on every frame, so the decode path has no error of its own;
    3. decoding the cached rows reproduces `reconstruct`'s uint8 frame on every
       frame whose cached row equals `reconstruct`'s ids.

    Reads one episode, not a split, and one batch on the GPU, so it stays short.
    Skipped, not failed, without R1's checkpoint and cache or without the
    generated set, the way `mirage.dynamics` skips R1's cache.
    """
    # Imported here: `dynamics` owns the name of the run Phase 2 inherits.
    from mirage.dynamics import TOKENIZER_RUN

    cfg, shard_dir, fixture = data.self_check_config()
    run_dir = ROOT / "runs" / TOKENIZER_RUN
    man_path = run_dir / "tokens" / "manifest.json"
    if fixture or not (run_dir / "model.pt").exists() or not man_path.exists():
        why = "the fixture has no token cache" if fixture else f"no R1 checkpoint and cache in {run_dir}"
        print(f"R1 decode: skipped - {why}")
        print("fsq_eval self-check ok (nothing to check)")
        return

    if not torch.cuda.is_available():
        # The cache was encoded on CUDA, whose TF32 convolutions round
        # differently from the CPU's, so check 1 would fail on the device alone.
        print("R1 decode: skipped - no CUDA device, and the cache was encoded on one")
        print("fsq_eval self-check ok (nothing to check)")
        return
    dev = torch.device("cuda")
    # `load_run` loads with `load_state_dict`'s default `strict=True`, so this
    # line is the strict-load check: an extra or missing buffer raises here.
    model, _ = load_run(TOKENIZER_RUN, cfg, dev)
    n_keys = len(model.state_dict())
    print(f"R1 {TOKENIZER_RUN}: strict load_state_dict ok, {n_keys} entries on {dev}")

    man = json.loads(man_path.read_text(encoding="utf-8"))
    batch = man["batch"]
    shards = data.load_shards(shard_dir, cfg.data_hash)
    episodes = data.split_episodes(data.episode_index(shards), "val", cfg.data["val_fraction"])
    # The first batch the cache encoded entirely inside one val episode, so the
    # decode batch and the cache's encode batch hold the same frames.
    ep, start = next((e, a) for e in episodes
                     for a in [-(-e.start // batch) * batch]
                     if a + batch <= e.start + e.length)
    sh = shards[ep.shard]
    cached = np.load(run_dir / "tokens" / f"shard_{sh.index:03d}.npy")[start:start + batch]

    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
    idx, lut_np = data.preload(shards, [ep], "val", cfg.data["val_fraction"], palette.rgb)
    lut = torch.from_numpy(lut_np).to(dev).float()
    rows = np.arange(start - ep.start, start - ep.start + batch)
    recon = reconstruct(model, idx, lut, rows, batch=batch)

    def to_u8(ids: np.ndarray) -> np.ndarray:
        # `reconstruct`'s uint8 conversion, applied to `decode` instead of `forward`.
        y = model.decode(torch.from_numpy(ids.astype(np.int64)).to(dev))
        return (y * PEAK).round().clamp(0, PEAK).byte().permute(0, 2, 3, 1).cpu().numpy()

    # 1. write_token_cache's input path, on the same frames and batch.
    px = np.ascontiguousarray(sh.pixels[start:start + batch, ::-1])
    x_cache = torch.from_numpy(px).to(dev).permute(0, 3, 1, 2).float() / PEAK
    redo = model.encode(x_cache).cpu().numpy()
    flips_redo = int((redo != cached).sum())
    assert flips_redo == 0, f"re-encoding the cache's input path flips {flips_redo} tokens"

    # 2. reconstruct's own ids, through decode.
    own = model.encode(_batch(idx, lut, rows) / PEAK).cpu().numpy()
    gap_own = int(np.abs(to_u8(own).astype(np.int16) - recon).max())
    assert gap_own == 0, f"decode of reconstruct's ids differs from it by {gap_own}"

    # 3. the cached rows, through decode.
    decoded = to_u8(cached)
    same = (own == cached).reshape(batch, -1).all(1)
    gap_same = int(np.abs(decoded[same].astype(np.int16) - recon[same]).max()) if same.any() else 0
    assert same.any(), "no frame's cached row matches reconstruct's ids - nothing compared"
    assert gap_same == 0, f"decoded cache rows differ from reconstruct by {gap_same} on matching frames"
    flips = int((own != cached).sum())
    gap_all = int(np.abs(decoded.astype(np.int16) - recon).max())

    frames = f"shard {sh.index} frames {start}..{start + batch - 1}"
    print(f"R1 decode, val episode {ep.episode_id}, {frames}, batch {batch} (the manifest's):")
    print(f"  cache-layout re-encode vs cached rows: {flips_redo} of {cached.size:,} tokens differ")
    print(f"  decode(reconstruct's ids) vs reconstruct: max abs diff {gap_own} over all {batch} frames")
    print(f"  decode(cached row) vs reconstruct: max abs diff {gap_same} on the {int(same.sum())} "
          f"frames whose cached row equals reconstruct's ids")
    print(f"  the other {int((~same).sum())} frames differ by up to {gap_all} on {flips} tokens "
          f"({flips / cached.size:.2%}) that reconstruct's contiguous input encodes differently")
    print("fsq_eval self-check ok")


if __name__ == "__main__":
    _self_check()
