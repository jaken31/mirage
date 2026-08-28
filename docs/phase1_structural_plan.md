# Phase 1: Structural Plan

Guidance for implementing the six Phase 1 items - a three-line change to `sim/`,
a second config, and three Python files - one at a time. Names the calls and the
order; does not write the code. Design rationale is in
`world_model_architecture.md` - not repeated here.

Numbering below matches `AGENDA.md` exactly. If the two ever disagree, AGENDA is
the order of record and this file is the stale one.

**Vocabulary, once.** A *tokenizer* here is an autoencoder: an *encoder* squeezes
a 64x64 picture down to an 8x8 grid of whole numbers, a *decoder* rebuilds the
picture from that grid. *FSQ* (finite scalar quantization) is how the squeezing
is made discrete: the encoder emits 3 real numbers per grid cell, each is rounded
to one of 8 fixed levels, and 8x8x8 = 512 gives the vocabulary. There is no
learned dictionary, which is the whole reason it was chosen - a dictionary can
collapse onto a few entries and FSQ has none to collapse. Rounding has zero
gradient everywhere, so the backward pass uses a *straight-through estimator*: it
pretends the rounding was not there. *PSNR* is reconstruction quality,
`10*log10(255^2 / MSE)`, higher is better. A cell's *receptive field* is how many
input pixels it can see. The *token cache* is the whole dataset re-expressed as
tokens instead of pixels, and it is what Phase 2 trains on.

---

## The dataflow, in one line

Shards on disk -> `preload` recodes them to palette indices in RAM -> `encoder`
-> `FSQ` -> `decoder` -> loss. After training: encode every frame once to the
token cache, and re-run the F-9 sweep against reconstructions.

Everything flows one direction. Nothing calls backwards.

---

## What each file owns

| File | Owns | Explicitly does not own |
|---|---|---|
| `scene/arm_blocks.xml` | nothing new. **It is frozen** - see the gotcha table | - |
| `sim/main.cpp` | writing `cfg.height`/`cfg.width` into `model->vis.global.offheight`/`offwidth` before the context exists | the XML, which no longer states the render size |
| `mirage/configs/base96.json` | the 96x96 arm: `sim.height`/`width` 96, `data.shard_dir` `data/shards96` | anything the 64x64 config owns; they are two datasets with two `data_hash`es |
| `mirage/data.py` | `preload(shards, index, split, val_fraction, palette_rgb)` -> palette-index array plus the byte LUT | anything model-shaped; it still does not interpret pixels. It does not *load* the palette either - `validator` imports `data`, so the rgb array is passed in |
| `mirage/logging.py` | `log(dict)` -> one jsonl line always, W&B only behind a flag | deciding what is worth logging |
| `mirage/fsq.py` | the quantizer, encoder, decoder, train loop, eval, the gate table, the token cache writer | loading shards, and the palette |
| `mirage/configs/base.json` | after item 6, the calibrated `validator` thresholds | the measurement that produced them |

`preload` belongs in `data.py`, not `fsq.py`, because Phase 2 will want the same
palette-index array for its own eval frames, and a copy in two files is the same
class of bug as two validator implementations.

---

## Build order, with the calls each file needs

Same numbering as `AGENDA.md`. Riskiest first, which here means "the thing that
could invalidate 300,000 frames" first.

### 1. `sim/main.cpp` - the offscreen size comes from config, not the XML

Three lines, and they have to sit between `mj_loadXML` and the `GlContext`
constructor:

```cpp
mjModel* model = mj_loadXML(...);
model->vis.global.offwidth  = cfg.width;
model->vis.global.offheight = cfg.height;
GlContext context(model);
```

`offwidth` and `offheight` are plain mutable `int` fields on `mjModel`
(`mjmodel.h`, `struct mjVisual_`, the `global` sub-struct), and `mjr_makeContext`
reads them to size the offscreen framebuffer. Verified by reading the header
shipped with the pinned MuJoCo 3.12.0.

**Why this and not editing the XML.** `data_hash` is
`sha256(canon(sim) + canon(data) + xml_bytes)`, so raising `offwidth` from 64 to
96 in the file would change the hash of the 64x64 dataset too, and invalidate
300,000 frames in order to make a *second* dataset possible. Setting it from
config leaves the XML byte-identical, and both resolutions become a config-only
change.

**One existing check does lose its teeth, found by making the change.**
`gl_context.cpp` compares `mjr_maxViewport` against `model->vis.global.offwidth`
and `main.cpp` cross-checks the viewport against `cfg.width` - and those are now
the same number, so what used to be a third independent corner is one fact
checked twice. Both are kept, because they cost nothing and they still catch a
driver that hands back a viewport other than the one asked for, but the comment
in `main.cpp` now says plainly that it is a duplicate. Do not read the pair as
corroboration.

Doc page: **API reference -> `mjModel`, `mjVisual`**. Nothing new to learn.

**Working when:** the 64x64 run is a no-op - regenerate all 300,000 frames and
every `.pixels` and `.meta` blob is **byte-identical** to what is on disk now,
and `load_shards(dir, cfg.data_hash)` accepts them. That comparison is the whole
reason this item goes first: it is the proof the XML stayed frozen.

**Done 2026-08-28.** All 14 blobs byte-identical, `git_sha` the only sidecar
field that moved, the fixture reproduced under the ASan build too. The full
regeneration measured **60.2 s at 4,980 fps** - budget from that, not from the
45-50 s recorded earlier the same day.

### 2. `mirage/configs/base96.json` - the second dataset

Copy `base.json`, change three values: `sim.height` 96, `sim.width` 96,
`data.shard_dir` `data/shards96`. Nothing else, and specifically not
`tokenizer.stride` - `Shapes.token_grid` is `(height // stride, width // stride)`,
so stride 8 at 96x96 already yields the 12x12 = 144 token grid.

`frames_per_shard` stays 50,000. It counts **frames, not bytes**, so the shard
split is byte-for-byte the same decision at both resolutions: **7 shards**, three
of 43,200 frames and four of 42,600, because episodes are spread evenly over the
shards rather than packed to the 50,000 ceiling. The largest 96x96 shard is
1.19 GB. Do not derive the shard count from `frames_per_shard` - read a sidecar.

Generation measured **65.8 s at 4,560 fps** on 2026-08-28 and lands **8.294 GB**
on disk, inside R-4's 20 GB with the 64x64 set alongside it. Two earlier figures
for this line were wrong in opposite directions: "~45 s at 6,775 fps" came from a
superseded throughput number, and extrapolating the measured 64x64 cost by pixel
count predicted 1.5-2 min. **2.25x the pixels costs 9% more wall clock** - 65.8 s
against 60.2 s - because generation is dominated by physics stepping and fixed
per-frame cost, not by pixel throughput. Do not size a resolution change by
pixel count.

**Working when:** `load_shards("data/shards96", cfg96.data_hash)` returns **7**
shards summing to 300,000 frames with the same per-shard frame counts as the
64x64 set, the two directories report **different** `data_hash`es, and
`mirage/validator.py` over the new set still reports at most 7 unique colours per
frame. F-2 is resolution-independent, and a failure here means the render config
did not survive the size change.

**Done 2026-08-28**, all of the above, at `data_hash 35e5b8627987a2bb`. Both
self-checks take an optional config path now, so this is
`python -m mirage.validator mirage/configs/base96.json` rather than a throwaway
script; an explicit path skips the fixture fallback on purpose, so a missing set
fails by name instead of quietly checking 40 other frames.

**One number moved, and it is the one that matters to the fork: F-7 fell from
5.35% to 4.78%**, margin 1.6x the 3% floor against 1.8x at 64x64. That is the
resolution working as intended - a bigger frame makes total occlusion rarer - but
it means the 144-token path buys edge fidelity and spends occlusion headroom.
F-6 is unchanged at 16.63%, contact being a physics fact rather than a pixel one.

### 3. `mirage/data.py` - `preload`

One function. Returns a `(n, h, w)` `uint8` array of palette indices plus the
`(7, 3)` `uint8` LUT that inverts it.

Why indices and not RGB: there are exactly **7** distinct byte triples across all
300,000 frames - measured as a union over the whole set, not a per-frame count -
so one byte per pixel is lossless, and the train split becomes **1.16 GB instead
of 3.49 GB**. That matters because this machine has ~4.9 GB free, and the loader
reads **6,804 frames/s cold against 109,682 warm**. Training needs ~13,000, so
the cold case is a real failure and the page cache cannot be relied on at a 3.5
GB working set.

Two things it must do:

1. **Flip the rows.** The blob is bottom-up. `WindowSampler.__getitem__` already
   flips; `preload` bypasses the sampler, so it has to flip itself or every
   downstream angle measurement is mirrored.
2. **Assert losslessness, do not assume it.** Map each pixel to its nearest
   palette entry by `argmin` over squared distance, then check that
   `LUT[indices]` reproduces the original block **exactly**, and that the worst
   distance any pixel sits from its assigned entry is under 1.0. Measured worst
   is 0.75. Exact RGB equality does not work here, and the reason is recorded in
   the architecture doc - `rgba * 255` does not land on integers.

Split by episode, via the existing `episode_index` and `is_val`. Do not add a
second split rule: `is_val` is a pure function of `episode_id`, which is what
makes the tokenizer's val set and Phase 2's the same set by construction. At
`val_fraction` 0.05 that is 473 train / 27 val episodes, 283,800 / 16,200 frames.

Doc page: **NumPy -> fancy indexing, `np.take`**.

**Working when:** `LUT[preload(...)]` equals a straight
`np.array(shard.pixels[i, ::-1])` byte for byte on a few thousand random frames,
the train array is 1.16 GB, and the same call at 96x96 returns 2.62 GB with the
same 7-entry LUT.

**Done 2026-08-28**, all of it, every figure landing exactly as predicted -
1.16 GB against 3.49 raw at 64x64, 2.62 against 7.85 at 96x96, one LUT of seven
entries shared by both. `preload` takes the palette rgb as a parameter, not from
`load_palette`, because `validator` imports this module.

`python -m mirage.data` runs the round-trip check on the **val** split only.
Materialising 1.16 GB (2.62 at 96x96) on every self-check run would buy no
coverage the val pass does not already give - it is the same code over 17x the
frames - and the size is arithmetic. The full build is measured once and lives
in `runs.jsonl`: **37.5 s at 64x64, 87.8 s at 96x96**, about 90 MB/s both times,
so it is packing-bound rather than disk-bound. One-time against a ~6 min run,
but the 96x96 ladder pays it per rung.

### 4. `mirage/logging.py`

`log(dict)` appends one JSON object per line to a run-scoped jsonl, always, and
mirrors to W&B only when a flag is set. Roughly 40 lines. Every record carries
the run id and the relevant hash, so E-4 and E-5 are satisfied by construction
rather than by remembering.

This is the first phase with a training loop, which is the only reason the file
lands now rather than in Phase 0.

Doc page: none. Stdlib `json`, `pathlib`, `time`.

**Working when:** a run writes a readable jsonl with W&B absent from the
environment entirely, and `pandas.read_json(..., lines=True)` parses it.

### 5. `mirage/fsq.py`

The whole phase in one file, built in five stages. Roughly 360 lines, which is
inside the 400-line comfortable band; if it passes 500, the eval half wants to
become `fsq_eval.py` and not before.

**5a. The quantizer.** ~40 lines, and the math is already verified numerically:

```
half_l = (levels - 1) * (1 + eps) / 2
shift  = atanh(offset / half_l)          offset = 0.5 for even levels, else 0
bound(z) = tanh(z + shift) * half_l - offset
q = bound(z); q = q + (q.round() - q).detach(); return q / (levels // 2)
```

Checked over `[-25, 25]` for `[8,8,8]`, `[8,6,5]`, `[5,5,5]` and `[4,4,4]`: each
yields exactly `prod(levels)` distinct values per dimension, and
`codes_to_indices` is a bijection onto `0..prod(levels)-1`. No commitment loss,
no codebook loss, no EMA, no dead-code restart - FSQ has no dictionary to
maintain, and adding an auxiliary loss here undoes the reason it was picked.

**5b. Encoder and decoder.** Stride 8 forces three stride-2 stages. Channels
3 -> 64 -> 128 -> 256, then a 1x1 conv to `len(levels)` = 3. Decoder mirrors it.
About 1.5M parameters, so R-1 is a non-issue at any batch size worth using.

Three choices that are not stylistic:

- **Nearest-upsample plus 3x3 conv in the decoder, never `ConvTranspose2d`.**
  Transposed-conv checkerboarding presents as misplaced edges, and misplaced
  edges are the exact signal that decides the 64-vs-144 fork. A checkerboard
  would send the project to 96x96 on a false diagnosis.
- **One self-attention layer on the 8x8 grid.** 64 positions, so the attention
  matrix is 64x64 and costs nothing. It is also the only mechanism by which the
  64 codes describe the frame *jointly* rather than independently, and
  independently is measured: a k-means codebook of 512 entries over real 8x8
  patches reaches **29.02 dB** against Q-1's 30 dB bar. The **0.98 dB** gap is what
  context has to buy - and because that margin is inside a training run's noise,
  the attention layer's value shows up in gate row 2, not row 1.
- **`GroupNorm` + `SiLU`, no residual blocks.** Residual blocks are the first
  capacity lever if rung R2 falls short, not a starting assumption.

Input scaled to `[0, 1]`; final conv linear with **no** output activation and
**no** clamp inside the loss. `tanh` on the output would saturate on exactly the
values this scene is made of - pure black void and saturated blocks - and
clamping inside the loss removes the gradient that penalises overshoot. Clamp
only when materialising `uint8`.

**5c. Loss and the training loop.** Plain MSE and nothing else. PSNR is a
monotone function of MSE, so the loss *is* the gate.

The alternative was checked and it loses: per-pixel 7-way cross-entropy gives
hard edges, but on this palette the mean squared distance between two distinct
entries is **47,814**, so classification needs ~99.6% pixel accuracy to clear
30 dB while regression can hedge with a blend.

MSE's hedging is a real hole: it rewards blurring edges, and edges are where
**99.95%** of the error lives. The counterweight is item 6 - `offpalette_px` on
reconstructions punishes precisely the blur that PSNR rewards, and the two
numbers cannot both be gamed. Report them together or neither means much.

**fp32, not bf16.** `ingredients.md`'s bf16 line is about the 15M-parameter
dynamics model at context 1024, where it is necessary. Here the model is 1.5M
parameters and activations at batch 128 are ~400 MB, so fp32 is free, and it
removes a class of numerical doubt from the one number the whole phase turns on.
Add autocast only if a measured step time asks for it.

**Starting hyperparameters - a starting point, not a measurement.** Nothing below
is verified. They are here so the first run is reproducible, and they are expected
to move:

| Knob | Start at | Why this and not something else |
|---|---|---|
| batch | 128 | 2,217 steps/epoch. Preloaded, so the loader is not the constraint |
| optimizer | AdamW, `weight_decay` 1e-4 | boring default; no evidence yet that decay matters at this data-to-parameter ratio |
| lr | 3e-4, cosine to 3e-5 | the standard small-conv-autoencoder starting point |
| warmup | 5% linear | the bottleneck `tanh` can saturate early on a cold start; cheap insurance |
| epochs, ladder rungs | 15 | ~6 min each, enough to *rank* architectures |
| epochs, the winner | 60 | ~25 min, the run whose PSNR is quoted |

**Ranking at 15 epochs assumes the ordering survives longer training.** That
usually holds for capacity comparisons and is not guaranteed. If two rungs land
within ~0.5 dB of each other at 15, they are tied and both need the long run.

**5d. The ladder.** Four runs, each answering exactly one question. Run R0 before
FSQ is wired in at all:

| Rung | Config | Answers |
|---|---|---|
| R0 | continuous bottleneck, quantizer bypassed, no attention | the architecture's ceiling. **If R0 misses 30 dB, no levels table will ever help** and the encoder is what needs work |
| R1 | FSQ `[8,8,8]`, no attention | what quantization costs, and it is directly comparable to the **29.02 dB** k-means floor |
| R2 | R1 plus attention at 8x8 | what joint coding buys |
| R3 | only if R2 falls short | residual blocks, wider channels, or the levels ladder - with the paired LR check below |

Then R1 through R3 again on the 96x96 dataset, which is what turns the fork into
a measurement instead of a prediction.

**The paired LR check, and why it is not optional.** The straight-through
gradient at zero is **0.858** for `[8,8,8]`, **1.001** for `[5,5,5]` and
**0.668** for `[4,4,4]` - the `tanh` derivative is inside it, because the STE
bypasses only the rounding. A levels change therefore rescales the effective
bottleneck learning rate by up to 1.5x, and a comparison run at one LR reports a
levels result that is partly an LR result. Run each levels variant at the base LR
*and* at the base LR scaled by `0.858 / g_new`.

**5e. The token cache.** After the winning run, encode all 300,000 frames once:
one `.npy` of `uint16` per shard, `(frames, 8, 8)`, in a directory named by the
**run id**, with a manifest recording `tokenizer_hash`, the checkpoint, and the
per-shard frame counts. 38.4 MB total.

Per-shard and not one flat array, because a flat array needs a cumulative frame
offset to address and that is an off-by-one factory; per-shard makes
`len(tokens) == shard.frames` a loud assert. Named by run and not by
`tokenizer_hash`, because two runs at identical config and different seeds share
a hash and produce different tokens - the architecture doc settled this, and the
checkpoint carries the hash inside it for provenance.

Doc pages: **PyTorch -> `nn.Conv2d`, `nn.GroupNorm`,
`F.scaled_dot_product_attention`, `torch.optim.AdamW`**. Read the FSQ paper's
levels table before touching item 5a's `levels`.

**Working when:** `python -m mirage.fsq` self-checks the quantizer without
touching data - `prod(levels)` distinct values per dimension, index bijection,
gradient finite - and `python -m mirage.fsq --eval` prints the gate table.

### 6. F-9 recalibration, and thresholds finally land in config

The one Phase 0 item left open, and it is not a chore. It is the counterweight to
5c.

`validator.sweep(frames, metas, palette, tau)` needs no change: pass the
**reconstructed** frames with the **original** metas, because ground truth came
from the simulator and is still true. What changes is the expected magnitude -
`offpalette_px` reads 0 on every ground-truth frame at tau 8, and an FSQ decoder
emits continuous colour, so it will not read 0 here.

**Do not apply F-2 to reconstructions.** The 24-colour bar is a statement about
the renderer. A healthy decoder emits hundreds of colours and F-2 would fail it.

Write the resulting set into `configs/base.json`'s `validator` section. That
section branches off `data_hash`, not `dynamics_hash`, so writing it does not
invalidate any checkpoint.

**Working when:** the sweep reports a threshold set with zero false positives on
reconstructions, `px_count_margin` is printed next to it so the tightness is
visible, and editing the numbers leaves `data_hash` and `tokenizer_hash`
unchanged.

---

## Gotchas, and how you would notice

| Gotcha | What breaks | How you notice |
|---|---|---|
| **`scene/arm_blocks.xml` is frozen, comments included** | Any byte changes `data_hash` and orphans 300,000 frames | `load_shards(dir, cfg.data_hash)` raises - but only if you pass the hash. Pass it. Notes about `offwidth` belong in `main.cpp` |
| The FSQ straight-through gradient is not 1.0 | A levels comparison silently becomes an LR comparison | Two levels tables rank differently at two LRs. Measured: 0.858 / 1.001 / 0.668 for `[8,8,8]` / `[5,5,5]` / `[4,4,4]` |
| `Shapes.token_grid` is floor division | `height=100` would silently drop 4 pixels off every frame | **Already guarded** - `config.py`'s `_check_values` rejects `sim.height`/`width` not divisible by `tokenizer.stride`, and `_self_check` covers it. Verified 2026-08-28: `100/8` and `64/6` rejected, `96/8` accepted as `token_grid=(12, 12)`. Do not re-add it |
| The pixel blob is bottom-up | `preload` bypasses the sampler's flip | Every angle and `link_extent` mirrors. Visible immediately if you look at one frame, silent forever if you do not |
| PSNR computed on the raw float output | Reports ~0.01 dB the pipeline never delivers | It does not fail, it just is not the number. Round to `uint8` first, and hand the validator the same frames |
| `ConvTranspose2d` in the decoder | Checkerboard artifacts read as edge error | Sends the project to 96x96 for a week on a false diagnosis |
| F-2 applied to reconstructions | A healthy decoder fails a renderer check | `n_unique_colors` reads in the hundreds. Mode 1 only |
| Adding an entropy or commitment loss to FSQ | Undoes the reason FSQ was chosen over VQ | Q-2 improves and you can no longer tell whether the codebook would have collapsed. If Q-2 misses, shrink the vocabulary instead |
| A page cache assumption | Training runs 16x slower than the probe promised | Cold 6,804 frames/s vs warm 109,682. This is why item 3 exists |
| Two runs writing one token-cache directory | Phase 2 trains on a mixture of two tokenizers | The manifest's `tokenizer_hash` disagrees with the checkpoint's. Name the directory by run id |

---

## The numbers already taken, so item 5 does not re-derive them

All from the 300,000 frames on disk, before any of Phase 1 exists. Re-measured
2026-08-28 at `data_hash 18a76531` by `bench/patch_probe.py`, which is also where
the k-means rows' method lives - **quote a k-means number from anywhere else and
you are probably quoting a random-init run**, which reads 2.6 dB low.

| Measure | Number | What it settles |
|---|---|---|
| Union of distinct byte triples, whole set | **7**, worst palette distance 0.75 | item 3's palette-index preload is lossless, 1.16 GB not 3.49 GB |
| 8x8 patches that are one flat colour | 63.47% | most of the frame is free to reconstruct |
| Interior cells with a fully flat 22x22 receptive field | 20.28%, all table | Q-2's provable ceiling is 94.3% of uniform, so **the data does not force Q-2 to fail** |
| k-means, 512 centroids, real patches | **29.02 dB**, all 512 centroids live | the floor item 5 must beat by **0.98 dB**. The superseded 26.39 dB / 150-live pair came from an unrecorded random initialisation, and was the only direct evidence Q-2 was at risk |
| same, 1024 centroids | **30.51 dB** | doubling the vocabulary clears Q-1 outright, so vocabulary *is* a lever - it is just one the token budget forbids, since codes per token is fixed at 512 by the Phase 2 handoff |
| Share of that error in non-flat patches | **99.95%**, in 36.53% of patches | the fork diagnostic is close to predetermined - if Q-1 misses, 96x96 is indicated |
| Loader at ctx=0, cold / warm | 6,804 / 109,682 frames/s | against a ~13,000 need: preload, do not hope |
| FSQ levels tables checked | `[8,8,8]` `[8,6,5]` `[5,5,5]` `[4,4,4]` | `prod(levels)` codes exactly, index bijection holds |

Restating Q-1 in something physical, since a wrong pixel costs 47,814 squared
error on this palette: **30 dB is about 17 of 4,096 pixels completely wrong**,
35 dB is 5.3, and the k-means floor is 38.4. Phase 1's task is to halve the count
a dumb dictionary produces.

Record the GPU power state next to every timing. A timing without it is not a
number.
