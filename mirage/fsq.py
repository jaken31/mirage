"""The tokenizer: FSQ quantizer, conv encoder/decoder, train loop, PSNR eval.

An *autoencoder*. The encoder squeezes a 64x64 RGB frame to an 8x8 grid of
`len(levels)` numbers per cell; the decoder rebuilds the frame from that grid.
FSQ (finite scalar quantization) makes the grid discrete by rounding each number
to one of a fixed set of levels, so `prod(levels)` is the vocabulary and there is
no learned dictionary to collapse. Rounding has zero gradient everywhere, so the
backward pass uses a straight-through estimator - it pretends the rounding was
not there.

Built in the order `docs/phase1_structural_plan.md` section 5 names:

- **5a** `FSQ` - the quantizer, and `codes_to_indices`.
- **5b** `Tokenizer` - encoder, optional 8x8 self-attention, decoder.
- **5c** `train` - MSE, AdamW, cosine schedule, held-out PSNR.
- **5d** `RUNGS` - R0, R1 and R2 as one flag pair each.
- **5e** and the gate table live in `fsq_eval.py`, split out when this file
  passed the 500 lines the plan names as the trigger. The division is by when
  the code runs: this file builds and trains, that one reads an artifact.

R3 is deliberately not a rung: "residual blocks, wider channels, or the levels
ladder" is three different runs, and which one it means is decided by R2's number
rather than in advance.

    python -m mirage.fsq              # 5a/5b self-check, touches no data
    python -m mirage.fsq --run r0     # continuous bottleneck - the ceiling
    python -m mirage.fsq --run r1     # FSQ [8,8,8], no attention
    python -m mirage.fsq --run r2     # R1 plus attention on the 8x8 grid
    python -m mirage.fsq --tokens ID  # encode all 300,000 frames from that run
    python -m mirage.fsq --eval ID    # the gate table

PSNR is `10*log10(255^2 / MSE)` on **uint8** reconstructions, not on the raw
float output. The float number is ~0.01 dB better and the pipeline never delivers
it, so it is not the number.
"""

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mirage import config, data, validator
from mirage.logging import Run

ROOT = Path(__file__).resolve().parent.parent
PEAK = 255.0

# Q-1's bar, and the floor gate row 2 charges a rung against. The floor is
# k-means++ at 512 codes fit on the 473 train episodes and scored on the 27 val
# ones - the same treatment a tokenizer gets - measured 2026-08-28 by
# `bench/patch_probe.py`. The 29.02 dB still quoted in older prose was fit *and*
# scored on a sample that straddled the split; 0.75 dB of that difference is the
# leak and 0.86 dB is the advantage of scoring a codebook on its own patches.
PSNR_BAR_DB = 30.0
KMEANS_FLOOR_DB = 28.27

# One rung is one question, and the only thing that varies is these two flags.
# R3 is deliberately absent: it is "residual blocks, wider channels, or the
# levels ladder", which is three different runs and is only defined once R2's
# number says which one is needed.
RUNGS = {
    "r0": dict(quantize=False, attention=False),  # the architecture's ceiling
    "r1": dict(quantize=True, attention=False),   # what quantization costs
    "r2": dict(quantize=True, attention=True),    # what joint coding buys
}

# GroupNorm needs a divisor of every channel count it is handed. 8 divides 64,
# 128 and 256, and the exact value has never mattered in a conv autoencoder this
# small - it is a named constant so it is not eight magic 8s.
GN_GROUPS = 8


# ------------------------------------------------------------------- 5a. FSQ

class FSQ(nn.Module):
    """Finite scalar quantization, one levels count per latent channel.

        half_l   = (levels - 1) * (1 + eps) / 2
        offset   = 0.5 for an even levels count, else 0
        shift    = atanh(offset / half_l)
        bound(z) = tanh(z + shift) * half_l - offset
        q        = round(bound(z)) via straight-through, then / (levels // 2)

    `eps` widens the bound just past the outermost level so the `tanh` asymptote
    does not sit exactly on it; `offset` and `shift` re-centre an even levels
    count, whose levels straddle zero rather than including it.

    **No auxiliary loss.** No commitment, no codebook loss, no EMA, no dead-code
    restart. FSQ has no dictionary to maintain, and adding one of those undoes
    the reason it was chosen over VQ - a Q-2 improvement would no longer
    distinguish "the vocabulary is well used" from "the loss propped it up". If
    Q-2 misses, shrink the vocabulary instead.

    The straight-through gradient is **not 1.0**: the STE bypasses only the
    rounding, so the `tanh` derivative survives into the backward pass. At zero
    it is 0.858 for [8,8,8], 1.001 for [5,5,5] and 0.668 for [4,4,4] - a levels
    change silently rescales the effective bottleneck learning rate by up to
    1.5x, which is why every levels comparison runs at two LRs. `_self_check`
    reproduces those three numbers.
    """

    # Annotated at class level so the buffers below type as plain tensors.
    # register_buffer is declared as returning None, so without these pyright
    # reads every `self.half_l` as `Tensor | Module | None`.
    half_l: torch.Tensor
    offset: torch.Tensor
    shift: torch.Tensor
    scale: torch.Tensor

    def __init__(self, levels, eps: float = 1e-3) -> None:
        super().__init__()
        lv = torch.tensor([float(v) for v in levels])
        half_l = (lv - 1) * (1 + eps) / 2
        offset = torch.where(lv % 2 == 0, 0.5, 0.0)
        self.levels = [int(v) for v in levels]
        self.codebook_size = math.prod(self.levels)
        # Buffers, not plain tensors: they have to follow .to(device) and land
        # in the checkpoint, so a reloaded model quantizes identically.
        self.register_buffer("half_l", half_l)
        self.register_buffer("offset", offset)
        self.register_buffer("shift", torch.atanh(offset / half_l))
        self.register_buffer("scale", lv.div(2).floor())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> the same shape, values on a `levels[c]`-point grid."""
        if z.shape[1] != len(self.levels):
            raise ValueError(
                f"latent has {z.shape[1]} channels, levels names {len(self.levels)}"
            )
        v = (1, -1, 1, 1)  # broadcast the per-channel constants over NCHW
        q = torch.tanh(z + self.shift.view(v)) * self.half_l.view(v) - self.offset.view(v)
        q = q + (q.round() - q).detach()
        return q / self.scale.view(v)

    def codes_to_indices(self, q: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) of `forward` outputs -> (B, H, W) ids in 0..codebook_size-1.

        `forward` hands back `round(bound(z)) / (levels//2)`, a *normalised*
        value in roughly [-1, 1], because that is what the decoder consumes. The
        digit is recovered by undoing exactly that: `q * scale + scale` puts the
        `levels[c]` grid points on `0..levels[c]-1`, and the mixed-radix sum
        against the place values reads those digits off as one number - channel 0
        is the ones place, channel 1 the `levels[0]`s place, and so on.

        A bijection, and for free: every integer in `0..prod(levels)-1` has
        exactly one mixed-radix expansion, so there is no dictionary and nothing
        to collide. `_self_check` enumerates all `prod(levels)` code tuples and
        asserts the ids come back as `arange` - cheap at 512, and it is the check
        that catches a wrong un-shift. Without it a sign error emits negative
        digits that wrap into wrong-but-valid ids and surface 300,000 frames
        later as a corrupt token cache.

        The bound check is per channel, not against `max(levels)`: a mixed table
        like [8,6,5] has three different digit ranges and a single bound would
        wave through a digit 7 in the 6-level channel.

        Nothing is registered as a buffer here. The place values are rebuilt per
        call - two tiny tensors against a conv forward, so free - because a new
        buffer would change `state_dict` and make R0's existing checkpoint fail
        a strict load for no gain.
        """
        v = (1, -1, 1, 1)
        lv = torch.tensor(self.levels, device=q.device).view(v)
        digits = (q * self.scale.view(v) + self.scale.view(v)).round().long()
        if int(digits.min()) < 0 or bool((digits >= lv).any()):
            raise ValueError(
                f"digit out of range [{int(digits.min())}, {int(digits.max())}] for "
                f"levels {self.levels} - these are not this quantizer's outputs"
            )
        basis = torch.cat([lv.new_ones(1), lv.flatten()[:-1]]).cumprod(0).view(v)
        return (digits * basis).sum(1)


# ---------------------------------------------------- 5b. encoder and decoder

def _stage(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1),
        nn.GroupNorm(GN_GROUPS, cout),
        nn.SiLU(),
    )


def _up(cin: int, cout: int) -> nn.Sequential:
    """Nearest-neighbour upsample plus a 3x3 conv. **Never `ConvTranspose2d`.**

    Transposed-conv checkerboarding presents as misplaced edges, and misplaced
    edges are the exact signal that decides the 64-vs-144 resolution fork. A
    checkerboard would send the project to 96x96 on a false diagnosis.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, 3, 1, 1),
        nn.GroupNorm(GN_GROUPS, cout),
        nn.SiLU(),
    )


class GridAttention(nn.Module):
    """One single-head self-attention layer over the 8x8 latent grid.

    64 positions, so the attention matrix is 64x64 and costs nothing. It is also
    the only mechanism by which the 64 codes describe the frame *jointly* rather
    than independently, and independently is measured: a 512-entry k-means
    codebook over real 8x8 patches, fit on the train episodes and scored on the
    val ones, reaches **28.27 dB** against the 30 dB Q-1 bar. The **1.73 dB** gap
    is what context has to buy. (The 29.02 dB / 0.98 dB pair quoted elsewhere is
    the same measurement fit *and* scored on a sample that straddled the split -
    0.75 dB of leak. `bench/patch_probe.py` prints both.)

    **REFUTED as a quality lever, VERIFIED as an entropy one** - measured
    2026-08-29 with R1 and R2 both run to convergence at 60 epochs.

    The claim that used to sit here - that this layer's value shows up in gate
    row 2 and not row 1, because the margin is inside a training run's noise -
    was wrong three ways. It could not show up in row 2, which `evaluate` makes
    row 1 minus a constant by charging against the recorded floor. It buys
    **+0.087 dB** for **+263,680 parameters**, about a sixth of the plan's own
    "within ~0.5 dB means tied" threshold, so it is a measured non-lever for
    quality - and **R1 passes every gate row without it**. And that "training
    run's noise" was never measured; no seed has been repeated to this day.

    Where it does show up is **row 3**: **+3.5 pp of token entropy** at
    convergence, +8.2 pp at 15 epochs, by decorrelating the three FSQ digits -
    redundancy falls 1.339 -> 0.781 bits when it is added at 15 epochs. Training
    length is a *substitute* for it there, not a complement: without attention,
    60 epochs reaches 0.890 bits on its own.

    It also costs an E-1 caveat R1 does not have: with attention, re-encoding a
    shard at a different batch size changes ~2 tokens in 100,000. See
    `fsq_eval` and the verification log.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(GN_GROUPS, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        # (b, 1, hw, c): one head, so the head axis is a literal 1.
        heads = (t.transpose(1, 2).unsqueeze(1) for t in (q, k, v))
        a = F.scaled_dot_product_attention(*heads)
        return x + self.proj(a.squeeze(1).transpose(1, 2).reshape(b, c, h, w))


class Tokenizer(nn.Module):
    """3 -> 64 -> 128 -> 256 down, 1x1 to the latent, then the mirror back up.

    Stride 8 forces exactly three stride-2 stages. `GroupNorm` + `SiLU`, no
    residual blocks - residual blocks are the first capacity lever if a rung
    falls short, not a starting assumption.

    The final conv is **linear**: no output activation and no clamp. `tanh` would
    saturate on exactly the values this scene is made of, pure black void and
    saturated blocks, and clamping removes the gradient that penalises overshoot.
    Clamp only when materialising uint8.

    `quantize=False` is rung R0 - the same architecture with the bottleneck left
    continuous, which measures the ceiling the quantizer is then charged against.
    """

    def __init__(self, levels=(8, 8, 8), attention: bool = False,
                 quantize: bool = True, width: int = 64) -> None:
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        d = len(levels)
        self.fsq = FSQ(levels)
        self.quantize = quantize
        self.encoder = nn.Sequential(
            _stage(3, c1, 2), _stage(c1, c2, 2), _stage(c2, c3, 2),
            *([GridAttention(c3)] if attention else []),
            nn.Conv2d(c3, d, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(d, c3, 1), nn.GroupNorm(GN_GROUPS, c3), nn.SiLU(),
            _up(c3, c2), _up(c2, c1),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c1, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(self.fsq(z) if self.quantize else z)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [0, 1] -> (B, h, w) token ids, one per latent cell.

        Refuses a continuous bottleneck rather than quantizing one on the way
        past. An R0 model's latent was never trained against a rounding step, so
        the ids it produces would be well-formed and meaningless - the failure
        this guard exists to stop is a Phase 2 dataset that looks fine.
        """
        if not self.quantize:
            raise ValueError(
                "this model has a continuous bottleneck (R0) and has no token ids"
            )
        return self.fsq.codes_to_indices(self.fsq(self.encoder(x)))


# -------------------------------------------------- 5c. loss, loop, and PSNR

def psnr_db(sse: float, values: int) -> float:
    return 10.0 * math.log10(PEAK * PEAK / (sse / values))


def _batch(idx: np.ndarray, lut: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
    """Palette indices -> (B, 3, H, W) of 0..255 floats on the LUT's device.

    The indices live in CPU RAM - 1.16 GB for the train split, against ~5 GB free
    on an 8 GB card that is also holding activations - and one batch is 512 KB,
    so the transfer is noise next to the step. The LUT expansion runs on the GPU
    because it is a gather over 7 rows.
    """
    rows_u8 = np.ascontiguousarray(idx[rows])
    b = torch.from_numpy(rows_u8).to(lut.device, non_blocking=True).long()
    return lut[b].permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def reconstruction_psnr(model: nn.Module, idx: np.ndarray, lut: torch.Tensor,
                        batch: int = 256) -> tuple[float, float]:
    """(PSNR in dB on uint8, mean squared error in [0,1] units) over `idx`.

    Rounded to uint8 before the error is taken, because that is what the pipeline
    delivers and what item 6 will hand the validator. PSNR computed on the raw
    float output reads ~0.01 dB better and the pipeline never produces it. The
    float MSE comes back alongside only so the training loss and the gate number
    can be read on one line.
    """
    was_training = model.training
    model.eval()
    sse = 0.0
    mse01 = 0.0
    for i in range(0, len(idx), batch):
        x8 = _batch(idx, lut, np.arange(i, min(i + batch, len(idx))))
        y = model(x8 / PEAK)
        y8 = (y * PEAK).round().clamp(0, PEAK)
        sse += float((y8 - x8).pow(2).sum())
        mse01 += float((y - x8 / PEAK).pow(2).sum())
    model.train(was_training)
    values = idx.size * 3
    return psnr_db(sse, values), mse01 / values


def _keep_awake() -> None:
    """Ask Windows not to suspend the machine while a rung trains.

    A 60-epoch rung is ~90 min, and on 2026-08-29 this laptop entered Modern
    Standby mid-run and froze epoch 8 for **49 minutes** - Windows event log
    `Kernel-Power` 506 at 01:07:29 and 507 at 01:56:46, against a measured
    3050.7 s epoch where every neighbour took 77-98 s. The wall clock keeps
    counting through a suspend, so the run survives but every timing it reports
    is void.

    `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` is a *per-process* request that lapses
    when the process exits. It changes no user setting and no power plan.

    ponytail: no-op off Windows, and a failure to get it is not worth losing a
    run over - the run is still correct, only its wall clock is suspect.
    """
    if sys.platform != "win32":
        return
    try:
        # ES_CONTINUOUS 0x80000000 | ES_SYSTEM_REQUIRED 0x00000001
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            print("  (keep-awake refused; a standby would void this run's timings)")
    except (AttributeError, OSError) as e:
        print(f"  (keep-awake unavailable: {e}; a standby would void the timings)")


def train(rung: str, cfg: config.Config, levels=(8, 8, 8), attention: bool = False,
          quantize: bool = True, epochs: int = 15, batch: int = 128, lr: float = 3e-4,
          lr_floor: float = 3e-5, weight_decay: float = 1e-4, warmup: float = 0.05,
          seed: int = 0, eval_frames: int = 4096, log_every: int = 100,
          device: str | None = None, resume: str | None = None) -> dict:
    """One rung. Plain MSE, AdamW, cosine to `lr_floor` after linear warmup.

    Plain MSE and nothing else: PSNR is a monotone function of MSE, so the loss
    *is* the gate. Per-pixel 7-way cross-entropy was considered and loses - the
    mean squared distance between two distinct palette entries is 47,814, so
    classification needs ~99.6% pixel accuracy to clear 30 dB while regression
    can hedge with a blend.

    The hedging is a real hole: MSE rewards blurring edges, and 99.95% of the
    k-means floor's error lives in the 36.53% of patches that are not flat. The
    counterweight is item 6 - `offpalette_px` on reconstructions punishes exactly
    the blur PSNR rewards, and the two cannot both be gamed. Neither number means
    much alone.

    **fp32, no autocast.** Under a million parameters and a few hundred MB of
    activations at batch 128, so fp32 is free here, and it removes a class of
    numerical doubt from the one number the whole phase turns on. Add autocast
    only if a measured step time asks for it.

    **It asked on 2026-08-29, and the answer is still no - for a reason that is
    about comparability, not about speed.** bf16 autocast was measured on this
    card at both resolutions, same model, same batch 128:

    | | fp32 | bf16 | 60-epoch rung |
    |---|---|---|---|
    | 64x64 R1 | 37.7 ms/step | 25.8 | 1.39 h -> 0.95 |
    | 64x64 R2 | 40.2 ms/step | 28.6 | 1.48 h -> 1.06 |
    | 96x96 R1 | 75.8 ms/step | 60.4 | 2.80 h -> 2.23 |
    | 96x96 R2 | 85.1 ms/step | 68.3 | 3.14 h -> 2.52 |

    So 1.4-1.5x at 64x64 and only **1.25x at 96x96** - bf16 accelerates the
    tensor-core matmuls and not the `nn.Upsample` / `GroupNorm` / `SiLU` chain,
    which is bandwidth-bound, and the bigger image shifts the mix toward the part
    it cannot help. TF32 alone is 1.07x and `cudnn.benchmark` is worth nothing
    here, the shapes being fixed. **Not adopted**, because R1 and R2 at 60 epochs
    differ by **0.087 dB** and are the comparison item 5 rests on; changing the
    arithmetic underneath makes every later rung incomparable to both. Whoever
    starts the 96x96 arm should decide it there, where a rung costs 3 h, and pay
    for it by re-running one baseline rather than by assuming bf16 is neutral.

    **What is *not* free to assume: this loop is not bit-reproducible.** Two
    1-epoch r1 runs at seed 0, same machine, nothing else changed, read **25.66625
    and 25.66792 dB** - 0.00167 dB apart, from nondeterministic cuDNN backward
    reductions (`torch.use_deterministic_algorithms` is not set, and setting it
    would cost throughput for a property nothing here needs). That is the first
    measurement of a quantity `AGENDA.md` correctly flags as never taken. It is a
    **1-epoch** figure and a lower bound on the 60-epoch spread, so it does not
    by itself license calling 0.087 dB significant - but it does put the noise
    two orders of magnitude below it rather than nowhere.

    The measured data path, for the same reason: `_batch` is **0.47 ms of a 40 ms
    step, 1.2%**. A pinned staging buffer takes it to 0.20 ms and buys nothing.
    `non_blocking=True` in `_batch` is a documented no-op on pageable memory.
    Do not spend time there.

    Every hyperparameter default is a *starting point, not a measurement* - the
    plan says so explicitly and they are expected to move.
    """
    _keep_awake()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])

    t0 = time.perf_counter()
    train_idx, lut_np = data.preload(shards, index, "train",
                                     cfg.data["val_fraction"], palette.rgb)
    val_idx, lut_val = data.preload(shards, index, "val",
                                    cfg.data["val_fraction"], palette.rgb)
    assert np.array_equal(lut_np, lut_val), "the two splits disagree on the palette LUT"
    lut = torch.from_numpy(lut_np).to(dev).float()
    load_s = time.perf_counter() - t0

    model = Tokenizer(levels, attention=attention, quantize=quantize).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_idx) // batch
    total = steps_per_epoch * epochs
    warm = max(1, int(warmup * total))

    def lr_at(step: int) -> float:
        """Linear warmup then cosine. The warmup is cheap insurance: a cold
        start can drive the bottleneck `tanh` straight into saturation."""
        if step < warm:
            return lr * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return lr_floor + 0.5 * (lr - lr_floor) * (1 + math.cos(math.pi * p))

    hashes = {"data_hash": cfg.data_hash, "tokenizer_hash": cfg.tokenizer_hash}
    knobs = dict(rung=rung, levels=list(levels), attention=attention, quantize=quantize,
                 epochs=epochs, batch=batch, lr=lr, lr_floor=lr_floor,
                 weight_decay=weight_decay, warmup=warmup, seed=seed,
                 image_size=list(cfg.shapes.image_size),
                 token_grid=list(cfg.shapes.token_grid), params=params,
                 train_frames=len(train_idx), val_frames=len(val_idx),
                 steps_per_epoch=steps_per_epoch, device=str(dev))

    print(f"{rung}: {params:,} parameters, {len(train_idx):,} train / "
          f"{len(val_idx):,} val frames, {steps_per_epoch} steps/epoch x {epochs}, "
          f"preload {load_s:.1f}s")
    print(f"  levels {list(levels)} quantize={quantize} attention={attention} on {dev}")

    # A fixed train subsample, so the train-val gap compares two numbers
    # measured the same way rather than a moving loss average against a
    # full-split eval.
    train_eval = np.sort(rng.choice(len(train_idx),
                                    size=min(eval_frames, len(train_idx)), replace=False))

    start_epoch = 0
    if resume is not None:
        ck = torch.load(ROOT / "runs" / resume / "model.pt", map_location=dev,
                        weights_only=False)
        # Every knob that changes the computation being resumed. `lr_floor`,
        # `warmup` and `weight_decay` were missing from this list until
        # 2026-08-29, and all three are load-bearing: the first two are read by
        # `lr_at` on every step and the third by AdamW, so resuming with a
        # different one silently changed the schedule mid-run while the flag's
        # own help text promised "every knob must match". Not reachable from the
        # CLI, which exposes none of the three - but `train()` is called directly
        # by anything driving a ladder from Python, which is how R1 and R2 ran.
        #
        # `rung`, `device` and the derived entries (`params`, `steps_per_epoch`,
        # the frame counts) are deliberately still unchecked: resuming onto a
        # second machine is a case this has to keep allowing, and those differ
        # without changing the computation.
        for k in ("levels", "attention", "quantize", "batch", "lr", "lr_floor",
                  "warmup", "weight_decay", "epochs", "seed"):
            assert k in ck["knobs"], (
                f"resume {resume}: its checkpoint carries no {k!r}, so it predates "
                f"this check and cannot be shown to match - retrain, do not resume"
            )
            assert ck["knobs"][k] == knobs[k], (
                f"resume {resume}: {k} is {ck['knobs'][k]}, this call asks for {knobs[k]}"
            )
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]
        # `.cpu()` is load-bearing, not defensive. `torch.load(map_location=dev)`
        # above moves **every** tensor in the checkpoint to the GPU, the two RNG
        # states included, and `set_rng_state` takes a CPU ByteTensor only - so
        # both calls raised `TypeError: RNG state must be a torch.ByteTensor` and
        # `--resume` could not survive its own first line on CUDA.
        #
        # Found 2026-08-29, by writing the first test that ever called it. The
        # path was added after the `nn.Upsample` crash to make a mid-flight death
        # cost one epoch instead of ninety minutes, was never exercised, and
        # would have failed at the moment it was finally needed - the R1 60-epoch
        # rerun started from scratch at epoch 0 rather than resuming the run that
        # had just died at epoch 6.
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] is not None and dev.type == "cuda":
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        train_eval = ck["train_eval"]
        start_epoch = ck["epoch"] + 1
        print(f"  resumed {resume} at epoch {start_epoch}/{epochs}")

    with Run(rung, hashes, config=knobs) as run:
        step = start_epoch * steps_per_epoch
        wall = time.perf_counter()
        for epoch in range(start_epoch, epochs):
            order = rng.permutation(len(train_idx))
            for s in range(steps_per_epoch):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                x = _batch(train_idx, lut, order[s * batch:(s + 1) * batch]) / PEAK
                loss = F.mse_loss(model(x), x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                step += 1
                if step % log_every == 0:
                    run.log({"step": step, "epoch": epoch, "loss": float(loss.detach()),
                             "lr": lr_at(step),
                             "wall_s": round(time.perf_counter() - wall, 1)})

            val_db, val_mse = reconstruction_psnr(model, val_idx, lut)
            tr_db, _ = reconstruction_psnr(model, train_idx[train_eval], lut)
            run.log({"step": step, "epoch": epoch, "val_psnr_db": val_db,
                     "train_psnr_db": tr_db, "gap_db": tr_db - val_db,
                     "val_mse01": val_mse,
                     "wall_s": round(time.perf_counter() - wall, 1)})
            print(f"  epoch {epoch + 1:>2}/{epochs}  val {val_db:6.3f} dB  "
                  f"train {tr_db:6.3f} dB  gap {tr_db - val_db:+.3f}  "
                  f"{time.perf_counter() - wall:6.1f}s")
            # Overwrite every epoch, not only at the end, and carry enough to
            # *resume* rather than only to evaluate. Two runs have now been lost
            # mid-flight to the `nn.Upsample` frame-corruption crash recorded in
            # the verification log, which is a native-layer fault this code
            # cannot prevent - so the answer is to make it cost one epoch
            # instead of ninety minutes. 4 MB and ~40 ms.
            #
            # The RNG states are saved rather than the epoch reseeded, because
            # reseeding per epoch would change the data order and make a resumed
            # rung incomparable to the rungs already measured - and the whole
            # point of R1 here is a 0.06 dB comparison against R2.
            torch.save({"state_dict": model.state_dict(), "knobs": knobs, **hashes,
                        "epoch": epoch, "opt": opt.state_dict(),
                        "np_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": (torch.cuda.get_rng_state()
                                     if dev.type == "cuda" else None),
                        "train_eval": train_eval},
                       run.dir / "model.pt")

        val_db, val_mse = reconstruction_psnr(model, val_idx, lut)
        tr_db, _ = reconstruction_psnr(model, train_idx[train_eval], lut)
        out = dict(knobs, run_id=run.run_id, val_psnr_db=val_db, train_psnr_db=tr_db,
                   gap_db=tr_db - val_db, val_mse01=val_mse,
                   train_s=round(time.perf_counter() - wall, 1), **hashes)
        # `epoch` too, because the final record also matches a `val_psnr_db`
        # filter and a reader that groups by epoch would trip over its absence.
        run.log({"final": True, "epoch": epochs - 1, **out})
        torch.save({"state_dict": model.state_dict(), "knobs": knobs, **hashes},
                   run.dir / "model.pt")
        (run.dir / "result.json").write_text(json.dumps(out, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    return out


# ----------------------------------------------------------------- self-check

def _self_check() -> None:
    """5a and 5b without touching data, per the plan's working-when.

    The gradient row is the load-bearing one: 0.858 / 1.001 / 0.668 are recorded
    measurements, and reproducing all three at once pins `eps`, `offset`, `shift`
    and the normalisation simultaneously. `codes_to_indices` and its bijection
    are not here - nothing needs them until R1.
    """
    torch.manual_seed(0)
    z = torch.linspace(-25, 25, 20001)

    for levels, recorded in (((8, 8, 8), 0.858), ((8, 6, 5), None),
                             ((5, 5, 5), 1.001), ((4, 4, 4), 0.668)):
        q = FSQ(levels)
        codes = q(z.view(-1, 1, 1, 1).expand(-1, len(levels), 1, 1))
        for c, n in enumerate(levels):
            got = torch.unique(codes[:, c]).numel()
            assert got == n, f"levels {levels} dim {c}: {got} distinct values, expected {n}"
        # The bijection 5a's working-when names. `unique` comes back sorted, so
        # the cartesian product is every code tuple exactly once; the ids must
        # then be 0..prod(levels)-1 with nothing missing and nothing doubled.
        # Asserting on the *sorted* ids checks injective and onto together - a
        # wrong place value would still produce prod(levels) values, just not
        # those ones.
        vals = [torch.unique(codes[:, c]) for c in range(len(levels))]
        tuples = torch.cartesian_prod(*vals) if len(levels) > 1 else vals[0][:, None]
        ids = q.codes_to_indices(tuples.T.reshape(1, len(levels), -1, 1)).flatten()
        assert torch.equal(ids.sort().values, torch.arange(math.prod(levels))), \
            f"levels {levels}: codes_to_indices is not a bijection onto 0..{math.prod(levels) - 1}"

        z0 = torch.zeros(1, len(levels), 1, 1, requires_grad=True)
        q(z0).sum().backward()
        assert z0.grad is not None, f"levels {levels}: backward produced no gradient"
        g = z0.grad.flatten()
        assert torch.isfinite(g).all(), f"levels {levels}: non-finite STE gradient"
        note = ""
        if recorded is not None:
            assert abs(float(g[0]) - recorded) < 5e-4, \
                f"levels {levels}: STE gradient {float(g[0]):.4f}, recorded {recorded}"
            note = f", matches the recorded {recorded}"
        print(f"FSQ {list(levels)}: {math.prod(levels)} codes, "
              f"{list(levels)} distinct values per dim, indices bijective onto "
              f"0..{math.prod(levels) - 1}, STE gradient at 0 "
              f"{[round(float(v), 4) for v in g]}{note}")

    cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
    assert math.prod((8, 8, 8)) == cfg.tokenizer["codebook_size"], \
        "prod(levels) disagrees with tokenizer.codebook_size"
    print(f"[8,8,8] gives {math.prod((8, 8, 8))} codes == tokenizer.codebook_size "
          f"{cfg.tokenizer['codebook_size']}")

    h, w = cfg.shapes.image_size
    grid = tuple(cfg.shapes.token_grid)
    x = torch.rand(2, 3, h, w)
    for attention in (False, True):
        for quantize in (False, True):
            m = Tokenizer((8, 8, 8), attention=attention, quantize=quantize)
            with torch.no_grad():
                zed = m.encoder(x)
                y = m(x)
            assert tuple(zed.shape) == (2, 3) + grid, \
                f"latent {tuple(zed.shape)}, expected (2, 3) + {grid}"
            assert y.shape == x.shape, f"output {tuple(y.shape)} != input {tuple(x.shape)}"
            assert torch.isfinite(y).all(), "non-finite reconstruction"
            n = sum(p.numel() for p in m.parameters())
            print(f"Tokenizer attention={attention} quantize={quantize}: {n:,} params, "
                  f"{tuple(x.shape[1:])} -> {tuple(zed.shape[1:])} -> {tuple(y.shape[1:])}, "
                  f"output [{float(y.min()):+.3f}, {float(y.max()):+.3f}]")

    # No tanh and no clamp on the output, so an untrained decoder must be free to
    # leave [0, 1]. A range pinned inside it would mean an activation crept in.
    wide = Tokenizer((8, 8, 8))
    out_conv = wide.decoder[-1]
    assert isinstance(out_conv, nn.Conv2d) and out_conv.bias is not None, \
        "the decoder does not end in a biased conv - something follows it"
    with torch.no_grad():
        out_conv.bias.fill_(3.0)
        reach = float(wide(x).max())
    assert reach > 1.0, "the output is clamped or squashed somewhere"
    print(f"output is unbounded: a +3.0 output bias reaches {reach:.3f}, "
          f"so no tanh and no clamp")

    for attention in (False, True):
        m = Tokenizer((8, 8, 8), attention=attention, quantize=True)
        m(x).sum().backward()
        grads = {name: p.grad for name, p in m.named_parameters()}
        missing = sorted(name for name, g in grads.items() if g is None)
        assert not missing, f"the STE did not reach {missing}"
        assert all(torch.isfinite(g).all() for g in grads.values() if g is not None), \
            "a parameter gradient is non-finite"
        n = len(grads)
        print(f"attention={attention}: gradient reaches all {n} parameter tensors "
              f"through the quantizer, all finite")

    # 5e rides entirely on `encode`, so check the three things a token cache
    # needs from it before spending a pass over 300,000 frames finding out.
    tok = Tokenizer((8, 8, 8), quantize=True)
    tok.eval()
    ids = tok.encode(x)
    assert ids.shape == (len(x),) + grid, f"encode gave {tuple(ids.shape)}, expected {(len(x),) + grid}"
    assert 0 <= int(ids.min()) and int(ids.max()) < 512, \
        f"token ids span [{int(ids.min())}, {int(ids.max())}], outside 0..511"
    assert torch.equal(ids, tok.encode(x)), "encode is not deterministic on identical input"
    try:
        Tokenizer((8, 8, 8), quantize=False).encode(x)
        raise AssertionError("encode accepted a continuous bottleneck - R0 would emit fake tokens")
    except ValueError:
        pass
    print(f"encode: ids {tuple(ids.shape)} in [{int(ids.min())}, {int(ids.max())}], "
          f"deterministic, and refuses a continuous bottleneck")

    # A hand-computable case: total SSE 1.0 over 100 values, so MSE = 1/100.
    assert abs(psnr_db(1.0, 100) - 10 * math.log10(255 * 255 * 100)) < 1e-9
    print("psnr_db agrees with 10*log10(255^2 / MSE) by hand")
    print("fsq self-check ok (5a, 5b, and encode; no data touched)")


def main() -> None:
    ap = argparse.ArgumentParser(description="the FSQ tokenizer: self-check, or train a ladder rung")
    ap.add_argument("--run", choices=sorted(RUNGS),
                    help="train a rung: r0 continuous, r1 quantized, r2 quantized + attention")
    ap.add_argument("--tokens", metavar="RUN_ID",
                    help="write the token cache for an already-trained run")
    ap.add_argument("--eval", metavar="RUN_ID",
                    help="print the gate table for a run whose token cache exists")
    ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "base.json"))
    ap.add_argument("--levels", type=int, nargs="+", default=[8, 8, 8],
                    help="the FSQ levels table; ignored by r0, which does not quantize")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="continue a run killed mid-flight, from its last per-epoch "
                         "checkpoint; every knob must match the run being resumed")
    args = ap.parse_args()

    if args.run is None and args.tokens is None and args.eval is None:
        _self_check()
        return

    cfg = config.load(args.config)

    # Imported here, not at module level: `fsq_eval` imports this file, so a
    # top-level import back would be a cycle. Function-local is the boring fix.
    if args.tokens is not None:
        from mirage.fsq_eval import write_token_cache
        write_token_cache(args.tokens, cfg, batch=args.batch)
        if args.eval is None:
            return

    if args.eval is not None:
        from mirage.fsq_eval import evaluate
        # Nonzero exit when a pass/fail row misses, so the gate is usable from a
        # script and not only by reading the table.
        raise SystemExit(1 if evaluate(args.eval, cfg)["failed_rows"] else 0)

    assert args.run is not None
    rung = RUNGS[args.run]
    out = train(args.run, cfg, levels=tuple(args.levels), epochs=args.epochs,
                batch=args.batch, lr=args.lr, seed=args.seed, resume=args.resume,
                quantize=rung["quantize"], attention=rung["attention"])
    db = out["val_psnr_db"]
    # Row 1 and row 2 of the gate, the two that can disagree. Row 1 passing while
    # row 2 fails means the val split got easier, not that the model got better -
    # which is why both are printed rather than the headline alone.
    print(f"\n{args.run.upper()} {out['run_id']}: held-out {db:.3f} dB")
    print(f"  row 1  vs the {PSNR_BAR_DB} dB Q-1 bar:            {db - PSNR_BAR_DB:+.3f} dB")
    print(f"  row 2  vs the {KMEANS_FLOOR_DB} dB held-out floor: {db - KMEANS_FLOOR_DB:+.3f} dB "
          f"(needs >= {PSNR_BAR_DB - KMEANS_FLOOR_DB:+.2f})")
    print(f"  train-val gap {out['gap_db']:+.3f} dB, {out['train_s']}s")


if __name__ == "__main__":
    main()
