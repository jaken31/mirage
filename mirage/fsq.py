"""The tokenizer: FSQ quantizer, conv encoder/decoder, training loop, PSNR eval.

An *autoencoder*. The encoder squeezes a 64x64 RGB frame to an 8x8 grid of
`len(levels)` numbers per cell; the decoder rebuilds the frame from that grid.
FSQ (finite scalar quantization) makes the grid discrete by rounding each number
to one of a fixed set of levels, so `prod(levels)` is the vocabulary and there is
no learned dictionary that could collapse. Rounding has zero gradient, so the
backward pass uses a straight-through estimator: it acts as if the rounding were
not there.

Contents, in the order of `docs/phase1_structural_plan.md` section 5:

- `FSQ` - the quantizer, and `codes_to_indices`.
- `Tokenizer` - encoder, optional 8x8 self-attention, decoder.
- `train` - MSE loss, AdamW, cosine schedule, held-out PSNR.
- `RUNGS` - the experiment ladder. Each "rung" is one training run that
  differs from the others by one or two flags: R0 (no quantization), R1
  (quantized), R2 (R1 plus attention), and two normalisation variants.
- Token caching and the pass/fail "gate table" live in `fsq_eval.py`. This
  file builds and trains; that one reads a trained run.

There is no R3 rung on purpose: it could mean residual blocks, wider channels,
or different levels, which are three different runs, and R2's result decides
which one is needed.

    python -m mirage.fsq              # self-check, touches no data
    python -m mirage.fsq --run r0     # continuous bottleneck - the ceiling
    python -m mirage.fsq --run r1     # FSQ [8,8,8], no attention
    python -m mirage.fsq --run r2     # R1 plus attention on the 8x8 grid
    python -m mirage.fsq --tokens ID  # encode all 300,000 frames from that run
    python -m mirage.fsq --eval ID    # the gate table

PSNR (peak signal-to-noise ratio) is `10*log10(255^2 / MSE)` on **uint8**
reconstructions, not the raw float output. The float value is ~0.01 dB better,
but the pipeline never produces floats, so it is not the number that counts.
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

# The reconstruction target: held-out PSNR of at least 30 dB.
PSNR_BAR_DB = 30.0

# The baseline a tokenizer must beat: a plain 512-entry k-means codebook over
# 8x8 pixel patches, fit on the training episodes and scored on the held-out
# ones, the same treatment a tokenizer gets (`bench/patch_probe.py`). Scoring
# it on a sample that mixed the two splits once gave a misleadingly higher
# number.
#
# **Keyed by resolution, because the baseline depends on the frames.** A rung is
# compared with the baseline at *its own* resolution; comparing a 96x96 rung
# with the 64x64 baseline would look like a free win. Not in `config.json`: it is
# a measurement, not a setting, and a new key in the `tokenizer` section would
# change `tokenizer_hash` and orphan every checkpoint on disk.
KMEANS_FLOOR_DB: dict[tuple[int, int], float] = {
    (64, 64): 28.27,
    # Same probe, same 179,200-patch budget. **Higher than at 64x64, not
    # lower**: an 8x8 patch covers 2.25x less of the scene at 96x96, so 73% of
    # patches are one flat colour (against 63% at 64x64) and a patch codebook
    # finds the frames easier. So at 96x96 the baseline alone almost reaches
    # 30 dB, the "beat the baseline" row says little, and the plain 30 dB row
    # carries the whole question.
    (96, 96): 29.97,
}


def kmeans_floor_db(cfg: "config.Config") -> float:
    """The recorded held-out k-means-512 baseline for this config's resolution.

    Raises instead of extrapolating. Numbers guessed by scaling with image area
    are exactly the kind this project has repeatedly had to retract.
    """
    size = tuple(cfg.shapes.image_size)
    if size not in KMEANS_FLOOR_DB:
        raise KeyError(
            f"no recorded k-means floor at {size[0]}x{size[1]}. Measure it with "
            f"`python bench/patch_probe.py --config <cfg>` and record the "
            f"held-out 512-centroid uint8 PSNR here - do not scale the 64x64 one")
    return KMEANS_FLOOR_DB[size]

# Each rung answers one question, and only these flags vary between them. No
# R3 on purpose; see the module docstring.
RUNGS = {
    "r0": dict(quantize=False, attention=False),  # the architecture's ceiling
    "r1": dict(quantize=True, attention=False),   # what quantization costs
    "r2": dict(quantize=True, attention=True),    # what joint coding buys
    # R1 without whole-image statistics in the encoder. GroupNorm normalises over
    # (channel group, H, W), so every token depends on every pixel of the frame:
    # measured with autograd, one token's gradient reaches all 4,096 px with it
    # and exactly 15x15 without it. That explains spurious token changes (a token
    # changing although its own 15x15 patch did not), and this rung measures what
    # removing it costs.
    "r1c": dict(quantize=True, attention=False, encoder_norm="channel"),
    # Halfway between the two, and it is the only in-between point available. A
    # KxK normalisation window widens each stage's reach by (K-1) at that
    # stage's resolution, so a token sees 15 + 2*(K-1)*(4+2+1) px: K=3 gives
    # 43x43 = 1,849, and K=5 gives 71, already wider than the 64-px frame.
    # Measured, and it agrees.
    "r1w3": dict(quantize=True, attention=False, encoder_norm="local3"),
}

# GroupNorm's group count must divide every channel count. 8 divides 64, 128
# and 256, and the exact value does not matter for a network this small. Named
# so it is not eight unexplained 8s.
GN_GROUPS = 8


# ----------------------------------------------------------------------- FSQ

class FSQ(nn.Module):
    """Finite scalar quantization, one levels count per latent channel.

        half_l   = (levels - 1) * (1 + eps) / 2
        offset   = 0.5 for an even levels count, else 0
        shift    = atanh(offset / half_l)
        bound(z) = tanh(z + shift) * half_l - offset
        q        = round(bound(z)) via straight-through, then / (levels // 2)

    `eps` widens the range slightly past the outermost level so the `tanh`
    limit does not sit exactly on it. `offset` and `shift` re-centre an even
    level count, whose levels sit either side of zero instead of including it.

    **No extra losses.** No commitment loss, codebook loss, moving averages or
    dead-code resets. FSQ has no dictionary to maintain, and adding one of these
    would undo why it was chosen over VQ (vector quantization): better codebook
    usage could no longer be told apart from "the extra loss propped it up". If
    codebook usage is too low, shrink the vocabulary instead.

    The straight-through gradient is **not 1.0**: it skips only the rounding, so
    the `tanh` slope still applies. At zero it is 0.858 for [8,8,8], 1.001 for
    [5,5,5] and 0.668 for [4,4,4]. So changing levels silently rescales the
    bottleneck's effective learning rate by up to 1.5x, which is why every
    levels comparison is run at two learning rates. `_self_check` reproduces
    those three numbers.
    """

    # Declared here so type checkers treat the buffers below as plain tensors.
    # Without these, pyright types every `self.half_l` as
    # `Tensor | Module | None`.
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
        # Buffers, not plain tensors: they must move with .to(device) and be
        # saved in the checkpoint, so a reloaded model quantizes identically.
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

        `forward` returns `round(bound(z)) / (levels//2)`, a value scaled to
        roughly [-1, 1] because that is what the decoder takes. This undoes that
        scaling: `q * scale + scale` maps the `levels[c]` values onto digits
        `0..levels[c]-1`. The digits are then combined like a number in a mixed
        base: channel 0 is the ones place, channel 1 the `levels[0]`s place, and
        so on.

        Every id in `0..prod(levels)-1` has exactly one such digit expansion, so
        the mapping is one-to-one with no dictionary and no collisions.
        `_self_check` runs every code combination and asserts the ids come back
        as `arange`. That catches a wrong un-scaling, which would otherwise make
        negative digits that wrap into wrong-but-valid ids, only noticed much
        later as a corrupt token cache.

        The range check is per channel, not against `max(levels)`: a mixed table
        like [8,6,5] has three digit ranges, and a single bound would let a
        digit 7 through in the 6-level channel.

        The place values are rebuilt on every call instead of stored as a buffer
        (two tiny tensors, so free). A new buffer would change `state_dict` and
        make existing R0 checkpoints fail a strict load.
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


# -------------------------------------------------------- encoder and decoder

class ChannelNorm(nn.Module):
    """`GroupNorm` with statistics taken **per pixel**, so no mixing across space.

    Same groups, same learned per-channel scale and shift, same eps. The only
    change: mean and variance are over the group's channels at one (h, w),
    not over the group's channels *and the whole feature map*. So a cell's
    output no longer depends on pixels outside its conv window, which is what
    rung `r1c` tests.

    ponytail: written out instead of using `F.group_norm`, which would need a
    permute and a copy of the activation to give per-pixel statistics. This is
    a view plus two reductions.
    """

    def __init__(self, groups: int, ch: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert ch % groups == 0, f"{ch} channels do not split into {groups} groups"
        self.groups, self.eps = groups, eps
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = x.view(b, self.groups, c // self.groups, h, w)
        g = (g - g.mean(2, keepdim=True)) * (
            g.var(2, unbiased=False, keepdim=True) + self.eps).rsqrt()
        g = g.view(b, c, h, w)
        return g * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class LocalNorm(nn.Module):
    """In between: statistics over a `window` x `window` patch around each pixel.

    `ChannelNorm` and `GroupNorm` are the two extremes (window 1 is per-pixel,
    a window covering the whole map is `GroupNorm`), so this is one dial
    between rungs `r1c` and `r1`, not a third idea. Rung `r1w3` uses it.

    The patch average is `avg_pool2d` at stride 1 with
    `count_include_pad=False`, so an edge pixel averages over the neighbours it
    really has, not over zero padding. Variance is `E[x^2] - E[x]^2` over the
    group's channels times the window: the same values `GroupNorm` uses, just
    restricted to the patch.

    ponytail: in fp32 that variance can come out slightly negative when the
    patch is nearly constant, which the black void band always is, so it is
    clamped at 0 before adding eps. Without the clamp `rsqrt` returns NaN and
    the run dies in its first epoch.
    """

    def __init__(self, groups: int, ch: int, window: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert ch % groups == 0, f"{ch} channels do not split into {groups} groups"
        assert window % 2 == 1 and window >= 1, f"window {window} must be odd and >= 1"
        self.groups, self.window, self.eps = groups, window, eps
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))

    def _box(self, t: torch.Tensor) -> torch.Tensor:
        if self.window == 1:
            return t
        return F.avg_pool2d(t, self.window, 1, self.window // 2,
                            count_include_pad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = x.view(b, self.groups, c // self.groups, h, w)
        # Average over the group's channels first, then over the patch. Same
        # result as one joint average, but pools `groups` maps instead of `c`.
        m = self._box(g.mean(2))
        v = (self._box(g.pow(2).mean(2)) - m.pow(2)).clamp_min(0)
        g = (g - m.unsqueeze(2)) * (v + self.eps).rsqrt().unsqueeze(2)
        g = g.view(b, c, h, w)
        return g * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _norm(kind: str, ch: int) -> nn.Module:
    """`group`, `channel`, or `localK` for a K x K window - see `LocalNorm`."""
    if kind == "group":
        return nn.GroupNorm(GN_GROUPS, ch)
    if kind == "channel":
        return ChannelNorm(GN_GROUPS, ch)
    if kind.startswith("local"):
        return LocalNorm(GN_GROUPS, ch, int(kind[len("local"):]))
    raise ValueError(
        f"unknown normalisation {kind!r}, expected 'group', 'channel' or 'localK'")


def _stage(cin: int, cout: int, stride: int, norm: str = "group") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1),
        _norm(norm, cout),
        nn.SiLU(),
    )


def _up(cin: int, cout: int) -> nn.Sequential:
    """Nearest-neighbour upsample plus a 3x3 conv. **Never `ConvTranspose2d`.**

    Transposed convolutions cause checkerboard artifacts that look like
    misplaced edges, and misplaced edges are exactly the signal used to decide
    whether to move from 64x64 (64 tokens) to 96x96 (144 tokens). A checkerboard
    would send the project to 96x96 on a false diagnosis.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, 3, 1, 1),
        nn.GroupNorm(GN_GROUPS, cout),
        nn.SiLU(),
    )


class GridAttention(nn.Module):
    """One single-head self-attention layer over the 8x8 latent grid.

    64 positions, so the attention matrix is 64x64 and costs nothing. It is the
    only way the 64 codes can describe the frame *together* rather than each
    patch on its own. Patches on their own were measured: the k-means patch
    baseline (see `KMEANS_FLOOR_DB`) falls short of the 30 dB target by 1.73 dB,
    and that gap is what shared context would have to close.

    **Measured: it does not improve quality, but it does improve token usage**
    (R1 and R2 both trained to convergence at 60 epochs).

    For quality it adds **+0.087 dB** for **+263,680 parameters**, about a sixth
    of the plan's own "within ~0.5 dB means tied" rule, and **R1 passes every
    gate row without it**. (An earlier claim here, that its value hid inside run
    to run noise, was wrong; that noise had never been measured.)

    Where it helps is token entropy (how evenly the vocabulary is used): **+3.5
    points** at convergence and +8.2 at 15 epochs, by making the three FSQ digits
    less redundant (1.339 -> 0.781 bits of redundancy at 15 epochs). Longer
    training does the same job instead: without attention, 60 epochs reaches
    0.890 bits on its own.

    It also has a determinism caveat R1 does not: with attention, re-encoding a
    shard at a different batch size changes about 2 tokens in 100,000. See
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
        # (b, 1, hw, c): one head, so the head axis has size 1.
        heads = (t.transpose(1, 2).unsqueeze(1) for t in (q, k, v))
        a = F.scaled_dot_product_attention(*heads)
        return x + self.proj(a.squeeze(1).transpose(1, 2).reshape(b, c, h, w))


class Tokenizer(nn.Module):
    """Channels 3 -> 64 -> 128 -> 256 going down, 1x1 conv to the latent, then mirrored back up.

    Stride 8 means exactly three stride-2 stages. `GroupNorm` + `SiLU`, no
    residual blocks: those are the first thing to add if a rung falls short,
    not a starting assumption.

    The final conv is **linear**: no output activation and no clamp. `tanh`
    would saturate on exactly the colours this scene is made of (pure black
    void, saturated blocks), and clamping removes the gradient that punishes
    overshoot. Clamp only when converting to uint8.

    `quantize=False` is rung R0: the same network with no rounding in the
    middle. It measures the best this architecture can do, which the quantized
    rungs are then compared against.
    """

    def __init__(self, levels=(8, 8, 8), attention: bool = False,
                 quantize: bool = True, width: int = 64,
                 encoder_norm: str = "group") -> None:
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4
        d = len(levels)
        self.fsq = FSQ(levels)
        self.quantize = quantize
        self.encoder = nn.Sequential(
            _stage(3, c1, 2, encoder_norm), _stage(c1, c2, 2, encoder_norm),
            _stage(c2, c3, 2, encoder_norm),
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

        Refuses an unquantized (R0) model instead of rounding its output anyway.
        R0 was never trained with rounding, so its ids would look valid and mean
        nothing, giving a token dataset for the next phase that looks fine but
        is not.
        """
        if not self.quantize:
            raise ValueError(
                "this model has a continuous bottleneck (R0) and has no token ids"
            )
        return self.fsq.codes_to_indices(self.fsq(self.encoder(x)))


# ------------------------------------------------------ loss, loop, and PSNR

def psnr_db(sse: float, values: int) -> float:
    return 10.0 * math.log10(PEAK * PEAK / (sse / values))


def _batch(idx: np.ndarray, lut: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
    """Palette indices -> (B, 3, H, W) of 0..255 floats on the lookup table's device.

    The indices stay in CPU RAM (1.16 GB for the training split, against ~5 GB
    free on an 8 GB card that also holds activations). One batch is 512 KB, so
    copying it is negligible next to a training step. Expanding indices to
    colours runs on the GPU because it is a lookup into 7 rows.
    """
    rows_u8 = np.ascontiguousarray(idx[rows])
    b = torch.from_numpy(rows_u8).to(lut.device, non_blocking=True).long()
    return lut[b].permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def reconstruction_psnr(model: nn.Module, idx: np.ndarray, lut: torch.Tensor,
                        batch: int = 256) -> tuple[float, float]:
    """(PSNR in dB on uint8, mean squared error in [0,1] units) over `idx`.

    Rounded to uint8 before measuring the error, because that is what the
    pipeline delivers and what the validator sees. PSNR on the raw float output
    reads ~0.01 dB better, but nothing downstream gets floats. The float MSE is
    returned only so the training loss and the gate number can be compared
    side by side.
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
    """Ask Windows not to put the machine to sleep while a rung trains.

    A 60-epoch rung takes about 90 min, and this laptop once went into standby
    mid-run and froze one epoch for **49 minutes** (every other epoch took
    77-98 s). The clock keeps running during sleep, so the run survives but all
    its timings are meaningless.

    `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` is a *per-process* request that ends
    when the process exits. It changes no user setting and no power plan.

    ponytail: does nothing off Windows, and failing to get it is not worth
    losing a run over: the results are still correct, only the timings are
    suspect.
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
          quantize: bool = True, encoder_norm: str = "group", epochs: int = 15, batch: int = 128, lr: float = 3e-4,
          lr_floor: float = 3e-5, weight_decay: float = 1e-4, warmup: float = 0.05,
          seed: int = 0, eval_frames: int = 4096, log_every: int = 100,
          device: str | None = None, resume: str | None = None) -> dict:
    """Train one rung. Plain MSE, AdamW, cosine decay to `lr_floor` after linear warmup.

    Plain MSE and nothing else: PSNR is a direct function of MSE, so the loss
    *is* the pass/fail number. Per-pixel 7-way classification was considered
    and loses: two different palette colours are on average 47,814 apart in
    squared distance, so classification would need ~99.6% pixel accuracy to
    reach 30 dB, while regression can hedge with a blend.

    That hedging is a real weakness: MSE rewards blurry edges, and almost all
    of the k-means baseline's error (99.95%) is in the 37% of patches that are
    not flat. The counterweight is the off-palette check on reconstructions,
    which punishes exactly the blur PSNR rewards; the two cannot both be gamed.
    Neither number means much alone.

    **fp32, no mixed precision.** Under a million parameters and a few hundred
    MB of activations at batch 128, so fp32 costs little, and it removes one
    source of numerical doubt from the number this phase depends on. Add mixed
    precision only if a measured step time calls for it.

    **Measured, and still no, for comparability rather than speed.** bf16
    autocast on this card at both resolutions, same model, batch 128:

    | | fp32 | bf16 | 60-epoch rung |
    |---|---|---|---|
    | 64x64 R1 | 37.7 ms/step | 25.8 | 1.39 h -> 0.95 |
    | 64x64 R2 | 40.2 ms/step | 28.6 | 1.48 h -> 1.06 |
    | 96x96 R1 | 75.8 ms/step | 60.4 | 2.80 h -> 2.23 |
    | 96x96 R2 | 85.1 ms/step | 68.3 | 3.14 h -> 2.52 |

    So 1.4-1.5x faster at 64x64 but only **1.25x at 96x96**: bf16 speeds up the
    matrix multiplies, not the `nn.Upsample` / `GroupNorm` / `SiLU` chain, which
    is limited by memory bandwidth, and bigger images spend more time there.
    TF32 alone gives 1.07x, and `cudnn.benchmark` gives nothing since the shapes
    are fixed. **Not adopted** because R1 and R2 at 60 epochs differ by only
    **0.087 dB**, and changing the arithmetic would make every later rung
    incomparable with both. Decide it when starting 96x96 work, where a rung
    costs 3 h, and pay for it by re-running one baseline rather than assuming
    bf16 changes nothing.

    **Not safe to assume: this loop is not bit-reproducible.** Two 1-epoch r1
    runs at seed 0, same machine, nothing changed, gave **25.66625 and 25.66792
    dB**, 0.00167 dB apart, because cuDNN's backward pass is not deterministic
    (`torch.use_deterministic_algorithms` is off, and turning it on would cost
    speed for no benefit here). That is a **1-epoch** figure and only a lower
    bound on the 60-epoch spread, so it does not prove 0.087 dB is significant,
    but it puts the noise about 50x below it rather than unknown.

    The data path is measured too: `_batch` is **0.47 ms of a 40 ms step
    (1.2%)**. A pinned staging buffer brings it to 0.20 ms and gains nothing.
    `non_blocking=True` in `_batch` does nothing on ordinary (unpinned) memory.
    Do not spend time there.

    Every hyperparameter default is a *starting point, not a measurement*; they
    are expected to change.
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

    model = Tokenizer(levels, attention=attention, quantize=quantize,
                      encoder_norm=encoder_norm).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = len(train_idx) // batch
    total = steps_per_epoch * epochs
    warm = max(1, int(warmup * total))

    def lr_at(step: int) -> float:
        """Linear warmup, then cosine decay. The warmup is cheap insurance: a
        cold start can push the bottleneck `tanh` straight into saturation."""
        if step < warm:
            return lr * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return lr_floor + 0.5 * (lr - lr_floor) * (1 + math.cos(math.pi * p))

    hashes = {"data_hash": cfg.data_hash, "tokenizer_hash": cfg.tokenizer_hash}
    knobs = dict(rung=rung, levels=list(levels), attention=attention, quantize=quantize,
                 encoder_norm=encoder_norm, epochs=epochs, batch=batch, lr=lr, lr_floor=lr_floor,
                 weight_decay=weight_decay, warmup=warmup, seed=seed,
                 image_size=list(cfg.shapes.image_size),
                 token_grid=list(cfg.shapes.token_grid), params=params,
                 train_frames=len(train_idx), val_frames=len(val_idx),
                 steps_per_epoch=steps_per_epoch, device=str(dev))

    print(f"{rung}: {params:,} parameters, {len(train_idx):,} train / "
          f"{len(val_idx):,} val frames, {steps_per_epoch} steps/epoch x {epochs}, "
          f"preload {load_s:.1f}s")
    print(f"  levels {list(levels)} quantize={quantize} attention={attention} on {dev}")

    # A fixed training subsample, so the train-vs-val gap compares two numbers
    # measured the same way, not a running loss average against a full eval.
    train_eval = np.sort(rng.choice(len(train_idx),
                                    size=min(eval_frames, len(train_idx)), replace=False))

    start_epoch = 0
    if resume is not None:
        ck = torch.load(ROOT / "runs" / resume / "model.pt", map_location=dev,
                        weights_only=False)
        # Every setting that changes the computation being resumed. All of them
        # matter: `lr_floor` and `warmup` are read by `lr_at` every step and
        # `weight_decay` by AdamW, so resuming with a different value would
        # silently change the schedule mid-run. The CLI does not expose those
        # three, but `train()` is also called directly from Python, which is how
        # R1 and R2 were run.
        #
        # `rung`, `device` and derived values (`params`, `steps_per_epoch`, the
        # frame counts) are deliberately not checked: resuming on another machine
        # must stay possible, and those can differ without changing the maths.
        #
        # `encoder_norm` changes the architecture, so a mismatch would not even
        # load. Checkpoints older than this key are refused by the
        # `k in ck["knobs"]` assert below, which is right: they cannot be *shown*
        # to match.
        for k in ("levels", "attention", "quantize", "batch", "lr", "lr_floor",
                  "warmup", "weight_decay", "epochs", "seed", "encoder_norm"):
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
        # `.cpu()` is required. `torch.load(map_location=dev)` above moves
        # **every** tensor in the checkpoint to the GPU, including both random
        # states, and `set_rng_state` only takes a CPU ByteTensor. Without it,
        # `--resume` on CUDA fails immediately with `TypeError: RNG state must be
        # a torch.ByteTensor`. That bug went unnoticed until the first test that
        # actually exercised resume.
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
            # Saved every epoch, not only at the end, with enough state to
            # *resume*, not just evaluate. Two runs were lost mid-run to a native
            # `nn.Upsample` crash (see the verification log) that this code cannot
            # prevent, so the fix is to make a crash cost one epoch instead of
            # ninety minutes. 4 MB and ~40 ms.
            #
            # The random states are saved instead of reseeding each epoch, because
            # reseeding would change the data order and make a resumed rung
            # incomparable with those already measured, and R1 vs R2 is a
            # comparison of under 0.1 dB.
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
        # Include `epoch` too: the final record also has `val_psnr_db`, and a
        # reader grouping those records by epoch would break without it.
        run.log({"final": True, "epoch": epochs - 1, **out})
        torch.save({"state_dict": model.state_dict(), "knobs": knobs, **hashes},
                   run.dir / "model.pt")
        (run.dir / "result.json").write_text(json.dumps(out, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    return out


# ----------------------------------------------------------------- self-check

def _self_check() -> None:
    """Checks the quantizer and the network without touching any data.

    The key check is the gradient: 0.858 / 1.001 / 0.668 are recorded
    measurements, and reproducing all three at once pins down `eps`, `offset`,
    `shift` and the scaling together. It also checks that `codes_to_indices`
    is one-to-one and that `encode` behaves.
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
        # codes_to_indices must be one-to-one. `unique` returns sorted values, so
        # the cartesian product is every code combination exactly once, and the
        # ids must be 0..prod(levels)-1 with none missing and none repeated.
        # Comparing the *sorted* ids checks both at once: a wrong place value
        # would still give prod(levels) ids, just not those ones.
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

    # No tanh and no clamp on the output, so an untrained decoder must be able to
    # leave [0, 1]. If it cannot, an activation has crept in.
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

    # The token cache depends entirely on `encode`, so check the three things it
    # needs here, before spending a pass over 300,000 frames finding out.
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

    # The point of the `r1c` rung, asserted rather than assumed. Backpropagate
    # from one latent cell and count input pixels with a nonzero gradient. Three
    # stride-2 3x3 convs see 2*(2*(2*1+1)+1)+1 = 15 px, so channel-only
    # normalisation must reach exactly 15x15 for a central cell, and GroupNorm,
    # whose statistics cover the whole map, must reach the whole 64x64 frame.
    # (The receptive field of 22 used in `bench/patch_probe.py` is neither.)
    def _support(encoder_norm: str) -> int:
        m = Tokenizer((8, 8, 8), encoder_norm=encoder_norm)
        xin = torch.rand(1, 3, h, w, requires_grad=True)
        m.encoder(xin)[0, :, grid[0] // 2, grid[1] // 2].sum().backward()
        gin = xin.grad  # read once: `.grad` is a property, so a type narrowing on it does not stick
        assert gin is not None, "backward produced no input gradient"
        return int((gin.abs().sum(1)[0] > 0).sum())

    assert (2 * (2 * (2 * 1 + 1) + 1) + 1) == 15, "the conv-field arithmetic moved"
    got_ch, got_gn = _support("channel"), _support("group")
    assert got_ch == 225, f"channel-only norm reaches {got_ch} pixels, expected 15x15=225"
    assert got_gn == h * w, f"GroupNorm reaches {got_gn} pixels, expected the whole {h}x{w}"
    print(f"one latent cell's gradient support: channel-only {got_ch} px (15x15), "
          f"GroupNorm {got_gn} px ({h}x{w}, the whole frame)")

    # `LocalNorm` is only "in between" if its extremes match: window 1 must
    # reproduce `ChannelNorm`, and window 3 must land strictly between. A token
    # sees 15 + 2*(K-1)*7 px; K=5 gives 71, wider than the frame, so **r1w3 is
    # the only in-between rung possible**, a fact of the architecture rather
    # than a choice.
    torch.manual_seed(0)
    probe = torch.randn(2, 64, grid[0], grid[1])
    cn, ln = ChannelNorm(GN_GROUPS, 64), LocalNorm(GN_GROUPS, 64, 1)
    with torch.no_grad():
        delta = float((cn(probe) - ln(probe)).abs().max())
    assert delta < 1e-5, f"LocalNorm(window=1) differs from ChannelNorm by {delta:.2e}"
    side_w3 = 15 + 2 * (3 - 1) * 7  # the conv window plus the norm window's reach
    got_w3 = _support("local3")
    assert side_w3 == 43 and got_w3 == side_w3 * side_w3 == 1849, \
        f"local3 reaches {got_w3} pixels, expected {side_w3}x{side_w3}"
    assert got_ch < got_w3 < got_gn, "local3 is not strictly between the endpoints"
    assert _support("local5") == h * w, "local5 was expected to saturate the frame"
    print(f"LocalNorm interpolates: window 1 matches ChannelNorm to {delta:.1e}, "
          f"window 3 reaches {got_w3} px (43x43), window 5 already saturates")

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

    # Imported here, not at the top: `fsq_eval` imports this file, so a
    # top-level import back would be circular.
    if args.tokens is not None:
        from mirage.fsq_eval import write_token_cache
        write_token_cache(args.tokens, cfg, batch=args.batch)
        if args.eval is None:
            return

    if args.eval is not None:
        from mirage.fsq_eval import evaluate
        # Nonzero exit when a pass/fail row fails, so scripts can use the gate
        # without reading the table.
        raise SystemExit(1 if evaluate(args.eval, cfg)["failed_rows"] else 0)

    assert args.run is not None
    rung = RUNGS[args.run]
    out = train(args.run, cfg, levels=tuple(args.levels), epochs=args.epochs,
                batch=args.batch, lr=args.lr, seed=args.seed, resume=args.resume,
                quantize=rung["quantize"], attention=rung["attention"],
                encoder_norm=rung.get("encoder_norm", "group"))
    db = out["val_psnr_db"]
    # Gate rows 1 and 2 (the 30 dB target, and the margin over the k-means
    # baseline), which can disagree. Row 1 passing while row 2 fails means the
    # validation split got easier, not that the model got better, so both are
    # printed, not just the headline.
    print(f"\n{args.run.upper()} {out['run_id']}: held-out {db:.3f} dB")
    print(f"  row 1  vs the {PSNR_BAR_DB} dB Q-1 bar:            {db - PSNR_BAR_DB:+.3f} dB")
    floor = kmeans_floor_db(cfg)
    print(f"  row 2  vs the {floor} dB held-out floor: {db - floor:+.3f} dB "
          f"(needs >= {PSNR_BAR_DB - floor:+.2f})")
    print(f"  train-val gap {out['gap_db']:+.3f} dB, {out['train_s']}s")


if __name__ == "__main__":
    main()
