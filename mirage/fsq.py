"""The tokenizer: FSQ quantizer, conv encoder/decoder, train loop, PSNR eval.

An *autoencoder*. The encoder squeezes a 64x64 RGB frame to an 8x8 grid of
`len(levels)` numbers per cell; the decoder rebuilds the frame from that grid.
FSQ (finite scalar quantization) makes the grid discrete by rounding each number
to one of a fixed set of levels, so `prod(levels)` is the vocabulary and there is
no learned dictionary to collapse. Rounding has zero gradient everywhere, so the
backward pass uses a straight-through estimator - it pretends the rounding was
not there.

Built in the order `docs/phase1_structural_plan.md` section 5 names, and this
file currently stops after 5c. What is here:

- **5a** `FSQ` - the quantizer.
- **5b** `Tokenizer` - encoder, optional 8x8 self-attention, decoder.
- **5c** `train` - MSE, AdamW, cosine schedule, held-out PSNR.

What is deliberately **not** here yet: `codes_to_indices`, the R1/R2/R3 ladder,
the gate table and the token-cache writer. R0 - continuous bottleneck, quantizer
bypassed, no attention - answers whether this architecture can reach Q-1's 30 dB
*at all*. If it cannot, no levels table helps and the encoder is what needs work,
so writing the index mapping and the cache first would spend a day finding that
out the expensive way.

    python -m mirage.fsq            # 5a/5b self-check, touches no data
    python -m mirage.fsq --run r0   # the R0 rung

PSNR is `10*log10(255^2 / MSE)` on **uint8** reconstructions, not on the raw
float output. The float number is ~0.01 dB better and the pipeline never delivers
it, so it is not the number.
"""

import argparse
import json
import math
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
    codebook over real 8x8 patches reaches 29.02 dB against the 30 dB Q-1 bar.
    The 0.98 dB gap is what context has to buy.

    Because that margin is inside a training run's noise, this layer's value
    shows up in gate row 2 and not row 1. R0 and R1 run without it.

    UNVERIFIED as a quality lever - R0 does not use it. `_self_check` covers only
    its shapes, its finiteness and that gradients flow through it.
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


def train(rung: str, cfg: config.Config, levels=(8, 8, 8), attention: bool = False,
          quantize: bool = True, epochs: int = 15, batch: int = 128, lr: float = 3e-4,
          lr_floor: float = 3e-5, weight_decay: float = 1e-4, warmup: float = 0.05,
          seed: int = 0, eval_frames: int = 4096, log_every: int = 100,
          device: str | None = None) -> dict:
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

    Every hyperparameter default is a *starting point, not a measurement* - the
    plan says so explicitly and they are expected to move.
    """
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

    with Run(rung, hashes, config=knobs) as run:
        step = 0
        wall = time.perf_counter()
        for epoch in range(epochs):
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
              f"{list(levels)} distinct values per dim, STE gradient at 0 "
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
    with torch.no_grad():
        wide.decoder[-1].bias.fill_(3.0)
        reach = float(wide(x).max())
    assert reach > 1.0, "the output is clamped or squashed somewhere"
    print(f"output is unbounded: a +3.0 output bias reaches {reach:.3f}, "
          f"so no tanh and no clamp")

    for attention in (False, True):
        m = Tokenizer((8, 8, 8), attention=attention, quantize=True)
        m(x).sum().backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, f"the STE did not reach {missing}"
        assert all(torch.isfinite(p.grad).all() for p in m.parameters()), \
            "a parameter gradient is non-finite"
        n = len(list(m.parameters()))
        print(f"attention={attention}: gradient reaches all {n} parameter tensors "
              f"through the quantizer, all finite")

    # A hand-computable case: total SSE 1.0 over 100 values, so MSE = 1/100.
    assert abs(psnr_db(1.0, 100) - 10 * math.log10(255 * 255 * 100)) < 1e-9
    print("psnr_db agrees with 10*log10(255^2 / MSE) by hand")
    print("fsq self-check ok (5a, 5b; no data touched)")


def main() -> None:
    ap = argparse.ArgumentParser(description="the FSQ tokenizer: self-check, or train a ladder rung")
    ap.add_argument("--run", choices=["r0"],
                    help="train a rung (r0: continuous bottleneck, no attention)")
    ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "base.json"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    if args.run is None:
        _self_check()
        return

    cfg = config.load(args.config)
    out = train("r0", cfg, quantize=False, attention=False,
                epochs=args.epochs, batch=args.batch, lr=args.lr)
    bar = 30.0
    print(f"\nR0 {out['run_id']}: held-out {out['val_psnr_db']:.3f} dB against the "
          f"{bar} dB Q-1 bar ({out['val_psnr_db'] - bar:+.3f}), train-val gap "
          f"{out['gap_db']:+.3f} dB, {out['train_s']}s")


if __name__ == "__main__":
    main()
