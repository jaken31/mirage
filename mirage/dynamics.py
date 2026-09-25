"""The dynamics model: the next frame's tokens, from earlier frames and actions.

Phase 2, built in the order of `docs/phase2_structural_plan.md`. This file holds
item 1, the sequence layout and the token/action window sampler; item 3, the
model; and item 4, the training loop with its instrument. The model never sees a pixel: it reads
a tokenizer run's token cache, which `fsq_eval.write_token_cache` writes, and
nothing here re-encodes a frame.

**Item 1 is irreversible, and silent when wrong.** Every checkpoint, rollout and
gate number is stated in this layout's terms, and a layout one position or one
record off trains without complaint. So `_self_check` asserts every part of it
instead of trusting the construction.

A *window* is `ctx + 1` consecutive frames of one episode: the context, plus the
frame it predicts. Window `i` holds exactly the frames `data.WindowSampler`
returns at index `i`, because both find it through one `data.WindowIndex`.

**The stream (decision 2, no shift).** `action[t]` sits immediately before frame
`t`'s 64 tokens, and both are read at record index `t`. That is the recorded
alignment - `qpos[t] - qpos[t-1]` is the result of `action[t]`, see the note at
the top of `mirage/data.py` - so the stream needs no index shift against the
record.

**The blocks (decision 9, block-causal).** The stream is cut before the window's
last frame and grouped into `ctx` blocks of 65 positions: frame `t`'s 64 tokens,
then `action[t+1]`. Attention is full inside a block and causal across blocks.
Frame `t+1`'s cell `i` is scored at the position holding frame `t`'s cell `i`, so
block `t` predicts all 64 cells of frame `t+1` in one pass. Each of those
predictions sees every earlier frame and `action[t+1]`, the action that produced
frame `t+1`, and nothing of frame `t+1` itself. `action[0]` is dropped: it
explains the step into frame 0, which nothing predicts. No target is an action.
At `ctx` 15 that is 15 blocks, **975 positions** and 960 targets. The plan's
1,040 is the same window under strictly-causal attention, which the mask
measurement rejected.

This is `bench/mask_probe.py`'s `layout("block", ...)`, the arrangement that
measurement trained and selected. It is written out again here rather than
imported, so that `_self_check` can compare two independent constructions array
for array.

**The acceptance test is the phase assertion.** Each action is held for
`sim.action_hold_steps` frames, so a one-record shift still pairs most frames
with the right action, and agreement statistics score the shifted reading
higher. What settles it is timing: the policy draws a new action only when its
hold runs out, so every action change in the stream the windows read must sit at
`step_idx % action_hold_steps == 0`. `TokenWindowSampler` refuses to build
otherwise, and `_self_check` asserts that copies shifted one record either way
fail the same check.

**The model (item 3)** is `bench/mask_probe.py`'s block-causal model, the one
that measurement trained and selected: pre-norm blocks, RoPE, attention through
`F.scaled_dot_product_attention` with the block-causal mask, and an untied head,
built from `cfg.dynamics`. Its parameter count is the sizing probe's
`rope_untied`, 14,593,152, exactly. It is scored only where a target is a frame
token - the 64 frame-token positions of each block, never the action slot - and
`_self_check` asserts the causality claim per block rather than trusting the
mask, including that deliberately wrong masks fail it.

**The 500-line trigger fired with item 3**, the one that split `fsq_eval.py` out
of `fsq.py`. That split is by when the code runs, and everything here runs
before a checkpoint exists, so nothing moves out. Rollout and the gate table
(item 6) are what run against a finished checkpoint, and they start in
`mirage/dynamics_eval.py`.

**The training loop (item 4)** trains in bf16 and stops by decision 4a
(`stop_decision`): a 10-epoch cap, or two epochs after held-out loss has risen
two epochs running. Held-out loss and the train/val gap are logged every epoch.
Every 1,000 steps it scores gate row 1's measure - the generated next frame's
accuracy, one forward pass under block-causal - on the mask measurement's 512
fixed val windows, with persistence on the same windows beside it, and keeps
the best point's weights apart from the per-epoch resumable checkpoint. When
the run stops, `mirage.dynamics_eval` scores that best checkpoint on the full
population. `--resume` takes a run id.

    python -m mirage.dynamics                     # self-check: the generated set if present, else the fixture
    python -m mirage.dynamics --train             # the first run: 10-epoch cap, decision 4a
    python -m mirage.dynamics --train --resume RUN_ID
"""

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mirage import config, data
from mirage.fsq import _keep_awake
from mirage.logging import Run

ROOT = Path(__file__).resolve().parent.parent

# The tokenizer Phase 2 inherits (`world_model_architecture.md`, "Phase 2
# inherits R1, and the encoder keeps `GroupNorm`"). Its token cache is the
# dynamics model's whole input.
TOKENIZER_RUN = "20260829-005439-r1"

# 3 torque directions ^ 2 joints, numbered 0..8 (`sim/policy.h`). Action `a` is
# token id `codebook_size + a`, after the frame codes.
N_ACTIONS = 9

# The sizing probe's count for RoPE with an untied head (`runs.jsonl` r49,
# `rope_untied`), the variant decisions 2 and 3 took. The model here must
# match it exactly while it keeps the probe's modules.
SIZED_PARAMS = 14_593_152


# ---------------------------------------------------------------- token cache

def load_token_cache(cfg: config.Config, shards: Sequence[data.Shard],
                     run_id: str = TOKENIZER_RUN,
                     runs_dir: Path | str = ROOT / "runs") -> list[np.ndarray]:
    """A tokenizer run's token cache: one `(frames, cells)` uint16 array per shard.

    Row `r` of array `k` is the tokens of frame `r` of `shards[k]`, cell `i` at
    grid row `i // w`, column `i % w`, with the rows already right-side up
    (`write_token_cache` flips them on the way in).

    Refused unless it is the cache for exactly these shards and this config, the
    way `fsq_eval.load_run` refuses a checkpoint. Otherwise a cache written at a
    superseded `data_hash` trains silently, and one from the 96x96 fork - 144
    tokens a frame, not 64 - fails somewhere after the load instead of at it.
    """
    tok_dir = Path(runs_dir) / run_id / "tokens"
    path = tok_dir / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    codes = cfg.tokenizer["codebook_size"]
    grid = tuple(cfg.shapes.token_grid)
    for key, want in (("run_id", run_id), ("data_hash", cfg.data_hash),
                      ("tokenizer_hash", cfg.tokenizer_hash), ("token_grid", list(grid)),
                      ("codebook_size", codes), ("dtype", "uint16")):
        if man.get(key) != want:
            raise ValueError(f"{path}: {key} is {man.get(key)!r}, expected {want!r}")

    by_index = {s["shard"]: s for s in man["shards"]}
    have = sorted(sh.index for sh in shards)
    if len(by_index) != len(man["shards"]) or sorted(by_index) != have:
        raise ValueError(f"{path}: lists shards {[s['shard'] for s in man['shards']]}, "
                         f"the data holds {have}")

    out = []
    for sh in shards:
        rec = by_index[sh.index]
        toks = np.load(tok_dir / rec["file"])
        # One token row per frame is gate row 4 of the tokenizer: the whole
        # handoff indexes tokens by frame, so an extra or missing row is silent.
        if toks.dtype != np.uint16 or toks.shape != (sh.frames, *grid) or rec["frames"] != sh.frames:
            raise ValueError(f"shard {sh.index}: {toks.dtype} tokens of shape {toks.shape}, "
                             f"manifest frames {rec['frames']}, expected uint16 "
                             f"{(sh.frames, *grid)}")
        if hashlib.sha256(toks.tobytes()).hexdigest() != rec["sha256"]:
            raise ValueError(f"shard {sh.index}: token bytes do not match the manifest sha256")
        # A code at or past `codes` would read as an action token.
        if int(toks.max()) >= codes:
            raise ValueError(f"shard {sh.index}: token {int(toks.max())} is not below {codes}")
        out.append(toks.reshape(sh.frames, math.prod(grid)))
    return out


# --------------------------------------------------------------------- layout

class Layout(NamedTuple):
    """A window's arrangement, as index arrays into the window's *pool*.

    The pool is one window flattened: `pool[t * cells + i]` is frame `t`'s cell
    `i`, and `pool[(ctx + 1) * cells + t]` is `action[t]` as token id
    `codes + action[t]`. Both are read from record `t`, so no arrangement of pool
    indices can put an action beside another record's frame.
    """
    ctx: int
    cells: int
    src: np.ndarray    # (L,) pool index each input position holds
    read: np.ndarray   # (R,) input position each target is scored at
    tgt: np.ndarray    # (R,) pool index of each target
    block: np.ndarray  # (L,) attention block: a position sees every position in blocks <= its own


def layout(ctx: int, cells: int) -> Layout:
    """The block-causal layout of a `ctx + 1`-frame window: `ctx` blocks of `cells + 1`."""
    frames = ctx + 1
    t = np.repeat(np.arange(ctx), cells + 1)             # the block of each position
    k = np.tile(np.arange(cells + 1), ctx)               # its place inside the block
    # Places 0..cells-1 hold frame t, place `cells` holds action[t+1].
    src = np.where(k < cells, t * cells + k, frames * cells + t + 1)
    tgt = np.arange(cells, frames * cells)               # frames 1..ctx, every cell
    # Frame t+1's cell i is scored at frame t's cell i.
    read = (tgt // cells - 1) * (cells + 1) + tgt % cells
    return Layout(ctx, cells, src, read, tgt, t)


def assemble(lay: Layout, tokens: np.ndarray, actions: np.ndarray,
             codes: int) -> tuple[np.ndarray, np.ndarray]:
    """Tokens `(..., ctx+1, cells)` and actions `(..., ctx+1)` -> inputs `(..., L)`, targets `(..., R)`.

    Both int64. The actions must be the ones read at the same records as the
    tokens, which is what `TokenWindowSampler` returns.
    """
    if tokens.shape[-2:] != (lay.ctx + 1, lay.cells) or actions.shape != tokens.shape[:-1]:
        raise ValueError(f"tokens {tokens.shape} and actions {actions.shape} do not fit a "
                         f"{lay.ctx + 1}-frame window of {lay.cells} cells")
    lead = tokens.shape[:-2]
    pool = np.concatenate([tokens.reshape(*lead, -1).astype(np.int64),
                           codes + actions.astype(np.int64)], axis=-1)
    return pool[..., lay.src], pool[..., lay.tgt]


# ------------------------------------------------------------------ alignment

def check_alignment(streams: Iterable[tuple[np.ndarray, np.ndarray]], hold: int) -> int:
    """Raise unless every action change sits at `step_idx % hold == 0`; return the count.

    `streams` holds one `(action, step_idx)` pair per episode, each covering the
    whole episode in order. An episode's first record always counts as a change
    (the policy zeroes its hold when an episode begins), and it sits at step 0,
    which is on the beat.
    """
    phases: set[int] = set()
    changes = 0
    for act, step in streams:
        changed = np.concatenate(([True], act[1:] != act[:-1]))
        phases |= set(np.unique(step[changed].astype(np.int64) % hold).tolist())
        changes += int(changed.sum())
    if phases != {0}:
        raise ValueError(f"actions change at step_idx phases {sorted(phases)} mod {hold}, "
                         f"expected {{0}} - the actions are out of step with the frames")
    return changes


# -------------------------------------------------------------------- sampler

class TokenWindow(NamedTuple):
    tokens: np.ndarray   # (ctx+1, cells) uint16, frame t's cell i at [t, i]
    actions: np.ndarray  # (ctx+1,) uint8, action[t] read at the same record as tokens[t]
    episode_id: int
    offset: int          # the window's first frame, counted from the episode's start


class TokenWindowSampler:
    """`data.WindowSampler`'s windows, read from the token cache instead of the pixels.

    Window `i` is found by the same `data.WindowIndex`, so it is the same
    episode and offset `WindowSampler` returns at `i`, split by the same
    `data.is_val`. Its tokens and actions come from the same shard rows `lo:hi`
    as that sampler's frames: tokens from the cache, actions from the meta
    records the frames came from.

    Builds only if the action stream its windows read passes `check_alignment`.
    Its arrays are read-only views, so a window `__getitem__` returns cannot
    edit the cache.
    """

    def __init__(
        self,
        shards: Sequence[data.Shard],
        index: Iterable[data.Episode],
        tokens: Sequence[np.ndarray],
        ctx: int,
        hold: int,
        split: str = "all",
        val_fraction: float = 0.0,
    ) -> None:
        if len(tokens) != len(shards) or any(len(t) != sh.frames for t, sh in zip(tokens, shards)):
            raise ValueError(f"{[len(t) for t in tokens]} token rows for shards of "
                             f"{[sh.frames for sh in shards]} frames")
        self.windows = data.WindowIndex(index, ctx, split, val_fraction)
        self.window = self.windows.window
        self.split = split
        self.tokens = [np.asarray(t).view() for t in tokens]
        self.actions = [np.array(sh.meta["action"]) for sh in shards]
        self.steps = [np.array(sh.meta["step_idx"]) for sh in shards]
        for a in (*self.tokens, *self.actions, *self.steps):
            a.setflags(write=False)

        top = max(int(self.actions[ep.shard][ep.start:ep.start + ep.length].max())
                  for ep in self.windows.episodes)
        if top >= N_ACTIONS:
            raise ValueError(f"action {top} in the stream, but there are only {N_ACTIONS}")
        self.action_changes = check_alignment(self.action_streams(), hold)

    def action_streams(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """`(action, step_idx)` for each of this split's episodes: the stream the windows read."""
        return [(self.actions[ep.shard][ep.start:ep.start + ep.length],
                 self.steps[ep.shard][ep.start:ep.start + ep.length])
                for ep in self.windows.episodes]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> TokenWindow:
        ep, offset = self.windows.locate(i)
        lo = ep.start + offset
        hi = lo + self.window
        return TokenWindow(self.tokens[ep.shard][lo:hi], self.actions[ep.shard][lo:hi],
                           ep.episode_id, offset)


# ---------------------------------------------------------------------- model

# The `dynamics` choices this file implements. `config.py` may admit more - it
# admits `strict_causal` so the mask measurement's strict arm stays runnable -
# and `build` refuses those rather than training a model the config misnames.
# `rope_base` is the value the mask measurement trained with.
IMPLEMENTED = {"pos_encoding": "rope", "output_head": "untied", "mask": "block_causal",
               "rope_base": 10_000.0}


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each `(i, i + d/2)` channel pair of `x` by an angle proportional to position.

    Computed at the angle tables' precision, then cast back: at position ~1,000
    a bf16 angle table has lost the low bits that tell neighbouring positions
    apart.
    """
    a, b = x.to(cos.dtype).chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1).to(x.dtype)


class Block(nn.Module):
    """Pre-norm transformer block: RoPE attention through SDPA, then a GELU MLP.

    The attention is the sizing probe's `nn.MultiheadAttention` written out as
    its two projections, biases included, so the parameter count is the one it
    priced. Written out because RoPE has to rotate q and k between the
    projection and the attention, and so that `F.scaled_dot_product_attention`
    takes the mask instead of a materialized attention matrix.
    """

    def __init__(self, d: int, heads: int, mlp_ratio: int) -> None:
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)       # MHA's in_proj
        self.out = nn.Linear(d, d)           # MHA's out_proj
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(),
                                 nn.Linear(mlp_ratio * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.heads, d // self.heads) \
            .permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v,
                                           attn_mask=mask)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


class Dynamics(nn.Module):
    """The dynamics model: token ids in, logits over the frame codes out.

    Vocabulary `codes + N_ACTIONS` in (the frame codes, then the actions) and
    `codes` out, through an untied head (decision 3), so no logit is ever an
    action. RoPE (decision 2) has no parameters; its tables are sized for
    `max_len` positions and kept out of `state_dict`, because they follow from
    the shape.

    `forward` takes the mask and the read positions rather than holding them,
    so one checkpoint runs at any context up to `max_len` - the
    configurable-context requirement is a rollout argument, not a config edit.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, mlp_ratio: int,
                 codes: int, max_len: int, rope_base: float) -> None:
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError(f"d_model {d_model} over {n_heads} heads is not an even head width, "
                             f"which RoPE's channel pairs need")
        self.embed = nn.Embedding(codes + N_ACTIONS, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, codes, bias=False)
        half = d_model // n_heads // 2
        f64 = dict(dtype=torch.float64)
        ang = torch.arange(max_len, **f64)[:, None] * rope_base ** (-torch.arange(half, **f64) / half)
        self.register_buffer("cos", ang.cos().float(), persistent=False)
        self.register_buffer("sin", ang.sin().float(), persistent=False)
        # GPT-2's initialisation, as the mask measurement trained with.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor,
                read: torch.Tensor) -> torch.Tensor:
        """`(B, L)` ids, an `(L, L)` mask and `(R,)` positions -> `(B, R, codes)` logits there."""
        n = tok.shape[1]
        if n > len(self.cos):
            raise ValueError(f"{n} positions, but the RoPE tables hold {len(self.cos)}")
        cos, sin = self.cos[:n], self.sin[:n]
        x = self.embed(tok)
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        return self.head(self.norm(x[:, read]))


def build(cfg: config.Config) -> Dynamics:
    """The model `cfg.dynamics` names, sized for a `ctx + 1`-frame window.

    Refuses a `dynamics` value this file does not implement, so a checkpoint's
    `dynamics_hash` always names the model that produced it.
    """
    dyn = cfg.dynamics
    for key, want in IMPLEMENTED.items():
        if dyn[key] != want:
            raise ValueError(f"dynamics.{key} is {dyn[key]!r}; mirage.dynamics implements "
                             f"only {want!r}")
    max_len = len(layout(cfg.data["ctx"], math.prod(cfg.shapes.token_grid)).src)
    return Dynamics(dyn["d_model"], dyn["n_layers"], dyn["n_heads"], dyn["mlp_ratio"],
                    cfg.tokenizer["codebook_size"], max_len, dyn["rope_base"])


def attention_mask(lay: Layout) -> torch.Tensor:
    """`(L, L)` bool, True where a query row may attend to a key column: block-causal.

    Full attention inside a block, causal across blocks. A shorter context's
    mask is the top-left corner of a longer one's, because blocks never move.
    """
    blk = torch.from_numpy(lay.block)
    return blk[None, :] <= blk[:, None]


def frame_loss(model: Dynamics, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
               read: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over a batch's frame targets, and at no other position.

    `x`, `y` are `assemble`'s inputs `(B, L)` and targets `(B, R)`, `read` the
    layout's `(R,)` read positions. The model computes logits only at `read`,
    which are frame-token positions, and every target is a frame code, so an
    action position is never scored and no action id is ever a target. That is
    the whole loss mask, and why there is no clamp: the sizing probe's
    cross-entropy over every position with `clamp(max=511)` would train action
    ids 512-520 as code 511.
    """
    logits = model(x, mask, read)
    return F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())


def check_block_causal(model: Dynamics, lay: Layout, mask: torch.Tensor, x: torch.Tensor,
                       positions: Iterable[int]) -> None:
    """Raise unless `model` under `mask` is block-causal over `lay`, altering each of `positions`.

    Altering the token at position `p`, in block `b`, must leave every logit in
    the blocks before `b` bit-identical, and must change the logits at every
    position from `b`'s first on - all of block `b`, which attends to `p` in
    full, and every later block. The first half is what stops the model reading
    the answer; the second is what fails a mask that is merely too strict.
    """
    positions = list(positions)
    if x.shape[0] != 1:
        raise ValueError(f"one sequence to alter, not a batch of {x.shape[0]}")
    # Row 0 is `x`, row 1 + j is `x` with positions[j] altered: one forward pass.
    y = x.repeat(1 + len(positions), 1)
    for j, p in enumerate(positions):
        y[1 + j, p] = (y[1 + j, p] + 1) % model.embed.num_embeddings
    with torch.no_grad():
        out = model(y, mask, torch.arange(len(lay.src)))
    for j, p in enumerate(positions):
        moved = (out[1 + j] != out[0]).any(-1)
        first = int(np.flatnonzero(lay.block == lay.block[p])[0])
        if moved[:first].any():
            raise ValueError(f"altering position {p} (block {lay.block[p]}) moved a logit in "
                             f"an earlier block, at position {int(moved[:first].nonzero()[0])}")
        if not moved[first:].all():
            raise ValueError(f"altering position {p} (block {lay.block[p]}) left position "
                             f"{first + int((~moved[first:]).nonzero()[0])} unchanged, "
                             f"which should attend to it")


# ------------------------------------------------------------------- training

# Decision 4a, the first run's stopping rule: a 10-epoch cap, or a trip when
# held-out loss rises two epochs running, after which the run trains two more
# epochs and stops - whichever comes first. `stop_decision` is the whole rule.
EPOCH_CAP = 10
TRIP_RISES = 2
TRIP_EXTRA = 2

# The mask measurement's schedule (`bench/mask_probe.py`, r54), so a run at its
# budget trains exactly the model it measured. Starting points, not
# measurements; they travel in each checkpoint's `knobs`, never in a hash.
BATCH = 16
LR, LR_FLOOR, WARMUP = 3e-4, 3e-5, 0.05
WEIGHT_DECAY = 1e-4
CLIP = 1.0

# Gate row 1's measure is scored every `EVAL_EVERY` steps on `EVAL_WINDOWS` val
# windows drawn with `EVAL_SEED`, which are the mask measurement's 512 curve
# windows (r59 re-read persistence on them). Fixed for the whole run, so
# persistence on them is computed once and every point is comparable.
EVAL_EVERY = 1000
EVAL_WINDOWS = 512
EVAL_SEED = 0
EVAL_BATCH = 64   # the mask measurement's eval batch; a batch shape can pick other bf16 kernels


def stop_decision(val_ce: Sequence[float], cap: int) -> dict | None:
    """Decision 4a over the held-out losses so far: None to keep training, else why it stops.

    `val_ce[e - 1]` is the held-out loss after epoch `e`. The trip fires at the
    first epoch `e` whose loss rose for `TRIP_RISES` epochs running -
    `val_ce[e-3] < val_ce[e-2] < val_ce[e-1]`, so epoch 3 at the earliest, and
    an equal loss is not a rise. The run then trains `TRIP_EXTRA` more epochs,
    because decision 4 wants the size of the gap and not just where it turns,
    and the cap bounds those epochs too. A trip whose extra epochs end exactly at
    the cap is recorded as the trip: the trip's stop is reached, and the cap
    alone would have stopped there anyway.

    Returns `{"rule": "trip" | "cap", "epoch": last epoch trained, "trip_epoch": e or None}`.
    """
    trip = next((e for e in range(TRIP_RISES + 1, len(val_ce) + 1)
                 if all(val_ce[k] > val_ce[k - 1] for k in range(e - TRIP_RISES, e))), None)
    if trip is not None and trip + TRIP_EXTRA <= cap:
        if len(val_ce) >= trip + TRIP_EXTRA:
            return {"rule": "trip", "epoch": trip + TRIP_EXTRA, "trip_epoch": trip}
        return None
    if len(val_ce) >= cap:
        return {"rule": "cap", "epoch": cap, "trip_epoch": trip}
    return None


class TokenSplits(NamedTuple):
    """The windows a run trains and scores on. `shards` is None for synthetic windows."""
    train: TokenWindowSampler
    val: TokenWindowSampler
    shards: Sequence[data.Shard] | None


def load_splits(cfg: config.Config, run_id: str = TOKENIZER_RUN) -> TokenSplits:
    """R1's token cache, refused unless it matches the config, as train and val windows."""
    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    tokens = load_token_cache(cfg, shards, run_id)
    ctx, hold, vf = cfg.data["ctx"], cfg.sim["action_hold_steps"], cfg.data["val_fraction"]
    return TokenSplits(TokenWindowSampler(shards, index, tokens, ctx, hold, "train", vf),
                       TokenWindowSampler(shards, index, tokens, ctx, hold, "val", vf), shards)


def _stack(windows: TokenWindowSampler, idx: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Window indices -> tokens `(B, ctx+1, cells)` and actions `(B, ctx+1)`."""
    got = [windows[int(i)] for i in idx]
    return np.stack([w.tokens for w in got]), np.stack([w.actions for w in got])


def _autocast(dev: torch.device):
    """bf16 on CUDA, which is what training runs in; plain fp32 on CPU."""
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


@torch.no_grad()
def score_windows(model: Dynamics, windows: TokenWindowSampler, idx: np.ndarray, lay: Layout,
                  mask: torch.Tensor, dev: torch.device,
                  batch: int = EVAL_BATCH) -> dict[str, np.ndarray]:
    """Held-out loss and gate row 1's measure on windows `idx`, one forward pass each.

    Per window: `ce`, the mean cross-entropy over all its frame targets - the
    training loss, measured in eval mode. Per cell of each window's last frame:
    `pred`, the model's greedy prediction; `target`, the true token; `prev`, the
    token one frame earlier, which is what persistence predicts.

    **`pred` is the generated next frame, not a teacher-forced one.** Under
    block-causal the layout stops before the window's last frame, so no cell of
    it is in the input at all (`_self_check` asserts that changing it leaves the
    input unchanged), and one pass is the whole generation - the mask
    measurement confirmed generated equals this pass exactly (r54).
    """
    was = model.training
    model.eval()
    codes, cells = model.head.out_features, lay.cells
    read = torch.from_numpy(lay.read).to(dev)
    out = {"ce": np.empty(len(idx)), **{k: np.empty((len(idx), cells), np.int64)
                                        for k in ("pred", "target", "prev")}}
    for j in range(0, len(idx), batch):
        tok, act = _stack(windows, idx[j:j + batch])
        x, y = (torch.from_numpy(a).to(dev) for a in assemble(lay, tok, act, codes))
        with _autocast(dev):
            logits = model(x, mask, read)
        logits = logits.float()
        part = slice(j, j + len(tok))
        out["ce"][part] = F.cross_entropy(logits.transpose(1, 2), y,
                                          reduction="none").mean(1).cpu().numpy()
        out["pred"][part] = logits[:, -cells:].argmax(-1).cpu().numpy()
        out["target"][part], out["prev"][part] = tok[:, -1], tok[:, -2]
    model.train(was)
    return out


def last_frame_summary(pred: np.ndarray, target: np.ndarray, prev: np.ndarray) -> dict:
    """Gate row 1's measure against persistence on the same cells, and the copy overlap.

    `margin_points` is accuracy minus persistence in percentage points, the
    number the bar is about. `copy_overlap` is row 10's share of predictions
    that copy the previous frame: a model clearing the bar while copying almost
    everywhere is winning on the cells persistence already gets right.
    """
    hit, same, changed = pred == target, prev == target, prev != target
    return {"cells": int(target.size), "acc": float(hit.mean()), "persistence": float(same.mean()),
            "margin_points": 100 * float(hit.mean() - same.mean()),
            "margin_cells": int(hit.sum() - same.sum()),
            "copy_overlap": float((pred == prev).mean()),
            "acc_changed": float(hit[changed].mean()) if changed.any() else None}


class GpuMonitor:
    """Samples SM clock and power draw on a background thread, and other GPU processes.

    Compute timings are gated on SM clock plus power draw, never on `pstate`,
    which follows the memory clock and reads P4 during correct compute-bound
    work (`AGENTS.md`). A thread rather than a sample at each log step, so the
    training loop never waits on `nvidia-smi` and samples land inside the steps
    rather than beside them. The loop sets `phase`; `take` summarises only the
    samples taken while it said "train", so held-out passes do not dilute the
    clock state recorded beside the step time. Every sample also lists the other
    compute processes on the GPU, so a run that shared the card says so.
    """

    QUERY = "clocks.sm,clocks.max.sm,power.draw,enforced.power.limit,temperature.gpu"

    def __init__(self, every_s: float = 2.0) -> None:
        self.every_s, self.phase = every_s, "train"
        self._samples: list[dict] = []
        self._others: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> "GpuMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def _smi(self, *args: str) -> list[list[str]]:
        out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [[v.strip() for v in line.split(",")] for line in out.splitlines() if line.strip()]

    def _loop(self) -> None:
        me = os.getpid()
        while True:
            phase = self.phase
            try:
                sm, sm_max, power, cap, temp = self._smi(f"--query-gpu={self.QUERY}")[0]
                apps = self._smi("--query-compute-apps=pid,process_name,used_memory")
                sample = {"phase": phase, "sm_mhz": int(sm), "sm_max_mhz": int(sm_max),
                          "power_w": float(power), "power_cap_w": float(cap), "temp_c": float(temp)}
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                sample, apps = None, []
            with self._lock:
                if sample is not None:
                    self._samples.append(sample)
                for pid, name, mib in apps:
                    if pid.isdigit() and int(pid) != me:
                        seen = self._others.setdefault(int(pid), {"name": name, "max_mib": 0})
                        seen["max_mib"] = max(seen["max_mib"], int(mib) if mib.isdigit() else 0)
            if self._stop.wait(self.every_s):
                return

    def take(self) -> dict | None:
        """The training-phase clock state since the last call, and every other process seen."""
        with self._lock:
            got = [s for s in self._samples if s["phase"] == "train"]
            self._samples.clear()
            others = {str(k): dict(v) for k, v in self._others.items()}
        if not got:
            return None
        sm, pw = np.array([s["sm_mhz"] for s in got]), np.array([s["power_w"] for s in got])
        return {"samples": len(got), "every_s": self.every_s,
                "sm_mhz_median": float(np.median(sm)), "sm_mhz_min": int(sm.min()),
                "sm_mhz_max": int(sm.max()), "sm_max_mhz": got[-1]["sm_max_mhz"],
                "power_w_median": float(np.median(pw)), "power_w_min": float(pw.min()),
                "power_cap_w_median": float(np.median([s["power_cap_w"] for s in got])),
                "temp_c_max": max(s["temp_c"] for s in got), "other_processes": others}


def _environment(dev: torch.device) -> dict:
    return {"platform": platform.platform(), "python": platform.python_version(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"}


# Every knob that changes the computation a resumed run continues. `device`,
# the environment and derived counts may differ, so resuming on another
# machine stays possible (`fsq.train` draws the same line).
RESUME_KNOBS = ("tokenizer_run", "epochs", "batch", "lr", "lr_floor", "warmup", "weight_decay",
                "clip", "seed", "eval_every", "eval_windows", "eval_seed", "steps_per_epoch",
                "seq_len")


def held_out(model: Dynamics, splits: TokenSplits, gap_idx: np.ndarray, lay: Layout,
             mask: torch.Tensor, dev: torch.device) -> dict:
    """The end-of-epoch pass: held-out loss over every val window, and the train/val gap.

    The train side is a fixed subset of as many train windows, scored the same
    way in eval mode, so the gap compares two numbers measured alike rather
    than a running training loss against a held-out pass (`fsq.train` does the
    same). The val pass also yields gate row 1's measure on the full population
    for free, so it is logged too.
    """
    val = score_windows(model, splits.val, np.arange(len(splits.val)), lay, mask, dev)
    train_ce = float(score_windows(model, splits.train, gap_idx, lay, mask, dev)["ce"].mean())
    val_ce = float(val["ce"].mean())
    full = last_frame_summary(val["pred"], val["target"], val["prev"])
    return {"val_ce": val_ce, "train_ce": train_ce, "gap_ce": val_ce - train_ce,
            "val_acc_last_all": full["acc"], "persistence_all": full["persistence"],
            "margin_points_all": full["margin_points"]}


def train(cfg: config.Config, epochs: int = EPOCH_CAP, batch: int = BATCH, lr: float = LR,
          lr_floor: float = LR_FLOOR, warmup: float = WARMUP, weight_decay: float = WEIGHT_DECAY,
          clip: float = CLIP, seed: int = 0, eval_every: int = EVAL_EVERY,
          eval_windows: int | None = EVAL_WINDOWS, eval_seed: int = EVAL_SEED,
          steps_per_epoch: int | None = None, log_every: int = 100, resume: str | None = None,
          wandb_project: str | None = None, device: str | None = None,
          runs_dir: Path | str = ROOT / "runs", splits: TokenSplits | None = None,
          session_epochs: int | None = None, score: bool = True, name: str = "dyn") -> dict:
    """Train the dynamics model: bf16, AdamW, warmup then cosine to `lr_floor` over `epochs`.

    **The stopping rule is decision 4a** (`stop_decision`), with `epochs` as its
    cap and as the schedule's length. Held-out loss over every val window is
    logged after every epoch from the first, beside the train/val gap, the
    phase's headline warning sign.

    **Gate row 1's measure is scored every `eval_every` steps** and at each
    epoch's end, on `eval_windows` fixed val windows (all of them if None),
    with persistence on exactly those windows beside every point. The best
    point's weights are kept as `best.pt`, apart from the per-epoch resumable
    `model.pt`, and when the run stops `mirage.dynamics_eval` scores both on the
    full population against persistence re-measured there by
    `bench/token_stability_probe.py` (unless `score` is False).

    **`resume` is a run id.** The run continues from that run's last per-epoch
    checkpoint in a new run directory - `logging.Run` never reopens one -
    carrying the history, the sub-epoch points and the best weights forward,
    and naming the run it continues as `resumed_from`. Data order, weights,
    optimiser and random states are restored, so a resumed run takes the steps
    the uninterrupted one would have (bit-identical on CPU, `_self_check`).

    `steps_per_epoch` shortens an epoch for smoke runs; `session_epochs` ends
    this process after that many epochs, leaving the checkpoint to resume -
    neither changes what a step computes. `splits` replaces R1's cache and
    `name` the run id's suffix, for the self-check.

    Keep the machine awake: `fsq._keep_awake` does it on Windows; on Linux,
    launch under `systemd-inhibit --what=idle:sleep`.
    """
    _keep_awake()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = dev.type == "cuda"
    runs_dir = Path(runs_dir)
    splits = splits or load_splits(cfg)
    lay = layout(cfg.data["ctx"], math.prod(cfg.shapes.token_grid))
    mask, read = attention_mask(lay).to(dev), torch.from_numpy(lay.read).to(dev)
    codes = cfg.tokenizer["codebook_size"]

    torch.manual_seed(seed)   # immediately before `build`: the mask measurement's initial weights
    model = build(cfg).to(dev)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    per_epoch = steps_per_epoch or len(splits.train) // batch
    if not 0 < per_epoch * batch <= len(splits.train):
        raise ValueError(f"{per_epoch} steps of batch {batch} do not fit "
                         f"{len(splits.train)} train windows")
    total = per_epoch * epochs
    warm = max(1, int(warmup * total))

    def lr_at(step: int) -> float:
        if step < warm:
            return lr * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return lr_floor + 0.5 * (lr - lr_floor) * (1 + math.cos(math.pi * p))

    # The fixed windows, drawn the way the mask measurement drew its curve's,
    # and a fixed train subset as large as the val split for the epoch gap.
    ev = np.random.default_rng(eval_seed)
    n_eval = len(splits.val) if eval_windows is None else eval_windows
    val_sub = np.sort(ev.choice(len(splits.val), n_eval, replace=False))
    train_sub = np.sort(ev.choice(len(splits.train), min(n_eval, len(splits.train)), replace=False))
    gap_idx = np.sort(np.random.default_rng([eval_seed, 1]).choice(
        len(splits.train), min(len(splits.val), len(splits.train)), replace=False))
    tok_sub = _stack(splits.val, val_sub)[0]
    persistence = float((tok_sub[:, -1] == tok_sub[:, -2]).mean())

    hashes = {"data_hash": cfg.data_hash, "tokenizer_hash": cfg.tokenizer_hash,
              "dynamics_hash": cfg.dynamics_hash}
    knobs = dict(tokenizer_run=TOKENIZER_RUN, epochs=epochs, batch=batch, lr=lr, lr_floor=lr_floor,
                 warmup=warmup, weight_decay=weight_decay, clip=clip, seed=seed,
                 eval_every=eval_every, eval_windows=eval_windows, eval_seed=eval_seed,
                 steps_per_epoch=per_epoch, seq_len=len(lay.src), targets=len(lay.tgt),
                 params=params, precision="bf16 autocast" if cuda else "fp32",
                 train_windows=len(splits.train), val_windows=len(splits.val),
                 action_changes=splits.train.action_changes + splits.val.action_changes,
                 resumed_from=resume, **_environment(dev))

    history: list[dict] = []    # one record per finished epoch
    points: list[dict] = []     # one record per sub-epoch point
    best: dict | None = None
    best_state: dict | None = None
    start = 0
    if resume is not None:
        ck = torch.load(runs_dir / resume / "model.pt", map_location="cpu", weights_only=True)
        for k, v in hashes.items():
            if ck[k] != v:
                raise ValueError(f"resume {resume}: {k} is {ck[k]}, the config's is {v}")
        for k in RESUME_KNOBS:
            if ck["knobs"][k] != knobs[k]:
                raise ValueError(f"resume {resume}: {k} is {ck['knobs'][k]!r}, "
                                 f"this call asks for {knobs[k]!r}")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]
        # `.cpu()` because `set_rng_state` takes a CPU ByteTensor - the bug that
        # left `fsq --resume` broken on CUDA until it was first executed.
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] is not None and cuda:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        history, points, best, best_state = ck["history"], ck["points"], ck["best"], ck["best_state"]
        start = ck["epoch"]
        done = stop_decision([h["val_ce"] for h in history], epochs)
        if done is not None:
            raise ValueError(f"resume {resume}: the run already stopped by the {done['rule']} "
                             f"rule at epoch {done['epoch']}; training past a stop is a new "
                             f"decision (decision 4a), not a resume")
        print(f"resumed {resume} after epoch {start}/{epochs}")

    print(f"dynamics: {params:,} parameters, {len(lay.src)} positions, {len(splits.train):,} train "
          f"/ {len(splits.val):,} val windows, {per_epoch:,} steps/epoch x {epochs} at batch "
          f"{batch} on {dev} ({knobs['precision']})")
    print(f"  gate row 1 every {eval_every:,} steps on {n_eval:,} fixed val windows, "
          f"persistence there {persistence:.4%}")

    monitor = GpuMonitor() if cuda else None
    with Run(name, hashes, config=knobs, root=runs_dir, wandb_project=wandb_project) as run, \
            (monitor or contextlib.nullcontext()):
        if best_state is not None:
            torch.save({"state_dict": best_state, "knobs": knobs, **hashes, **best},
                       run.dir / "best.pt")
        wall = time.perf_counter()
        step = start * per_epoch
        stop = None
        peak = {"alloc": 0, "reserved": 0}

        def point(epoch: int) -> None:
            """Score the fixed windows, log the point beside persistence, keep the best."""
            nonlocal best, best_state
            if cuda:
                peak["alloc"] = max(peak["alloc"], torch.cuda.max_memory_allocated(dev))
                peak["reserved"] = max(peak["reserved"], torch.cuda.max_memory_reserved(dev))
                monitor.phase = "eval"
            sv = score_windows(model, splits.val, val_sub, lay, mask, dev)
            st = score_windows(model, splits.train, train_sub, lay, mask, dev)
            s = last_frame_summary(sv["pred"], sv["target"], sv["prev"])
            assert s["persistence"] == persistence, "the fixed windows moved"
            rec = {"step": step, "epoch": epoch, "val_acc_last": s["acc"], "persistence": persistence,
                   "margin_points": s["margin_points"], "margin_cells": s["margin_cells"],
                   "copy_overlap": s["copy_overlap"], "val_acc_changed": s["acc_changed"],
                   "val_ce_sub": float(sv["ce"].mean()), "train_ce_sub": float(st["ce"].mean())}
            rec["best"] = best is None or rec["val_acc_last"] > best["val_acc_last"]
            if rec["best"]:
                best = {k: rec[k] for k in ("step", "epoch", "val_acc_last", "persistence",
                                            "margin_points", "copy_overlap")}
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"state_dict": best_state, "knobs": knobs, **hashes, **best},
                           run.dir / "best.pt")
            points.append(rec)
            run.log({"point": True, **rec, "wall_s": round(time.perf_counter() - wall, 1)})
            print(f"  step {step:>7,}  acc(last) {s['acc']:.4%}  persistence {persistence:.4%}  "
                  f"{s['margin_points']:+.2f} pts  copy {s['copy_overlap']:.2%}  val ce(sub) "
                  f"{rec['val_ce_sub']:.4f}{'  best' if rec['best'] else ''}", flush=True)
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
                monitor.phase = "train"

        for epoch in range(start + 1, epochs + 1):
            if session_epochs is not None and epoch > start + session_epochs:
                break
            t_epoch, step_s = time.perf_counter(), []
            if cuda:
                torch.cuda.reset_peak_memory_stats(dev)
                peak = {"alloc": 0, "reserved": 0}
                monitor.take()   # drop samples from before this epoch
            order = rng.permutation(len(splits.train))[:per_epoch * batch].reshape(per_epoch, batch)
            for s in range(per_epoch):
                timed = cuda and step % 50 == 0
                if timed:
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                tok, act = _stack(splits.train, order[s])
                x, y = (torch.from_numpy(a).to(dev) for a in assemble(lay, tok, act, codes))
                with _autocast(dev):
                    loss = frame_loss(model, x, y, mask, read)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), clip))
                opt.step()
                if timed:
                    torch.cuda.synchronize(dev)
                    step_s.append(time.perf_counter() - t0)
                step += 1
                if step % log_every == 0:
                    run.log({"step": step, "epoch": epoch, "loss": float(loss.detach()),
                             "lr": lr_at(step - 1), "grad_norm": gnorm,
                             "wall_s": round(time.perf_counter() - wall, 1)})
                if step % eval_every == 0 or s == per_epoch - 1:
                    point(epoch)

            train_s = time.perf_counter() - t_epoch
            gpu = monitor.take() if cuda else None
            if cuda:
                monitor.phase = "eval"
            rec = {"epoch": epoch, "step": step,
                   **held_out(model, splits, gap_idx, lay, mask, dev),
                   "ms_per_step_median": (round(1e3 * float(np.median(step_s)), 1)
                                          if step_s else None),
                   "timed_steps": len(step_s), "batch": batch, "gpu": gpu,
                   "peak_vram_alloc_gb": round(peak["alloc"] / 2**30, 3) if cuda else None,
                   "peak_vram_reserved_gb": round(peak["reserved"] / 2**30, 3) if cuda else None,
                   "epoch_train_s": round(train_s, 1),
                   "epoch_s": round(time.perf_counter() - t_epoch, 1)}
            if cuda:
                monitor.phase = "train"
            history.append(rec)
            stop = stop_decision([h["val_ce"] for h in history], epochs)
            run.log({"epoch_end": True, **rec, "stop": stop,
                     "wall_s": round(time.perf_counter() - wall, 1)})
            print(f"epoch {epoch}/{epochs}  val ce {rec['val_ce']:.4f}  train ce "
                  f"{rec['train_ce']:.4f}  gap {rec['gap_ce']:+.4f}  acc(last, all val) "
                  f"{rec['val_acc_last_all']:.4%}  {rec['ms_per_step_median']} ms/step  "
                  f"{rec['epoch_s']:.0f}s{f'  STOP: {stop}' if stop else ''}", flush=True)
            # Every epoch, with everything a resume needs. The random states are
            # restored rather than reseeded, so a resumed run draws the data
            # order the uninterrupted one would have.
            torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                        "step": step, "knobs": knobs, **hashes,
                        "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if cuda else None,
                        "history": history, "points": points, "best": best,
                        "best_state": best_state},
                       run.dir / "model.pt")
            if stop is not None:
                break

        out = {**knobs, **hashes, "run_id": run.run_id, "stop": stop,
               "epochs_done": len(history), "history": history, "points": points, "best": best,
               "train_s": round(time.perf_counter() - wall, 1)}
        if stop is not None and score:
            if cuda:
                monitor.phase = "eval"
            # Imported here: `dynamics_eval` imports this file.
            from mirage import dynamics_eval
            out["population"] = dynamics_eval.score_run(run.dir, cfg, splits, dev)
        run.log({"final": True, **{k: v for k, v in out.items() if k not in ("history", "points")}})
        (run.dir / "result.json").write_text(json.dumps(out, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="the dynamics model: self-check, or train it")
    ap.add_argument("--train", action="store_true", help="train (decision 4a's stopping rule)")
    ap.add_argument("--config", default=str(ROOT / "mirage" / "configs" / "base.json"))
    ap.add_argument("--epochs", type=int, default=EPOCH_CAP,
                    help=f"the epoch cap and the schedule's length (default {EPOCH_CAP})")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-windows", default=str(EVAL_WINDOWS),
                    help=f"fixed val windows gate row 1 is scored on every {EVAL_EVERY:,} "
                         f"steps, or 'all' (default {EVAL_WINDOWS})")
    ap.add_argument("--steps-per-epoch", type=int, help="shorten each epoch (smoke runs only)")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="continue that run from its last per-epoch checkpoint; every "
                         "knob must match")
    ap.add_argument("--wandb", metavar="PROJECT", help="mirror the jsonl to this W&B project")
    args = ap.parse_args()
    if not args.train:
        _self_check()
        return
    train(config.load(args.config), epochs=args.epochs, batch=args.batch, seed=args.seed,
          eval_windows=None if args.eval_windows == "all" else int(args.eval_windows),
          steps_per_epoch=args.steps_per_epoch, resume=args.resume, wandb_project=args.wandb)


# ----------------------------------------------------------------- self-check

def _refused(build: Callable[..., object], *args: object) -> bool:
    """True if `build(*args)` raises ValueError, the error every refusal here uses."""
    try:
        build(*args)
    except ValueError:
        return True
    return False


def _bench_module(name: str):
    """`bench/<name>.py`, imported by path: `bench/` is scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The sizing probe's names for the same parameters: it keeps attention as one
# `nn.MultiheadAttention`, this file as that module's two projections.
_SIZING_NAMES = {"attn.in_proj_weight": "qkv.weight", "attn.in_proj_bias": "qkv.bias",
                 "attn.out_proj.weight": "out.weight", "attn.out_proj.bias": "out.bias"}


def _param_diff(model: nn.Module, sized: nn.Module) -> list[str]:
    """Every parameter whose size differs between `model` and the sizing probe's `sized`, by name.

    Empty when the two are the same modules. Otherwise it names the module, so a
    count that moves is a module difference to state rather than a total to
    round.
    """
    def sizes(m: nn.Module, rename: dict[str, str]) -> dict[str, int]:
        out = {}
        for name, p in m.named_parameters():
            for old, new in rename.items():
                name = name.replace(old, new)
            out[name] = p.numel()
        return out

    mine, theirs = sizes(model, {}), sizes(sized, _SIZING_NAMES)
    return [f"{n}: {mine.get(n, 0):,} here, {theirs.get(n, 0):,} in the sizing probe"
            for n in sorted(mine.keys() | theirs.keys()) if mine.get(n) != theirs.get(n)]


def _identity_cache(shards: Sequence[data.Shard], cells: int) -> list[np.ndarray]:
    """Token rows that name their own frame: cells 0-2 hold the row and the shard.

    The known-value control for the window checks. The real cache repeats
    whole rows wherever the scene is still, so a read one frame off can match it
    by coincidence. Here every row differs, so it cannot.
    """
    out = []
    for k, sh in enumerate(shards):
        rows = np.tile(np.arange(cells, dtype=np.uint16), (sh.frames, 1))
        r = np.arange(sh.frames)
        rows[:, 0], rows[:, 1], rows[:, 2] = r % 512, r // 512, k
        out.append(rows)
    return out


def _write_cache(runs_dir: Path, run_id: str, cfg: config.Config,
                 shards: Sequence[data.Shard], cache: Sequence[np.ndarray],
                 edit: dict | None = None) -> None:
    """Write `cache` the way `write_token_cache` lays it out, with manifest fields replaced by `edit`."""
    tok_dir = runs_dir / run_id / "tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for sh, rows in zip(shards, cache):
        toks = rows.reshape(len(rows), *cfg.shapes.token_grid)
        np.save(tok_dir / f"shard_{sh.index:03d}.npy", toks)
        recs.append({"shard": sh.index, "frames": len(toks), "file": f"shard_{sh.index:03d}.npy",
                     "sha256": hashlib.sha256(toks.tobytes()).hexdigest()})
    man = {"run_id": run_id, "data_hash": cfg.data_hash, "tokenizer_hash": cfg.tokenizer_hash,
           "token_grid": list(cfg.shapes.token_grid),
           "codebook_size": cfg.tokenizer["codebook_size"], "dtype": "uint16", "shards": recs}
    (tok_dir / "manifest.json").write_text(json.dumps({**man, **(edit or {})}), encoding="utf-8")


def _check_windows(cfg: config.Config, shards: Sequence[data.Shard],
                   index: Sequence[data.Episode], tokens: Sequence[np.ndarray],
                   splits: Sequence[str], identity: bool) -> None:
    """Every window of every split in `splits`, against `WindowSampler` at the same index."""
    ctx, hold, vf = cfg.data["ctx"], cfg.sim["action_hold_steps"], cfg.data["val_fraction"]
    where = {ep.episode_id: ep for ep in index}
    for split in splits:
        tw = TokenWindowSampler(shards, index, tokens, ctx, hold, split, vf)
        ws = data.WindowSampler(shards, index, ctx, split, vf)
        assert len(tw) == len(ws) and tw.window == ws.window == ctx + 1
        for i in range(len(tw)):
            t, w = tw[i], ws[i]
            # The shared addressing names the window WindowSampler returned, read
            # back from that window's own meta records rather than recomputed.
            assert (w.meta["episode_id"] == t.episode_id).all(), f"window {i} straddles an episode"
            assert int(w.meta["step_idx"][0]) == t.offset, f"window {i}: offset {t.offset}"
            ep = where[t.episode_id]
            rows = ep.start + w.meta["step_idx"].astype(np.int64)
            assert t.offset + tw.window <= ep.length and np.array_equal(
                rows, np.arange(ep.start + t.offset, ep.start + t.offset + tw.window)), i
            assert np.array_equal(t.tokens, tokens[ep.shard][rows]), f"window {i}: token rows"
            assert np.array_equal(t.actions, w.meta["action"]), f"window {i}: actions"
            if identity:
                named = t.tokens[:, 0].astype(np.int64) + 512 * t.tokens[:, 1]
                assert np.array_equal(named, rows) and (t.tokens[:, 2] == ep.shard).all(), i
        print(f"  {split}: all {len(tw):,} windows are WindowSampler's (episode, offset), "
              f"one episode each, tokens and actions from its records")


class _Synthetic:
    """Random token windows behind `TokenWindowSampler`'s interface, for the CPU training checks."""

    def __init__(self, n: int, ctx: int, cells: int, codes: int, seed: int) -> None:
        r = np.random.default_rng(seed)
        self.tokens = r.integers(0, codes, (n, ctx + 1, cells)).astype(np.uint16)
        self.actions = r.integers(0, N_ACTIONS, (n, ctx + 1)).astype(np.uint8)
        self.action_changes = 0

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int) -> TokenWindow:
        return TokenWindow(self.tokens[i], self.actions[i], 0, i)


def _records(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()]


def _check_training(base: config.Config) -> None:
    """Item 4's working-when, as far as it can be shown on CPU without data.

    The stopping rule against decision 4a case by case; then the real `train`
    on a tiny model and synthetic windows: a run killed after epoch 1 and
    resumed takes bit-identical steps to one never interrupted, and the rule
    trips and stops where decision 4a says through the loop itself, across a
    resume.
    """
    cases = [  # held-out losses, epoch cap -> decision
        ([5, 4, 3, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5], 10, {"rule": "cap", "epoch": 10, "trip_epoch": None}),
        ([5, 4, 3, 2], 10, None),
        ([5, 6, 7], 10, None),                                    # trips at 3, trains 4 and 5
        ([5, 6, 7, 6, 5], 10, {"rule": "trip", "epoch": 5, "trip_epoch": 3}),
        ([5, 4, 3, 3.5, 3.6, 3.0], 10, None),
        ([5, 4, 3, 3.5, 3.6, 3.0, 2.0], 10, {"rule": "trip", "epoch": 7, "trip_epoch": 5}),
        ([5, 6, 5, 6, 5, 6, 5, 6, 5, 6], 10, {"rule": "cap", "epoch": 10, "trip_epoch": None}),
        ([5, 5, 5, 5, 5, 5, 5, 5, 5, 5], 10, {"rule": "cap", "epoch": 10, "trip_epoch": None}),
        ([9, 8, 7, 6, 5, 4, 3, 3.1, 3.2, 3.3], 10, {"rule": "cap", "epoch": 10, "trip_epoch": 9}),
        ([9, 8, 7, 6, 5, 3.0, 3.1, 3.2, 3.0, 2.9], 10, {"rule": "trip", "epoch": 10, "trip_epoch": 8}),
        ([2.0], 1, {"rule": "cap", "epoch": 1, "trip_epoch": None}),
    ]
    for losses, cap, want in cases:
        got = stop_decision(losses, cap)
        assert got == want, f"stop_decision({losses}, {cap}) = {got}, decision 4a says {want}"
        # The decision is taken the first epoch it can be, never later.
        assert all(stop_decision(losses[:k], cap) is None for k in range(len(losses))), losses
    print(f"stopping rule: {len(cases)} cases - the cap at 10; a trip on two consecutive rises "
          f"then two more epochs; no trip on one rise, a flat loss or a zigzag; a trip at 9 "
          f"bounded by the cap; a trip at 8 stopping at 10 as the trip")

    ctx, cells, codes = base.data["ctx"], math.prod(base.shapes.token_grid), 512
    lay = layout(ctx, cells)
    tok = np.random.default_rng(1).integers(0, codes, (2, ctx + 1, cells))
    act = np.zeros((2, ctx + 1), np.int64)
    other = tok.copy()
    other[:, -1] = (other[:, -1] + 1) % codes
    assert np.array_equal(assemble(lay, tok, act, codes)[0], assemble(lay, other, act, codes)[0]), \
        "the last frame reached the input: one pass would not be a generated frame"
    print("gate row 1: no cell of a window's last frame is in its input, so one pass generates it")

    tiny = dataclasses.replace(base, dynamics={**base.dynamics, "d_model": 16, "n_layers": 1,
                                               "n_heads": 2})
    splits = TokenSplits(_Synthetic(24, ctx, cells, codes, 0), _Synthetic(8, ctx, cells, codes, 1),
                         None)
    common = dict(epochs=3, batch=2, steps_per_epoch=3, eval_every=2, eval_windows=4,
                  log_every=1, device="cpu", splits=splits, score=False)
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        whole = train(tiny, **common, runs_dir=runs, name="whole")
        first = train(tiny, **common, runs_dir=runs, name="first", session_epochs=1)
        rest = train(tiny, **common, runs_dir=runs, name="rest", resume=first["run_id"])
        assert first["stop"] is None and first["epochs_done"] == 1
        assert whole["stop"] == rest["stop"] and (whole["stop"]["rule"], whole["stop"]["epoch"]) == ("cap", 3)

        def losses(r: dict) -> list:
            return [(x["step"], x["loss"]) for x in _records(runs / r["run_id"]) if "loss" in x]
        assert losses(whole) == losses(first) + losses(rest), "the resumed losses differ"
        assert [x[0] for x in losses(rest)] == list(range(4, 10)), "the resume skipped or repeated steps"
        def untimed(r: dict) -> list:
            return [{k: v for k, v in h.items() if not k.endswith("_s")} for h in r["history"]]
        assert untimed(whole) == untimed(rest), "the epoch records differ"
        assert whole["points"] == rest["points"] and whole["best"] == rest["best"]
        for f in ("model.pt", "best.pt"):
            a, b = (torch.load(runs / r["run_id"] / f, weights_only=True) for r in (whole, rest))
            assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"]), f
        # Every record names its run and carries the hashes, dynamics_hash included.
        for r in (whole, first, rest):
            for x in _records(runs / r["run_id"]):
                assert x["run_id"] == r["run_id"] and x["dynamics_hash"] == tiny.dynamics_hash, x
        meta = json.loads((runs / rest["run_id"] / "meta.json").read_text("utf-8"))
        assert meta["config"]["resumed_from"] == first["run_id"]
        assert first["history"][0]["val_ce"] > 0 and "val_ce" in _records(runs / first["run_id"])[-2]
        assert _refused(lambda: train(tiny, **{**common, "lr": 1e-3}, runs_dir=runs, name="x",
                                      resume=first["run_id"])), "resumed with another learning rate"
        assert _refused(lambda: train(tiny, **common, runs_dir=runs, name="y",
                                      resume=rest["run_id"])), \
            "resumed a run the stopping rule had already stopped"
        print(f"train: ended after epoch 1 and resumed, {len(losses(whole))} steps bit-identical to "
              f"an uninterrupted run - losses, epoch records, sub-epoch points, model.pt and "
              f"best.pt; every record carries dynamics_hash; refuses another knob or a stopped run")

        # The trip through the loop, across a resume: scripted held-out losses
        # stand in for a model that starts to overfit at epoch 4.
        script = iter([5.0, 4.0, 3.0, 3.5, 3.6, 3.0, 2.0, 1.0, 0.5, 0.4])
        real = globals()["held_out"]

        def scripted(*a: object, **k: object) -> dict:
            return {**real(*a, **k), "val_ce": next(script)}
        globals()["held_out"] = scripted
        try:
            trip = dict(common, epochs=10, steps_per_epoch=1, eval_every=100)
            a = train(tiny, **trip, runs_dir=runs, name="trip-a", session_epochs=6)
            b = train(tiny, **trip, runs_dir=runs, name="trip-b", resume=a["run_id"])
        finally:
            globals()["held_out"] = real
        assert a["stop"] is None and a["epochs_done"] == 6
        assert b["stop"] == {"rule": "trip", "epoch": 7, "trip_epoch": 5}, b["stop"]
        assert b["epochs_done"] == 7 and [h["epoch"] for h in b["history"]] == list(range(1, 8))
        final = _records(runs / b["run_id"])[-1]
        assert final["final"] and final["stop"] == b["stop"]
        print("train: held-out loss rising at epochs 4 and 5 trips at 5 and stops after epoch 7 "
              "of 10, across a resume at 6; the final record says trip, epoch 7")


def _self_check() -> None:
    """Items 1 and 3's working-when lists (`docs/phase2_structural_plan.md`)."""
    codes, cells = 512, 64

    # ---- the layout: no data needed
    lay = layout(15, cells)
    frames = lay.ctx + 1
    assert len(lay.src) == len(lay.block) == 15 * 65 == 975, len(lay.src)
    assert len(lay.tgt) == len(lay.read) == 15 * 64 == 960
    assert set(np.bincount(lay.block).tolist()) == {65}, "a block is not 65 positions"
    assert len(np.unique(lay.src)) == len(lay.src), "a pool slot is fed twice"
    # Every target is a frame token, frames 1..15, each cell once.
    assert np.array_equal(lay.tgt, np.arange(cells, frames * cells))
    # Frame t+1's cell i is read where frame t's cell i sits.
    assert np.array_equal(lay.src[lay.read], lay.tgt - cells)
    # Block t is frame t's cells in order, then action[t+1]. So action[t]
    # immediately precedes frame t, for every frame in the input.
    for t in range(lay.ctx):
        b = lay.src[lay.block == t]
        assert np.array_equal(b[:cells], np.arange(t * cells, (t + 1) * cells)), t
        assert b[cells] == frames * cells + t + 1, t
    # No target is visible where it is scored: a position sees blocks up to its
    # own, and a target's own slot, where it has one, is in a later block. And
    # the action that produced the target frame is in the scoring block.
    slot = np.full(frames * cells + frames, -1)
    slot[lay.src] = np.arange(len(lay.src))
    held = slot[lay.tgt] >= 0
    assert held.sum() == 14 * 64, "frames 1..14 should be both input and target"
    assert (lay.block[slot[lay.tgt][held]] > lay.block[lay.read][held]).all(), "a target is visible"
    act_slot = slot[frames * cells + lay.tgt // cells]
    assert np.array_equal(lay.block[act_slot], lay.block[lay.read]), "a frame's action is not in its block"
    assert slot[frames * cells] == -1, "action[0] explains no target and is dropped"

    # Round trip: every token and action in a window lands at its pool slot,
    # and the pool rebuilds the window.
    rng = np.random.default_rng(0)
    tok = rng.integers(0, codes, (2, frames, cells))
    act = rng.integers(0, N_ACTIONS, (2, frames))
    x, y = assemble(lay, tok, act, codes)
    assert x.shape == (2, 975) and y.shape == (2, 960) and y.max() < codes
    pool = np.full((2, frames * cells + frames), -1)
    pool[:, lay.src] = x
    assert np.array_equal(pool[:, lay.tgt][:, held], y[:, held]), "an input and its target disagree"
    pool[:, lay.tgt] = y
    assert np.array_equal(pool[:, :frames * cells].reshape(tok.shape), tok)
    assert np.array_equal(pool[:, frames * cells + 1:], codes + act[:, 1:])
    assert (pool[:, frames * cells] == -1).all()
    assert _refused(assemble, lay, tok[:, 1:], act[:, 1:], codes), \
        "assemble took a 15-frame window for a 16-frame layout"

    # The mask measurement's arrangement, array for array, at the full context
    # and the shorter ones a rollout is asked to run at.
    probe = _bench_module("mask_probe")
    for c in (4, 8, 15):
        mine, theirs = layout(c, cells), probe.layout("block", c + 1, cells)
        for f in ("src", "read", "tgt", "block"):
            assert np.array_equal(getattr(mine, f), getattr(theirs, f)), f"ctx {c}: {f} differs"
    print("layout: 15 blocks of 65 = 975 positions, 960 frame targets, action[t] before "
          "frame t, round-trips; equals bench/mask_probe.py's block layout at ctx 4, 8, 15")

    # ---- the model (item 3): no data needed
    torch.manual_seed(0)
    base = config.load(ROOT / "mirage" / "configs" / "base.json")
    model = build(base).eval()
    params = sum(p.numel() for p in model.parameters())
    sizing = _bench_module("dyn_size_probe")
    dyn = base.dynamics

    def sized(tied: bool) -> nn.Module:
        return sizing.Dynamics(dyn["d_model"], dyn["n_layers"], dyn["n_heads"],
                               codes + sizing.N_ACTIONS, codes, len(lay.src),
                               learned_pos=False, tied=tied, mlp_ratio=dyn["mlp_ratio"])

    diff = _param_diff(model, sized(tied=False))
    assert not diff, "a module differs from the sizing probe's rope_untied:\n  " + "\n  ".join(diff)
    assert params == SIZED_PARAMS, f"{params:,} parameters, the sizing probe's r49 row says {SIZED_PARAMS:,}"
    # The control: against the tied variant the one difference is the head, named.
    assert _param_diff(model, sized(tied=True)) == \
        ["head.weight: 196,608 here, 0 in the sizing probe"], _param_diff(model, sized(tied=True))
    assert not any(isinstance(m, nn.MultiheadAttention) for m in model.modules())
    print(f"model: {params:,} parameters, the sizing probe's rope_untied (r49) exactly, "
          f"parameter for parameter; its tied variant differs by head.weight alone")

    # The model the mask measurement trained and selected, function for
    # function: the same weights from the same seed, and bit-identical logits
    # under the same mask. A count cannot see a post-norm block, another RoPE
    # base or another initialisation; this can.
    torch.manual_seed(0)   # the seed `model` was built from
    measured = probe.build_model(base, len(lay.src)).eval()
    sd, sd_m = model.state_dict(), measured.state_dict()
    assert list(sd) == list(sd_m) and all(torch.equal(sd[k], sd_m[k]) for k in sd), \
        "not bench/mask_probe.py's parameters and initialisation"
    mask = attention_mask(lay)
    assert torch.equal(mask, probe.layout("block", frames, cells).mask()), \
        "the mask differs from bench/mask_probe.py's block mask"
    x_any = torch.randint(0, codes + N_ACTIONS, (1, len(lay.src)))
    with torch.no_grad():
        assert torch.equal(model(x_any, mask, torch.from_numpy(lay.read)),
                           measured(x_any, mask, torch.from_numpy(lay.read))), \
            "logits differ from bench/mask_probe.py's model"
    del measured
    print("model: bench/mask_probe.py's block-causal model - same parameters from the same "
          "seed, same mask, bit-identical logits")

    # It builds from cfg.dynamics and refuses what it does not implement,
    # including the strict mask config.py admits for the mask measurement.
    for key, other in (("mask", "strict_causal"), ("pos_encoding", "learned"),
                       ("output_head", "tied"), ("rope_base", 500.0)):
        cfg_x = dataclasses.replace(base, dynamics={**base.dynamics, key: other})
        assert _refused(build, cfg_x), f"built a model with dynamics.{key} = {other!r}"
    assert _refused(Dynamics, 384, 1, 5, 4, codes, 975, 1e4), "built 384 channels over 5 heads"
    assert _refused(Dynamics, 18, 1, 6, 4, codes, 975, 1e4), "built an odd RoPE head width"
    x, y = (torch.from_numpy(a) for a in assemble(lay, tok[:1], act[:1], codes))
    read = torch.from_numpy(lay.read)
    assert _refused(model, torch.cat((x, x), 1), mask, read), "ran past the RoPE tables"
    print("build: refuses a strict mask, learned positions, a tied head, another RoPE base, "
          "and head widths RoPE cannot pair")

    # RoPE is applied, and attention depends on distance, not absolute position.
    cos, sin = model.cos.double(), model.sin.double()
    q, k = torch.randn(2, 2 * cos.shape[1], dtype=torch.float64)

    def score(i: int, j: int) -> float:
        return float((rope(q, cos[i], sin[i]) * rope(k, cos[j], sin[j])).sum())
    assert abs(score(10, 3) - score(900, 893)) < 1e-5 < 1e-2 < abs(score(10, 3) - score(10, 4)), \
        (score(10, 3), score(900, 893), score(10, 4))

    # Causality, per block, on the model built from the config. Double
    # precision on CPU, so "unchanged" can mean bit-identical and a change far
    # down the sequence is not rounded away.
    m64 = build(base).double().eval()
    probes = (0, 63, 64, 65, 500, 973, 974)   # block edges, the action slots, the last position
    check_block_causal(m64, lay, mask, x, probes)
    # The controls: each wrong mask must fail the same assert.
    n = len(lay.src)
    shifted = torch.from_numpy((np.arange(n) + 1) // (cells + 1))
    wrong = {
        "no mask": torch.ones(n, n, dtype=torch.bool),
        "strictly causal": torch.ones(n, n, dtype=torch.bool).tril(),
        "blocks shifted one position": shifted[None, :] <= shifted[:, None],
        "no attention across blocks": torch.from_numpy(lay.block[None, :] == lay.block[:, None]),
    }
    for name, bad in wrong.items():
        assert _refused(check_block_causal, m64, lay, bad, x, probes), f"passed with {name}"
    print(f"causality: altering positions {', '.join(map(str, probes))} leaves every "
          f"earlier-block logit bit-identical and moves all of its own block and later; "
          f"fails with {', '.join(wrong)}")

    # The loss is scored at frame-token positions only: 64 a block, never the
    # action slot, and every target a frame code. Frame 0 is only context, and
    # action[0] is not in the input (asserted with the layout above).
    is_action = lay.src >= frames * cells
    assert is_action.sum() == lay.ctx and not is_action[lay.read].any(), "an action slot is scored"
    assert np.bincount(lay.block[lay.read]).tolist() == [cells] * lay.ctx
    assert lay.tgt.min() == cells and lay.tgt.max() < frames * cells, "a target outside frames 1..15"
    assert np.array_equal(np.bincount(lay.tgt // cells, minlength=frames), [0] + [cells] * lay.ctx)
    y_act = assemble(lay, tok[:1], np.full_like(act[:1], N_ACTIONS - 1), codes)[1]
    assert y_act.max() < codes and np.array_equal(y_act, y.numpy()), "an action reached a target"
    with torch.no_grad():
        got = frame_loss(m64, x, y, mask, read)
        full = m64(x, mask, torch.arange(n))
    assert m64(x, mask, read).shape == (1, lay.ctx * cells, codes)
    assert torch.equal(got, F.cross_entropy(full[0, lay.read].float(), y[0])), \
        "the loss read other positions"
    print(f"loss: {lay.ctx} x {cells} frame targets, read only at frame-token positions, "
          f"no action slot and no action target; frame 0 is context only")
    del m64

    # ---- training (item 4): no data needed
    _check_training(base)

    # ---- the data: the generated set if present, else the committed fixture
    cfg, shard_dir, fixture = data.self_check_config()
    shards = data.load_shards(shard_dir, cfg.data_hash)
    index = data.episode_index(shards)
    ctx, hold, vf = cfg.data["ctx"], cfg.sim["action_hold_steps"], cfg.data["val_fraction"]
    codes, cells = cfg.tokenizer["codebook_size"], math.prod(cfg.shapes.token_grid)
    assert len(layout(ctx, cells).src) == 975, "the config's window is not 975 positions"
    kind = "fixture" if fixture else "generated set"
    print(f"{kind}: {len(shards)} shards, {len(index)} episodes, ctx {ctx}")

    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        ident = _identity_cache(shards, cells)
        _write_cache(runs, "selfcheck", cfg, shards, ident)
        tokens = load_token_cache(cfg, shards, "selfcheck", runs)

        # The acceptance test: every action change in the stream the windows
        # read sits on the hold's beat, and a copy shifted one record either way
        # fails the same check.
        every = TokenWindowSampler(shards, index, tokens, ctx, hold)
        streams = every.action_streams()
        later = [(np.concatenate((a[:1], a[:-1])), s) for a, s in streams]
        earlier = [(np.concatenate((a[1:], a[-1:])), s) for a, s in streams]
        assert _refused(check_alignment, later, hold), "a shift of +1 passed the phase assertion"
        assert _refused(check_alignment, earlier, hold), "a shift of -1 passed the phase assertion"
        print(f"alignment: all {every.action_changes:,} action changes in the windows' stream "
              f"sit at step_idx % {hold} == 0; shifts of +1 and -1 both fail")

        # The split is data.is_val's, compared against it.
        want = {s: {ep.episode_id for ep in index if ep.length >= ctx + 1
                    and data.is_val(ep.episode_id, vf) == (s == "val")} for s in ("train", "val")}
        for s in ("train", "val"):
            if not want[s]:
                assert _refused(TokenWindowSampler, shards, index, tokens, ctx, hold, s, vf)
                assert _refused(data.WindowSampler, shards, index, ctx, s, vf)
                continue
            got = TokenWindowSampler(shards, index, tokens, ctx, hold, s, vf).windows.episodes
            assert {ep.episode_id for ep in got} == want[s], f"{s} episodes are not data.is_val's"
            # And in data.split_episodes' order, the row order preload uses.
            assert got == [ep for ep in data.split_episodes(index, s, vf) if ep.length >= ctx + 1]
        assert not want["train"] & want["val"]
        assert want["train"] | want["val"] == {ep.episode_id for ep in index}
        print(f"split: {len(want['train'])} train / {len(want['val'])} val episodes, "
              f"exactly data.is_val's in data.split_episodes' order, disjoint, nothing dropped")

        splits = [s for s in ("train", "val") if want[s]]
        print("windows against WindowSampler, identity-coded cache:")
        _check_windows(cfg, shards, index, tokens, splits, identity=True)

        # One window through assemble and the model: the sequence it feeds
        # round-trips to the token rows and actions of the records
        # WindowSampler returns at the same index, the rows item 1 asserted.
        i = len(every) // 2
        t, w = every[i], data.WindowSampler(shards, index, ctx)[i]
        ep = next(e for e in index if e.episode_id == t.episode_id)
        rows = ep.start + w.meta["step_idx"].astype(np.int64)
        lay_c = layout(ctx, cells)
        xw, yw = assemble(lay_c, t.tokens, t.actions, codes)
        pool = np.full((ctx + 1) * cells + ctx + 1, -1)
        pool[lay_c.src], pool[lay_c.tgt] = xw, yw
        assert np.array_equal(pool[:(ctx + 1) * cells].reshape(ctx + 1, cells), tokens[ep.shard][rows])
        assert np.array_equal(pool[(ctx + 1) * cells + 1:] - codes, w.meta["action"][1:])
        named = pool[:(ctx + 1) * cells].reshape(ctx + 1, cells)
        assert np.array_equal(named[:, 0] + 512 * named[:, 1], rows) and \
            (named[:, 2] == ep.shard).all(), "a row is another frame's"
        with torch.no_grad():
            lw = build(cfg).eval()(torch.from_numpy(xw)[None], attention_mask(lay_c),
                                   torch.from_numpy(lay_c.read))
        assert lw.shape == (1, ctx * cells, codes) and torch.isfinite(lw).all()
        print(f"model input: window {i} (episode {t.episode_id}, offset {t.offset}) assembles "
              f"to {len(xw)} positions that round-trip to its {ctx + 1} cache rows and "
              f"actions; the model returns {ctx * cells} x {codes} logits")

        # The cache refusals, on one shard: each edit must stop the load.
        one = shards[:1]
        first = ident[0]
        bad_code = first.copy()
        bad_code[0, 5] = codes
        cases = {
            "data_hash": ({"data_hash": "0" * 64}, first),
            "tokenizer_hash": ({"tokenizer_hash": "0" * 64}, first),
            "token_grid": ({"token_grid": [12, 12]}, first),
            "codebook_size": ({"codebook_size": 1024}, first),
            "run_id": ({"run_id": "another-run"}, first),
            "missing row": ({}, first[:-1]),
            "code >= codebook": ({}, bad_code),
            "missing shard": ({"shards": []}, first),
        }
        for n, (name, (edit, rows)) in enumerate(cases.items()):
            _write_cache(runs, f"bad{n}", cfg, one, [rows], edit)
            assert _refused(load_token_cache, cfg, one, f"bad{n}", runs), f"a bad {name} loaded"
        _write_cache(runs, "bitflip", cfg, one, [first])
        path = runs / "bitflip" / "tokens" / f"shard_{one[0].index:03d}.npy"
        flipped = np.load(path)
        flipped[3, 3, 3] ^= 1
        np.save(path, flipped)
        assert _refused(load_token_cache, cfg, one, "bitflip", runs), "a sha256 mismatch loaded"
        print(f"cache: refused on a wrong {', '.join(cases)} or sha256")

    real = ROOT / "runs" / TOKENIZER_RUN / "tokens" / "manifest.json"
    if fixture or not real.exists():
        print(f"R1 token cache: skipped - {'the fixture has none' if fixture else f'no {real}'}")
        print(f"dynamics self-check ok ({kind})")
        return
    tokens = load_token_cache(cfg, shards)
    assert all(len(t) == sh.frames for t, sh in zip(tokens, shards))
    print(f"R1 token cache: {sum(len(t) for t in tokens):,} rows, one per frame of every shard, "
          f"manifest and sha256 match")
    print("windows against WindowSampler, R1's cache:")
    _check_windows(cfg, shards, index, tokens, splits, identity=False)
    print("dynamics self-check ok")


if __name__ == "__main__":
    main()
