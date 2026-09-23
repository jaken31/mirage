"""Strictly-causal against block-causal attention, at a fixed step budget.

The measurement ordered 2026-09-22 to run before Phase 2 build item 1, the
sequence layout: `docs/phase2_structural_plan.md`, "Before item 1", and the F-11
row under "Requirements at risk" in `docs/world_model_requirements.md`. The mask
is baked into every checkpoint trained under it, and the result decides whether
F-11's "predicts next token" still describes the model. **This is not Phase 2's
model.** `mirage/dynamics.py` does not exist, and this file deliberately does
not become it - it trains the shape r49 priced, twice, and compares.

**The two arms feed the identical token stream.** Decision 2's layout for a
16-frame window (`ctx + 1`, item 1's recommendation) is
`F_0, a_1, F_1, a_2, ..., a_15, F_15`: each action immediately before the 64
tokens of the frame it produced, both read at the same record index. The
leading `a_0` is dropped because it only explains the transition into `F_0`,
which no arm predicts. Both arms score the same 960 targets, frames 1..15 at
every cell. Only the mask and the position each target is read from differ:

- **strict** - 1,039 positions, each seeing only those before it. Frame t's
  cell i is read at the position just before it: `a_t` for cell 0, otherwise
  cell i - 1 of the same frame.
- **block** - the same stream stopped before `F_15`: 975 positions, **r49's
  priced length**. It is grouped into 15 blocks of 65: frame t's 64 tokens plus
  the action after them, `a_{t+1}`. Attention is full inside a block and causal
  across blocks. Frame t+1's cell i is read at the position holding frame t's
  cell i. So all 64 cells of frame t+1 come out of one pass, and each sees every
  earlier frame and `a_{t+1}` and nothing of frame t+1. A frame cannot be both
  the input and the target of its own block, and this is the arrangement that
  keeps it from being both.

**The score is the generated frame, not the teacher-forced one.** Under teacher
forcing the strict arm reads the true cells 0..i-1 of the very frame it is
predicting, which it never has at rollout. So its teacher-forced accuracy is
inflated by construction, and scoring on it would bias the comparison toward
strict. The primary score is held-out per-cell accuracy of the *generated* next
frame from ground-truth context, decoded greedily (decision 5), against the
persistence baseline on the same cells. The block arm decodes it in one pass,
the strict arm in 64. That is F-12's unit - one full next frame from previous
frames plus one action - and for the block arm it is exactly gate row 1's
measure. So the result still reads against F-11. The teacher-forced figure,
accuracy on the cells whose token changes, the marginal-frequency top-1 and
the held-out cross-entropy curve are reported alongside.

**The population:** every window of `data.WindowSampler`'s val split, scored on
its last frame. That is every frame of the 27 val episodes with a full 15-frame
history - t = 15..599, 15,795 frames, 1,010,880 cells. Persistence on those
exact cells is re-measured by `bench/token_stability_probe.py --episodes all
--first-target 15`, the instrument F-11 names, and `evaluate` asserts that it
agrees with this file's own count.

**The budget:** one epoch of train windows at batch 16, 17,294 steps. Both arms
see identical batches in identical order and start from identical weights at a
given seed (the arms have the same parameter shapes). They share one schedule
with no per-arm tuning. Two seeds per arm give the first seed spread ever
measured for a dynamics model.

**The decision rule, fixed before the run** (`decide`): the arm with the higher
mean primary score is selected only if its margin exceeds the seed spread - the
larger of the two arms' ranges across seeds. Otherwise the arms are not
separated and strictly-causal stands, because every item in the plan is written
for it. Block-causal selected means F-11's description gets a second, separately
dated amendment. Strictly-causal means it stands.

Keep the machine awake for the duration. `fsq._keep_awake` covers Windows; on
Linux, launch under `systemd-inhibit --what=idle:sleep`.

    python bench/mask_probe.py --self-check          # no dataset, no GPU needed
    python bench/mask_probe.py train --arm strict --seed 0 [--resume RUN_ID]
    python bench/mask_probe.py eval RUN_ID           # re-score a finished run
    python bench/mask_probe.py compare RUN_ID RUN_ID RUN_ID RUN_ID
"""
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mirage import config, data  # noqa: E402
from mirage.fsq import _keep_awake  # noqa: E402
from mirage.logging import Run  # noqa: E402
import dyn_size_probe  # noqa: E402  - the priced model, for the count cross-check
import token_stability_probe  # noqa: E402  - F-11's named persistence instrument

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "mirage" / "configs" / "base.json"
TOKENIZER_RUN = "20260829-005439-r1"   # R1, which Phase 2 inherits
R49_ROPE_UNTIED = 14_593_152           # runs.jsonl r49, the variant decisions 2 and 3 take
ROPE_BASE = 10_000.0
ARMS = ("strict", "block")

# The shared schedule. Phase 1's values for everything Phase 1 had
# (`fsq.train`), because that is the project's only precedent. Weight decay
# stays at its token 1e-4: decision 4 measures the train/val gap before any
# remedy is chosen, and heavier regularisation is one of the remedies.
BATCH = 16
LR, LR_FLOOR, WARMUP = 3e-4, 3e-5, 0.05
WEIGHT_DECAY = 1e-4
CLIP = 1.0            # bf16 insurance against a single spike, shared by both arms
EVAL_EVERY = 1000
EVAL_WINDOWS = 512    # per split, for the curve - fixed across every run
EVAL_SUBSET_SEED = 0  # not the run's seed: every curve is scored on the same windows


# ----------------------------------------------------------------- the layout

class Layout(NamedTuple):
    """One arm's arrangement of a window, as index arrays over a *pool*.

    The pool is one window flattened: `pool[t * 64 + i]` is frame t's cell i,
    and `pool[frames * 64 + t]` is `a_t` as token id `codes + action`. Both are
    read from the same record t, so the layout cannot shift an action against
    its frame.
    """
    arm: str
    src: np.ndarray    # (L,) pool index each input position holds
    read: np.ndarray   # (R,) input position each target is scored at
    tgt: np.ndarray    # (R,) pool index of each target
    block: np.ndarray  # (L,) attention group; a position sees groups <= its own

    def mask(self) -> torch.Tensor:
        """(L, L) bool, True where query row may attend to key column."""
        if self.arm == "strict":
            return torch.ones(len(self.src), len(self.src), dtype=torch.bool).tril()
        g = torch.from_numpy(self.block)
        return g[None, :] <= g[:, None]


def layout(arm: str, frames: int, cells: int) -> Layout:
    """The stream `F_0, a_1, F_1, ..., a_{n-1}, F_{n-1}`, cut and read per arm."""
    n = frames
    stream = list(range(cells))                          # F_0
    for t in range(1, n):
        stream.append(n * cells + t)                     # a_t, same record as F_t
        stream += range(t * cells, (t + 1) * cells)      # F_t
    stream = np.array(stream)
    pos = {int(p): k for k, p in enumerate(stream)}      # pool index -> position
    tgt = np.arange(cells, n * cells)                    # frames 1..n-1, every cell

    if arm == "strict":
        src = stream
        read = np.array([pos[int(p)] - 1 for p in tgt])
        block = np.arange(len(src))
    elif arm == "block":
        src = stream[:(n - 1) * (cells + 1)]             # stop before F_{n-1}
        # Frame t+1's cell i is read where frame t's cell i sits.
        read = np.array([pos[int(p) - cells] for p in tgt])
        block = np.arange(len(src)) // (cells + 1)
    else:
        raise ValueError(f"arm is {arm!r}, expected one of {ARMS}")
    return Layout(arm, src, read, tgt, block)


# ------------------------------------------------------------------ the model

def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (i, i + d/2) channel pair by an angle proportional to position.

    In the tables' precision, then back to the input's: at position ~1,000 a
    bf16 angle table has lost the low bits that tell neighbours apart.
    """
    a, b = x.to(cos.dtype).chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1).to(x.dtype)


class Block(nn.Module):
    """Pre-norm block with the probe's modules - r49's `nn.MultiheadAttention`
    unrolled into its own two projections, biases included, so RoPE can reach q
    and k and `F.scaled_dot_product_attention` can take the mask."""

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
        a = F.scaled_dot_product_attention(_rope(q, cos, sin), _rope(k, cos, sin), v,
                                           attn_mask=mask)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


class Dynamics(nn.Module):
    """r49's `rope_untied`: RoPE, untied output head, vocab 521 in and 512 out."""

    def __init__(self, d: int, layers: int, heads: int, vocab_in: int, vocab_out: int,
                 max_len: int, mlp_ratio: int = dyn_size_probe.MLP_RATIO) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_in, d)
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_out, bias=False)
        half = d // heads // 2
        f64 = dict(dtype=torch.float64)
        ang = torch.arange(max_len, **f64)[:, None] * ROPE_BASE ** (-torch.arange(half, **f64) / half)
        # Not persistent: derived from the shape, so no state_dict entry.
        self.register_buffer("cos", ang.cos().float(), persistent=False)
        self.register_buffer("sin", ang.sin().float(), persistent=False)
        # GPT-2's init, shared by both arms.
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, tok: torch.Tensor, mask: torch.Tensor,
                read: torch.Tensor) -> torch.Tensor:
        """(B, L) ids -> (B, len(read), vocab_out) logits at the read positions."""
        n = tok.shape[1]
        cos, sin = self.cos[:n], self.sin[:n]
        x = self.embed(tok)
        for blk in self.blocks:
            x = blk(x, mask, cos, sin)
        return self.head(self.norm(x[:, read]))


def build_model(cfg: config.Config, max_len: int) -> Dynamics:
    codes = cfg.tokenizer["codebook_size"]
    return Dynamics(cfg.dynamics["d_model"], cfg.dynamics["n_layers"],
                    dyn_size_probe.N_HEADS, codes + dyn_size_probe.N_ACTIONS, codes,
                    max_len)


# -------------------------------------------------------- scoring a window batch

def assemble(lay: Layout, tok: np.ndarray, act: np.ndarray, codes: int) -> tuple[np.ndarray, np.ndarray]:
    """(B, frames, cells) tokens + (B, frames) actions -> inputs (B, L), targets (B, R)."""
    b = tok.shape[0]
    pool = np.concatenate([tok.reshape(b, -1).astype(np.int64),
                           codes + act.astype(np.int64)], axis=1)
    return pool[:, lay.src], pool[:, lay.tgt]


@torch.no_grad()
def generate(model: Dynamics, lay: Layout, x: torch.Tensor, mask: torch.Tensor,
             cells: int, dummy: int = 0) -> torch.Tensor:
    """Greedy last frame from ground-truth context: (B, L) -> (B, cells).

    Every input position holding a cell of the frame being generated is
    overwritten with `dummy` first, and then with each generated token as it
    comes out. So no ground truth of the target frame is in the input at all,
    whatever the mask does. The block arm holds none of it and finishes in one
    pass; the strict arm needs one pass per cell, each cut to the prefix that
    can influence it.
    """
    x = x.clone()
    reads = lay.read[-cells:]
    held = {int(p): k for k, p in enumerate(lay.src)}
    feed = [held.get(int(p)) for p in lay.tgt[-cells:]]    # input slot of each cell, if any
    for k in (f for f in feed if f is not None):
        x[:, k] = dummy
    if all(f is None for f in feed):
        return model(x, mask, torch.from_numpy(reads).to(x.device)).argmax(-1)
    out = torch.empty(x.shape[0], cells, dtype=torch.long, device=x.device)
    for i, r in enumerate(reads):
        n = int(r) + 1
        out[:, i] = model(x[:, :n], mask[:n, :n], torch.tensor([n - 1], device=x.device))[:, 0].argmax(-1)
        if feed[i] is not None:
            x[:, feed[i]] = out[:, i]
    return out


# ---------------------------------------------------------------------- data

class Data:
    """R1's token cache and the meta records, addressed through `WindowSampler`.

    The sampler is the addressing authority: a window's (episode, offset) is read
    off the meta it returns, never recomputed here, and the window's actions are
    that same meta's `action` column. So the tokens and the actions of one
    window come from the records `WindowSampler` would have returned, by
    construction rather than by a second copy of its arithmetic.
    """

    def __init__(self, cfg: config.Config, run_id: str = TOKENIZER_RUN) -> None:
        self.cfg = cfg
        self.shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
        self.index = data.episode_index(self.shards)
        self.cells = math.prod(cfg.shapes.token_grid)
        self.codes = cfg.tokenizer["codebook_size"]
        self.ctx = cfg.data["ctx"]
        load_manifest(cfg, self.shards, run_id)
        tok_dir = ROOT / "runs" / run_id / "tokens"
        cache = [np.load(tok_dir / f"shard_{sh.index:03d}.npy").reshape(sh.frames, self.cells)
                 for sh in self.shards]
        self.tokens: dict[int, np.ndarray] = {}
        for ep in self.index:
            steps = np.asarray(self.shards[ep.shard].meta["step_idx"][ep.start:ep.start + ep.length])
            assert np.array_equal(steps, np.arange(ep.length)), \
                f"episode {ep.episode_id}: step_idx is not 0..{ep.length - 1}"
            self.tokens[ep.episode_id] = cache[ep.shard][ep.start:ep.start + ep.length]
        vf = cfg.data["val_fraction"]
        self.train = data.WindowSampler(self.shards, self.index, self.ctx, "train", vf)
        self.val = data.WindowSampler(self.shards, self.index, self.ctx, "val", vf)

    def windows(self, sampler: data.WindowSampler, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Window indices -> tokens (B, ctx+1, cells), actions (B, ctx+1), episode ids (B,)."""
        toks, acts, eids = [], [], []
        for i in idx:
            m = sampler[int(i)].meta
            eid, s0 = int(m["episode_id"][0]), int(m["step_idx"][0])
            assert (m["episode_id"] == eid).all(), f"window {i} straddles an episode"
            toks.append(self.tokens[eid][s0:s0 + sampler.window])
            acts.append(np.asarray(m["action"]))
            eids.append(eid)
        return np.stack(toks), np.stack(acts), np.array(eids)


def load_manifest(cfg: config.Config, shards, run_id: str) -> dict:
    """R1's cache manifest, refused on any provenance mismatch.

    The refusals `fsq_eval.load_run` makes for a checkpoint, made for the cache:
    a cache written at another `data_hash` or by another tokenizer trains
    silently otherwise, and one from the 96x96 fork fails somewhere downstream
    of the load rather than at it. Every shard's bytes are checked against the
    manifest's sha256 too, so the cache is the one R1 wrote.
    """
    path = ROOT / "runs" / run_id / "tokens" / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    for key, want in (("data_hash", cfg.data_hash), ("tokenizer_hash", cfg.tokenizer_hash),
                      ("token_grid", list(cfg.shapes.token_grid)),
                      ("codebook_size", cfg.tokenizer["codebook_size"]), ("run_id", run_id)):
        if man[key] != want:
            raise ValueError(f"{path}: {key} is {man[key]!r}, expected {want!r}")
    by_index = {s["shard"]: s for s in man["shards"]}
    for sh in shards:
        rec = by_index[sh.index]
        toks = np.load(path.parent / rec["file"])
        if len(toks) != sh.frames or rec["frames"] != sh.frames:
            raise ValueError(f"shard {sh.index}: {len(toks)} token rows, {sh.frames} frames")
        if hashlib.sha256(toks.tobytes()).hexdigest() != rec["sha256"]:
            raise ValueError(f"shard {sh.index}: token bytes do not match the manifest sha256")
    return man


def action_phase(shards, hold: int, shift: int = 0) -> tuple[set[int], int]:
    """Phases mod `hold` at which the action stream changes, and how many changes.

    `data._self_check`'s assertion, over the records the windows read. `shift`
    moves each action one record later within its episode - the negative
    control, which must land off phase 0.
    """
    phases: set[int] = set()
    changes = 0
    for sh in shards:
        a = np.asarray(sh.meta["action"]).astype(np.int64)
        st = np.asarray(sh.meta["step_idx"]).astype(np.int64)
        ep = np.asarray(sh.meta["episode_id"])
        if shift:
            a = np.concatenate(([a[0]], a[:-1]))
        changed = np.concatenate(([True], (a[1:] != a[:-1]) | (ep[1:] != ep[:-1])))
        phases |= set(np.unique(st[changed] % hold).tolist())
        changes += int(changed.sum())
    return phases, changes


# --------------------------------------------------------------------- train

def gpu_state() -> dict | None:
    """SM clock and power, which is what a compute-bound timing is gated on -
    never `pstate` (refuted 2026-08-23). The fields `bench/gpu_probe.py` samples."""
    fields = ["clocks.current.sm", "clocks.max.sm", "power.draw", "enforced.power.limit",
              "temperature.gpu", "pstate"]
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        sm, smx, pw, lim, temp, ps = [v.strip() for v in out.splitlines()[0].split(",")]
        return {"sm_mhz": int(sm), "sm_max_mhz": int(smx), "power_w": float(pw),
                "power_cap_w": float(lim), "temp_c": float(temp), "pstate": ps}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def environment() -> dict:
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"platform": platform.platform(), "python": platform.python_version(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": dev}


@torch.no_grad()
def curve_point(model: Dynamics, lay: Layout, mask: torch.Tensor, d: Data,
                subsets: dict[str, np.ndarray], dev: torch.device) -> dict:
    """Teacher-forced cross-entropy over all targets, and on the last frame alone."""
    model.eval()
    read = torch.from_numpy(lay.read).to(dev)
    out = {}
    for split, idx in subsets.items():
        sampler = d.train if split == "train" else d.val
        ce_all = ce_last = hit_last = n_win = 0.0
        for j in range(0, len(idx), 32):
            tok, act, _ = d.windows(sampler, idx[j:j + 32])
            x, y = assemble(lay, tok, act, d.codes)
            x, y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(x, mask, read).float()
            ce = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
            ce_all += float(ce.mean(1).sum())
            ce_last += float(ce[:, -d.cells:].mean(1).sum())
            hit_last += float((logits[:, -d.cells:].argmax(-1) == y[:, -d.cells:]).float().mean(1).sum())
            n_win += len(x)
        out[f"{split}_ce"] = ce_all / n_win
        out[f"{split}_ce_last"] = ce_last / n_win
        out[f"{split}_acc_tf_last"] = hit_last / n_win
    out["gap_ce"] = out["val_ce"] - out["train_ce"]
    model.train()
    return out


def train(arm: str, seed: int, cfg: config.Config, resume: str | None = None,
          steps: int | None = None) -> str:
    _keep_awake()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Data(cfg)
    hold = cfg.sim["action_hold_steps"]
    phases, changes = action_phase(d.shards, hold)
    assert phases == {0}, f"action changes at phases {sorted(phases)} mod {hold}"
    shifted, _ = action_phase(d.shards, hold, shift=1)
    assert shifted != {0}, "a one-record shift passed the phase assertion - it cannot catch one"
    print(f"alignment: {changes:,} action changes, all at step_idx % {hold} == 0; "
          f"a shift of one lands at {sorted(shifted)[:4]}...")

    lay = layout(arm, d.ctx + 1, d.cells)
    mask = lay.mask().to(dev)
    read = torch.from_numpy(lay.read).to(dev)
    total = steps or len(d.train) // BATCH
    warm = max(1, int(WARMUP * total))

    torch.manual_seed(seed)
    model = build_model(cfg, len(lay.src)).to(dev)
    params = sum(p.numel() for p in model.parameters())
    assert params == R49_ROPE_UNTIED, f"{params:,} parameters, r49 priced {R49_ROPE_UNTIED:,}"
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # One epoch without replacement; the same order for both arms at a seed.
    order = np.random.default_rng(seed).permutation(len(d.train))[:total * BATCH].reshape(total, BATCH)
    ev = np.random.default_rng(EVAL_SUBSET_SEED)
    subsets = {"val": np.sort(ev.choice(len(d.val), EVAL_WINDOWS, replace=False)),
               "train": np.sort(ev.choice(len(d.train), EVAL_WINDOWS, replace=False))}

    def lr_at(step: int) -> float:
        if step < warm:
            return LR * (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return LR_FLOOR + 0.5 * (LR - LR_FLOOR) * (1 + math.cos(math.pi * p))

    hashes = {"data_hash": cfg.data_hash, "tokenizer_hash": cfg.tokenizer_hash,
              "dynamics_hash": cfg.dynamics_hash}
    knobs = dict(arm=arm, seed=seed, tokenizer_run=TOKENIZER_RUN, frames=d.ctx + 1,
                 seq_len=len(lay.src), targets=len(lay.tgt), params=params,
                 d_model=cfg.dynamics["d_model"], n_layers=cfg.dynamics["n_layers"],
                 n_heads=dyn_size_probe.N_HEADS, mlp_ratio=dyn_size_probe.MLP_RATIO,
                 rope_base=ROPE_BASE, head="untied", precision="bf16 autocast",
                 steps=total, batch=BATCH, lr=LR, lr_floor=LR_FLOOR, warmup=WARMUP,
                 weight_decay=WEIGHT_DECAY, clip=CLIP, eval_every=EVAL_EVERY,
                 eval_windows=EVAL_WINDOWS, train_windows=len(d.train),
                 action_changes=changes, resumed_from=resume, **environment())

    start = 0
    if resume is not None:
        ck = torch.load(ROOT / "runs" / resume / "model.pt", map_location=dev, weights_only=True)
        for k in ("arm", "seed", "steps", "batch", "lr", "lr_floor", "warmup",
                  "weight_decay", "clip", "tokenizer_run", "seq_len"):
            assert ck["knobs"][k] == knobs[k], f"resume {resume}: {k} is {ck['knobs'][k]}, not {knobs[k]}"
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
        print(f"resumed {resume} at step {start:,}/{total:,}")

    print(f"{arm} seed {seed}: {params:,} params, {len(lay.src)} positions, "
          f"{len(lay.tgt)} targets/window, {total:,} steps at batch {BATCH} on {dev}")
    with Run(f"mask-{arm}-s{seed}", hashes, config=knobs) as run:
        wall, step_s = time.perf_counter(), []
        cuda = dev.type == "cuda"
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        for step in range(start, total):
            timed = cuda and step % 50 == 0
            if timed:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            tok, act, _ = d.windows(d.train, order[step])
            x, y = assemble(lay, tok, act, d.codes)
            x, y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(x, mask, read)
            loss = F.cross_entropy(logits.float().transpose(1, 2), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), CLIP))
            opt.step()
            if timed:
                torch.cuda.synchronize()
                step_s.append(time.perf_counter() - t0)
            done = step + 1
            if done % 100 == 0:
                run.log({"step": done, "loss": float(loss.detach()), "lr": lr_at(step), "grad_norm": gnorm,
                         "wall_s": round(time.perf_counter() - wall, 1)})
            if done % EVAL_EVERY == 0 or done == total:
                pt = curve_point(model, lay, mask, d, subsets, dev)
                run.log({"step": done, **pt, "gpu": gpu_state(),
                         "wall_s": round(time.perf_counter() - wall, 1)})
                print(f"  step {done:>6,}/{total:,}  val ce {pt['val_ce']:.4f}  "
                      f"train ce {pt['train_ce']:.4f}  val acc(tf, last) "
                      f"{pt['val_acc_tf_last']:.2%}  {time.perf_counter() - wall:7.1f}s", flush=True)
                torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(),
                            "step": done, "knobs": knobs, **hashes}, run.dir / "model.pt")
        train_info = {"train_s": round(time.perf_counter() - wall, 1),
                      "ms_per_step_median": round(1e3 * float(np.median(step_s)), 1) if step_s else None,
                      "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                       if cuda else None)}
        (run.dir / "train.json").write_text(json.dumps({**knobs, **hashes, **train_info}, indent=1)
                                            + "\n", encoding="utf-8", newline="\n")
        run.log({"final_train": True, **train_info})
        run_id = run.run_id
    evaluate(run_id, cfg, d)
    return run_id


# ------------------------------------------------------------------ evaluate

@torch.no_grad()
def evaluate(run_id: str, cfg: config.Config, d: Data | None = None,
             stride: int = 1, batch: int = 64) -> dict:
    """Score a finished run on the population, and write `result.json`.

    `stride` > 1 scores the generated frame on every stride-th window only - the
    fallback for when the strict arm's 64-pass decode costs too much. It is
    recorded in the result, and every other number is still over all windows.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = d or Data(cfg)
    run_dir = ROOT / "runs" / run_id
    info = json.loads((run_dir / "train.json").read_text(encoding="utf-8"))
    ck = torch.load(run_dir / "model.pt", map_location=dev, weights_only=True)
    assert ck["step"] == info["steps"], f"{run_id} stopped at step {ck['step']} of {info['steps']}"
    for k in ("data_hash", "tokenizer_hash", "dynamics_hash"):
        assert ck[k] == getattr(cfg, k), f"{run_id}: {k} differs from the config"
    lay = layout(info["arm"], d.ctx + 1, d.cells)
    model = build_model(cfg, len(lay.src)).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mask = lay.mask().to(dev)
    last = torch.from_numpy(lay.read[-d.cells:]).to(dev)

    # The marginal-frequency baseline, fitted on the train split's frames.
    train_eps = data.split_episodes(d.index, "train", cfg.data["val_fraction"])
    counts = np.bincount(np.concatenate([d.tokens[e.episode_id].ravel() for e in train_eps]),
                         minlength=d.codes)
    top1 = int(counts.argmax())

    n = len(d.val)
    target = np.empty((n, d.cells), np.int64)
    prev = np.empty((n, d.cells), np.int64)
    tf = np.empty((n, d.cells), np.int64)
    gen = np.full((n, d.cells), -1, np.int64)
    eid = np.empty(n, np.int64)
    ce_sum, t0, gen_s = 0.0, time.perf_counter(), 0.0
    for j in range(0, n, batch):
        idx = np.arange(j, min(j + batch, n))
        tok, act, eid[idx] = d.windows(d.val, idx)
        target[idx], prev[idx] = tok[:, -1], tok[:, -2]
        x = torch.from_numpy(assemble(lay, tok, act, d.codes)[0]).to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(x, mask, last).float()
        ce_sum += float(F.cross_entropy(logits.transpose(1, 2),
                                        torch.from_numpy(target[idx]).to(dev), reduction="sum"))
        tf[idx] = logits.argmax(-1).cpu().numpy()
        keep = np.flatnonzero(idx % stride == 0)
        if len(keep):
            g0 = time.perf_counter()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                g = generate(model, lay, x[torch.from_numpy(keep).to(dev)], mask, d.cells)
            gen[idx[keep]] = g.cpu().numpy()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            gen_s += time.perf_counter() - g0
    eval_s = time.perf_counter() - t0
    if info["arm"] == "block":
        # The block arm reads no cell of the target frame, so generating it is
        # the teacher-forced pass - a control that this is true, not a shortcut.
        assert np.array_equal(gen[gen >= 0], tf[gen >= 0]), "block arm: generated != teacher-forced"

    probe = token_stability_probe.probe(TOKENIZER_RUN, cfg, None, d.ctx)
    persistence = float((prev == target).mean())
    assert probe["transitions"] == target.size and abs(probe["persistence"] - persistence) < 1e-12, \
        f"population disagrees with token_stability_probe: {probe} against {persistence}"

    g = gen >= 0
    rows = g.all(1)
    changed = target != prev
    per_ep = {int(e): float((gen[rows & (eid == e)] == target[rows & (eid == e)]).mean())
              for e in np.unique(eid)}
    score = {
        "acc_gen": float((gen[rows] == target[rows]).mean()),
        "acc_tf": float((tf == target).mean()),
        "acc_tf_changed": float((tf == target)[changed].mean()),
        "acc_gen_changed": float((gen[rows] == target[rows])[changed[rows]].mean()),
        "acc_gen_unchanged": float((gen[rows] == target[rows])[~changed[rows]].mean()),
        "copy_overlap_gen": float((gen[rows] == prev[rows]).mean()),
        "frame_exact_gen": float((gen[rows] == target[rows]).all(1).mean()),
        "ce_last_tf": ce_sum / target.size,
        "acc_gen_per_episode": per_ep,
    }
    population = {
        "split": "val", "episodes": int(len(np.unique(eid))), "windows": n,
        "first_target": d.ctx, "cells": int(target.size), "gen_stride": stride,
        "gen_cells": int(gen[rows].size), "changed_share": float(changed.mean()),
        "persistence": persistence, "persistence_probe": probe["persistence"],
        "persistence_on_gen_cells": float((prev[rows] == target[rows]).mean()),
        "marginal_top1_code": top1, "marginal_top1_acc": float((target == top1).mean()),
    }
    result = {**info, "run_id": run_id, "population": population, "score": score,
              "curve": curve(run_id), "eval_s": round(eval_s, 1), "gen_s": round(gen_s, 1),
              "eval_gpu": gpu_state()}
    np.save(run_dir / "generated.npy", gen.astype(np.int16))
    (run_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    print(f"{run_id}: generated {score['acc_gen']:.4%}  teacher-forced {score['acc_tf']:.4%}  "
          f"persistence {persistence:.4%}  changed-cell {score['acc_gen_changed']:.2%}  "
          f"marginal top-1 {population['marginal_top1_acc']:.2%}  ({eval_s:.0f}s, decode {gen_s:.0f}s)")
    return result


def curve(run_id: str) -> list[dict]:
    """The held-out loss curve, followed back through any `--resume` seams."""
    points, rid = [], run_id
    while rid is not None:
        run_dir = ROOT / "runs" / rid
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if "val_ce" in r:
                points.append({k: r[k] for k in ("step", "val_ce", "train_ce", "gap_ce",
                                                 "val_acc_tf_last", "gpu")})
        rid = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["config"]["resumed_from"]
    return sorted({p["step"]: p for p in points}.values(), key=lambda p: p["step"])


# ------------------------------------------------------------------- compare

def decide(scores: dict[str, list[float]]) -> dict:
    """The rule fixed before the run. `scores` is arm -> primary score per seed."""
    assert set(scores) == set(ARMS), f"need both arms, got {sorted(scores)}"
    assert all(len(v) >= 2 for v in scores.values()), "a seed spread needs two seeds per arm"
    mean = {a: float(np.mean(v)) for a, v in scores.items()}
    spread = max(float(max(v) - min(v)) for v in scores.values())
    margin = mean["block"] - mean["strict"]
    separated = abs(margin) > spread
    selected = "block" if separated and margin > 0 else "strict"
    return {"mean": mean, "margin_block_minus_strict": margin, "seed_spread": spread,
            "separated": separated, "selected": selected,
            "f11": ("second, separately dated amendment to the description" if selected == "block"
                    else "description stands")}


def compare(run_ids: list[str]) -> dict:
    res = [json.loads((ROOT / "runs" / r / "result.json").read_text(encoding="utf-8")) for r in run_ids]
    for key in ("data_hash", "tokenizer_hash", "dynamics_hash", "steps", "batch", "lr",
                "tokenizer_run", "frames"):
        vals = {json.dumps(r[key]) for r in res}
        assert len(vals) == 1, f"runs disagree on {key}: {vals}"
    pops = {json.dumps({k: v for k, v in r["population"].items() if k != "gen_cells"}) for r in res}
    assert len(pops) == 1, "runs were scored on different populations"
    scores: dict[str, list[float]] = {}
    for r in sorted(res, key=lambda r: (r["arm"], r["seed"])):
        scores.setdefault(r["arm"], []).append(r["score"]["acc_gen"])
        s = r["score"]
        print(f"{r['arm']:<7} seed {r['seed']}  {r['run_id']:<32} gen {s['acc_gen']:.4%}  "
              f"tf {s['acc_tf']:.4%}  changed {s['acc_gen_changed']:.2%}  "
              f"copy-overlap {s['copy_overlap_gen']:.2%}  ce {s['ce_last_tf']:.4f}")
    p = res[0]["population"]
    print(f"persistence {p['persistence']:.4%}, marginal top-1 {p['marginal_top1_acc']:.2%}, "
          f"over {p['cells']:,} cells of {p['windows']:,} val windows")
    verdict = decide(scores)
    print(json.dumps(verdict, indent=1))
    return verdict


# ---------------------------------------------------------------- self-check

def _self_check() -> None:
    """No dataset and no GPU: the layout, the mask, the model and the rule."""
    torch.manual_seed(0)
    frames, cells, codes = 16, 64, 512
    lays = {a: layout(a, frames, cells) for a in ARMS}
    s, b = lays["strict"], lays["block"]

    # The arms feed one stream, score one target set, and differ in read-out only.
    assert len(s.src) == 64 + 15 * 65 == 1039 and len(b.src) == 15 * 65 == 975
    assert np.array_equal(b.src, s.src[:len(b.src)]), "block's stream is not strict's, cut"
    assert np.array_equal(s.tgt, b.tgt) and len(s.tgt) == 15 * 64
    assert np.array_equal(s.src[s.read + 1], s.tgt), "strict reads each target one position early"
    assert np.array_equal(b.src[b.read], b.tgt - cells), "block reads frame t+1 where frame t sits"
    # Decision 2's order: every action immediately before the frame it produced.
    for t in range(1, frames):
        k = int(np.flatnonzero(s.src == frames * cells + t)[0])
        assert np.array_equal(s.src[k + 1:k + 1 + cells], np.arange(t * cells, (t + 1) * cells))
    assert set(np.bincount(b.block).tolist()) == {65}
    print("layout: one stream, strict 1,039 / block 975 positions, 960 shared targets, "
          "action[t] before frame t")

    # The model is the priced one, module for module.
    cfg = config.load(CONFIG)
    full = build_model(cfg, 1039)
    priced = dyn_size_probe.Dynamics(cfg.dynamics["d_model"], cfg.dynamics["n_layers"],
                                     dyn_size_probe.N_HEADS, codes + dyn_size_probe.N_ACTIONS,
                                     codes, 975, learned_pos=False, tied=False)
    n_full = sum(p.numel() for p in full.parameters())
    assert n_full == sum(p.numel() for p in priced.parameters()) == R49_ROPE_UNTIED, n_full
    print(f"model: {n_full:,} parameters, r49's rope_untied exactly")
    del full, priced

    # RoPE: a score depends on the offset between two positions, not where they sit.
    m = Dynamics(64, 2, 2, codes + 9, codes, 1039).double().eval()
    q, k = torch.randn(1, 1, 1, 32, dtype=torch.float64), torch.randn(1, 1, 1, 32, dtype=torch.float64)
    def dot(i: int, j: int) -> float:
        return float((_rope(q, m.cos[i], m.sin[i]) * _rope(k, m.cos[j], m.sin[j])).sum())
    assert abs(dot(10, 3) - dot(900, 893)) < 1e-5 < 1e-2 < abs(dot(10, 3) - dot(10, 4)), \
        (dot(10, 3), dot(900, 893), dot(10, 4))

    # Causality, per position for strict and per block for block. Double
    # precision on CPU so "unchanged" can mean bit-identical.
    tok = torch.randint(0, codes, (1, frames, cells)).numpy()
    act = torch.randint(0, 9, (1, frames)).numpy()
    for arm, lay in lays.items():
        mask = lay.mask()
        x = torch.from_numpy(assemble(lay, tok, act, codes)[0])
        every = torch.arange(len(lay.src))
        base = m(x, mask, every)
        for pos in (0, 63, 64, 65, 500, 973, len(lay.src) - 2):
            y = x.clone()
            y[0, pos] = (y[0, pos] + 1) % codes
            moved = (m(y, mask, every) != base).any(-1)[0]
            first = pos if arm == "strict" else int(lay.block[pos]) * 65
            assert not moved[:first].any(), f"{arm}: altering position {pos} moved an earlier logit"
            assert moved[pos], f"{arm}: altering position {pos} left its own logit alone"
            if arm == "block":
                sib = first + (1 if pos == first else 0)
                assert moved[sib], f"block: position {pos} did not reach its block sibling {sib}"
        # No target is visible where it is scored: altering the input slot that
        # holds a target leaves that target's logit bit-identical.
        slot = {int(p): i for i, p in enumerate(lay.src)}
        leaks = 0
        for kk in range(0, len(lay.tgt), 37):
            if int(lay.tgt[kk]) in slot:
                r = torch.tensor([int(lay.read[kk])])
                y = x.clone()
                y[0, slot[int(lay.tgt[kk])]] = (y[0, slot[int(lay.tgt[kk])]] + 1) % codes
                assert torch.equal(m(y, mask, r), m(x, mask, r)), f"{arm}: target {kk} leaks"
                leaks += 1
        assert leaks >= 20, f"{arm}: only {leaks} targets tested"
        # And the premise of scoring the generated frame: under teacher forcing
        # strict sees the true earlier cells of its own target frame, block never.
        # Frame 14, which both arms hold in their input.
        fr = slice((frames - 3) * cells, (frames - 2) * cells)
        reads = torch.from_numpy(lay.read[fr])
        y = x.clone()
        for p in lay.tgt[fr][:8]:
            y[0, slot[int(p)]] = (y[0, slot[int(p)]] + 1) % codes
        seen = not torch.equal(m(y, mask, reads), m(x, mask, reads))
        assert seen == (arm == "strict"), f"{arm}: in-frame visibility is {seen}"
    print("mask: strict causal per position, block causal per block and full inside it; "
          "no target visible where it is scored")

    # Generation: never reads the target frame's truth, and strict's cut prefixes
    # match a full-length greedy decode token for token.
    for arm, lay in lays.items():
        mask = lay.mask()
        x = torch.from_numpy(assemble(lay, tok, act, codes)[0])
        g = generate(m, lay, x, mask, cells)
        other = tok.copy()
        other[0, -1] = (other[0, -1] + 7) % codes
        assert torch.equal(g, generate(m, lay, torch.from_numpy(assemble(lay, other, act, codes)[0]),
                                       mask, cells)), f"{arm}: generation read the target frame"
        if arm == "block":
            tf = m(x, mask, torch.from_numpy(lay.read[-cells:])).argmax(-1)
            assert torch.equal(g, tf), "block: one-pass generation != teacher-forced argmax"
        else:
            y = x.clone()
            held = {int(p): i for i, p in enumerate(lay.src)}
            slots = [held[int(p)] for p in lay.tgt[-cells:]]
            y[0, slots] = 0
            for i, r in enumerate(lay.read[-cells:]):
                y[0, slots[i]] = m(y, mask, torch.tensor([int(r)]))[0, 0].argmax()
                assert int(y[0, slots[i]]) == int(g[0, i]), f"strict: cut decode differs at cell {i}"
    print("generate: no target-frame truth in the input; strict's cut prefixes match a full decode")

    # The rule.
    assert decide({"strict": [0.90, 0.91], "block": [0.93, 0.935]})["selected"] == "block"
    assert decide({"strict": [0.90, 0.91], "block": [0.905, 0.92]})["separated"] is False
    assert decide({"strict": [0.90, 0.91], "block": [0.905, 0.92]})["selected"] == "strict"
    assert decide({"strict": [0.95, 0.951], "block": [0.90, 0.901]})["selected"] == "strict"
    print("decide: block only when it clears the seed spread; otherwise strict stands")

    # The phase assertion and its negative control, on whatever set is here.
    fcfg, shard_dir, fixture = data.self_check_config()
    shards = data.load_shards(shard_dir, fcfg.data_hash)
    hold = fcfg.sim["action_hold_steps"]
    phases, changes = action_phase(shards, hold)
    assert phases == {0} and action_phase(shards, hold, shift=1)[0] != {0}
    print(f"alignment: {changes:,} action changes at phase 0 on the "
          f"{'fixture' if fixture else 'generated set'}; a shift of one fails")
    print("mask_probe self-check ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-check", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--resume", metavar="RUN_ID")
    t.add_argument("--steps", type=int, help="override the one-epoch budget (smoke runs only)")
    e = sub.add_parser("eval")
    e.add_argument("run_id")
    e.add_argument("--stride", type=int, default=1)
    c = sub.add_parser("compare")
    c.add_argument("run_ids", nargs="+")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return
    cfg = config.load(CONFIG)
    if a.cmd == "train":
        train(a.arm, a.seed, cfg, a.resume, a.steps)
    elif a.cmd == "eval":
        evaluate(a.run_id, cfg, stride=a.stride)
    elif a.cmd == "compare":
        compare(a.run_ids)
    else:
        ap.error("pass --self-check or a subcommand")


if __name__ == "__main__":
    main()
