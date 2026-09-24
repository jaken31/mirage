"""The dynamics model: the next frame's tokens, from earlier frames and actions.

Phase 2, built in the order of `docs/phase2_structural_plan.md`. So far this file
holds item 1, the sequence layout and the token/action window sampler. The model
and its training loop are items 3 and 4. The model never sees a pixel: it reads a
tokenizer run's token cache, which `fsq_eval.write_token_cache` writes, and
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

    python -m mirage.dynamics    # self-check: the generated set if present, else the fixture
"""

import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

import numpy as np

from mirage import config, data

ROOT = Path(__file__).resolve().parent.parent

# The tokenizer Phase 2 inherits (`world_model_architecture.md`, "Phase 2
# inherits R1, and the encoder keeps `GroupNorm`"). Its token cache is the
# dynamics model's whole input.
TOKENIZER_RUN = "20260829-005439-r1"

# 3 torque directions ^ 2 joints, numbered 0..8 (`sim/policy.h`). Action `a` is
# token id `codebook_size + a`, after the frame codes.
N_ACTIONS = 9


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


# ----------------------------------------------------------------- self-check

def _refused(build: Callable[..., object], *args: object) -> bool:
    """True if `build(*args)` raises ValueError, the error every refusal here uses."""
    try:
        build(*args)
    except ValueError:
        return True
    return False


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


def _self_check() -> None:
    """Item 1's working-when list (`docs/phase2_structural_plan.md`, item 1)."""
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
    spec = importlib.util.spec_from_file_location("mask_probe", ROOT / "bench" / "mask_probe.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    for c in (4, 8, 15):
        mine, theirs = layout(c, cells), probe.layout("block", c + 1, cells)
        for f in ("src", "read", "tgt", "block"):
            assert np.array_equal(getattr(mine, f), getattr(theirs, f)), f"ctx {c}: {f} differs"
    print("layout: 15 blocks of 65 = 975 positions, 960 frame targets, action[t] before "
          "frame t, round-trips; equals bench/mask_probe.py's block layout at ctx 4, 8, 15")

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
    _self_check()
