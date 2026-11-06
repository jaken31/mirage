"""What holds a greedy rollout still? Four measurements, each on real val data.

r72-r74 showed three remedies for exposure bias leave rollouts 91-95% frozen
(a frozen frame repeats the one before, token for token; the truth's share is
20.1%). This probe asks where the freeze comes from, on gate row 1's population
(every val window's last frame, `dynamics_eval.population`) and on the gate's
rollout population (`dynamics_eval.val_episodes`):

- **onestep**: the frozen-frame and changed-cell shares of one greedy step from
  real context, beside the truth's with quiet-field flips held at the previous
  token (what a `--clean-flips` run trains toward).
- **actions**: the same step with the actions shuffled across windows, set to
  id 0, or with a stopped arm for context (every context frame replaced by the
  last one) and real actions. Accuracy is on moving cells (pixel field changed).
- **steps**: the rollout per step bucket - frozen share, changed cells, accuracy
  against the truth and against the seed frame held still, the decode ->
  encode round-trip mismatch, and P(frozen next | frozen).
- **swap**: one generated frame at the end of real context. Window i-1's greedy
  prediction replaces window i's last context frame - whole, on its moving
  cells only, or on its quiet cells only - beside the real frame with its quiet
  cells held at the frame before (the tokenizer's flips removed). "Frozen" is
  against the frame actually fed.

Decoding is untouched: `dynamics_eval.rollout` and row 1 stay greedy (decision 5).

    python bench/freeze_probe.py RUN_ID/CHECKPOINT [...] [--parts onestep actions steps swap]
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mirage import config, dynamics, dynamics_eval as de, fsq_eval, validator  # noqa: E402

CONFIG = ROOT / "mirage" / "configs" / "dyn192x4.json"
PARTS = ("onestep", "actions", "steps", "swap")
BUCKETS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, None)]
PEAK = 255.0


class Ctx:
    """Everything the parts share: config, splits, population, layout, device."""

    def __init__(self, config_path: Path):
        self.cfg = config.load(str(config_path))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.splits = dynamics.load_splits(self.cfg)
        self.pop = de.population(self.cfg, self.splits)
        self.lay = dynamics.layout(self.cfg.data["ctx"], math.prod(self.cfg.shapes.token_grid))
        self.mask = dynamics.attention_mask(self.lay).to(self.dev)
        self.read = torch.from_numpy(self.lay.read).to(self.dev)
        self.n, self.cells = self.pop["target"].shape


@torch.no_grad()
def predict(c: Ctx, model, edit=None, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Greedy last-frame predictions over every val window, and the last context frame fed.

    `edit(i, tokens, actions)` may change window i's tokens and actions in place.
    """
    codes = model.head.out_features
    out, fed = np.empty((c.n, c.cells), np.int64), np.empty((c.n, c.cells), np.int64)
    for j in range(0, c.n, batch):
        toks, acts = [], []
        for i in range(j, min(c.n, j + batch)):
            w = c.splits.val[i]
            tok, act = w.tokens.copy(), w.actions.copy()
            if edit is not None:
                edit(i, tok, act)
            toks.append(tok)
            acts.append(act)
            fed[i] = tok[-2]
        x, _ = (torch.from_numpy(a).to(c.dev)
                for a in dynamics.assemble(c.lay, np.stack(toks), np.stack(acts), codes))
        with dynamics._autocast(c.dev):
            logits = model(x, c.mask, c.read).float()
        out[j:j + len(toks)] = logits[:, -c.cells:].argmax(-1).cpu().numpy()
    return out, fed


def line(label: str, p: np.ndarray, fed: np.ndarray, c: Ctx, keep: np.ndarray | None = None) -> str:
    keep = np.ones(c.n, bool) if keep is None else keep
    pk, vk, tk, mk = p[keep], fed[keep], c.pop["target"][keep], ~c.pop["quiet"][keep]
    return (f"  {label:<11} frozen {(pk == vk).all(1).mean():5.1%}  changed {(pk != vk).mean():5.2%}  "
            f"moving-cell acc {(pk == tk)[mk].mean():6.2%}")


def onestep(c: Ctx, model) -> None:
    p, fed = predict(c, model)
    print(line("real", p, fed, c))


def actions(c: Ctx, model) -> None:
    perm = np.random.default_rng(0).permutation(c.n)

    def shuffle(i, tok, act):
        act[:] = c.splits.val[int(perm[i])].actions

    def zero(i, tok, act):
        act[:] = 0

    def static(i, tok, act):
        tok[:-1] = tok[-2]

    real = None
    for label, edit in (("real", None), ("shuffle", shuffle), ("zero", zero), ("static", static)):
        p, fed = predict(c, model, edit)
        real = p if real is None else real
        same = (p == real)[~c.pop["quiet"]].mean()
        print(f"{line(label, p, fed, c)}  same as real on moving {same:6.2%}")


def swap(c: Ctx, model) -> None:
    quiet, val = c.pop["quiet"], c.splits.val
    keep = np.array([val.windows.locate(i)[1] > 0 for i in range(c.n)])
    real, fed = predict(c, model)
    g, r, rp, q = (a[:-1][keep[1:]] for a in (real, c.pop["target"], c.pop["prev"], quiet))
    print(f"  its frame ctx-1 vs the real one: quiet cells differ {(g != r)[q].mean():.2%}, moving "
          f"cells differ {(g != r)[~q].mean():.2%}, moving cells left at the frame before "
          f"{(g == rp)[~q].mean():.2%} (real {(r == rp)[~q].mean():.2%})")

    def put(which):
        def edit(i, tok, act):
            if not keep[i]:
                return
            gen, qi = real[i - 1], quiet[i - 1]   # window i-1's target is frame ctx-1 here
            sel = {"gen": slice(None), "gen-moving": ~qi, "gen-quiet": qi}.get(which)
            if which == "clean":
                tok[-2, qi] = tok[-3, qi]
            else:
                tok[-2, sel] = gen[sel]
        return edit

    print(line("real", real, fed, c, keep))
    for which in ("gen", "gen-moving", "gen-quiet", "clean"):
        p, fd = predict(c, model, put(which))
        print(line(which, p, fd, c, keep), flush=True)


class Steps:
    """The rollout population, the tokenizer, and the truth's own per-step row."""

    def __init__(self, c: Ctx):
        palette = validator.load_palette(ROOT / c.cfg.sim["scene_xml"])
        self.eps = de.val_episodes(c.splits, len(palette.blocks))
        self.ctx, self.grid = c.cfg.data["ctx"], tuple(c.cfg.shapes.token_grid)
        self.steps = self.eps.actions.shape[1] - self.ctx
        self.tok, _ = fsq_eval.load_run(dynamics.TOKENIZER_RUN, c.cfg, c.dev)
        self.truth = self.per_step(c, self.eps.tokens)

    @torch.no_grad()
    def round_trip(self, c: Ctx, frames: np.ndarray) -> np.ndarray:
        """(E, T, cells) -> (E, T) share of cells whose token moves under decode -> uint8 -> encode."""
        flat = frames.reshape(-1, c.cells)
        px = de.decode_u8(self.tok, flat, self.grid, c.dev)
        back = []
        for i in range(0, len(px), 256):
            x = torch.from_numpy(px[i:i + 256]).to(c.dev).permute(0, 3, 1, 2).float() / PEAK
            back.append(self.tok.encode(x).reshape(len(x), -1).cpu().numpy())
        return (np.concatenate(back) != flat).mean(1).reshape(frames.shape[:2])

    def per_step(self, c: Ctx, gen: np.ndarray) -> dict[str, np.ndarray]:
        """Each measure as (E, steps), step k being frame ctx + k."""
        k, truth = self.ctx, self.eps.tokens
        g, before = gen[:, k:], gen[:, k - 1:-1]
        return {"frozen": (g == before).all(-1), "changed": (g != before).mean(-1),
                "acc": (g == truth[:, k:]).mean(-1), "still": (truth[:, k - 1:k] == truth[:, k:]).mean(-1),
                "rt": self.round_trip(c, gen)[:, k:]}

    def table(self, m: dict[str, np.ndarray]) -> None:
        print(f"  {'steps':<10}{'frozen':>8}{'changed':>9}{'acc':>8}{'still':>8}{'roundtrip':>11}")
        for a, b in BUCKETS:
            s, end = slice(a, b), (b or self.steps)
            lab = f"{a}-{end - 1}" if end - a > 1 else f"{a}"
            print(f"  {lab:<10}{m['frozen'][:, s].mean():>8.1%}{m['changed'][:, s].mean():>9.2%}"
                  f"{m['acc'][:, s].mean():>8.2%}{m['still'][:, s].mean():>8.2%}{m['rt'][:, s].mean():>11.2%}")
        fz = m["frozen"]
        stay = (fz[:, 1:] & fz[:, :-1]).sum() / max(fz[:, :-1].sum(), 1)
        streak = [max((len(s) for s in "".join("1" if v else "0" for v in r).split("0")), default=0)
                  for r in fz]
        first = [int(np.argmax(r)) if r.any() else -1 for r in fz]
        print(f"  P(frozen next | frozen) {stay:.1%}; first frozen step median {int(np.median(first))}; "
              f"longest frozen streak median {int(np.median(streak))} of {self.steps}")

    def run(self, c: Ctx, model) -> None:
        gen = de.rollout(model, self.eps.tokens[:, :self.ctx], self.eps.actions, self.steps, c.dev).frames
        self.table(self.per_step(c, gen))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="RUN_ID/best.pt or RUN_ID/model.pt under runs/")
    ap.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()
    c = Ctx(args.config)
    tgt, prev, quiet = c.pop["target"], c.pop["prev"], c.pop["quiet"]
    clean = np.where(quiet, prev, tgt)
    print(f"windows {c.n:,}; truth: frozen {(tgt == prev).all(1).mean():.1%}, changed {(tgt != prev).mean():.2%}; "
          f"flips held: frozen {(clean == prev).all(1).mean():.1%}, changed {(clean != prev).mean():.2%}; "
          f"persistence on moving cells {(prev == tgt)[~quiet].mean():.2%}")
    st = Steps(c) if "steps" in args.parts else None
    if st is not None:
        print(f"\ntruth rollout population ({len(st.eps.ids)} val episodes, {st.steps} steps)")
        st.table(st.truth)
    for ck in args.checkpoints:
        model, meta = de.load_checkpoint(ROOT / "runs" / ck, c.cfg, c.dev)
        print(f"\n=== {ck} (step {meta['step']:,})")
        for part in args.parts:
            print(f"[{part}]")
            st.run(c, model) if part == "steps" else {"onestep": onestep, "actions": actions,
                                                       "swap": swap}[part](c, model)


if __name__ == "__main__":
    main()
