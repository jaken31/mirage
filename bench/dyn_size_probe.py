"""How big is the Phase 2 dynamics model, and what does one epoch cost?

Records the sizing an earlier session computed in conversation and never wrote
down: parameter count, sequence length, token-cache size, distance from the
Chinchilla-optimal token budget, and fp32 against bf16 throughput.  Nothing here
is a Phase 2 decision - it prices the model the docs already specify
(`world_model_ingredients.md`: d_model 384, 8 layers, 6 heads, plain MHA;
`configs/base.json` `dynamics`) so that a plan can be written against numbers
instead of recollections.

**Four variants, on purpose.** The parameter count depends on two choices that
are still open - `docs/handoff_tokenizer_decision.md` section 7, decision 4,
sequence layout and position encoding, which it calls irreversible.  Quoting one
number would decide it by accident, so the table prices tied against untied
output embeddings and RoPE against learned positions, and the plan can pick.

**The throughput number is measured, not modelled**, because that is the one an
estimate gets wrong: fp32 matmul on this machine does not use TF32 by default,
and the thermal state moves the answer (repo `CLAUDE.md`).  `bench/gpu_probe.py`
is what says whether the machine was clocked up when this ran.

    python bench/dyn_size_probe.py [--batch 16] [--steps 20]
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mirage import config, data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_HEADS = 6          # world_model_ingredients.md; not in config
MLP_RATIO = 4        # the transformer default the ingredients doc assumes
N_ACTIONS = 9        # 3 levels ^ 2 joints, sim/policy.h
CHINCHILLA = 20      # tokens per parameter


class Block(nn.Module):
    """Pre-norm transformer block, plain MHA - what the ingredients doc names.

    `nn.MultiheadAttention` rather than a hand-rolled block: this probe exists
    to count parameters and time a matmul, and a second attention implementation
    written here would be a third place the shape can be wrong.
    """

    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, MLP_RATIO * d), nn.GELU(),
                                 nn.Linear(MLP_RATIO * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Dynamics(nn.Module):
    """The sizing model.  Not the Phase 2 implementation - `mirage/dynamics.py`
    does not exist yet and this deliberately does not become it."""

    def __init__(self, d: int, layers: int, heads: int, vocab_in: int,
                 vocab_out: int, seq: int, learned_pos: bool, tied: bool) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_in, d)
        self.pos = nn.Parameter(torch.zeros(seq, d)) if learned_pos else None
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        # Tied: reuse the input embedding's first `vocab_out` rows as the output
        # projection.  Only sound because the frame codes are the first block of
        # the vocabulary and the action tokens are appended after them.
        self.head = None if tied else nn.Linear(d, vocab_out, bias=False)
        self.vocab_out = vocab_out

    def forward(self, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(tok)
        if self.pos is not None:
            x = x + self.pos[:tok.shape[1]]
        for b in self.blocks:
            x = b(x, mask)
        x = self.norm(x)
        return (x @ self.embed.weight[:self.vocab_out].T if self.head is None
                else self.head(x))


def _time_step(model: nn.Module, tok: torch.Tensor, mask: torch.Tensor,
               dtype: torch.dtype | None, steps: int) -> float:
    """Mean seconds per forward+backward step, after a warmup."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def one() -> None:
        with torch.autocast("cuda", dtype=dtype or torch.float32,
                            enabled=dtype is not None):
            logits = model(tok, mask)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                tok.reshape(-1).clamp(max=logits.shape[-1] - 1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def probe(cfg: config.Config, batch: int, steps: int) -> dict:
    d = cfg.dynamics["d_model"]
    layers = cfg.dynamics["n_layers"]
    ctx = cfg.data["ctx"]
    per_frame = cfg.shapes.token_grid[0] * cfg.shapes.token_grid[1]
    seq = ctx * (per_frame + 1)          # F-11: interleaved frame and action tokens
    vocab_in = cfg.tokenizer["codebook_size"] + N_ACTIONS
    vocab_out = cfg.tokenizer["codebook_size"]

    variants = {}
    for learned_pos in (False, True):
        for tied in (True, False):
            m = Dynamics(d, layers, N_HEADS, vocab_in, vocab_out, seq, learned_pos, tied)
            name = f"{'learned' if learned_pos else 'RoPE'}+{'tied' if tied else 'untied'}"
            variants[name] = sum(p.numel() for p in m.parameters())

    shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
    index = data.episode_index(shards)
    windows = len(data.WindowSampler(shards, index, ctx, "train",
                                     cfg.data["val_fraction"]))
    frames = sum(s.frames for s in shards)
    dataset_tokens = frames * (per_frame + 1)
    epoch_tokens = windows * seq

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = "RoPE+untied"
    model = Dynamics(d, layers, N_HEADS, vocab_in, vocab_out, seq, False, False).to(dev)
    tok = torch.randint(0, vocab_in, (batch, seq), device=dev)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=dev), 1)

    timing = {}
    if dev.type == "cuda":
        for label, dtype in (("fp32", None), ("bf16", torch.bfloat16)):
            s = _time_step(model, tok, mask, dtype, steps)
            timing[label] = {
                "s_per_step": s,
                "tokens_per_s": batch * seq / s,
                "epoch_hours": windows / batch * s / 3600,
            }

    params = variants[ref]
    return {
        "d_model": d, "n_layers": layers, "n_heads": N_HEADS,
        "ctx_frames": ctx, "tokens_per_frame": per_frame, "seq_len": seq,
        "vocab_in": vocab_in, "vocab_out": vocab_out,
        "params": variants, "params_reference": ref, "params_ref_value": params,
        "train_windows": windows, "dataset_frames": frames,
        "dataset_tokens": dataset_tokens,
        "token_cache_bytes": frames * per_frame * 2,   # uint16 per token
        "epoch_tokens": epoch_tokens,
        "chinchilla_tokens": params * CHINCHILLA,
        "chinchilla_ratio": params * CHINCHILLA / dataset_tokens,
        "batch": batch, "timing": timing,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
    d = probe(cfg, a.batch, a.steps)

    print(f"d_model {d['d_model']}, {d['n_layers']} layers, {d['n_heads']} heads, "
          f"ctx {d['ctx_frames']} frames x ({d['tokens_per_frame']}+1) = "
          f"{d['seq_len']} tokens, vocab {d['vocab_in']} in / {d['vocab_out']} out\n")
    for k, v in d["params"].items():
        tag = "   <- quoted below" if k == d["params_reference"] else ""
        print(f"  {k:<16} {v:>12,} params{tag}")
    print(f"\ntoken cache {d['token_cache_bytes'] / 1e6:.1f} MB "
          f"({d['dataset_frames']:,} frames x {d['tokens_per_frame']} uint16)")
    print(f"dataset {d['dataset_tokens'] / 1e6:.1f}M tokens against a Chinchilla-optimal "
          f"{d['chinchilla_tokens'] / 1e6:.1f}M - {d['chinchilla_ratio']:.1f}x under")
    print(f"one epoch = {d['train_windows']:,} train windows = "
          f"{d['epoch_tokens'] / 1e6:.1f}M tokens (windows overlap, so this "
          f"exceeds the dataset)\n")
    for label, t in d["timing"].items():
        print(f"  {label:<5} {t['s_per_step'] * 1e3:8.1f} ms/step at batch {d['batch']}  "
              f"{t['tokens_per_s'] / 1e3:8.1f} k tok/s  {t['epoch_hours']:6.2f} h/epoch")


if __name__ == "__main__":
    main()
