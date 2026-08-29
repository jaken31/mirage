"""Every figure the tokenizer write-up needs, from artifacts already on disk.

Reads three things a finished run leaves behind and writes eight PNGs to
`docs/figures/`:

  * `runs/<id>/metrics.jsonl` - the per-epoch curves;
  * `runs/<id>/tokens/manifest.json` - the 512-bin code histogram and its shards;
  * `runs/<id>/model.pt` - reconstructions, per-frame PSNR, the edge/flat split.

It measures nothing new. Every number a figure draws also appears in the gate
table that `python -m mirage.fsq --eval RUN_ID` prints, so a figure disagreeing
with that table is a bug in this file and not a new result. Two of them assert
that agreement rather than trusting it - see `per_frame_psnr` and
`token_position_entropy`.

    python bench/fsq_figures.py                      # all eight, R1 and R2 at 60 epochs
    python bench/fsq_figures.py --only curves,codebook
    python bench/fsq_figures.py --r1 RUN_ID --r2 RUN_ID

Each figure prints the numbers it encodes, so the write-up can quote them without
reading pixels off a chart.

Colour is assigned once, at the top, and follows the *run*: R1 is blue and R2 is
orange in every figure that shows both. Where a figure needs a second distinction
inside one run - validation against training, flat pixels against edge pixels - it
uses line style or hatch, never a third hue, because a hue that means "R2" in one
figure and "edge pixels" in the next is how a reader mis-reads a whole document.
The three-hue set is validated for colour-vision deficiency (worst pair delta-E
9.2 deutan, 24.0 normal-vision, on this light surface).

ponytail: light mode only. A dark variant is a second validated set of steps, not
an inversion, and nothing here is going into a dark document yet.
"""

import argparse
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # no display on this box, and a figure script must not need one
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mirage import config, data, validator  # noqa: E402
from mirage.fsq import (KMEANS_FLOOR_DB, PEAK, PSNR_BAR_DB, _batch,  # noqa: E402
                        psnr_db)
from mirage.fsq_eval import _flat_mask, entropy_split, load_run  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "figures"

# The two 60-epoch runs: same data_hash, same seed, attention off and on.
R1_DEFAULT = "20260829-005439-r1"
R2_DEFAULT = "20260828-230015-r2"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8e88", "#e5e4e0"
SURFACE = "#fcfcfb"
# One hue, light to dark, for the two magnitude encodings (error maps, entropy grid).
SEQ = LinearSegmentedColormap.from_list(
    "mirage_blue",
    ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


# ----------------------------------------------------------------- artifacts

def epoch_rows(run_id: str) -> list[dict]:
    """The per-epoch validation rows of one run, in order.

    Skips the `final` row, which repeats the last epoch's numbers with the whole
    config attached; plotting it would draw epoch 60 twice.
    """
    rows = []
    with open(RUNS / run_id / "metrics.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "val_psnr_db" in r and not r.get("final"):
                rows.append(r)
    return rows


def manifest(run_id: str) -> dict:
    return json.loads((RUNS / run_id / "tokens" / "manifest.json").read_text(encoding="utf-8"))


def result(run_id: str) -> dict:
    return json.loads((RUNS / run_id / "result.json").read_text(encoding="utf-8"))


@torch.no_grad()
def per_frame_psnr(model, idx: np.ndarray, lut: torch.Tensor,
                   batch: int = 256) -> tuple[np.ndarray, float]:
    """One uint8 PSNR per val frame, plus the pooled number those frames imply.

    Per-frame PSNR is *not* what gate row 1 reports: the gate pools squared error
    over every frame and takes one logarithm, while the mean of these is a mean of
    16,200 logarithms. The two differ by a fraction of a dB and neither is wrong,
    so the pooled value comes back alongside for the caller to check against
    `result.json` - that is what catches a wrong LUT or a stale checkpoint, which
    is what would actually go wrong here.
    """
    out = np.empty(len(idx), dtype=np.float64)
    values = 3 * idx.shape[1] * idx.shape[2]
    total_sse = 0.0
    for i in range(0, len(idx), batch):
        rows = np.arange(i, min(i + batch, len(idx)))
        x8 = _batch(idx, lut, rows)
        y8 = (model(x8 / PEAK) * PEAK).round().clamp(0, PEAK)
        sse = (y8 - x8).pow(2).sum(dim=(1, 2, 3))
        total_sse += float(sse.sum())
        out[rows] = (10.0 * torch.log10(PEAK * PEAK * values / sse.clamp_min(1e-9))).cpu().numpy()
    return out, psnr_db(total_sse, len(idx) * values)


@torch.no_grad()
def edge_flat_split(model, idx: np.ndarray, lut: torch.Tensor, patch: int,
                    batch: int = 256) -> tuple[float, float, float]:
    """(flat dB, edge dB, edge share of squared error) - gate row 7, redrawn.

    Flatness comes from the ground truth's palette indices, never from the
    reconstruction: a blurry decoder allowed to call its own mistakes edges would
    flatter the flat-pixel number.
    """
    sse = {True: 0.0, False: 0.0}
    values = {True: 0, False: 0}
    for i in range(0, len(idx), batch):
        rows = np.arange(i, min(i + batch, len(idx)))
        x8 = _batch(idx, lut, rows)
        y8 = (model(x8 / PEAK) * PEAK).round().clamp(0, PEAK)
        err = (y8 - x8).pow(2).sum(1)
        flat = torch.from_numpy(_flat_mask(np.ascontiguousarray(idx[rows]), patch)).to(err.device)
        for is_flat in (True, False):
            m = flat if is_flat else ~flat
            sse[is_flat] += float(err[m].sum())
            values[is_flat] += int(m.sum()) * 3
    total = sse[True] + sse[False]
    return psnr_db(sse[True], values[True]), psnr_db(sse[False], values[False]), sse[False] / total


def token_position_entropy(run_id: str, grid: tuple[int, int]) -> np.ndarray:
    """Per-grid-cell token entropy in bits, over all 300,000 cached frames.

    The joint entropy gate row 3 reports is one number for the whole 8x8 grid.
    This is the same statistic per cell, and it answers a question the single
    number cannot: whether the codebook is spent evenly across the frame, or
    hoarded in the cells the arm actually moves through.
    """
    man = manifest(run_id)
    h, w = grid
    counts = np.zeros((h * w, man["codebook_size"]), dtype=np.int64)
    seen = 0
    for s in man["shards"]:
        arr = np.load(RUNS / run_id / "tokens" / s["file"])
        flat = arr.reshape(len(arr), -1)
        for p in range(h * w):
            counts[p] += np.bincount(flat[:, p], minlength=man["codebook_size"])
        seen += len(arr)
    assert seen == man["frames"], f"{run_id}: {seen} frames on disk, manifest says {man['frames']}"
    p = counts / counts.sum(1, keepdims=True)
    return -(p * np.log2(np.where(p > 0, p, 1.0))).sum(1).reshape(h, w)


# -------------------------------------------------------------------- figures

def _end_labels(ax, x, ends):
    """Direct labels at the right end of each line: ink text, colour on the dot.

    Series colour never lands on text - a reader with a colour deficiency reads
    the label, and the dot beside it is only confirmation.

    `ends` is [(y, text, colour)], and two labels are nudged apart when they would
    overlap. R1 and R2 finish 0.087 dB apart, which is the whole point of the
    figure and also close enough to print one label on top of the other. The
    proximity test runs in display pixels rather than data units, so the same
    branch works on the log axis in `fig_codebook`. Call this *after* the last
    line is drawn - it reads the axis transform.
    """
    ends = sorted(ends, key=lambda e: e[0])
    px = [ax.transData.transform((x, y))[1] for y, _, _ in ends]
    close = len(ends) == 2 and abs(px[1] - px[0]) < 14
    for k, (y, text, color) in enumerate(ends):
        ax.plot([x], [y], "o", color=color, ms=5, zorder=5, clip_on=False)
        ax.annotate(text, (x, y), xytext=(7, (-9, 9)[k] if close else 0),
                    textcoords="offset points", va="center", color=INK,
                    fontsize=8.5, clip_on=False)


def fig_curves(runs):
    """1. Val and train PSNR per epoch, against the bar and the k-means floor."""
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(wspace=0.34)   # the left panel's direct labels overhang
    series, ends = {}, []
    for label, rid, color in runs:
        ep = epoch_rows(rid)
        x = [r["epoch"] + 1 for r in ep]
        val = [r["val_psnr_db"] for r in ep]
        series[label] = (x, val)
        ax.plot(x, [r["train_psnr_db"] for r in ep], color=color, lw=1.1,
                ls=(0, (4, 2)), alpha=0.6)
        ax.plot(x, val, color=color, lw=1.8)
        ends.append((val[-1], f"{label}  {val[-1]:.2f} dB", color))

    for y, name, style in ((PSNR_BAR_DB, f"Q-1 bar {PSNR_BAR_DB} dB", (0, (5, 3))),
                           (KMEANS_FLOOR_DB, f"k-means-512 floor {KMEANS_FLOOR_DB} dB",
                            (0, (1, 2)))):
        ax.axhline(y, color=MUTED, lw=1.0, ls=style, zorder=0)
        ax.annotate(name, (38, y), xytext=(0, 4), textcoords="offset points",
                    color=INK2, fontsize=8)   # x=38: under the converged curves
    ax.set_xlim(0, 78)
    ax.set_xticks(range(0, 61, 10))   # the run ends at 60; the rest is label room
    ax.set_xlabel("epoch")
    ax.set_ylabel("held-out PSNR, uint8 (dB)")
    ax.set_title("Reconstruction quality per epoch", loc="left", color=INK)
    ax.grid(axis="y")
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=1.8, label="validation"),
                       Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 2)), label="train")],
              loc="lower right")
    _end_labels(ax, 60, ends)

    (x1, v1), (_, v2) = series[runs[0][0]], series[runs[1][0]]
    n = min(len(v1), len(v2))
    delta = np.asarray(v2[:n]) - np.asarray(v1[:n])
    axd.axhline(0, color=MUTED, lw=1.0)
    axd.plot(x1[:n], delta, color=INK2, lw=1.5)
    axd.fill_between(x1[:n], 0, delta, color=INK2, alpha=0.10)
    axd.set_xlabel("epoch")
    axd.set_ylabel("attention advantage (dB)")
    axd.set_title(f"What attention buys: {delta[-1]:+.3f} dB at 60 epochs",
                  loc="left", color=INK)
    axd.grid(axis="y")
    print(f"  curves: final val {v1[-1]:.3f} / {v2[-1]:.3f} dB, "
          f"delta min {delta.min():+.3f} max {delta.max():+.3f} final {delta[-1]:+.3f} dB")
    return fig


def fig_frame_psnr(runs, frame_db, pooled):
    """3. The spread behind the single held-out number."""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bins = np.linspace(min(d.min() for d in frame_db.values()),
                       max(d.max() for d in frame_db.values()), 70)
    for label, _, color in runs:
        db = frame_db[label]
        ax.hist(db, bins=bins, histtype="step", lw=1.8, color=color)
        print(f"  frame PSNR {label}: pooled {pooled[label]:.3f} dB, "
              f"median {np.median(db):.3f}, p1 {np.percentile(db, 1):.3f}, "
              f"p99 {np.percentile(db, 99):.3f}, "
              f"{float((db < PSNR_BAR_DB).mean()):.1%} of frames under {PSNR_BAR_DB} dB")
    ax.axvline(PSNR_BAR_DB, color=MUTED, lw=1.0, ls=(0, (5, 3)))
    ax.annotate(f"Q-1 bar {PSNR_BAR_DB} dB", (PSNR_BAR_DB, 0), xytext=(4, 6),
                textcoords="offset points", color=INK2, fontsize=8)
    ax.legend(handles=[Line2D([], [], color=c, lw=1.8,
                              label=f"{lab}  pooled {pooled[lab]:.2f} dB, "
                                    f"{float((frame_db[lab] < PSNR_BAR_DB).mean()):.1%} under bar")
                       for lab, _, c in runs], loc="upper left")
    ax.set_xlabel("per-frame PSNR, uint8 (dB)")
    ax.set_ylabel("val frames")
    ax.set_title("Per-frame quality across the 16,200 held-out frames", loc="left", color=INK)
    ax.grid(axis="y")
    return fig


@torch.no_grad()  # this one runs the model; the others only read numbers
def fig_recon(label, model, val_idx, lut, frame_db, vmax=64):
    """2. Truth, reconstruction and error at five points of the quality spread.

    Frames picked by percentile and labelled with their dB rather than hand-chosen:
    a gallery of five frames the author liked is not evidence about 16,200.
    """
    order = np.argsort(frame_db)
    picks = [("worst", order[0]), ("p5", order[len(order) // 20]),
             ("median", order[len(order) // 2]),
             ("p95", order[-len(order) // 20]), ("best", order[-1])]
    fig, axes = plt.subplots(3, len(picks), figsize=(2.05 * len(picks), 6.3))
    ims = []
    for col, (name, i) in enumerate(picks):
        x8 = _batch(val_idx, lut, np.asarray([i]))
        y8 = (model(x8 / PEAK) * PEAK).round().clamp(0, PEAK)
        axes[0, col].imshow(x8[0].permute(1, 2, 0).cpu().numpy() / PEAK)
        axes[1, col].imshow(y8[0].permute(1, 2, 0).cpu().numpy() / PEAK)
        ims.append(axes[2, col].imshow((y8 - x8).abs().amax(1)[0].cpu().numpy(),
                                      cmap=SEQ, vmin=0, vmax=vmax))
        axes[0, col].set_title(f"{name}\n{frame_db[i]:.2f} dB", fontsize=9, color=INK)
        for row in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for sp in axes[row, col].spines.values():
                sp.set_visible(False)
    # Row labels stay short: a rotated label wider than its own row gets clipped
    # by savefig's tight bbox on this matplotlib, twice observed. Units go in the
    # horizontal title, which does not clip.
    for row, name in enumerate(("ground truth", "reconstruction", f"|error| 0-{vmax}")):
        axes[row, 0].set_ylabel(name, fontsize=9, color=INK2)
    # Its own axes rather than `ax=axes[2, :]`: stealing width from row 3 alone
    # leaves its panels narrower than the two rows above it, which is visible.
    fig.subplots_adjust(right=0.86)   # room for the colourbar and its label
    box = axes[2, -1].get_position()
    cax = fig.add_axes((box.x1 + 0.012, box.y0, 0.011, box.height))
    fig.colorbar(ims[-1], cax=cax)
    fig.suptitle(f"{label}: reconstructions across the quality spread "
                 f"(bottom row: max abs channel error, uint8, clipped at {vmax})",
                 x=0.02, ha="left", fontsize=10.5, color=INK)
    print(f"  recon {label}: " + ", ".join(f"{n} {frame_db[i]:.2f} dB" for n, i in picks))
    return fig


def fig_codebook(runs):
    """4. Is the 512-code book used, and how unevenly."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ends = []
    for label, rid, color in runs:
        man = manifest(rid)
        c = np.asarray(man["counts"], dtype=float)
        p = np.sort(c / c.sum())[::-1]
        ax.plot(np.arange(1, len(p) + 1), p, color=color, lw=1.8)
        ends.append((p[-1], label, color))
        print(f"  codebook {label}: entropy {man['entropy_bits']:.3f} bits "
              f"({man['entropy_ratio']:.1%} of 9), "
              f"{man['live_codes']}/{man['codebook_size']} live, "
              f"{int((c == 0).sum())} never used, top code {p[0]:.2%}, "
              f"top 32 codes {p[:32].sum():.1%} of mass")
    n = manifest(runs[0][1])["codebook_size"]
    ax.axhline(1 / n, color=MUTED, lw=1.0, ls=(0, (5, 3)))
    ax.annotate(f"uniform, 1/{n}", (n * 0.60, 1 / n), xytext=(0, 5),
                textcoords="offset points", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlim(0, n * 1.14)
    ax.set_xticks(range(0, n + 1, 128))
    ax.set_xlabel("code, ranked by frequency")
    ax.set_ylabel("share of all 19.2M tokens")
    ax.set_title("Codebook usage - flat would be the full 9 bits", loc="left", color=INK)
    ax.grid(axis="y")
    _end_labels(ax, n, ends)
    return fig


def fig_channel_digits(runs):
    """5. Per-channel digit distributions - where the marginal skew lives.

    A token id is the mixed-radix number `d0 + 8*d1 + 64*d2`, so each channel's
    digit histogram falls straight out of the same 512-bin count vector.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(10.5, 3.9), sharey=True)
    top = 0.0
    for ax, (label, rid, _) in zip(axes, runs):
        man = manifest(rid)
        levels = man["levels"]
        c = np.asarray(man["counts"], dtype=float)
        p = c / c.sum()
        ids = np.arange(len(c))
        place = 1
        width = 0.27
        es = entropy_split(man["counts"], levels)
        for ch, (n, color) in enumerate(zip(levels, (BLUE, ORANGE, AQUA))):
            digits = np.bincount((ids // place) % n, weights=p, minlength=n)
            top = max(top, float(digits.max()))
            ax.bar(np.arange(n) + (ch - 1) * width, digits, width * 0.92, color=color,
                   label=f"channel {ch}  {es['channel_bits'][ch]:.2f} of 3 bits")
            place *= n
        ax.set_xticks(range(levels[0]))
        ax.set_xlabel("FSQ digit (0 = most negative)")
        ax.set_title(f"{label}  -  marginals sum to {es['marginal_sum_bits']:.2f} bits",
                     loc="left", color=INK)
        ax.legend(loc="upper center")
        ax.grid(axis="y")
        print(f"  channels {label}: "
              + " / ".join(f"{b:.3f}" for b in es["channel_bits"]) + " bits of 3.000 each")
    axes[0].set_ylim(0, top * 1.38)   # sharey: one limit, clearing the legend
    axes[0].set_ylabel("share of tokens")
    # Deliberately not "the channel sits off centre": R1 channel 2 piles 40% of its
    # mass on digit 7 and R1 channel 0 piles 56% on digits 3-4, so both a centred
    # and a saturated channel are on this figure. Uneven is the claim; where is not.
    fig.suptitle("Per-channel digit use - an uneven digit histogram is the marginal skew",
                 x=0.02, y=1.04, ha="left", fontsize=10.5, color=INK)
    return fig


def fig_bits_budget(runs):
    """6. The 9-bit budget split three ways: achieved, skew, redundancy."""
    fig, ax = plt.subplots(figsize=(8.4, 2.7))
    parts = [("joint entropy (achieved)", BLUE), ("marginal skew", ORANGE),
             ("redundancy between channels", AQUA)]
    for row, (label, rid, _) in enumerate(runs):
        man = manifest(rid)
        es = entropy_split(man["counts"], man["levels"])
        uniform = math.log2(man["codebook_size"])
        vals = [es["joint_bits"], uniform - es["marginal_sum_bits"], es["redundancy_bits"]]
        left = 0.0
        for (_, color), v in zip(parts, vals):
            # edgecolor=SURFACE gives the 2px gap that keeps adjacent fills legible.
            ax.barh(row, v, left=left, height=0.52, color=color,
                    edgecolor=SURFACE, linewidth=2)
            ax.annotate(f"{v:.2f}", (left + v / 2, row), ha="center", va="center",
                        color="white" if color != AQUA else INK, fontsize=8.5)
            left += v
        print(f"  budget {label}: {vals[0]:.3f} achieved + {vals[1]:.3f} skew + "
              f"{vals[2]:.3f} redundancy = {uniform:.0f} bits")
    ax.set_yticks(range(len(runs)), [r[0] for r in runs])
    ax.set_xlim(0, math.log2(manifest(runs[0][1])["codebook_size"]))
    ax.set_xlabel("bits per token")
    ax.set_title("Where the missing bits go - the two deficits have different fixes",
                 loc="left", color=INK)
    ax.legend(handles=[Patch(facecolor=c, label=n) for n, c in parts],
              loc="upper center", bbox_to_anchor=(0.5, -0.42), ncols=3)
    ax.invert_yaxis()
    ax.grid(axis="x")
    return fig


def fig_edge_flat(runs, splits):
    """7. Edge pixels against flat pixels - the 64-vs-96 fork's diagnostic."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.34
    for i, (label, _, color) in enumerate(runs):
        flat_db, edge_db, share = splits[label]
        bars = ax.bar([0 + (i - 0.5) * width, 1 + (i - 0.5) * width], [flat_db, edge_db],
                      width * 0.92, color=color, label=label)
        bars[1].set_hatch("///")          # texture, so edge-vs-flat is not colour-only
        bars[1].set_edgecolor(SURFACE)
        for b, v in zip(bars, (flat_db, edge_db)):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=INK, fontsize=8.5)
        print(f"  edge/flat {label}: flat {flat_db:.3f} dB, edge {edge_db:.3f} dB, "
              f"{share:.2%} of squared error at edges")
    ax.axhline(PSNR_BAR_DB, color=MUTED, lw=1.0, ls=(0, (5, 3)))
    ax.annotate(f"Q-1 bar {PSNR_BAR_DB} dB", (0.01, PSNR_BAR_DB),
                xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xticks([0, 1], ["flat pixels", "edge pixels (hatched)"])
    ax.set_ylabel("PSNR, uint8 (dB)")
    ax.set_ylim(0, 46)
    ax.set_title(f"Error concentrates at edges: "
                 f"{splits[runs[0][0]][2]:.1%} of squared error lives there",
                 loc="left", color=INK)
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    return fig


def fig_token_map(runs, grid):
    """8. Per-cell token entropy over the 8x8 grid, both runs on one scale."""
    ents = {lab: token_position_entropy(rid, grid) for lab, rid, _ in runs}
    lo = min(e.min() for e in ents.values())
    hi = max(e.max() for e in ents.values())
    fig, axes = plt.subplots(1, len(runs), figsize=(9.8, 4.6))
    ims = []
    for ax, (label, *_) in zip(axes, runs):
        e = ents[label]
        ims.append(ax.imshow(e, cmap=SEQ, vmin=lo, vmax=hi))
        for r in range(e.shape[0]):
            for c in range(e.shape[1]):
                ax.text(c, r, f"{e[r, c]:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if e[r, c] > lo + 0.62 * (hi - lo) else INK)
        ax.set_xticks(range(e.shape[1]))
        ax.set_yticks(range(e.shape[0]))
        ax.set_title(f"{label}  -  {e.min():.2f} to {e.max():.2f} bits", loc="left", color=INK)
        print(f"  token map {label}: min {e.min():.3f}, max {e.max():.3f}, "
              f"mean {e.mean():.3f} bits per cell")
    fig.colorbar(ims[-1], ax=axes.tolist(), fraction=0.025, pad=0.02, label="entropy (bits)")
    fig.suptitle("Token entropy per grid cell, all 300,000 frames "
                 "(values printed - colour is confirmation, not the encoding)",
                 x=0.02, ha="left", fontsize=10.5, color=INK)
    return fig


# ----------------------------------------------------------------------- main

FIGURES = ("curves", "recon", "frames", "codebook", "channels", "budget", "edgeflat", "tokenmap")
NEEDS_MODEL = ("recon", "frames", "edgeflat")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--r1", default=R1_DEFAULT)
    ap.add_argument("--r2", default=R2_DEFAULT)
    ap.add_argument("--only", default=",".join(FIGURES),
                    help=f"comma-separated subset of {', '.join(FIGURES)}")
    args = ap.parse_args()
    want = [f.strip() for f in args.only.split(",") if f.strip()]
    bad = [f for f in want if f not in FIGURES]
    if bad:
        ap.error(f"unknown figure(s) {bad}; pick from {', '.join(FIGURES)}")

    runs = [("R1 no attention", args.r1, BLUE), ("R2 attention", args.r2, ORANGE)]
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load(ROOT / "mirage" / "configs" / "base.json")
    print(f"data_hash {cfg.data_hash[:8]} - "
          + " / ".join(f"{lab} {rid} ({result(rid)['params']:,} params, "
                       f"{result(rid)['val_psnr_db']:.3f} dB)" for lab, rid, _ in runs))

    made = {}
    if "curves" in want:
        made["fig1_curves"] = fig_curves(runs)
    if "codebook" in want:
        made["fig4_codebook"] = fig_codebook(runs)
    if "channels" in want:
        made["fig5_channel_digits"] = fig_channel_digits(runs)
    if "budget" in want:
        made["fig6_bits_budget"] = fig_bits_budget(runs)
    if "tokenmap" in want:
        made["fig8_token_entropy_map"] = fig_token_map(runs, tuple(cfg.shapes.token_grid))

    if any(f in want for f in NEEDS_MODEL):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
        palette = validator.load_palette(ROOT / cfg.sim["scene_xml"])
        val_idx, lut_np = data.preload(shards, data.episode_index(shards), "val",
                                       cfg.data["val_fraction"], palette.rgb)
        lut = torch.from_numpy(lut_np).to(dev).float()
        patch = cfg.shapes.image_size[0] // cfg.shapes.token_grid[0]
        print(f"{len(val_idx):,} val frames on {dev}")

        frame_db, pooled, splits = {}, {}, {}
        for label, rid, _ in runs:
            model, _ = load_run(rid, cfg, dev)
            if "frames" in want or "recon" in want:
                frame_db[label], pooled[label] = per_frame_psnr(model, val_idx, lut)
                assert abs(pooled[label] - result(rid)["val_psnr_db"]) < 0.01, \
                    (f"{rid}: recomputed {pooled[label]:.4f} dB against "
                     f"result.json {result(rid)['val_psnr_db']:.4f} dB")
            if "edgeflat" in want:
                splits[label] = edge_flat_split(model, val_idx, lut, patch)
            if "recon" in want:
                made[f"fig2_recon_{rid.rsplit('-', 1)[-1]}"] = fig_recon(
                    label, model, val_idx, lut, frame_db[label])
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        if "frames" in want:
            made["fig3_frame_psnr"] = fig_frame_psnr(runs, frame_db, pooled)
        if "edgeflat" in want:
            made["fig7_edge_vs_flat"] = fig_edge_flat(runs, splits)

    for name, fig in sorted(made.items()):
        path = OUT / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
