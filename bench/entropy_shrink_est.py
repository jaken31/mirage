"""What a shrink step and what attention would do to Q-2 at 96x96 - by arithmetic.

    python bench/entropy_shrink_est.py

**This is a derived row, not a measurement.** It answers a question that would
otherwise cost two 3-hour training runs, and it answers it off the token caches
already on disk. Nothing here trains anything. `runs.jsonl` r45 carries the
result and says the same thing about its status.

R1 at 96x96 misses **Q-2** - "token entropy vs uniform over 512 codes, >= 70%" -
at 55.4%, and `AGENDA.md` names exactly two remedies: the Q-2 shrink ladder, and
(for quality) the attention rung. This prices both before either is run.

Three kinds of number come out, and they are not equally trustworthy:

  * **Model-independent bounds.** `H_joint <= sum of the channel marginals` is an
    identity, so the marginal sum is a hard ceiling on any method that only
    *decorrelates* channels - attention included. And coarsening a distribution
    can only destroy information, so `current bits / log2(new codes)` is a hard
    upper bound on any shrink step whatever the re-binning. **Trust these.**
  * **The coarsening model.** FSQ puts each channel's bounded latent on L evenly
    spaced levels, so dropping 8 -> M re-bins the same latent onto a coarser
    grid; map each level-8 bin centre to its nearest level-M centre and
    accumulate the joint. Exact under one assumption that is certainly false:
    it holds the latent distribution **fixed**, where a retrained tokenizer would
    adapt. So it reads **pessimistic**.
  * **The transfer estimates for attention**, which carry the 64x64 R1 -> R2
    measurement across to 96x96 three different ways. Weakest of the three, and
    it does not matter, because the identity above already settles that case.

**Known artifact, so nobody reads it as a finding:** the shrink ladder is
non-monotonic - `[8,6,5]` scores below `[4,4,4]`. 8 -> 4 is an exact 2:1 merge
while 8 -> 6 and 8 -> 5 are ragged, and a ragged merge concentrates mass. Treat
the non-divisor rungs as pessimistic by a few points on top of everything else.

The one available check on the model is `R1 64x64`, where row 3 passes today: the
model returns 75.2% and 71.5% for `[8,6,5]` and `[5,5,5]` against a real 74.1% at
`[8,8,8]`, so it is in the right neighbourhood. That is a sanity check and not a
validation - **no shrunk rung has ever been trained**, which is the whole reason
this file is arithmetic.
"""
import json, math, pathlib, itertools
import numpy as np

ROOT = pathlib.Path("C:/Users/nguye/Documents/Dev_and_Projects/mirage")
RUNS = {
    "R1 64x64": "20260829-005439-r1",
    "R2 64x64": "20260828-230015-r2",
    "R1 96x96": "20260829-132014-r1",
}

def load(run):
    m = json.loads((ROOT / "runs" / run / "tokens" / "manifest.json").read_text())
    return np.asarray(m["counts"], float), m["entropy_bits"], m["entropy_ratio"]

def h(p):
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())

def joint3(counts, levels=(8, 8, 8)):
    """counts over mixed-radix ids -> (l0,l1,l2) probability array."""
    p = counts / counts.sum()
    ids = np.arange(len(p))
    d = []
    place = 1
    for n in levels:
        d.append((ids // place) % n)
        place *= n
    a = np.zeros(levels)
    np.add.at(a, tuple(d), p)
    return a

def coarsen(a, target):
    """Re-bin each axis from its current L onto `target` evenly spaced levels."""
    out = a
    for ax, m in enumerate(target):
        L = out.shape[ax]
        if m == L:
            continue
        # bin k of L sits at k/(L-1) of the bounded range; nearest bin of m
        j = np.rint(np.arange(L) / (L - 1) * (m - 1)).astype(int)
        new = np.zeros(out.shape[:ax] + (m,) + out.shape[ax + 1:])
        np.add.at(new, (slice(None),) * ax + (j,), out)
        out = new
    return out

def split(a):
    """joint bits, per-channel bits, redundancy."""
    per = [h(a.sum(axis=tuple(i for i in range(3) if i != c))) for c in range(3)]
    j = h(a.ravel())
    return j, per, sum(per) - j

print("=== what is on disk ===")
A = {}
for name, run in RUNS.items():
    c, bits, ratio = load(run)
    a = joint3(c)
    j, per, red = split(a)
    A[name] = a
    assert abs(j - bits) < 1e-6, f"{name}: decoded {j} vs manifest {bits}"
    print(f"{name}: {j:.3f} bits = {ratio:.1%} of 9 | channels "
          f"{' / '.join(f'{x:.3f}' for x in per)} | skew {9 - sum(per):.3f} "
          f"| redundancy {red:.3f}")

BAR = 0.70
print(f"\n=== OPTION 2: shrink the vocabulary at 96x96 (bar {BAR:.0%}) ===")
print("Model: same latent, coarser bins. Assumes retraining does NOT redistribute mass.")
print(f"{'levels':<12}{'codes':>6}{'uniform':>9}{'bits':>8}{'ratio':>8}   verdict")
ladder = [(8, 8, 8), (8, 6, 5), (5, 5, 5), (4, 4, 4), (4, 4, 3), (3, 3, 3)]
for tgt in ladder:
    a = coarsen(A["R1 96x96"], tgt)
    j, per, red = split(a)
    codes = math.prod(tgt)
    uni = math.log2(codes)
    r = j / uni
    print(f"{str(list(tgt)):<12}{codes:>6}{uni:>9.3f}{j:>8.3f}{r:>8.1%}   "
          f"{'PASS' if r >= BAR else 'fail'}")

print("\nSame model applied to R1 64x64, where row 3 already passes at [8,8,8] -")
print("this is the only check available on the model, since no shrunk rung exists:")
for tgt in [(8, 8, 8), (8, 6, 5), (5, 5, 5)]:
    a = coarsen(A["R1 64x64"], tgt)
    j, per, red = split(a)
    print(f"  {str(list(tgt)):<12}{j:>8.3f} bits{j / math.log2(math.prod(tgt)):>8.1%}")

print("\n=== OPTION 3: attention at 96x96 ===")
j1, per1, red1 = split(A["R1 64x64"])
j2, per2, red2 = split(A["R2 64x64"])
j96, per96, red96 = split(A["R1 96x96"])
print(f"measured at 64x64: redundancy {red1:.3f} -> {red2:.3f} ({red2 - red1:+.3f} bits), "
      f"skew {9 - sum(per1):.3f} -> {9 - sum(per2):.3f} ({sum(per2) - sum(per1):+.3f} bits)")
print(f"total gain {j2 - j1:+.3f} bits = {(j2 - j1) / 9:+.1%} of uniform "
      f"(matches the recorded +3.5 pp)")
print(f"\n96x96 R1 now: {j96:.3f} bits, skew {9 - sum(per96):.3f}, redundancy {red96:.3f}")
for label, dr, ds in [
    ("absolute transfer  (same bits moved)", red2 - red1, sum(per2) - sum(per1)),
    ("relative transfer  (same fraction)", red96 * (red2 / red1 - 1),
     (9 - sum(per96)) * ((9 - sum(per2)) / (9 - sum(per1)) - 1) * -1),
    ("redundancy to ZERO (upper bound)", -red96, 0.0),
]:
    est = j96 - dr + ds
    print(f"  {label:<38} {est:6.3f} bits = {est / 9:5.1%}   "
          f"{'PASS' if est / 9 >= BAR else 'fail'}")
print(f"\nneeded to pass at 512 codes: {BAR * 9:.3f} bits, i.e. {BAR * 9 - j96:+.3f} from here")

print("\n=== the model-independent bounds ===")
print("Coarsening can only DESTROY information, so `current bits / log2(new codes)`")
print("is an upper bound on any shrink step, whatever the re-binning model:")
for tgt in ladder[1:]:
    codes = math.prod(tgt)
    ub = j96 / math.log2(codes)
    a = coarsen(A["R1 96x96"], tgt)
    got, _, _ = split(a)
    print(f"  {str(list(tgt)):<12}{codes:>6} codes   upper bound {ub:6.1%}   "
          f"model says {got / math.log2(codes):6.1%}   "
          f"{'possible' if ub >= BAR else 'IMPOSSIBLE'}")

print("\nAnd the ceiling on any method that only DECORRELATES channels:")
print(f"  sum of marginals = {sum(per96):.3f} bits = {sum(per96) / 9:.1%} - "
      f"reached only at zero redundancy")
print(f"  so no amount of attention, however perfect, passes {BAR:.0%} at 96x96")
print(f"\nWhat WOULD pass: recover marginal skew. Skew is {9 - sum(per96):.3f} bits;")
need = BAR * 9 - j96
print(f"  {need:.3f} bits of it ({need / (9 - sum(per96)):.0%}) is enough, holding redundancy fixed")
print(f"  fixing skew entirely gives {9 - red96:.3f} bits = {(9 - red96) / 9:.1%}")
