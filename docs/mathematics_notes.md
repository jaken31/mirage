# Mirage: Mathematics Notes

**What this is.** Every piece of mathematics the project actually runs, explained
from scratch, with a pointer to the code that computes it. It is a companion to
`docs/decision_notes.md`, which covers *why* each choice was made. This file
covers *what the formula is and how to read it*.

**Derived, not authoritative.** The formulas here are transcriptions of code. When
this file and the code disagree, the code is right and this file is stale.

## Doc class: Live

`docs/canonical_numbers.md` (the register) sorts every doc into Log, Register,
Live or Frozen. This file is Live: it states current values in plain words, and
the register is where to check that a value is still current. This file is not
yet named in the register's class table; adding it there is the register owner's
call, not this file's.

Two conventions:

- **Prose states the quantity and its current value**, for example "the held-out
  k-means floor, 28.27 dB".
- **Fenced worked blocks substitute values to show the arithmetic.** Every number
  inside a fenced block below is **recomputed** from the inputs named in the
  prose. A recomputation is not a second source of truth. If an input moves, the
  block is stale and recomputing it is the fix.

A handful of figures here have no register entry yet. They are marked
**unregistered** where they are used and listed at the end, because the register's
own rule is to add an entry when a number reaches its second live doc.

---

## The one thread through all of it

| Quantity | Shape of the math | Where |
|---|---|---|
| palette distance | Euclidean norm in RGB space | `mirage/validator.py`, `_label` |
| MSE | squared Euclidean norm over one frame's values, averaged | `mirage/fsq.py`, `reconstruction_psnr` |
| PSNR | log of a ratio of the above | `mirage/fsq.py`, `psnr_db` |
| k-means floor | minimise that same squared norm over cluster assignments | `bench/patch_probe.py` |
| token entropy | log of a probability, averaged | `mirage/fsq_eval.py`, `write_token_cache` |
| channel redundancy | difference of two entropies | `mirage/fsq_eval.py`, `entropy_split` |
| compactness | eigenvectors of a covariance, itself a squared-distance object | `mirage/validator.py`, `_oriented` |

It is squared Euclidean distance and logarithms, over and over. The one exception
is the straight-through estimator in the quantizer, which is not really
mathematics at all. It is an admitted lie about a derivative, adopted because
rounding has no useful one.

---

## 1. Euclidean distance in colour space

Used by: the frame validator's palette check and `offpalette_tau`.

A pixel is a point in 3D space, `(R, G, B)`, each channel 0..255. The distance
between two colours is Pythagoras extended to three axes:

```
d(p, q) = sqrt( (Rp-Rq)^2 + (Gp-Gq)^2 + (Bp-Bq)^2 )
```

The palette is the scene XML's six `rgba` colours plus an implicit `void` entry
for MuJoCo's clear colour. The clear colour shows through past the far table
edge and appears in no `rgba` attribute. That makes 7 entries, which is the
distinct-colour count measured over the whole dataset. Without the void entry, a
flawless frame reads about 578 off-palette pixels (**unregistered**,
`mirage/validator.py` docstring), and the zero-false-positive bar can never be
met.

For each pixel, find the closest palette entry:

```
nearest(p) = argmin over k of  ||p - c_k||^2
```

**Squared, not square-rooted, on purpose.** `sqrt` only ever increases, so
whichever `k` minimises `d^2` also minimises `d`. Skipping the root saves one
square root per palette entry per colour. The root is taken once, for the winner
only, because `tau` is expressed in real distance units.

**A second optimisation worth noticing.** `_label` maps the frame's *distinct*
colours rather than its pixels. A ground-truth frame holds 7 of them, so the
argmin is 7x7 instead of 4096x7. It stays correct on decoder output that emits
thousands of colours, just slower.

### tau is the radius of a ball

Draw a sphere of radius `tau` around each palette point. A pixel inside some
sphere is legal. A pixel outside all of them counts toward `offpalette_px`.

`offpalette_tau` lives in `mirage/configs/base.json` and is currently **32.0**.
It was first set against ground-truth frames, where the worst distance any
rendered pixel sits from its own palette entry is **0.75 RGB units**. The ratio
between the two is the slack, now **43x**. Build order item 6 has since re-run
the palette sweep against reconstructions rather than ground truth. tau rose from
8.0 to 32.0 as predicted. The other prediction, that false positives on ground
truth would stop reading zero, was wrong: they still read 0 px, because ground
truth sits at 0.75 and clears any tau in play. What is non-zero is a different
count on decoder output.

> **Two unrelated 0.75s, and mixing them up has already caused one misreading.**
> The worst render distance is **0.75 RGB units of colour distance**. The k-means
> train/val split leak in section 7 is **0.75 dB of PSNR**. They share no
> derivation. Check the unit before quoting either. The register carries this
> warning on both rows.

Recomputed from `tau`:

```
legal volume per entry = (4/3) * pi * tau^3
tau = 0.75  ->      1.8 colour-cube units
tau = 8.0   ->  2,145
tau = 32.0  -> 137,258
```

Volume grows as `tau^3`. Raising tau from 8 to 32 makes the legal region **64x**
larger per palette entry. Every doubling buys tolerance for decoder blur and
throws away detection power cubically fast. That tradeoff is the whole content of
the build-order item 6 calibration.

### Why nearest-palette rather than exact equality

`rgba * 255` does not land on an integer. link0's `0.90 0.75 0.10` renders as
(229, 191, 25), not (230, 191, 26), and not by any rule worth modelling: 0.65
rounds up to 166 while 0.90 rounds down to 229. Under exact equality against a
byte-rounded palette, **four of the seven entries match zero pixels** on a
flawless frame, and the validator reports block0, block2, link1 and table as
missing. (All **unregistered**, `mirage/validator.py` docstring.)

This is also why `Palette.rgb` stays unrounded float 0..255. Measured against
229.5 rather than 230, the worst rendered pixel distance is 0.75. Rounding the
palette doubles that for no gain.

---

## 2. MSE and PSNR

Used by: the tokenizer PSNR bar (30.0 dB) and the training loss.

### Mean squared error

Line up the original frame and the reconstruction, subtract, square, average:

```
MSE = (1/N) * sum_i (x_i - xhat_i)^2
N = 64 * 64 * 3 = 12,288 values per frame     (recomputed from configs/base.json)
```

Squared rather than absolute, for two reasons: it punishes one large error more
than many small ones, and it is differentiable everywhere. `|x|` has a kink at
zero that gradient descent handles badly.

### Peak signal-to-noise ratio

```
PSNR = 10 * log10( PEAK^2 / MSE )        PEAK = 255.0
```

Read it right to left:

| Piece | Why it is there |
|---|---|
| `PEAK^2 / MSE` | a ratio: largest possible error over your actual error. Unitless, bigger is better |
| `log10` | image errors span orders of magnitude; a log makes "10x better" a fixed step |
| `x 10` | converts to decibels. This is the only reason the unit is dB |

Worked backwards from the 30.0 dB bar:

```
MSE = 255^2 / 10^(30.0/10) = 65,025 / 1000 = 65.03
```

So the bar means the average squared error per colour channel is about 65: a
typical channel is off by about sqrt(65) = 8 out of 255.

**The rule to memorise: every 3 dB halves the MSE**, because `10*log10(2) =
3.01`. So the gap between R1 at convergence (31.095 dB) and the held-out k-means
floor (28.27 dB) works out to:

```
31.095 - 28.27 = 2.825 dB  ->  10^(-2.825/10) = 0.52
```

The tokenizer makes roughly **half** the squared error of the baseline. The bar
that gap has to clear is gate row 2's, +1.73 dB. The register marks it derived:
recompute it as the 30.0 dB bar minus the k-means floor rather than restating it.

### One detail that is easy to get wrong

PSNR is computed on **uint8** reconstructions, after rounding, not on the raw
float output. That is what the pipeline delivers and what item 6 hands the
validator. Measuring before rounding reports a number nothing downstream ever
sees.

---

## 3. Converting decibels into a pixel budget

The mean squared distance between two distinct palette entries is **47,814**. That
is the cost, in error units, of painting one pixel a completely different palette
colour, and it turns any dB figure into a count of ruined pixels:

```
SSE per frame      = MSE * 12,288
fully wrong pixels = SSE / 47,814
```

Recomputed at three levels, per 4,096-pixel frame:

| Level | PSNR | MSE | SSE per frame | Equivalent wrong pixels |
|---|---|---|---|---|
| held-out k-means floor | 28.27 dB | 96.85 | 1,190,040 | **24.9** |
| the PSNR bar | 30.0 dB | 65.03 | 799,027 | **16.7** |
| R1 at convergence | 31.095 dB | 50.53 | 620,957 | **13.0** |

This is where "cut the error count by a third" comes from: about 25 wrong pixels
down to about 17.

### The design argument this number settled

47,814 per wrong pixel is large enough that per-pixel 7-way classification with
cross-entropy needs roughly 99.6% pixel accuracy to clear 30 dB
(**unregistered**, `docs/phase1_structural_plan.md`), because every miss costs
the full amount. MSE regression can hedge: it outputs a blend and pays a small
penalty instead of a catastrophic one. That is why the tokenizer trains on plain
MSE and nothing else.

The hedge has a real cost. MSE rewards blurring edges, and 99.95% of the floor's
squared error already lives in the 36.53% of patches that are not a single flat
colour. The counterweight is item 6: `offpalette_px` on reconstructions punishes
exactly the blur that PSNR rewards, and the two cannot both be gamed.

---

## 4. Shannon entropy

Used by: the token-entropy bar (70%) and gate row 3.

Count each token id over all 300,000 frames, convert to probabilities
`p_i = c_i / sum(c)`, then:

```
H = - sum_i p_i * log2(p_i)      bits
```

`H` is the average number of yes/no questions needed to identify which token you
received.

| Situation | H |
|---|---|
| all 512 tokens equally likely | `log2(512) = 9` bits, the maximum |
| exactly one token ever used | 0 bits, you already know the answer |
| R2 measured | 77.6% of that maximum |
| R1 measured | 74.1% |

The entropy bar is the ratio `H / log2(512)` against 70%: a normalised score of
how much of the vocabulary is doing work. R2 in bits:

```
0.776 * 9 = 6.98 bits of the 9 available
```

**The ceiling is not 100%.** The provable ceiling is 94.25%, because 20.28% of
interior cells have a fully flat receptive field, and a flat cell has little to
say. That ceiling sits comfortably above 70%, which is why the data does not
force an entropy miss.

### Splitting the entropy: skew versus redundancy

A token id is three digits packed together, so the per-channel digit
distributions come out of the same counts vector, with no GPU and no re-encode:

```
H(d0) + H(d1) + H(d2)  -  H(d0, d1, d2)  =  redundancy  >= 0
   marginal sum              joint
```

This quantity is called total correlation, or sometimes multi-information. It is
zero only when the three digits are statistically independent, and it can never
be negative. The two terms fail for different reasons and have different fixes:

- **Marginal skew.** One channel's latent sits off centre in the `tanh` bound and
  never reaches most of its levels. R2's channel 2 puts 81% of its mass on digits
  0 and 1 and returns 1.964 of 3 bits.
- **Redundancy.** The channels encode copies of each other. The attention layer
  fixes this: from R1 to R2 it falls from 1.339 to 0.781 bits.

Both bullets are **unregistered**; the source is `mirage/fsq_eval.py`,
`entropy_split`, reproduced by `python -m mirage.fsq --eval`.

That 0.558-bit drop is 76% of attention's whole entropy gain of +3.5 pp, and it
explains a result no design document predicted. Attention bought only +0.087 dB
of quality for 263,680 parameters - a measured non-lever - and paid for itself in
entropy instead.

Splitting the two matters because the planned remedy for an entropy miss, the
shrink ladder, fixes *collapse*, and neither term here is collapse. No code has a
zero count in either rung. That is also what retired the collapse worry the old
150-of-512 live-centroid figure used to support (it is now 486 of 512).

---

## 5. FSQ: quantizing without killing the gradient

Used by: the tokenizer bottleneck. This is the densest mathematics in the
project.

### The problem

The encoder produces a continuous number `z`. Tokens must be integers. But
`round()` has derivative zero almost everywhere, so gradient descent gets no
signal through it and training dies.

### Step 1: squash into a bounded range

With `L` levels for this channel:

```
half     = (L - 1) * (1 + eps) / 2
offset   = 0.5 if L is even else 0
shift    = atanh(offset / half)
bound(z) = tanh(z + shift) * half - offset
```

`tanh` maps the whole real line into (-1, 1), so multiplying by `half` maps any
input into a bounded interval containing exactly `L` integer grid points.

- `eps` widens the bound just past the outermost level, so the `tanh` asymptote
  does not sit exactly on a grid point it can never reach.
- `offset` and `shift` re-centre an even number of levels, whose grid points
  straddle zero rather than including it.

### Step 2: the straight-through estimator

```python
q = q + (q.round() - q).detach()
```

Forward: `q + (round(q) - q)` equals `round(q)`, so you get the integer.
Backward: `.detach()` zeroes the bracket's derivative, so autograd sees `q` alone
and passes the gradient through unchanged.

It is a deliberate lie about the derivative. It works because the lie is unbiased
on average.

### The subtlety that invalidates naive levels comparisons

The straight-through estimator skips only the rounding. The `tanh` derivative
survives into the backward pass:

```
d/dz bound(z) = half * sech^2(z + shift)
```

Evaluated at `z = 0` (all three **unregistered**, reproduced by
`FSQ._self_check`):

| Levels table | Derivative at 0 |
|---|---|
| `[8,8,8]` | 0.858 |
| `[5,5,5]` | 1.001 |
| `[4,4,4]` | 0.668 |

Switching `[5,5,5]` to `[4,4,4]` silently multiplies the bottleneck's effective
learning rate by `0.668 / 1.001 = 0.67`, and the spread is up to 1.5x across the
full table. Compare two levels tables at a single LR, and part of the "levels
result" is really an LR result. **Every step of the shrink ladder needs a paired
LR check.**

### Step 3: mixed radix, packing digits into one id

This is the same idea as reading `347` as `3*100 + 4*10 + 7`, except each place
has its own base:

```
id = d0 + L0*d1 + L0*L1*d2
```

For `[8,8,8]` that is plain base 8, `d0 + 8*d1 + 64*d2`, giving ids 0..511. For a
mixed table like `[8,6,5]` the place values are 1, 8, 48, giving 0..239.

**This is why FSQ has no codebook.** Classic VQ-VAE stores learned vectors and
needs a commitment loss, EMA updates and dead-code restarts to keep them alive.
FSQ's mapping is a one-to-one correspondence guaranteed by arithmetic: every
integer in `[0, prod(levels))` has exactly one mixed-radix expansion, so no two
codes can collide and none can die from neglect. The whole maintenance apparatus
is replaced by a base conversion.

It is also why no auxiliary entropy loss is allowed if the entropy bar misses. An
auxiliary loss undoes the reason FSQ was chosen over VQ: a better entropy score
would no longer tell "the vocabulary is well used" apart from "the loss propped it
up". Shrink the vocabulary instead, and pay the floor cost the shrink ladder
prices: 27.09 dB at 240 codes against 28.27 dB at 512.

When recovering digits, the bound check is per channel, not against
`max(levels)`. A mixed table has three different digit ranges, and a single bound
would let a wrong digit wrap into a valid-looking id that surfaces only much
later, across 300,000 frames, as a corrupt token cache. `_self_check` enumerates
all `prod(levels)` code tuples and asserts the ids come back as `arange`.

---

## 6. PCA and compactness

Used by: the validator's shattered-object detector.

Given the pixel coordinates of one block, decide whether it is a solid object or
has broken into confetti.

1. Stack the coordinates as a 2 x n matrix, and subtract the mean so it is
   centred.
2. Covariance `Sigma = P @ P.T / n`, a symmetric 2x2.
3. Eigendecompose. The eigenvectors are the object's own principal axes: the
   directions of greatest and least spread.
4. Project onto those axes and take the extent along each.

```
compactness = pixel_count / (major * minor)
```

A solid rectangle fills its own oriented box, so compactness is about 1.0.
Scattered debris fills almost none of it, about 0.05.

**Oriented, not axis-aligned, and this is not a style preference.** An
axis-aligned box around a square rotated 45 degrees has twice the area: the
diagonal is sqrt(2) times the side, and sqrt(2)^2 = 2. Compactness would read
about 0.5 for a perfectly intact rotated block, colliding with the genuinely
partially-occluded case that the occlusion floor makes common. Both arm
links rotate and a free-joint block rotates when pushed, so rotation is the
normal case here.

**The +1.0 in the extent.** A single row of pixels spans one pixel, not zero.
Without the +1, a 1-px-wide blob divides by zero and compactness comes back `inf`.

### The angle gotcha, which affects action-following (Q-4)

A PCA eigenvector is defined only up to sign: `v` and `-v` describe the same
axis. So the angle is only defined modulo pi, and it is canonicalised into
[0, pi). A link rotating through that boundary shows a jump of nearly pi.

Any code taking `sign(theta_t+1 - theta_t)` must **unwrap** the difference into
(-pi/2, pi/2] before taking the sign. Skip the unwrap, and roughly one step in
every half-turn reports the wrong direction of rotation.

Separately, `y` runs downward in image coordinates, so the angle increases
clockwise on screen. Action-following compares against a commanded joint sign
and must calibrate that sign against the data rather than assume it. Note that
ground truth itself only scores 83.1% on this measure, below the absolute 90%
bar the action-following requirement once set.

---

## 7. k-means, and the two ways this project got it wrong

Used by: gate row 2's floor.

Chop frames into 8x8x3 = 192-dimensional patch vectors and find 512 centroids
that minimise:

```
J = sum_i  min_k  ||x_i - mu_k||^2
```

Lloyd's algorithm alternates two steps until nothing changes: assign each point to
its nearest centroid, then move each centroid to the mean of its members. Neither
step can increase `J`, so it converges. But only to a **local** minimum, and which
one depends entirely on where you started.

That dependence caused two separate corrections. The register's history for the
k-means floor holds both, with their sizes; they are not restated here.

**k-means++** seeds the centroids with probability proportional to `D(x)^2`, the
squared distance to the nearest centroid already chosen. That spreads them out on
purpose instead of by luck, and it is the half of the correction that came from
initialisation.

**The other half, 0.75 dB, is a textbook train/test leak.** Scoring a codebook
partly on the patches it was fit to flatters it, because those patches are
exactly what the centroids were placed to cover. The held-out figure fits on the
473 train episodes and scores on the 27 val ones (16,200 frames), which is how a
tokenizer is scored too.

The floors at three vocabulary sizes are 27.09 dB (240 codes), 28.27 dB (512) and
29.39 dB (1024). All three remain **lower bounds**: 25 Lloyd iterations may not
have converged. `bench/patch_probe.py` prints both the leaked and the held-out
column, and the 0.75 dB leak is the difference between them.

---

## Where the numbers live

`docs/canonical_numbers.md` is the register. Every current value above is checked
against it, and each register entry names its source row in `runs.jsonl` or the
verification log. If a value above disagrees with the register, the register
wins.

### Unregistered figures used above

These have no register entry, because the register's rule is to add one when a
number reaches its **second** live doc. Each row names the live doc it already
appears in, so the register owner can decide. This file is not a second register
and should not become one.

| Figure | Also appears in | Verified by |
|---|---|---|
| ~~worst render-rounding palette distance~~ | **registered 2026-08-29 as `NUM-VAL-WORSTDIST`** | `python -m mirage.validator` |
| ~~`offpalette_tau` current setting~~ | **registered 2026-08-29 as `NUM-VAL-TAU`** | `mirage/configs/base.json`, `CALC` |
| STE derivatives at zero, three levels tables | `AGENDA.md` shrink ladder | `FSQ._self_check` |
| classification needs ~99.6% pixel accuracy | `docs/phase1_structural_plan.md` | palette arithmetic on the 47,814 wrong-pixel cost |
| channel redundancy, 1.339 to 0.781 bits | this file only | `mirage/fsq_eval.py`, `entropy_split` |
| R2 channel 2 skew, 81% mass, 1.964 of 3 bits | this file only | same |
| ~578 off-palette px without the void entry | this file only | `mirage/validator.py` docstring |
| the (229, 191, 25) render rounding example | this file only | same |

Two of the top four were registered on 2026-08-29, as part of the item 6
preparation. They are struck through above rather than deleted, so the reason
they were flagged stays readable. **The two that remain still qualify**, and
registering them is the standing recommendation to the register owner. The bottom
four do not qualify, and adding them would be the duplication the register exists
to prevent.

### One figure here is arithmetic, not measurement

The RGB-cube volume numbers in section 1 are computed from `tau` for intuition.
No code prints them and nothing depends on them.
