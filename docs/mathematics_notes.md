# Mirage: Mathematics Notes

**What this is.** Every piece of mathematics the project actually runs, explained
from scratch, with a pointer to the code that computes it. Companion to
`docs/decision_notes.md`, which covers *why* each choice was made; this file
covers *what the formula is and how to read it*.

**Derived, not authoritative.** Formulas here are transcriptions of code. When
this file and the code disagree, the code is right and this file is stale.

## Doc class: Live

`docs/canonical_numbers.md` sorts every doc into Log, Register, Live or Frozen,
and a **Live** doc may not state a registered figure inline - it cites the name
and the `NUM-` id instead. This file is Live. It is not yet named in the
register's class table; adding it there is the register owner's call, not this
file's.

Two conventions follow from that, and they are the only ones you need:

- **Prose names the quantity and cites the id.** Never the bare value. So "the
  held-out k-means floor (`NUM-TOK-FLOOR512`)", not the number.
- **Fenced worked blocks substitute the value once, to show the arithmetic.**
  Every number inside a fenced block below is a **recomputation** from the cited
  inputs, in the sense the register already uses for `NUM-BAR-ROW2` ("derived -
  recompute, do not restate"). A recomputation is not a second source of truth.
  If a cited input moves, the block is stale and recomputing it is the fix.

A handful of figures here are in no register entry yet. They are marked
**unregistered** at the point of use and listed at the end, because the register's
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

Squared Euclidean distance and logarithms, over and over. The single exception is
the straight-through estimator in the quantizer, which is not really mathematics
at all: it is an admitted lie about a derivative, adopted because rounding has no
useful one.

---

## 1. Euclidean distance in colour space

Used by: the frame validator, F-9, `offpalette_tau`.

A pixel is a point in 3D space, `(R, G, B)`, each channel 0..255. Distance
between two colours is Pythagoras extended to three axes:

```
d(p, q) = sqrt( (Rp-Rq)^2 + (Gp-Gq)^2 + (Bp-Bq)^2 )
```

The palette is the scene XML's six `rgba` colours plus an implicit `void` entry
for MuJoCo's clear colour, which shows through past the far table edge and is in
no `rgba` attribute. That total is the distinct-colour count F-2 measures
(`NUM-DATA-COLOURS`). Without the void entry a flawless frame reads about 578
off-palette pixels (**unregistered**, `mirage/validator.py` docstring) and F-9
can never be met.

For each pixel, find the closest palette entry:

```
nearest(p) = argmin over k of  ||p - c_k||^2
```

**Squared, not square-rooted, on purpose.** `sqrt` is monotonically increasing,
so whichever `k` minimises `d^2` also minimises `d`. Skipping the root saves one
square root per palette entry per colour. The root is taken once, for the winner
only, because `tau` is expressed in real distance units.

**A second optimisation worth noticing.** `_label` maps the frame's *distinct*
colours rather than its pixels. A ground-truth frame holds `NUM-DATA-COLOURS` of
them, so the argmin is 7x7 instead of 4096x7. It stays correct on a decoder
output that emits thousands, just slower.

### tau is the radius of a ball

Draw a sphere of radius `tau` around each palette point. A pixel inside some
sphere is legal; outside all of them, it counts toward `offpalette_px`.

`offpalette_tau` lives in `mirage/configs/base.json` and is now registered as
`NUM-VAL-TAU`. Its current setting was chosen against ground-truth frames, where
the worst distance any rendered pixel sits from its own palette entry is
`NUM-VAL-WORSTDIST`. The ratio between them, `NUM-VAL-HEADROOM`, is the slack
available. **All three move when build order item 6 lands**: it re-runs the F-9
sweep against reconstructions rather than ground truth, so expect tau to rise and
`NUM-VAL-FALSEPOS` to stop reading zero.

> **Digit collision, and it has already caused one misreading.**
> `NUM-VAL-WORSTDIST` is **RGB units of colour distance**. `NUM-TOK-LEAK` carries
> the same digits but is **dB of PSNR**, and describes the k-means train/val
> split leak in section 7. The two are unrelated and share no derivation. Check
> the unit before quoting either - the register now carries this warning on both
> rows, which is the only place someone looking up one would see the other.

Recomputation, from `tau` as configured today:

```
legal volume per entry = (4/3) * pi * tau^3
tau = 0.75  ->      1.8 colour-cube units
tau = 8.0   ->  2,145
tau = 32.0  -> 137,258
```

Volume grows as `tau^3`. Raising tau from 8 to 32 makes the legal region **64x**
larger per palette entry. Every doubling buys tolerance for decoder blur and
throws away detection power cubically fast. That tradeoff is the entire content
of the build-order item 6 calibration.

### Why nearest-palette rather than exact equality

`rgba * 255` does not land on an integer. link0's `0.90 0.75 0.10` renders as
(229, 191, 25), not (230, 191, 26), and not by a rule worth modelling: 0.65
rounds up to 166 while 0.90 rounds down to 229. Under exact equality against a
byte-rounded palette, **four of the seven entries match zero pixels** on a
flawless frame, and the validator reports block0, block2, link1 and table as
missing. (All **unregistered**, `mirage/validator.py` docstring.)

This is also why `Palette.rgb` stays unrounded float 0..255. Measured against
229.5 rather than 230, the worst rendered pixel distance is `NUM-VAL-WORSTDIST`.
Rounding the palette doubles that for no gain.

---

## 2. MSE and PSNR

Used by: Q-1, the gate bar `NUM-BAR-Q1`, the training loss.

### Mean squared error

Line up the original frame and the reconstruction, subtract, square, average:

```
MSE = (1/N) * sum_i (x_i - xhat_i)^2
N = 64 * 64 * 3 = 12,288 values per frame     (recomputed from configs/base.json)
```

Squared rather than absolute for two reasons: it punishes one large error more
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

Worked backwards from the Q-1 bar (`NUM-BAR-Q1`):

```
MSE = 255^2 / 10^(30.0/10) = 65,025 / 1000 = 65.03
```

So the bar means the average squared error per colour channel is about 65, i.e. a
typical channel is off by about sqrt(65) = 8 out of 255.

**The rule to memorise: every 3 dB halves the MSE**, because `10*log10(2) =
3.01`. So the gap between R1 at convergence (`NUM-TOK-R1-60`) and the held-out
k-means floor (`NUM-TOK-FLOOR512`) recomputes to:

```
31.095 - 28.27 = 2.825 dB  ->  10^(-2.825/10) = 0.52
```

The tokenizer makes roughly **half** the squared error of the baseline. The bar
that gap has to clear is `NUM-BAR-ROW2`, which the register marks derived:
recompute it as `NUM-BAR-Q1` minus `NUM-TOK-FLOOR512` rather than restating it.

### One detail that is easy to get wrong

PSNR is computed on **uint8** reconstructions, after rounding, not on the raw
float output. That is what the pipeline delivers and what item 6 hands the
validator. Measuring before the round reports a number nothing downstream ever
sees.

---

## 3. Converting decibels into a pixel budget

The mean squared distance between two distinct palette entries is
`NUM-TOK-PIXELCOST`. That is the cost, in error units, of painting one pixel a
completely different palette colour, and it converts any dB figure into a count
of ruined pixels:

```
SSE per frame      = MSE * 12,288
fully wrong pixels = SSE / NUM-TOK-PIXELCOST
```

Recomputation at three cited levels, per 4,096-pixel frame:

| Level | id | MSE | SSE per frame | Equivalent wrong pixels |
|---|---|---|---|---|
| held-out k-means floor | `NUM-TOK-FLOOR512` | 96.85 | 1,190,040 | **24.9** |
| the Q-1 bar | `NUM-BAR-Q1` | 65.03 | 799,027 | **16.7** |
| R1 at convergence | `NUM-TOK-R1-60` | 50.53 | 620,957 | **13.0** |

This is where "cut the error count by a third" comes from: about 25 wrong pixels
down to about 17.

### The design argument this number settled

`NUM-TOK-PIXELCOST` is large enough that per-pixel 7-way classification with
cross-entropy needs roughly 99.6% pixel accuracy to clear `NUM-BAR-Q1`
(**unregistered**, `docs/phase1_structural_plan.md`), because every miss costs
the full amount. MSE regression can hedge, outputting a blend and paying a small
penalty instead of a catastrophic one. That is why the tokenizer trains on plain
MSE and nothing else.

The cost of that hedge is real: MSE rewards blurring edges, and
`NUM-TOK-EDGESHARE` says almost all of the floor's squared error already lives in
the minority of patches that are not a single flat colour. The counterweight is
item 6. `offpalette_px` on reconstructions punishes precisely the blur that PSNR
rewards, and the two cannot both be gamed.

---

## 4. Shannon entropy

Used by: Q-2, the gate bar `NUM-BAR-Q2`, gate row 3.

Take the count of each token id over `NUM-DATA-FRAMES`, convert to probabilities
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
| R2 measured | `NUM-TOK-ENT-R2` of that maximum |
| R1 measured | `NUM-TOK-ENT-R1` |

The Q-2 bar is the ratio `H / log2(512)` against `NUM-BAR-Q2`, a normalised "how
much of the vocabulary is doing work" score. Recomputing R2 in bits:

```
0.776 * 9 = 6.98 bits of the 9 available
```

**The ceiling is not 100%.** `NUM-TOK-Q2CEIL` is the provable ceiling, because
`NUM-TOK-FLAT` of interior cells have a fully flat receptive field and a flat cell
has little to say. That ceiling sits comfortably above `NUM-BAR-Q2`, which is why
the data does not force a Q-2 miss.

### Splitting the entropy: skew versus redundancy

A token id is three digits packed together, so the per-channel digit
distributions fall out of the same counts vector, with no GPU and no re-encode:

```
H(d0) + H(d1) + H(d2)  -  H(d0, d1, d2)  =  redundancy  >= 0
   marginal sum              joint
```

This quantity is total correlation, sometimes multi-information. It is zero only
when the three digits are statistically independent, and it can never be
negative. The two terms fail for different reasons and have different fixes:

- **Marginal skew.** One channel's latent sits off centre in the `tanh` bound and
  never reaches most of its levels. R2's channel 2 puts 81% of its mass on digits
  0 and 1 and returns 1.964 of 3 bits.
- **Redundancy.** The channels encode copies of each other. This is what the
  attention layer fixes: R1 to R2 it falls from 1.339 to 0.781 bits.

Both bullets are **unregistered**; source is `mirage/fsq_eval.py`,
`entropy_split`, reproduced by `python -m mirage.fsq --eval`.

That 0.558 bit drop is 76% of `NUM-TOK-ATTNENT`, attention's entire entropy gain,
and it is the mechanism behind a result no design document predicted. Attention
bought `NUM-TOK-ATTN` of quality for `NUM-TOK-ATTNPARAM` parameters, a measured
non-lever, and paid for itself in entropy instead.

Splitting the two matters because the planned remedy for a Q-2 miss, the shrink
ladder, is a *collapse* fix, and neither term here is collapse. Zero of 512 codes
have zero count in either rung, which is also what retired the collapse worry
that `NUM-TOK-LIVE512` used to support.

---

## 5. FSQ: quantizing without killing the gradient

Used by: the tokenizer bottleneck. This is the densest mathematics in the
project.

### The problem

The encoder produces a continuous number `z`. Tokens must be integers. But
`round()` has derivative zero almost everywhere, so gradient descent receives no
signal through it and training dies.

### Step 1: squash into a bounded range

With `L` levels for this channel:

```
half     = (L - 1) * (1 + eps) / 2
offset   = 0.5 if L is even else 0
shift    = atanh(offset / half)
bound(z) = tanh(z + shift) * half - offset
```

`tanh` maps all of the real line into (-1, 1), so multiplying by `half` maps any
input into a bounded interval containing exactly `L` integer grid points.

- `eps` widens the bound just past the outermost level, so the `tanh` asymptote
  does not sit exactly on a grid point it can never reach.
- `offset` and `shift` re-centre an even levels count, whose grid points straddle
  zero rather than including it.

### Step 2: the straight-through estimator

```python
q = q + (q.round() - q).detach()
```

Forward: `q + (round(q) - q)` equals `round(q)`. You get the integer.
Backward: `.detach()` zeroes the bracket's derivative, so autograd sees `q` alone
and passes the gradient through unchanged.

It is a deliberate lie about the derivative. It works because the lie is unbiased
on average.

### The subtlety that invalidates naive levels comparisons

The STE bypasses only the rounding. The `tanh` derivative survives into the
backward pass:

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

Switching `[5,5,5]` to `[4,4,4]` silently multiplies the effective bottleneck
learning rate by `0.668 / 1.001 = 0.67`, up to 1.5x across the full table.
Compare two levels tables at a single LR and part of the "levels result" is
actually an LR result. **Every step of the shrink ladder needs a paired LR
check.**

### Step 3: mixed radix, packing digits into one id

Same idea as reading `347` as `3*100 + 4*10 + 7`, except each place has its own
base:

```
id = d0 + L0*d1 + L0*L1*d2
```

For `[8,8,8]` that is plain base 8, `d0 + 8*d1 + 64*d2`, giving ids 0..511. For a
mixed table like `[8,6,5]` the place values are 1, 8, 48, giving 0..239.

**This is why FSQ has no codebook.** Classic VQ-VAE stores learned vectors and
needs commitment loss, EMA updates and dead-code restarts to keep them alive.
FSQ's mapping is a bijection guaranteed by arithmetic: every integer in
`[0, prod(levels))` has exactly one mixed-radix expansion, so no two codes can
collide and none can die from neglect. The entire maintenance apparatus is
replaced by a base conversion.

It is also why no auxiliary entropy loss is allowed if Q-2 misses. An auxiliary
loss undoes the reason FSQ was chosen over VQ: a Q-2 improvement would no longer
distinguish "the vocabulary is well used" from "the loss propped it up". Shrink
the vocabulary instead, and pay the floor cost the shrink ladder prices as
`NUM-TOK-FLOOR240` against `NUM-TOK-FLOOR512`.

The bound check when recovering digits is per channel, not against
`max(levels)`. A mixed table has three different digit ranges and a single bound
would let a wrong digit wrap into a valid-looking id, surfacing
`NUM-DATA-FRAMES` later as a corrupt token cache. `_self_check` enumerates all
`prod(levels)` code tuples and asserts the ids come back as `arange`.

---

## 6. PCA and compactness

Used by: the validator's shattered-object detector.

Given the pixel coordinates of one block, decide whether it is a solid object or
has broken into confetti.

1. Stack coordinates as a 2 x n matrix, subtract the mean so it is centred.
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
partially-occluded case that F-7 makes common. Both arm links revolve and a
free-joint block rotates when pushed, so rotation is the normal case here.

**The +1.0 in the extent.** A single row of pixels spans one pixel, not zero.
Without the +1 a 1-px-wide blob divides by zero and compactness comes back `inf`.

### The angle gotcha, which affects Q-4

A PCA eigenvector is defined up to sign: `v` and `-v` describe the same axis. So
the angle is only defined modulo pi, and is canonicalised into [0, pi). A link
rotating through that boundary shows a jump of nearly pi.

Any code taking `sign(theta_t+1 - theta_t)` must **unwrap** the difference into
(-pi/2, pi/2] before taking the sign. Skip the unwrap and roughly one step in
every half-turn reports the opposite direction of rotation.

Separately: `y` runs downward in image coordinates, so the angle increases
clockwise on screen. Q-4 compares against a commanded joint sign and must
calibrate that sign against the data rather than assume it. Note that Q-4's own
ceiling, `NUM-DATA-Q4CEIL`, sits below the score ground truth would need.

---

## 7. k-means, and the two ways this project got it wrong

Used by: gate row 2's floor.

Chop frames into 8x8x3 = 192-dimensional patch vectors and find 512 centroids
minimising:

```
J = sum_i  min_k  ||x_i - mu_k||^2
```

Lloyd's algorithm alternates two steps until stable: assign each point to its
nearest centroid, then move each centroid to the mean of its members. Both steps
can only decrease `J`, so it converges. But only to a **local** minimum, and
which one depends entirely on where you started.

That dependence caused two separate corrections. The register's supersession
chain for `NUM-TOK-FLOOR512` holds both, with their sizes; do not restate them
here.

**k-means++** seeds centroids with probability proportional to `D(x)^2`, the
squared distance to the nearest already-chosen centroid. That spreads them out
deliberately instead of by luck, and it is the half of the correction that came
from initialisation.

**The other half, `NUM-TOK-LEAK`, is a textbook train/test leak.** Scoring a
codebook partly on the patches it was fit to flatters it, because those patches
are exactly what the centroids were placed to cover. The held-out figure fits on
the train episodes and scores on the val ones (`NUM-DATA-SPLIT`,
`NUM-DATA-VALFRAMES`), which is the treatment a tokenizer gets.

The floors at three vocabulary sizes are `NUM-TOK-FLOOR240`, `NUM-TOK-FLOOR512`
and `NUM-TOK-FLOOR1024`. All three remain **lower bounds**: 25 Lloyd iterations
may not have converged. `bench/patch_probe.py` prints both the leaked and the
held-out column, and the difference between them is `NUM-TOK-LEAK` made visible.

---

## Where the numbers live

`docs/canonical_numbers.md` is the register. Every `NUM-` id above resolves
there, with its source row in `runs.jsonl` or the verification log, and
`python check.py` fails if any id cited here is undefined.

### Unregistered figures used above

These are stated inline because the register's rule is to add an entry when a
number reaches its **second** live doc. Each row names the live doc it already
appears in, so the register owner can decide. This file is not a second register
and should not become one.

| Figure | Also appears in | Verified by |
|---|---|---|
| ~~worst render-rounding palette distance~~ | **registered 2026-08-29 as `NUM-VAL-WORSTDIST`** | `python -m mirage.validator` |
| ~~`offpalette_tau` current setting~~ | **registered 2026-08-29 as `NUM-VAL-TAU`** | `mirage/configs/base.json`, `CALC` |
| STE derivatives at zero, three levels tables | `AGENDA.md` shrink ladder | `FSQ._self_check` |
| classification needs ~99.6% pixel accuracy | `docs/phase1_structural_plan.md` | palette arithmetic on `NUM-TOK-PIXELCOST` |
| channel redundancy, 1.339 to 0.781 bits | this file only | `mirage/fsq_eval.py`, `entropy_split` |
| R2 channel 2 skew, 81% mass, 1.964 of 3 bits | this file only | same |
| ~578 off-palette px without the void entry | this file only | `mirage/validator.py` docstring |
| the (229, 191, 25) render rounding example | this file only | same |

Two of the top four were registered on 2026-08-29, as part of the item 6
preparation - they are struck above rather than deleted, so the reason they were
flagged stays legible. **The two that remain still qualify** and are the standing
recommendation to the register owner. The bottom four do not, and adding them
would be the duplication the register exists to prevent.

### One figure here is arithmetic, not measurement

The RGB-cube volume numbers in section 1 are computed from `tau` for intuition.
No code prints them and nothing depends on them.
