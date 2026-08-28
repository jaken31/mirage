# Mirage: Decision Notes

**What this is.** Every design choice the project has already made, in plain
words, with the reason it was made, the specific thing that would tell us it was
wrong, and one or two fallbacks if it fails. Companion to `docs/timeline.md`.

**How to read an entry.** Four lines, always the same shape:

- **Chose** - what we are doing.
- **Why** - the argument, usually one sentence.
- **Wrong if** - the observable signal that this was a mistake. If a decision has
  no such signal, that is itself a problem worth naming.
- **Fallback** - what we do instead, in preference order.

**One rule that applies to every entry.** Fallbacks are not built in advance.
Each one has a trigger written next to it. Building fallbacks speculatively is
how a five-day stage becomes a three-week one, and it is the most common way this
kind of project fails.

---

## A. Where the project runs, and how pictures get drawn

### A1. Run directly on Windows, not inside the Linux compatibility layer

- **Chose** everything runs natively on Windows.
- **Why** the compatibility layer on this machine cannot reach the graphics
  hardware at all. The system log fails at adapter enumeration on every boot, no
  graphics device node is ever created, and so the graphics stack silently
  substitutes a software imitation that runs on the main processor - roughly fifty
  times too slow. Verified across two restarts, a pre-release update of the
  compatibility layer, and every available override. The evidence is in
  `CLAUDE.md` and `bench/egl_probe.py`.
- **Wrong if** nothing. This is measured, not assumed, and it should not be
  revisited. Note that the general-purpose computation path inside the
  compatibility layer still works fine - it uses a different mechanism - but
  there is no longer any reason to use it.
- **Fallback** (1) if a future driver update creates the missing device node, the
  probe script re-runs in about a minute and tells us. (2) A second machine with
  a desktop graphics card removes the problem entirely, at the cost of every
  measurement in the plan needing to be retaken there.

### A2. Use the simulator's standard drawing path, with a hand-built one held in reserve

- **Chose** the simulator's default drawing library on Windows, and refuse the
  software-only alternative it also ships.
- **Why** on Windows there are exactly two options and one of them draws on the
  main processor. There is no third choice to evaluate.
- **Wrong if** copying a finished picture off the graphics card takes anywhere
  near 30 milliseconds, which is what a public discussion thread reports for
  exactly this path. Our whole budget is 2 milliseconds per picture.
- **Fallback** (1) hand-build a direct connection to the graphics driver,
  bypassing the standard library's window machinery. Three to five days. (2) If
  even that is slow, generate the pictures on a machine that can use the
  well-supported path, once, and copy the finished dataset over. The dataset is
  under 20 GB and is generated exactly once, so this is a genuinely reasonable
  escape hatch that costs nothing after day one.

### A3. Draw twice per picture now - once in colour, once colour-coded by object

- **Chose** two drawing passes: the normal picture, plus a second pass where each
  object is painted a flat identifying colour so we can count how many pixels of
  each block are visible.
- **Why** it is the direct, obvious way to know when a block is fully hidden,
  which is what makes the most interesting result in the project measurable at
  all. Storing the colour-coded pictures would double the dataset for nothing, so
  we count pixels and throw the second picture away.
- **Wrong if** copying a picture off the graphics card costs more than about half
  a millisecond. Two passes pay that fixed cost twice, and at that point the two
  copies consume the entire per-picture budget.
- **Fallback** (1) Draw *only* the colour-coded pass and build the real picture
  from a colour lookup table. This works because the scene is deliberately flat -
  no shadows, no textures, no blended edges - so every pixel belongs to exactly
  one object, and the colour picture genuinely *is* a lookup over the coded one.
  Bonus: the colour-count ceiling becomes structural rather than something we
  test for, and repeatability strengthens because no shading arithmetic touches
  the pixels. Cost: reversibility - if we ever want shading or textures, the
  second pass has to come back. Switching later costs one regeneration, about ten
  minutes. (2) If both passes and the lookup trick are too slow, the drawing path
  itself is the problem and the answer is A2's fallback, not a change to what we
  draw.

### A4. Turn off the simulator's debugging overlays

- **Chose** contact markers, joint axes, and similar visual aids are explicitly
  disabled.
- **Why** they add colours that are not in our palette, which corrupts both the
  colour-count ceiling and the object-identification pass.
- **Wrong if** never; this is a one-line setting with no downside.
- **Fallback** none needed.

---

## B. Where the code lives

### B1. The picture factory is a standalone program, not a plug-in

- **Chose** a self-contained executable that writes files to disk. The scripting
  side only ever reads files. No bridging layer between the two languages.
- **Why** the decisive argument is the memory-error checker. We require it to run
  clean over a full generation run, and running it through a plug-in inside a
  scripting interpreter requires loading its runtime before the interpreter
  starts plus a list of exceptions for the interpreter's own internals. On a
  standalone program it is one command. The exact size of that gap is unverified,
  and the decision does not depend on it - the standalone program is simpler
  regardless, and it makes "delete the simulator afterwards" literal: after the
  first stage, nothing on the scripting side touches the simulator at all.
- **Wrong if** we find ourselves wanting to drive the simulator interactively
  from the scripting side, which would mean a bridging layer after all.
- **Fallback** (1) Add a thin command-line interface to the standalone program
  and drive it by launching it. Covers almost every interactive need without a
  bridge. (2) Add the bridging layer, and accept a list of memory-checker
  exceptions as the price.

### B2. Two separate build systems, no unifying tooling

- **Chose** the low-level side and the scripting side each use their own
  conventional build tooling. No monorepo framework on top.
- **Why** there are two components with one file-shaped connection between them.
  A unifying layer would be pure overhead.
- **Wrong if** the number of components grows past three or four, or if setting
  up a fresh machine becomes an undocumented ritual.
- **Fallback** (1) Write the whole setup as a script in the readme. (2) A
  container image, which is the honest answer if setup ever takes more than
  fifteen minutes.

### B3. The measurement code is separate from the file-writing code

- **Chose** the part that reads the simulator's internal state and the part that
  serialises it to disk are separate files.
- **Why** the measuring part is the only low-level code that genuinely *must*
  exist, because its inputs disappear the moment the simulator is deleted. The
  writing part is generic. Mixing them means the irreplaceable code is tangled
  with the replaceable code.
- **Wrong if** never; this is free.
- **Fallback** none needed.

---

## C. How the data sits on disk

### C1. Three files per batch: pictures, per-picture notes, summary

- **Chose** raw picture bytes in one file, fixed-width per-picture notes in a
  second, and a small human-readable summary in a third.
- **Why** the pictures can be read straight off disk with no header parsing and
  no arithmetic about where one picture ends and the next begins.
- **Wrong if** the number of files becomes unmanageable, or if reading turns out
  to be a bottleneck during training.
- **Fallback** (1) Increase the batch size so there are fewer, larger files. (2)
  Adopt an off-the-shelf columnar file format, which buys compression and tooling
  at the cost of a dependency and a much less obvious read path.

### C2. Separate files rather than one interleaved file - for simplicity, not speed

- **Chose** picture data and note data live in different files.
- **Why** honestly recorded: the speed argument for this was **wrong and has been
  corrected**. We measured it. Reading a single field out of an interleaved file
  gives you a non-contiguous view as expected, but the framework accepts it
  anyway, and the cost is one extra in-memory copy of about 197 kilobytes per
  training batch - which is noise. Separate files win because reading them
  requires no stride arithmetic and no interleaved-format subtleties. That is the
  entire argument.
- **Wrong if** we need to read pictures and notes together in a tight loop where
  two file handles cost something measurable. We do not currently.
- **Fallback** (1) Interleave them; the measurement says it works. (2) Keep both
  and pick per use case, which is worse than either and should not be done.

### C3. Store how many pixels of each block are visible, not a yes/no "hidden" flag

- **Chose** a count per block per picture.
- **Why** a yes/no flag bakes a threshold into the dataset permanently. Counts let
  the "how often is a block hidden" measurement and the "does the program
  remember hidden blocks" measurement both be recomputed at *any* threshold,
  forever, without regenerating 300,000 pictures.
- **Wrong if** never realistically; the counts cost 6 bytes per picture.
- **Fallback** none needed. This is the cheap version of a general principle used
  three times in this project: **store the measurement, derive the verdict.**

### C4. Episode number and step number are mandatory, not optional

- **Chose** every picture records which episode it belongs to and where in that
  episode it sits.
- **Why** training uses windows of fifteen consecutive pictures. A window that
  straddles a reset between episodes is pure noise, and without these two numbers
  the reader cannot detect one. The failure mode is the worst available: it
  produces a plausible-looking training curve while quietly teaching the program
  nonsense.
- **Wrong if** never.
- **Fallback** none. If these were missing we would have to regenerate the
  dataset.

### C5. The summary file is written last, and its existence means "complete"

- **Chose** picture and note data are written first; the summary file is written
  after both are closed. The reader skips any batch without a summary file.
- **Why** it makes a crash safe with zero extra machinery. No lock files, no
  partial-write detection, no cleanup pass.
- **Wrong if** we ever need to read a batch *while* it is being written.
- **Fallback** (1) Write to a temporary name and rename at the end - renaming is
  atomic on both platforms. (2) A real lock file, which is what we are
  deliberately avoiding.

### C6. Per-picture notes are 46 bytes with a named consumer for every field

- **Chose** eight fields, and the rule that **a field with no named consumer does
  not ship.** Total overhead: 0.37% of the picture data.
- **Why** it is the only way to stop this record growing without limit. Every
  field currently in it is pointed at by a specific measurement or check
  elsewhere in the plan.
- **Wrong if** a later stage needs a field we did not store - which would mean
  regenerating.
- **Fallback** (1) Regenerate; **measured at 45-50 seconds**, not the ten minutes
  first guessed here, which is the real reason this is a low-stakes decision.
  (2) Add a second notes file alongside the existing one for the new field, keyed
  by picture index, avoiding the regeneration entirely.
- **Held up, and grew a ninth field for free.** On 2026-08-28 a ninth piece of
  information was wanted - which of the two behaviours produced an episode, the
  scripted reach or the random one - and it went into a **spare bit of an existing
  field** rather than a new one. The record is still 46 bytes. The cost is that
  one byte now means two things, so every piece of code that reads it has to be
  told; miss one, and the "arm touched a block" rate silently reads over 50%
  instead of 16.63%.

---

## D. Naming things so results can be traced

### D1. Every output file's name contains a fingerprint of the settings that produced it

- **Chose** a short fingerprint computed from the relevant settings is part of
  every filename.
- **Why** this is the highest-leverage twenty lines in the project. It means
  **no cache-invalidation code exists at all** - a stale file is unreachable
  because nothing ever computes its name. "Have I already computed this?" becomes
  "does this file exist?" Two runs cannot collide. And every number in the final
  report walks back to the exact pictures it came from.
- **Wrong if** we change an implementation without changing any setting. Then the
  fingerprint is identical while behaviour differs - see D3.
- **Fallback** (1) Include a manually incremented version number in the
  fingerprint. Rejected below, in D4. (2) Include the code version in the
  fingerprint, which invalidates every stored result on every commit and makes
  the project unworkable.

### D2. The fingerprints form a tree, not a flat list

- **Chose** the dataset's fingerprint is computed first; the compressor's includes
  the dataset's; the predictor's includes the compressor's; and the frame
  checker's branches off the dataset's independently.
- **Why** this is what makes two common operations non-destructive. Retuning a
  frame-checker threshold does not invalidate a trained program whose output
  never changed. Retraining the compressor does not invalidate the dataset. A
  flat scheme makes every change invalidate everything.
- **Wrong if** the dependency structure turns out to be different from what we
  assumed - for instance if the frame checker were to influence training, which
  it does not.
- **Fallback** (1) Recompute the tree; it is one function. (2) If the structure
  becomes genuinely tangled, fall back to storing the full settings inside each
  file and comparing them, which is slower but cannot be structurally wrong.

### D3. Record the code version *inside* every output file, but never in the fingerprint

- **Chose** the exact code revision is stored as a field in the file. It does not
  contribute to the name.
- **Why** the honest limitation of D1 is that the fingerprint covers settings, not
  code. That splits into two problems: *detecting* that a file is stale (not
  cheaply solvable) and *diagnosing* where a file came from (solvable). We solve
  the second and accept that the first stays manual.
- **Wrong if** stale files start causing confusion in practice. The symptom is a
  result that cannot be explained by any setting.
- **Fallback** (1) Fingerprint the compiled program itself for the low-level
  side, which is a real solution for that half. (2) Add a manual "regenerate
  everything downstream of X" command and use discipline, which is what most
  projects actually do.

### D4. No manually maintained version numbers

- **Chose** explicitly no hand-incremented version integers in the settings
  sections.
- **Why** seven hand-bumped numbers across seven sections will rot - someone
  forgets - and they only ever answered half of the unsolvable problem anyway.
- **Wrong if** we end up wanting exactly that. Then D3's second fallback is the
  same thing with a better name.
- **Fallback** (1) The compiled-program fingerprint from D3. (2) Hand-bumped
  numbers, accepting that they will rot.

### D5. The compressed-picture cache is named after the trained program's actual bytes, not its settings

- **Chose** the cache of pre-compressed pictures is keyed by a fingerprint of the
  trained compressor file's contents.
- **Why** this is a corrected bug, worth reading twice. Two training runs with
  *identical settings but different random starting points* share a
  settings-based fingerprint and produce completely different compressed output.
  Under a settings-based name, the next stage would silently load the wrong cache
  - precisely the failure the whole naming scheme exists to prevent.
- **Wrong if** never; this is strictly more correct.
- **Fallback** none needed. The trained file still records its settings
  fingerprint internally, so we get content-addressing for correctness and
  settings-linkage for traceability, both.

### D6. The scene file is now frozen, and the picture size moved into settings

- **Chose** the size of the drawing buffer is written into the simulator from the
  settings file at startup, rather than stated in the scene file where it used to
  live. Added 2026-08-28.
- **Why** the dataset's fingerprint covers **the scene file's raw bytes**, so any
  edit to it - including a comment - changes the fingerprint and orphans the
  300,000 pictures already on disk. Measuring both picture sizes (I1) needs the
  buffer at the larger size, which the scene file stated literally as the smaller
  one. Moving that one value into settings costs three lines of low-level code,
  leaves the scene file byte-for-byte unchanged, and turns "a second picture size"
  into a settings-only change. The alternative was regenerating the existing
  dataset in order to make a second one possible.
- **Wrong if** someone later needs to change the scene itself - different block
  positions, a third arm link. That is a real regeneration and always was; this
  decision does not make it worse, it just means the *buffer size* is no longer a
  reason to trigger one.
- **Fallback** none needed, but one consequence has to be carried: **the scene
  file's stated buffer size is now decorative, and the comment explaining that
  cannot be added to the scene file.** It lives in the low-level code and in the
  design document.
- **Watch for** the two existing safety checks still work and must not be
  removed - one compares the buffer the graphics driver actually gave us against
  the requested size, the other compares it against the settings file. Both now
  read the settings value, so both still catch a mismatch.

---

## E. What behaviour gets recorded

### E1. The random-versus-scripted mix is chosen per episode, not per picture

- **Chose** each episode is entirely random arm movement or entirely a scripted
  reach, decided by a coin flip at the start.
- **Why** a scripted reach needs several consecutive steps to be a reach at all.
  Flipping a coin every picture destroys the exact property the mix exists to
  create. The requirement that all nine actions appear roughly evenly still holds
  in aggregate, because the random half carries it.
- **Wrong if** the action histogram comes out lopsided.
- **Fallback** (1) Adjust the mix ratio away from 50/50 until the histogram
  flattens. (2) Add a third behaviour - a slow sweep - to fill whichever actions
  are underrepresented.

### E2. Episodes are much longer than the training window

- **Chose** episodes of roughly 300 steps, against a 15-picture training window.
- **Why** direct arithmetic: at 300 steps, 4.7% of pictures are unusable because
  they sit too close to a reset. At 100 steps it is 14%. Longer episodes waste
  less.
- **Wrong if** long episodes drift into a degenerate state - the arm parked in a
  corner, nothing happening.
- **Fallback** (1) Shorten episodes and accept the higher waste; 14% is
  survivable. (2) Reset the blocks mid-episode without resetting the counter,
  which keeps windows valid while refreshing the scene.

### E3. No dedicated replay mode

- **Chose** repeatability is tested by generating twice with the same starting
  seed and comparing the picture files byte for byte. No separate playback
  feature.
- **Why** the test is one command against the real code path. A replay feature
  would be new code that could itself be wrong.
- **Wrong if** we need to reproduce a *specific* picture from the middle of a run
  for debugging without regenerating everything before it.
- **Fallback** (1) Generate a single short episode with that episode's seed -
  seeds are per-batch, so this already works. (2) Build the replay feature, if a
  real need appears.

---

## F. The automatic frame checker

### F1. It emits measurements, never verdicts

- **Chose** the checker reports a fixed list of numbers per picture. "This picture
  is bad" is a threshold rule written in the settings file, layered on top.
- **Why** three payoffs. The longest-sequence result can be recomputed under any
  threshold set without re-running a single sequence - about 3 megabytes of stored
  numbers buys that permanently. Thresholds become tunable data rather than code
  changes. And the "zero false alarms" requirement becomes an executable
  procedure rather than a judgement call.
- **Wrong if** the measurements turn out not to capture a failure mode we care
  about - a picture that is obviously broken to a human but scores fine on every
  number.
- **Fallback** (1) Add a measurement, then recalibrate. Since the stored numbers
  are per-picture, this is additive. (2) Add human spot-checking of a random
  sample as a separate, honest, non-automated gate.

### F2. Count pixels by colour rather than finding connected shapes

- **Chose** count how many pixels of each palette colour are present, and where.
  Explicitly do **not** group adjacent same-coloured pixels into objects.
- **Why** this is a correctness argument, not an effort-saving one. We *require*
  blocks to be fully hidden by the arm in at least 3% of pictures, so partial
  hiding is common. When the arm crosses a block, shape-grouping splits it into
  two disconnected pieces and reports **two blocks** - a phantom object.
  Suppressing that needs a "same-coloured pieces are one object" merge rule,
  which reduces shape-grouping back to colour counting with an extra pass bolted
  on the front. Colour counting is immune to this by construction.
- **Wrong if** we need to distinguish two same-coloured objects, which our
  palette design specifically prevents.
- **Fallback** (1) Add shape-grouping later as a *diagnostic on already-failing
  pictures* - "this block fragmented into 5 pieces, the largest being 12 pixels".
  That invalidates no earlier measurement, and designing it against real observed
  failures beats designing it now against guesses. (2) Give every object its own
  colour, which the palette budget currently allows.

### F3. Fragmentation is measured with a rotation-tolerant bounding shape

- **Chose** compute how tightly a colour's pixels are packed, using a bounding box
  aligned to the object's own orientation rather than to the screen.
- **Why** both arm segments rotate, and a block spins when pushed. A
  screen-aligned box around a square rotated 45 degrees has twice the area, which
  scores about 0.5 - colliding exactly with the score for a partially hidden
  object. Two very different failures would be indistinguishable. The
  orientation-aware version is immune, is about five lines, and shares its
  machinery with two other measurements we need anyway.
- **Wrong if** the machinery misbehaves on very small or very thin pixel groups,
  which is plausible for a nearly fully hidden block.
- **Fallback** (1) Skip the measurement below a minimum pixel count and report it
  as unavailable rather than as a low score. (2) Use the ratio of the object's two
  principal extents instead, which is cruder but degrades more gracefully.

### F4. Assign each pixel to its *nearest* palette colour, not an exact match

- **Chose** nearest-colour assignment with the distance recorded.
- **Why** when the prediction program draws a block in a slightly wrong shade,
  exact matching counts zero pixels and reports "block missing" - conflating two
  completely different failures. Nearest-colour separates "the colours drifted"
  from "the block is gone".
- **Wrong if** two palette colours are close enough that pixels get assigned to
  the wrong one.
- **Fallback** (1) Space the palette colours further apart in the scene file; we
  have colour budget to spare. (2) Report both the exact-match count and the
  nearest-match count and treat the gap as its own signal.

### F5. The measurement steps run in a fixed order

- **Chose** count distinct colours on the raw picture *first*, then do
  nearest-colour assignment, then everything else.
- **Why** after assignment, the number of distinct colours cannot exceed the
  palette size by definition, so measuring it afterwards would always pass and
  measure nothing. And "off-palette pixels" only has meaning as "distance to the
  nearest palette colour exceeds a limit", which requires the assignment to have
  happened.
- **Wrong if** never; this is forced by the definitions.
- **Fallback** none needed.

### F6. The measurement list is generous; the pass/fail rule is minimal

- **Chose** measure many things, but build the actual pass/fail rule from as few
  uncorrelated measurements as possible. Concretely, the distinct-colour count is
  measured always but only *enforced* on original pictures, never on rebuilt or
  imagined ones.
- **Why** two reasons. First, the distinct-colour ceiling is a statement about the
  drawing setup; a rebuilt picture emits hundreds of slightly different shades by
  construction and that is not a violation. Second, and more general: under a rule
  of "fail if any measurement exceeds its limit", the chance of a false alarm
  rises with every limit added. Every extra threshold makes "zero false alarms"
  strictly harder to achieve, and one of our existing measurements already
  responds to the same failure.
- **Wrong if** the minimal rule misses a real failure the generous list caught.
- **Fallback** (1) Promote one more measurement into the rule and recalibrate.
  (2) Use two rules - a strict one for alerting and a loose one for gating -
  which is more machinery than it sounds and should be a last resort.

### F7. Two checker modes, and calibrate twice

- **Chose** one mode that can see the true state of the world from the recorded
  notes, and one that sees only pixels. Calibrate the pixels-only mode against the
  truth-aware mode - **twice**, once on original pictures and once on the
  compressor's rebuilt ones.
- **Why** the "zero false alarms" requirement is literally a threshold sweep of
  one mode against the other; without both modes there is no procedure for it.
  The second calibration exists because original pictures are perfect while
  imagined pictures carry real artefacts, so zero false alarms on perfect
  pictures says nothing about the rate on imperfect ones. The rebuilt pictures
  cost nothing extra - the compressor stage produces them anyway for its own
  quality measurement, and they exist before the prediction stage needs them.

  **Strengthened 2026-08-28: the second calibration is not merely cheap, it is
  the only thing checking a hole in the compressor's own quality score.** That
  score rewards *blurring* the boundaries between objects - a soft blend across an
  edge costs it less than a crisp edge placed one pixel wrong - and boundaries are
  where 99.9% of the error lives. So a compressor could score well by smearing
  every edge. The checker's off-palette pixel count measures exactly what a smear
  produces, so the two numbers cannot both be satisfied by cheating. **Report them
  together or neither means much.** This is why the second calibration is a
  pass/fail row in the compressor stage's gate rather than a follow-up chore.
- **Wrong if** even the rebuilt pictures are too clean to represent what the
  prediction program will produce.
- **Watch for** the "at most 24 distinct colours per picture" rule must **not** be
  applied to rebuilt pictures. That rule describes the drawing program, which
  emits exactly 7 colours; a healthy compressor emits hundreds, because it
  outputs continuous colour rather than picking from a list. Applying it to
  rebuilt pictures would fail a perfectly good compressor.
- **Fallback** (1) Calibrate a third time against the prediction program's own
  early output, once it exists. (2) Set the thresholds deliberately loose and
  accept missing subtle failures, prioritising zero false alarms - which is the
  requirement's actual wording.

---

## G. Logging, and measuring honestly

### G1. Three logging layers; the human-written one is a hard requirement

- **Chose** a hand-written run notebook (one line per run: what changed, what the
  number was, what we concluded), plus an automatic machine-readable metric
  stream, plus an optional dashboard.
- **Why** the hand-written notebook is a shipping requirement and no tool
  produces it. It is the only layer that records *why* a run happened.
- **Wrong if** the notebook stops being maintained, which is the normal fate of
  such things.
- **Fallback** (1) Generate a skeleton entry automatically at the end of every
  run and require it be filled in. (2) Accept the automatic metric stream as the
  record and lose the reasoning, which is a real loss.

### G2. The machine-readable stream is the source of truth; the dashboard is optional

- **Chose** one logging function writes a line-per-event text file
  unconditionally and additionally forwards to the dashboard when a flag is set.
  About fifteen lines.
- **Why** the local file survives independently of any account or service, and the
  final results table is produced by a small script over that file - no interface
  produces it.
- **Wrong if** the local file becomes unwieldy at scale. It will not, at this
  project's size.
- **Fallback** (1) A local database file, still no service dependency. (2) Rely on
  the dashboard, accepting the account dependency.

### G3. Nothing is logged inside a timed section

- **Chose** timings are recorded into a pre-allocated block of memory and written
  out after the measurement finishes.
- **Why** two of our requirements are about the *slowest* pictures, not the
  average. A single one-millisecond hiccup from writing a log line is 2.5% of the
  frame budget and lands squarely in the slowest 1% - exactly the number we are
  trying to measure.
- **Wrong if** never; this is free.
- **Fallback** none needed.

### G4. Peak memory is read from the framework's own high-water mark

- **Chose** ask the computation framework for the highest memory it ever
  allocated, rather than reading the dashboard's graphics-card metric or polling
  the driver.
- **Why** the dashboard samples the driver on an interval and can miss a
  short-lived peak entirely. A high-water mark cannot. Relatedly, the dashboard's
  background sampling process must be turned off during timing runs, because it
  polls the graphics card on its own schedule and quietly inflates the
  slowest-1% number.
- **Wrong if** memory is consumed outside the framework's allocator, which the
  high-water mark would not see.
- **Fallback** (1) Cross-check once against the driver's own reported peak under
  a deliberately memory-heavy run. (2) Report both numbers and explain the gap.

### G5. A measurement refuses to run when the graphics card is not at full power

- **Chose** every timing row records the card's power state, clock speeds, power
  draw, and temperature, and the measurement harness refuses to start if the card
  is not at full power or its clocks are below a configured fraction of maximum.
- **Why** this came out of a real, invalid measurement. We measured raw memory
  bandwidth at 66-77 GB/s against an assumed 448 (since refuted - the real ceiling is 384 and the card delivers 308.3; that first attempt was invalid because it was drawing **6.16
  watts** at 36% of its maximum clock speed during a heavy computation - it never
  left a low-power state. That does not refute the assumed figure; it means the
  measurement was meaningless. Two consequences beyond the number itself: the
  "any measurement repeats within 5%" requirement is *unachievable* while the
  clock drifts, since the swing is more than an order of magnitude; and the
  slowest-1% requirement can be blown by a single power-state transition
  mid-run, which would look exactly like a programming problem. Cost of the fix:
  one system query per run, outside the timed section.
- **Wrong if** the laptop simply cannot *sustain* full power through a
  thousand-picture run. That is a project-level constraint worth discovering in
  week one rather than week ten, because it compresses every performance margin
  in the plan simultaneously.
- **Fallback** (1) Run measurements in short bursts with cooldowns between,
  reporting burst performance and stating the caveat plainly. (2) Take the
  performance measurements on a desktop machine and report laptop numbers
  separately, which changes what the project claims but keeps the claims true.

---

## H. The shape of the prediction program

### H1. The program is a pure function; a separate component owns the loop

- **Chose** the prediction program takes its inputs and returns its outputs with
  no hidden state. A separate component owns the loop, the remembered work, the
  pre-recorded instruction batches, and the scheduling.
- **Why** this one line assigns all five speed-ups to one side or the other -
  three belong to the loop owner, two are transformations of the program itself
  (a change to the stored numbers at load time, and a swapped-in component). That
  is why five independent switches produce five behaviours rather than
  thirty-two tangled code paths.
- **Wrong if** a speed-up genuinely needs to cross the boundary. The
  several-numbers-at-once technique is the one to watch.
- **Fallback** (1) That technique is already accommodated - see H2 - by passing
  the relevant control information in as an argument. (2) If a future technique
  truly needs both sides, widen the interface once, deliberately, rather than
  eroding it.

### H2. The "what may look at what" control information is always passed in explicitly

- **Chose** it is an argument, never baked into the program.
- **Why** baking it in blocks the several-numbers-at-once speed-up at the program
  level, no matter how cleverly the loop is scheduled - and that speed-up is
  *mandatory* if we end up on the larger-picture path. This is the single most
  expensive thing to retrofit in the whole design, which is why it is a
  from-the-first-commit constraint.
- **Wrong if** never. The cost is one extra argument.
- **Fallback** none needed.

### H3. The remembered-work buffer is a fixed size, allocated up front

- **Chose** when the remembered-work speed-up is enabled, its buffer is allocated
  at maximum size once and the unused part is masked off. It never grows.
- **Why** a growing buffer breaks the pre-recorded-instruction-batch speed-up,
  which is the headline win of the entire performance stage. Discovering that
  after building a growing buffer means rewriting it.
- **Wrong if** the maximum size is wasteful in memory. It is not - the buffer is
  under 30 megabytes on either path, against a 2 gigabyte allowance.
- **Fallback** none needed. Note this is *not* "the buffer is always on" - see
  H4.

### H4. Both a remembering path and a recompute-everything path ship

- **Chose** the slower recompute-everything path is kept permanently, not deleted
  once the fast path works.
- **Why** it is the baseline that the required 3x improvement is measured
  *against*. Delete it and the headline claim becomes unverifiable.
- **Wrong if** maintaining two paths causes them to drift apart and disagree.
- **Fallback** (1) An automated check that both paths produce identical output on
  a fixed input. (2) Keep the slow path but only exercise it at release time,
  accepting slower detection of drift.

### H5. No hardcoded picture sizes anywhere

- **Chose** picture size, the number of compressed numbers per picture, how many
  pictures of history, and how many possible actions all come from settings.
  Trained files store the settings that produced them.
- **Why** required by the picture-size fork - see I1 - which is a live decision
  until the end of the compressor stage. Also required for the "one trained
  program runs at three different history lengths" requirement.
- **Wrong if** never. This is standard hygiene and cheap on day one, expensive on
  day thirty.
- **Fallback** none needed.

### H6. The dataset outlives the simulator

- **Chose** a short recorded clip of real pictures ships permanently alongside the
  trained program.
- **Why** traced during review, and it corrects a sloppy earlier claim. The
  playable demo runs with the simulator absent, but the prediction program needs
  about fifteen real pictures of context before it can predict anything, and those
  come from a real recorded episode. "Delete the simulator" is exact. "Delete the
  data" is not. The cold-start time requirement has to include loading that clip.
- **Wrong if** the clip is large enough to hurt the cold-start requirement.
  Fifteen small pictures is a few tens of kilobytes, so no.
- **Fallback** (1) Ship a single starting picture and repeat it fifteen times,
  accepting a worse first few predictions. (2) Ship a compressed clip rather than
  raw pictures, skipping a step at startup.

### H7. On the larger-picture path, use the framework's built-in attention routine

- **Chose** if we take the 144-number path, the plain implementation must be the
  framework's optimised built-in rather than the obvious hand-written version.
- **Why** the hand-written version materialises a large intermediate table, and at
  the longer sequence length that caps the training batch size at about 16 against
  our memory allowance. The built-in removes the cap entirely. Note this is a
  batch-size ceiling, not a pass/fail - nothing in the requirements specifies a
  batch size - but it is a real constraint discovered by computation, not guessed.
- **Wrong if** the built-in is unavailable or misbehaves on this hardware
  generation.
- **Fallback** (1) Accept the batch-size cap; batch 16 trains fine, just slower.
  (2) Accumulate gradients over several smaller batches to simulate a larger one.

---

## I. The picture-size fork

### I1. Measure both picture sizes in Week 2, rather than starting small and switching on a diagnosis

- **Chose** build and score the compressor at both 64-by-64 and the larger size
  in the same week. **Revised 2026-08-28 - this used to read "start at the
  smaller size; switch only on a specific diagnosis."**
- **Why** two computed facts, plus two measured ones that changed the decision.

  The computed facts, unchanged: **the compression ratio is provably identical at
  both sizes** - exactly 170.667 to 1, and the arithmetic cancels to a
  size-independent constant. So the larger size does not give the compressor an
  easier ratio. It helps only because the scene becomes oversampled relative to
  the size of the things in it: a block that occupies exactly one small tile at
  the smaller size spans a bit over two tiles at the larger, so each tile carries
  a simpler piece of an edge. And the larger size makes the naive speed **251% of
  the entire budget rather than 103%** (both figures recalculated against the
  measured memory bandwidth of 308 GB/s, not the 448 originally assumed), which
  promotes the several-numbers-at-once speed-up from optional to mandatory. That
  is real added scope.

  The measured facts, which are new. First, **the simplest possible compressor
  falls clearly short**: a fixed dictionary of the 512 most representative tiles
  scores 26.4 against a bar of 30, and 1,024 entries reach only 27.6. So this is
  not a formality that a week of training clears. Second, and decisively, **99.9%
  of that shortfall sits at the boundaries between objects** rather than in flat
  areas. Misplaced edges are precisely what the larger size fixes. So the branch
  this note used to treat as the unlikely one is the branch the evidence points
  at, and holding the larger size "in reserve" would mean discovering that late.

  Measuring both costs one settings file, about 45 seconds of regenerating
  pictures, and one extra training run of about six minutes. Against that, what
  it buys is that Stage 5's difficulty is known before Stage 3 starts rather than
  after.
- **Wrong if** the larger size turns out to help less than the edge-localisation
  evidence suggests, in which case we spent one training run and about 8 GB of
  disk finding that out. That is a far cheaper way to be wrong than the previous
  version of this decision allowed for.
- **Fallback** (1) If both sizes fall short, the lever is the compressor's shape -
  more capacity, and the mechanism that lets tiles describe each other - not the
  picture size and not the internal precision settings. The control run described
  in I2 is what tells these apart. (2) Take the larger size *and* accept 20
  pictures per second rather than 30, reporting the curve honestly instead of
  cutting a speed-up from the list. That remains the pre-agreed answer if the
  larger path cannot reach the frame rate.

### I2. If the vocabulary collapses, shrink it - never reward spreading it out

- **Chose** if the compressor uses too little of its 512-value vocabulary, walk
  the vocabulary down - 512, then 240, then 125, then 64 - and stop at the first
  size that passes. Do not add a term to the training that rewards using more of
  it. Added 2026-08-28.
- **Why** three things. **The risk is real and measured**: the best simple
  dictionary keeps only **150 of its 512 entries** in genuine use, which is direct
  evidence that this scene does not need 512 distinct values. **Shrinking is
  free**: the speed budget for the prediction program is set by *how many* numbers
  each picture becomes - 64, which never changes - not by how many distinct values
  each number may take. A smaller vocabulary actually makes the prediction program
  slightly smaller and its final step slightly cheaper. And **the alternative
  destroys the measurement**: this compressor was chosen over the older, more
  complicated family specifically because it has no learned dictionary and
  therefore cannot collapse the way those do. A training term that rewards
  spreading the vocabulary out would improve the number while hiding whether the
  underlying problem exists, which is the one outcome worth avoiding.
- **Wrong if** the check is measuring the wrong thing. It compares the vocabulary
  actually used against a perfectly even spread, and a perfectly even spread is
  not something this scene has any reason to produce - so a smaller vocabulary
  scores better on it than a large one does, which is a slightly perverse
  incentive. It is kept as a pass/fail bar anyway, because the failure it is
  really watching for - the compressor ignoring almost all of its vocabulary - is
  a genuine failure whichever way the bar is phrased.
- **Fallback** (1) Report the number of vocabulary entries in real use alongside
  the evenness figure, so a reader can see which of the two is doing the work.
  (2) If shrinking all the way to 64 still fails, that is evidence the compressor
  itself is too weak, and the answer moves to I1's first fallback.
- **Watch for** changing the vocabulary size quietly changes how strongly the
  compressor learns. The rounding step passes a gradient of 0.858 at 512 values,
  1.001 at 125 and 0.668 at 64 - a spread of about 1.5x. So every vocabulary
  comparison has to be run at two learning rates, or the result is partly a
  learning-rate result wearing a vocabulary label.

---

## J. Keeping the low-level code safe without paying for it

### J1. Memory-error checking is a separate build mode, not the default

- **Chose** the error-checking build exists from the very first low-level file,
  but as an alternative build rather than the standard one, because the real
  generation run needs full optimisation to hit its throughput target.
- **Why** the checking is required to run clean over a full generation run, so it
  cannot be an afterthought - but if it were the default, every run would pay for
  it.
- **Wrong if** developers forget the checking build exists and only discover
  problems at the gate.
- **Fallback** (1) Make the checking build the default during development and
  switch explicitly for the real run, which is the inverse and equally valid. (2)
  Run the checking build automatically on a nightly schedule.

### J2. Use a curated subset of the undefined-behaviour checks

- **Chose** the default set, deliberately not the extended one.
- **Why** published measurements: the full set of nineteen sub-checks costs up to
  **228%** on a standard benchmark suite. Adding the integer-overflow checks and
  similar is the path to that figure. The memory checker is separately about 73%
  to 103%. Both those figures are from processor-only benchmarks though, and our
  loop is dominated by drawing and picture-copying that the checker does not
  instrument - so the real end-to-end slowdown should land well below them.
  **Measure it; do not extrapolate.**
- **Wrong if** a real bug slips through that one of the excluded checks would
  have caught.
- **Fallback** (1) Add the specific excluded check that would have caught it, and
  only that one. (2) Run the full set once, overnight, as a one-off audit rather
  than a gate.

### J3. Collect all findings in one pass for the gate run; stop at the first fault during development

- **Chose** the gate run is configured to keep going and report everything; during
  development the inverse.
- **Why** the full generation run is long enough that finding one problem per run
  is unacceptable. During development, stopping at the fault is what you want.
- **Wrong if** an early error cascades into hundreds of spurious later ones,
  making the report unreadable.
- **Fallback** (1) Fix the first finding, rerun. (2) Cap the number of reports.

### J4. The simulator library itself is left unchecked

- **Chose** the prebuilt simulator library is not instrumented. Only our code is.
- **Why** it is not our code, and - importantly - the case that actually matters
  is *still caught*, because the memory checker intercepts allocation globally.
  If the simulator writes past the end of a buffer that **we** allocated, it trips
  a guard region and gets reported.
- **Wrong if** the bug is entirely internal to the simulator, which we could not
  fix anyway.
- **Fallback** (1) Report it upstream. (2) Build the simulator from source with
  instrumentation, as a one-off investigation only.

### J5. All low-level code is confined to the first stage

- **Chose** the low-level language appears only in the picture factory. Nothing
  later.
- **Why** the memory checker and the general-purpose graphics-card computation
  runtime coexist poorly, and the first stage is the only one with no
  graphics-card computation in it. Confining the low-level code to that stage is
  what makes the clean-checker requirement achievable at all.
- **Wrong if** a later stage needs low-level code for speed.
- **Fallback** (1) Write that piece as a custom graphics-card routine from the
  scripting side, which is the planned mechanism for the merge-computation-steps
  speed-up anyway. (2) Write low-level code without the checker for that piece
  specifically, and document the gap.

### J6. Never enable aggressive floating-point shortcuts

- **Chose** the compiler flag that permits reordering floating-point arithmetic
  stays off, permanently.
- **Why** it breaks byte-identical repeatability, which is a shipping requirement.
- **Wrong if** never.
- **Fallback** none. If we ever need the speed badly enough to give up
  repeatability, that is a requirements change, not an implementation choice.

---

## K. The scene itself

### K1. The two arm segments get different colours

- **Chose** each segment is its own colour.
- **Why** three separate measurements - does each segment keep its length, does
  the arm move in the commanded direction, is a segment fragmented - all become a
  few lines of pixel counting per segment. A shared colour would require
  splitting one blob into two segments, which is exactly the shape-grouping
  problem we rejected in F2.
- **Wrong if** the colour budget runs out. It will not; we are using about seven
  of twenty-four.
- **Fallback** (1) Colour only the outer segment distinctly and measure the inner
  one by subtraction. (2) Add a small marker geometry at the joint in a third
  colour.

### K2. The palette has exactly one home: the scene file

- **Chose** the colours are defined only as attributes in the scene file. The
  frame checker reads them from there with standard library tooling, about five
  lines.
- **Why** duplicating the colour list into the settings file is the same class of
  bug as having two frame-checker implementations - the copies drift, and the
  symptom is the checker reporting "block missing" for a block that is plainly
  present.
- **Wrong if** parsing the scene file at runtime becomes awkward.
- **Fallback** (1) Extract the palette from the scene file into the settings file
  as an automated build step, so it is still derived rather than duplicated. (2)
  Generate the scene file *from* the settings, inverting which one is the source
  of truth - a bigger change, but equally single-homed.

### K3. Every object gets a distinct saturated colour, and the total stays under 24

- **Chose** distinct, well-separated colours; current count about seven
  (background, table, two arm segments, three blocks) against a ceiling of 24.
- **Why** distinct colours are what let "how many blocks are present" be a
  per-colour pixel count instead of shape analysis, and it is why no additional
  numerical library is needed anywhere in the shipped design. The 24-colour
  ceiling is what protects the compressor's job.
- **Wrong if** we add enough objects to approach the ceiling. Every new
  distinctly-coloured object spends from a budget that is also protecting the
  compressor.
- **Fallback** (1) Reuse a colour for two objects that can never overlap, and
  disambiguate by position. (2) Raise the ceiling and accept a harder compression
  job, which is a requirements change.

### K4. Contact frequency and hiding frequency are tuned by moving objects, not by changing code

- **Chose** the two behavioural bars - the arm touches a block in more than 5% of
  pictures, and fully hides one in at least 3% - are hit by adjusting arm reach
  and block placement in the scene file.
- **Why** these are the two checks most likely to fail on the first attempt, and
  recognising in advance that they are *scene* problems rather than *code*
  problems is what keeps the fix to minutes instead of days.
- **Wrong if** no placement satisfies both at once - they pull in the same
  direction, so this is unlikely, but blocks too close to the arm could hurt the
  action-variety bar.
- **Fallback** (1) Bias the scripted-reach behaviour toward the blocks, raising
  both rates without moving anything. (2) Lengthen the arm segments, which raises
  both rates at the cost of a slightly different scene than originally planned.
