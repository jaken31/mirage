# Mirage: Timeline

**Audience.** Anyone who needs to know what happens when, in what order, and what
could go wrong - without reading the engineering docs. Deliberately written in
plain words. The reasoning behind each choice lives in
`docs/decision_notes.md`; the study plan it draws on is
`docs/world_model_learning_roadmap.md`.

---

## What the project is, in four sentences

We are building a program that can *imagine* a robot arm. First we set up a
physics simulation of a two-joint arm on a table pushing three coloured blocks
around, and we record 300,000 small pictures of it. Then we train a program to
predict the next picture given the last few pictures and whatever the operator
just pressed. When it works, you can unplug the physics simulation entirely,
press arrow keys, and the arm still moves - because the program learned how the
world behaves instead of being told.

The point is not that a program can imagine a robot arm. Other people have done
that. The point is doing it **fast enough to feel like a live video game** - at
least 30 new pictures every second, on one laptop graphics card. That speed
target is the whole thesis, and it is why the back half of the timeline is
almost entirely performance work.

---

## How to read this timeline

- **Weeks are relative, not calendar dates.** Week 1 begins the day the three
  Week 0 blockers clear. They may not clear in one week.
- **Every stage has a gate** - a specific, checkable thing that must be true
  before the next stage starts. No stage is "done" on a feeling.
- **There is exactly one real branch point in the plan.** It happens at the end
  of Week 2 and it decides how hard the back half is. Everything else is a
  smaller, local decision.
- **Study time is a separate lane, not slack.** Roughly 75 hours of learning is
  spread across the project, timed so each topic arrives about a week before it
  is needed. Two weeks are genuinely overloaded and are flagged where they occur.

---

## The shape of the whole thing

| Stage | Length | Done when | The thing that could go wrong |
|---|---|---|---|
| **0. Unblock the machine** | 2-4 days, possibly 2 weeks | The graphics card runs at full power on demand, pictures come out of the real graphics hardware rather than a slow software imitation, and the compiler toolchain works | Copying finished pictures off the graphics card may be far slower than assumed. This is the single riskiest unknown in the project |
| **1. Build the picture factory** | 5 days | 300,000 pictures on disk, the same settings produce byte-identical pictures twice, and the automatic frame checker never wrongly flags a good picture | The arm may not touch or hide the blocks often enough. Fixed by moving things in the scene file, not by changing code |
| **2. Build the compressor** | 1 week | A picture survives being squeezed to a short list of numbers and rebuilt, at a measured quality bar | **This is the branch point, and measurement has made it the likeliest stage to force a change.** The simplest version of this falls clearly short of the bar, and the reason it falls short is the exact reason we would switch to larger pictures - which makes the back half materially harder |
| **3. Build the predictor** | 2 weeks | It can imagine 200 pictures in a row without the scene falling apart | Output looks bad for a while before it looks good. Expected, not a signal to change plans |
| **4. Make it playable** | 3 days | Keyboard drives the arm with the physics simulation switched off, and we have written down the honest starting speed | Nothing much. This stage exists to produce the number the next stage optimises |
| **5. Make it fast** | 6 weeks | Five specific speed-ups measured one at a time, at least 3x faster than the starting point, and 30 pictures per second sustained | The hardest and longest stage. Progress often looks flat at first |
| **6. Slack** | 2 weeks | - | Reserved for the overruns above, not for extra features |

**Total: about 14 weeks.** Twelve of work, two of slack, plus however long Week 0
takes.

---

## Week 0 - Unblock the machine

**Why this stage exists at all.** The original plan assumed the project would run
inside the Linux compatibility layer that ships with Windows, using a
well-supported path for drawing pictures without a visible window. That
assumption was tested and **it failed**. The compatibility layer on this specific
machine cannot reach the graphics hardware at all - it silently falls back to
drawing pictures on the main processor instead, which is roughly fifty times too
slow. This was verified across two restarts and every available override; the
evidence is in `CLAUDE.md` and `bench/egl_probe.py`. It is settled and should not
be revisited.

So everything moves to running directly on Windows. That is a smaller change than
it sounds, except in one respect covered below.

**Work, in order:**

1. **Get the graphics card to run at full power.** It currently idles at a low
   power state and stays there even under heavy load - drawing about 6 watts when
   it should draw 60 to 115. Every speed measurement taken in that state is
   worthless, and one of our engineering commitments is that any measurement can
   be repeated within 5%. A card that drifts between power states varies by far
   more than 5%. Fix: mains power, the performance profile in the system
   settings, close the roughly fifty background programs holding onto the graphics
   card (browsers, chat apps, game overlays), then lock the clock speeds for the
   duration of a measurement.
2. **Confirm pictures come from real graphics hardware.** Create a drawing
   context, then read back the name of the device that will do the drawing and
   check it names the actual graphics card. Two specific names mean we got the
   slow software imitation instead, and the program should refuse to run when it
   sees them. This has to be a hard check, not a note in a document: the software
   fallback produces correct pictures, just fifty times too slowly, so without
   this check the failure is invisible until the throughput number comes in low
   for reasons nobody can explain.
3. **Measure how long it takes to copy one finished picture off the graphics
   card.** This is the highest-risk number in the entire plan and it is the
   reason Week 0 has a wide range. Details in the next section.
4. **Measure how long one step of physics takes**, separately from drawing.
   Needed to know how much of the time budget is left.
5. **Re-measure raw memory speed and raw arithmetic speed, at full power.**
   Every performance estimate later in the plan is built on an assumed memory
   speed that has never actually been confirmed on this machine.
6. **Get a compiler working.** The picture factory is written in a
   lower-level language for speed and for a memory-checking tool that only
   works well there. Which compiler works on this machine is currently unknown.
   It does not block the measurements above, which are done from the scripting
   language.

**Gate:** all five numbers written down, each recorded next to the power state of
the graphics card at the moment it was taken. A timing without its power state is
not a valid number.

### The one measurement that can reshape Week 0

We need to copy each finished picture out of the graphics card's memory so we can
save it to disk. A public discussion thread about the simulator reports this
operation taking about **30 milliseconds per picture** on the exact drawing path
we are now forced to use - and, strangely, the cost barely changes with picture
size, which means it is a fixed overhead per copy rather than a
data-transfer problem.

Our budget is **2 milliseconds per picture, total, for everything**. So:

| Measured result | What it means | What we do |
|---|---|---|
| Under 0.5 ms per copy | Comfortable. About 6x margin | Keep the straightforward two-pass approach. No extra work |
| 0.5 ms to a few ms | Two copies per picture eats the whole budget | Switch to the single-pass approach already designed and held in reserve. Roughly a day of work |
| Near 30 ms per copy | The standard drawing path itself is the problem, and no change to *what* we draw saves us | Hand-build a direct connection to the graphics driver, bypassing the standard path. **3 to 5 extra days, and this is why Week 0 might become two weeks** |

**Do not build the fallbacks in advance.** Each one has a specific measured
trigger. Building them speculatively is how a five-day stage becomes a
three-week one.

**Study alongside (about 15 hours, ideally done before Week 0):** work through the
from-scratch language-model video course listed first in
`docs/world_model_learning_roadmap.md`, typing the code out rather than watching.
The prediction program we build in Weeks 3-4 is that exact design with pictures
substituted for words. Skipping this makes every later decision guesswork.

**Study alongside (about 12 hours, spilling into Week 1):** the lower-level
language basics - only the chapters the roadmap names, not the whole book - plus
the three relevant sections of the simulator's documentation and the one sample
program that already does roughly what our picture factory does. Read that sample
before writing anything.

> **Load warning.** 27 hours of study plus a five-day unblocking stage does not
> fit in one week. Either do the 15-hour course before the project formally
> starts, or accept that Week 0 and Week 1 run together into about three weeks.

---

## Week 1 - Build the picture factory

**Goal:** 300,000 small pictures of the arm pushing blocks, on disk, with a
record of what was happening in each one.

**Work, in order.** This order is chosen so the riskiest piece is proven early:

1. **The scene.** A two-joint arm, three blocks, a fixed camera, deliberately
   ugly flat lighting - no shadows, no textures, no smooth shading. Ugly is a
   feature: it keeps the number of distinct colours in a picture under two dozen,
   which is what makes the pictures cheap to compress later. Each arm segment
   gets its own colour, which turns several later measurements into simple pixel
   counting rather than shape analysis.
2. **The settings file and naming scheme.** One settings file split into
   sections, and a naming scheme where every output file's name contains a
   fingerprint of the settings that produced it. This is a small piece of work
   with an outsized payoff, explained in `docs/decision_notes.md`.
3. **The drawing connection.** The riskiest file, done early. Creates the drawing
   context and refuses to continue if it is not talking to real graphics
   hardware.
4. **The behaviour generator.** Half the recorded episodes are random arm
   movements; half are sloppy scripted attempts to reach for a block. The mix is
   chosen per episode, not per picture - a scripted reach needs several
   consecutive steps to make any sense.
5. **The ground-truth recorder.** For each picture, note which action was
   commanded, where the blocks actually are, how many pixels of each block are
   visible, and whether the arm was touching anything. Forty-six bytes per
   picture, less than half a percent of the picture data itself.
6. **The file writer.** Picture data and the per-picture notes are written
   first; a small summary file is written last, and its existence is the signal
   that the batch is complete. A crashed run leaves a batch with no summary file
   and the reader skips it. No lock files, no cleanup logic.
7. **The reader.** Reads the files back and knows never to hand the training
   program a window of pictures that spans a reset between episodes.
8. **The automatic frame checker.** Looks at a picture and reports a fixed list of
   measurements - how many pixels of each colour, where they are, how spread out
   they are, how long each arm segment appears, what angle it is at. It reports
   *measurements*, never a verdict. "The picture is bad" is a rule written in the
   settings file, on top of those measurements.

**Gate - all of these must pass:**

| Check | Bar |
|---|---|
| Builds cleanly from scratch on a fresh machine | Documented, done once |
| Pictures come from real graphics hardware | Hard assertion in the program |
| Same settings, run twice, produce byte-identical pictures | Exact match |
| The memory-error detector reports nothing over a full run | Zero reports |
| Throughput | At least 500 pictures per second |
| Total dataset size | Under 20 GB |
| Distinct colours per picture | At most 24 |
| The nine possible actions are used roughly evenly | Roughly uniform |
| The arm touches a block | More than 5% of pictures |
| A block is completely hidden by the arm | At least 3% of pictures |
| The frame checker never wrongly flags a good picture | Zero false alarms |

**What will probably go wrong.** The last two behavioural bars - the arm touching
blocks often enough, and the arm hiding blocks often enough - are the most likely
to fail on the first attempt. Both are fixed by moving objects around in the
scene file, not by changing any code. **Budget one or two extra rounds of this.**
Hiding matters specifically because the most interesting question in the whole
project is whether the program remembers a block it can no longer see - and that
question is unmeasurable if the arm never hides anything.

**Study alongside (about 8 hours, running into Week 2):** how to turn a picture
into a short list of whole numbers. Read the older, more complicated approach
first even though we are building the newer, simpler one - the new one only makes
sense as "the thing that deletes the old one's problems."

---

## Week 2 - Build the compressor, and hit the branch point

**Goal:** a program that squeezes each 64-by-64 picture down to 64 small numbers
and rebuilds it, losing as little as possible. The compression is aggressive -
roughly 170 to 1.

**Why this matters more than it sounds.** The prediction program in Weeks 3-4
does not work on pictures. It works on these short number lists, and it has to
produce one number at a time, in sequence. So the *length* of that list sets the
speed budget for everything after. Sixty-four numbers per picture is comfortable.
The fallback is 144, and that is not comfortable.

**What we measured before writing any of it, and what it changed.** We built the
simplest thing that could possibly work and scored it: take the 512 most
representative 8-by-8 tiles from the real pictures, keep them as a fixed
dictionary, and replace every tile with its closest dictionary entry. That scores
**26.4 on the quality scale where the bar is 30**. Doubling the dictionary to
1,024 entries only reaches 27.6, so a bigger vocabulary is not the answer.

Put in pixels rather than a scale: **clearing the bar means getting all but about
17 of a picture's 4,096 pixels exactly right. The simple version gets about 38
wrong.** The week's job is to halve that.

The consequence is that a compressor which looks at each tile *in isolation*
cannot pass, and we know that before starting. The entire gap has to come from
the tiles being allowed to describe each other - each number carrying
information about its neighbours, and the rebuilding step reading a whole
neighbourhood rather than one tile at a time. That is a specific thing to build,
not a matter of training longer.

**And the failure has an address.** Of all the mistakes the simple version makes,
**99.9% sit at the boundaries between objects** - the edges of blocks and arm
links. Almost none are in the large flat areas. That matters because misplaced
edges are the one problem larger pictures fix, and the one problem that retuning
the compressor's internal settings does not. So the branch the plan had been
treating as an unlikely fallback is the branch the evidence points at.

**Work:** build the compressor, train it on the pictures from Week 1, measure how
closely rebuilt pictures match the originals on pictures it has never seen. Also
check that it actually uses its whole vocabulary rather than collapsing onto a
handful of favourites.

Four short training runs rather than one, each answering exactly one question.
Each takes about six minutes, so the whole set is an afternoon:

1. **The control run**, with the number-rounding switched off entirely. This tells
   us whether a shortfall is the compressor's shape or the rounding. **If the
   control run cannot clear the bar, no amount of vocabulary tuning will** - the
   problem is the compressor and nothing else. Run this one first.
2. The real thing, tiles treated independently. Directly comparable to the 26.4
   above, so it says what the rounding costs.
3. The real thing plus the mechanism that lets tiles describe each other. This
   is where the missing quality is supposed to come from.
4. Only if 3 falls short: more capacity.

**Then runs 2-4 again at the larger picture size.** This is a change from the
earlier plan, and the measurement above is why: rather than hold the larger size
in reserve and switch later on a diagnosis, we measure both this week. It costs
one settings file, about 45 seconds of regenerating pictures, and one extra
training run - and it replaces a guess about how hard the back half will be with
a number.

**Gate - and this is the branch point:**

| Result | Meaning | Consequence |
|---|---|---|
| Quality clears the bar at the small size | 64 numbers per picture works | The straightforward path. The starting speed in Week 5 will already be close to the 30-pictures-per-second target, and the remaining work is proving a 3x improvement on top. **Now the less likely of the two outcomes** |
| Quality falls short **because edges and small details are misplaced** | The pictures are too small relative to the things in them | Move to larger pictures and 144 numbers each. **This makes the back half materially harder**: the naive speed becomes about two and a half times the entire budget rather than just over it, and one of the optional speed-ups becomes mandatory. **This is the expected failure, not the surprising one** |
| Quality falls short **because colours drift or the overall layout is lost** | Bigger pictures will not help | Do not switch sizes; retune the compressor's internal settings instead. **Measurement has made this branch unlikely** - 99.9% of the error in the simple version is already at edges, so there is very little colour-and-layout error left for this branch to describe |

The compression ratio is provably identical at both picture sizes, so bigger
pictures do not give the compressor an easier job in general. They help with one
specific problem, and that problem is the one the evidence says we have.

**The vocabulary check is at genuine risk, and its fix is free.** The best simple
dictionary keeps only **150 of its 512 entries** in real use, which is direct
evidence the scene does not need 512 distinct values. If the trained compressor
does the same, the answer is to shrink the vocabulary - 512, then 240, then 125,
then 64 - and stop at the first size that passes. **That costs the project
nothing**, because the speed budget for Weeks 3-6 is set by *how many* numbers
each picture becomes, which never changes, not by how many distinct values each
number may take. Shrinking actually makes the prediction program slightly
smaller. What we must *not* do is add a term to the training that rewards
spreading the vocabulary out: that would hide the very collapse the check exists
to detect.

**Also this week, and cheap:** re-run the frame checker's calibration against the
compressor's rebuilt pictures, not just the original clean ones. The clean ones
are perfect; the rebuilt ones have real artefacts. A checker calibrated only on
perfect pictures will cry wolf constantly once it sees imperfect ones - and
imperfect is all it will ever see from Week 3 onward.

**Study alongside (about 8 hours, background reading through Weeks 2-3):** four
or five papers on programs that imagine worlds, listed in the roadmap. One of
them is essentially our architecture. One well-known family of approaches on that
list should be read for vocabulary and deliberately **not** built - it solves a
different problem, and knowing that clearly is worth the reading time.

---

## Weeks 3 and 4 - Build the predictor

**Goal:** given the last fifteen pictures (as number lists) and one action, the
program produces the next picture, one number at a time, and keeps doing it
without the scene disintegrating.

**Work:** train the prediction program - about 15 million adjustable values,
small by current standards and deliberately so. Then measure four things by
watching it imagine long sequences:

| Question | Bar |
|---|---|
| How many pictures can it imagine before the scene falls apart? | At least 200, ideally 500 |
| Does the arm move in the direction the operator commanded? | At least 90% of the time |
| Do the arm segments keep their length, or does it hallucinate the arm's shape? | Length drift under 10% |
| Does a block reappear in the right place after the arm stops hiding it? | At least 80% of the time - **and this one is optional** |

**The last one is the interesting result and the one most likely to fail.** It is
deliberately not a shipping requirement. If it does not happen, we measure it,
report the negative result honestly, and ship anyway. A 15-million-value program
may simply be too small for it.

**Expect this to look bad for a while.** Early on, imagined sequences drift, the
arm smears, blocks dissolve. That is the normal shape of this work, not a signal
that the plan is wrong.

**Study alongside (about 8 hours, background):** the first two items of the
graphics-card programming track - an interactive puzzle set that needs no
graphics card to start, and one long, excellent written walkthrough of how
graphics-card memory actually behaves. This is pulled forward *deliberately*.
Weeks 6-11 are a difficulty cliff and this is how we make it smaller. Doing this
reading during Weeks 6-11 instead is the single most likely way for that stage to
overrun.

---

## Week 5 - Make it playable

**Goal:** press an arrow key, watch the arm move, with the physics simulation not
running at all. Three days.

This stage is short and it is not really about the demo. It exists to produce one
artefact: **an honest baseline measurement.** How many pictures per second do we
get with no optimisation whatsoever, and - critically - how is the time split
between the main processor preparing work and the graphics card doing work?

**We have a prediction for that split, and it is a test of the measurement rather
than of the program.** On the 64-number path we expect roughly 80% of the time
spent on the main processor just issuing instructions, and only 20% on the
graphics card actually computing. If the measurement says something very
different, suspect the measurement before suspecting the program.

That 80/20 split is the entire justification for the next six weeks. The graphics
card is barely working; it is waiting around for instructions. Nearly all the
available speed comes from reducing the waiting, not from making the computation
faster.

**One structural consequence worth flagging:** we said we would delete the
physics simulation after Week 1. That is true. But the prediction program needs
about fifteen real pictures before it can predict anything, so we must keep a
short recorded clip forever. "Delete the simulator" is exact. "Delete the data"
is not.

**Gate:** keyboard control works with the simulator absent, and the starting speed
plus the processor/graphics-card split are written down.

**Study alongside (about 4 hours):** how to read a timing trace - specifically,
how to look at what the main processor was doing next to what the graphics card
was doing, and see the gaps. That one skill is the whole point. Everything else in
the profiling tools is a distraction until there is a specific question.

---

## Weeks 6 to 11 - Make it fast

**Goal:** at least 3x faster than the Week 5 baseline, 30 pictures per second
sustained, and the slowest 1% of pictures no slower than 40 milliseconds.

**Five specific improvements, applied and measured one at a time.** Each is behind
its own switch so we can report the effect of each in isolation - a single command
produces the table.

| # | Improvement, in plain terms | Roughly |
|---|---|---|
| 1 | Stop recomputing work already done for earlier pictures. Remember it instead | 1 week |
| 2 | **Pre-record the whole batch of instructions once and replay it**, instead of issuing thousands of instructions per picture. Given the 80/20 split, this is the headline win | 1-2 weeks |
| 3 | Store the program's values in a smaller number format, so less data moves | 1 week |
| 4 | Merge many small computation steps into one big custom one | 1-2 weeks |
| 5 | Produce several of a picture's numbers at once instead of strictly one at a time | 1 week, **or mandatory and first if we took the 144-number path** |

**Ordering rule, and it is not negotiable: profile before changing anything.**
Guessing at optimisations is how weeks disappear. Also expect the numbers to look
flat at first - the first improvement often shows almost nothing until the second
one lands, because they attack the same bottleneck from different sides.

**The measurement discipline this stage depends on.** Three rules, all learned
from things that have already burned us:

1. **Nothing whatsoever is recorded inside the timed section.** Two of our
   requirements are about the *slowest* pictures, not the average. A single
   one-millisecond hiccup from writing a log line is 2.5% of the frame budget and
   lands squarely in the slowest 1%. Timings go into a pre-allocated block of
   memory and get written out afterwards.
2. **Turn off the monitoring dashboard's background sampling during measurements.**
   It quietly polls the graphics card on its own schedule, which inflates exactly
   the number we are trying to measure.
3. **Refuse to run a measurement when the graphics card is not at full power.**
   This is the lesson from Week 0. A card that boosts mid-run can blow the
   slowest-1% requirement all by itself, and it would look exactly like a
   programming problem.

**A sanity check on ambition.** The closest comparable public system runs a
program twenty times larger than ours and takes about the same amount of time per
number as we are targeting. That is not a coincidence and it is not discouraging -
it is the thesis. Both systems are limited by fixed overhead rather than by
computation. Which means: a small program is *not automatically fast*, and the
interesting engineering is removing the overhead. That is what these five
improvements do.

**Study alongside (about 8 hours, ideally just before this stage):** one small,
excellent public codebase does almost exactly this ladder of improvements in
under a thousand lines. Read the accompanying write-up first, then the code - it
explains one trap we would otherwise walk straight into. Then write our own
version rather than copying it; copying it skips the part that teaches anything.

**Study alongside (about 12 hours, during):** the remaining graphics-card
programming material - selected lectures rather than all of them, plus a second
puzzle set that builds up to the exact kind of merged computation step that
improvement #4 needs.

---

## Weeks 12 and 13 - Slack

Reserved for overruns from the stages above, in this order of likelihood:

1. Week 0 becoming two weeks because the picture-copy measurement came in badly
2. Week 1 needing extra rounds of scene adjustment to hit the contact and hiding
   bars
3. Weeks 6-11 running long, which is the historical norm for performance work

**Not for new features.** There is a list of tempting additions - a third arm
segment, blocks that collide with each other, a moving camera, a camera mounted
on the arm itself. Every one of them is deferred until the shipping criteria are
met, and the moving camera in particular makes the hardest problem in the project
substantially harder.

---

## The study lane, all in one place

Total honest estimate: **65 to 85 hours**, on top of the build rather than instead
of it. Full details and links in `docs/world_model_learning_roadmap.md` - this is
only the schedule.

| When | Topic | Hours | Why then |
|---|---|---|---|
| Before Week 0 | Build a small language program from scratch, following along in code | 15 | Non-negotiable. Weeks 3-4 build exactly this with pictures instead of words |
| Week 0-1 | Lower-level language basics, the simulator's docs, one sample program | 12 | Needed to write the picture factory at all |
| Week 1-2 | How to turn a picture into a short list of numbers | 8 | Needed for Week 2 |
| Week 2-3 | Papers on programs that imagine worlds | 8 | Background. One of them is our architecture |
| Weeks 3-4 | Graphics-card programming, first two items | 8 | **Pulled forward on purpose** to shrink the Weeks 6-11 cliff |
| Week 5 | Reading timing traces | 4 | Needed the same week to interpret the baseline |
| Week 5-6 | The reference implementation of our speed-up ladder | 8 | Needed before Weeks 6-11, not during |
| Weeks 6-11 | Graphics-card programming, remaining items | 12 | Directly feeds improvements #4 and #5 |

**Deliberately not studied.** There is an explicit skip list in the roadmap and
it is as important as the reading list. Highlights: one entire well-known family
of world-imagining approaches (right field, wrong branch - it learns behaviour,
we build a simulator); the currently fashionable image-generation approach (it
would predict better than ours and be far too slow, which is the argument our
whole project makes); text-handling techniques (our inputs are pictures); and
multi-machine training (we have one graphics card).

---

## The three moments that can change the plan

Everything else is execution. These three are genuine forks:

| When | The question | If it goes badly |
|---|---|---|
| **Week 0** | How long does it take to copy one finished picture off the graphics card? | Below half a millisecond: proceed. Above: switch to the cheaper drawing approach. Near 30 milliseconds: hand-build a direct connection to the graphics driver, and add 3-5 days |
| **End of Week 2** | Is the rebuilt-picture quality good enough at the small size? | Switch to larger pictures, 144 numbers per picture, and the back half gets harder: one optional speed-up becomes mandatory and the naive starting speed goes from "just over budget" to "two and a half times budget". **Measurement has moved this from an unlikely fork to the expected one** - 99.9% of the error in the simplest version of the compressor is the exact kind larger pictures fix. Both sizes are measured in Week 2 rather than one being held in reserve, so this fork is answered with a number instead of a diagnosis |
| **Week 5** | Is the time really split 80/20 between the main processor and the graphics card? | A very different split means the measurement is wrong, not the program. Fix the measurement before optimising anything, or six weeks get spent on the wrong bottleneck |

---

## Risk register, plain

| Risk | Likelihood | Impact | What we do about it |
|---|---|---|---|
| Copying pictures off the graphics card is fixed-cost slow | Medium-high. Public reports say it is | Week 0 becomes two weeks | Measure on day one, in isolation. Three pre-decided responses, each with a numeric trigger |
| The graphics card refuses to hold full power under sustained load | Medium. It is a laptop card | Every performance margin in the plan compresses at once | Discovered in Week 0 rather than Week 10, which is the entire reason Week 0 exists |
| The arm does not touch or hide blocks often enough | High on the first attempt | One or two extra rounds in Week 1 | Fixed by editing the scene, not the code. Already budgeted |
| Rebuilt-picture quality falls short at the small size | **Medium, lowered from medium-high on 2026-08-28.** The simplest version of the compressor scores 29.0 against a bar of 30. The earlier 26.4 came from a sloppier version of that same simple method, not from the pictures - re-run properly it does much better, so the gap one specific mechanism has to earn is about a quarter of what was budgeted | Larger pictures, harder back half | Both sizes are measured in Week 2, so the switch is a measurement rather than a diagnosis. A control run with rounding switched off, taken first, separates "the compressor is too weak" from "the rounding costs too much" |
| The compressor collapses onto a handful of its vocabulary | **Unknown, downgraded from medium on 2026-08-28.** The evidence for it - a simple dictionary using only 150 of its 512 entries - turned out to be an artifact of how that dictionary was built. Built properly it uses all 512, so there is now no measurement pointing either way | None, if handled correctly | Shrink the vocabulary until the check passes - 512, 240, 125, 64. Costs nothing, because the speed budget depends on how many numbers per picture, not how many values each may take. Never add a training term that rewards spreading out; that hides the collapse instead of fixing it |
| The program never learns to remember hidden blocks | Medium-high | The most interesting result is a negative one | Explicitly optional. Measure it, report it either way, ship without it |
| Performance work runs long | High. It always does | Eats the slack | Two weeks of slack, plus the graphics-card study pulled forward into Weeks 3-4 |
| Bit-identical repeats stop being bit-identical after a driver update | Low, and expected | A caveat rather than a failure | Documented up front: exact repeatability holds for a fixed driver and a fixed build |
