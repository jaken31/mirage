# Phase 0: Technical Debt Checklist

Derived from `docs/phase0_report.md` plus a fresh audit of the tree at `af1d6c1`.
Work happens on branch `worktree-phase0-debt`.

**Ordering principle: cost of fixing it later, divided by cost now.** Everything
in Tier 1 is baked into 300,000 frames on disk - regenerating today costs 44
seconds, and regenerating after Phase 1 and Phase 2 invalidates `data_hash`, every
hash below it, every checkpoint and every eval row. Everything else is flat cost:
it will cost the same next month. Only Tier 1 has a deadline.

Each item carries its evidence, because a debt claim without one is an opinion -
`CLAUDE.md`, "Every requirement claim carries its evidence".

---

## Tier 1 - Baked into the dataset. Only these get more expensive.

### [ ] D1. `action_hold_steps = 20` is an unmeasured guess, and Q-4 depends on it

**Evidence.** Architecture doc verification log, row `sim.action_hold_steps = 20`:
marked **"unverified - a guess"**. Estimated from the compiled model's
`dof_damping = 0.5` and `dof_armature = 0.01` as `inertia / damping` ~ 15 steps,
then rounded up to 20. Never measured.

**Why it is debt.** Q-4 requires >= 90% action-following, measured as
`sign(theta_t+1 - theta_t)` against the commanded sign. For roughly one settling
time after every sign flip the joint is still moving the *old* way, so the
measurement disagrees with the command through no fault of the model. Too short a
hold and **Q-4 is unreachable by any model**, which is a dataset defect that
presents as a modelling failure.

**Done when.**
1. Measure the settling time directly: drive one joint at constant `+1`, log
   `qvel`, take the step count at 63% of terminal velocity. Call it `tau_s`.
2. Sweep `action_hold_steps` over roughly `{tau_s, 1.5 tau_s, 2 tau_s, 3 tau_s}`,
   logging commanded sign against `sign(delta theta)` agreement.
3. Pick the point where agreement crosses 90% **with margin**, the same way F-5's
   2.5 was picked at the knee of a measured frontier rather than chosen.
4. Record it in the verification log, replacing the "a guess" row.

**Cost.** Now: one sweep plus a 44-second regeneration. Later: `data_hash`
changes, so every tokenizer checkpoint, dynamics checkpoint, token cache, eval row
and bench row computed from the current shards becomes unreachable by name.

---

### [ ] D2. 58.7% of ctx=15 training windows contain no action change at all

**Evidence - measured this session** against `data/shards/shard_000.meta`:

| | |
|---|---|
| Windows examined | 23,440 (ctx=15, 40 episodes) |
| Contain >= 1 action change | 9,674 = **41.3%** |
| Constant-action | 13,766 = **58.7%** |
| Observed hold length | mean 20.69, median 20, max 40 |
| Distinct actions used per episode | min **2**, median 7, max 9 |

The arithmetic predicts better than this: 14 change-opportunities per window
against a boundary every ~20.7 steps gives ~68%. The per-episode fraction is
**strongly bimodal** - 21 episodes at mean 63.6% (matching the arithmetic) and 51
at mean 29.4%. The effective action-change interval is ~34 steps, not the
configured 20.

**Mechanism.** The scripted reach is state-deterministic: it recomputes
`sign(gain)` on every re-draw and usually gets the same corner action back, so
consecutive holds repeat and the boundary produces no change. The high cluster is
the random half, where draws are uniform over 9 and the arithmetic holds. One
episode used **2 distinct actions across 600 steps**.

**Why it is debt.** An action-conditioned model learns what an action *does*
largely from windows where the action changes. Nearly three in five windows carry
no such event. If F-11 (beat the marginal-frequency baseline by 3x) or Q-4
underperforms, this is a leading candidate cause - and by then it is a dataset
problem found after two phases of training on it.

**Honest counterweight.** This is not obviously fatal. The action token is present
in every frame, the model still observes dynamics, and 41.3% of 300k windows is
still ~124,000 windows containing a transition. **The point is that this should be
a known number before Phase 2, not a discovery after it.**

**Done when** - one of:
- **(a) Accept it.** Record the measurement in the architecture doc next to `ctx`,
  with the trigger "if F-11 or Q-4 misses, revisit this first." Zero code.
- **(b) Fix it in D1's sweep.** Add constant-window fraction as a second reported
  quantity alongside action-following agreement, and pick a hold that satisfies
  both. Note `action_hold_steps < ctx` makes every window contain a boundary by
  construction.
- **(c) Fix the mechanism.** Have the scripted half re-draw a *different* action on
  repeat. Cheapest in code, but it perturbs F-5's measured histogram, so F-5 must
  be re-run at its own 2,000-episode sample size afterward.

**Do this with D1, not separately** - both are settled by the same sweep and the
same regeneration.

---

### [ ] D3. The meta record cannot say which policy half produced an episode

**Evidence.** The 46-byte record holds `action`, `qpos`, `block_xy`, `visible_px`,
`contact_mask`, `episode_id`, `step_idx`. There is no scripted-vs-random flag.
D2's split above had to be inferred from a bimodal histogram with a hand-chosen
0.55 cutoff, which is why that section reports clusters rather than a clean 50/50.

**Why it is debt.** Any per-half question is guesswork from the dataset alone:
does F-6's 20.69% contact come mostly from the scripted half? Does Q-4 fail on
random episodes specifically? The 50/50 coin is the single biggest structural
choice in the policy and the dataset cannot report on it.

**Honest counterweight.** The architecture doc's own rule is "a field with no named
consumer does not ship." This now has one - D2's analysis - but it is a
**diagnostic, not a requirement**, and it costs a byte (46 -> 47) plus a
regeneration.

**Done when.** Only if D1 forces a regeneration anyway: add `is_scripted` as a
`u8` (or a bit in `contact_mask`'s spare high bits, which costs zero bytes),
update `meta_dtype`, update the record layout in the architecture doc's shard
table. **If D1 concludes 20 was fine and no regeneration happens, skip this
entirely.**

---

## Tier 2 - Must-tier requirements with no evidence

These are ship criteria. The project's own rule is that a requirement claim
without a measurement is written as unverified, not as a requirement - and two of
them currently have no artifact at all.

### [ ] D4. E-2 (clean build from scratch) has never been verified

**Evidence.** `README.md:48` - "## Build / Not yet implemented. E-2 requires a
verified clean-build-from-scratch recipe here." E-2 is **Must**, and the
requirements doc names the README as its home: "Documented in README, verified
once."

**Why it is debt.** Every ingredient is known and scattered - MSVC via CMake
generator `Visual Studio 18 2026`, C++20, `sim/build/` and `sim/build-asan/`, the
ASan runtime DLL copied beside the binary, `/Zi` plus linker `/DEBUG` mandatory
because `C5072` is fatal under `/WX`. Nobody has run the sequence from an empty
directory. The failure mode is a recipe that works only on a machine that already
built it once.

**Done when.** Clone to a fresh directory, run the documented commands, both build
types compile and the binary runs. Paste the **exact** commands into README's
Build section. One entry in `runs.jsonl` (D5).

---

### [x] D5. E-5 (append-only run log) has no file - `runs.jsonl` does not exist

**Evidence.** `test -f runs.jsonl` returns missing. The architecture doc's repo
layout lists it at root, and `.gitignore` goes out of its way to protect it:
"NOT ignored: runs.jsonl. It is the E-5 append-only lab notebook, hand-authored,
and belongs in version control. Do not add a `*.jsonl` rule here." The intent was
committed; the artifact never was.

**Why it is debt.** E-5 is **Must**, one entry per run. Every Phase 0 number
currently lives only as prose in the architecture doc's verification log. Phase 1
produces PSNR sweeps, codebook entropy and reconstruction runs, and there is no
notebook to put them in - so they will land as more prose, and the log's value
falls with every phase it does not cover.

**Done when.** `runs.jsonl` exists at root and is backfilled with the Phase 0 gate
run and the four day-1 probes - one line each: config hash, change, number,
conclusion.

**Do not conflate with `mirage/logging.py`.** That is the Phase 1 `log(dict)`
helper (jsonl always, W&B behind a flag) and is machinery. E-5 is hand-authored by
a person. Build `logging.py` when Phase 1 needs it, not now.

**LANDED.** `runs.jsonl` exists with **18 entries**, chronological from 2026-08-21
to 2026-08-27, covering 19 requirement IDs. Schema: `date`, `run`, `hash`,
`requirement`, `change`, `number`, `conclusion` - the middle four are E-5's, and
`date` and `requirement` were added for chronology and ship-criteria tracing.
`number` is an object, not prose, so F-17's jsonl-to-markdown script can read it.
`hash` is `null` on the eleven runs that read no config: the `bench/` probes do not
take one, and the 6-episode shard-writer run's hash was never recorded.

**Backfilled beyond this item's stated scope**, which named the gate run plus the
four day-1 probes - five entries. E-5 says *one entry per run*, and a notebook
showing 5 of 18 known runs misrepresents the record. The boundary drawn instead:
**every verification-log row that carries both a date and a number.** Excluded on
purpose - rows that are arithmetic rather than runs (the budget recomputation, the
closed-form compression ratio), rows sourced from docs rather than execution (W&B
behaviour, published ASan overheads), and `action_hold_steps = 20`, which has no
run because nobody measured it. That one is D1.

The architecture doc's observability table now names this schema, so the two do not
drift.

---

## Tier 3 - Documentation that now actively misleads

### [x] D6. README tells the reader to gate benchmarks on `pstate == P0`. That was refuted

**Evidence.** `README.md:38-43` - "**Before taking any timing measurement**,
confirm the GPU is at P0." Against the architecture doc and `CLAUDE.md`: "**Do not
gate benchmarks on `pstate == P0`** - that rule was refuted 2026-08-23." The
pstate follows the *memory* clock domain, so a correct compute-bound run reads P4
while the SMs sit at 86% of max drawing 99 W of a 100 W cap.

**Why it is debt.** This is the worst class of doc rot: not merely stale, but
instructing the wrong action. Anyone following the README rejects every valid
compute number the machine can produce.

**Done when.** Replaced with the two-domain rule - compute gated on **SM clock +
power draw**, bandwidth on **memory clock == max** - pointing at
`bench/gpu_probe.py`, which already does both.

**LANDED.** README now carries a two-row gate table, the thermal-counter rule, the
Phase 3/4 preconditions, and the F-4 determinism caveat - the last two because the
architecture doc asserts the README states them and it did not.

**Scope grew, and it had to.** The same refuted rule survived in the *authoritative*
doc: `world_model_architecture.md` item 3 of the benchmark-validity section still
instructed the bench harness to "refuse to run when `pstate != P0`", contradicting
its own verification log two hundred lines below. Fixing only the derived README
would have left the source wrong, which is the exact failure D8 describes. Both are
now corrected, in the doc's own `**Corrected <date>:**` convention. A stale
"MSVC or MinGW - is unverified" line in the same section went with it.

---

### [x] D7. README's environment table calls three verified things "unverified"

**Evidence.**

| Line | Says | Actually |
|---|---|---|
| `README.md:30` | GL backend "unverified - `mjr_readPixels` per-call latency is the day-1 blocker" | verified 2026-08-26, `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2`, and readback measured at 25.4 us |
| `README.md:31` | MuJoCo "wheel resolves for this Python; not yet exercised" | 300,000 frames generated through it |
| `README.md:32` | Compiler "unverified - MSVC or MinGW, remaining Phase 0 prerequisite" | MSVC, C++20 confirmed by the binary printing `202002` |

**Done when.** Table reflects the verification log. **Bundle with D6 and D11 in a
single README pass** - they are the same file and the same sitting.

**LANDED.** All three rows now carry their evidence. MuJoCo pinned at the installed
**3.12.0** rather than the vaguer "3.x, exercised".

---

### [x] D8. The verification log and the doc body drift, in both directions

**Investigated 2026-08-28, after D6 turned up one instance by accident.** The
sweep method: every verification-log row that reports a refutation or a
correction names a claim the rest of the tree may still assert. Grep the tree for
each refuted spelling.

**Scale: 19 of the log's 43 rows (44%) carry a refutation or correction.**

#### Type A - the body still asserts what the log refuted

Twelve sites across five files.

| Site | Asserts | Refuted by |
|---|---|---|
| `world_model_architecture.md:180` | "disable visualization decorations in `mjvOption`" | log: `mjv_defaultOption` already leaves every decoration off, and zeroing the flag array clears `mjVIS_STATIC` so worldbody geoms stop drawing entirely |
| `phase0_structural_plan.md:267` | "Turn them off in `mjvOption`" | same |
| `phase0_structural_plan.md:270` | "Prebuilt MuJoCo plus the leak checker ... expect a small suppression file" | log: MSVC has **no leak detection at all**, so the situation cannot arise |
| `phase0_structural_plan.md:139` | "Read the `record.cc` sample before writing" | log: not in the pip wheel - `mujoco` 3.12.0 ships headers and two test XMLs |
| `phase0_structural_plan.md:264` | "`record.cc` handles it" | same |
| `phase0_structural_plan.md:286` | `mj_step` timing is "**Next**" | measured 2026-08-23 |
| `phase0_structural_plan.md:289` | "the GPU pstate blocker" | pstate gate refuted 2026-08-23 |
| `world_model_ingredients.md:80` | "C++ harness: **EGL context**, step loop" | EGL is out; the backend is GLFW |
| `world_model_ingredients.md:128` | Phase 0 gate is "**EGL verified**" | the gate was hardware-render verified under GLFW |
| `world_model_learning_roadmap.md:62` | "the **EGL context**. This is where your C++ harness lives" | same |
| `world_model_learning_roadmap.md:151` | "where the **EGL** and offscreen-rendering answers actually live" | low severity - points a reader at the wrong search term |
| `.gitattributes:6` | "Work happens on Windows and **builds in WSL2**" | WSL2 is out. The LF conclusion survives; its stated reason does not |

#### Type B - the log holds findings the body never absorbed

`phase0_structural_plan.md` has a **"Gotchas, and how you would notice"** table -
the designated sink for exactly this. It holds **8 rows, every one of them a
pre-Phase-0 prediction read out of documentation**. It holds **zero** of the
gotchas Phase 0 discovered by running code:

| Discovered gotcha | Why it belongs in a gotchas table |
|---|---|
| `mjv_updateScene` before any `mj_forward`/`mj_step` | Renders an entirely black frame while `scene.ngeom` reads a correct 6 |
| `<camera mode="targetbody">` without `target=` | Compiles clean, `cam_targetbodyid = -1`, aims nowhere, no error |
| The segmentation pass leaves the framebuffer holding id colours | Reverse the order and the shard stores id colours; nothing downstream fails |
| Slicing an `np.memmap` is lazy and touches no pages | A probe reported 3.4M fps having timed slice arithmetic |
| `jnt_qposadr` and `jnt_dofadr` diverge for free joints | Computing either silently indexes the wrong element |
| `rgba * 255` does not land exactly, and not by a modellable rule | Byte-rounded equality calls 4 of 7 palette entries missing on a flawless frame |

**And 3 of the table's 8 existing rows are themselves now stale** - the two
`mjvOption` and `record.cc` rows plus the leak-checker row above. So the table is
**0 for 6 on current findings and 3 for 8 stale on old ones.**

This is time-sensitive. AGENDA names the Phase 0 structural plan as the standing
template - "same shape as the Phase 0 one" - and Phase 1's plan is being drafted
from it now.

#### Why it drifted - the mechanism, and it is not carelessness

**Corrections propagate by string match, not by meaning.** The evidence is that
drift is not uniform *within a single file*:

- **`castshadow`** is a distinctive, greppable token. Fixed in **all six** places
  it appears. Nothing survives.
- **`record.cc`** was corrected inline in `world_model_learning_roadmap.md:63`,
  with a "checked:" note - and **not** in its sibling `phase0_structural_plan.md`,
  where the same claim is phrased differently.
- **The `mjvOption` row is titled "by zeroing the flag array".** The body never
  says "zeroing"; it says "disable". Grepping the refutation's own key term does
  not find the assertion it refutes.
- **`EGL`**: the architecture doc added a blanket escape hatch - "read every 'EGL'
  below as 'the offscreen GL context'" - which **licensed leaving the occurrences
  in place**. That instruction scopes to one file. The ingredients and roadmap
  docs kept theirs with no such license.
- `world_model_ingredients.md` had its header (448 GB/s) and three table rows
  corrected while rows 80 and 128 were missed. Same file, same session. "Nobody
  updated this doc" does not explain that.

The predictor of survival is **whether the refuted claim is phrased in the same
words as the refutation.**

#### LANDED 2026-08-28 - items 2 and 3 below

**Ten of the twelve Type A sites corrected**, plus the gotchas table.
`world_model_architecture.md:180` now says the decorations are *already* off and
warns against zeroing the array; `.gitattributes` keeps the LF rule and drops the
WSL2 reason; the four EGL sites name GLFW; the two stale status markers in the
structural plan are current. Verified by re-running the same sweep - the only
surviving `EGL` hits are the architecture doc's own historical framing at `:8` and
`:639`, both immediately adjacent to their corrections.

**One of the twelve was a false positive.** `phase0_structural_plan.md:139` already
read "It lives in the MuJoCo GitHub repo under `sample/`, not in the pip wheel" -
the grep matched the token `record.cc`, not a stale claim. Its sibling at `:264`
was left as-is for the same reason: `record.cc handles it` is accurate once `:139`
has told the reader where the file lives. **So the real count was ten, not twelve** -
a keyword sweep over-reports, which is itself an argument for item 1 below.

**The gotchas table is now two tables.** The eight predicted-from-docs rows stay,
with the three stale ones marked **Corrected** in place rather than deleted, since
the wrong version is the useful part. Six run-discovered rows added, and the
section says the thing that matters about them: **five of the six fail silently** -
the run completes, the numbers look plausible, nothing reports an error.

**Item 1 landed too - D8 is closed.** The verification log now carries a fourth
column, **Asserted at**, backfilled across all 43 rows: the 19 refutation rows name
their sites, the other 24 read `-`. Sites are cited as `file, "Section name"`,
never as line numbers, matching the convention `sim/policy.h` and `sim/truth.cpp`
already use - a line reference rots exactly the way the claims did.

The column's rule sits above the table, because a column with no stated convention
is a column nobody fills: a refutation row lists every site asserting the refuted
claim and marks each corrected or standing; it is filled **when the row is
written**, since filling it later means grepping this row's wording against an
assertion phrased differently, which is precisely how the `mjvOption` and
`record.cc` sites survived; and **an empty sites column on a real finding is the
signal to give that finding a home** rather than a clean bill.

Two things the backfill exposed that the prose audit had not:

- **The `pstate == P0` row is the worst case in the table.** Three sites asserted
  the refuted gate for five days *after* the row refuting it was written, and one
  of them was an instruction to the reader.
- **A pre-existing markdown bug.** The F-9 row's result cell contains a literal
  `max |diff| 0`, so that row had been rendering as six columns, splitting its own
  result text. Escaped. Found only because adding a column forced a structural
  check of the table - a schema change is a free excuse to validate the rows.

#### Done when - and the fix is not a sync tool

1. **Add an "asserted at" column to the verification log.** The log records what
   was wrong and what came back; it never records **where the wrong thing is
   asserted**. Filling that in when the row is written makes propagation
   mechanical instead of dependent on a later reader guessing the right synonym.
   This is the change that would have prevented all twelve Type A sites.
2. **Route run-discovered gotchas into the gotchas table**, and fix its three
   stale rows. One line in the standing practice note: a phase's structural plan
   is updated at phase close, not only written at phase start.
3. ~~Correct the twelve Type A sites.~~ **DONE** - ten real ones; see above.

**Reprioritised from "lowest priority item on this list".** It was written as doc
hygiene on the assumption of one instance. Twelve Type A sites and a gotchas table
that is 0-for-6 on real findings is a different item, and item 2 above decays as
Phase 1's plan gets written from the stale template.

---

## Tier 4 - Structural, cheap now, annoying later

### [ ] D9. `tau = 8.0` is hardcoded in `validator.py`, not in config

**Evidence.** `mirage/validator.py:201`, `:248`, `:285` all carry
`tau: float = 8.0` as a default parameter. The `validator` section of
`base.json` holds only `contact_rate_min` and `occlusion_rate_min`.

**Why it is debt.** It contradicts the design statement the validator was built
from - "the validator is a feature extractor, not a predicate: ... 'the validator
failed' is a threshold expression over that vector, **defined in config rather
than in code**." `validator_hash` exists precisely so that changing a threshold
produces a new eval row against the same checkpoint. **With tau in code, changing
tau does not move `validator_hash`**, and two Q-3 coherence-horizon rows that are
not comparable will claim they are.

**Done when.** `validator.offpalette_tau` exists in `base.json`, is read through
`Config`, and the defaults are gone from the three signatures.

**This is about where the number lives, not what it is.** The *value* stays
uncalibrated until the Phase 1 recalibration - see Tier 5. ~10 lines now; later,
every Q-3 row taken before the move is unattributable.

---

### [ ] D10. Two of the three Python self-checks cannot run in a clean checkout

**Evidence - demonstrated in this worktree.** `python -m mirage.data` gives
`FileNotFoundError: no committed shards in ...\data\shards`. `mirage.validator` is
the same. Only `python -m mirage.config` passes. `data/` is gitignored, correctly -
it is 3.5 GB.

**Why it is debt.** `python -m mirage.data` **is F-8's acceptance test** and
`python -m mirage.validator` **is F-9's**. Both are unrunnable by anyone who has
not first generated 300,000 frames - which includes every fresh clone, every CI
run, and this worktree.

**Done when.** A tiny committed fixture - one shard of ~20 frames, ~250 KB,
produced by the real binary - plus both self-checks falling back to it when
`data/shards` is empty. `shard_writer_self_check` already proves the C++ side can
write a small shard to a temp directory; this is the Python-side mirror of it.

**Cheaper alternative, and why not to take it:** the self-checks could *synthesize*
a shard in temp from `meta_dtype` plus random pixels. That loses the "real bytes
from the real writer" property, which is most of what F-8 is testing. Prefer the
fixture.

---

### [x] D11. `MUJOCO_LOG.TXT` is a committed error log

**Evidence.** Tracked at repo root. Contents are one line from a deliberate
negative test on 2026-08-26: `Failed to load model from 'scene/nope.xml'`.

**Done when.** `git rm --cached MUJOCO_LOG.TXT` and add it to `.gitignore` -
MuJoCo writes it unconditionally next to the working directory, so it will come
back. One line, do it in the README pass.

**LANDED.** Untracked and ignored. The file stays on disk; MuJoCo rewrites it on
every run regardless.

---

## Tier 5 - Explicitly do NOT do these

Each already has a recorded trigger, and none has fired. Doing them now is the
speculative-fallback failure the architecture doc warns about.

| Leave alone | Why |
|---|---|
| Writing validator threshold **values** into config | The build order recalibrates against Phase 1 tokenizer reconstructions first; ground-truth frames are perfectly rendered while Q-3's inputs carry decoder artifacts. **D9 moves where tau lives, not what it is** |
| Phase 2 and Phase 4 structural plans | Phase 2's numbers wait on the tokenizer PSNR; Phase 4's whole plan derives from the Phase 3 profile. Drafting these is guessing |
| Connected components, parallel generation, WGL pbuffer, single-pass render, clang-cl UBSan | Triggers recorded, none fired |
| `mirage/logging.py` | Phase 1 machinery. Build it when Phase 1 needs it |
| E-4's 5% reproducibility for `mj_step` | Needs a quiescent-machine protocol, and it gates a *bench* number, not the dataset. Do it when Phase 3/4 timing starts mattering |
| F-7's knocked-off-table bias | Recorded. Q-6 re-derives from `visible_px` counts at any threshold, and separating the two cases needs a "block is on the table" field. Not worth a regeneration on its own - **but fold it in if D1 regenerates anyway** |

---

## Suggested order

**One sitting, roughly half a day, flat cost - but it clears the actively wrong
instructions and closes two Must rows:**

1. ~~**D6 + D7 + D11** - a single README and cleanup pass.~~ **DONE** - and it
   also corrected the same refuted rule in `world_model_architecture.md`, which
   was not in the original scope.
2. **D5** - create `runs.jsonl`, backfill Phase 0. Closes a Must row and gives
   step 3 somewhere to land.
3. **D4** - verify the clean build from scratch. Closes the other Must row.
4. **D9** - tau into config. ~10 lines, and it has to happen before any Q-3 row
   exists or that row is unattributable.
5. **D10** - the small fixture, so step 3's fresh clone can actually run F-8 and
   F-9.

**Then, before Phase 1 trains on the current shards:**

6. **D1 + D2**, and **D3 + F-7's bias** only if those force a regeneration.

**If Phase 1 is starting imminently, invert this and do 6 first.** Items 1-5 cost
the same next month. Item 6 is the only one whose price goes up, and it goes up by
the whole downstream hash tree.

---

## Not debt - recorded so it is not re-litigated

- **No test framework, no fixtures, self-checks per module.** A stated design
  choice, not an omission. D10 fixes the *inputs* to two self-checks, not the
  approach.
- **`json.tar.xz` at repo root.** Vendored `nlohmann/json`, pinned by a locally
  computed SHA256, which is the deliberate supply-chain choice.
- **`sim/` being deletable and `mirage/` never importing MuJoCo.** Working as
  designed - that is what makes "delete the simulator" literal.
- **Empty `engine: {}` in config.** It feeds `engine_hash` and is filled in Phase
  4. Correct as is.
