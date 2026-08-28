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

### [ ] D5. E-5 (append-only run log) has no file - `runs.jsonl` does not exist

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

---

## Tier 3 - Documentation that now actively misleads

### [ ] D6. README tells the reader to gate benchmarks on `pstate == P0`. That was refuted

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

---

### [ ] D7. README's environment table calls three verified things "unverified"

**Evidence.**

| Line | Says | Actually |
|---|---|---|
| `README.md:30` | GL backend "unverified - `mjr_readPixels` per-call latency is the day-1 blocker" | verified 2026-08-26, `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2`, and readback measured at 25.4 us |
| `README.md:31` | MuJoCo "wheel resolves for this Python; not yet exercised" | 300,000 frames generated through it |
| `README.md:32` | Compiler "unverified - MSVC or MinGW, remaining Phase 0 prerequisite" | MSVC, C++20 confirmed by the binary printing `202002` |

**Done when.** Table reflects the verification log. **Bundle with D6 and D11 in a
single README pass** - they are the same file and the same sitting.

---

### [ ] D8. Nine docs, overlapping, with nothing keeping the derived ones in sync

**Evidence.** `docs/` now holds 9 files. `timeline.md`, `decision_notes.md` and
`phase0_report.md` all declare themselves derived from the architecture and
requirements docs, but nothing enforces it. `CLAUDE.md` names this exact failure:
"expect the stale copies to outnumber the decision that caused them."

**Why it is debt.** D6 and D7 *are* this failure, already happening, in the most
visible file in the repo.

**Done when.** Pick one and stop:
- a `> Derived from: X, Y` header line on each derived file, plus one line in
  AGENDA saying a decision change requires a re-derivation pass; **or**
- accept the drift explicitly and say so.

**Do not build a sync tool.** The real mitigation is doing D6 and D7. Lowest
priority item on this list.

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

### [ ] D11. `MUJOCO_LOG.TXT` is a committed error log

**Evidence.** Tracked at repo root. Contents are one line from a deliberate
negative test on 2026-08-26: `Failed to load model from 'scene/nope.xml'`.

**Done when.** `git rm --cached MUJOCO_LOG.TXT` and add it to `.gitignore` -
MuJoCo writes it unconditionally next to the working directory, so it will come
back. One line, do it in the README pass.

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

1. **D6 + D7 + D11** - a single README and cleanup pass. Removes guidance that
   makes a reader reject valid measurements.
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
