# Handoff: the tokenizer decision, and the locality/entropy curve behind it

> **Derived explainer, never a citable source.** Same standing as
> `decision_notes.md` and `timeline.md`: it restates the authoritative files in
> plain words for a reader picking the project back up. The decision itself lives
> in `world_model_architecture.md`, the evidence in `runs.jsonl` r46 and the
> verification log. **When this file disagrees with those, they win** - and unlike
> them it is a snapshot, so its "still open" and "on disk" sections go stale.

**Session date:** 2026-08-29 into 2026-08-30. **Repo:** `C:\Users\nguye\Documents\Dev_and_Projects\mirage`,
branch `main`. It grew out of an agent-session handoff that lived outside the repo and is
probably gone; nothing here depends on it, and sections 6 and 7 carry forward
the only parts of it that mattered.

**Two commits landed, both on `main`:**

| commit | what |
|---|---|
| `1660129` | `feat:` rung r1c, `bench/token_stability_probe.py`, `runs.jsonl` r46, one verification-log row |
| `e0484fa` | `docs:` the R1 decision with its reversal trigger, plus `LocalNorm` and rung r1w3 |

`python check.py` passes at both: all 5 self-checks ok, register clean.

---

## 1. The decision, and where it actually lives

**Phase 2's tokenizer is R1 `20260829-005439-r1`** - FSQ `[8,8,8]`, no attention,
`GroupNorm` in the encoder, 60 epochs. Decided by the user on 2026-08-30.

**The authoritative record is `docs/world_model_architecture.md`, section
"Phase 2 inherits R1, and the encoder keeps `GroupNorm`".** `AGENDA.md` points at
it. **This handoff is not a source** - it is a derived explainer under the same
rule as `decision_notes.md`, and if it ever disagrees with the architecture doc,
the architecture doc wins. Do not cite this file.

### Why R1 and not R2

R2 wins *both* reported gate numbers, so the decision has to be argued rather
than read off a table.

| | R1 `20260829-005439-r1` | R2 `20260828-230015-r2` |
|---|---|---|
| Q-1 held-out PSNR | 31.095 dB | **31.182** |
| Q-2 token entropy | 74.1% | **77.6%** |
| tokens stable across batch size | **yes, structurally** | ~2 in 100,000 move |
| spurious token flips | **8.86%** | 18.75% |
| parameters | **744,966** | 1,008,646 |

- R2's Q-1 win is **+0.087 dB for +263,680 parameters**, about a sixth of the
  architecture doc's own "within ~0.5 dB means tied" threshold. **A measured
  non-lever.** Do not re-argue this on quality.
- **The load-bearing argument is E-1 determinism.** R2's encoder attention makes
  tokens batch-size dependent, and **Phase 3 encodes a seed clip at batch 1**.
  R1 structurally cannot have that exposure.
- Stability is the third argument and points the same way.

### Why not r1c, the tokenizer that is arguably better

This is the part worth understanding, because the surface reading is wrong.

r1c reaches within **0.282 dB** of R1 while using **26% fewer bits per frame**
(314.2 against 426.9). On a rate-distortion basis it is the better tokenizer, and
it is not collapsed - 463 codes live, zero unused. **It was still rejected**,
because it fails Q-2 at 54.6% against the 70% bar.

**The tempting move - "Q-2 is measuring the wrong thing for a dynamics model" -
was considered and rejected on purpose.** Lower token entropy is an *easier*
prediction problem for Phase 2, and the fidelity that matters is Q-1, which barely
moved. That argument is real. It was rejected because:

1. This project's discipline says moving or reinterpreting a bar because a run
   missed it is exactly the failure mode the discipline exists to prevent, and
   the 96x96 arm was held to that standard the previous day.
2. **r1c has a strictly weaker case than 96x96 did.** The 96x96 arm failed the
   *statistic* while satisfying the *rationale* - it delivered 1.68x the bits per
   frame with zero dead codes. r1c delivers **fewer** bits per frame than R1. It
   fails the bar and the rationale together, so there is no escape hatch.

**`NUM-BAR-Q2` is not moved.** If a future session wants to reopen this, it needs
new evidence about what Phase 2 actually needs, not a re-reading of the same
numbers.

---

## 2. The curve: what was measured, and what it means

The finding that drove all of this: **`GroupNorm` in the encoder makes every token
depend on every pixel of the frame.** Autograd on one latent cell, now asserted in
`python -m mirage.fsq`'s self-check:

| encoder normalisation | gradient support of one latent cell |
|---|---|
| channel-only (window 1) | **225 px**, exactly 15x15 |
| local, window 3 | **1,849 px**, 43x43 |
| local, window 5 or `GroupNorm` | **4,096 px**, the whole frame |

The conv field is 15 by arithmetic, `2*(2*(2*1+1)+1)+1`. A K x K normalisation
window adds `2*(K-1)*7` px on top, so **K=3 is the only interior point the
architecture admits** - K=5 already overshoots the 64 px frame. That is a property
of three stride-2 stages, not a choice, and it caps how finely this curve can ever
be sampled without changing the encoder's stage structure.

### The three measured points

12 held-out episodes, 460,032 cell-transitions, 396,013 of them with a *quiet*
receptive field. All three rungs are **744,966 parameters** - the same model with
one thing changed.

| | R1 (GroupNorm) | R2 (+attention) | r1c (channel-only) |
|---|---|---|---|
| support | 4,096 px | 4,096 px | 225 px |
| Q-1 PSNR | 31.095 dB | 31.182 | 30.813 |
| Q-2 entropy | 74.1% | 77.6% | **54.6% FAIL** |
| bits per frame | 426.9 | 447.2 | 314.2 |
| persistence | 85.67% | 77.28% | **93.22%** |
| P(flip given quiet field) | 8.86% | 18.75% | **0.00%** |
| spurious share of flips | 53.21% | 71.06% | **0.00%** |
| live codes | 485 | 460 | 463 |

**The zero is literal: 0 spurious flips out of 396,013.** That is the whole result.
Spatial coupling was not *a* mechanism behind tokens flipping for no local reason,
it was the *entire* cause. **It is also the probe's own control** - an encoder that
respects its receptive field is *required* to read exactly 0, so a nonzero reading
would have condemned `bench/token_stability_probe.py` rather than the model. Keep
using controls like this.

### Why entropy and stability are the same knob

They are two readings of one quantity: **how much of the frame each token sees.**
`GroupNorm` lets a token encode global context, so the same 8x8 patch gets
different codes depending on the rest of the frame - which raises entropy *and*
means the token changes when something far away moves. **You cannot buy one
without paying the other.** The trade is steep: locality costs 0.282 dB of Q-1 and
19.5 points of Q-2.

### The shortfall is skew, not collapse - and that closes the remedies

| | R1 | r1c |
|---|---|---|
| marginal skew | 1.440 bits | **2.959** |
| redundancy | 0.890 bits | 1.131 |

**The skew term doubled and redundancy barely moved** - the same signature r45
priced at 96x96, reached by a completely different route. By r45's arithmetic the
two named Q-2 remedies are a *collapse* fix (the shrink ladder) and a *redundancy*
fix (attention), and **neither touches skew**. Worse, r1c's **sum of channel
marginals is 6.041 bits = 67.1%, below the 70% bar**, so a *perfect* decorrelator
on top of this encoder still fails, model-independently. **Do not run an attention
or shrink rung on top of r1c.** It is refuted by arithmetic, at zero GPU cost.

---

## 3. What this decision explicitly does NOT claim

**Read this before quoting the stability numbers.**

**Nothing has measured what spurious flips cost a dynamics model, in either
direction.** The architecture doc says this outright, and it should stay said. The
case for caring is an argument, not a measurement:

- *for*: a model roughly 16x under Chinchilla spends scarce capacity learning
  global dependencies that carry no physics
- *against*: the flips are **deterministic functions of the whole frame**, and a
  transformer over a 975-token sequence can see the whole frame. They are not
  noise to it, and it may learn them for free

**Stability was a tiebreaker, not a driver.** It agreed with a choice the E-1
argument had already decided. A future reader will find three arguments all
pointing at R1 and could reasonably conclude stability was load-bearing evidence.
It was not.

**The trigger is one-directional: a rung can promote itself above R1, nothing
demotes R1.** Any tokenizer passing *every* gate row with materially fewer than
8.86% spurious flips is a legitimate replacement, being strictly better at the
same parameter count. Two other triggers are recorded: Phase 2 evidence that
instability costs accuracy would turn the tiebreaker into a driver, and if Phase 3
ever encodes its seed clip at the training batch size, R2's determinism objection
weakens.

---

## 4. State on disk

**Committed and clean.** `python check.py` passes.

- `mirage/fsq.py` - `ChannelNorm`, `LocalNorm`, `_norm`, rungs `r1c` and `r1w3`,
  `encoder_norm` threaded through `Tokenizer` / `train` / checkpoint knobs /
  `--resume`, and four new self-check assertions (support 225 / 4,096 / 1,849,
  `LocalNorm(window=1)` == `ChannelNorm` to 8.3e-07, `local5` saturates)
- `mirage/fsq_eval.py` - `load_run` reads `encoder_norm` with `.get(..., "group")`
  so **every pre-existing checkpoint still loads**
- `bench/token_stability_probe.py` - new, no GPU, reads the token cache and shard
  pixels. `python bench/token_stability_probe.py <run_id> [more...]`
- `runs.jsonl` **r46**, and one verification-log row at the end of
  `docs/world_model_architecture.md`
- `docs/world_model_architecture.md` + `AGENDA.md` - the decision

**Runs on disk.** `20260830-000842-r1c` is the finished r1c rung with its token
cache. `20260829-224438-r1c` and `20260829-233129-r1c` are dead partial runs from
crashes; they carry resumable checkpoints and are otherwise junk.
**`20260830-075300-r1w3` is the abandoned r1w3 rung, killed at epoch 4 of 60
(28.330 dB).** It is resumable:

    python -m mirage.fsq --run r1w3 --epochs 60 --seed 0 --resume 20260830-075300-r1w3

Re-issue it after each `nn.Upsample` crash, against the newest `runs/*-r1w3` id, then
`--tokens` / `--eval` and the stability probe. Section 5 has the crash budget.

which auto-resumes through crashes, then runs the token cache, gate, and probe.
**Only run it if you have decided stability matters.** Under the recorded trigger
r1w3 can only promote itself above R1, and by the doc, if it misses Q-2 then **no
further rungs should be run chasing this**.

**Not touched, and not mine:** `docs/writeup_part1.md` (modified before this
session) and `docs/research_token_stability.md` (untracked, from the previous
session). Both still uncommitted.

---

## 5. The `nn.Upsample` fault is now a routine operational cost

`AttributeError: 'str' object has no attribute 'align_corners'` inside
`Upsample.forward` - a native-layer frame corruption, not this project's bug.
**It fired twice during the r1c rung, at epochs 22 and 36**, roughly every 14
epochs. `--resume` carried both seams with no visible discontinuity (30.343 ->
30.381 dB across the first). **This is the first time `--resume` has carried a
rung to completion rather than merely been tested.**

**Budget for it: a 60-epoch rung is ~2 hours of training plus 2-3 crash restarts.**
A shell loop that resumes until `result.json` appears is worth the four lines; babysitting
it wastes a session. Two operational notes paid for in this one: `--resume` takes a **run
id, not a path**, and killing the wrapper does **not** kill the child python - kill the
trainer pid explicitly or it keeps training.

---

## 6. Still unrecorded, still not citable

**These findings were measured in an earlier session and exist only in its
conversation.** r46
records A and C in passing; **D, E and F had no `runs.jsonl` row, no
verification-log entry, and no register id.** By this project's own rules they
could not be quoted until reproduced or recorded.

**All three were re-run and recorded on 2026-08-30** - r47, r48, r49, with r50
added when the Q-5 repair was refuted. **Two of the three came back with
different numbers**, which is the argument for this table rather than against it:
a recollection that is 2.2x out on one figure and 1.3x out in the other direction
on its neighbour would have been quoted as fact by whoever wrote the Phase 2 plan.
The status column now carries what moved.

| | finding | status |
|---|---|---|
| A | F-11's 3x-marginal bar is beaten by a zero-parameter persistence baseline | **recorded** in r46 as evidence; the restatement itself is *not* decided |
| B | `GroupNorm` couples tokens globally | **recorded and superseded** by r46, which measured it properly |
| C | `bench/patch_probe.py:60` sets `RF = 22`; the true conv field is **15** | **recorded in r46, NOT FIXED.** See below |
| D | Q-3 cannot see dynamics failure; F-9 fires 0.0% on tokens from 300 steps later | **recorded 2026-08-30 in r48, and acted on**: Q-3's terminator is restated in `world_model_requirements.md`. Reproduced exactly - 0.00%, with both controls holding |
| E | Q-5's 10% link-drift bar fails the simulator's own frames (35.6% mean) | **recorded 2026-08-30 in r47, and acted on**: Q-5 is now relative, Q-4's treatment. **The number moved** - 44.2% on link1 and 23.0% on link0, not one 35.6% figure; the recollection appears to have averaged two links that fail for different reasons. r50 additionally refutes the obvious repair |
| F | Phase 2 sizing: 14,592,384 params, 975-token sequence, 38.4 MB cache, ~16x under Chinchilla, fp32 6.6 h/epoch vs bf16 49 min | **recorded 2026-08-30 in r49, and THREE OF ITS NUMBERS MOVED.** Params 14,593,152 for RoPE+untied (the recollection is 768 short, and the four layout variants span only 571,008 in total). Chinchilla shortfall 15.0x, not ~16x. **The epoch times were wrong in both directions**: fp32 is 2.99 h, not 6.6, and bf16 is 1.06 h, not 49 min. Sequence length and cache size reproduce exactly. Nothing depended on the two wrong ones |

**No `NUM-` id was minted for anything this session.** Registering is a separate,
deliberate act; quote `r46` until then.

### The `RF = 22` correction, deliberately not applied

`bench/patch_probe.py:60` is wrong and "22x22" is quoted in several documents.
The Q-2 ceiling derivation rests on it - "cells with identical receptive fields
must share a code" - and that premise is **vacuous while `GroupNorm` is in the
encoder**, because the effective field is the whole frame. **The direction is
safe** (a larger true field means more room than registered, not less) and **no
passed gate moves**, which is why it was logged rather than hot-fixed. Whoever
touches `NUM-TOK-Q2CEIL` next has to deal with it.

---

## 7. Open decisions, carried forward

The tokenizer-checkpoint decision is **closed** by this document. Seven others
were open alongside it and are untouched; each needs its own working through, and
what follows is only the reminder that they exist:

| # | decision | note |
|---|---|---|
| 2 | data volume: 500 episodes or regenerate 3x / 5x | gated on the `data_hash` provenance story |
| 3 | restate F-11 against the persistence baseline | evidence now recorded in r46; the decision is not taken |
| 4 | sequence layout and position encoding | **irreversible.** RoPE is the lazy correct answer |
| ~~5~~ | ~~Q-3's verdict expression and calibration population~~ | **closed 2026-08-30.** Finding D is recorded in r48 and Q-3 now terminates on frame-to-frame continuity rather than on F-9's palette verdict. The calibration population is settled with it: **reconstructions, not renders** - see the Q-3 risk row in `world_model_requirements.md`. What is still to be written is the expression's own thresholds, which is implementation, not a decision |
| 6 | rollout sampling: greedy vs temperature | sim is deterministic (E-1), greedy is the strong default |
| 7 | file split `dynamics.py` / `dynamics_eval.py` | do **not** start `engine.py` (Phase 3) |
| 8 | exposure bias: mitigate now or name a trigger | |

**The writeup remains the highest-priority item on `AGENDA.md`** and has an outside
deadline of 2026-09-09. Nothing in this session changed that, and the user was
explicit in the previous session that the deadline must not pressure Phase 2
decisions.

---

## 8. Traps

- **`docs/writeup_part1.md` is frozen.** Never cite it; a moved number gets an
  erratum, not an edit. Note it quotes R1 and R2 as "two checkpoints either of
  which could ship" - **that is now decided**, and if it matters it is an erratum.
- **Do not re-litigate WSL2 rendering or the `pstate == P0` gate.** Both settled,
  see repo `CLAUDE.md`.
- **Do not re-argue R2 on quality.** +0.087 dB is a measured non-lever.
- **Do not run attention or a shrink rung on top of r1c.** Refuted by arithmetic:
  its marginal sum is 67.1%, below the bar.
- **`--resume` refuses any checkpoint written before `encoder_norm` existed**, by
  the `k in ck["knobs"]` assert. That is the honest answer - those runs cannot be
  *shown* to match - but it means the pre-2026-08-30 partial checkpoints are not
  resumable. Nothing was in flight when it landed.
- **A wrong constant was used and caught only by a control** (the 22x22 field).
  The probe's zero reading for r1c is the same trick. **Keep building controls
  that must read a known value.**
- **The stability probe does not exactly reproduce the previous session's
  in-conversation figures** (85.67% vs 86.62% persistence for R1). Transition
  counts match, so it is the same episodes; the difference is edge-clipping of the
  15x15 window. The recorded probe is the citable one.
