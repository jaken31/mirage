# Phase 2: Structural Plan

Guidance for implementing Phase 2 - the dynamics model, `mirage/dynamics.py` in
the architecture doc's file map - one item at a time. Names the calls and the
order; does not write the code. Design rationale is in
`world_model_architecture.md` and is not repeated here.

Numbering below is this file's own. `AGENDA.md` opens Phase 2 and points here
rather than restating the list, because that file's rule is to stay short and
carry no history. If the two ever disagree, AGENDA is the order of record and
this file is the stale one.

**What gated this plan, and no longer does.**
`world_model_architecture.md`, "Which later-phase plans can be drafted now",
rates Phase 2 *structure yes, numbers no*, gated on the 64/144 fork - which is a
*Phase 1* result. That fork resolved to **64x64** (`runs.jsonl` r44 for the arm,
r45 for the decision), the checkpoint is chosen, and **`runs.jsonl` r49 priced
the model the docs already specify**. The numbers exist now, so this file quotes
them instead of predicting them.

**What Phase 2 inherits, all settled - do not reopen.** The R1 checkpoint
`20260829-005439-r1`, the fixed 512-code budget, and the 64-token path. The
reversal trigger is in `world_model_architecture.md` under "Phase 2 inherits R1,
and the encoder keeps `GroupNorm`", and it is **one-directional**: another rung
may promote itself above R1 by passing every gate row with fewer spurious token
flips, and nothing demotes R1. Nothing in this file fires it.

**Vocabulary, once.** The *dynamics model* is a decoder-only transformer: given
the last `ctx` frames as grids of token ids plus the action taken at each, it
predicts the next frame's tokens. It never sees a pixel. *Autoregressive* means
each position predicts the token at the next position; during training every
position is scored at once against the true next token, which is *teacher
forcing*, and during a *rollout* the model's own output is fed back instead -
the gap between those two regimes is *exposure bias*. *Interleaved* means one
sequence carries both frame tokens and action tokens, so an action is just
another token in the stream. A *causal mask* is the arithmetic that stops a
position attending to a later one; without it the model reads the answer.
*Position encoding* is how a token learns where it sits: *learned positions* is
one trainable vector per absolute position, capped at the longest sequence
trained on, while *RoPE* rotates each query and key by an angle proportional to
position, carries no parameters, and has no cap. A *tied output embedding*
reuses the input embedding matrix as the output projection rather than training a
second one. *Chinchilla-optimal* is the rule of thumb of about 20 training tokens
per parameter; well below it a model is **data-bound**, and its failure mode is
memorisation rather than underfitting. The *KV cache* belongs to Phases 3 and 4
and is not built here.

---

## The dataflow, in one line

R1's token cache on disk (`runs/20260829-005439-r1/tokens/shard_NNN.npy` plus its
`manifest.json`) and the shard meta records' `action` column -> one window of
`ctx` frames interleaved with their actions -> `Dynamics` (embed, pre-norm
blocks, causal mask) -> logits over the 512 codes at every frame-token position
-> cross-entropy. After training: roll out autoregressively from a seed clip,
decode each predicted token grid through R1's decoder, and score the rollout on
the decoded frames.

Everything flows one direction. Nothing calls backwards.

**Note what is *not* in that line: pixels.** Phase 2 trains on 38.4 MB of
`uint16` (r49), so nothing about the data path needs engineering - the whole
cache fits in VRAM many times over. The Phase 1 problem of a 3.5 GB working set
against the page cache does not recur, and `mirage.data.preload` is not on
Phase 2's training path at all. It returns to the path only for eval, where
decoded frames are compared against ground-truth ones.

---

## What each file owns

| File | Owns | Explicitly does not own |
|---|---|---|
| `mirage/dynamics.py` | the sequence layout, the token/action window sampler, the model, the train loop | pixels, the tokenizer, and *writing* the token cache - that is `fsq_eval.write_token_cache`, and Phase 2 is a reader of it |
| `mirage/dynamics_eval.py` | rollout and the gate table - everything that runs against a finished checkpoint. **Only if the 500-line trigger fires**, which is the same trigger that split `fsq_eval.py` out of `fsq.py` | training anything |
| `mirage/configs/base.json`, `dynamics` section | the shape knobs that must sit inside `dynamics_hash` | the training knobs, which travel in the checkpoint's `knobs` dict as Phase 1's do |
| `mirage/config.py` | the `dynamics` key set and its validators | anything model-shaped. It gains keys in item 2 and nothing else |
| `mirage/fsq.py` | the inverse of `FSQ.codes_to_indices`, which does not exist yet (item 5) | the rollout. The token-to-pixel path belongs beside the pixel-to-token path, not in a second copy |
| `mirage/data.py` | the window index arithmetic, *if* item 1 takes the shared option | anything token-shaped. It reads the shard format and knows nothing about codes |
| `mirage/validator.py` | the per-frame measurements Q-3's continuity verdict is built from | the verdict itself. Phase 0's rule stands: the validator emits measurements and the verdict is a threshold expression in config |
| `runs.jsonl` and the verification log | one row per run, and one verification row per claim | - |

`dynamics.py` reads the token cache and never re-encodes. Re-encoding would put a
second copy of the encode path in the tree, and the cache already carries the
provenance that makes the tokens checkable: `manifest.json` records `run_id`,
`checkpoint`, `tokenizer_hash`, `data_hash`, `levels`, `token_grid`, the
per-shard frame counts and sha256s, and the `batch` the cache was written at.

**Check those the way `fsq_eval.load_run` checks a checkpoint**, which raises
when the checkpoint's `data_hash` disagrees with the config rather than loading
anyway. The dynamics loader needs the same refusal against the manifest's
`data_hash`, `tokenizer_hash` and `token_grid`. Without it a cache written at a superseded
`data_hash` trains silently, and one from the 96x96 fork - 144 tokens per frame,
not 64 - fails somewhere downstream of the load rather than at it.

---

## Build order, with the calls each file needs

Riskiest first. In Phase 0 that meant "the thing that could invalidate 300,000
frames"; in Phase 1, the same. **Here nothing can**: the dataset and the
tokenizer are both frozen, and `data_hash` and `tokenizer_hash` are upstream of
every hash Phase 2 moves. What is at risk instead is **irreversibility** - the
sequence layout is baked into every checkpoint trained under it, into the engine's
decode schedule in Phase 4, and into what a Q-4 verdict means. So the order runs
irreversible-and-silent first, expensive second, and reported-only last.

### 1. The sequence layout, and the token/action window sampler

**Why first: it is irreversible, and getting it wrong is silent.** Every
checkpoint, every rollout and every gate number below is stated in terms of this
layout. `runs.jsonl` r49 exists so that the choice is not made by accident - it
prices four variants of the model and finds them **14,396,544 to 14,967,552
parameters, a spread of 571,008**, under 4%, so **the irreversible choice is not
a capacity choice** and has to be made on its merits.

The calls, all of which exist:

```python
shards = data.load_shards(ROOT / cfg.data["shard_dir"], cfg.data_hash)
index  = data.episode_index(shards)
for e in data.split_episodes(index, "train", cfg.data["val_fraction"]):
    # actions and phase come out of the same meta records the pixels came from
    act  = shards[e.shard].meta["action"][e.start:e.start + e.length]
    step = shards[e.shard].meta["step_idx"][e.start:e.start + e.length]
```

plus the cache rows, `np.load(f"runs/{run_id}/tokens/shard_{i:03d}.npy")`, and
`data.is_val` for the split. **Reuse `data.is_val`, never a second split rule.**
It is a pure function of `episode_id`, which is what makes the tokenizer's val
set and Phase 2's the same set by construction rather than by coincidence -
`NUM-DATA-SPLIT` episodes, `NUM-DATA-VALFRAMES` held-out frames.

**The alignment is a recorded fact, not a free choice, and a sweep inverts it.**
`mirage/data.py`'s module header and the alignment row in the verification log
at the end of `world_model_architecture.md` both state it: `sim/main.cpp` picks
the action, writes `ctrl`, calls `mj_step`, then reads truth, so within one
record

    qpos[t] - qpos[t-1]  is the result of  action[t]      <- the same record

Therefore the action token that conditions frame `t` is `action[t]`, and
whichever of the two interleavings is chosen, **exactly one of them needs an
index shift against the record**. This cannot be settled by scoring: an action is
held for `sim.action_hold_steps` frames, so a one-step shift leaves 14 of every
15 frames unchanged, and the measured agreement is **93.9% same-record against
95.6% next-record** - **the wrong reading scores higher**, because shifting hands
each delta the previous, already-settled action instead of the fresh one whose
transient Q-4 cannot win, and both clear Q-4's bar. What does settle it is
**phase**: `Policy::step` redraws only when its hold expires, so all **13,242**
action changes sit at `step_idx % action_hold_steps == 0`, and the negative
control - shifting by one - introduces phase 1.

**Two options for the window index arithmetic, and they are a real choice.**

- **Share it.** `data.WindowSampler` already owns episode-aware indexing: the
  cumulative start positions, the split filter, the refusal of any episode
  shorter than the window, and `__getitem__` as a pure function of the index so
  that a shuffle and a resumed run address the same window by the same number.
  Factor that addressing into something `dynamics.py` can call, leave
  `WindowSampler` using it, and assert the two agree on every window's
  (episode, offset). **This is what the project's own precedent asks for** -
  `preload` lives in `data.py` rather than `fsq.py` because "a copy in two files
  is the same class of bug as two validator implementations", and
  `split_episodes` was factored out for exactly this reason.
- **Copy it.** `dynamics.py` recomputes its own cumulative offsets over token
  rows. Cheaper to write, touches no file that two passed gates depend on, and
  it is the option that keeps `mirage/data.py` closed.

**Recommended: share it**, because the failure mode of the copy is a cumulative
frame offset that is off by one, which the architecture doc already calls "an
off-by-one factory" in the one place it was allowed (and refused: the token cache
is per-shard for precisely this reason). But note honestly what share costs:
`data.py` is 702 lines, its `_self_check` is F-8's acceptance test - "shard writer
emits packed frames and actions", accepted when the numpy round trip matches the
C++ buffer byte for byte - and any edit there is an edit to a file two gates
rest on. Nothing about the sampler's *indexing* touches the meta decode F-8
checks, which is the reason to believe the cost is small.

**Working when:** the action stream read back out of an assembled window changes
only where `step_idx % sim.action_hold_steps == 0`, and a deliberately shifted
copy fails that same assertion - a control that must read a known value, which
is the discipline r46's zero reading rests on, and the one that would have caught
the `RF = 22` constant before an autograd measurement had to. Also: each
window's token rows equal the cache rows for the frames `WindowSampler` would
have returned at the same index; no window straddles an
`episode_id` boundary; `len(tokens) == shard.frames` for every shard in the
manifest; and the val episode set is `data.is_val`'s, compared against it rather
than reimplemented.

### 2. `dynamics_hash` covers every shape knob, or E-4 and E-5 have a hole

Small, and it goes before the model because it decides where the model's shape
is written down.

**E-4** - "every bench number reproducible from a config hash", accepted when a
rerun matches within `NUM-BAR-E4` - and **E-5** - "append-only run log: config
hash, change, number, conclusion, one entry per run" - are both satisfied by
construction only if the hash names the thing that changed. Today the `dynamics`
section holds `d_model` and `n_layers` and nothing else, so **`n_heads` is in no
hash at all**: it lives as `N_HEADS = 6` in `bench/dyn_size_probe.py`, whose own
comment says "not in config". The same is true of whatever item 1 decides about
layout and position encoding.

Two lines of consequence, both verifiable by reading `mirage/config.py`:

- `EXPECTED_KEYS` is checked with `_check_keys`, which rejects **unknown** keys
  as well as missing ones. So a new `dynamics` knob is a `config.py` edit plus a
  JSON edit, never a JSON edit alone, and a count knob belongs in
  `POSITIVE_INT_KEYS` beside `d_model` and `n_layers`.
- `dynamics_hash = sha256(tokenizer_hash + canon(dynamics))` and
  `engine_hash = sha256(dynamics_hash + canon(engine))`, so a dynamics knob moves
  those two and **nothing upstream**. That is the hash tree working as designed:
  Phase 2 cannot invalidate the tokenizer or the data.

**Do not put the context length here.** `data.ctx` is inside the `data` section,
which is a term in `data_hash` - so editing it moves `data_hash`, and then
`load_shards(dir, cfg.data_hash)` refuses the 300,000 frames on disk and
`fsq_eval.load_run` refuses the R1 checkpoint. **F-13** - "configurable context
length at load time", accepted when one checkpoint rolls out at 4, 8 and 15
frames - must therefore be a **rollout argument**, not a config edit. This is
derived from the term order in `config.load` and is reproducible by hashing a
`ctx`-varied copy of `base.json`; it is not a measurement and no value of the
moved hash is quoted here.

**Working when:** editing a `dynamics` knob leaves `data_hash`,
`tokenizer_hash` and `validator_hash` unchanged and moves `dynamics_hash` and
`engine_hash`; dropping a `dynamics` key or adding an unknown one raises;
`python -m mirage.config` passes. The first three are the shape of assertions
`config._self_check` already makes for a `tokenizer` change - extend that block
rather than writing a new one.

### 3. The model

`d_model` 384, 8 layers, 6 heads, MLP ratio 4, pre-norm blocks, a causal mask,
vocab **521 in and 512 out** (512 codes plus the 9 actions in, codes only out),
and **15 frames x (64 + 1) = 975 positions** - all from r49, which measured the
parameter count for each of the four layout variants. **R-3** - "dynamics model
parameters <= 20M bf16" - passes at every one of them, 14.4 to 15.0 M, and the
requirement's own fallback says never above 40M, so capacity is not the pressure
here. `NUM-TOK-PARAMS-R1` is the tokenizer's count for comparison: the dynamics
model is about 19x it.

Three choices that are not stylistic:

- **`F.scaled_dot_product_attention`, not `nn.MultiheadAttention`.** The
  architecture doc's seam note requires this on the 144-token path because
  materialized attention caps the training batch size there; at 64 tokens the
  reason is the same in kind and smaller in degree. **Derived, not measured:** a
  materialized 975x975 attention matrix at 6 heads and batch 16 is about 91 M
  entries per layer, and r49's throughput was measured through
  `nn.MultiheadAttention`, so its figures describe the materializing path. Whether
  SDPA is faster here, and by how much, is **unmeasured**. Take the measurement on
  the first run rather than assuming either sign.
- **Score the loss on frame-token positions only.** The action at inference comes
  from the operator - **F-14** is "control loop reads keyboard and drives the
  model with MuJoCo not running" - so a next-token loss over action positions
  optimises a distribution the engine never samples, and spends capacity that
  `runs.jsonl` r49 says is already 15x short of its data. r49's probe takes
  cross-entropy over every position with a clamp, which is correct for timing a
  matmul and is explicitly not the implementation.
- **Keep the causality claim asserted, not assumed.** A causal mask that is
  subtly wrong trains a model that reads the answer and then fails only at
  rollout, hours later.

**Working when:** `python -m mirage.dynamics` self-checks with **no dataset and
no checkpoint**, the way `mirage.config`, `mirage.logging` and `mirage.fsq`
already do - the parameter count reproduces r49's figure for the variant item 1
chose, exactly; altering a token at position `k` leaves every logit at positions
`< k` bit-identical and changes at least one at `k`; and the sequence assembled
for one window round-trips to the same token rows item 1 asserted.

### 4. The training loop, and the instrument for the risk that actually exists

**bf16, and this is not the Phase 1 question re-asked.** Phase 1 trains in fp32
because the tokenizer is under a million parameters and because changing the
arithmetic under a 0.087 dB comparison would make later rungs incomparable. Here
r49 measures **221.6 ms/step in bf16 against 622.4 in fp32 at batch 16, 2.81x**,
for **1.06 h/epoch against 2.99 h**. Nothing in Phase 2 rests on a sub-tenth-dB
comparison, and the architecture doc's own condition for bf16 - "the 15M-parameter
dynamics model at context 1024, where it is necessary" - is the model being built.
Read that 1024 as the round number it is: the measured layout is 975 positions
(r49), and 1024 has never been a measurement of anything here.

**Quote r49's throughput as a lower bound and say so every time.**
`bench/gpu_probe.py` ran alongside and returned **compute FAIL**: the SMs held
2385 MHz of a 3090 MHz maximum and fp16 matmul read 20.6 TFLOP/s against
`NUM-HW-FP16`, the figure this machine records when cool. A clocked-up run is
faster by up to about a third. **Re-measure before any schedule leans on it**,
and gate the re-measurement on **SM clock plus power draw**, never on
`pstate == P0` - the pstate follows the memory clock domain and reads P4 during
correct compute-bound work. `NUM-HW-POWER` is the enforced limit to judge power
against.

**The risk to manage is overfitting, and it is measured rather than feared.**
r49: the cache carries **19.5 M tokens** against a Chinchilla-optimal
**291.9 M** for this parameter count, **15.0x under**. One epoch draws
**276,705 train windows totalling 269.8 M tokens**, which exceeds the dataset
**13.8x from window overlap alone** - that is repetition, not new information.
**So the first run's job is to locate the train/val gap, not to close it.** Log
held-out loss every epoch from the first, and treat the gap as the phase's
headline canary the way the tokenizer's train-val PSNR gap is row 8 of its gate.

**r49 prices the shortfall and decides nothing about it.** Three answers are
open - more data, a smaller model, heavier regularisation - and the row says so
explicitly. Picking one before the gap is measured would be choosing a remedy for
a magnitude nobody has yet seen; note that more data is also entangled with the
`data_hash` provenance story, since regenerating the set moves the hash and
orphans both the checkpoint and the cache.

Operational shape, all of it precedent rather than invention:

- **Per-epoch resumable checkpoint plus `--resume`, from the first run.** At
  ~1 h/epoch against `NUM-TOK-EPOCH64` for a tokenizer epoch, a run is hours
  long, and Phase 1 paid for this lesson twice: `--resume` carried the r1c rung
  across two mid-flight faults with no visible discontinuity (r46), and before
  that it was broken on CUDA and had never once been executed. `--resume` takes a
  **run id, not a path**.
- **Keep the machine awake for the duration.** `fsq._keep_awake()` is the
  per-process request that does it, and a dynamics run needs the same - import it
  or lift it somewhere both can reach, rather than writing a second copy. Modern
  Standby froze one Phase 1 epoch for 49 minutes; the wall clock keeps counting
  through a suspend, so the run survives and every timing it reports is void.
- **`logging.Run` for the jsonl, W&B behind the flag.** Every record carries the
  run id and the caller's hashes, so E-4 and E-5 are satisfied by construction.
  Pass `dynamics_hash`. Note the credential hazard the verification log records:
  with no key **on a terminal** wandb prompts without a timeout, so
  `Run` bounds it with `WANDB_LOGIN_TIMEOUT_S` and raises - which matters more
  here than in Phase 1, because the thing that hangs before step 0 is an
  overnight run.
- **`R-1`** - "peak training VRAM <= 7.5 GB", against `NUM-HW-VRAM` on the card -
  is **unmeasured for this model**: r49 recorded no VRAM figure, and its batch of
  16 was the probe's choice rather than a measured optimum. Measure both on the
  first run, at the batch actually used.

**One requirement needs its terms stated before it can be scored, and this plan
does not restate it. P-7** - "full 300k-frame epoch <= 30 min", tier S - names a
300,000-frame pass. A dynamics epoch as r49 defines it is 276,705 **overlapping
windows**, 13.8x the dataset in tokens, and prices at 1.06 h. Those are two
different objects, and the honest options are to score P-7 against a
300,000-frame equivalent, to score it against the windowed epoch, or to record it
as not applicable to Phase 2. **No bar moves either way**: naming an ambiguity is
not the same act as moving a threshold, and moving one because a run missed it is
the failure mode this project's discipline exists to prevent.

**Working when:** a one-epoch run writes a resumable checkpoint and a jsonl
carrying `dynamics_hash`; `--resume` continues it with no visible discontinuity
in the loss curve; the val loss is in the log from epoch 1; and peak VRAM and the
GPU clock state are recorded next to the step time.

### 5. The token-to-pixel path, which does not exist yet

Phase 2's output is token ids. Every quality requirement below is measured on
**pixels**. There is no code in the tree that turns ids back into pixels:
`FSQ.codes_to_indices` has no inverse, `Tokenizer` has `encode` and no
`decode`, and `fsq_eval.reconstruct` starts from pixels. `bench/q3_blind_probe.py`
says outright that it round-trips frame `t + lag`'s **pixels** rather than
decoding its cached token row, "instead of adding a second decode implementation".

So the inverse lands **once, in `mirage/fsq.py` beside `codes_to_indices`**, and
never as a copy in `dynamics.py`. It is the mixed-radix expansion that function
already documents, run backwards: recover each channel's digit from the id by the
place values, then undo the normalisation `codes_to_indices` undoes in the other
direction. Two cautions carry over from that docstring: the digit bound is **per
channel**, because a mixed table like `[8,6,5]` has three different digit ranges
and a single bound waves through an out-of-range digit; and **register no new
buffer**, because a new entry in `state_dict` makes existing checkpoints fail a
strict load for no gain.

**Working when:** enumerating all `prod(levels)` ids and round-tripping them
through both directions is exact - the forward direction's `_self_check` already
enumerates them, so extend it; and decoding a held-out frame's cached token row
reproduces `fsq_eval.reconstruct`'s `uint8` frame for that frame, **at the same
batch size**. Compare at the same batch because that is the confound gate row 5
was taught to avoid - it once re-encoded at 256 against a cache written at 128
and failed a determinism row on the batch difference rather than on
nondeterminism. With the batch pinned, any mismatch is the digit recovery being
wrong rather than the decoder, since both paths then feed the same convolutions
the same codes.

### 6. Rollout, and the gate

`dynamics_eval.py` if the 500-line trigger fires.

**Rollout shape.** A seed clip of `ctx` frames from a **val** episode, then one
action per step taken from that episode's own action column, decoded greedily.
The architecture doc's seam note is the reason the clip exists at all: **F-14**
drives the model with MuJoCo absent, but the model needs context before
predicting anything, so "delete the simulator" is exact and "delete the data" is
not. **F-12** - "generates a full next frame from previous frames plus one
action, fixed step count, no fallback path" - fixes the shape of one step: all 64
token positions are decoded, every time, with no early exit. Any parallel decode
schedule is Phase 4's ladder, not this.

**Greedy is the default and the alternative is open.** The simulator is
deterministic - **E-1**, "deterministic sim given a seed", defined as same seed
and action sequence give bit-identical frames - so the target distribution is a
point mass and greedy decoding is the strong default; it also makes a rollout
exactly reproducible from a checkpoint plus a seed clip, which is what lets E-4
compare two runs at all. Temperature sampling is the alternative, and what would
decide for it is evidence that greedy decoding collapses - a rollout that freezes
or falls into a short loop rather than one that drifts. Not chosen here.

**F-13's interaction with item 1's irreversible choice is the one operational
difference between the two position encodings**, and it is worth stating because
F-13 is scored from one checkpoint: with learned positions a shorter context has
to index the same absolute position table, and nothing past the trained 975
positions exists at all; with RoPE both directions are free. That is not an
argument that settles item 1 by itself, but it is a cost that belongs on the
table when it is settled.

### The proposed gate - one command

`python -m mirage.dynamics --eval RUN_ID` prints the table and exits nonzero when
a pass/fail row misses, mirroring `python -m mirage.fsq --eval`, which is what
makes a gate usable from a script rather than only by reading it.

**These rows are proposals against the requirements that already exist. They are
not new requirements, and none of them moves a bar.**

| # | Measure | Bar | What the requirement actually says |
|---|---|---|---|
| 1 | Held-out next-token accuracy, against the marginal-frequency baseline **and** against copy-the-previous-frame's-token, both reported | **see the open decision below** | **F-11** - "dynamics model consumes interleaved frame and action tokens, predicts next token", accepted when held-out accuracy beats the marginal-frequency baseline by 3x. r46 measured the zero-parameter copy baseline at **85.67%** on the checkpoint Phase 2 inherits, far above 3x the marginal top-1. **Whether F-11 is restated because of that is a Phase 2 decision and this plan does not take it** |
| 2 | One full next frame from `ctx` frames plus one action, fixed step count | **exact**, no fallback path | **F-12** - as quoted above. All 64 positions decoded every step |
| 3 | Rollout at `ctx` 4, 8 and 15 from one checkpoint | **runs** | **F-13** (S) - "configurable context length at load time". Item 2's gotcha is why this is an argument and not a config edit |
| 4 | Frames until the frame-to-frame continuity verdict fires | **>= 200** | **Q-3** - the coherence horizon. The verdict bounds per-step change in `link_angle`, `link_extent` and each block's `bbox` centroid, calibrated so that **zero windows of ground-truth frames fire** - the same acceptance shape **F-9** uses, F-9 being "frame validator reports block count, arm pose plausibility, palette adherence", accepted at zero false positives on ground-truth frames. Calibrate on **reconstructions, not renders**; `bench/q3_blind_probe.py` is the regression test and must fire on 100% of 300-step substitutions and 0% of clean reconstructions |
| 5 | Action-following agreement, and the simulator's own agreement on the same subset | **>= 90% of the ground-truth term, both numbers reported** | **Q-4** - agreement is `sign(theta_t+1 - theta_t)` against the commanded sign, on an **action-balanced** subset drawn from the val split. An absolute bar there fails a model that is exactly right, which is why the bar is relative. `bench/hold_probe.py` measures the ground-truth term - **re-measure it, see the gotcha** |
| 6 | Link-length drift over a 200-step rollout, per link, and the simulator's own drift | **<= 1.1x the ground-truth term, per link, both numbers reported** | **Q-5** - the statistic is the pixel-measured major extent's `(max - min) / median` over non-overlapping 200-frame windows. Ground truth reads 23.0% on link0 and 44.2% on link1, so a perfect model fails any absolute bar. `bench/link_drift_probe.py` measures the ground-truth term. **Do not re-attempt the deprojection** - r50 records it as measured and refuted |
| 7 | Block reappears in the correct position after full occlusion | **>= 80% of events** (S) | **Q-6** - object permanence, the memory result. Tier S: the project ships without it and the negative result gets reported either way. `mirage.data.seen_later` owns the recoverable-occlusion split, and `NUM-DATA-F7` is the event rate it scores over |
| 8 | Parameter count, and peak training VRAM | **<= 20M bf16**, **<= 7.5 GB** | **R-3** and **R-1**. r49 settles R-3 at 14.4-15.0 M; R-1 is **unmeasured** for this model and item 4 takes it |
| 9 | Rollout reproduced from the checkpoint plus the seed clip | **identical** | **E-1** and **E-4** - a rerun matching within `NUM-BAR-E4`. Greedy decoding makes this exact rather than statistical; a temperature decision would change this row's shape |
| 10 | Train-val loss gap; share of predictions the copy baseline also gets right | **reported** | not requirements - the overfitting and triviality canaries. The first is item 4's headline instrument; the second is what keeps row 1 honest whatever F-11 ends up saying |

Rows 1 to 6 and 8 to 9 are the pass/fail candidates; 7 is S-tier and reported;
10 is reported. Row 1's bar is blocked on the open decision below and the row
cannot be finalised without it.

---

## Open decisions this plan does not take

Each one changes an item above, so each is named rather than quietly resolved.

**1. F-11's acceptance test, and it is escalated rather than decided here.**
F-11 asks the dynamics model to beat a marginal-frequency baseline by 3x.
`runs.jsonl` r46 recorded, for Phase 2 and deliberately not acted on, that
**copying the previous frame's token at the same cell scores 85.67%** on the R1
checkpoint Phase 2 inherits, over 460,032 held-out cell-transitions from 12 val
episodes - far above 3x the marginal top-1, **at zero parameters**. The options:
leave F-11 as written and report the copy baseline beside it as a canary; restate
F-11 against the persistence baseline; or restate it against a baseline that is
neither. The evidence for all three is r46 and it is the same evidence; what
differs is what the phase's M-tier row is then claiming. **This decision is not
this plan's to take and no side is taken.**

**2. Sequence layout and position encoding** - r49 records these as the two
choices already called irreversible, and prices them rather than taking them:
four variants, 571,008 parameters apart, so **not a capacity choice**. Item 1's
alignment fact constrains the layout, and F-13 costs the two position encodings
differently. Item 1 takes it.

**3. Tied against untied output embedding** - the other axis of the same table,
same 4% spread. Note the caution in `bench/dyn_size_probe.py`, the probe r49
records: tying is "only sound because the frame codes are the first block of the
vocabulary and the action tokens are appended after them", which is a property of
item 1's vocabulary layout and not a free one.

**4. The answer to the Chinchilla shortfall** - more data, a smaller model, or
heavier regularisation. r49 prices the gap at 15.0x and decides none of them.
Item 4's position is that the first run measures the gap before anything answers
it.

**5. Greedy against temperature sampling at rollout** - item 6's default is
greedy, on E-1's determinism, and the trigger for revisiting is named there.

**6. Whether `dynamics_eval.py` splits out** - the same 500-line trigger that
split `fsq_eval.py`, applied when it fires and not before.

**7. Exposure bias: mitigate now, or name a trigger** - teacher forcing trains on
ground-truth context and the rollout feeds the model its own output. The cheap
position is to name the trigger (a coherence horizon that collapses well before
Q-3's 200 while held-out next-token accuracy looks healthy) rather than to buy a
mitigation before that signature appears.

**8. What P-7 means for a windowed epoch** - item 4 states the three readings and
takes none.

---

## Gotchas, and how you would notice

| Gotcha | What breaks | How you notice |
|---|---|---|
| **A one-step action misalignment** | Every checkpoint conditions each frame on the wrong action, and Q-4 scores the wrong thing | **You do not**, from agreement: 93.9% against 95.6%, and the **wrong** reading scores higher, with both clearing Q-4's bar. Only the phase assert catches it - all 13,242 action changes sit at `step_idx % action_hold_steps == 0`. Assert it, and assert that a shift of one breaks it |
| Putting the context length in `data.ctx` for F-13 | `data_hash` moves, `load_shards` refuses the 300,000 frames and `load_run` refuses the R1 checkpoint | Loudly, on the next run - which is the good case. The bad case is a session spent editing the register instead of passing an argument |
| A shape knob outside the `dynamics` section | `dynamics_hash` does not name the model that produced the number, so E-4 has a hole | **You do not.** Two runs with different head counts log the same hash. `n_heads` is in this state today, as a constant in `bench/dyn_size_probe.py` |
| Calibrating Q-3's continuity verdict on **renders** | The verdict is tuned in the wrong regime and fires on ordinary decoder output | The same two-regime trap that cost Phase 1's build-order item 6 its obvious recipe: renders sit at `NUM-VAL-WORSTDIST` from the palette, reconstructions at `NUM-VAL-RECONDIST`. Calibrate on reconstructions |
| Reusing **F-9** - the per-frame plausibility check, accepted at zero false positives on ground truth - as the rollout terminator | The horizon measures decoder artifacts and nothing about dynamics | **Refuted and measured**: r48, F-9 fires on **0.00%** of frames substituted from 300 steps away, against 100% on its noise control. F-9 itself is unchanged and still does its own job. `bench/q3_blind_probe.py` is the regression test that catches a replacement going blind the same way |
| **Quoting `NUM-DATA-Q4CEIL` as Q-4's ground-truth term** | The Q-4 row is scored against the superseded physics | The id reads **83.1%**, measured before the `gear 6 / damping 1.5` change, while **r20 - the row the register cites - measures 91.5% after it**, with a relative bar of 82.3%; `docs/phase0_debt_checklist.md` records the same before/after pair. Q-4's acceptance test says to re-measure the ground-truth term on the same subset anyway, so **re-measure with `bench/hold_probe.py` and report both numbers**. Do not copy either stored value, and do not silently edit the register from a Phase 2 plan |
| A second decode implementation in `dynamics.py` | Two token-to-pixel paths that will eventually disagree | The disagreement is a wrong picture, which nothing crashes on. Item 5 exists so there is one path |
| Training on token rows as if they were upside down | Well-formed tokens for mirrored frames | **Already handled** - `write_token_cache` flips the rows on the way in, exactly as `preload` does, and says so. Do not add a second flip: the blob is bottom-up and the flip lives in one place |
| Hardcoding the re-encode batch | An E-1 check that tests "re-encode at a different batch size" instead of determinism | Gate row 5's own history: the check originally re-encoded at 256 against a cache written at 128 and failed R2 on a false alarm. Read `batch` from the manifest |
| Mixing token caches from two runs | The model trains on a mixture of two tokenizers | The manifest's `tokenizer_hash` and `run_id` disagree with the checkpoint's. The cache directory is named by run id for this reason |
| The `nn.Upsample` native-layer fault | It is a fault in the **tokenizer's decoder**, so the dynamics train loop cannot hit it - the **rollout decode path can** | `AttributeError: 'str' object has no attribute 'align_corners'` inside `Upsample.forward`. It fired at epochs 22 and 36 of one 60-epoch tokenizer rung (r46), so roughly every 14 epochs of exercising that decoder. Not this project's bug; budget restarts for any long eval that decodes |
| Modern Standby mid-run | Every timing the run reports is void while the run itself survives | A single epoch reading many times its neighbours. `fsq._keep_awake` is the per-process request that prevents it, and it is a no-op off Windows |
| Reading `pstate == P0` as a valid-benchmark gate | A correct compute-bound run is rejected, or a throttled one accepted | Refuted 2026-08-23. Gate compute on SM clock plus power draw, bandwidth on memory clock, and record them next to the number - which is exactly why r49's own throughput is a lower bound |
| **Touching `bench/patch_probe.py`'s `RF` constant, or anything quoting 22x22** | The constant is wrong in a way the register depends on | r46 measured the true conv field at **15x15 = 225 px** and the effective field under the encoder's `GroupNorm` at the whole **4,096 px** frame. `RF = 22` is neither, and `NUM-TOK-Q2CEIL`'s derivation rests on the premise that cells with identical receptive fields must share a code - **vacuous while `GroupNorm` is in the encoder**. r46 logged it **deliberately unrepaired**: the direction is safe, since a larger true field means more room than registered, and no passed gate moves. **Phase 2 does not fix it**, and a Phase 2 item that quotes 22x22 is quoting a known-wrong number |
| Reading a falling train loss as progress | The 13.8x window overlap means an epoch is mostly repetition | Train loss falls while held-out loss rises. This is the phase's expected failure and item 4's instrument exists for it |

---

## The numbers already taken, so item 1 does not re-derive them

Every row traces to a `runs.jsonl` row, a `NUM-` register id, or the verification
log at the end of `world_model_architecture.md`. **Cite the id, do not copy the
value**, and where no id exists the row says which notebook row to quote.
Nothing here was measured by this plan: there is no dataset and no checkpoint on
the machine it was written on.

| Measure | Number | What it settles |
|---|---|---|
| Model shape | `d_model` 384, 8 layers, 6 heads, MLP 4x - r49 | item 3 does not choose these; they were specified and are now priced |
| Sequence | 15 frames x (64 + 1) = **975** positions, vocab **521** in / **512** out - r49 | item 1's arithmetic, and the 975 that caps a learned position table |
| Parameters | **14,396,544** (RoPE + tied) to **14,967,552** (learned + untied), spread **571,008** - r49 | **the irreversible layout choice is not a capacity choice**, and R-3 passes at all four |
| Step and epoch cost | bf16 **221.6 ms/step** at batch 16, **1.06 h/epoch**; fp32 **622.4 ms** and **2.99 h** - r49 | bf16 for item 4. **A lower bound**: `gpu_probe` returned compute FAIL, 2385 of 3090 MHz and 20.6 TFLOP/s against `NUM-HW-FP16` |
| Data against capacity | **19.5 M** cache tokens against a Chinchilla-optimal **291.9 M**, **15.0x under**; one epoch draws **276,705** windows totalling **269.8 M** tokens, **13.8x** the dataset from overlap - r49 | **the phase's risk is overfitting, not throughput**, and the three answers stay open |
| Token cache size | **38.4 MB** - r49 | the whole cache fits in VRAM many times over; nothing about the data path needs engineering |
| The inherited checkpoint | `20260829-005439-r1`, `NUM-TOK-R1-60` held-out PSNR at `NUM-TOK-ENT-R1` token entropy | the tokenizer is fixed, and so are the 512 codes and the 64-token grid |
| The zero-parameter baseline | **85.67%** token persistence on R1, over **460,032** transitions from 12 val episodes, **396,013** of them quiet-field - r46 | F-11's open decision. **Evidence, not a verdict** |
| Token instability on R1 | **8.86%** of transitions flip with no change in the cell's own 15x15 field, **53.21%** of all flips - r46 | context for a rollout that looks noisy. r46 is explicit that **nothing has measured what spurious flips cost a dynamics model**, in either direction, and that Phase 2 evidence either way is a recorded trigger to reopen the tokenizer choice |
| The split | `NUM-DATA-SPLIT` train/val episodes, `NUM-DATA-VALFRAMES` held-out frames of `NUM-DATA-FRAMES` total, at `NUM-DATA-HASH64` | reuse `data.is_val`; the tokenizer's val set and Phase 2's are the same set by construction |
| Action alignment | same-record **93.9%** against next-record **95.6%**, and all **13,242** action changes at phase 0 - verification log, the alignment row | item 1. The agreement figures are there to prove agreement cannot settle it |
| ctx=15 window coverage | **60.9%** of windows carry any evidence of what an action does - r20, at the shipped physics (`gear 6 / damping 1.5`, `action_hold_steps` 15). r23 re-ran the gate at `NUM-DATA-HASH64` with the scene and the policy untouched and found every M-tier row unchanged, so an r20 figure still describes the set on disk - but **window coverage is not one of the rows r23 re-measured** | about 39% of training windows contain no action change at all, which is what an action-balanced Q-4 subset is drawn against |
| Q-4's ground-truth term | **91.5%**, relative bar **82.3%** - r20's after-values at the shipped physics. **`NUM-DATA-Q4CEIL` disagrees**, see the gotcha | re-measure with `bench/hold_probe.py` and report both numbers |
| Q-5's ground-truth term | **23.0%** on link0 and **44.2%** on link1 - r47, quoted in Q-5's requirement row | a perfect model fails any absolute bar; `bench/link_drift_probe.py` produces the term |
| Q-3's terminator | F-9 fires on **0.00%** of 300-step substitutions against **100%** on the noise control - r48 | the continuity verdict replaces it, and the probe stays as its regression test |
| Run-to-run noise | `NUM-PERF-NOISE`, and it is the **tokenizer's** 1-epoch figure | **unmeasured for a dynamics rung.** Do not call a margin "inside the noise" here; no seed has been repeated on this model |
| Peak training VRAM | **unmeasured** for this model - r49 recorded none | item 4 takes it, at the batch actually used. Batch 16 is the probe's choice, not an optimum |
| SDPA against materialized attention | **unmeasured** - r49 timed `nn.MultiheadAttention` | item 3 measures it rather than assuming a sign |
| Rollout throughput, and any P-row | **unmeasured**, and deliberately - **P-1** to **P-5** are the interactive rows (sustained frame rate, p99 frame time, input-to-display latency, the p99/p50 jitter ratio, and the eager-to-engine speedup), which belong to Phase 3's baseline and Phase 4's ladder | Phase 2 produces a checkpoint, not a frame rate. `NUM-BAR-P7` is the only P-row this phase touches, and item 4 says what is ambiguous about it |

Record the GPU power state next to every timing. A timing without it is not a
number - and gate compute numbers on **SM clock plus power draw**, bandwidth on
**memory clock == max**.

---

## What this plan does not do

No part of `mirage/dynamics.py` is written here, no run was launched, and no
measurement was taken - there is no dataset and no checkpoint on the machine this
was written on, so every number above is quoted from the record rather than
earned here. **No `NUM-` id is minted**: registering a number is a separate,
deliberate act, and until then r46 and r49 are what to quote. **No bar is
moved.** Phases 3 and 4 stay undrafted, which is "profile before changing
anything" applied to planning.
