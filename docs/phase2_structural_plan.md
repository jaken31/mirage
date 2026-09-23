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
*Strictly causal* lets each position see only the positions before it;
*block-causal* also lets the positions of one frame see each other, so a frame's
tokens are predicted together rather than one at a time. *Position encoding* is
how a token learns where it sits: *learned positions* is one trainable vector
per absolute position, capped at the longest sequence trained on, while *RoPE*
rotates each query and key by an angle proportional to position, carries no
parameters, and has no cap. A *tied output embedding* reuses the input embedding
matrix as the output projection rather than training a second one.
*Chinchilla-optimal* is the rule of thumb of about 20 training tokens per
parameter; well below it a model is **data-bound**, and its failure mode is
memorisation rather than underfitting. The *KV cache* belongs to Phases 3 and 4
and is not built here.

---

## The dataflow, in one line

R1's token cache on disk (`runs/20260829-005439-r1/tokens/shard_NNN.npy` plus its
`manifest.json`) and the shard meta records' `action` column -> one window of
`ctx` frames interleaved with their actions -> `Dynamics` (embed, pre-norm
blocks, causal mask) -> logits over the 512 codes at every position whose
target is a frame token -> cross-entropy. After training: roll out autoregressively from a seed clip,
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
| `mirage/data.py` | the window index arithmetic, shared with `WindowSampler` by the decision in item 1 | anything token-shaped. It reads the shard format and knows nothing about codes |
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

### Before item 1: strictly-causal against block-causal attention

**Ordered 2026-09-22 to run before item 1, and it ran: block-causal is
selected** - `runs.jsonl` r54, `bench/mask_probe.py`, 2026-09-23. It went ahead
of item 1 for item 1's own reason - the mask is baked into every checkpoint
trained under it - and because it decided whether F-11's description still
holds. It does not: `world_model_requirements.md` carries the second,
separately dated amendment the requirement prescribed, and the 2026-09-22
restatement of F-11's acceptance test is untouched.

**What r54 found, in one paragraph - quote r54, not this.** Both arms fed one
token stream and were scored on the next frame each *generated* from
ground-truth context. Block-causal led by 2.40 points against a seed spread of
0.20. Strictly-causal reads the true earlier cells of its own target frame under
teacher forcing and its own guesses at rollout, and the gap between those two
readings of one checkpoint is larger than its whole deficit to block-causal. So
**gate row 1's measure taken as teacher-forced accuracy would have picked the
wrong arm**. Under
block-causal the two readings are one number. Neither arm beat persistence at
r54's one-epoch endpoint, which is past both arms' held-out minimum. That is
item 4's overfitting risk showing up on schedule, not an F-11 verdict. **The
recorded trigger to reopen the mask:** a generated-frame comparison at each
arm's held-out optimum, on r54's population, with strictly-causal ahead by more
than the seed spread. r54 did not measure that.

**The items below were written for strictly-causal attention**, which is what
r49 timed: `bench/dyn_size_probe.py` builds its mask as `torch.triu` over all
975 positions. The block-causal result changes four things, listed here so they
are found in one place, and **these now govern wherever an item below says
otherwise**:

- **item 3's causality assert** moves from per position to per block, a block
  being frame `t`'s 64 tokens plus `action[t+1]`: altering a token in block `b`
  leaves every logit in the blocks before `b` identical, and may change any
  logit inside block `b`.
- **item 3's loss and item 6's decode step**: a frame's 64 predictions come out
  together rather than one position at a time. F-12's fixed step count and "all
  64 positions every step" hold under either mask.
- **how a frame's positions are fed in training.** A frame cannot be both the
  input and the target of its own block. r54's arrangement keeps decision 2's
  stream exactly and only regroups it: blocks of 65, frame t's 64 tokens plus
  the action after them, `action[t+1]`, full attention inside a block and
  causal across blocks. Frame t+1's cell i is read at the position holding frame
  t's cell i. `bench/mask_probe.py`'s `layout` is that arrangement, and its
  `--self-check` asserts it. Decision 2's alignment - both tokens read at the
  same record index, and the phase assertion - holds under either mask.
- **how long a training sequence is.** A `ctx + 1` window of 16 frames feeds
  15 blocks, **975 positions** - r49's priced length - because the last frame
  is only ever a target. Item 1's derived 1,040 is the strictly-causal figure.

The order recorded "at a fixed step budget" and no more. r54 states the
budget, the score and the population: one epoch at batch 16, two seeds per arm,
per-cell accuracy of the generated next frame against persistence, over every
val window's last frame. The plan's suggestion - gate row 1's measure - was
taken in that generated form, for the reason in the paragraph above.

### 1. The sequence layout, and the token/action window sampler

**Why first: it is irreversible, and getting it wrong is silent.** Every
checkpoint, every rollout and every gate number below is stated in terms of this
layout. `runs.jsonl` r49 exists so that the choice is not made by accident - it
prices four variants of the model and finds them **14,396,544 to 14,967,552
parameters, a spread of 571,008**, under 4%, so **the irreversible choice is not
a capacity choice**. It was made on its merits, as decision 2 below: **RoPE, and
the no-shift interleaving.**

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

**The layout, by decision 2.** Each frame step is **65 positions** - r49's 64
tokens plus one action - in this order:

    action[t], then frame t's 64 tokens        both read at record index t

The one action token that conditions a frame sits immediately before that
frame's 64 positions, and both come from the same record, so the layout needs
**no index shift against the record**. Under strictly-causal attention every
position of frame `t` then sees `action[t]`. This is the recorded alignment, not
a free choice: `mirage/data.py`'s module header and the alignment row in the
verification log at the end of `world_model_architecture.md` both state it.
`sim/main.cpp` picks the action, writes `ctrl`, calls `mj_step`, then reads
truth, so within one record

    qpos[t] - qpos[t-1]  is the result of  action[t]      <- the same record

**Agreement statistics must not be used to confirm it, because the wrong reading
scores higher.** An action is held for `sim.action_hold_steps` frames, so a
one-step shift leaves 14 of every 15 frames unchanged, and the measured agreement
is **93.9% same-record against 95.6% next-record**. Shifting hands each delta the
previous, already-settled action instead of the fresh one whose transient Q-4
cannot win, and both readings clear Q-4's bar. What does settle it is **phase**,
and the phase assertion is **mandatory - it is this item's acceptance test**.
`Policy::step` redraws only when its hold expires, so all **13,242** action
changes sit at `step_idx % action_hold_steps == 0`, and the negative control -
shifting by one - introduces phase 1.

**How many frames a window holds is the one layout number r49 priced two
ways.** r49 prices a sequence at `ctx x 65` = 975 positions, but counts its
epoch in `WindowSampler` windows, and those hold **`ctx + 1`** frames - the
context plus the frame it predicts. `WindowSampler.__init__` sets
`self.window = ctx + 1`, and r49's 276,705 train windows are that count. Sharing
the addressing, below, makes a Phase 2 window the sampler's window, and under
teacher forcing its last frame is the one predicted from a full `ctx` of
context, which is what F-13's rollout at 15 asks for. **Derived, not measured:**
that sequence is 16 x 65 = **1,040** positions at `ctx` 15 - under
strictly-causal attention. **Block-causal, selected by r54, feeds the same window
as 975 positions**; see "Before item 1". Under RoPE the
parameter count does not depend on sequence length, so r49's counts stand. Its
tokens per epoch, step time and epoch time were all priced at 975 positions a
window, and at 1,040 all three are higher by an amount r49 did not measure -
re-take them on the first run. **Recommended: `ctx + 1`.** The alternative is to
train on `ctx`-frame windows and match r49's pricing, at the cost of never
training the full-context prediction F-13 scores. It is named here rather than
taken silently, and whichever is built, the first run's row records it.

**The window index arithmetic is shared, not copied - DECIDED 2026-09-21.**
`data.WindowSampler` already owns episode-aware indexing: the cumulative start
positions, the split filter, the refusal of any episode shorter than the window,
and `__getitem__` as a pure function of the index so that a shuffle and a
resumed run address the same window by the same number. Factor that addressing
into something `dynamics.py` can call, leave `WindowSampler` using it, and
**assert that the shared addressing and `WindowSampler` agree on every window's
(episode, offset)**. This is the project's own precedent - `preload` lives in
`data.py` rather than `fsq.py` because "a copy in two files is the same class of
bug as two validator implementations", and `split_episodes` was factored out for
exactly this reason. The rejected alternative was a copy in `dynamics.py`,
cheaper to write and leaving `data.py` closed. Its failure mode is a cumulative
frame offset that is off by one, which the architecture doc already calls "an
off-by-one factory" in the one place it was allowed (and refused: the token
cache is per-shard for precisely this reason).

**What sharing costs, recorded honestly: it edits `mirage/data.py`.** That file
is 702 lines, and its `_self_check` is F-8's acceptance test - "shard writer
emits packed frames and actions", accepted when the numpy round trip matches the
C++ buffer byte for byte - so any edit there is an edit to a file a passed gate
rests on. **The cost is believed small** because the edit moves indexing and
nothing else. The meta decode F-8 checks, the structured dtype and the row flip
in `__getitem__` are all untouched, and `_self_check` runs unchanged afterwards
on both the generated set and the fixture. If it does not pass unchanged, the
edit reached further than indexing.

**Working when - and the first sentence is the acceptance test.** Every action
change in the action stream the windows are assembled from sits at
`step_idx % sim.action_hold_steps == 0` - over the generated set, the alignment
row's 13,242 of them - and a deliberately shifted copy **fails that same
assertion**. No agreement figure enters this test. A control that must read a
known value is the discipline r46's zero reading rests on, and the one that
would have caught the `RF = 22` constant before an autograd measurement had to.
With no dataset, the same pair runs on the committed fixture, whose 3 action
changes `python -m mirage.data` already asserts at phase 0. Also: the shared
addressing and `WindowSampler` agree on every window's (episode, offset); each
window's token rows equal the cache rows for the frames `WindowSampler` would
have returned at the same index; no window straddles an `episode_id` boundary;
`len(tokens) == shard.frames` for every shard in the manifest; the val episode
set is `data.is_val`'s, compared against it rather than reimplemented; and
`python -m mirage.data` still passes.

### 2. `dynamics_hash` covers every shape knob, or E-4 and E-5 have a hole

Small, and it goes before the model because it decides where the model's shape
is written down.

**E-4** - "every bench number reproducible from a config hash", accepted when a
rerun matches within `NUM-BAR-E4` - and **E-5** - "append-only run log: config
hash, change, number, conclusion, one entry per run" - are both satisfied by
construction only if the hash names the thing that changed. Today the `dynamics`
section holds `d_model` and `n_layers` and nothing else, so **`n_heads` is in no
hash at all**: it lives as `N_HEADS = 6` in `bench/dyn_size_probe.py`, whose own
comment says "not in config". The same is true of what decisions 2 and 3 fixed -
RoPE and the untied head - and of the mask the measurement before item 1
selects: each has to be named in the `dynamics` section, or two checkpoints that
differ in it log the same hash.

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
parameter count for each of the four layout variants (item 1 has why a training
window is 16 frames, not 15). **Decisions 2 and 3 take RoPE with an untied output
head, which r49 records as 14,593,152** (`rope_untied`) - the variant r49's
Chinchilla figure is computed against, and the parameter layout it timed.
**R-3** - "dynamics model parameters <= 20M bf16" - passes at every one of the
four, 14.4 to 15.0 M, and the requirement's own fallback says never above 40M,
so capacity is not the pressure here. `NUM-TOK-PARAMS-R1` is the tokenizer's
count for comparison: the dynamics model is about 19x it.

Three choices that are not stylistic:

- **`F.scaled_dot_product_attention`, not `nn.MultiheadAttention`.** The
  architecture doc's seam note requires this on the 144-token path because
  materialized attention caps the training batch size there; at 64 tokens the
  reason is the same in kind and smaller in degree. **Derived, not measured:** a
  materialized 975x975 attention matrix at 6 heads and batch 16 is about 91 M
  entries per layer, and r49's throughput was measured through
  `nn.MultiheadAttention`, so its figures describe the materializing path. Whether
  SDPA is faster here, and by how much, is **unmeasured**. Take the measurement on
  the first run rather than assuming either sign. **RoPE's own cost is unmeasured
  too**: the probe's RoPE variant is `pos=None`, no position signal at all, which
  is right for counting parameters and times no rotation.
- **Score the loss only at positions whose target is a frame token.** This is a
  next-token loss, so the position that predicts a token is the one before it.
  Under decision 2's layout that is action[t]'s position and the first 63
  positions of frame t; the last position of frame t predicts action[t+1] and is
  masked out, so **no target is ever an action token**. Scoring "the frame-token
  positions themselves" would drop one frame target and keep one action target
  per frame, and under r49's clamp action ids 512-520 would silently train as
  code 511. The action at inference comes
  from the operator - **F-14** is "control loop reads keyboard and drives the
  model with MuJoCo not running" - so a next-token loss over action positions
  optimises a distribution the engine never samples, and spends capacity that
  `runs.jsonl` r49 says is already 15x short of its data. r49's probe takes
  cross-entropy over every position with a clamp, which is correct for timing a
  matmul and is explicitly not the implementation.
- **Keep the causality claim asserted, not assumed.** A causal mask that is
  subtly wrong trains a model that reads the answer and then fails only at
  rollout, hours later. The measurement before item 1 selected block-causal
  (r54), so the assert is the per-block form under "Before item 1", not the
  strictly-causal one written below.

**Working when:** `python -m mirage.dynamics` self-checks with **no dataset and
no checkpoint**, the way `mirage.config`, `mirage.logging` and `mirage.fsq`
already do - it prints the parameter count and the first run's row records it,
and the count equals r49's `rope_untied` 14,593,152 exactly while `dynamics.py`
keeps the probe's modules, attention biases included. Any difference is a module
difference to name in that row, not to round away. Also: altering a token at
position `k` leaves every logit at positions `< k` bit-identical and changes at
least one at `k` - the strictly-causal form, with the block-causal one under
"Before item 1"; and the sequence assembled for one window round-trips to the
same token rows item 1 asserted.

### 4. The training loop, and the instrument for the risk that actually exists

**bf16, and this is not the Phase 1 question re-asked.** Phase 1 trains in fp32
because the tokenizer is under a million parameters and because changing the
arithmetic under a 0.087 dB comparison would make later rungs incomparable. Here
r49 measures **221.6 ms/step in bf16 against 622.4 in fp32 at batch 16, 2.81x**,
for **1.06 h/epoch against 2.99 h**. Nothing in Phase 2 rests on a sub-tenth-dB
comparison, and the architecture doc's own condition for bf16 - "the 15M-parameter
dynamics model at context 1024, where it is necessary" - is the model being built.
Read that 1024 as the round number it is: r49 priced 975 positions, item 1
derives 1,040 for the sampler's window, and 1024 has never been a measurement of
anything here.

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

**r49 prices the shortfall and decides nothing about it; decision 4, taken
2026-09-21, is that the first run measures the train/val gap before any remedy
is chosen.** Picking one of the three - more data, a smaller model, heavier
regularisation - before the gap is measured would be choosing a remedy for a
magnitude nobody has yet seen. They are not equal in what they cost, and the
difference is recorded now so it is not rediscovered then: **more data is the
remedy entangled with `data_hash` provenance**, since regenerating the set moves
the hash and orphans both the checkpoint and the token cache - `fsq_eval.load_run`
refuses R1 at a moved hash, and the manifest carries the old one - while a
smaller model and heavier regularisation are not.

**The first run's stopping rule - decision 4a, taken 2026-09-21: an epoch cap or
a divergence trip, whichever fires first.**

- **The cap is 10 epochs.** At r49's 1.06 h/epoch - the figure r49 calls
  pessimistic, because its throughput is a lower bound, which makes its time an
  upper bound at 975 positions a window only - that is one overnight window at
  r49's 975-position figure. Derived, not measured: 10.6 h of training steps at
  975 positions a window, before the per-epoch held-out pass, which r49 did not
  time. At the 1,040 positions item 1 recommends the epoch time is unmeasured,
  and r49's figure does not bound it.
- **The trip is held-out loss rising for two consecutive epochs.** When it
  fires, the run continues a further two epochs and then stops, because
  decision 4 wants the **magnitude** of the gap and not merely where it turns.
  Whichever fires first still governs, so the cap bounds those two epochs too.
- **Stopping is a decision point, not the end of the run.** The per-epoch
  resumable checkpoint and `--resume` by run id, below, mean a run stopped by
  either rule continues from its last epoch if the gap it recorded asks for
  more. Neither rule has to be right the first time; it has to be recorded.

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

**P-7 is scored against a 300,000-frame equivalent - decision 8, taken
2026-09-21.** P-7 - "full 300k-frame epoch <= 30 min", tier S - names a
300,000-frame pass, while a dynamics epoch as r49 defines it is 276,705
**overlapping windows**, 13.8x the dataset in tokens. Those are two different
objects, and the decision is which one P-7 is read against. **It is a
scoring-method decision only, and no bar moves**: `NUM-BAR-P7` stays where it
is, and moving a bar because a run missed it is the failure mode this project's
discipline exists to prevent. What is timed is the train loop working through
the dataset's 300,000 frames once, and the row that measures it states how it
counts a frame inside a window, because that conversion is where a verdict could
be manufactured. **The equivalent figure is unmeasured.** It is deliberately not
derived from the windowed epoch's 1.06 h, which rests on a throughput lower
bound, leaves out the held-out pass, times no rotation and was priced at 975
positions a window. Scaling it would imply a verdict nobody measured.

**Working when:** a one-epoch run writes a resumable checkpoint and a jsonl
carrying `dynamics_hash`; `--resume` continues it with no visible discontinuity
in the loss curve; the val loss is in the log from epoch 1; the stopping rule
stops where decision 4a says, and the run's row records which rule fired and at
which epoch; and peak VRAM and the GPU clock state are recorded next to the step
time.

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

**Greedy - decision 5, taken 2026-09-21.** The simulator is deterministic -
**E-1**, "deterministic sim given a seed", defined as same seed and action
sequence give bit-identical frames - so the target distribution is a point mass
and greedy decoding is the strong default. It also makes a rollout exactly
reproducible from a checkpoint plus a seed clip, which is what lets E-4 compare
two runs at all, and it keeps gate row 9 an exact-reproduction row rather than a
statistical one. **The revisit trigger stays in place:** evidence that greedy
decoding collapses - a rollout that freezes or falls into a short loop, not one
that merely drifts. That is what would reopen temperature sampling.

**F-13 is why decision 2 took RoPE**, and it is worth stating here because F-13
is scored from one checkpoint. With learned positions a shorter context has to
index the same absolute position table, and nothing past the trained length
exists at all; with RoPE both directions are free. A rollout at `ctx` 15 grows
to the same `ctx + 1` frames item 1 recommends training on, so the longest
rollout sequence is one the model has trained at.

### The proposed gate - one command

`python -m mirage.dynamics --eval RUN_ID` prints the table and exits nonzero when
a pass/fail row misses, mirroring `python -m mirage.fsq --eval`, which is what
makes a gate usable from a script rather than only by reading it.

**These rows are proposals against the requirements that already exist, not new
requirements, and none of them moves a bar.** Row 1 is written against F-11 as
`world_model_requirements.md` restated it on 2026-09-22.

| # | Measure | Bar | What the requirement actually says |
|---|---|---|---|
| 1 | Held-out next-token accuracy against the **persistence baseline** - copying the previous frame's token at the same cell - on the same held-out population as the model, with the marginal-frequency baseline reported alongside | **above the persistence baseline, re-measured with `bench/token_stability_probe.py` on the model's population.** r46 reads **85.67%** over its 12 val episodes on R1; that figure is the bar only on that population. Quote r46, do not restate it | **F-11** - "dynamics model consumes interleaved frame and action tokens, predicts next token", accepted when held-out accuracy beats the persistence baseline scored like-for-like on the same population, with the marginal-frequency baseline reported alongside. **Restated 2026-09-22** from "beats marginal-frequency baseline by 3x". Its description was amended 2026-09-23 on r54 - see "Before item 1". Under block-causal the teacher-forced and generated accuracies are one number |
| 2 | One full next frame from `ctx` frames plus one action, fixed step count | **exact**, no fallback path | **F-12** - as quoted above. All 64 positions decoded every step |
| 3 | Rollout at `ctx` 4, 8 and 15 from one checkpoint | **runs** | **F-13** (S) - "configurable context length at load time". Item 2's gotcha is why this is an argument and not a config edit |
| 4 | Frames until the frame-to-frame continuity verdict fires | **>= 200** | **Q-3** - the coherence horizon. The verdict bounds per-step change in `link_angle`, `link_extent` and each block's `bbox` centroid, calibrated so that **zero windows of ground-truth frames fire** - the same acceptance shape **F-9** uses, F-9 being "frame validator reports block count, arm pose plausibility, palette adherence", accepted at zero false positives on ground-truth frames. Calibrate on **reconstructions, not renders**; `bench/q3_blind_probe.py` is the regression test and must fire on 100% of 300-step substitutions and 0% of clean reconstructions |
| 5 | Action-following agreement, and the simulator's own agreement on the same subset | **>= 90% of the ground-truth term, both numbers reported** | **Q-4** - agreement is `sign(theta_t+1 - theta_t)` against the commanded sign, on an **action-balanced** subset drawn from the val split. An absolute bar there fails a model that is exactly right, which is why the bar is relative. `bench/hold_probe.py` measures the ground-truth term - **re-measure it, see the gotcha** |
| 6 | Link-length drift over a 200-step rollout, per link, and the simulator's own drift | **<= 1.1x the ground-truth term, per link, both numbers reported** | **Q-5** - the statistic is the pixel-measured major extent's `(max - min) / median` over non-overlapping 200-frame windows. Ground truth reads 23.0% on link0 and 44.2% on link1, so a perfect model fails any absolute bar. `bench/link_drift_probe.py` measures the ground-truth term. **Do not re-attempt the deprojection** - r50 records it as measured and refuted |
| 7 | Block reappears in the correct position after full occlusion | **>= 80% of events** (S) | **Q-6** - object permanence, the memory result. Tier S: the project ships without it and the negative result gets reported either way. `mirage.data.seen_later` owns the recoverable-occlusion split, and `NUM-DATA-F7` is the event rate it scores over |
| 8 | Parameter count, and peak training VRAM | **<= 20M bf16**, **<= 7.5 GB** | **R-3** and **R-1**. r49 settles R-3: 14,593,152 for the chosen variant, 14.4-15.0 M across all four. R-1 is **unmeasured** for this model and item 4 takes it |
| 9 | Rollout reproduced from the checkpoint plus the seed clip | **identical** | **E-1** and **E-4** - a rerun matching within `NUM-BAR-E4`. Greedy decoding, decision 5, makes this an exact-reproduction row rather than a statistical one; only item 6's revisit trigger would change its shape |
| 10 | Train-val loss gap; share of predictions the copy baseline also gets right | **reported** | not requirements - the overfitting and triviality canaries. The first is item 4's headline instrument; the second is what keeps row 1 honest now that its bar *is* a baseline - a model that clears the bar while agreeing with the copy baseline almost everywhere is winning on the cells the baseline already gets right, and the overlap is what says so |

Rows 1 to 6 and 8 to 9 are the pass/fail candidates; 7 is S-tier and reported;
10 is reported. Row 1's bar is F-11's restated one, and it is **materially
harder than the requirement originally promised**: r46 records the
zero-parameter copy baseline as far above 3x the marginal top-1, so a model that
predicts every cell unchanged passed the old wording and fails this one. **That
comparison is asserted, not measured** - neither r46 nor
`bench/token_stability_probe.py` computes the marginal top-1 - so row 1's
marginal-frequency column is where it first gets a number.

---

## Decisions: eight taken, one on its trigger

Each one changes an item above, so each is named rather than quietly resolved.
Decision 1 was taken 2026-09-18, decision 9 on 2026-09-23 by measurement, and
decision 7 on 2026-09-23.
The other five were taken 2026-09-21, after a walkthrough of this plan's draft,
and each is written down with its rationale.

**1. F-11's acceptance test - DECIDED 2026-09-18: restated against the
persistence baseline, and `world_model_requirements.md` carries it as of
2026-09-22.** F-11 had asked the dynamics model to beat a marginal-frequency
baseline by 3x. `runs.jsonl` r46 recorded, for Phase 2 and deliberately not
acted on, that **copying the previous frame's token at the same cell scores
85.67%** on the R1 checkpoint Phase 2 inherits, over 460,032 held-out
cell-transitions from 12 val episodes, **at zero parameters** - and records that
as far above 3x the marginal top-1, which is asserted rather than measured. That
baseline is now the bar, and gate row 1 is written against it.

**The rationale, recorded because the bar moved on it: an acceptance test a
zero-parameter baseline already passes cannot show the model learned dynamics.**
Beating the marginal frequency by 3x asks only that the model know which codes
are common; beating persistence asks it to know when a token *changes*, which is
the only part of the sequence that carries the physics. Note what this costs:
it is a **materially harder bar than the requirements document originally
promised**. A model that predicts every cell as unchanged scores the baseline
itself, and now fails.

**Three constraints on quoting it.** Quote r46 for the figure rather than
restating it from memory: r49 exists for that reason and says so in its own
words, recording sizing "computed in an earlier session and never written down"
so that a plan is written against numbers instead of recollections. **No `NUM-`
id is minted for it**, so r46 stays the citation until
registering one is done as its own deliberate act. And the comparison is
**like-for-like or it is nothing**: the requirement re-measures the baseline with
`bench/token_stability_probe.py` on the same held-out population the model is
scored on, so r46's 85.67% is the bar only when that population is r46's 12 val
episodes. Otherwise the row compares two different statistics and reports the
difference as skill. The baseline also belongs to the **tokenizer checkpoint**,
not to F-11 - r46 reads 85.67% on R1, 93.22% on r1c and 77.28% on R2 - so a
checkpoint promoted above R1 means re-running the probe, never reusing R1's
figure.

**Where it is written down.** `world_model_requirements.md` restated F-11 on
2026-09-22 with the dated paragraph under its Models table and an F-11 row in
"Requirements at risk", so the requirement and this plan agree; where their
wording differs, the requirement wins. The one piece still missing is the
verification-log row naming the acceptance procedure, which cannot be written
until gate row 1 exists in code. **No Phase 2 F-11 verdict exists, and none is
quoted here.**

**2. Sequence layout and position encoding - DECIDED 2026-09-21: RoPE, and the
no-shift interleaving.** The irreversible one. **RoPE**, because it keeps
F-13's shorter-context rollout free in both directions, and because r49 shows
the four variants span under 4% of parameters, so this is not a capacity trade.
**The interleaving is the no-shift reading of the recorded alignment**: the one
action token that conditions a frame immediately precedes that frame's 64 token
positions, and both are read at the same record index. That is the convention
`mirage/data.py`'s module header already states - `qpos[t] - qpos[t-1]` is the
result of `action[t]`, same record - and it is consistent with r49's 65
positions per frame step. **The phase assertion is mandatory and is item 1's
acceptance test**: all 13,242 action changes must sit at
`step_idx % sim.action_hold_steps == 0`, and a deliberately shifted copy must
fail that same assertion. **Agreement statistics must not be used to confirm
the alignment**, because the wrong reading scores higher - 95.6% against
93.9% - and both clear Q-4's bar. Two things this decision does not settle:
the window's frame count, which item 1 names, and the mask, which decision 9's
measurement selects and which may change how a frame's positions are fed but not
which record they are read from.

**Item 1's window index arithmetic - DECIDED 2026-09-21: shared, not copied.**
The plan's own recommendation and the project's precedent. The shared
addressing and `WindowSampler` must agree on every window's (episode, offset).
It edits `mirage/data.py`, whose `_self_check` is F-8's acceptance test, and the
cost is believed small because the edit moves indexing and nothing F-8 checks.
Item 1 has the detail.

**3. Output embedding - DECIDED 2026-09-21: untied, two separate matrices.**
Tying is only sound because the frame codes occupy the first block of the
vocabulary with the action tokens appended after - `bench/dyn_size_probe.py`
says so in its own comment - and that is a hidden coupling to the layout. The
parameter cost stays inside the same sub-4% spread, and R-3 has headroom.
**The count is on record**: r49 prices all four variants, and RoPE-and-untied
is its `rope_untied`, **14,593,152**. An earlier draft of this plan quoted only
the two endpoints, 14,396,544 for RoPE-and-tied and 14,967,552 for
learned-and-untied, which made the chosen variant look unpriced. The first run
still reports its own count, per item 3's done-when.

**4. The answer to the Chinchilla shortfall - DECIDED 2026-09-21: measure the
gap first.** Unchanged from item 4's position: the first run measures the
train/val gap before any remedy is chosen. Recorded alongside it: **more data
is the remedy entangled with `data_hash` provenance**, since regenerating the
set moves the hash and orphans both the checkpoint and the token cache, while a
smaller model and heavier regularisation are not.

**4a. The first run's stopping rule - DECIDED 2026-09-21**, the part decision 4
otherwise leaves open. An epoch cap or a divergence trip, whichever fires first:
the cap is 10 epochs, one overnight window at r49's 1.06 h/epoch at 975
positions a window (the time at 1,040 is unmeasured); the trip is
held-out loss rising for two consecutive epochs, after which the run continues a
further two epochs before stopping, because decision 4 wants the magnitude of
the gap and not merely the location of the turn. Item 4 states it in full,
beside the per-epoch resumable checkpoint and `--resume` by run id that make
stopping a decision point rather than the end of the run.

**5. Rollout decoding - DECIDED 2026-09-21: greedy.** The revisit trigger in
item 6 stays in place: a rollout that freezes or falls into a short loop, not
one that merely drifts. Greedy keeps gate row 9 an exact-reproduction row rather
than a statistical one.

**6. Whether `dynamics_eval.py` splits out - governed by its trigger, not taken
now.** The same 500-line trigger that split `fsq_eval.py` out of `fsq.py`,
applied when it fires and not before.

**7. Exposure bias - DECIDED 2026-09-23: name a trigger, build no mitigation
now.** Teacher forcing trains on ground-truth context and the rollout feeds the
model its own output. Under decision 9's block-causal mask a frame's 64 tokens
come out of one pass from earlier frames only, so the gap applies across frames,
not within one. Item 4's first real training run trains with no exposure-bias
mitigation, which keeps its train/val gap measurement clean. **Trigger:** gate
row 4's coherence horizon (Q-3, frames until the continuity verdict fires) comes
in under 100 frames while gate row 1 passes (held-out accuracy beats the
persistence baseline). **If it fires**, the first remedy is context-token
corruption during training (randomly replacing a small share of context tokens),
and scheduled sampling is the second. The accepted cost is one retrain.

**8. What P-7 means for a windowed epoch - DECIDED 2026-09-21: scored against a
300,000-frame equivalent.** A scoring-method decision only; `NUM-BAR-P7` does
not move. The equivalent figure is **unmeasured**, and item 4 says why it is
not derived from the windowed epoch's 1.06 h.

**9. Strictly-causal against block-causal attention - DECIDED 2026-09-23 by the
measurement ordered 2026-09-22: block-causal**, taken by it rather than by
argument - `runs.jsonl` r54, under a rule fixed in code before the first step.
"Before item 1" says what it changes and the trigger that would reopen it.
F-11's description carries the second, separately dated amendment.

---

## Gotchas, and how you would notice

| Gotcha | What breaks | How you notice |
|---|---|---|
| **A one-step action misalignment** | Every checkpoint conditions each frame on the wrong action, and Q-4 scores the wrong thing | **You do not**, from agreement: 93.9% against 95.6%, and the **wrong** reading scores higher, with both clearing Q-4's bar. Only the phase assert catches it - all 13,242 action changes sit at `step_idx % action_hold_steps == 0`. Assert it, and assert that a shift of one breaks it |
| **Scoring gate row 1 against r46's 85.67% on another population, or on another tokenizer** | The row compares two statistics and reports the difference as skill | The row names episodes other than r46's 12 val ones, or a checkpoint other than R1. Re-measure the baseline with `bench/token_stability_probe.py` on the model's own population and checkpoint; r46's figure is the bar only on its own |
| **Quoting r49's 975 positions as the training sequence** | Tokens per epoch, step time and epoch time are all understated, and a schedule built on them runs long | r49 prices `ctx x 65`, while `WindowSampler` holds `ctx + 1` frames - 16 at `ctx` 15 - and r49's own window count is the sampler's. **Under the block-causal mask r54 selected, a 16-frame window is 975 input positions**, so the gap is the strictly-causal case's. Re-take the timings at the sequence actually built |
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
| Sequence | 15 frames x (64 + 1) = **975** positions, vocab **521** in / **512** out - r49 | the 65 positions per frame step decision 2 lays out. **975 is `ctx x 65`, and the sampler's window is `ctx + 1` frames** - item 1 - so read it as r49's pricing, not as the training sequence |
| Parameters | **14,396,544** RoPE + tied, **14,593,152** RoPE + untied, **14,770,944** learned + tied, **14,967,552** learned + untied; spread **571,008** - r49 | **the irreversible layout choice is not a capacity choice**, and R-3 passes at all four. Decisions 2 and 3 take **RoPE + untied** |
| Step and epoch cost | bf16 **221.6 ms/step** at batch 16, **1.06 h/epoch**; fp32 **622.4 ms** and **2.99 h** - r49 | bf16 for item 4, and decision 4a's cap. **A lower bound on throughput, so an upper bound on time at 975 positions a window only**, and not a bound at item 1's 1,040: `gpu_probe` returned compute FAIL, 2385 of 3090 MHz and 20.6 TFLOP/s against `NUM-HW-FP16`. Timed on the RoPE + untied parameter layout with no rotation, through `nn.MultiheadAttention`, at 975 positions a window |
| Data against capacity | **19.5 M** cache tokens against a Chinchilla-optimal **291.9 M**, **15.0x under**; one epoch draws **276,705** windows totalling **269.8 M** tokens, **13.8x** the dataset from overlap - r49 | **the phase's risk is overfitting, not throughput**. Decision 4: the first run measures the gap before a remedy is chosen. The epoch's token figures are priced at 975 positions a window - item 1 |
| Token cache size | **38.4 MB** - r49 | the whole cache fits in VRAM many times over; nothing about the data path needs engineering |
| The inherited checkpoint | `20260829-005439-r1`, `NUM-TOK-R1-60` held-out PSNR at `NUM-TOK-ENT-R1` token entropy | the tokenizer is fixed, and so are the 512 codes and the 64-token grid |
| The zero-parameter baseline | **85.67%** token persistence on R1, over **460,032** transitions from 12 val episodes, **396,013** of them quiet-field - r46 | **F-11's restated bar on r46's population**, and gate row 1. Quote r46 for it; on any other population, re-measure with `bench/token_stability_probe.py` |
| Token instability on R1 | **8.86%** of transitions flip with no change in the cell's own 15x15 field, **53.21%** of all flips - r46 | context for a rollout that looks noisy. r46 is explicit that **nothing has measured what spurious flips cost a dynamics model**, in either direction, and that Phase 2 evidence either way is a recorded trigger to reopen the tokenizer choice |
| The split | `NUM-DATA-SPLIT` train/val episodes, `NUM-DATA-VALFRAMES` held-out frames of `NUM-DATA-FRAMES` total, at `NUM-DATA-HASH64` | reuse `data.is_val`; the tokenizer's val set and Phase 2's are the same set by construction |
| Action alignment | same-record **93.9%** against next-record **95.6%**, and all **13,242** action changes at phase 0 - verification log, the alignment row | item 1. The agreement figures are there to prove agreement cannot settle it |
| ctx=15 window coverage | **60.9%** of windows carry any evidence of what an action does - r20, at the shipped physics (`gear 6 / damping 1.5`, `action_hold_steps` 15). r23 re-ran the gate at `NUM-DATA-HASH64` with the scene and the policy untouched and found every M-tier row unchanged, so an r20 figure still describes the set on disk - but **window coverage is not one of the rows r23 re-measured** | about 39% of training windows contain no action change at all, which is what an action-balanced Q-4 subset is drawn against |
| Q-4's ground-truth term | **91.5%**, relative bar **82.3%** - r20's after-values at the shipped physics. **`NUM-DATA-Q4CEIL` disagrees**, see the gotcha | re-measure with `bench/hold_probe.py` and report both numbers |
| Q-5's ground-truth term | **23.0%** on link0 and **44.2%** on link1 - r47, quoted in Q-5's requirement row | a perfect model fails any absolute bar; `bench/link_drift_probe.py` produces the term |
| Q-3's terminator | F-9 fires on **0.00%** of 300-step substitutions against **100%** on the noise control - r48 | the continuity verdict replaces it, and the probe stays as its regression test |
| Run-to-run noise | `NUM-PERF-NOISE`, and it is the **tokenizer's** 1-epoch figure | **unmeasured for a dynamics rung.** Do not call a margin "inside the noise" here; no seed has been repeated on this model |
| Peak training VRAM | **unmeasured** for this model - r49 recorded none | item 4 takes it, at the batch actually used. Batch 16 is the probe's choice, not an optimum |
| SDPA against materialized attention, and RoPE's rotation | **unmeasured** - r49 timed `nn.MultiheadAttention` and applied no rotation | item 3 measures both rather than assuming a sign |
| P-7's 300,000-frame equivalent | **unmeasured** | decision 8 fixes how P-7 is scored, not what it reads. Not derived from r49's windowed epoch |
| Rollout throughput, and any P-row | **unmeasured**, and deliberately - **P-1** to **P-5** are the interactive rows (sustained frame rate, p99 frame time, input-to-display latency, the p99/p50 jitter ratio, and the eager-to-engine speedup), which belong to Phase 3's baseline and Phase 4's ladder | Phase 2 produces a checkpoint, not a frame rate. `NUM-BAR-P7` is the only P-row this phase touches, and item 4 says how it is scored |

Record the GPU power state next to every timing. A timing without it is not a
number - and gate compute numbers on **SM clock plus power draw**, bandwidth on
**memory clock == max**.

---

## What this plan does not do

No part of `mirage/dynamics.py` is written here, no run was launched, and no
measurement was taken - there is no dataset and no checkpoint on the machine this
was written on, so every number above is quoted from the record rather than
earned here. **No `NUM-` id is minted**: registering a number is a separate,
deliberate act, and until then r46, r49 and r54 are what to quote. **No bar is
moved here.** F-11's acceptance test was raised, not lowered, and
`world_model_requirements.md` restated it on 2026-09-22; gate row 1 is written
against that. Moving a bar *down* because a run missed it is the failure mode
this project's discipline exists to prevent, and nothing here does that - the
one run since, r54, selected a mask and moved no bar. **The decisions above are
recorded here, not taken here**: they were taken 2026-09-18, 2026-09-21 and
2026-09-23, the last by r54's measurement, and the one new call this plan makes -
`ctx + 1` frames a window, in item 1 - is written as a recommendation with its
alternative. Phases 3 and 4 stay undrafted, which is "profile before changing
anything" applied to planning.
