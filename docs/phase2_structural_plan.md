# Phase 2: Structural Plan

Guidance for implementing Phase 2 - the dynamics model, `mirage/dynamics.py` in
the architecture doc's file map - one item at a time. It names the calls and the
order, and does not write the code. The design reasons are in
`world_model_architecture.md` and are not repeated here.

The numbering below is this file's own. `AGENDA.md` opens Phase 2 and points here
rather than restating the list, because AGENDA's rule is to stay short and carry
no history. If the two ever disagree, AGENDA is the order of record and this file
is the stale one.

**What gated this plan, and no longer does.**
`world_model_architecture.md`, "Which later-phase plans can be drafted now",
rates Phase 2 *structure yes, numbers no*, gated on the 64/144 fork - a *Phase 1*
result. That fork resolved to **64x64** (the 96x96 tokenizer run, then the
arithmetic that closed the fork), the checkpoint is chosen, and
**`bench/dyn_size_probe.py` priced the model the docs already specify** (its row
in `runs.jsonl` is "the sizing probe" below). The numbers exist now, so this file
quotes them instead of predicting them.

**What Phase 2 inherits, all settled - do not reopen.** The R1 checkpoint
`20260829-005439-r1`, the fixed 512-code budget, and the 64-token path. The
trigger that would reverse this is in `world_model_architecture.md` under "Phase 2
inherits R1, and the encoder keeps `GroupNorm`", and it **only works one way**:
another rung may promote itself above R1 by passing every gate row with fewer
spurious token flips, and nothing demotes R1. Nothing in this file fires it.

**Vocabulary, once.**

- The *dynamics model* is a decoder-only transformer. Given the last `ctx` frames
  as grids of token ids, plus the action taken at each, it predicts the next
  frame's tokens. It never sees a pixel.
- *Autoregressive* means each position predicts the token at the next position.
  In training, every position is scored at once against the true next token; that
  is *teacher forcing*. In a *rollout*, the model's own output is fed back
  instead. The gap between those two regimes is *exposure bias*.
- *Interleaved* means one sequence carries both frame tokens and action tokens,
  so an action is just another token in the stream.
- A *causal mask* stops a position attending to a later one; without it the model
  reads the answer. *Strictly causal* lets each position see only the positions
  before it. *Block-causal* also lets the positions of one frame see each other,
  so a frame's tokens are predicted together rather than one at a time.
- *Position encoding* is how a token learns where it sits. *Learned positions*
  are one trainable vector per absolute position, capped at the longest sequence
  trained on. *RoPE* rotates each query and key by an angle proportional to
  position, has no parameters, and has no cap.
- A *tied output embedding* reuses the input embedding matrix as the output
  projection instead of training a second one.
- *Chinchilla-optimal* is the rule of thumb of about 20 training tokens per
  parameter. Well below it a model is **data-bound**, and it fails by memorising
  rather than by underfitting.
- The *KV cache* belongs to Phases 3 and 4 and is not built here.

---

## The dataflow, in one line

R1's token cache on disk (`runs/20260829-005439-r1/tokens/shard_NNN.npy` plus its
`manifest.json`) and the shard meta records' `action` column -> one window of
`ctx` frames interleaved with their actions -> `Dynamics` (embed, pre-norm
blocks, causal mask) -> logits over the 512 codes at every position whose target
is a frame token -> cross-entropy. After training: roll out autoregressively from
a seed clip, decode each predicted token grid through R1's decoder, and score the
rollout on the decoded frames.

Everything flows one direction. Nothing calls backwards.

**Note what is *not* in that line: pixels.** Phase 2 trains on 38.4 MB of
`uint16`, so nothing about the data path needs engineering - the whole cache fits
in VRAM many times over. Phase 1's problem, a 3.5 GB working set against the page
cache, does not come back, and `mirage.data.preload` is not on Phase 2's training
path at all. It returns only for eval, where decoded frames are compared against
ground-truth ones.

---

## What each file owns

| File | Owns | Explicitly does not own |
|---|---|---|
| `mirage/dynamics.py` | the sequence layout, the token/action window sampler, the model, the train loop | pixels, the tokenizer, and *writing* the token cache - that is `fsq_eval.write_token_cache`, and Phase 2 only reads it |
| `mirage/dynamics_eval.py` | rollout and the gate table - everything that runs against a finished checkpoint. **The 500-line trigger fired at item 3**, the same trigger that split `fsq_eval.py` out of `fsq.py`, so item 6 starts here | training anything |
| `mirage/configs/base.json`, `dynamics` section | the shape knobs that must sit inside `dynamics_hash` | the training knobs, which travel in the checkpoint's `knobs` dict as Phase 1's do |
| `mirage/config.py` | the `dynamics` key set and its validators, and the `validator` section's continuity bounds for gate row 4 | anything model-shaped. It gained keys in item 2, and the four `continuity_*` keys in item 6 (decided 2026-09-25), which move `validator_hash` and nothing else |
| `mirage/fsq.py` | the inverse of `FSQ.codes_to_indices`, `FSQ.indices_to_codes`, and `Tokenizer.decode` over it (item 5, landed 2026-09-25) | the rollout. The token-to-pixel path belongs beside the pixel-to-token path, not in a second copy |
| `mirage/data.py` | the window index arithmetic, shared with `WindowSampler` by the decision in item 1 | anything token-shaped. It reads the shard format and knows nothing about codes |
| `mirage/validator.py` | the per-frame measurements the coherence horizon's continuity check is built from | the verdict itself. Phase 0's rule stands: the validator emits measurements, and the verdict is a threshold expression in config |
| `runs.jsonl` and the verification log | one row per run, and one verification row per claim | - |

`dynamics.py` reads the token cache and never re-encodes. Re-encoding would put a
second copy of the encode path in the tree, and the cache already carries the
provenance that makes the tokens checkable: `manifest.json` records `run_id`,
`checkpoint`, `tokenizer_hash`, `data_hash`, `levels`, `token_grid`, the
per-shard frame counts and sha256s, and the `batch` the cache was written at.

**Check those the way `fsq_eval.load_run` checks a checkpoint.** It raises when
the checkpoint's `data_hash` disagrees with the config, rather than loading
anyway. The dynamics loader needs the same refusal against the manifest's
`data_hash`, `tokenizer_hash` and `token_grid`. Without it, a cache written at a
superseded `data_hash` trains silently, and one from the 96x96 fork - 144 tokens
per frame, not 64 - fails somewhere after the load rather than at it.

---

## Build order, with the calls each file needs

Riskiest first. In Phases 0 and 1 that meant "the thing that could invalidate
300,000 frames". **Here nothing can**: the dataset and the tokenizer are both
frozen, and `data_hash` and `tokenizer_hash` sit above every hash Phase 2 moves.
The risk instead is **irreversibility**. The sequence layout is baked into every
checkpoint trained under it, into the engine's decode schedule in Phase 4, and
into what an action-following verdict means. So the order is: irreversible
and silent first, expensive second, reported-only last.

### Before item 1: strictly-causal against block-causal attention

**Ordered 2026-09-22 to run before item 1, and it ran: block-causal is
selected** (`bench/mask_probe.py`, 2026-09-23; its row in `runs.jsonl` is "the
mask measurement" below). It went ahead of item 1 for item 1's own reason - the
mask is baked into every checkpoint trained under it - and because it decided
whether the dynamics requirement's description still holds. It does not:
`world_model_requirements.md` carries the second, separately dated amendment the
requirement called for, and the 2026-09-22 restatement of its acceptance test is
untouched.

**What the mask measurement found, in one paragraph. For the figures, go to its
`runs.jsonl` row, not this summary.** Both arms were fed one token stream and
scored on the next frame each *generated* from ground-truth context.
Block-causal led by 2.40 points against a seed spread of 0.20. Strictly-causal
reads the true earlier cells of its own target frame under teacher forcing, but
its own guesses during a rollout, and the gap between those two readings of one
checkpoint is larger than its whole deficit to block-causal. So **gate row 1's
measure, taken as teacher-forced accuracy, would have picked the wrong arm**.
Under block-causal the two readings are one number. Neither arm beat persistence
at the measurement's one-epoch endpoint, which is past both arms' held-out
minimum. **Overfitting is not the whole story (2026-09-24, `runs.jsonl` r59).**
Under block-causal the held-out curve's last-frame accuracy is the generated
score, and re-read against persistence on the curve's own 512 windows, neither
block-causal seed was above it at any of 18 checkpoints - its best-accuracy point
(step 1,000) and its lowest-loss point (step 10,000) included. It is still not a
verdict on the dynamics requirement: the probe is a one-epoch model, not Phase
2's, and item 4's instrument is amended to measure this directly. **The recorded
trigger to reopen the mask:** a generated-frame comparison at each arm's held-out
optimum, on the same population, with strictly-causal ahead by more than the seed
spread. The record holds half of it: block-causal's generated score at its
optimum is on its curve (r59), strictly-causal's is not, because strict's curve
accuracy is teacher-forced.

**The items below were written for strictly-causal attention**, which is what the
sizing probe timed: `bench/dyn_size_probe.py` builds its mask as `torch.triu`
over all 975 positions. The block-causal result changes four things, listed here
so they are in one place, and **these govern wherever an item below says
otherwise**:

- **Item 3's causality assert** moves from per position to per block, a block
  being frame `t`'s 64 tokens plus `action[t+1]`. Altering a token in block `b`
  leaves every logit in the blocks before `b` identical, and may change any logit
  inside block `b`.
- **Item 3's loss and item 6's decode step**: a frame's 64 predictions come out
  together rather than one position at a time. The fixed step count and "all 64
  positions every step" of the full-frame requirement hold under either mask.
- **How a frame's positions are fed in training.** A frame cannot be both the
  input and the target of its own block. The mask measurement's arrangement keeps
  decision 2's stream exactly and only regroups it: blocks of 65 - frame t's 64
  tokens plus the action after them, `action[t+1]` - with full attention inside a
  block and causal attention across blocks. Frame t+1's cell i is read at the
  position holding frame t's cell i. `bench/mask_probe.py`'s `layout` is that
  arrangement, and its `--self-check` asserts it. Decision 2's alignment - both
  tokens read at the same record index, and the phase assertion - holds under
  either mask.
- **How long a training sequence is.** A `ctx + 1` window of 16 frames feeds 15
  blocks, **975 positions** - the length the sizing probe priced - because the
  last frame is only ever a target. Item 1's derived 1,040 is the strictly-causal
  figure.

The order said "at a fixed step budget" and no more. The mask measurement's row
states the budget, the score and the population: one epoch at batch 16, two seeds
per arm, per-cell accuracy of the generated next frame against persistence, over
every val window's last frame. The plan suggested gate row 1's measure, and it was
taken in that generated form, for the reason in the paragraph above.

### 1. The sequence layout, and the token/action window sampler

**Why first: it is irreversible, and getting it wrong is silent.** Every
checkpoint, every rollout and every gate number below is stated in terms of this
layout. The sizing probe exists so the choice is not made by accident. It prices
four variants of the model at **14,396,544 to 14,967,552 parameters, a spread of
571,008**, under 4%, so **the irreversible choice is not a capacity choice**. It
was made on its merits, as decision 2 below: **RoPE, and the no-shift
interleaving.**

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
It is a pure function of `episode_id`, which is what makes the tokenizer's val set
and Phase 2's the same set by construction rather than by coincidence: 473 train
and 27 val episodes, 16,200 held-out frames.

**The layout, by decision 2.** Each frame step is **65 positions** - 64 tokens
plus one action - in this order:

    action[t], then frame t's 64 tokens        both read at record index t

The one action token that conditions a frame sits immediately before that frame's
64 positions, and both come from the same record, so the layout needs **no index
shift against the record**. Under strictly-causal attention, every position of
frame `t` then sees `action[t]`. This is the recorded alignment, not a free
choice: `mirage/data.py`'s module header and the alignment row in the
verification log at the end of `world_model_architecture.md` both state it.
`sim/main.cpp` picks the action, writes `ctrl`, calls `mj_step`, then reads
truth, so within one record

    qpos[t] - qpos[t-1]  is the result of  action[t]      <- the same record

**Do not use agreement statistics to confirm it, because the wrong reading scores
higher.** An action is held for `sim.action_hold_steps` frames, so a one-step
shift leaves 14 of every 15 frames unchanged, and measured agreement is **93.9%
same-record against 95.6% next-record**. Shifting hands each delta the previous,
already-settled action instead of the fresh one whose early transient the
agreement measure cannot win, and both readings clear the action-following bar.
What does settle it is **phase**, and the phase assertion is **mandatory - it is
this item's acceptance test**. `Policy::step` redraws only when its hold expires,
so all **13,242** action changes sit at `step_idx % action_hold_steps == 0`, and
the negative control - shifting by one - puts them at phase 1.

**How many frames a window holds is the one layout number the sizing probe
priced two ways.** It prices a sequence at `ctx x 65` = 975 positions, but counts
its epoch in `WindowSampler` windows, and those hold **`ctx + 1`** frames: the
context plus the frame it predicts. `WindowSampler.__init__` sets
`self.window = ctx + 1`, and the probe's 276,705 train windows are that count.
Sharing the addressing (below) makes a Phase 2 window the sampler's window. Under
teacher forcing, its last frame is then the one predicted from a full `ctx` of
context, which is what the configurable-context requirement's rollout at 15 asks
for.

**Derived, not measured:** that sequence is 16 x 65 = **1,040** positions at
`ctx` 15, under strictly-causal attention. **Block-causal, now selected, feeds
the same window as 975 positions**; see "Before item 1". Under RoPE the parameter
count does not depend on sequence length, so the probe's counts stand. Its tokens
per epoch, step time and epoch time were all priced at 975 positions a window. At
1,040 all three are higher by an amount the probe did not measure, so re-take
them on the first run. **Recommended: `ctx + 1`.** The alternative is to train on
`ctx`-frame windows and match the probe's pricing, at the cost of never training
the full-context prediction that requirement scores. It is named here rather than
taken silently, and whichever is built, the first run's row records it.

**The window index arithmetic is shared, not copied - DECIDED 2026-09-21.**
`data.WindowSampler` already owns episode-aware indexing: the cumulative start
positions, the split filter, the refusal of any episode shorter than the window,
and `__getitem__` as a pure function of the index, so that a shuffle and a
resumed run address the same window by the same number. Factor that addressing
into something `dynamics.py` can call, leave `WindowSampler` using it, and
**assert that the shared addressing and `WindowSampler` agree on every window's
(episode, offset)**. This follows the project's own precedent: `preload` lives in
`data.py` rather than `fsq.py` because "a copy in two files is the same class of
bug as two validator implementations", and `split_episodes` was factored out for
exactly this reason. The rejected alternative was a copy in `dynamics.py`,
cheaper to write and leaving `data.py` closed. Its failure mode is a cumulative
frame offset that is off by one - which the architecture doc already calls "an
off-by-one factory" in the one place it came up (and refused it: the token cache
is per-shard for exactly this reason).

**What sharing costs: it edits `mirage/data.py`.** That file is 702 lines, and
its `_self_check` is the acceptance test for the shard round-trip requirement
("shard writer emits packed frames and actions", accepted when the numpy
round trip matches the C++ buffer byte for byte). So any edit there is an edit to
a file a passed gate rests on. **The cost is believed small**, because the edit
moves indexing and nothing else. The meta decode that requirement checks, the
structured dtype and the row flip in `__getitem__` are all untouched, and
`_self_check` runs unchanged afterwards on both the generated set and the
fixture. If it does not pass unchanged, the edit reached further than indexing.

**Working when - and the first sentence is the acceptance test.** Every action
change in the action stream the windows are built from sits at
`step_idx % sim.action_hold_steps == 0` - over the generated set, the alignment
row's 13,242 of them - and a deliberately shifted copy **fails that same
assertion**. No agreement figure enters this test. A control that must read a
known value is the discipline the r1c rung's zero spurious-flip reading rests on,
and the kind that would have caught the `RF = 22` constant before an autograd
measurement had to. With no dataset, the same pair runs on the committed fixture,
whose 3 action changes `python -m mirage.data` already asserts at phase 0.

Also: the shared addressing and `WindowSampler` agree on every window's
(episode, offset); each window's token rows equal the cache rows for the frames
`WindowSampler` would have returned at the same index; no window straddles an
`episode_id` boundary; `len(tokens) == shard.frames` for every shard in the
manifest; the val episode set is `data.is_val`'s, compared against it rather than
reimplemented; and `python -m mirage.data` still passes.

### 2. `dynamics_hash` covers every shape knob, or E-4 and E-5 have a hole

Small, and it goes before the model because it decides where the model's shape
is written down.

Two engineering requirements are only met by construction if the hash names the
thing that changed. **Bench reproducibility** is "every bench number reproducible
from a config hash", accepted when a rerun matches within 5%. **The run log** is
"append-only run log: config hash, change, number, conclusion, one entry per
run". Until this item the `dynamics` section held `d_model` and `n_layers` and
nothing else, so **`n_heads` was in no hash at all**. It lived as `N_HEADS = 6` in
`bench/dyn_size_probe.py`, whose own comment said "not in config". The same was
true of what decisions 2 and 3 fixed - RoPE and the untied head - and of the mask
the measurement before item 1 selected. Each has to be named in the `dynamics`
section, or two checkpoints that differ in it log the same hash - as both arms of
the mask measurement did.

**Landed 2026-09-24.** The section now also holds `n_heads` 6 and `mlp_ratio` 4,
both positive ints, and `pos_encoding` `"rope"`, `output_head` `"untied"` and
`mask` `"block_causal"`, each checked against a closed set in `config.py` that
holds only the decided value - except `mask`, which also admits
`"strict_causal"` so the mask measurement's strict arm stays runnable. Admitting
any other alternative is a `config.py` edit, made when its decision's trigger
fires. Both bench probes build from `n_heads` and `mlp_ratio` in the config and
dropped their own constants. `bench/mask_probe.py` derives each arm's config
from `base.json` with `mask` set to the arm's value, loaded through
`config.load`, so the strict arm runs with no edit to `base.json`, each arm's
run logs the hash of its own mask, and `compare` accepts a run only under its own
arm's hash - which proves the arms differ in the mask alone. The move changed
`dynamics_hash`, so the sizing probe's and the mask measurement's `runs.jsonl`
rows carry the hash from before it.

**RoPE's base followed during item 3**, 2026-09-24: it had been a
`ROPE_BASE = 10_000.0` constant in `mirage/dynamics.py` and
`bench/mask_probe.py`, so no hash named it. It is now `rope_base` 10000.0 in the
`dynamics` section, a positive float, and both files build from it and refuse any
other value - the only base the mask measurement trained with. No behaviour
changed, but `dynamics_hash` moved again.

Two consequences, both verifiable by reading `mirage/config.py`:

- `EXPECTED_KEYS` is checked with `_check_keys`, which rejects **unknown** keys as
  well as missing ones. So a new `dynamics` knob is a `config.py` edit plus a JSON
  edit, never a JSON edit alone, and a count knob belongs in `POSITIVE_INT_KEYS`
  beside `d_model` and `n_layers`.
- `dynamics_hash = sha256(tokenizer_hash + canon(dynamics))` and
  `engine_hash = sha256(dynamics_hash + canon(engine))`, so a dynamics knob moves
  those two and **nothing upstream**. That is the hash tree working as designed:
  Phase 2 cannot invalidate the tokenizer or the data.

**Do not put the context length here.** `data.ctx` is inside the `data` section,
which feeds `data_hash`. Editing it moves `data_hash`, and then
`load_shards(dir, cfg.data_hash)` refuses the 300,000 frames on disk and
`fsq_eval.load_run` refuses the R1 checkpoint. **The configurable-context
requirement** - "configurable context
length at load time", accepted when one checkpoint rolls out at 4, 8 and 15
frames - must therefore be a **rollout argument**, not a config edit. This is
derived from the term order in `config.load`, and anyone can reproduce it by
hashing a `ctx`-varied copy of `base.json`. It is not a measurement, and no value
of the moved hash is quoted here.

**Working when:** editing a `dynamics` knob leaves `data_hash`, `tokenizer_hash`
and `validator_hash` unchanged and moves `dynamics_hash` and `engine_hash`;
dropping a `dynamics` key or adding an unknown one raises; and
`python -m mirage.config` passes. The first three are the same shape as the
assertions `config._self_check` already makes for a `tokenizer` change, so extend
that block rather than writing a new one.

### 3. The model

`d_model` 384, 8 layers, 6 heads, MLP ratio 4, pre-norm blocks, a causal mask,
vocab **521 in and 512 out** (512 codes plus the 9 actions in, codes only out),
and **15 frames x (64 + 1) = 975 positions**. All of these come from the sizing
probe, which measured the parameter count for each of the four layout variants
(item 1 has why a training window is 16 frames, not 15). **Decisions 2 and 3 take
RoPE with an untied output head, which the probe records as 14,593,152**
(`rope_untied`). That is the variant its Chinchilla figure is computed against,
and the parameter layout it timed. The parameter bar ("dynamics model
parameters <= 20M bf16") passes at all four, 14.4 to 15.0 M, and the
requirement's own fallback says never above 40M, so capacity is not the pressure
here. For comparison, the tokenizer has 744,966 parameters; the dynamics model is
about 19x that.

Three choices that are not stylistic:

- **`F.scaled_dot_product_attention` (SDPA), not `nn.MultiheadAttention`.** The
  architecture doc's seam note requires this on the 144-token path, because
  materialized attention caps the training batch size there. At 64 tokens the
  reason is the same in kind and smaller in degree. **Derived, not measured:** a
  materialized 975x975 attention matrix at 6 heads and batch 16 is about 91 M
  entries per layer, and the sizing probe measured its throughput through
  `nn.MultiheadAttention`, so its figures describe the materializing path.
  Whether SDPA is faster here, and by how much, is **unmeasured**. Take the
  measurement on the first run rather than assuming either sign. **RoPE's own
  cost is unmeasured too**: the probe's RoPE variant is `pos=None`, no position
  signal at all, which is right for counting parameters and times no rotation.
- **Score the loss only at positions whose target is a frame token.** This is a
  next-token loss, so the position that predicts a token is the one before it.
  Under decision 2's layout that is action[t]'s position and the first 63
  positions of frame t. The last position of frame t predicts action[t+1] and is
  masked out, so **no target is ever an action token**. Scoring "the frame-token
  positions themselves" would drop one frame target and keep one action target
  per frame, and under the probe's clamp, action ids 512-520 would silently train
  as code 511. At inference the action comes from the operator - the control-loop
  requirement is "control loop reads keyboard and drives the model with MuJoCo
  not running" - so a next-token loss over action positions optimises a
  distribution the engine never samples. It also spends capacity the sizing probe
  says is already 15x short of its data. The probe takes cross-entropy over every
  position with a clamp, which is correct for timing a matmul and is explicitly
  not the implementation.
- **Keep the causality claim asserted, not assumed.** A causal mask that is
  subtly wrong trains a model that reads the answer and then fails only at
  rollout, hours later. The measurement before item 1 selected block-causal, so
  the assert is the per-block form under "Before item 1", not the strictly-causal
  one written below.

**Landed 2026-09-24**, in `mirage/dynamics.py`: `Dynamics`, built by `build(cfg)`
from the `dynamics` section. `build` refuses any value the file does not
implement, including `strict_causal`, which `config.py` admits only for the mask
measurement's strict arm. It is the mask measurement's block-causal model, and
the self-check asserts so: the same parameters from the same seed and
bit-identical logits. Under block-causal, "action[t]'s position and the first 63
positions of frame t" becomes **the 64 frame-token positions of each block**,
because block `t`'s cell `i` predicts frame `t+1`'s cell `i`. The action slot
closing each block is the one position left unscored, so the loss is still 64
frame targets a frame step and no action target, and frame 0 is context only.
`frame_loss` reads logits only there, with no clamp. SDPA's speed against
materialized attention, and RoPE's own cost, stay unmeasured for the first run.
Item 3 moved `dynamics.py` past 500 lines; decision 6 says what that changes.

**Working when:** `python -m mirage.dynamics` self-checks with **no dataset and no
checkpoint**, the way `mirage.config`, `mirage.logging` and `mirage.fsq` already
do. It prints the parameter count, the first run's row records it, and the count
equals the probe's `rope_untied` 14,593,152 exactly as long as `dynamics.py` keeps
the probe's modules, attention biases included. Any difference is a module
difference to name in that row, not to round away. Also: altering a token at
position `k` leaves every logit at positions `< k` bit-identical and changes at
least one at `k` - the strictly-causal form, with the block-causal one under
"Before item 1"; and the sequence assembled for one window round-trips to the
same token rows item 1 asserted.

### 4. The training loop, and the instrument for the risk that actually exists

**bf16, and this is not the Phase 1 question asked again.** Phase 1 trains in
fp32 because the tokenizer is under a million parameters, and because changing
the arithmetic under a 0.087 dB comparison would make later rungs incomparable.
Here the sizing probe measures **221.6 ms/step in bf16 against 622.4 in fp32 at
batch 16, 2.81x**, for **1.06 h/epoch against 2.99 h** (the chosen model's own
bf16 step, timed since, is below). Nothing in Phase 2 rests on a sub-tenth-dB
comparison, and the architecture doc's own condition for bf16 -
"the 15M-parameter dynamics model at context 1024, where it is necessary" - is the
model being built. Read that 1024 as the round number it is: the probe priced 975
positions, item 1 derives 1,040 for the sampler's window, and 1024 has never been
a measurement of anything here.

**The sizing probe's throughput is a lower bound, and schedules now use the
chosen model's own timing.** `bench/gpu_probe.py` ran alongside the sizing probe
and returned **compute FAIL**: the SMs held 2385 MHz of a 3090 MHz maximum, and
fp16 matmul read 20.6 TFLOP/s against the 27.6 TFLOP/s this machine records when
cool on Windows (31.6 on Linux, r58). That run's own row records an **85 W**
enforced power limit (r49), not the 99.86 W measured after the cooling fix.
**The re-measurement this paragraph asked for exists (updated 2026-09-24).** The
mask measurement timed the chosen block-causal model itself - SDPA with RoPE
applied, 975 positions, batch 16, bf16 - at **159.6-160.1 ms/step**, with the
SMs at a median 2606-2617 MHz drawing 98.8-98.9 W of a 100 W limit, and one
epoch took 2,813-2,822 s including 18 curve evaluations, about **0.78 h/epoch**
(r54, on Linux). Gate any further re-measurement on **SM clock plus power
draw**, never on `pstate == P0`. The pstate follows the memory clock domain and
reads P4 during correct compute-bound work. **`bench/gpu_probe.py`'s compute
verdict is not that gate as it stands**: its clock-decay statistic counts a
sample taken before the load starts, so it fails a GPU that holds its clock
under load (-5.7% as the probe computes it, +1.1% over load samples only,
`runs.jsonl` r58). The enforced limit also moves with the platform: on Linux it
reads 85 W at idle and 100 W under load (r58).

**The risk to manage is overfitting, and it is measured rather than feared.** The
cache carries **19.2 M** frame tokens (`manifest.json`). The sizing probe counts
**19.5 M**, 300,000 frames x 65, because it adds one action token per frame, and
actions come from the meta records, not the cache. Against a Chinchilla-optimal
**291.9 M** for this parameter count that is **15.0x under** counting actions and
15.2x on frame tokens alone. One epoch draws **276,705
train windows totalling 269.8 M tokens**, which exceeds the dataset **13.8x from
window overlap alone**. That is repetition, not new information. **So the first
run's job is to find the train/val gap, not to close it.** Log held-out loss
every epoch from the first, and treat the gap as the phase's headline warning
sign, the way the tokenizer's train-val PSNR gap is row 8 of its gate.

**The sizing probe prices the shortfall and decides nothing about it. Decision 4,
taken 2026-09-21, is that the first run measures the train/val gap before any
remedy is chosen.** Picking one of the three remedies - more data, a smaller
model, heavier regularisation - before the gap is measured would be choosing a
remedy for a size nobody has seen yet. They do not cost the same, and the
difference is recorded now so it is not rediscovered later: **more data is the
remedy tangled up with `data_hash` provenance**. Regenerating the set moves the
hash and orphans both the checkpoint and the token cache - `fsq_eval.load_run`
refuses R1 at a moved hash, and the manifest carries the old one. A smaller model
and heavier regularisation do not.

**The first run's stopping rule - decision 4a, taken 2026-09-21: an epoch cap or
a divergence trip, whichever fires first.**

- **The cap is 10 epochs.** At the chosen model's measured 0.78 h/epoch (r54),
  which includes its 18 curve evaluations, that is one overnight window.
  Derived, not measured: about 7.8 h plus the per-epoch held-out passes.
  Under block-causal a 16-frame window is 975 positions, the length r54 timed.
  The sizing probe's 1.06 h/epoch, from a lower-bound throughput, was the
  figure this cap was first set against.
- **The trip is held-out loss rising for two consecutive epochs.** When it fires,
  the run continues for two more epochs and then stops, because decision 4 wants
  the **size** of the gap, not just where it turns. Whichever rule fires first
  still governs, so the cap bounds those two epochs too.
- **Stopping is a decision point, not the end of the run.** The per-epoch
  resumable checkpoint and `--resume` by run id (below) mean a run stopped by
  either rule continues from its last epoch if the gap it recorded asks for more.
  Neither rule has to be right the first time; it has to be recorded.

**The instrument also scores gate row 1's own measure - amended 2026-09-24,
before item 4 runs, by addition.** Decision 4a's per-epoch loss rule stands as
written and still decides when the run stops. What a per-epoch instrument cannot
do is see or keep an optimum inside an epoch. In the mask measurement, held-out
loss was lowest at step 10,000, 0.58 of one epoch, and last-frame accuracy peaked
at step 1,000, 0.06 of one. Re-read against persistence on the same windows
(`runs.jsonl` r59), neither block-causal seed was above persistence at any of its
18 curve points, the best-accuracy and lowest-loss points included. So a
checkpoint chosen by held-out loss is not the one gate row 1 rewards. Alongside
the loss rule, the first run:

1. **Scores gate row 1's measure during training**, not only held-out loss:
   held-out accuracy of the generated next frame. Under block-causal that is the
   last frame's accuracy from one forward pass, so each point costs one pass.
2. **Logs persistence on exactly the same windows beside each point.** The
   windows are fixed, so it is computed once. Without it a curve cannot say
   whether a point is above the bar, and r54's curve could not.
3. **Evaluates at sub-epoch intervals.** The mask measurement used every 1,000
   steps, about 2.7 minutes of training at its 160 ms/step.
4. **Keeps the best checkpoint by that measure**, separately from the per-epoch
   resumable checkpoint, and scores it on the full population against a baseline
   re-measured on that population. For every val window's last frame,
   `bench/token_stability_probe.py --episodes all --first-target 15` reads
   **86.69%** (r54, reproduced in r57). Never score it against 85.67%, which is
   the probe's default 12-episode population (r46).
5. **Reports the false-flip rate on static cells** - the share of cells whose
   token and own 15x15 pixel field did not change that the model predicts as
   changed - beside row 10's copy overlap. On r54's final block-causal seed 0
   checkpoint it read 6.4%, and those cells lost 51,946 cells against the 35,420
   its correct flips won back (r59).
6. **Plans for the answer being "no".** At its best checkpoint the mask
   measurement's model was 0.10 to 0.19 points below persistence on the curve's
   windows (r59). Decision 4 already chooses a remedy only once the gap is
   measured, and the gap has one measurement: validation minus train
   cross-entropy of 0.26 at one epoch (r54).

This moves no bar. Row 1's bar is still the persistence baseline on the model's
own population.

The operational shape, all of it taken from precedent:

- **A per-epoch resumable checkpoint plus `--resume`, from the first run.** At
  ~1 h/epoch, against 87.6 s for a tokenizer epoch, a run is hours long. Phase 1
  paid for this lesson twice: `--resume` carried the r1c rung across two
  mid-flight faults with no visible discontinuity, and before that it was broken
  on CUDA and had never once been executed. `--resume` takes a **run id, not a
  path**.
- **Keep the machine awake for the whole run.** `fsq._keep_awake()` is the
  per-process request that does it, and a dynamics run needs the same. Import it,
  or move it somewhere both can reach, rather than writing a second copy. Modern
  Standby froze one Phase 1 epoch for 49 minutes. The wall clock keeps counting
  through a suspend, so the run survives and every timing it reports is void.
- **`logging.Run` for the jsonl, W&B behind the flag.** Every record carries the
  run id and the caller's hashes, so bench reproducibility and the run log hold
  by construction. Pass `dynamics_hash`. Note the credential hazard the
  verification log records: with no key **on a terminal**, wandb prompts without
  a timeout, so `Run` bounds the login with `WANDB_LOGIN_TIMEOUT_S` and raises.
  That matters more here than in Phase 1, because what hangs before step 0 is an
  overnight run.
- **The training VRAM bar ("peak training VRAM <= 7.5 GB", on an 8 GB card)
  has no measurement for this model yet.** The sizing probe recorded no VRAM
  figure. The mask measurement recorded a peak of 2.227 GB for block-causal and
  2.333 GB for strict at batch 16 (r54), and states it is not item 4's
  measurement of that bar. Batch 16 was the probes' choice rather than a measured
  optimum. Measure both on the first run, at the batch actually used.

**The epoch-time bar is scored against a 300,000-frame equivalent -
decision 8, taken 2026-09-21.** The bar - "full 300k-frame epoch <= 30 min", tier
S - names a 300,000-frame pass. A dynamics epoch, as the sizing probe defines it,
is 276,705 **overlapping windows**, 13.8x the dataset in tokens. Those are two
different things, and the decision is which one the bar is read against. **It is
a scoring-method decision only, and no bar moves**: the 30-minute bar stays where
it is, and moving a bar because a run missed it is the failure this project's
rules exist to prevent. What is timed is the train loop working through the
dataset's 300,000 frames once, and the row that measures it states how it counts
a frame inside a window, because that conversion is where a verdict could be
manufactured. **The equivalent figure is unmeasured.** It is deliberately not
derived from the windowed epoch's 1.06 h, which rests on a throughput lower
bound, leaves out the held-out pass, times no rotation and was priced at 975
positions a window. Scaling it would imply a verdict nobody measured.

**Working when:** a one-epoch run writes a resumable checkpoint and a jsonl
carrying `dynamics_hash`; `--resume` continues it with no visible discontinuity
in the loss curve; the val loss is in the log from epoch 1; the stopping rule
stops where decision 4a says, and the run's row records which rule fired and at
which epoch; gate row 1's measure and persistence on the same windows are in the
log at every sub-epoch point, and the best checkpoint by that measure is kept and
scored on the full population against the baseline re-measured there; and peak
VRAM and the GPU clock state are recorded next to the step time.

**Landed 2026-09-25**, in `mirage/dynamics.py` (`train`, `stop_decision`, the
sub-epoch scorer) and a new `mirage/dynamics_eval.py`, which scores the best
checkpoint on the full population - decision 6's split, since that step runs
against a finished checkpoint. `_keep_awake` is imported from `mirage.fsq`, as
`bench/mask_probe.py` already did. The sub-epoch windows are the mask
measurement's 512 curve windows (subset seed 0), the population r59 re-read
persistence on, chosen by the captain 2026-09-25. Its `runs.jsonl` row, "item
4's training loop", has the numbers; in brief:

- **The one-epoch run** (`--epochs 1` at seed 0: the mask measurement's
  initial weights, data order and schedule) retraced r54's block-causal seed 0
  curve - the same 86.28% at step 1,000, held-out loss lowest at step 10,000 -
  and **no point of 18 was above persistence**, whose 86.47% on those windows
  is r59's figure exactly. The best checkpoint, step 1,000, scored on the full
  population, is **0.19 points below persistence** (86.50% against the probe's
  re-measured 86.69%), with 0.31% false flips on static cells; the final one
  is 2.19 points below, copying 89.4% of cells and flipping 6.2% of static
  ones. The epoch's gap is 0.27 in cross-entropy. That is item 4's point 6
  arriving: the answer on one epoch is "no".
- **Peak training VRAM** is 2.20 GB allocated (2.36 GB reserved) at batch 16,
  the batch used - the training VRAM bar's first measurement for this model.
  Step time 158.5 ms, with the SMs at a median 2,610 MHz drawing 98.4 W of a
  100 W limit. **Other compute processes shared the GPU** during that run
  (four, unidentified, up to 760 MiB each), so that timing is not clean.
- **`--resume`** continues a run killed mid-epoch from its last epoch with no
  visible discontinuity against an uninterrupted twin, and `_self_check`
  proves it bit-identical on CPU. The stopping rule is `stop_decision`, checked
  case by case and tripped through the loop itself across a resume.

Still unmeasured, and not item 4's working-when: SDPA against materialized
attention and RoPE's own cost, and decision 8's 300,000-frame epoch
equivalent. **Continuing a run past a stop** is refused by `--resume`: the
plan calls stopping a decision point, and going on past it changes the rule or
the schedule, which is a new decision rather than a resume.

### 5. The token-to-pixel path, which does not exist yet

Phase 2's output is token ids. Every quality requirement below is measured on
**pixels**. No code in the tree turns ids back into pixels: `FSQ.codes_to_indices`
has no inverse, `Tokenizer` has `encode` and no `decode`, and
`fsq_eval.reconstruct` starts from pixels. `bench/q3_blind_probe.py` says
outright that it round-trips frame `t + lag`'s **pixels** rather than decoding its
cached token row, "instead of adding a second decode implementation".

So the inverse lands **once, in `mirage/fsq.py` beside `codes_to_indices`**, and
never as a copy in `dynamics.py`. It is the mixed-radix expansion that function
already documents, run backwards: recover each channel's digit from the id by the
place values, then undo the normalisation that `codes_to_indices` undoes in the
other direction. Two cautions carry over from that docstring. The digit bound is
**per channel**, because a mixed table like `[8,6,5]` has three different digit
ranges and a single bound lets an out-of-range digit through. And **register no
new buffer**, because a new entry in `state_dict` makes existing checkpoints fail
a strict load for no gain.

**Working when**, restated 2026-09-25 to what was actually proven: enumerating
all `prod(levels)` ids and round-tripping them through both directions is exact
(the forward direction's `_self_check` already enumerates them, and now does
this too); decoding the ids `fsq_eval.reconstruct` itself produced reproduces its
`uint8` frames exactly, **at the same batch size**; and decoding a held-out
frame's **cached** token row differs from `reconstruct`'s frame only where the
two encodings of that frame differ - by the input-layout confound below, whose
size is measured.

~~Decoding a held-out frame's cached token row reproduces
`fsq_eval.reconstruct`'s `uint8` frame for that frame, at the same batch size.~~
The batch is pinned because that is the confound gate row 5 was taught to avoid:
it once re-encoded at 256 against a cache written at 128, and failed a
determinism row on the batch difference rather than on nondeterminism.
~~With the batch pinned, any mismatch means the digit recovery is wrong, not the
decoder, since both paths then feed the same convolutions the same codes.~~
**Refuted 2026-09-25: the batch is not the only confound.** Pinned at the
manifest's 128, R1's cached rows and `reconstruct` still disagree on 29 of 8,192
tokens (0.35%) of one held-out batch, and 17 of its 128 frames decode up to 67
uint8 levels apart. The cause is the encoder's **input memory layout**:
`write_token_cache` feeds it a channels-last view (`permute`, no `contiguous`),
`reconstruct`'s `_batch` a contiguous tensor, and under cuDNN's default TF32
convolutions the two pick different kernels. Over shards 0 and 1 (43,200 frames),
the cache's own input path re-encodes the cache with **0** flips and the
contiguous one flips 18,523 tokens (0.33%); with TF32 off, both flip about
16,900. The digit recovery is exact either way.

**Decided 2026-09-25: both encoder paths stay as they are.** Aligning
`reconstruct` to the cache would move the recorded gate numbers, and rewriting
the cache would move every shard's sha256 under the cache Phase 2 inherits. The
confound is handled where it bites instead: see the gotcha row.

**Landed 2026-09-25**: `FSQ.indices_to_codes` beside `codes_to_indices`, sharing
one place-value helper and registering no buffer, and `Tokenizer.decode(ids)`,
which returns `forward`'s float output, so the caller converts to uint8 exactly
as `reconstruct` does. `python -m mirage.fsq` round-trips every id of `[8,8,8]`,
`[8,6,5]`, `[5,5,5]` and `[4,4,4]` both ways with zero mismatches, and asserts
that ids decode to exactly the codes `forward` gave over the whole `tanh` range.
`python -m mirage.fsq_eval`, now in `check.py`, strict-loads R1 and decodes one
held-out batch of its cache at the manifest's batch: the cache's input path
reproduces the cached rows exactly, decoding `reconstruct`'s own ids reproduces
its uint8 frames on all 128 frames, and decoding the cached rows reproduces them
on the 111 frames whose rows agree. **What this leaves for item 6:** it scores
`decode(cached rows)`, never `reconstruct` output - see the gotcha row.

### 6. Rollout, and the gate

`dynamics_eval.py`, since the 500-line trigger fired at item 3.

**Rollout shape.** A seed clip of `ctx` frames from a **val** episode, then one
action per step taken from that episode's own action column, decoded greedily.
The architecture doc's seam note is the reason the clip exists at all: the
control loop drives the model with MuJoCo absent, but the model needs
context before predicting anything. So "delete the simulator" is exact, and
"delete the data" is not. **The full-frame requirement** - "generates a full next
frame from previous frames plus one action, fixed step count, no fallback path" -
fixes the shape of one step: all 64 token positions are decoded, every time, with
no early exit. Any parallel decode schedule is Phase 4's ladder, not this.

**Greedy - decision 5, taken 2026-09-21.** The simulator is deterministic
("deterministic sim given a seed": the same seed and action sequence
give bit-identical frames), so the target distribution is a single point and
greedy decoding is the strong default. It also makes a rollout exactly
reproducible from a checkpoint plus a seed clip, which is what lets bench
reproducibility compare two runs at all, and it keeps gate row 9 an
exact-reproduction row rather than a statistical one. **The revisit trigger stays
in place:** evidence that greedy decoding collapses - a rollout that freezes or
falls into a short loop, not one that merely drifts. That is what would reopen
temperature sampling.

**The configurable-context requirement is why decision 2 took RoPE**, and it is
worth stating here because that requirement is scored from one checkpoint. With
learned positions, a shorter context has to index the same absolute position
table, and nothing past the trained length exists at all. With RoPE, both
directions are free. A rollout at `ctx` 15 grows to the same `ctx + 1` frames
item 1 recommends training on, so the longest rollout sequence is one the model
has trained at.

### The proposed gate - one command

`python -m mirage.dynamics --eval RUN_ID` prints the table and exits nonzero when
a pass/fail row misses, mirroring `python -m mirage.fsq --eval`. That is what
makes a gate usable from a script rather than only by reading it.

**These rows are proposals against the requirements that already exist, not new
requirements, and none of them moves a bar.** Row 1 is written against the
dynamics requirement as `world_model_requirements.md` restated it on 2026-09-22.

| # | Measure | Bar | What the requirement actually says |
|---|---|---|---|
| 1 | Held-out next-token accuracy against the **persistence baseline** - copying the previous frame's token at the same cell - on the same held-out population as the model, with the marginal-frequency baseline reported alongside | **above the persistence baseline, re-measured with `bench/token_stability_probe.py` on the model's population.** On R1, over that probe's 12 val episodes, it reads **85.67%**; that figure is the bar only on that population. Take it from the probe's `runs.jsonl` row, not from memory | **The dynamics requirement** - "dynamics model consumes interleaved frame and action tokens, predicts next token", accepted when held-out accuracy beats the persistence baseline scored like-for-like on the same population, with the marginal-frequency baseline reported alongside. **Restated 2026-09-22** from "beats marginal-frequency baseline by 3x". Its description was amended 2026-09-23 on the mask measurement - see "Before item 1". Under block-causal the teacher-forced and generated accuracies are one number |
| 2 | One full next frame from `ctx` frames plus one action, fixed step count | **exact**, no fallback path | **The full-frame requirement** - as quoted above. All 64 positions decoded every step |
| 3 | Rollout at `ctx` 4, 8 and 15 from one checkpoint | **runs** | **Configurable context** (S) - "configurable context length at load time". Item 2's gotcha is why this is an argument and not a config edit |
| 4 | Frames until the frame-to-frame continuity check fires | **>= 200** - **reported, not pass/fail, from 2026-09-25** until the coherence-horizon requirement is restated; see "Landed" below | **The coherence horizon.** The check bounds per-step change in `link_angle`, `link_extent` and each block's `bbox` centroid, calibrated so that **zero windows of ground-truth frames fire**. That is the same acceptance shape as the frame validator ("frame validator reports block count, arm pose plausibility, palette adherence", accepted at zero false positives on ground-truth frames). Calibrate on **reconstructions, not renders**; `bench/q3_blind_probe.py` is the regression test and must fire on 100% of 300-step substitutions and 0% of clean reconstructions |
| 5 | Action-following agreement, and the simulator's own agreement on the same subset | **>= 90% of the ground-truth term, both numbers reported** - **reported, not pass/fail, this phase** (2026-09-25); see "Landed" below | **Action-following** - agreement is `sign(theta_t+1 - theta_t)` against the commanded sign, on an **action-balanced** subset drawn from the val split. An absolute bar there fails a model that is exactly right, which is why the bar is relative. `bench/hold_probe.py` measures the ground-truth term - **re-measure it, see the gotcha** |
| 6 | Link-length drift over a 200-step rollout, per link, and the simulator's own drift | **<= 1.1x the ground-truth term, per link, both numbers reported** | **Link-length drift** - the statistic is the pixel-measured major extent's `(max - min) / median` over non-overlapping 200-frame windows. Ground truth reads 23.0% on link0 and 44.2% on link1, so a perfect model fails any absolute bar. `bench/link_drift_probe.py` measures the ground-truth term. **Do not re-attempt the deprojection** - it was measured and refuted |
| 7 | Block reappears in the correct position after full occlusion | **>= 80% of events** (S) | **Object permanence** - the memory result. Tier S: the project ships without it, and the negative result gets reported either way. `mirage.data.seen_later` owns the recoverable-occlusion split, and recoverable occlusion (5.35% of frames) is the event rate it scores over |
| 8 | Parameter count, and peak training VRAM | **<= 20M bf16**, **<= 7.5 GB** | **The parameter and training VRAM bars.** The sizing probe settles the parameter bar: 14,593,152 for the chosen variant, 14.4-15.0 M across all four. The VRAM bar's first measurement is item 4's one-epoch run: 2.20 GB allocated at batch 16 |
| 9 | Rollout reproduced from the checkpoint plus the seed clip | **identical** | **Determinism** and **bench reproducibility** - a rerun matching within 5%. Greedy decoding, decision 5, makes this an exact-reproduction row rather than a statistical one; only item 6's revisit trigger would change its shape |
| 10 | Train-val loss gap; share of predictions the copy baseline also gets right; false-flip rate on static cells | **reported** | not requirements - the warning signs for overfitting and for a trivial model. The first is item 4's headline instrument. The second keeps row 1 honest now that its bar *is* a baseline: a model that clears the bar while agreeing with the copy baseline almost everywhere is winning on the cells the baseline already gets right, and the overlap is what shows it. The third, added 2026-09-24 with item 4's amendment, is where the mask measurement's model lost to copying (r59) |

Rows 1 to 6 and 8 to 9 are the pass/fail candidates; 7 is S-tier and reported;
10 is reported. Row 1's bar is the dynamics requirement's restated one, and it is
**much harder than the requirement originally promised**. The token-stability
probe records the zero-parameter copy baseline as far above 3x the marginal
top-1, so a model that predicts every cell unchanged passed the old wording and
fails this one. **That comparison was asserted, not measured, until the mask
measurement computed the marginal top-1 on its own population**: code 474 at
7.00% against persistence's 86.69%, 12.4x (r54). `bench/token_stability_probe.py`
still does not compute it, so on any other population row 1's marginal-frequency
column is where it first gets a number.

**Landed 2026-09-25**, in `mirage/dynamics_eval.py` (`rollout`, `evaluate`),
run as `python -m mirage.dynamics --eval RUN_ID`, with `--checkpoint best` (the
default, item 4's `best.pt`) or `final` (the last `model.pt`). Its `runs.jsonl`
row, "item 6's gate" (r61), has the numbers. What was built, and what differs
from the table above:

- **The rollouts.** One per val episode, 27 in all, seeded with frames `0 ..
  ctx - 1` and run to the episode's end: 585 steps at `ctx` 15, 592 at 8, 596 at
  4. Every pixel row decodes tokens through `Tokenizer.decode`, the ground truth
  included, per item 5's gotcha. The self-check proves the rollout's shape on a
  tiny model with stand-ins whose answer names their input: frame `ctx + k`
  reads `action[ctx + k]`, and each step's output is the next step's input.
- **Row 2** compares the rollout's first frame with `score_windows`' one pass
  over the same 27 windows at the same batch, and they are identical.
- **Row 4 is reported, not pass/fail (decided 2026-09-25), because its check
  cannot meet its own acceptance test.** The bounds are four new `validator`
  keys, `continuity_link_angle_max`, `continuity_link_major_max`,
  `continuity_link_minor_max` and `continuity_block_centre_max`, one value per
  link or block, set by `python -m mirage.dynamics_eval --calibrate-continuity`
  to the largest per-step change over all 16,173 transitions of the val split's
  decoded cached rows, so no ground-truth window fires. With those bounds,
  `bench/q3_blind_probe.py`'s 300-step substitutions fire on **53.7%**, not the
  100% the coherence-horizon requirement asks. Nothing tried reaches 100%, even
  allowing false alarms: gating on pixel counts, medians over 3 to 9 frames, a
  joint statistic, or bounds at the 99th percentile (90%). The arm moves about
  0.007 rad a frame, while on decoded frames the pixel link angle jumps up to
  0.39 rad on link0 and a full pi/2 on link1 when it is partly hidden, and a
  block's bbox centre jumps up to 30 px. So link1's angle bound is the largest
  change there is and never fires. Restating the requirement is the captain's
  call; until then the row prints the horizon beside the regression result.
- **Row 5 is reported this phase (decided 2026-09-25), because its instrument
  is at chance.** It scores one-step predictions from true context on an
  action-balanced subset of row 1's population (the neutral action left out,
  9,480 frames), and the truth on the same frames, both through the pixel link
  angle. On the truth that angle's per-step sign agrees with the command 47.1%
  of the time, where the simulator's own joint angles agree 89.7%. `bench/hold_probe.py`
  re-measured its term at the shipped hold at 90.7%, on its own random-policy
  episodes, which is not the same subset. Over a whole held action link0's sign
  does work (91.3% at 14 frames) and link1's still does not.
- **Row 7** scores every reappearance at the reappearance frame itself, with
  no shift forward and no event dropped. Correct position is the block's bbox
  centre within 4 px of the decoded truth's at that frame, and the 2 px share
  is reported beside it. A block coming back shows a pixel or two that the
  decoder often drops; an event whose decoded truth does not show the block at
  that frame counts as a miss, and the row prints how many events were in that
  case. 21 events fall inside the rollouts.
- **Row 8** reads peak training VRAM as the run's largest reserved figure, with
  allocated beside it. **Row 10**'s gap is the run's last epoch.

**On item 4's one-epoch run** (`20260925-100507-dyn`) the gate prints every row
and exits 1, on row 1 alone, as expected: `best.pt` is 0.19 points under
persistence. Rows 2, 3, 6, 8 and 9 pass. **Its rollouts freeze**: every
generated frame repeats the one before it, since that checkpoint copies 99.6%
of cells, and with `--checkpoint final` 92.0% still do. That is decision 5's
revisit trigger on an undertrained checkpoint, not yet a verdict on greedy
decoding, and it shows two rows a frozen rollout games:
row 4 reads the whole 585 frames and row 6 reads no drift at all, which passes.
The gate prints the frozen share under the table for that reason. The
link-drift requirement's risk row already warned the bar might not tell a bad
model from the simulator.

`dynamics_eval.py` is now past 500 lines. Decision 6's split is by when code
runs, and everything in it runs against a finished checkpoint, so nothing moves.

---

## Decisions: eight taken, one settled by its trigger

Each one changes an item above, so each is named rather than quietly resolved.
Decision 1 was taken 2026-09-18, decision 9 on 2026-09-23 by measurement, and
decision 7 on 2026-09-23. Decision 6 was never taken: its trigger fired at item
3, on 2026-09-24.
The other five were taken 2026-09-21, after a walkthrough of this plan's draft,
and each is written down with its rationale.

**1. The dynamics model's acceptance test - DECIDED 2026-09-18: restated against
the persistence baseline, and `world_model_requirements.md` carries it as of
2026-09-22.** The requirement had asked the dynamics model to beat a
marginal-frequency baseline by 3x. The token-stability probe's `runs.jsonl` row
recorded, for Phase 2 and deliberately without acting on it, that **copying the
previous frame's token at the same cell scores 85.67%** on the R1 checkpoint
Phase 2 inherits, over 460,032 held-out cell-transitions from 12 val episodes,
**at zero parameters**. It records that as far above 3x the marginal top-1, which
it asserted rather than measured; the mask measurement has since measured it at
12.4x on its own population (r54). That baseline is now the bar, and gate row 1 is
written against it.

**The rationale, recorded because the bar moved on it: an acceptance test a
zero-parameter baseline already passes cannot show the model learned dynamics.**
Beating the marginal frequency by 3x only asks the model to know which codes are
common. Beating persistence asks it to know when a token *changes*, which is the
only part of the sequence that carries the physics. Note the cost: it is a **much
harder bar than the requirements document originally promised**. A model that
predicts every cell as unchanged scores the baseline itself, and now fails.

**Three constraints on quoting it.**

- **Take the figure from the probe's `runs.jsonl` row, not from memory.** The
  sizing probe's row exists for exactly this reason, and says so: it records
  sizing "computed in an earlier session and never written down", so that a plan
  is written against numbers instead of recollections. **Registered 2026-09-24**,
  by a separate, deliberate act: `canonical_numbers.md` now carries 85.67% for
  this population and 86.69% for every val window's last frame, each with its
  population and checkpoint.
- **The comparison is like-for-like or it is nothing.** The requirement
  re-measures the baseline with `bench/token_stability_probe.py` on the same
  held-out population the model is scored on, so 85.67% is the bar only when that
  population is the probe's 12 val episodes. Otherwise the row compares two
  different statistics and reports the difference as skill.
- **The baseline belongs to the tokenizer checkpoint, not to the requirement.**
  It reads 85.67% on R1, 93.22% on r1c and 77.28% on R2, so a checkpoint promoted
  above R1 means re-running the probe, never reusing R1's figure.

**Where it is written down.** `world_model_requirements.md` restated the
requirement on 2026-09-22, with the dated paragraph under its Models table and an
F-11 row in "Requirements at risk", so the requirement and this plan agree. Where
their wording differs, the requirement wins. The one piece still missing is the
verification-log row naming the acceptance procedure, which cannot be written
until gate row 1 exists in code. **No Phase 2 verdict on it exists, and none is
quoted here.**

**2. Sequence layout and position encoding - DECIDED 2026-09-21: RoPE, and the
no-shift interleaving.** The irreversible one. **RoPE**, because it keeps the
configurable-context requirement's shorter-context rollout free in both
directions, and because the sizing probe shows the four variants span under 4% of
parameters, so this is not a capacity trade. **The interleaving is the no-shift
reading of the recorded alignment**: the one action token that conditions a frame
immediately precedes that frame's 64 token positions, and both are read at the
same record index. That is the convention `mirage/data.py`'s module header
already states - `qpos[t] - qpos[t-1]` is the result of `action[t]`, same
record - and it matches the probe's 65 positions per frame step. **The phase
assertion is mandatory and is item 1's acceptance test**: all 13,242 action
changes must sit at `step_idx % sim.action_hold_steps == 0`, and a deliberately
shifted copy must fail that same assertion. **Do not use agreement statistics to
confirm the alignment**, because the wrong reading scores higher - 95.6% against
93.9% - and both clear the action-following bar. Two things this decision does
not settle: the window's frame count, which item 1 names, and the mask, which
decision 9's measurement selects and which may change how a frame's positions are
fed, but not which record they are read from.

**Item 1's window index arithmetic - DECIDED 2026-09-21: shared, not copied.**
The plan's own recommendation and the project's precedent. The shared addressing
and `WindowSampler` must agree on every window's (episode, offset). It edits
`mirage/data.py`, whose `_self_check` is the round-trip requirement's
acceptance test, and the cost is believed small because the edit moves indexing
and nothing that requirement checks. Item 1 has the detail.

**3. Output embedding - DECIDED 2026-09-21: untied, two separate matrices.** Tying
is only sound because the frame codes occupy the first block of the vocabulary,
with the action tokens appended after - `bench/dyn_size_probe.py` says so in its
own comment - and that is a hidden coupling to the layout. The parameter cost
stays inside the same sub-4% spread, and the parameter bar has headroom.
**The count is on record**: the sizing probe prices all four variants, and
RoPE-and-untied is its `rope_untied`, **14,593,152**. An earlier draft of this
plan quoted only the two endpoints, 14,396,544 for RoPE-and-tied and 14,967,552
for learned-and-untied, which made the chosen variant look unpriced. The first run
still reports its own count, per item 3's working-when.

**4. The answer to the Chinchilla shortfall - DECIDED 2026-09-21: measure the gap
first.** Unchanged from item 4: the first run measures the train/val gap before
any remedy is chosen. Recorded alongside it: **more data is the remedy tangled up
with `data_hash` provenance**, since regenerating the set moves the hash and
orphans both the checkpoint and the token cache, while a smaller model and
heavier regularisation do not.

**4a. The first run's stopping rule - DECIDED 2026-09-21**, the part decision 4
otherwise leaves open. An epoch cap or a divergence trip, whichever fires first.
The cap is 10 epochs, one overnight window at the chosen model's measured
0.78 h/epoch (r54; the sizing probe's lower-bound figure was 1.06 h). The trip is
held-out loss rising for two consecutive epochs, after which the run continues
for two more epochs before stopping, because decision 4 wants the size of the
gap and not just where it turns. Item 4 states it in full, beside the per-epoch
resumable checkpoint and `--resume` by run id that make stopping a decision
point rather than the end of the run. **Amended 2026-09-24, by addition:**
alongside this rule the run scores gate row 1's measure against persistence on
the same windows at sub-epoch intervals and keeps the best checkpoint by it. The
rule itself is unchanged; item 4 has the amendment.

**5. Rollout decoding - DECIDED 2026-09-21: greedy.** The revisit trigger in item
6 stays in place: a rollout that freezes or falls into a short loop, not one that
merely drifts. Greedy keeps gate row 9 an exact-reproduction row rather than a
statistical one.

**6. Whether `dynamics_eval.py` splits out - governed by its trigger, which fired
at item 3 (2026-09-24).** The same 500-line trigger that split `fsq_eval.py` out
of `fsq.py`, applied when it fires and not before. The model took `dynamics.py`
past 500 lines. That split is by when the code runs, and nothing in
`dynamics.py` yet runs against a finished checkpoint, so nothing moved: rollout
and the gate table, item 6, start in `dynamics_eval.py`.

**7. Exposure bias - DECIDED 2026-09-23: name a trigger, build no mitigation
now.** Teacher forcing trains on ground-truth context, and the rollout feeds the
model its own output. Under decision 9's block-causal mask, a frame's 64 tokens
come out of one pass from earlier frames only, so the gap applies across frames,
not within one. Item 4's first real training run trains with no exposure-bias
mitigation, which keeps its train/val gap measurement clean. **Trigger:** gate
row 4's coherence horizon (frames until the continuity check fires) comes in
under 100 frames while gate row 1 passes (held-out accuracy beats the persistence
baseline). **If it fires**, the first remedy is context-token corruption during
training (randomly replacing a small share of context tokens), and scheduled
sampling is the second. The accepted cost is one retrain.

**8. What the epoch-time bar means for a windowed epoch - DECIDED 2026-09-21:
scored against a 300,000-frame equivalent.** A scoring-method decision only; the
30-minute bar does not move. The equivalent figure is **unmeasured**, and item 4
says why it is not derived from the windowed epoch's 1.06 h.

**9. Strictly-causal against block-causal attention - DECIDED 2026-09-23 by the
measurement ordered 2026-09-22: block-causal**, taken by that measurement rather
than by argument, under a rule fixed in code before the first step
(`bench/mask_probe.py`). "Before item 1" says what it changes and the trigger that
would reopen it. The dynamics requirement's description carries the second,
separately dated amendment.

---

## Gotchas, and how you would notice

| Gotcha | What breaks | How you notice |
|---|---|---|
| **A one-step action misalignment** | Every checkpoint conditions each frame on the wrong action, and action-following scores the wrong thing | **You do not**, from agreement: 93.9% against 95.6%, and the **wrong** reading scores higher, with both clearing the action-following bar. Only the phase assert catches it - all 13,242 action changes sit at `step_idx % action_hold_steps == 0`. Assert it, and assert that a shift of one breaks it |
| **Scoring gate row 1 against the probe's 85.67% on another population, or on another tokenizer** | The row compares two statistics and reports the difference as skill | The row names episodes other than the probe's 12 val ones, or a checkpoint other than R1. Re-measure the baseline with `bench/token_stability_probe.py` on the model's own population and checkpoint; 85.67% is the bar only on its own population |
| **Quoting the sizing probe's 975 positions as the training sequence** | Tokens per epoch, step time and epoch time are all understated, and a schedule built on them runs long | The probe prices `ctx x 65`, while `WindowSampler` holds `ctx + 1` frames - 16 at `ctx` 15 - and the probe's own window count is the sampler's. **Under the block-causal mask now selected, a 16-frame window is 975 input positions**, so the gap applies to the strictly-causal case. Re-take the timings at the sequence actually built |
| Putting the context length in `data.ctx` for the configurable-context requirement | `data_hash` moves, `load_shards` refuses the 300,000 frames, and `load_run` refuses the R1 checkpoint | Loudly, on the next run - which is the good case. The bad case is a session spent editing the register instead of passing an argument |
| A shape knob outside the `dynamics` section | `dynamics_hash` does not name the model that produced the number, so bench reproducibility has a hole | **You do not.** Two runs with different head counts log the same hash. `n_heads` was in this state until item 2 put it in the `dynamics` section, which both bench probes now read |
| Calibrating the coherence horizon's continuity check on **renders** | The check is tuned on the wrong population and fires on ordinary decoder output | The same two-population trap that cost Phase 1's build-order item 6 its obvious recipe: rendered pixels sit at most 0.75 RGB units from the palette, reconstructed ones up to 154.9. Calibrate on reconstructions |
| Reusing the frame validator (the per-frame plausibility check accepted at zero false positives on ground truth) as the rollout terminator | The horizon measures decoder artifacts and nothing about dynamics | **Measured and refuted** (`bench/q3_blind_probe.py`): the validator fires on **0.00%** of frames substituted from 300 steps away, against 100% on its noise control. The validator itself is unchanged and still does its own job. The probe is the regression test that catches a replacement going blind the same way. **The replacement, the continuity check, reads 53.7% on it, not 100%** (item 6, 2026-09-25), so gate row 4 is reported until the requirement is restated |
| **Quoting the register's action-agreement ceiling (`NUM-DATA-Q4CEIL`) as action-following's ground-truth term** | The action-following gate row is scored against the superseded physics | That entry reads **83.1%**, measured before the `gear 6 / damping 1.5` change, while the `runs.jsonl` row it cites (the re-run at that change) measures **91.5% after it**, with a relative bar of 82.3%. `docs/phase0_debt_checklist.md` records the same before/after pair. The action-following requirement says to re-measure the ground-truth term on the same subset anyway, so **re-measure with `bench/hold_probe.py` and report both numbers**. Do not copy either stored value, and do not silently edit the register from a Phase 2 plan |
| **A frozen rollout scored as a coherent one** | Row 4 reads the full horizon and row 6 reads zero drift, which passes, for a model that only copies | Item 6's first gate run: `best.pt` of item 4's one-epoch run copies 99.6% of cells, and every generated frame repeats the one before it. The gate prints the share of frozen frames under the table; read it before rows 4 and 6 |
| A second decode implementation in `dynamics.py` | Two token-to-pixel paths that will eventually disagree | The disagreement is a wrong picture, which nothing crashes on. Item 5 exists so there is one path |
| Training on token rows as if they were upside down | Well-formed tokens for mirrored frames | **Already handled** - `write_token_cache` flips the rows on the way in, exactly as `preload` does, and says so. Do not add a second flip: the blob is bottom-up, and the flip lives in one place |
| Hardcoding the re-encode batch | A determinism check that tests "re-encode at a different batch size" instead of determinism | Gate row 5's own history: the check originally re-encoded at 256 against a cache written at 128 and failed R2 on a false alarm. Read `batch` from the manifest |
| Calibrating or scoring item 6 on `fsq_eval.reconstruct` output | The coherence horizon's continuity check, and any other pixel-level calibration, is tuned on tokens that differ from the cache in 0.33% of cells, while every rollout is decoded from cache-like tokens: the same two-population trap as renders against reconstructions, one level down | Item 5's measurement: `reconstruct` feeds the encoder a contiguous tensor, `write_token_cache` a channels-last view, and under TF32 that flips 0.33% of R1's tokens at the same batch, moving a frame by up to 67 uint8 levels. **Item 6 and any pixel calibration score `Tokenizer.decode(cached rows)`** - the population the model trains on and a rollout produces. Decided 2026-09-25: neither encoder path changes. A re-encode check must also copy `write_token_cache`'s input path (`permute` without `contiguous`) |
| Mixing token caches from two runs | The model trains on a mixture of two tokenizers | The manifest's `tokenizer_hash` and `run_id` disagree with the checkpoint's. The cache directory is named by run id for this reason |
| The `nn.Upsample` native-layer fault | It is a fault in the **tokenizer's decoder**, so the dynamics train loop cannot hit it - but the **rollout decode path can** | `AttributeError: 'str' object has no attribute 'align_corners'` inside `Upsample.forward`. It fired at epochs 22 and 36 of one 60-epoch tokenizer rung (r1c), so roughly every 14 epochs of running that decoder. Not this project's bug; budget restarts for any long eval that decodes |
| Modern Standby mid-run | Every timing the run reports is void, while the run itself survives | A single epoch reading many times its neighbours. `fsq._keep_awake` is the per-process request that prevents it, and it does nothing off Windows |
| Reading `pstate == P0` as a valid-benchmark gate | A correct compute-bound run is rejected, or a throttled one accepted | Refuted 2026-08-23. Gate compute on SM clock plus power draw, bandwidth on memory clock, and record them next to the number - which is exactly why the sizing probe's throughput is a lower bound |
| **Touching `bench/patch_probe.py`'s `RF` constant, or anything quoting 22x22** | The constant is wrong in a way the register depends on | The token-stability run measured the true conv field at **15x15 = 225 px**, and the effective field under the encoder's `GroupNorm` at the whole **4,096 px** frame. `RF = 22` is neither. The register's 94.25% entropy ceiling rests on the premise that cells with identical receptive fields must share a code, which **says nothing while `GroupNorm` is in the encoder**. It was logged and **deliberately left unrepaired**: the error points the safe way, since a larger true field means more room than registered, and no passed gate moves. **Phase 2 does not fix it**, and a Phase 2 item that quotes 22x22 is quoting a known-wrong number |
| Reading a falling train loss as progress | The 13.8x window overlap means an epoch is mostly repetition | Train loss falls while held-out loss rises. This is the phase's expected failure, and item 4's instrument exists for it |

---

## The numbers already taken, so item 1 does not re-derive them

Every row traces to a `runs.jsonl` row, a register entry in
`canonical_numbers.md`, or the verification log at the end of
`world_model_architecture.md`, and the row names which. Nothing here was measured
by this plan: there is no dataset and no checkpoint on the machine it was written
on.

| Measure | Number | What it settles |
|---|---|---|
| Model shape | `d_model` 384, 8 layers, 6 heads, MLP 4x - sizing probe | item 3 does not choose these; they were specified and are now priced |
| Sequence | 15 frames x (64 + 1) = **975** positions, vocab **521** in / **512** out - sizing probe | the 65 positions per frame step that decision 2 lays out. **975 is `ctx x 65`, and the sampler's window is `ctx + 1` frames** (item 1), so read it as the probe's pricing, not as the training sequence |
| Parameters | **14,396,544** RoPE + tied, **14,593,152** RoPE + untied, **14,770,944** learned + tied, **14,967,552** learned + untied; spread **571,008** - sizing probe | **the irreversible layout choice is not a capacity choice**, and the parameter bar passes at all four. Decisions 2 and 3 take **RoPE + untied** |
| Step and epoch cost | bf16 **221.6 ms/step** at batch 16, **1.06 h/epoch**; fp32 **622.4 ms** and **2.99 h** - sizing probe | bf16 for item 4. **A lower bound on throughput, so an upper bound on time, at 975 positions a window only**: `gpu_probe` returned compute FAIL, 2385 of 3090 MHz and 20.6 TFLOP/s against the cool machine's 27.6, under an 85 W enforced limit (r49). Timed on the RoPE + untied parameter layout with no rotation, through `nn.MultiheadAttention`, at 975 positions a window. **Superseded for schedules by the row below** |
| The chosen model's step and epoch cost | block-causal bf16 **159.6-160.1 ms/step** at batch 16, 975 positions, through SDPA with RoPE applied, SMs at a median 2606-2617 MHz and 98.8-98.9 W of 100 W; one epoch **2,813-2,822 s, about 0.78 h**, including 18 curve evaluations - mask measurement (r54, Linux) | decision 4a's cap and any schedule. Gated on SM clock plus power draw. `gpu_probe`'s own compute FAIL in that row comes from its decay statistic counting a pre-load sample, not from the GPU (r58) |
| Data against capacity | **19.5 M** tokens, 300,000 frames x 65 counting one action token per frame (the cache itself holds **19.2 M** frame tokens, `manifest.json`), against a Chinchilla-optimal **291.9 M**, **15.0x under** (15.2x on frame tokens alone); one epoch draws **276,705** windows totalling **269.8 M** tokens, **13.8x** the dataset from overlap - sizing probe | **the phase's risk is overfitting, not throughput**. Decision 4: the first run measures the gap before a remedy is chosen. The epoch's token figures are priced at 975 positions a window (item 1) |
| Token cache size | **38.4 MB** - sizing probe | the whole cache fits in VRAM many times over; nothing about the data path needs engineering |
| The inherited checkpoint | `20260829-005439-r1`, 31.095 dB held-out PSNR at 74.1% token entropy - register | the tokenizer is fixed, and so are the 512 codes and the 64-token grid |
| The zero-parameter baseline | **85.67%** token persistence on R1, over **460,032** transitions from 12 val episodes, **396,013** of them quiet-field - token-stability probe | **the dynamics requirement's restated bar on that probe's population**, and gate row 1. On any other population, re-measure with `bench/token_stability_probe.py` |
| Token instability on R1 | **8.86%** of transitions flip with no change in the cell's own 15x15 field; **53.21%** of all flips - token-stability probe | context for a rollout that looks noisy. The probe's row is explicit that **nothing has measured what spurious flips cost a dynamics model**, in either direction, and that Phase 2 evidence either way is a recorded trigger to reopen the tokenizer choice |
| The split | 473 / 27 train/val episodes, 16,200 held-out frames of 300,000 total, at `data_hash` `18a76531` - register | reuse `data.is_val`; the tokenizer's val set and Phase 2's are the same set by construction |
| Action alignment | same-record **93.9%** against next-record **95.6%**, and all **13,242** action changes at phase 0 - verification log, the alignment row | item 1. The agreement figures are there to prove agreement cannot settle it |
| ctx=15 window coverage | **60.9%** of windows carry any evidence of what an action does - the gate re-run at the shipped physics (`gear 6 / damping 1.5`, `action_hold_steps` 15). The later gate re-run at `data_hash` `18a76531`, with the scene and the policy untouched, found every M-tier row unchanged, so that figure still describes the set on disk - but **window coverage is not one of the rows the later re-run measured** | about 39% of training windows contain no action change at all, which is what an action-balanced action-following subset is drawn against |
| Action-following's ground-truth term | **91.5%**, relative bar **82.3%** - the after-values of the `gear 6 / damping 1.5` re-run. **The register's 83.1% disagrees**; see the gotcha | re-measure with `bench/hold_probe.py` and report both numbers |
| Link-length drift's ground-truth term | **23.0%** on link0 and **44.2%** on link1 - link-drift probe, quoted in its requirement row | a perfect model fails any absolute bar; `bench/link_drift_probe.py` produces the term |
| The coherence horizon's terminator | The frame validator fires on **0.00%** of 300-step substitutions against **100%** on the noise control - coherence-horizon blind probe | the continuity check replaces it, and the probe stays as its regression test |
| Run-to-run noise | 0.00167 dB, and it is the **tokenizer's** 1-epoch figure - register | **unmeasured for a dynamics rung.** Do not call a margin "inside the noise" here; no seed has been repeated on this model |
| Peak training VRAM | **2.20 GB allocated, 2.36 GB reserved, at batch 16** - item 4's one-epoch run, the first measurement of the bar for this model. The mask measurement's 2.227 GB block-causal and 2.333 GB strict at batch 16 (r54) were not that measurement | the training VRAM bar (<= 7.5 GB) at the batch used. Batch 16 is still the probes' choice, not an optimum |
| SDPA against materialized attention, and RoPE's rotation | **unmeasured as an A/B.** The chosen model through SDPA with RoPE applied is timed at 159.6-160.1 ms/step (r54), against the sizing probe's 221.6 through `nn.MultiheadAttention` with no rotation - two changes at once, on different platforms, so neither's own share is known | the first run measures both rather than assuming a sign; item 3 built the model and took no GPU measurement |
| The epoch-time bar's 300,000-frame equivalent | **unmeasured** | decision 8 fixes how that bar is scored, not what it reads. Not derived from the probe's windowed epoch |
| Rollout throughput, and any performance row | **unmeasured**, deliberately. The five interactive performance rows (sustained frame rate, p99 frame time, input-to-display latency, the p99/p50 jitter ratio, and the eager-to-engine speedup) belong to Phase 3's baseline and Phase 4's ladder | Phase 2 produces a checkpoint, not a frame rate. The 30-minute epoch bar is the only performance row this phase touches, and item 4 says how it is scored |

Record the GPU power state next to every timing; a timing without it is not a
number. Gate compute numbers on **SM clock plus power draw**, and bandwidth on
**memory clock == max**.

---

## What this plan does not do

No part of `mirage/dynamics.py` is written here, no run was launched, and no
measurement was taken. There is no dataset and no checkpoint on the machine this
was written on, so every number above is quoted from the record rather than
earned here. **No register entry was created by this plan**: registering a
number is a separate, deliberate act. That act was taken 2026-09-24 for the
persistence baseline, now in `canonical_numbers.md` for both populations; for
everything else, the `runs.jsonl` rows for the token-stability probe, the sizing
probe and the mask measurement are the sources. **No bar is moved here.** The
dynamics model's acceptance test was raised, not lowered, and
`world_model_requirements.md` restated it on 2026-09-22; gate row 1 is written
against that. Moving a bar *down* because a run missed it is the failure this
project's rules exist to prevent, and nothing here does that - the one run
since, the mask measurement, selected a mask and moved no bar. **The decisions
above are recorded here, not taken here**: they were taken 2026-09-18,
2026-09-21 and 2026-09-23, the last by the mask measurement. The one new call
this plan makes - `ctx + 1` frames a window, in item 1 - is written as a
recommendation with its alternative. Phases 3 and 4 stay undrafted, which is
"profile before changing anything" applied to planning.
