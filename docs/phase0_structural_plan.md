# Phase 0: Structural Plan

Guidance for implementing the eight Phase 0 files - `scene/`, `sim/`, and
`mirage/` - one at a time. Names the calls and the order; does not write the code.
Design rationale is in `world_model_architecture.md` - not repeated here.

Numbering below matches `AGENDA.md` exactly. If the two ever disagree, AGENDA is
the order of record and this file is the stale one.

**Phase 0 closed 2026-08-27 - gate met on all three conditions.** This file is
kept as the record of how it was built and as the template for later phases.
Statuses below are final, not pending. The gotchas table was updated at phase
close with what running the code actually turned up; per `phase0_debt_checklist.md`
D8, that update is the practice, not a one-off.

**Vocabulary, once.** A *GL context* is the handle that lets you issue drawing
commands; nothing draws without one. *Offscreen* means drawing into a memory
buffer rather than a visible window. A *segid* is the integer MuJoCo assigns each
piece of geometry, used to identify objects in the colour-coded pass.

---

## The dataflow, in one line

`main` loads the model and creates the context, then per step: `policy` picks an
action → `mj_step` advances physics → render → `truth` reads what happened →
`shard_writer` appends. Sidecar JSON written once at the end.

Everything flows one direction. Nothing calls backwards.

---

## What each file owns

| File | Owns | Explicitly does not own |
|---|---|---|
| `scene/arm_blocks.xml` | Geometry, the flat-render config, **the palette** | Anything read at runtime by C++ other than the model |
| `mirage/config.py` | Loading sectioned JSON, the hash tree, `Shapes` | The palette - that lives in the XML |
| `sim/CMakeLists.txt` | Two build types: optimised default, sanitizer build separate | - |
| `sim/gl_context.{h,cpp}` | GLFW init, hidden window, context current, `mjrContext`, offscreen buffer selection, the renderer-name assert | The scene, the camera, pixels |
| `sim/policy.{h,cpp}` | Seeded generator, per-episode coin flip, the action for step *t* | Calling `mj_step` |
| `sim/truth.{h,cpp}` | Reading `mjData`, counting segmentation pixels per block | Writing files |
| `sim/shard_writer.{h,cpp}` | The three output files, sidecar last | Measuring anything |
| `sim/main.cpp` | Arg parse, seed, episode and step loops, wiring | The internals of any of the above |
| `mirage/data.py` | `np.memmap` over the blobs, episode-aware sampling | Interpreting pixels |
| `mirage/validator.py` | The measurement vector, both modes, the threshold sweep | Emitting a verdict - that is a config expression |

`truth` stays separate from `shard_writer` because `truth` is the only file whose
*inputs disappear* when the simulator is deleted. That is the seam worth
protecting.

---

## Build order, with the calls each file needs

Same numbering as `AGENDA.md`. Riskiest first.

`sim/CMakeLists.txt` and `sim/main.cpp` carry no numbers because they are
scaffolding rather than deliverables. CMakeLists comes before item 3 and is
**already done**: C++20, `/W4 /WX`, an optimised default plus a `MIRAGE_ASAN`
configuration, both compiling and running a `main` that prints and exits. Never
`/fp:fast` - the MSVC spelling of `-ffast-math` - in either.

The flags are MSVC, not GCC. Three of them are not obvious and were each found by
the build failing:

| Flag | Why |
|---|---|
| `/fsanitize=address` | MSVC's ASan. There is no `,undefined` to append |
| `/Zi` plus linker `/DEBUG` | without debug info MSVC emits `C5072`, which `/WX` makes fatal, and an ASan report without symbols is useless anyway |
| post-build copy of `clang_rt.asan_dynamic-x86_64.dll` | MSVC links the ASan runtime dynamically whatever the CRT setting, and the DLL is on `PATH` only inside a Developer prompt. Derived from `CMAKE_CXX_COMPILER`'s directory, so no MSVC version is hardcoded |

**Settled 2026-08-26: E-3 is ASan-clean only.** MSVC has no UBSan and the clang-cl
configuration was rejected. The one UBSan class that matters here - signed overflow
on shard byte offsets - is handled by 64-bit offsets plus a bounds assert at the
write site, which lands in `shard_writer` (item 6). Reasoning and reversal trigger
in `world_model_architecture.md`, "Sanitizer cost".

`main.cpp` grows alongside items 3 through 6 and is finished with item 6.

### 1. `scene/arm_blocks.xml`

First because everything downstream reads it: the model the simulator loads, the
palette the validator parses, and one of the three inputs to `data_hash`.

Four things it must carry, all measured rather than stylistic:

1. `<quality offsamples="0" shadowsize="0"/>` - together worth 12x on `mjr_render`.
   Not `castshadow`: that is a `<light>` attribute and MuJoCo rejects it on a geom.
2. An ambient-only headlight (`diffuse="0 0 0" specular="0 0 0"`). `offsamples="0"`
   alone does not give the <=24-colour palette; a diffuse-lit box shades per face,
   so one `rgba` becomes three entries. No materials required - plain `rgba` under
   ambient-only light measured 6 colours on the first working scene.
3. `offwidth`/`offheight` in `<visual><global>`, or the offscreen buffer stays at
   its 640x480 default.
4. Two arm links in different colours, three blocks in three more. Current count
   is ~7 including background and table, against F-2's ceiling of 24.

The `rgba` attributes are the palette's only home. The validator reads them with
`xml.etree.ElementTree`; nothing duplicates the list into config JSON.

Doc page: **XML reference → `visual`, `asset/material`, `geom`**.

**Working when:** `mj_loadXML` returns without error, and `np.unique` over one
rendered frame reshaped to `(-1, 3)` gives at most 24 distinct triples.

### 2. `mirage/config.py`

Sectioned JSON - `sim`, `data`, `tokenizer`, `dynamics`, `engine`, `validator` -
with the hash tree rooted at `data_hash`, and `Shapes` for the tensor dimensions
every later phase derives from. Roughly 20 lines for the hash tree.

`data_hash` covers `canon(sim)`, `canon(data)`, and the scene XML's bytes. The
XML is inside it or E-4 has a hole: a bench number from a different scene is not
comparable. `validator_hash` branches off `data_hash` rather than off
`dynamics_hash`, so re-tuning a threshold does not invalidate a checkpoint whose
rollouts never changed.

Doc page: none. Stdlib `json`, `hashlib`. The C++ side reads the same file with
`nlohmann/json`.

**Working when:** the same config hashes identically twice in a row, editing a
`validator` threshold leaves `data_hash` and `dynamics_hash` unchanged, and
touching the XML changes all of them.

### 3. `sim/gl_context.{h,cpp}`

Give it real time. It is the file most likely to cost you a day, though the day-1
readback probe already cleared GLFW, so this is a port of what
`bench/readback_probe.py` does rather than an open question. No pbuffer.

Order of operations:

1. `glfwInit`, then `glfwWindowHint` with `GLFW_VISIBLE` set to false, then
   `glfwCreateWindow`, then `glfwMakeContextCurrent`. The hidden window exists
   only to own the context - you never draw into it.
2. Read `glGetString(GL_RENDERER)` and assert it is neither `GDI Generic` nor
   `Microsoft Basic Render Driver`. Deny-list, not an allow-list on "RTX 5060":
   the allow-list form fails on any other machine that is perfectly fine, while
   those two spellings are what actually indicate the software fallback.
3. `mjr_defaultContext`, then `mjr_makeContext` against the loaded `mjModel`.
4. `mjr_setBuffer` with `mjFB_OFFSCREEN`, so rendering targets the offscreen
   buffer and not the window.

Doc page: **Programming → Rendering**. Read the `record.cc` sample before writing
this - it does exactly this sequence. It lives in the MuJoCo GitHub repo under
`sample/`, not in the pip wheel.

**Working when:** the binary prints the renderer name at startup and it names
your GPU.

### 4. `sim/policy.{h,cpp}`

One seeded `std::mt19937` per shard, seeded from the shard index. A coin flip at
episode start selects random-deltas or scripted-reach for the whole episode.
Returns an action index in the 9-action space; the caller maps it onto
`mjData.ctrl`.

Doc page: none needed. This is plain C++.

**Working when:** running with a fixed seed twice produces an identical action
sequence, and a histogram over a few thousand steps is roughly flat across all
nine.

### 5. `sim/truth.{h,cpp}`

Two jobs.

*Contact:* read `mjData.ncon`, walk `mjData.contact[i]` and check `geom1`/`geom2`
against your block and arm geom ids. Produces one bitmask byte.

*Visibility:* set `mjRND_SEGMENT` and `mjRND_IDCOLOR` in `mjvScene.flags`, render,
`mjr_readPixels`, then histogram the pixels by decoded segid. Store the counts,
throw the mask away. `mjRND_IDCOLOR` encodes segid+1 into the RGB channels, so
background reads as zero and you decode by packing the three bytes back into an
integer.

Also read `mjData.qpos` for the joint angles and the block positions.

Doc pages: **`mjtRndFlag` in the API type reference** for the two flags,
**Programming → Rendering** for the readback.

**Working when:** with a block deliberately parked behind the arm, its count
reads zero, and with it in the open the count is roughly its pixel area.

### 6. `sim/shard_writer.{h,cpp}`, and `main.cpp` with it

Open the pixel blob and the meta blob, append per frame, close both, *then* write
the sidecar JSON. The write order is the whole correctness argument - the sidecar
existing is what marks the shard complete.

Meta record is fixed-width, 46 bytes, fields in the order the architecture doc
lists. Write it with explicit widths, not by dumping a struct - padding differs
between compilers and the reader on the Python side assumes exact offsets.

Doc page: none. `nlohmann/json`, single header, for the sidecar.

**Working when:** the Python side reads the pixel blob with `np.memmap` and
reshapes it to `(-1, H, W, 3)` with no stride arithmetic, and a known test buffer
written in C++ compares byte-identical.

Finish `main.cpp` here, since item 6 is the first point at which the whole loop
can run: `mj_loadXML` → `mj_makeData` → context → loop episodes → `mj_resetData`
between them → per step: policy, `mj_step`, render, truth, write. At exit, free in
reverse order of creation: `mjr_freeContext`, `mjv_freeScene`, `mj_deleteData`,
`mj_deleteModel`, `glfwTerminate`. Print frames per second at exit.

**Also working when:** the throughput number appears and the sanitizer build
reports nothing over a short run.

### 7. `mirage/data.py`

`np.memmap` over the pixel blob, reshaped to `(-1, H, W, 3)` - no stride
arithmetic, which is the entire reason pixels and meta are separate files. Meta
is read as a structured dtype whose field widths match the C++ writer's explicit
widths exactly.

Episode-aware means a sampled window never straddles an `episode_id` boundary;
`step_idx` and `episode_id` in the meta record are what make that checkable
rather than assumed.

Doc page: **NumPy → `np.memmap`, structured dtypes**.

**Working when:** the byte-compare against a known C++-written buffer passes
(F-8), and a few thousand sampled windows all report a single `episode_id`.

### 8. `mirage/validator.py`

Roughly 50 lines with no dependencies beyond NumPy, and it emits measurements
rather than verdicts - the verdict is a threshold expression in config.

Per-frame vector: `px_count`, `bbox`, `compactness` per colour; `link_extent` and
`link_angle` per arm link; `offpalette_px` and `n_unique_colors` per frame.

Two orderings are load-bearing:

- **Compute `n_unique_colors` on the raw frame first**, then do
  nearest-palette assignment, then everything else. Post-mapping, the count
  cannot exceed the palette size, so computing it later silently stops serving
  F-2.
- **Nearest-palette by `np.argmin` over squared distances, not exact RGB
  equality.** Exact equality on a slightly off shade counts zero pixels and
  reports "block missing", conflating palette drift with a lost object.

`compactness` uses an oriented bbox from PCA on the mask coordinates, not an
axis-aligned one - both arm links revolve and a pushed block rotates, and an
axis-aligned box around a 45-degree-rotated square has 2x the area, which
collides with the occluded case. The same PCA yields `link_extent` and
`link_angle`.

Both modes are required, not optional: `measure_with_truth(frame, meta)` for
Phase 0 and `measure_pixels_only(frame)` for later phases. F-9's "zero false
positives" *is* the threshold sweep of mode 2 against mode 1 - without both modes
that criterion has no procedure.

Doc page: none. NumPy plus stdlib `xml.etree.ElementTree` for reading the palette
out of the XML.

**Working when:** mode 1 over all shards reports `n_unique_colors` <= 24, and the
sweep finds a threshold set with zero mode-2 false positives on ground-truth
frames.

---

## Gotchas, and how you would notice

### Predicted from the documentation, before writing any code

Kept because they were right. Three rows that were here originally have been
corrected against what actually happened - marked below.

| Gotcha | What breaks | How you notice |
|---|---|---|
| The offscreen buffer size defaults to 640x480 | You render into the wrong region, or get a clipped image | Pictures are the wrong size or partly black. Set `offwidth`/`offheight` in the XML `<visual><global>` block. `GlContext` now checks the two against each other and aborts, so this cannot reach a data run |
| `mjr_readPixels` returns rows bottom-up | Every picture is vertically mirrored | Obvious on first look, silent forever if you never look. **Settled: nothing in `sim/` flips it and the blob stays bottom-up; the flip lives in `WindowSampler`**, so `Shard.pixels` stays raw and F-8 has something byte-exact to compare |
| Forgetting `mjr_setBuffer` | You render to the hidden window instead of the buffer | Readback returns garbage or blank |
| `mjRND_IDCOLOR` without `mjRND_SEGMENT` | The colour-coded pass isn't colour-coded | Counts come out nonsensical |
| **Corrected.** Visualization decorations left on | Contact dots and joint axes add colours | They are already off - `mjv_defaultOption` leaves every decoration cleared, so call it and change nothing. **Do not zero the flag array to be sure:** that also clears `mjVIS_STATIC` and every worldbody geom stops drawing |
| A GL context belongs to one thread | Threads cannot share it | If you ever parallelise, use processes. This is why shards are the unit of both determinism and parallelism |
| `rand()` instead of a seeded generator | Determinism silently gone | The generate-twice-and-compare check fails, and only that check would catch it |
| **Corrected - does not apply.** Prebuilt MuJoCo plus the leak checker | Spurious leak reports from the graphics driver | MSVC ships **no leak detection at all**, so there is no suppression file to write. On this toolchain E-3 is ASan only |

### Found by running Phase 0

None of these came from reading documentation. Each is here because it was hit.
**Five of the six fail silently** - the run completes, the numbers look plausible,
and nothing reports an error.

| Gotcha | What breaks | How you notice |
|---|---|---|
| `mjv_updateScene` before any `mj_forward`/`mj_step` | An entirely black frame | **You do not.** `scene.ngeom` reads a correct 6, so a geom-count check passes. mjData's derived `xpos`/`xmat` are zero until forward dynamics runs - step or force the sim first, then update the scene |
| `<camera mode="targetbody">` with no `target=` | The camera aims nowhere and renders empty ground | **You do not.** It compiles clean and `cam_targetbodyid` is silently `-1`. Give `target=`, or set orientation with `xyaxes` as `arm_blocks.xml` does |
| The segmentation pass leaves the framebuffer holding id colours | The shard stores id colours instead of the RGB frame | **You do not.** Id colours look like a plausible flat-shaded image and fail no check downstream. Order is fixed: render RGB, read it back, *then* call `Truth::read` |
| Slicing an `np.memmap` returns a lazy view | A throughput probe measures nothing | **You do not** - it reports a wonderful number. A first loader probe read 3.4M fps having timed slice arithmetic. Force the copy: `__getitem__` must return `np.array(...)` |
| `jnt_qposadr` and `jnt_dofadr` diverge for free joints | Reads land on the wrong element | **You do not.** A free joint is 7 wide in `qpos` and 6 in `qvel`, so block1 is 9 vs 8 and block2 16 vs 14. Never compute either address; always read the array |
| `rgba * 255` does not land on exact byte values | Exact-equality palette matching reports objects missing | Byte-rounded equality calls **4 of 7 entries missing on a flawless frame** - 0.65 rounds up to 166 while 0.90 rounds down to 229, and there is no rule worth modelling. Keep the palette unrounded and match by nearest-entry `argmin` |

`glGetString` is core OpenGL 1.1, so it links directly against `opengl32.dll` on
Windows with no extension loader. Most modern GL functions do not - but you only
need this one.

---

## The two numbers to take before writing file 3

Both from Python, before any C++ exists, because they can change what you build:

1. **Per-call `mjr_readPixels` latency, timed in isolation.** **Done** - 25.4 us
   RGB, 49.6 us RGB+depth, 75.8 us with render, at P2. Two-pass render confirmed
   with a 13x margin, so neither the single-pass palette collapse nor the WGL
   pbuffer is needed. The threshold that would have forced them was ~0.5 ms.
2. **`mj_step` time alone**, to know the remaining headroom. **Done** - 10.5-10.8
   us median driven, p99 32-60 us, `bench/step_probe.py`. Render plus readback
   leaves ~1850 of the 2000 us frame, so this was the only day-1 number that could
   still break P-6; it came in 131-176x under. CPU work, so no GPU clock gate
   applies to it.

Record the GPU clock state next to each. A timing without it is not a number -
and gate it on **SM clock plus power draw** for compute, **memory clock == max**
for bandwidth. Not on `pstate == P0`, which follows the memory domain and reads
P4 during correct compute-bound work; see `README.md`, "Taking a measurement".
