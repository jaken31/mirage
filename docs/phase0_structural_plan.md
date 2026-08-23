# Phase 0: Structural Plan

Guidance for implementing `sim/`, one file at a time. Names the calls and the
order; does not write the code. Design rationale is in
`world_model_architecture.md` - not repeated here.

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
| `CMakeLists.txt` | Two build types: optimised default, sanitizer build separate | - |
| `gl_context.{h,cpp}` | GLFW init, hidden window, context current, `mjrContext`, offscreen buffer selection, the renderer-name assert | The scene, the camera, pixels |
| `policy.{h,cpp}` | Seeded generator, per-episode coin flip, the action for step *t* | Calling `mj_step` |
| `truth.{h,cpp}` | Reading `mjData`, counting segmentation pixels per block | Writing files |
| `shard_writer.{h,cpp}` | The three output files, sidecar last | Measuring anything |
| `main.cpp` | Arg parse, seed, episode and step loops, wiring | The internals of any of the above |

`truth` stays separate from `shard_writer` because `truth` is the only file whose
*inputs disappear* when the simulator is deleted. That is the seam worth
protecting.

---

## Build order, with the calls each file needs

Same order as `AGENDA.md`. Riskiest first.

### 1. `CMakeLists.txt`

Two build configurations from the start. Optimised is the default because the
real run needs it; the sanitizer build is a separate configuration you select,
carrying `-fsanitize=address,undefined -fno-omit-frame-pointer -g`. Never
`-ffast-math` in either.

**Working when:** both configurations build a trivial `main` that prints and
exits, and the sanitizer one reports nothing.

### 2. `gl_context.{h,cpp}`

Do this second and give it real time. It is the file most likely to cost you a
day.

Order of operations:

1. `glfwInit`, then `glfwWindowHint` with `GLFW_VISIBLE` set to false, then
   `glfwCreateWindow`, then `glfwMakeContextCurrent`. The hidden window exists
   only to own the context - you never draw into it.
2. Read `glGetString(GL_RENDERER)` and assert it names the RTX 5060. Reject
   `GDI Generic` and `Microsoft Basic Render Driver`.
3. `mjr_defaultContext`, then `mjr_makeContext` against the loaded `mjModel`.
4. `mjr_setBuffer` with `mjFB_OFFSCREEN`, so rendering targets the offscreen
   buffer and not the window.

Doc page: **Programming → Rendering**. Read the `record.cc` sample that ships
with MuJoCo before writing this - it does exactly this sequence.

**Working when:** the binary prints the renderer name at startup and it names
your GPU.

### 3. `policy.{h,cpp}`

One seeded `std::mt19937` per shard, seeded from the shard index. A coin flip at
episode start selects random-deltas or scripted-reach for the whole episode.
Returns an action index in the 9-action space; the caller maps it onto
`mjData.ctrl`.

Doc page: none needed. This is plain C++.

**Working when:** running with a fixed seed twice produces an identical action
sequence, and a histogram over a few thousand steps is roughly flat across all
nine.

### 4. `truth.{h,cpp}`

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

### 5. `shard_writer.{h,cpp}`

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

### 6. `main.cpp`

`mj_loadXML` → `mj_makeData` → context → loop episodes → `mj_resetData` between
them → per step: policy, `mj_step`, render, truth, write. At exit, free in
reverse order of creation: `mjr_freeContext`, `mjv_freeScene`, `mj_deleteData`,
`mj_deleteModel`, `glfwTerminate`. Print frames per second at exit.

**Working when:** the throughput number appears and the sanitizer build reports
nothing over a short run.

---

## Gotchas, and how you would notice

| Gotcha | What breaks | How you notice |
|---|---|---|
| The offscreen buffer size defaults to 640x480 | You render into the wrong region, or get a clipped image | Pictures are the wrong size or partly black. Set `offwidth`/`offheight` in the XML `<visual><global>` block |
| `mjr_readPixels` returns rows bottom-up | Every picture is vertically mirrored | Obvious on first look, silent forever if you never look. `record.cc` handles it |
| Forgetting `mjr_setBuffer` | You render to the hidden window instead of the buffer | Readback returns garbage or blank |
| `mjRND_IDCOLOR` without `mjRND_SEGMENT` | The colour-coded pass isn't colour-coded | Counts come out nonsensical |
| Visualization decorations left on | Contact dots and joint axes add colours | The 24-colour check fails and the segmentation counts are polluted. Turn them off in `mjvOption` |
| A GL context belongs to one thread | Threads cannot share it | If you ever parallelise, use processes. This is why shards are the unit of both determinism and parallelism |
| `rand()` instead of a seeded generator | Determinism silently gone | The generate-twice-and-compare check fails, and only that check would catch it |
| Prebuilt MuJoCo plus the leak checker | Spurious leak reports from the graphics driver | Expect to need a small suppression file. This is normal, not a bug in your code |

`glGetString` is core OpenGL 1.1, so it links directly against `opengl32.dll` on
Windows with no extension loader. Most modern GL functions do not - but you only
need this one.

---

## The two numbers to take before writing file 3

Both from Python, before any C++ exists, because they can change what you build:

1. **Per-call `mjr_readPixels` latency, timed in isolation.** Above ~0.5 ms the
   two-pass render cannot meet the throughput target and the single-pass palette
   render becomes mandatory. Near ~30 ms, GLFW itself is the problem and the fix
   is a hand-rolled WGL pbuffer context.
2. **`mj_step` time alone**, to know the remaining headroom.

Record the GPU power state next to each. A timing without it is not a number.
