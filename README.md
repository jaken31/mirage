# mirage

Train an action-conditioned world model on a MuJoCo manipulation scene, delete the
simulator, and control the learned model in real time at 30 fps.

## Docs

| File | What |
|---|---|
| [docs/writeup_part1.md](docs/writeup_part1.md) | **The writeup, part one.** Phases 0-1, the dataset and the tokenizer, refutation-first. Start here to know what this *is*; start at AGENDA to know what to *do* |
| [AGENDA.md](AGENDA.md) | What to do next. Start here. |
| [docs/world_model_architecture.md](docs/world_model_architecture.md) | How the pieces meet: interfaces, shard format, provenance, validator |
| [docs/world_model_requirements.md](docs/world_model_requirements.md) | Tiered requirements and acceptance tests |
| [docs/world_model_ingredients.md](docs/world_model_ingredients.md) | Scope, model config, latency budget, phases |
| [docs/world_model_learning_roadmap.md](docs/world_model_learning_roadmap.md) | Study plan, sequenced against the phases |
| [docs/timeline.md](docs/timeline.md) | **Plain English.** Schedule, gates, branch points, risks, study woven in. For non-engineering readers |
| [docs/decision_notes.md](docs/decision_notes.md) | **Plain English.** Every decision with its reason, its wrong-if signal, and 1-2 fallbacks |
| [docs/phase0_report.md](docs/phase0_report.md) | **Derived.** Phase 0 completion report: output, architecture, measured results, open items |

The last three are derived from the architecture and requirements docs. When a
decision changes, change it there first.

## Environment

| Item | Requirement | Status on this machine |
|---|---|---|
| GPU | RTX 5060, 8 GB, sm_120 | confirmed, capability (12,0) - **laptop variant** |
| OS | **Native Windows** | Windows 11, Python 3.14.2 |
| CUDA | 12.8+ (Blackwell) | 13.0 |
| PyTorch | cu128+ | 2.9.1+cu130 |
| GL backend | **GLFW**, offscreen. EGL is unavailable on Windows MuJoCo; OSMesa is a CPU rasterizer and an anti-choice | **verified 2026-08-26** - `sim/gl_context.cpp` reads `NVIDIA GeForce RTX 5060 Laptop GPU/PCIe/SSE2`. The day-1 blocker cleared: `mjr_readPixels` is **25.4 us** RGB at 64x64, not the ~30 ms the MuJoCo discussion reported |
| MuJoCo | 3.x, C API | **3.12.0, exercised** - 300,000 frames generated through it |
| Compiler | C++20 | **MSVC, verified** - CMake generator `Visual Studio 18 2026`, `sim/main.cpp` prints `202002`. Both `sim/build/` and `sim/build-asan/` compile and run |

**WSL2 is out, and this is settled.** Its GPU graphics path is broken on this
machine - no `/dev/dri` node, Mesa falls back to a CPU rasterizer. Evidence in
`CLAUDE.md`, produced by `bench/egl_probe.py`. Do not re-litigate it.

### Python packages

`requirements.txt` pins the exact stack every number in `runs.jsonl` was measured
on. torch is the CUDA 13.0 build and is not on PyPI, so it installs first and
separately:

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

```bash
pip install -r requirements.txt
```

Then confirm the GPU is actually reachable. A CPU-only torch installs quietly and
turns every timing in this repo into a meaningless number:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It must print `2.9.1+cu130 True`. The pins are `==` rather than `>=` on purpose:
the `nn.Upsample` use-after-free that killed two 60-epoch runs is specific to
this python/torch pair, and a reproduction attempt elsewhere is worth nothing
without them.

## Checks

Every module owns a `_self_check()` and stays runnable alone -
`python -m mirage.data`, `python -m mirage.validator`, and so on. To run all five:

```bash
python check.py
```

It exits nonzero if any fails, and orders them cheapest first, so a break in
`config` surfaces in seconds rather than after `data` has swept 300,000 frames.
`config`, `logging` and `fsq` need no dataset at all; `validator` and `data` fall
back to the committed 40-frame fixture in `mirage/fixtures/` when `data/shards`
is empty. This is a runner, not a test framework, and it does not reverse the
per-module choice recorded in `docs/phase0_debt_checklist.md`.

It also validates `docs/canonical_numbers.md`, the register holding the current
value of every number this project quotes in more than one place. Four checks, all
chosen because they cannot false-positive: no id defined twice, every entry names
a source, every `r<N>` source points at a `runs.jsonl` row that exists, and every
`NUM-` id cited anywhere in the tree is defined. **Cite the id, not the value** -
that is what lets a figure move in one place instead of seventeen.

## Taking a measurement

**Do not gate on `pstate == P0`.** That rule was refuted 2026-08-23. The reported
pstate follows the **memory** clock domain, so a correct compute-bound run reads
P4 while the SMs hold 2662 MHz of 3090 at 99 W of a 100 W cap - fully clocked up.
P0 appears only under memory-bound load. The old rule would have rejected every
valid compute number this machine can produce.

No single load clocks both domains, so gate them separately and record the gate
next to the number:

| Measuring | Gate on | Invalid when |
|---|---|---|
| Compute (TFLOP/s) | `clocks.current.sm` >= 80% of `clocks.max.sm`, and `power.draw` >= 80% of `enforced.power.limit` | SMs at idle clocks, or draw far under the enforced limit |
| Bandwidth (GB/s) | `clocks.current.memory` == `clocks.max.memory` | anything below max |

`bench/gpu_probe.py` runs both phases and prints the gate beside each figure.

**Sample the thermal counters, not the throttle flags.** The instantaneous flags
read `Not Active` through a 45 W cap. The evidence lives in the counters:

```bash
nvidia-smi -q -d PERFORMANCE
```

A chassis cooling fix moved the enforced power limit 55 -> 100 W and fp16 matmul
**3.0 -> 27.6 TFLOP/s - 9x from cooling alone**, with no throttle flag ever set.

**Preconditions for any Phase 3/4 timing run:** mains power, the Windows and
NVIDIA performance profiles selected, `nvidia-smi --lock-gpu-clocks` and
`--lock-memory-clocks` held for the run (needs admin), and desktop GPU consumers
closed - browsers, Teams, Discord, and the NVIDIA and Overwolf overlays all hold
GPU contexts under WDDM.

**Determinism caveat (F-4).** Bit-exact replay holds for a **fixed driver and
build**. `/fp:fast` stays off; enabling it forfeits the guarantee. F-4 is tested
by generating twice at one seed and comparing the pixel blobs - there is no
`--replay` mode.

## Build

**E-2, verified 2026-08-28** by running exactly these commands in a directory
that had never held this project. Nothing is vendored: MuJoCo 3.12.0, GLFW 3.5.1
and nlohmann/json 3.12.0 are all fetched by CMake at configure time, pinned by
SHA256 or tag in `sim/CMakeLists.txt`, so the first configure needs a network.

Needs Visual Studio 2026 with the "Desktop development with C++" workload
(measured against MSVC 19.50.35728, toolset 14.50.35717), CMake >= 3.20
(measured on 4.3.1), and git.

```bash
git clone <url> mirage
cd mirage
cmake -S sim -B sim/build -G "Visual Studio 18 2026" -A x64
cmake --build sim/build --config Release
```

The sanitizer build is a second, separate build directory:

```bash
cmake -S sim -B sim/build-asan -G "Visual Studio 18 2026" -A x64 -DMIRAGE_ASAN=ON
cmake --build sim/build-asan --config Release
```

Then smoke-test it. This is a real 40-frame generation run - it loads the scene,
runs both C++ self-checks, creates the GL context, and writes a shard - and it
regenerates the committed self-check fixture, so it is also the command to reach
for when `scene/arm_blocks.xml` or the `sim` config section changes:

```bash
python -c "from mirage.config import load; print(load('mirage/fixtures/fixture.json').data_hash)"
./sim/build/Release/mirage_sim.exe mirage/fixtures/fixture.json --data-hash <that hex> --git-sha $(git rev-parse HEAD)
```

It must print `GL_RENDERER: NVIDIA ...`. `GDI Generic` or `Microsoft Basic
Render Driver` means the software rasterizer, which is a failed build for our
purposes even though it compiled.

Four things the clean run measured, none of which a machine that has built once
would show you:

| | |
|---|---|
| **Do not build under `%TEMP%`** | MSBuild's FileTracker refuses: `MSB8029` then `FTK1011: could not create the new file tracking log file`. It fails during CMake's compiler probe, so it reads as "no working C++ compiler" rather than as a path problem |
| **Each build dir re-fetches** | `sim/build` and `sim/build-asan` keep separate `_deps/`, so the second configure downloads MuJoCo and re-clones GLFW again. 14-17 s each, not shared |
| **The DLLs are copied, not found** | `mujoco.dll` lands beside the binary by a POST_BUILD step, and the ASan build adds `clang_rt.asan_dynamic-x86_64.dll` from the MSVC bin directory. Both confirmed present in their output directories, and both binaries start from a plain shell rather than only from a Developer prompt |
| **Cost from empty** | Release 16.5 s configure + 11.6 s build; ASan 14.7 s + 9.7 s. Both from nothing, on this machine |

The ASan binary was run over the same 40-frame generation, into a throwaway
shard directory: no sanitizer report, and its `.pixels` and `.meta` blobs are
**byte-identical** to the Release build's. Same compiler, machine and driver, so
this widens F-4's determinism from "same build, twice" to "two build
configurations" - it does not say anything about a different toolchain.

Run every command from the repo root. Config paths are repo-relative and the
binary says so rather than guessing.
