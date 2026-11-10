# mirage

mirage trains a world model of a simulated robot arm. A MuJoCo scene generates
300,000 frames at 64x64, a tokenizer compresses each frame into tokens, and an
action-conditioned model learns to predict the next frame from the current one
and the arm's action. The end goal is to delete the simulator and drive the
learned model in real time at 30 fps.

## Where to start

- **To understand the project:** [docs/writeup_part1.md](docs/writeup_part1.md)
  covers Phases 0-1 (the dataset and the tokenizer) and what the measurements
  showed.
- **To work on it:** [AGENDA.md](AGENDA.md) is the ordered list of what to do
  next.

Other docs:

| File | What it covers |
|---|---|
| [docs/world_model_architecture.md](docs/world_model_architecture.md) | How the pieces fit: interfaces, shard format, provenance, validator, and the verification log |
| [docs/world_model_requirements.md](docs/world_model_requirements.md) | Requirements and their acceptance tests |
| [docs/world_model_ingredients.md](docs/world_model_ingredients.md) | Scope, model config, latency budget, phases |
| [docs/world_model_learning_roadmap.md](docs/world_model_learning_roadmap.md) | Study plan, sequenced against the phases |
| [docs/canonical_numbers.md](docs/canonical_numbers.md) | The current value of every number quoted in more than one place |
| [docs/timeline.md](docs/timeline.md) | Plain-English schedule, gates and risks |
| [docs/decision_notes.md](docs/decision_notes.md) | Plain-English list of every decision, why it was made, and what would change it |
| [docs/phase0_report.md](docs/phase0_report.md) | Phase 0 completion report |

The last three are summaries of the architecture and requirements docs. When a
decision changes, update the source doc first.

## Requirements

| | Needed | Tested on |
|---|---|---|
| GPU | NVIDIA, CUDA 12.8+ (Blackwell) | RTX 5060 Laptop, 8 GB, CUDA 13.0 |
| OS | Native Windows or native Linux | Windows 11 and Omarchy (Arch Linux), dual-boot |
| Python | 3.14 | 3.14.2 (Windows), 3.14.7 (Linux) |
| PyTorch | CUDA build, cu128+ | 2.9.1+cu130 |
| Compiler | C++20 | MSVC 19.50 (Windows); GCC 16.2.1 and Clang 22.1.8 (Linux) |
| CMake | 3.20+ | 4.3.1 (Windows), 4.4.3 (Linux) |
| MuJoCo | 3.x, C API | 3.12.0, fetched by CMake |

The simulator renders offscreen through GLFW on both platforms. Windows MuJoCo
has no EGL, and OSMesa is a CPU renderer, so neither is used.

WSL2 does not work: on this machine its GPU graphics path never creates a
`/dev/dri` node, so rendering falls back to the CPU. `bench/egl_probe.py` is the
evidence, and the details are in [AGENTS.md](AGENTS.md).

## Setup

### 1. Python packages

On Linux, create a venv first. Arch's system Python refuses `pip install`
(`externally-managed-environment`):

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

torch is the CUDA 13.0 build, which is not on PyPI, so install it first:

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

```bash
pip install -r requirements.txt
```

Then confirm torch can see the GPU. It must print `2.9.1+cu130 True`. A CPU-only
torch installs without complaint and makes every timing in this repo meaningless.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The versions are pinned with `==` on purpose. Every number in `runs.jsonl` was
measured on this exact stack, and the `nn.Upsample` use-after-free that killed
two 60-epoch runs is tied to this Python/torch pair.

### 2. Build the simulator

Nothing is vendored. CMake downloads MuJoCo 3.12.0, GLFW 3.5.1 and nlohmann/json
3.12.0 at configure time, pinned by SHA256 or tag in `sim/CMakeLists.txt`, so the
first configure needs a network. Run every command from the repo root; config
paths are repo-relative.

These steps were verified from a fresh clone on Windows (2026-08-28) and Linux
(2026-09-23).

**Windows** needs Visual Studio 2026 with the "Desktop development with C++"
workload, CMake and git.

```bash
cmake -S sim -B sim/build -G "Visual Studio 18 2026" -A x64
```

```bash
cmake --build sim/build --config Release
```

**Linux** needs CMake, Ninja, and a C++20 compiler. GLFW builds both its Wayland
and X11 backends, so it needs their development files. On Arch those come from
`wayland`, `wayland-protocols`, `libxkbcommon`, `libx11`, `libxrandr`,
`libxinerama`, `libxcursor` and `libxi` (which of these are strictly required
has not been checked).

```bash
cmake -S sim -B sim/build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build sim/build
```

### 3. Smoke-test the build

This generates 40 real frames: it loads the scene, runs the C++ self-checks,
creates the GL context and writes a shard. It also regenerates the committed
test fixture, so rerun it whenever `scene/arm_blocks.xml` or the `sim` config
section changes.

Get the config's data hash:

```bash
python -c "from mirage.config import load; print(load('mirage/fixtures/fixture.json').data_hash)"
```

Then run the simulator with it. On Windows:

```bash
./sim/build/Release/mirage_sim.exe mirage/fixtures/fixture.json --data-hash <hash> --git-sha $(git rev-parse HEAD)
```

On Linux:

```bash
./sim/build/mirage_sim mirage/fixtures/fixture.json --data-hash <hash> --git-sha $(git rev-parse HEAD)
```

It must print `GL_RENDERER: NVIDIA ...`. Anything else means the build is not
usable, even though it compiled:

- On Windows, `GDI Generic` or `Microsoft Basic Render Driver` is the software
  renderer.
- On Linux, the binary refuses to run on anything but NVIDIA. See
  [Linux GPU selection](#linux-gpu-selection).

### 4. Run the checks

```bash
python check.py
```

This runs every module's `_self_check()`, cheapest first, and exits nonzero if
any fails. Each module can also run alone, e.g. `python -m mirage.data`.

You do not need the full dataset. Without it, the checks that need frames fall
back to the 40-frame fixture in `mirage/fixtures/`, and the checks that need the
token cache or a GPU skip that part.

`check.py` also validates [docs/canonical_numbers.md](docs/canonical_numbers.md):
no id is defined twice, every entry names a source, every `r<N>` source exists in
`runs.jsonl`, and every `NUM-` id cited anywhere in the repo is defined. When a
number changes, update the register first, then search the docs for the old
value.

## Build

### Sanitizer build

The sanitizer build lives in its own build directory.

Windows (AddressSanitizer):

```bash
cmake -S sim -B sim/build-asan -G "Visual Studio 18 2026" -A x64 -DMIRAGE_ASAN=ON
```

```bash
cmake --build sim/build-asan --config Release
```

Linux (AddressSanitizer plus UndefinedBehaviorSanitizer). This needs Clang,
because MuJoCo 3.12.0's `mjsan.h` does not compile under GCC with ASan on; CMake
stops with that message if you try.

```bash
cmake -S sim -B sim/build-asan -G Ninja -DCMAKE_BUILD_TYPE=Release -DMIRAGE_ASAN=ON -DCMAKE_CXX_COMPILER=clang++
```

```bash
cmake --build sim/build-asan
```

On Linux, run the sanitizer build with `ASAN_OPTIONS=detect_leaks=0`. Every leak
LeakSanitizer reports is in a library, not this project:

- The NVIDIA driver leaks about 1.7 KB per run inside `libdbus` and an
  already-unloaded module, so a suppression file cannot name it.
- GLFW 3.5.1's Wayland backend leaks about 140 KB each time the compositor sends
  a new keymap, which happens on window focus changes (6.7 MB over one 5-minute
  run).

To leak-check this project's own code, run the fixture with leak checking on and
expect only the driver's ~1.7 KB.

The ASan build of the 40-frame fixture produced pixel and metadata blobs
byte-identical to the Release build, with no sanitizer report.

### Windows build pitfalls

These only show up on a machine that has never built the project:

- **Do not build under `%TEMP%`.** MSBuild's FileTracker fails with `MSB8029`
  and `FTK1011` during CMake's compiler probe, which looks like "no working C++
  compiler" rather than a path problem.
- **Each build directory downloads its own dependencies.** `sim/build` and
  `sim/build-asan` keep separate `_deps/`, about 15 s each.
- **DLLs are copied next to the binary.** `mujoco.dll` (and, for ASan,
  `clang_rt.asan_dynamic-x86_64.dll`) land beside the executable, so both
  binaries run from a plain shell, not only a Developer prompt.
- **Build time from empty:** Release 16.5 s configure + 11.6 s build; ASan 14.7 s
  + 9.7 s.

On Linux, a build from empty took 5.2 s to configure and 5.3 s to build, with
zero warnings under `-Werror`. CMake fetches MuJoCo's Linux tarball and copies
`libmujoco.so.3.12.0` next to the binary, which finds it through an `$ORIGIN`
RUNPATH.

### Linux GPU selection

The binary picks the NVIDIA GPU itself; you do not need to export anything.

This laptop has two GPUs. Under Wayland, GLFW picks the Intel one by default
(`Mesa Intel(R) Graphics (ARL)`). It is real hardware, but it runs about 10x
slower (roughly 1,000 vs 10,000 frames/s) and renders a slightly different
dataset under the same `data_hash`.

So `sim/gl_context.cpp` sets PRIME render offload (`__NV_PRIME_RENDER_OFFLOAD=1`,
plus `__GLX_VENDOR_LIBRARY_NAME=nvidia` for X11) before GLFW starts, overriding
the shell. It then refuses to render unless `GL_RENDERER` contains `NVIDIA`:

- Mesa's `llvmpipe` or `softpipe` stops with `software GL, not hardware`.
- Any other GPU stops with `GL_RENDERER is '...', not NVIDIA`.

On success it prints `GL_RENDERER: NVIDIA ...` and
`GL_VERSION: 4.6.0 NVIDIA <driver>`.

## Taking a measurement

GPU numbers on this laptop are only valid if the GPU was actually clocked up.
Gate each kind of measurement separately and record the gate next to the number:

| Measuring | Valid when |
|---|---|
| Compute (TFLOP/s) | SM clock >= 80% of max, and power draw >= 80% of the enforced limit |
| Bandwidth (GB/s) | Memory clock == max |

`bench/gpu_probe.py` runs both and prints the gate beside each figure. It also
requires the SM clock to stay within 5% across the run. `--self-check` tests the
verdict logic without a GPU.

**Do not gate on `pstate == P0`.** The reported pstate follows the memory clock,
so a correct compute-bound run reads P4 while the SMs are at 2662 of 3090 MHz,
drawing 99 W of a 100 W cap. That rule was refuted on 2026-08-23.

**Read the thermal counters, not the throttle flags.** The throttle flags read
`Not Active` even while the GPU was capped at 45 W. The real evidence is in:

```bash
nvidia-smi -q -d PERFORMANCE
```

Cooling matters a lot: a chassis cooling fix raised the power limit from 55 to
100 W and fp16 matmul from 3.0 to 27.6 TFLOP/s, with no throttle flag ever set.

**Before any Phase 3/4 timing run:** plug into mains power, select the Windows
and NVIDIA performance profiles, lock clocks with `nvidia-smi --lock-gpu-clocks`
and `--lock-memory-clocks` (needs admin), and close anything else using the GPU
(browsers, Teams, Discord, NVIDIA and Overwolf overlays).

## Determinism

The same seed produces the same bytes (requirement F-4), as long as the GPU
vendor stays the same. Test it by generating twice at one seed and comparing the
pixel blobs; there is no `--replay` mode. Keep `/fp:fast` off, or the guarantee
is gone.

Measured on 2026-09-23 (`runs.jsonl` r55):

- A GCC build on Linux, on the same NVIDIA GPU, regenerated all 300,000 frames
  byte-identical to the Windows set. OS and compiler do not change the output.
- The Intel integrated GPU did not: 442 frames differ by one pixel, and 71
  `visible_px` counts differ by 1. The GPU vendor changes the output.

Still open: whether the NVIDIA driver version matters. Linux ran 610.57.04, but
the Windows driver version was not recorded. Shards now store it in their
sidecar (`gl_version`), so the next comparison can answer it.
