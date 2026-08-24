# Robot World Model: Learning Roadmap

**Assumption:** strong Python, TypeScript, SQL, full-stack, LLM APIs. New to: training models from scratch, discrete representations, world models, GPU programming, C++, physics simulators, performance profiling.

**Rule: just-in-time, not up front.** Each track is placed a week before it becomes load-bearing. Only Tier 0 happens before you start.

**Honest total: 65 to 85 hours of study spread across the project.** That is on top of the build, not instead of it.

---

## Tier 0 — before writing any code (~15 hours)

**Karpathy, "Let's build GPT: from scratch, in code, spelled out"**
https://www.youtube.com/watch?v=kCc8FmEb1nY

Non-negotiable and non-substitutable. Your dynamics model is a GPT that eats image tokens instead of text tokens. If you do not understand attention, causal masking, and next-token prediction at the code level, every later decision is guesswork. Two hours of video, plan a full day to code along.

**Karpathy, "Neural Networks: Zero to Hero"** (course home: https://karpathy.ai/zero-to-hero.html)
Watch lectures 1 through 6 only if backprop, activations, or BatchNorm feel shaky. Skip if comfortable. Skip the tokenizer lecture entirely, BPE is irrelevant here.

**karpathy/build-nanogpt** — https://github.com/karpathy/build-nanogpt
Commits are kept atomic so you can walk the history. This is the structural reference for your training code. Read it, do not clone it.

---

## Track A — Discrete representations (before Phase 1, ~8 hours)

| Resource | Use |
|---|---|
| zalandoresearch/pytorch-vq-vae (notebook) | clearest walkthrough of the quantizer and straight-through estimator |
| explainingai-code/VQVAE-Pytorch | has a `run_simple_vqvae.py` minimal path, good second read |
| **FSQ paper 2309.15505** | what you are actually implementing. Read section 3 and the levels table |

Read VQ first even though you are building FSQ. FSQ only makes sense as "the thing that deletes VQ's problems," and the commitment-loss and codebook-collapse machinery is what it deletes.

---

## Track B — World models for robotics (during Phase 1, ~8 hours)

| Paper | Why | Depth |
|---|---|---|
| Ha & Schmidhuber, World Models (2018) | the origin, V/M/C decomposition, 64x64 latents | read fully |
| **Causal World Modeling for Robot Control (2601.21998)** | autoregressive video+action, KV cache, real-time. **This is your architecture** | read fully |
| MineWorld (2504.08388) | closest open implementation, same token-interleaving scheme | read fully, skim the code |
| World Model for Robot Learning survey (2605.00080) | positions the whole field, tells you what you are and are not doing | skim, use as a map |
| Genie (2402.15391) | ST-transformer, the fork you did not take | read the architecture section |

**Dreamer line moved off the skip list.** DreamerV1-V3, RSSM, PlaNet, DayDreamer. In a robotics framing this lineage is context you will be asked about, and Dreamer is the reference point everyone means by "world model" in robot learning. Read the DreamerV3 paper once for vocabulary. Do **not** implement any of it. It is the RL branch, latent imagination for policy learning, and you are building a simulator rather than an agent. Know it, skip it.

Useful framing to hold onto: diffusion video models predict better than you will and are far too slow to close a control loop. Autoregressive plus KV cache is the latency-driven choice. That is the sentence your whole project is an argument for.

---

## Track C — C++ and MuJoCo (Phase 0, ~12 hours)

**learncpp.com** — https://www.learncpp.com/
The standard free resource, actively maintained through C++23. Do not read all 28 chapters. You need: chapters 1 to 12 (basics through functions), 16 to 17 (arrays, `std::vector`), 14 to 15 (classes), and 20 (lambdas). Skip templates, inheritance, and metaprogramming until something forces you.

**MuJoCo documentation** — https://mujoco.readthedocs.io/
Three sections only:
- **Overview and XML reference** — enough to write `arm_blocks.xml`. Focus on `<worldbody>`, joints, geoms, actuators, and the `<visual>` block that controls your flat render.
- **Programming / Rendering** — `mjr_render`, offscreen buffers, and the EGL context. This is where your C++ harness lives.
- **`record.cc` sample** — in the MuJoCo GitHub repo under `sample/`, **not** in the pip wheel (checked: `mujoco` 3.12.0 ships headers and two test XMLs only). Does exactly what your harness does: steps physics, renders offscreen, dumps pixels. Read it before writing yours.

**Supplement with:** build with `/W4 /WX /fsanitize=address` from the very first file (MSVC has no UBSan). Sanitizers are the difference between C++ being tolerable and C++ being a nightmare when your instincts are not yet C++ instincts.

**Note:** you are not writing a physics engine. You are writing a data pipeline against a C API. That is more useful, and more like real infrastructure work, than hand-rolling collision detection.

---

## Track D — GPU programming (before Phase 4, ~20 hours)

This is the big one, and the sequence matters. **Pull steps 1 and 2 into Phase 2 background time.** Phase 4 is a difficulty cliff and this is how you make it smaller.

**1. srush/GPU-Puzzles** — https://github.com/srush/GPU-Puzzles
Interactive, Colab, no GPU needed to start. Uses Numba to map Python directly to CUDA kernels, so it looks like Python but teaches the real thread/block model. Builds intuition in a few hours. Start here, always.

**2. Simon Boehm, CUDA matmul walkthrough** — https://siboehm.com/articles/22/CUDA-MMM
The best single piece of writing on the GPU memory model. Ten kernel iterations, each faster than the last, each with the reason explained. Fork https://github.com/siboehm/SGEMM_CUDA and run them.

**3. GPU MODE lectures** — https://github.com/gpu-mode/lectures and the YouTube channel
Reading group run by Andreas Köpf and Mark Saroufim. Relevant ones for you, not all of them:
- Lecture 1: profiling and integrating CUDA kernels in PyTorch
- Lecture 8: CUDA performance checklist (coalescing, occupancy, divergence, tiling)
- Lecture 12: Flash Attention
- Lecture 14: Practitioner's guide to Triton
- Lecture 16: hands-on profiling
- Lecture 18: fused kernels

**4. gpu-mode/Triton-Puzzles** — https://github.com/gpu-mode/Triton-Puzzles
Runs on the Triton interpreter, no GPU required. Builds to Flash Attention and quantized kernels. Roughly two days. This is your direct preparation for the fused-block rung.

**5. PMPP book** (Programming Massively Parallel Processors) — reference, not a read-through. Look things up.

---

## Track E — Inference optimization (before Phase 4, ~8 hours)

**pytorch-labs/gpt-fast** — https://github.com/pytorch-labs/gpt-fast
**The single most important repo for this project.** Under 1000 lines of pure PyTorch implementing almost exactly your ladder: static KV cache, `torch.compile`, CUDA graphs, INT8 weight-only quantization, speculative decoding. Roughly 10x over baseline with no accuracy loss.

**Accompanying blog** — https://pytorch.org/blog/accelerating-generative-ai-2/
Read this before the code. It explains *why* each step, including one gotcha you will otherwise hit blind: a dynamically growing KV cache breaks CUDA graph capture, so you must statically allocate the max size and mask the unused portion. That constraint should be in your engine from the first commit.

**Diagonal Decoding (2503.14070)** — read the method section during Phase 3. If your tokenizer lands on the 144-token path this stops being optional and becomes a required rung, so know it before you need it.

Read gpt-fast for structure, then write your own. Copying it means you skip the part that teaches you anything.

---

## Track F — Profiling (Phase 3, ~4 hours)

`torch.profiler` first, Nsight Systems second. You need one specific capability: reading the CPU dispatch timeline against the GPU busy timeline and seeing the gaps. GPU MODE lectures 1 and 16 cover this directly. Everything else in Nsight is a distraction until you have a specific question.

---

## Sequencing against phases

| When | Study | Hours |
|---|---|---|
| Before start | Tier 0: build GPT | 15 |
| Phase 0 (sim + harness) | Track C: learncpp subset, MuJoCo docs, `record.cc` | 12 |
| Phase 0 to 1 | Track A: VQ then FSQ | 8 |
| Phase 1 (background) | Track B: robot world model papers | 8 |
| Phase 2 (background) | **Track D steps 1 and 2: GPU Puzzles, Boehm** | 8 |
| Phase 3 | Track F: profiling | 4 |
| Phase 3 to 4 | Track E: gpt-fast blog then code, DiagD method | 8 |
| Phase 4 | Track D steps 3 and 4: GPU MODE, Triton Puzzles | 12 |

---

## Explicitly skip

| Topic | Why |
|---|---|
| Implementing Dreamer / RSSM | read for vocabulary, do not build. Wrong branch |
| Diffusion video models | too slow for a control loop, which is your entire thesis |
| VLAs, policy learning, imitation learning | you are building the simulator, not the policy |
| Sim-to-real, domain randomization | explicit non-requirement |
| Grasping and contact-rich dexterity | your arm pushes blocks |
| BPE and text tokenization | your tokens are image patches |
| Distributed training, FSDP, NCCL | one GPU |
| CUTLASS, tensor core intrinsics | far past where Triton stops being enough |
| C++ templates, inheritance, metaprogramming | a data pipeline needs none of it |
| MuJoCo MJX / JAX | GPU-parallel batched sim, useful for RL, irrelevant for one data run |
| Full PMPP read-through | reference only |

## Communities

- **GPU MODE Discord** (https://discord.gg/gpumode) — the `#triton-puzzles` channel specifically, and generally where the kernel people are
- **MuJoCo GitHub Discussions** — where the EGL and offscreen-rendering answers actually live
- **Karpathy Zero to Hero Discord** — `#nanoGPT` channel

## Reality check

MineWorld's 300M model reaches about 5.9 fps and their 1.2B about 3.0 fps, on roughly 340 tokens per frame. You are running 64 to 144 tokens per frame with a 15M model. Your 30fps target is aggressive but the arithmetic supports it.

Three things that are normal and not signals to abandon the plan:

1. **Phase 2 output looks bad for a while.** Rollouts drift, the arm smears. Expected.
2. **Phase 4 numbers look flat at first.** Profile before changing anything. Guessing at optimizations is how weeks disappear.
3. **Object permanence (Q-6) may not emerge at all.** It is a "should," not a "must." Measure it, report it either way, and ship without it if it does not come.
