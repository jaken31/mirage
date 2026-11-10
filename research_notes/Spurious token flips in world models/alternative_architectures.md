# Alternative world-model architectures for copy-heavy, static-background scenes

Scope: what to switch to if cleaning R1's spurious flips out of the targets does not let the 192-dim / 4-layer token dynamics model beat copy-previous-frame. Setting: 64x64 MuJoCo arm frames, 300k frames, flat palette, mostly static background, one RTX 5060 Laptop (8 GB), about 2 weeks for the dynamics phase. Research date: 2026-11-01.

## Continuous-latent world models vs discrete tokens for small simulated scenes

### Takeaway
At this scale, the strongest evidence for consistent static content comes from DIAMOND, which predicts pixels directly with a 4M-parameter diffusion U-Net. Its paper names IRIS's per-frame discrete tokens as the source of frame-to-frame inconsistency. DreamerV3 and TD-MPC2 fit the GPU but answer a different question: they are trained for control, not for rendering frames. TD-MPC2 cannot render frames at all.

### Cited Findings
- **DIAMOND (Alonso et al., 2024, NeurIPS 2024).**
  - Atari model: a 2D U-Net with 4 stages of [64,64,64,64] channels and 2 residual blocks per stage. It is conditioned on the last 4 frames by frame stacking, and on actions and diffusion time through adaptive group norm. The diffusion model is about 4M parameters; the full agent is 13M. — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)
  - Uses EDM, not DDPM. With DDPM and fewer than about 10 denoising steps, imagined trajectories show "severe compounding error". EDM stays stable even at 1 step. The paper uses 3 Euler steps on Atari, and says 1 step blurs outcomes that could go several ways (Boxing). — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)
  - Compute: 64x64x3 frames, about 12 GB VRAM and 2.9 days per run on one RTX 4090. For comparison, the paper lists IRIS at 30M parameters and 4.1 days, and DreamerV3 at 18M parameters and under 1 day. — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)
  - Atari 100k: mean human-normalized score (HNS) 1.459, against IRIS 1.046, DreamerV3 1.097 and STORM 1.266. — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)
  - Qualitative claim: IRIS imaginations show inconsistencies between frames, such as enemies rendered as rewards, and bricks and score that change in Breakout. The paper says these do not occur with DIAMOND. — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)
  - The abstract's argument against discrete latents: "compression into a compact discrete representation may ignore visual details that are important for reinforcement learning". — [arXiv 2405.12399](https://arxiv.org/abs/2405.12399)
- **DreamerV3 (Hafner et al., 2023; Nature 2025).**
  - Latents are sampled from "a vector of softmax distributions", mixed with 1% uniform noise.
  - The KL terms use free bits clipped at 1 nat, with β_dyn = 1 and β_rep = 0.1.
  - The decoder reconstructs the inputs, using a symlog squared loss on vector inputs.
  - Model sizes run from 12M to 400M parameters. Each agent trained on one A100.
  - The paper notes that games with static backgrounds, where single pixels can matter, need a weak representation regularizer, and says free bits resolve this with fixed hyperparameters.
  - Source: [arXiv 2301.04104 html](https://arxiv.org/html/2301.04104)
- **TD-MPC2 (Hansen et al., 2023; ICLR 2024).**
  - Has no decoder and does no pixel reconstruction. It trains its latent by predicting future latents, rewards and value (TD) targets.
  - Its latent uses SimNorm, which splits the vector into groups of 8 and softmaxes each group.
  - Sizes run from 1M to 317M parameters; 1M takes 3.7 and 5M takes 4.2 GPU-days on an RTX 3090, for the 80-task multitask setup.
  - Supports 64x64 images through a 4-layer CNN encoder.
  - The authors argue that predicting raw future observations over long horizons is hard and does not necessarily lead to effective control.
  - Source: [arXiv 2310.16828 html](https://arxiv.org/html/2310.16828)

### Inferences
- **DIAMOND is the right-sized reference.** Its scale (64x64, 4M U-Net, about 400k Atari frames) is close to mirage's (64x64, 300k frames). Its 12 GB VRAM figure is for its own batch size, so fitting 8 GB probably means a smaller batch or bf16; this is not verified. The 2.9 days on a 4090 is for the full RL loop. A world model trained alone on a fixed dataset should cost less, but expect the laptop 5060 to be noticeably slower than a 4090.
- **Pixel prediction removes the flip problem at its source.** Because the U-Net sees the previous frames stacked in its input, "nothing changed" is a near-identity map it can learn. There are no tokenizer codes to flip.
- **A copy-dominant scene will not stop DreamerV3 from working.** Its latent is still discrete categoricals (as in its earlier versions; the size is not stated in the fetched text), but its loss is pixel reconstruction, not cross-entropy over per-frame codes. A quiet background costs it almost nothing. The catch is that it would discard the existing FSQ pipeline, and its strengths are in control.
- **TD-MPC2 is the wrong tool here.** It cannot produce frames, so it cannot be scored against copy-previous-frame in pixel space.

### Gaps
- DIAMOND's quantitative compounding-error analysis (Appendix K) was not read; the fetch truncated before it.
- No source found reports DIAMOND or DreamerV3 trained on an 8 GB GPU.
- The size of DreamerV3's categorical latent is not stated in the fetched text.

## Conditional / delta tokenization (Delta-IRIS, iVideoGPT, object-centric slots, keyframe + residual)

### Takeaway
Delta-IRIS and iVideoGPT both fix exactly the failure mirage has. Each encodes a frame *given* the previous frames, so an unchanged region costs no tokens. Delta-IRIS is the small-compute version: 4 tokens per frame, a 3-layer transformer, one A100. Object-centric slot models also handle static backgrounds well, but they need a slot encoder pretrained first, so they carry more integration risk than a 2-week budget allows.

### Cited Findings
- **Delta-IRIS (Micheli, Alonso, Fleuret, 2024, ICML 2024).**
  - Encoder: maps frame x_t to K discrete "Δ-tokens" (Δ for delta, the change between steps). It is conditioned on past frames and actions, with actions embedded and concatenated to the frames channel-wise.
  - Decoder: reconstructs x_t from past frames, actions and the Δ-tokens.
  - Settings: K = 4 tokens per frame, vocabulary 1024. IRIS used 16 or 64 tokens per frame; the paper compares 40 bits per frame against 160.
  - Losses: L1 + L2 + max-pixel + commitment, with an EMA codebook update.
  - Source: [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
  - The transformer's input interleaves continuous "I-tokens" (a per-frame CNN embedding of the full frame, trained end to end), action tokens and Δ-tokens. Transformer: 3 layers, width 512, 8 heads, 21-timestep context. — [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
  - Without I-tokens the model glitches through walls and water, and the Crafter score at 10M frames drops from 42.47 to 25.92. — [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
  - Compute: one A100 40GB. Collection runs at 20 FPS against 2 FPS for IRIS with 64 tokens, about 10x faster. Atari 100k takes 26 hours, a 5x speedup over IRIS. Atari 100k mean HNS is 1.39, against IRIS 1.05. — [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
  - Crafter: 17 of 22 tasks solved at 10M frames. Delta-IRIS beats DreamerV3 XL beyond about 3M frames, but not at the smallest budgets. — [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
  - Stated limitations:
    - K is fixed, even though how much changes varies over time; adaptive K is left to future work.
    - The paper has no ablation on K.
    - With randomly sampled Δ-tokens, stochastic elements behave incoherently (qualitative only).
    - The autoencoder's architecture and context size depend on the environment.
    - Source: [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
- **iVideoGPT (Wu et al., 2024, NeurIPS 2024).**
  - A conditional VQGAN. Context frames get 16x16 tokens each. Future frames get only 4x4 tokens each, encoded and decoded with multi-scale cross-attention to the context frames.
  - Codebook of 8192, at 64x64 resolution. The paper claims about 16x fewer tokens per sequence.
  - Source: [arXiv 2405.15223 html](https://arxiv.org/html/2405.15223v3)
  - Motivation: frames repeat each other heavily, so only the dynamics (positions and poses of moving objects) need encoding. The paper says this reduces the flicker seen with MaskViT's per-frame tokenization. — [arXiv 2405.15223 html](https://arxiv.org/html/2405.15223v3)
  - Tokenizer ablation on RoboNet:
    - A per-frame 4x4 tokenizer reconstructs poorly.
    - The compressive tokenizer reconstructs slightly worse than per-frame 16x16, but keeps the context more consistent and uses far fewer tokens.
    - With decoder cross-attention to the context removed, the motion is still reproduced but the scene is almost entirely missing. This shows that the future tokens carry only the dynamics.
    - Source: [arXiv 2405.15223 html](https://arxiv.org/html/2405.15223v3)
  - Large compute: 138M / 436M transformers and a 114M tokenizer, pretrained on about 1.4M trajectories (Open X-Embodiment + Something-Something v2). The 64x64 VQGAN took 17 GPU-days to pretrain and the transformer 19. Flag: the pretraining recipe is not feasible here; only the tokenizer idea transfers. — [arXiv 2405.15223 html](https://arxiv.org/html/2405.15223v3)
- **Object-centric: SOLD (Mosbach et al., ICML 2025).**
  - Encoder: SAVi with 128-dimensional slots, 2-10 slots per environment.
  - Dynamics: an OCVP-Seq transformer with 4 layers and width 256. The model totals about 12M parameters.
  - Trained on one A100 40GB. SAVi pretraining took about 1M frames and 400k steps.
  - Source: [arXiv 2410.08822 html](https://arxiv.org/html/2410.08822v2)
  - Evaluated on 8 MuJoCo robot-arm tasks plus Meta-World and DM-Control. Average success on Specific tasks is SOLD 89.4%, DreamerV3 81.7%, TD-MPC2 84.2%, though DreamerV3 is higher on Push and Pick. On Distinct tasks it is SOLD 74.6%, DreamerV3 33.1%, TD-MPC2 17.3%. — [arXiv 2410.08822 html](https://arxiv.org/html/2410.08822v2)
  - The background gets its own slot, and unused slots stay empty. — [arXiv 2410.08822 html](https://arxiv.org/html/2410.08822v2)
  - Limitations:
    - The slots are pretrained on random-behavior frames, and the frozen encoder fails on configurations it has not seen (lifted blocks), so SAVi must be fine-tuned.
    - Predictions are deterministic.
    - Scaling to real data is "a significant challenge".
    - Source: [arXiv 2410.08822 html](https://arxiv.org/html/2410.08822v2)
  - Related work: language-conditioned slot world models built on SlotFormer ([arXiv 2503.06170](https://arxiv.org/html/2503.06170v2)), and Slot-MPC (May 2026) for model-predictive control in simulated manipulation ([papers.cool 2605.14937](https://papers.cool/arxiv/2605.14937)). An August 2026 critique, "Better Slots, Better Worlds", says prior object-centric world models assume a good slot encoder, evaluate only in distribution, and lean on crutches such as proprioception tokens ([alphaxiv 2608.12078](https://www.alphaxiv.org/abs/2608.12078)). Its details were not verified.

### Inferences
- **Delta-IRIS maps closest onto mirage's existing parts.** It uses one small-codebook quantizer, and the existing FSQ quantizer could likely stand in for its VQ, though that is untested. Its transformer, at 3 layers and width 512, is the same size class as mirage's 192-dim / 4-layer model. Its compute was a single A100 at roughly Atari 100k scale.
- **What adopting it means.** The tokenizer has to be retrained as a *conditional* autoencoder (encoder and decoder both see x_{t-1} and a_t). The transformer gains a continuous per-frame embedding stream (the I-tokens). The existing per-frame FSQ codebook, and its token-accuracy metrics, stop being the prediction target.
- **Why it handles a static background.** A frame where the arm did not move needs Δ-tokens carrying almost no information, and a decoder that already sees the previous frame copies the background for free. The "bet on noise" failure has nothing to bet on.
- **iVideoGPT adds two ideas to borrow.** Spend the full token count on a keyframe and few tokens on what follows; its ablation is the clearest evidence that this keeps context consistent. Its model sizes are not feasible here.
- **Slots cost too much for 2 weeks.** The mirage scene (one arm on a flat background) is about the easiest case for slots. But pretraining SAVi, then fine-tuning it, adds a second representation-learning project on top of the dynamics phase.

### Gaps
- Neither Delta-IRIS nor iVideoGPT reports a "token change rate on static regions" or anything like it. The evidence that conditioning removes flicker is qualitative (iVideoGPT's comparison with MaskViT) or indirect (fewer tokens, better downstream scores).
- No source found pairs Delta-IRIS with FSQ.
- No keyframe + residual-token scheme was found reported at small scale, beyond iVideoGPT's context/future split.

## Pixel-space prediction with residual / copy structure (masks + flow, PredRNN, SimVP)

### Takeaway
Action-conditioned pixel-motion models have a long robot-arm track record (Finn et al. 2016: a 59k-interaction pushing dataset). They make "move existing pixels" the default operation, which suits a mostly static scene. Modern simple CNN predictors (SimVP) are cheap on one GPU. The main weakness is blur when the future is genuinely uncertain, and a MuJoCo arm conditioned on actions is close to deterministic.

### Cited Findings
- **Finn, Goodfellow, Levine (2016, NeurIPS): DNA / CDNA / STP.**
  - The model predicts pixel motion from previous frames, conditioned on actions.
  - CDNA, one of the three variants, has a ConvLSTM emit 10 normalized convolution kernels. Each kernel is applied to the previous frame, and the 10 results are composited with masks that are softmaxed to sum to 1 at each pixel.
  - The models learned to segment out moving objects without supervision, as the predicted background masks show.
  - Source: [NeurIPS 2016 paper](https://proceedings.nips.cc/paper_files/paper/2016/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf), [review summary](https://liner.com/review/unsupervised-learning-for-physical-interaction-through-video-prediction)
  - Introduced a dataset of 59,000 robot pushing interactions. The authors say explicitly modeling motion makes the model "partially invariant to object appearance", which lets it generalize to unseen objects. — [arXiv 1605.07157](https://arxiv.org/abs/1605.07157)
- **SimVP (Gao et al., CVPR 2022) and SimVPv2.**
  - SimVP is a pure-CNN encoder–translator–decoder. Its authors attribute its training speed to CNNs being well optimized and to avoiding step-by-step recurrence. — [arXiv 2206.05099](https://arxiv.org/pdf/2206.05099)
  - SimVPv2 reports fewer FLOPs, about half the training time and 60% faster inference than SimVP on Moving MNIST. — [arXiv 2211.12509](https://arxiv.org/html/2211.12509v4)
  - The OpenSTL repo has single-GPU training recipes. — [OpenSTL](https://openi.pcl.ac.cn/Goodman2023/SimVPv2/src/branch/master/README.md)
- **DIAMOND as a pixel predictor** (see the first section): it conditions on stacked past frames and stays stable over long rollouts with EDM. — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)

### Inferences
- **Copy is built in.** A model that outputs x_t = x_{t-1} + residual(x_{t-1..t-k}, a_t), or a mask-composited warp of x_{t-1}, has copy-previous-frame in its function class at zero output. A model trained this way should not lose to copy on quiet fields unless the training objective rewards hallucinating change. That is the structural fix for mirage's failure, gained by changing the output parameterization. It is not proven here.
- **Determinism decides between a regression model and diffusion.** The simulator is deterministic given the state, and an arm's state is mostly observable from the last few frames plus the action. So a deterministic MSE/L1 residual predictor (SimVP-sized or a small U-Net) may lose little to diffusion. DIAMOND-style EDM is the upgrade if multimodal blur shows up, for example in contact events. This rests on the project's own setup and is untested.
- **Extra cost.** Frame-level PSNR / MSE metrics on moving and quiet fields would replace token accuracy, and the bench already scores moving fields separately. It also means dropping the FSQ tokens from the dynamics phase.

### Gaps
- Not confirmed from a primary source whether Finn et al.'s CDNA includes an explicit "copy the first frame as background" mask; the fetched pages did not cover it.
- SimVP's parameter counts and absolute training times were not retrieved; Table 4 of 2206.05099 has them.
- PredRNN was not researched in depth. It is recurrent and slower than SimVP by SimVP's own account, so it is low priority.

## Small-scale token world models that worked, and their tricks

### Takeaway
The small token world models that succeeded fall into two groups. Some avoided independent per-frame tokens (Delta-IRIS's conditional Δ-tokens plus continuous I-tokens). Others lived with flicker because their benchmark score did not punish it (IRIS on Atari). None of the small Genie reproductions found report token-stability numbers.

### Cited Findings
- **IRIS** (the per-frame tokenizer baseline): 30M parameters, 4.1 days per Atari run, mean HNS 1.046. DIAMOND's authors observe flicker between frames in its imagined rollouts (bricks and score changing). — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)
- **Delta-IRIS:** cut tokens per frame from IRIS's 16-64 down to 4 by conditioning, and added continuous I-tokens. Without them, the Crafter score fell from 42.47 to 25.92. — [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
- **Jafar** (JAX Genie reimplementation, ICML 2024 workshop): reproduces Genie's CoinRun experiment on one L40S in under a week. — [Jafar GitHub](https://github.com/FLAIROx/jafar), [ICML 2024](https://icml.cc/virtual/2024/37980)
- **Jasmine** (successor codebase): claims an order of magnitude faster wall-clock convergence than Jafar, but targets multi-host scale. — [Jasmine mirror](https://gh.r-mj.com/p-doom/jasmine), [arXiv 2510.27002 listing](https://www.emergentmind.com/papers/2510.27002)
- **TinyWorlds** (AlmondGod, Sept 2025): a 3M-parameter Genie-style model with an FSQ-VAE video tokenizer, a latent action model, and a MaskGIT-style dynamics model. The source is a third-party blog; its 16 GB+ VRAM recommendation is unverified, and one report flagged the license as unclear. — [BrightCoding blog](https://www.blog.brightcoding.dev/2026/06/18/tinyworlds-the-secret-genie-clone-scaling-world-models-without-actions), [cgworld.jp](https://cgworld.jp/flashnews/01-202510-TinyWorlds.html)

### Inferences
- **Mirage has already hit TinyWorlds' likely failure.** TinyWorlds is the closest public analogue: a per-frame FSQ tokenizer plus a token dynamics model at a few million parameters. Nothing found suggests it solved spurious flips.
- **MaskGIT-style decoding** (predict all tokens at once and keep the confident ones) is a different choice from mirage's autoregressive one. No evidence was found on whether it reduces betting on noise.

### Gaps
- No small-scale paper or repo found reports a per-frame token-change rate on static regions, which is the metric mirage tracks. This is the biggest evidence gap. It means the literature cannot say how bad mirage's flip rate is compared with IRIS, Genie or TinyWorlds.
- TinyWorlds' architecture and results were not checked against its repo.

## Recommendation: lowest-risk pivot if target cleaning fails

### Takeaway
The lowest-risk pivot is a deterministic residual pixel predictor: a small U-Net or SimVP-style CNN that takes the last k frames and the action and outputs x_{t-1} + Δ. It keeps the dataset and benchmarks, removes the tokenizer from the loop, and has copy-previous-frame built in at zero residual. If its output blurs at contacts, upgrade it to DIAMOND-style EDM diffusion with the same backbone. Delta-IRIS is the right pivot only if keeping a discrete-token dynamics model is itself a goal.

### Cited Findings
- **DIAMOND shows a 4M-parameter frame-stacked U-Net is enough at 64x64.** It ran on one consumer GPU (RTX 4090, 2.9 days for full RL, 12 GB VRAM) and kept static elements consistent where IRIS's tokens did not. — [arXiv 2405.12399 html](https://arxiv.org/html/2405.12399v2)
- **Delta-IRIS shows that conditioning the tokenizer on the previous frame fixes the token-side problem** at the same compute class (one A100; 3-layer, 512-wide transformer). It also shows the transformer needed a continuous per-frame stream to stay physically coherent. — [arXiv 2406.19320 html](https://arxiv.org/html/2406.19320)
- **iVideoGPT shows that conditional tokens carry only the dynamics.** Its decoder ablation reproduced the motion while losing the scene, which is evidence that the context-conditioned tokens encode only what changed. — [arXiv 2405.15223 html](https://arxiv.org/html/2405.15223v3)

### Inferences
- **Ranking for mirage's constraints** (8 GB, 2 weeks, existing FSQ and dataset):
  1. **Residual pixel predictor.** About 2-4 days: new model and loss, and the bench scored in pixel space. Lowest risk because copy is in its function class and no new tokenizer is needed. Main risk: blur where outcomes are uncertain.
  2. **DIAMOND-style EDM on the same U-Net.** Adds about 2-3 days on top of option 1. Needs a few denoising steps at inference. VRAM is likely fine at a smaller batch than DIAMOND's (unverified).
  3. **Delta-IRIS conditional tokenizer.** About 1 week or more: retrain the tokenizer conditioned on x_{t-1} and a_t (FSQ could replace VQ, untested), add I-tokens, and rewrite the dynamics inputs. Keeps the discrete-token story, but replaces the existing tokenizer outright.
  4. **Slots (SOLD-style) or iVideoGPT-scale models.** Not recommended: a second pretraining project, or compute flagged as too large.
- **A cheaper step before any pivot, keeping the existing FSQ.** Give the current dynamics model an explicit per-token "unchanged" option: a copy head, or residual logits biased toward the previous token. This is the token-space version of the residual trick. It is not a sourced method from the papers above; it borrows the Delta-IRIS / residual idea. It costs about a day, and would show whether the problem is the output parameterization or the noise in the targets.
- **Time estimates.** These are judgments, not measurements, based on model sizes in the cited papers and the 5060 Laptop being slower than a 4090 or A100.

### Gaps
- No head-to-head comparison found between a deterministic residual CNN and diffusion on action-conditioned robot-arm frames at 64x64 with small compute.
- No published measurement found of any of these models on an 8 GB laptop GPU.
