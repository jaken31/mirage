# Residual / change-based and continuous-output next-frame predictors

Scope: alternatives to predicting a full grid of discrete tokens per frame. Source tags: "[abs]" means only the arXiv abstract was read; "[full]" means the paper's HTML text was read; "[recall, unverified]" means the claim is from background knowledge and was NOT re-checked in this session, so treat it as a lead and not a citation.

## 1. Pixel- and latent-space delta predictors (PredNet, SimVP/PhyDNet, Delta-IRIS, flow/warp)

### Takeaway
Delta prediction is well established and works. Its biggest strength is that "no change" costs almost nothing, and that is also its biggest danger: on data that is mostly static, delta and copy-based models fall back to copying the previous frame, and per-frame metrics reward them for doing so. The strongest recent evidence for change-only *tokens* is Delta-IRIS: 4 tokens per frame, conditioned on past frames and actions, beating IRIS at 16 to 64 tokens while training about 10x faster.

### Cited Findings
- **Delta-IRIS (arXiv 2406.19320, 2024, ICML 2024) [full]:** the tokenizer's encoder and decoder are "conditioned on past frames and actions," so the discrete tokens only need to capture what changes. It uses 4 tokens per frame and says it "is able to encode frames with minimal loss"; IRIS at 16 and 64 tokens "produces poor reconstructions" (Fig. 5). Design principle: "only the stochastic features of a transition should be encoded," because the decoder handles deterministic dynamics. — [arXiv 2406.19320](https://arxiv.org/html/2406.19320)
- Delta-IRIS adds continuous "I-tokens" from an auxiliary CNN, interleaved with the discrete delta tokens, because otherwise the transformer must integrate over all past deltas to know the current state. Removing I-tokens drops the Crafter score at 10M frames from 42.47 to 25.92. — [arXiv 2406.19320](https://arxiv.org/html/2406.19320)
- Delta-IRIS results: solves 17 of 22 Crafter tasks on average at 10M frames, beats DreamerV3 beyond 3M frames, and runs at 20 FPS throughput vs 2 FPS for IRIS-64 on one A100 40GB. Stated limitation: it "uses the same number of tokens to encode stochastic dynamics at each time step"; the authors propose a variable token count as future work. — [arXiv 2406.19320](https://arxiv.org/html/2406.19320)
- **PredNet (arXiv 1605.08104, 2016) [abs]:** each layer forwards only "deviations from those predictions," i.e. prediction errors, upward. — [arXiv 1605.08104](https://arxiv.org/abs/1605.08104)
- A critical review of PredNet (arXiv 1906.11902, 2019) reports that it falls back to copying the last frame on sequences with many still frames, so the frame rate becomes a key hyperparameter (via search summary; the PDF was not opened). — [arXiv 1906.11902](https://arxiv.org/pdf/1906.11902)
- **SimVP (arXiv 2206.05099, CVPR 2022):** a pure-CNN encoder-translator-decoder trained end to end with MSE. Moving MNIST MSE/SSIM: SimVP 23.8/0.948, PhyDNet 24.4/0.947, CrevNet 22.3/0.949. PhyDNet takes about 10 days to train with the official code vs about 2 days for SimVP (2k epochs) (via search summary). — [arXiv 2206.05099](https://arxiv.org/pdf/2206.05099)
- **Finn et al., CDNA/DNA/STP (arXiv 1605.07157, 2016) [abs]:** action-conditioned video prediction that explicitly models pixel motion from previous frames. This makes it partly invariant to appearance, so it generalizes to unseen objects. Trained on 59k robot pushing interactions. — [arXiv 1605.07157](https://arxiv.org/abs/1605.07157)
  - [recall, unverified] CDNA predicts a set of convolution kernels, each applied to the previous frame, plus a soft compositing mask. One mask channel selects the unmodified previous frame (background), making "copy" an explicit, learnable gate per pixel. Inputs are 64x64, it is trained with L2, and predictions blur at longer horizons. I could not extract the PDF text in this session to confirm.
- **SAVP (arXiv 1804.01523, 2018) [abs]:** naive deterministic models "average together possible futures into a single, blurry prediction." VAEs capture stochasticity but "can struggle to produce realistic results"; GANs look realistic but "fail to produce diverse predictions"; combining both works best. — [arXiv 1804.01523](https://arxiv.org/abs/1804.01523)
  - [recall, unverified] SAVP's generator builds on the CDNA-style warp-plus-composite design (transform previous and first frames, then blend with masks) and is evaluated on the BAIR robot-pushing dataset.
- **Copy-last-frame as a degenerate optimum:** Villegas et al. (arXiv 1911.01655, 2019) found that on Human3.6M, where most pixels are static background, copying the last frame "significantly outperforms" all models tested, so per-frame metrics are unreliable there. A transformation-based adversarial predictor (arXiv 2003.04035, 2020) notes models sometimes make motion "simply stop." Both via search summary. — [arXiv 1911.01655](https://arxiv.org/pdf/1911.01655); [arXiv 2003.04035](https://arxiv.org/pdf/2003.04035)
- A 2026 NeurIPS listing, "Your world model may be worse than copying the last frame," reportedly finds a learned model about 5x worse in squared error than repeating the previous frame and recommends reporting "lift" over a copy baseline. Seen only in a search snippet; not read. — [NeurIPS 2026](https://neurips.cc/virtual/2026/166122)
- The vp-suite library ships a copy-last-frame baseline model. — [vp-suite](https://ais-bonn.github.io/vp-suite/vp_suite.models.copy_last_frame.html)

### Inferences
- A residual predictor makes "hold still" the zero-effort output. That fixes a model that cannot hold still, but it worsens a model that already freezes: the literature's main failure mode for delta and copy models is under-predicting motion, not over-predicting it. If the project's rollouts already freeze, a residual parameterization alone points the wrong way unless the loss or sampling penalizes zero motion.
- Delta-IRIS is the closest published analogue to "keep discrete tokens but encode only change." Its key additions are (a) conditioning the tokenizer's decoder on the previous frame, so tokens are deltas by construction, and (b) a continuous state summary (I-tokens) beside the deltas. The ablation's large drop (42.5 to 25.9) suggests delta tokens without a state summary are a weak design.
- Explicit copy/warp masks (CDNA style) make the "copy" decision visible and measurable per pixel. That is useful for diagnosing freezing even if the mask is not kept.

### Gaps
- No numbers found on how often residual pixel predictors collapse to zero motion on robot-arm data specifically.
- CDNA mask details and SAVP architecture details were not verified (PDF extraction failed; only abstracts were read).
- PhyDNet's and SimVP's own papers were not read for whether they predict residuals explicitly; the search results do not say.

## 2. Continuous-latent and diffusion world models (DreamerV3, DIAMOND, GameNGen, Diffusion Forcing): frozen rollouts and the copy shortcut

### Takeaway
Diffusion world models are the best-documented fix for blur and drift at small scale. DIAMOND runs at 64x64 with about 4-13M parameters on one RTX 4090, and shows that the training *target* decides whether few-step sampling drifts: noise prediction (DDPM) "learn[s] the identity function when the noise is dominant," while predicting the clean image (EDM) stays stable at 3 steps. The standard defense against autoregressive drift is noise-augmenting the context frames (GameNGen, Diffusion Forcing). None of the sources reports "frozen" rollouts as such, but DIAMOND's identity-function finding is the closest mechanism to a copy shortcut.

### Cited Findings
- **DIAMOND (arXiv 2405.12399, NeurIPS 2024 spotlight) [full]:** pixel-space diffusion world model. Atari frames are 64x64x3, conditioned on the last 4 frames and actions by frame stacking, with 3 denoising steps ("we therefore set n=3 in all of our experiments"). — [arXiv 2405.12399](https://arxiv.org/html/2405.12399v2)
- DIAMOND Fig. 3: "ddpm-based generation suffers from compounding error, and that the smaller the number of denoising steps, the faster the error accumulates." The cause given: the noise-prediction objective "causes the model to learn the identity function when the noise is dominant." EDM's preconditioning makes the model predict the clean image at high noise instead. — [arXiv 2405.12399](https://arxiv.org/html/2405.12399v2)
- DIAMOND on single-step denoising: "The optimal single-step prediction is thus the expectation over possible reconstructions," and on Boxing "single-step denoising interpolates between possible outcomes and results in blurry predictions"; multi-step sampling drives generation to one mode and stays crisp. A single step was stable in Breakout. — [arXiv 2405.12399](https://arxiv.org/html/2405.12399v2)
- DIAMOND size and compute: 13M parameters (Table 4) vs 30M for IRIS and 18M for DreamerV3. The text also says "4M for our Atari models" without reconciling the two figures. Each Atari run used about 12 GB of VRAM and about 2.9 days on one RTX 4090. — [arXiv 2405.12399](https://arxiv.org/html/2405.12399v2)
- DIAMOND CS:GO failure modes: it drifts out of distribution in rarely visited areas, forgets state when approaching walls or losing visibility (limited memory), and "wrongly enables successive jumps." The dynamics model runs at 56x30 with a 51M upsampler, trained 12 days on an RTX 4090. — [arXiv 2405.12399](https://arxiv.org/html/2405.12399v2)
- **GameNGen (arXiv 2408.14837, 2024) [full]:** fine-tuned Stable Diffusion v1.4 at 320x240, with 64 context frames and actions and 4 DDIM steps (1 step degrades). Context frames are corrupted with Gaussian noise up to level 0.7, with the noise level discretized into 10 learned embedding buckets. Without this, "quality degrades quickly after 10-20 frames." Drift is attributed to the teacher-forcing vs autoregressive-sampling gap. Training: 128 TPU-v5e, 700k steps, batch 128. — [arXiv 2408.14837](https://arxiv.org/html/2408.14837)
- **Diffusion Forcing (arXiv 2407.01392, 2024) [abs]:** trains with "independent per-token noise levels," so a causal model can condition on partially noised past frames. Claims rollouts of video beyond the training horizon "where baselines diverge." — [arXiv 2407.01392](https://arxiv.org/abs/2407.01392)
- **DreamerV3 (arXiv 2301.04104, 2023)** [recall, unverified]: the latent state is a deterministic GRU state plus *categorical* (discrete, 32x32 one-hot) stochastic latents, and a CNN decoder reconstructs pixels from that state. It is therefore not a purely continuous-latent model. Its decoder is used for representation learning, not as the rollout output.
- Small objects in reconstruction-based world models: pixel reconstruction gives small objects little loss signal. A 2026 Pong study (arXiv 2607.15142) found frozen world models lose the ball (disappearance, wrong motion, invalid ball-paddle contact); a DreamerV3 policy trained in it fell from -5.5 to -20.9 in the real game. A ball-region auxiliary loss (CGSReg) helped some models but not STORM. Via search summary only. — [papers.cool 2607.15142](https://papers.cool/arxiv/2607.15142); related: [Segmentation Dreamer, arXiv 2410.09972](https://arxiv.org/html/2410.09972v1), [Dreaming, arXiv 2007.14535](https://pdf.arxiv.org/pdf/2007.14535)

### Inferences
- DIAMOND's identity-function result maps directly onto the project's concern. Any objective whose best low-information guess equals "the input" (noise prediction at high noise, or a delta regressed to zero) lets the model output "no change." Predicting the clean target, or using enough sampling steps, removes that easy way out.
- DIAMOND is the most relevant compute reference: same 64x64 resolution, similar parameter count (about 4-13M), and 12 GB VRAM on a 24 GB 4090. An 8 GB laptop GPU would likely need a smaller batch or gradient checkpointing; this is untested.
- Context-noise augmentation (GameNGen) targets drift, not freezing. Its relevance here depends on whether the frozen rollouts come from exposure bias (which it addresses) or from a decoding/averaging effect (which it does not).

### Gaps
- No source found that explicitly reports frozen/static rollouts in DIAMOND, GameNGen or DreamerV3, or measures how often they copy the last frame.
- Diffusion Forcing's full experiments (stability metrics, failure cases) were not read.
- DreamerV3 details were not re-verified this session.

## 3. Does regressing a delta with MSE cause blur or zero-motion collapse, and which losses fix it?

### Takeaway
Yes, for blur: MSE's optimum is the conditional mean, so under uncertainty it averages possible futures into blur, and for a delta target the mean of "moves left" and "moves right" is roughly zero, i.e. no motion. This is stated across Mathieu 2015, SAVP and DIAMOND. Fixes, roughly in order of how strong the evidence is: diffusion with multi-step sampling (DIAMOND), adversarial plus latent-variable losses (SAVP), gradient-difference and multi-scale losses (Mathieu).

### Cited Findings
- Mathieu, Couprie, LeCun (arXiv 1511.05440, 2015/ICLR 2016) [abs]: MSE gives "inherently blurry predictions." Proposed fixes: a multi-scale architecture, adversarial training, and an image gradient difference loss (GDL). — [arXiv 1511.05440](https://arxiv.org/abs/1511.05440)
- SAVP [abs]: deterministic models "average together possible futures into a single, blurry prediction"; VAE plus GAN gives both realism and diversity. — [arXiv 1804.01523](https://arxiv.org/abs/1804.01523)
- DIAMOND [full]: the single-step (regression-like) optimum "is the expectation over possible reconstructions," which blurs on multimodal outcomes; 3-step EDM sampling picks a mode and stays crisp. — [arXiv 2405.12399](https://arxiv.org/html/2405.12399v2)
- An early multi-scale video prediction work evaluated only moving regions, because predicting static pixels is trivial (via search summary, patent record). — [US patent 10430685](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10430685)

### Inferences
- For a robot arm whose next pose given the action is close to deterministic, MSE-on-delta blur may be mild: the conditional mean is close to a single mode. Blur and zero-motion get worse where motion is uncertain, for example ambiguous or noisy actions, or contact. This is an inference, not measured.
- MSE averaged over the whole frame is dominated by static background when the arm covers a small fraction of 64x64. A motion-masked or motion-weighted loss, plus a metric on moving pixels only, is the cheapest guard against zero-motion collapse.

### Gaps
- No paper found that directly measures "MSE on frame deltas collapses to zero motion" as a separate experiment from MSE on full frames.
- Perceptual (LPIPS) loss for small 64x64 robot frames: no specific evidence was found.

## 4. Hybrid designs: discrete tokens plus a copy mask / change gate, or tokens only for changed cells

### Takeaway
Limited direct evidence. Delta-IRIS is the strongest example (discrete change-only tokens plus a continuous state). "Skip unchanged patches" exists mainly as an *efficiency* trick in video understanding and diffusion inference, not as a world-model output head. I found no world-model paper with a per-cell copy gate over a discrete token grid. CDNA's compositing mask is the pixel-space precedent.

### Cited Findings
- Delta-IRIS: fixed 4 delta tokens per frame; the authors name a variable, context-dependent token count as future work. — [arXiv 2406.19320](https://arxiv.org/html/2406.19320)
- Inter-frame latent patch pruning in a video diffusion transformer (arXiv 2604.23858, 2026) reuses patches unchanged from the previous frame. Naive pruning caused artifacts, which needed an "Attention Recovery" fix; reported 1.44x throughput on Self-Forcing. LIPAR (arXiv 2603.05811) argues that pixel-unchanged patches are latent-unchanged. Both via search summary. — [arXiv 2604.23858](https://arxiv.org/html/2604.23858v1); [arXiv 2603.05811](https://arxiv.org/html/2603.05811)
- Input-side static-patch removal: Run-Length Tokenization (arXiv 2411.05222) thresholds pixel differences between consecutive patches; Gated Residual Tokenization (arXiv 2509.14199) uses an SSIM-based binary mask; EVS (arXiv 2510.14624) prunes static patches in VLMs. — [RLT](https://arxiv.org/pdf/2411.05222); [GRT](https://arxiv.org/pdf/2509.14199); [EVS](https://arxiv.org/html/2510.14624v1)

### Inferences
- A per-cell "copy vs re-predict" gate over the 8x8 FSQ grid is untested in the literature found. It also inherits the copy-collapse risk from section 1: if the gate's loss is dominated by static cells, it will learn "always copy." Supervising the gate on cells that actually changed, and balancing that loss, would be required.
- The 2026 pruning result (naive reuse causes artifacts) hints that hard-copying latent cells can create seams at the border between changed and unchanged cells. This is plausible for a convolutional FSQ decoder too, but untested.

### Gaps
- No world-model paper found that predicts tokens only for changed cells with an explicit gate, or that reports its effect on frozen rollouts.

## 5. Compute on an 8 GB laptop GPU

### Takeaway
The only directly comparable published number is DIAMOND: 64x64 frames, about 4-13M parameters, 3 denoising steps, about 12 GB VRAM and about 2.9 days per run on an RTX 4090. Delta-IRIS used an A100 40GB. GameNGen needs TPU-pod scale. Residual CNN predictors (SimVP-class) are far cheaper: about 2 days to train on Moving MNIST.

### Cited Findings
- DIAMOND Atari: about 12 GB VRAM, about 2.9 days on one RTX 4090; CS:GO: 12 days on an RTX 4090, inference 10 Hz on an RTX 3090. — [arXiv 2405.12399](https://arxiv.org/html/2405.12399v2)
- Delta-IRIS: one A100 40GB, 20 FPS training throughput. — [arXiv 2406.19320](https://arxiv.org/html/2406.19320)
- GameNGen: 128 TPU-v5e, 700k steps. — [arXiv 2408.14837](https://arxiv.org/html/2408.14837)
- SimVP: about 2 days (2k epochs) vs about 10 days for PhyDNet. — [arXiv 2206.05099](https://arxiv.org/pdf/2206.05099)

### Inferences
- A DIAMOND-scale diffusion head at 64x64 likely fits in 8 GB with a smaller batch. Sampling costs about 3x a single forward pass per frame (3 steps), which matters for rollout evaluation but probably not for training. Not measured.

### Gaps
- No published memory or throughput numbers on an 8 GB laptop GPU for any of these models.
