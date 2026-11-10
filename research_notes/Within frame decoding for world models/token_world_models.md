# Within-frame decoding in token-based (discrete-latent) video world models

Researched 2026-11-06. "Verified" means read on the cited page this session. Items marked
**[unverified, from memory]** were not re-read this session and need checking before they are cited.

## How does each system decode the tokens of one next frame?

### Takeaway
Three camps. Small RL world models (IRIS, Delta-IRIS, iVideoGPT) and large video LMs (GAIA-1, VideoPoet, Cosmos-AR) decode a frame's tokens one at a time in raster order, with sampling, not argmax. Genie uses MaskGIT: 25 rounds per frame at temperature 2. The only systems that predict a frame in one pass with independent per-unit outputs (STORM, and probably REM/POP) use very few latents per frame (STORM: one 32x32 categorical vector), and they report trouble with small moving objects. The newest large systems (Genie 2, Oasis) left discrete tokens for latent diffusion.

### Cited Findings
- **IRIS** (Micheli et al., ICLR 2023, arXiv 2209.00588): 64x64 frames, K=16 tokens per frame (4x4), vocab 512. The transformer generates frame tokens autoregressively, one at a time, each conditioned on history and the earlier tokens of the same frame ("the autoregressive process happens at the token level"). Imagination horizon 20. Training on 8 A100s took about 3.5 days per game. Freeway needed sampling temperature 0.01 to get early rewards, so tokens are sampled, not argmaxed. — [IRIS paper](https://arxiv.org/html/2209.00588v2)
- IRIS repo: the world model's forward pass takes a `past_keys_values` KV cache and offsets positions by its length, so token-by-token rollout reuses cached keys and values. The sampling loop lives outside `world_model.py`. — [iris/src/models/world_model.py](https://github.com/eloialonso/iris/blob/main/src/models/world_model.py)
- **Delta-IRIS** (Micheli, Alonso, Fleuret, ICML 2024, arXiv 2406.19320): K=4 delta tokens per frame, versus 16 or 64 for IRIS. The transformer predicts them one at a time, each from a categorical over the vocab. The paper gives no temperature or decoding scheme beyond this. — [Delta-IRIS paper](https://arxiv.org/html/2406.19320)
- **Genie** (Bruce et al., ICML 2024, arXiv 2402.15391): MaskGIT dynamics model. Quote: "we perform 25 MaskGIT steps for the sampling of each frame with a temperature of 2 using random sampling." The CoinRun case study uses 25 steps at temperature 1.0 (App. F.3, Table 17). Training masks input tokens with a Bernoulli rate drawn uniformly from 0.5 to 1. Tokenizer: 1024 codes, patch size 4, 160x90 frames; the token grid is not stated. Latent actions: 8 codes, added as embeddings. The 10.1B model runs at about 1 FPS. — [Genie paper](https://arxiv.org/html/2402.15391)
- **GAIA-1** (Hu et al., Wayve, 2023, arXiv 2309.17080): 576 tokens per frame (18x32 at 288x512), vocab 8192, plus 32 text and 2 action tokens per step. Each image token is conditioned on earlier tokens of the same frame, so decoding within a frame is sequential (raster order not named explicitly). Sampling is top-k=50; argmax was rejected (see next section). Not real time. Long videos use a sliding window; the paper does not mention KV caching. — [GAIA-1 paper](https://arxiv.org/html/2309.17080)
- **STORM** (Zhang et al., NeurIPS 2023, arXiv 2310.09615): one stochastic latent per frame, 32 categoricals x 32 classes, predicted from the transformer's hidden state. The transformer token is one action-fused latent per timestep, so there is no within-frame token sequence. Imagination is autoregressive over time with a KV cache. Context 8, horizon 16. Training takes 4.3 h on an RTX 3090, versus about 7 days per two IRIS runs on an A100. The paper does not say whether the 32 categoricals are sampled independently. — [STORM paper](https://arxiv.org/html/2310.09615)
- **iVideoGPT** (Wu et al., NeurIPS 2024, arXiv 2405.15223): context frames use 16x16 tokens; future frames use 4x4 tokens conditioned on the context ("asymptotic reduction of 16x"). Codebooks are 8192 each. Generation is autoregressive next-token prediction with temperature 1.0 and top-k 100 (Table 3). — [iVideoGPT paper](https://arxiv.org/html/2405.15223)
- **VideoPoet** (Kondratyuk et al., ICML 2024, arXiv 2312.14125): MAGVIT-v2 tokens; a 17-frame 128x128 clip is (5,16,16) = 1280 tokens, vocab 2^18. The base LM decodes them autoregressively. Super-resolution is non-autoregressive, with 24 sampling steps per stage. Speed: about 5 s of compute per second of output video on TPUv5p (unoptimized). — [VideoPoet paper](https://arxiv.org/html/2312.14125)
- **Cosmos autoregressive WFMs** (NVIDIA, 2025, arXiv 2501.03575): FSQ tokenizer with 8x temporal and 16x16 spatial compression, levels (8,8,8,5,5,5), vocab 64,000. Llama3-style GPT, 4B to 13B. Context is 12,800 tokens; my arithmetic says that is 5x40x64 for 33 frames at 640x1024. Inference uses a KV cache, tensor parallelism and torch.compile (gpt-fast style). Medusa speculative decoding with 9 heads gives up to 2.0x token throughput and 4.6x fewer forward passes on 4B, and up to 3.2x and 6.1x on 5B. More heads cut forward passes but can lower throughput. A 320x512 variant reaches about 10 frames/s. A 7B diffusion decoder maps the discrete tokens to continuous ones because heavy compression causes distortions. — [Cosmos paper](https://arxiv.org/html/2501.03575)
- **MAGVIT** (Yu et al., CVPR 2023, arXiv 2212.05199): non-autoregressive masked decoding, stated "60x" faster than autoregressive models and two orders of magnitude faster than diffusion. — [MAGVIT abstract](https://arxiv.org/abs/2212.05199)
- **REM / POP** (Cohen et al., arXiv 2402.05643, 2024): targets the "sequential token-by-token generation of next observations" bottleneck with a RetNet forward mode (Parallel Observation Prediction). Reports "15.4x faster imagination compared to prior TBWMs" and superhuman scores on 12 of 26 Atari 100K games in under 12 h of training. — [REM abstract](https://arxiv.org/abs/2402.05643)
- **Genie 2** (DeepMind blog, Dec 2024): an autoregressive latent diffusion model, not discrete tokens. An autoencoder produces latent frames, and a causal transformer dynamics model generates them frame by frame, with classifier-free guidance to improve action control. A distilled version runs in real time at lower quality. No paper. — [Gigazine summary of the DeepMind blog](https://wbgsv0a.gigazine.net/gsc_news/en/20241205-google-deepmind-genie-2); [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2024/12/genie-2/)

### Inferences
- No system I found decodes a large frame (16+ discrete tokens) in one pass with independent per-cell argmax. The closest is STORM, which uses one vector per frame and accepts poor small-object modeling. REM/POP is parallel within a frame, but whether its tokens are independent given the context is not confirmed (see Gaps).
- Every discrete system that decodes in one shot or in a few steps uses random sampling (Genie: temperature 2), and the AR systems use sampling with top-k. Nobody reports argmax working well. GAIA-1 reports that it fails.
- Genie's training masking rate of 0.5 to 1 means the model always sees at least some unmasked target-frame tokens during training. That is what lets MaskGIT rounds coordinate cells: each round conditions on the cells already committed.

### Gaps
- **[unverified, from memory]** MaskGIT (Chang et al., CVPR 2022, arXiv 2202.04200) uses about 8 to 12 steps with a cosine mask schedule. MAGVIT uses 12 steps on a 4x16x16 token grid with COMMIT conditional masking (the PDF was too large to fetch). MAGVIT-v2 (Yu et al., ICLR 2024, arXiv 2310.05737) introduced lookup-free quantization.
- **[unverified, from memory]** REM's POP predicts all K tokens of the next frame in one forward pass from learned "prediction" query tokens. Whether those predictions are independent given the context was not confirmed.
- **[unverified, from memory]** TWM (Robine et al., ICLR 2023, arXiv 2303.07109) uses a Transformer-XL over one categorical latent per frame, like DreamerV2.
- **[unverified, from memory]** Oasis (Decart/Etched, 2024) is a diffusion transformer with a ViT autoencoder, not discrete tokens. Genie 3 (DeepMind, Aug 2025) has only a blog post, with no published decoding details.
- Cosmos-AR within-frame order and top-k/temperature were not in the sections read.

## Do any report frozen/static rollouts, copying the last frame, or poor small-object motion, and how did they fix it?

### Takeaway
Yes, though frozen rollouts are usually reported as a decoding-time symptom, not a model failure. GAIA-1 states plainly that argmax decoding produces futures that "get stuck in a repetitive loop" and frames that "repeat themselves", and fixes it with top-k=50 sampling. VideoPoet reports a "repetitive token predictions" decoding collapse. STORM and VideoPoet report trouble with small moving objects. A 2026 line of work (ITC, DeltaTok, LaMo) treats copying the previous frame as a structure to exploit rather than a bug.

### Cited Findings
- GAIA-1: "maximization-based sampling (i.e. argmax) generates futures that get stuck in a repetitive loop"; Figure 6 says argmax frames "repeat themselves". Pure sampling picks tokens from "the unreliable tail of the probability distribution, which throws the model out-of-distribution." The fix is top-k sampling; "top-k=50 sampling produces tokens that have a similar perplexity distribution to real tokens", with k chosen from the tokens per frame and codebook size. — [GAIA-1 paper](https://arxiv.org/html/2309.17080)
- VideoPoet: a decoding collapse "characterized by repetitive token predictions", addressed by fine-tuning. Also: "Small objects and fine-grained details, especially when coupled with significant motions, remains difficult." The 1B model was unstable, with large subject changes over time. — [VideoPoet paper](https://arxiv.org/html/2312.14125)
- STORM: "faces challenges when handling a single small moving object, as observed in Pong and Breakout, due to the nature of autoencoders." Breakout scores 16 for STORM versus 84 for IRIS. — [STORM paper](https://arxiv.org/html/2310.09615)
- IRIS: with 16 tokens, "the autoencoder often erases the player, switches colors, and misplaces rewards" (App. E). 64 tokens helped reconstruction. — [IRIS paper](https://arxiv.org/html/2209.00588v2)
- Delta-IRIS: with randomly sampled delta tokens, "large quantities of cows appear and disappear from the screen incoherently" (Fig. 3). Without I-tokens, the agent glitches through walls and water in imagination. — [Delta-IRIS paper](https://arxiv.org/html/2406.19320)
- Cosmos-AR: the main failure is objects unexpectedly appearing from below. Failure rate with single-image conditioning is 15% (4B), 7% (5B), 2% (12B) and 3% (13B), and below 2% for all models with 9-frame video conditioning (Table 18). — [Cosmos paper](https://arxiv.org/html/2501.03575)
- Genie: reports hallucinated "unrealistic futures" and a 16-frame memory limit. It does not discuss static rollouts or copying. — [Genie paper](https://arxiv.org/html/2402.15391)
- iVideoGPT: per-frame tokenizers (MaskViT) show "temporal inconsistency and flicker artifacts"; conditioning future-frame tokens on context reduces this. Its bottleneck tokens encode "the position and pose of moving objects." Static or copying failures are not discussed. — [iVideoGPT paper](https://arxiv.org/html/2405.15223)
- **ITC: Identifiable Token Correspondence for World Models** (Kim et al., ICML 2026, arXiv 2605.16457): treats each next-frame token as either copied from a previous-frame token or newly generated, solved as optimal transport at decoding time. Architecture and training are unchanged. Reports beating the previous best token-based world model (Cohen et al., 2025) on Atari 100K, and 72.5% return / 35.6% score on Craftax-classic. Code: github.com/snu-mllab/Identifiable-Token-Correspondence. — [ITC arXiv](https://arxiv.org/html/2605.16457); [ICML poster](https://icml.cc/virtual/2026/poster/66113)
- **DeltaTok: "A Frame is Worth One Token"** (Kerssies et al., CVPR 2026): one delta token per frame, encoding the change from the previous frame's features, so the model avoids "re-predicting static parts of the scene". — [CVPR 2026 supplemental](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Kerssies_A_Frame_is_CVPR_2026_supplemental.pdf); [Emergent Mind summary](https://www.emergentmind.com/topics/deltatok)
- **LaMo** (ICLR 2026): continuous motion tokens encoding first-order change, motivated by dense latents "repeatedly carrying largely static information". — [LaMo, ICLR 2026](https://iclr.cc/virtual/2026/10018050)

### Inferences
- GAIA-1's failure, frames repeating under argmax, matches the project's symptom. It points at the decoding rule (mode-seeking per token), not only at a lack of coordination between cells. In a mostly static scene, the most likely value for each token is "same as last frame", so argmax per cell freezes even when the joint distribution puts most of its mass on "something moves". Both sequential decoding and sampling break this, by different routes: sequential or MaskGIT decoding lets later cells condition on cells already committed to motion, and sampling stops each cell from always picking the mode.
- So the published fix for frozen rollouts is sampling (top-k, or a temperature above 1 as in Genie's 2.0) together with a decoder that conditions cells on each other. Raster AR alone, still run with argmax, is not shown anywhere to fix freezing.
- ITC suggests "copy" is the right default for most cells, and that the hard part is choosing which cells to generate. A small model could do this cheaply with a per-cell "changed?" head.

### Gaps
- No paper found that measures "fraction of frozen frames" against decoding scheme (AR vs MaskGIT vs parallel) on the same model. Comparisons are indirect.
- No small-scale (~10M param) ablation of MaskGIT step count for a world model was found.

## What does Delta-IRIS do to encode only changes, and how did it do versus IRIS?

### Takeaway
Delta-IRIS conditions the tokenizer's encoder and decoder on past frames and actions, so the discrete tokens only need to carry the stochastic part of a transition. That takes 4 tokens per frame instead of 16 or 64. It also feeds continuous CNN frame embeddings ("I-tokens") to the transformer so it can see the current state. Result: better returns than IRIS at every budget, and about 10x faster.

### Cited Findings
- Encoder and decoder are conditioned on past frames and actions; "only the stochastic features of a transition should be encoded." K=4 delta tokens per frame. Continuous I-tokens (CNN embeddings of frames, not discretized) are interleaved with delta tokens so the transformer need not integrate past deltas to infer state. — [Delta-IRIS paper](https://arxiv.org/html/2406.19320)
- Crafter return (Table 1): Delta-IRIS 7.7 / 15.4 / 16.1 at 1M / 5M / 10M frames, 20 FPS. DreamerV3 XL 9.2 / 14.2 / 15.1, 30 FPS. IRIS-64 5.5 at 1M, 2 FPS. IRIS-16 4.4 at 1M, 6 FPS. IRIS runs stopped early as "prohibitively slow". Delta-IRIS is "10 times faster than iris". Crafter score at 10M is 42.47, with 17 of 22 achievements solved on average. Removing I-tokens drops the score to 25.92. — [Delta-IRIS paper](https://arxiv.org/html/2406.19320)

### Inferences
- The project's FSQ tokenizer encodes each frame independently. A Delta-IRIS-style conditional tokenizer would make "no change" cheap to represent and push the dynamics model's token budget onto motion. That attacks the same freeze from the representation side. It is a bigger change than switching the decoder.

### Gaps
- Delta-IRIS does not report a sampling temperature or any static-rollout analysis.

## Token counts per frame, and rollout speed tradeoffs

### Takeaway
Small RL models use 1 to 64 tokens per frame. Video LMs use 256 to 1280+ per frame or clip. Token-by-token AR cost grows linearly with tokens per frame even with a KV cache, which is why Delta-IRIS cut to 4 tokens, iVideoGPT to 4x4 for future frames, REM went parallel, and Cosmos added Medusa heads. MaskGIT's cost is fixed by the round count (8 to 25), not by token count.

### Cited Findings
- Tokens per frame: STORM 1 latent (32x32 categorical); Delta-IRIS 4; iVideoGPT 16 for future frames (256 for context); IRIS 16 or 64; VideoPoet 256 per image, 1280 per 17-frame clip; GAIA-1 576; Cosmos-AR 2,560 per latent frame (40x64, my arithmetic). — sources as cited above
- Speed: IRIS-16 6 FPS vs Delta-IRIS 20 FPS on Crafter ([Delta-IRIS](https://arxiv.org/html/2406.19320)). REM "15.4x faster imagination" than prior token-based world models ([REM](https://arxiv.org/abs/2402.05643)). MAGVIT is 60x faster than AR ([MAGVIT](https://arxiv.org/abs/2212.05199)). Genie runs at about 1 FPS at 10.1B parameters with 25 MaskGIT rounds ([Genie](https://arxiv.org/html/2402.15391)). Cosmos Medusa gives up to 3.2x throughput ([Cosmos](https://arxiv.org/html/2501.03575)).
- KV caching: used by IRIS (repo), STORM (paper) and Cosmos (paper). GAIA-1 uses a sliding window and does not mention a cache.

### Inferences
- For 64 tokens per frame on a ~10M model, raster AR costs 64 sequential forward steps per frame. With a KV cache, each step is cheap (one query over cached context), but the steps are still sequential. A MaskGIT schedule of 4 to 8 rounds costs 4 to 8 full-frame passes. Which is faster in practice depends on kernel launch overhead at this tiny scale, and should be measured, not assumed.
- IRIS is the closest published analogue to the project in scale and resolution: 64x64 frames, 16 tokens per frame, a 10-layer D=256 transformer. It used raster AR with sampling.

### Gaps
- No published per-frame wall-clock comparison of AR vs MaskGIT at ~10M parameters and 64 tokens per frame was found.
