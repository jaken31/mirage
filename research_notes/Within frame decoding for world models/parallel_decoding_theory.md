# Parallel (non-autoregressive) decoding of discrete tokens: why it produces partial outputs, and the fixes

Scope note: sources were read on 2026-11-06 via arXiv abstract pages and ar5iv/arXiv HTML renderings. Numbers below are quoted from those pages. Items I know from background but did not re-verify this session are marked "unverified" and placed under Inferences or Gaps.

## What is the multimodality problem, and how does it map onto per-cell independent prediction of a frame?

### Takeaway
A model that predicts every output token independently given the input can only learn per-position marginals; when the target has several valid joint outcomes, the argmax of each marginal can mix pieces of different outcomes into an output that matches none of them. An 8x8 token grid where each cell independently decides "keep" vs "change" is the same factorization, so a partly-moved arm is the image analogue of NAT's "Danke Dank".

### Cited Findings
- Gu et al. 2018 (ICLR 2018, arXiv 1711.02281) define the naive NAT model as a target-length distribution times a product of per-position token distributions, each conditioned only on the source ("complete conditional independence"). Example: "Thank you." can be "Danke.", "Danke schön." or "Vielen Dank."; an independent per-position model can emit invalid mixes like "Danke Dank." or "Vielen schön." — [Gu et al. 2018, ar5iv](https://ar5iv.labs.arxiv.org/html/1711.02281)
- Gu et al. BLEU (WMT14 En-De): NAT 17.35; +fine-tuning 17.69; +noisy parallel decoding s=10 18.66, s=100 19.17; autoregressive greedy 22.71, beam 4 23.45. WMT16 En-Ro: NAT 26.22 vs AR(b=4) 31.91 — [Gu et al. 2018](https://ar5iv.labs.arxiv.org/html/1711.02281)
- Gu et al.'s fix used a latent variable (input-token fertilities) to pin down the "mode" before the parallel decoder runs; switching decoder inputs from uniform copying to fertility-based copying added ~4 BLEU with ground-truth training, ~2 with distillation (IWSLT dev) — [Gu et al. 2018](https://ar5iv.labs.arxiv.org/html/1711.02281)
- Huang et al. 2022 (arXiv 2206.05975) show maximum-likelihood NAT training approximates the marginals but "drops all dependencies between tokens"; the information lost is the dataset's conditional total correlation. Prior NAT objectives (their framing) succeed by maximizing likelihood on a proxy distribution that has less of that lost information — [Huang et al. 2022, abstract](https://arxiv.org/abs/2206.05975)
- Symptom: token repetition. Mask-Predict's one-pass output (T=1) has 16.72% repeated tokens on WMT14 En-De (9.34% En-Ro); this drops to 5.40% at T=2, 2.03% T=3, 0.72% T=5, and the drop "correlates with the steep rise in translation quality", which the authors take as support for the multimodality explanation — [Ghazvininejad et al. 2019, ar5iv](https://ar5iv.labs.arxiv.org/html/1904.09324)
- Same failure named for discrete diffusion: discrete diffusion models denoise each variable without modelling dependencies between variables within a step, which is why they need hundreds to thousands of steps; adding a copula model that supplies the missing dependencies gave better text with ~8-32x fewer steps — [Liu et al., Discrete Copula Diffusion, arXiv 2410.01949](https://arxiv.org/abs/2410.01949)
- Token-Critic (2022) lists, as a MaskGIT weakness, that each token's accept/reject decision is made independently, which "impedes capturing rich correlations between tokens" — [Lezama et al. 2022, ar5iv](https://ar5iv.labs.arxiv.org/html/2209.04439)

### Inferences
- Mapping to the frame case: the "modes" are joint outcomes such as "arm stays this frame" vs "arm moves by one link-width this frame" (timing ambiguity, or uncertainty about the action's effect). Each changing cell sees a near 50/50 marginal; independent greedy argmax resolves each coin flip separately, so some cells move and others keep, giving a partial arm. This is the "Danke Dank" mix in 2D.
- Repetition in NAT and "keep the previous token" in a frame model are the same kind of error: falling back to a safe, high-marginal token when the model cannot commit to one mode jointly. Rollout freezing compounds it: a frame that is mostly a copy becomes the next context, which makes "keep" even more likely next step.
- Gu's fertility latent is the conceptual ancestor of conditioning on an explicit action/latent-action token: anything that tells the decoder *which* mode to produce reduces the per-cell ambiguity before the parallel step.

### Gaps
- No source found that measures multimodality (e.g., conditional total correlation) directly for video-frame token grids; the mapping above is my inference.

## What fixes exist, and what do they cost?

### Takeaway
Two families: (a) reduce the multimodality of what is predicted in one shot (distillation, latent variables, glancing training), and (b) stop deciding all tokens at once (iterative mask-predict / MaskGIT, critic-guided resampling, discrete diffusion). For images and world models, (b) with a cosine schedule and ~8-25 steps is the established default.

### Cited Findings
**Iterative refinement: Mask-Predict (Ghazvininejad et al., EMNLP 2019, arXiv 1904.09324)**
- Predict all target tokens in parallel, then repeatedly re-mask the least-confident ones and re-predict; within ~1 BLEU of a left-to-right Transformer — [abstract](https://arxiv.org/abs/1904.09324)
- Training: sample the number of masked target tokens uniformly between 1 and the sequence length, mask that many at random; loss on masked tokens (conditional masked LM) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.09324)
- Decoding schedule: linear decay, n = N·(T−t)/T masked at iteration t — [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.09324)
- BLEU vs iterations (base, distilled): En-De T=1 18.05, T=2 22.91, T=4 25.94, T=10 27.03 vs AR base 27.74. De-En 21.83 / 29.90 / 30.53 vs 31.09. En-Ro 27.32 / 31.08(T=2) / 32.53 / 33.08 vs 34.28 — [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.09324)

**MaskGIT (Chang et al., CVPR 2022, arXiv 2202.04200)** — bidirectional transformer over VQ image tokens
- Training: per example, sample mask ratio r in [0,1) via the schedule, mask those tokens, loss = NLL of masked tokens only — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.04200)
- Decoding: start fully masked; each iteration predicts all masked tokens in parallel, samples them, keeps the most confident, re-masks the rest by γ(t/T); temperature annealing on the confidence adds randomness "to encourage more diversity" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.04200)
- Schedule ablation (ImageNet 256, best T per schedule): cosine T=10 FID 6.06; square T=10 6.35; cubic T=9 7.26; exponential T=8 7.89; linear T=16 7.51; square-root T=32 12.33; logarithmic T=60 29.17. Concave schedules (few tokens committed early, many late) win — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.04200)
- "More iterations are not necessarily better": each schedule has a sweet spot then degrades; cosine peaks earliest, ~8-12 iterations — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.04200)
- vs autoregressive VQGAN: ImageNet 256 FID 6.18 (8 steps) vs 15.78 (256 steps); ImageNet 512 FID 7.32 (12 steps) vs 26.52 (1024 steps); 30-64x faster wall-clock — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.04200)

**Token-Critic (Lezama et al., ECCV 2022, arXiv 2209.04439)**
- A second transformer trained (BCE) to tell which tokens in a partially-regenerated real image were filled in by the generator; at sampling, the lowest-critic-score tokens are re-masked instead of using generator confidence. Generator frozen — [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.04439)
- Motivation: generator confidence is sensitive to modelling errors; accept/reject is per-token independent; greedy keeps cannot be revised — [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.04439)
- 18 steps: ImageNet 256 FID 6.56 (MaskGIT) → 4.69 (+critic); ImageNet 512 8.48 → 6.80. Cost: two forward passes per step — [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.04439)

**Knowledge distillation**
- Gu et al.: training on AR-teacher outputs instead of ground truth gave ~+5 BLEU consistently (IWSLT dev) — [Gu et al. 2018](https://ar5iv.labs.arxiv.org/html/1711.02281)
- Mask-Predict raw vs distilled, En-De: T=1 10.64 → 18.05; T=4 22.25 → 25.94; T=10 24.61 → 27.03. The gap shrinks with more iterations (7.4 BLEU at T=1, 2.4 at T=10) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.09324)
- Zhou et al. 2020 (ICLR, arXiv 1911.02727): distillation helps by reducing data complexity (conditional entropy of targets given aligned source, i.e. fewer modes per input). Weaker students do best with less complex distilled data; vanilla NAT peaks with a small teacher, LevT with a big one. Beam-search distillation best for vanilla NAT (19.5 BLEU) vs greedy 18.9, top-10 sampling 14.6, pure sampling 6.6 — [ar5iv](https://ar5iv.labs.arxiv.org/html/1911.02727)
- Side effect: distillation biases NAT toward high-frequency tokens and hurts low-frequency ones; exposing raw data restores them (Ding et al., ICLR 2021, arXiv 2012.14583; also "Rejuvenating Low-Frequency Words", ACL 2021) — [arXiv 2012.14583](https://arxiv.org/pdf/2012.14583); [ACL 2021](https://preview.aclanthology.org/fix_video/2021.acl-long.266/) (read via search summaries, tables not checked)

**Glancing training (GLAT, Qian et al., ACL 2021, arXiv 2008.07905)**
- Two-pass training: first parallel prediction Ŷ; then reveal S = λ·Hamming(Y, Ŷ) random reference tokens as decoder inputs and train on the rest. More tokens are revealed while the model is bad, fewer as it improves; λ annealed 0.5 → 0.3 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2008.07905)
- Single-pass BLEU WMT14 En-De/De-En: GLAT 25.21 / 29.84 vs NAT-base 20.36 / 24.81 and Transformer 27.48 / 31.27, at 15.3x speed-up; +NPD(m=7) 26.55 / 31.02 at 7.9x. All with distillation — [ar5iv](https://ar5iv.labs.arxiv.org/html/2008.07905)

**Discrete diffusion**
- D3PM (Austin et al., NeurIPS 2021, arXiv 2107.03006): discrete forward processes with uniform, absorbing (mask), discretized-Gaussian or nearest-neighbour transition matrices; the absorbing variant connects to masked generative models; "the choice of transition matrix is an important design decision"; loss = VLB + auxiliary cross-entropy — [abstract](https://arxiv.org/abs/2107.03006)
- MDLM (Sahoo et al., NeurIPS 2024, arXiv 2406.07524): masked diffusion objective reduces to a weighted mixture of masked-LM losses; approaches AR perplexity — [abstract](https://arxiv.org/abs/2406.07524)
- Copula diffusion (arXiv 2410.01949): see above — few-step quality limited by per-token independence; fix by an external dependency model — [abstract](https://arxiv.org/abs/2410.01949)

**World-model use of these methods**
- Genie (Bruce et al., ICML 2024, arXiv 2402.15391): dynamics model is a MaskGIT transformer predicting next-frame tokens; 25 MaskGIT steps per frame, temperature 2, random sampling; training masks input tokens with a Bernoulli rate sampled uniformly in [0.5, 1] — [arXiv HTML](https://arxiv.org/html/2402.15391)
- MaskViT (Gupta et al., ICLR 2023, arXiv 2206.11894): video prediction with iterative mask-schedule decoding; trains with a variable (not fixed) mask ratio to close the train/inference gap; up to 512x faster than autoregressive, enabling real-robot planning — [search summary of arXiv 2206.11894](https://arxiv.org/abs/2206.11894v2)
- Copilot4D (Zhang et al., ICLR 2024, arXiv 2311.01017): recasts MaskGIT as absorbing-uniform discrete diffusion: also uniform-noise some unmasked tokens (up to η=20%), compute loss on all tokens, and let the sampler revise already-decoded tokens. Mask count ⌈cos(u·π/2)·N⌉, u~U(0,1). Training mix: 50% condition on clean past and denoise future, 40% denoise past and future jointly, 10% single frame without context (for classifier-free guidance). At 10 steps/frame, Chamfer 0.82 (MaskGIT) → 0.58 (theirs). CFG: w=0 Chamfer 1.40 → w=1 0.56 — [arXiv HTML](https://arxiv.org/html/2311.01017v4)

### Inferences
- With only 64 tokens per frame, the step count can be small. Text (N≈20-30 tokens) recovers most of the AR gap by T=4 and nearly all by T=10; images (256 tokens) peak at 8-12 with cosine. A 64-token frame plausibly needs ~4-8 steps; worth sweeping 1/2/4/8/16 because MaskGIT shows too many steps can hurt.
- Training requirement for iterative decoding: the target frame must be trained under random partial masking (MaskGIT/Genie/MaskViT style), not only predicted from an all-mask or all-previous-frame input, otherwise later decode steps see input distributions never seen in training.
- Copilot4D's "noise some unmasked tokens + loss on all tokens + revisable sampling" directly addresses committing a wrong "keep" early: the model learns to fix tokens it is shown, not just fill blanks.
- Copilot4D's CFG result (Chamfer 1.40 → 0.56) suggests that sharpening the conditional on context/action is a strong lever for "move vs stay" decisions; this is inference for the robot-arm case, not measured there.
- Distillation's analogue for a world model would be training on targets from a model that has already picked one mode; in a deterministic simulator the data is nearly unimodal given full state+action, so most residual multimodality likely comes from missing information in the context (partial observability, action timing) rather than the data itself.

### Gaps
- Exact MaskViT step counts and schedule ablation numbers were not retrieved.
- MDLM's exact perplexity gap and sampling-step results were not retrieved (abstract only).
- Lee, Mansimov & Cho 2018 (iterative refinement as denoising, arXiv 1802.06901), CTC-based NAT (Libovický & Helcl 2018; Imputer, Saharia et al. 2020) and MUSE (arXiv 2301.00704) were not read this session; numbers for them are absent.
- Copilot4D does not ablate its three changes separately.

## How many refinement steps recover most of autoregressive quality?

### Takeaway
Text NAT: 4 steps recover most of the gap and 10 steps get within ~0.5-1 BLEU of AR. Images: MaskGIT at 8-12 steps beats a 256-step AR baseline on FID. World models in practice use 10-25 steps per frame. More steps are not monotonically better for images.

### Cited Findings
- Mask-Predict WMT14 En-De: T=1 18.05, T=2 22.91, T=4 25.94, T=10 27.03, AR 27.74 — i.e. T=4 recovers ~82% of the T=1→AR gap, T=10 ~93% — [ar5iv](https://ar5iv.labs.arxiv.org/html/1904.09324) (percentages are my arithmetic)
- MaskGIT: cosine best at ~8-12 steps; more steps degrade; 8 steps FID 6.18 vs AR VQGAN 15.78 at 256 steps — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.04200)
- Token-Critic: 18 steps best for critic sampling — [ar5iv](https://ar5iv.labs.arxiv.org/html/2209.04439)
- Genie: 25 steps per frame; Copilot4D: 10 steps per frame — [Genie](https://arxiv.org/html/2402.15391); [Copilot4D](https://arxiv.org/html/2311.01017v4)
- One-pass alternative: GLAT single pass gets within ~2.3 BLEU of AR on En-De (25.21 vs 27.48) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2008.07905)

### Inferences
- MaskGIT's FID beating AR is partly a sampling-quality effect (AR raster order is a poor fit for images), not proof that parallel is generally better; the text results are the cleaner "how close to AR" measurement.

### Gaps
- No source found giving step-count curves for a 64-token frame or for action-conditioned next-frame prediction specifically.

## Does independent per-token prediction collapse toward "copy the input" or "most frequent token"?

### Takeaway
There is solid evidence for a conservative bias in the adjacent settings: NAT repeats tokens and over-produces high-frequency words, and video prediction treats copy-last-frame as a strong trivial solution. I found no paper that measures "keep previous token" collapse in a parallel-decoded discrete world model directly.

### Cited Findings
- NAT one-pass outputs have high token repetition (16.72% En-De at T=1), removed by a few refinement steps — [Mask-Predict ar5iv](https://ar5iv.labs.arxiv.org/html/1904.09324)
- Distilled NAT is biased toward high-frequency tokens and loses low-frequency ones; the AT teacher itself generates more high-frequency and fewer low-frequency tokens; gains from the fix concentrate on low-frequency words — [Ding et al., arXiv 2012.14583](https://arxiv.org/pdf/2012.14583) (via search summary)
- Video prediction: copying the last input frame is a strong baseline on PSNR/SSIM because most pixels are static, which makes full-image metrics weak — [Mathieu et al. 2016, arXiv 1511.05440](https://arxiv.org/pdf/1511.05440) (via search summary); a NeurIPS 2018 review flags a loss whose trivial optimum is reproducing the current frame and asks for copy-last-frame baselines — [NeurIPS 2018 reviews](https://papers.neurips.cc/paper_files/paper/2018/file/ede7e2b6d13a41ddf9f4bdef84fdc737-Reviews.html)
- Huang et al.: ML training of NAT fits marginals and drops inter-token dependence — [arXiv 2206.05975](https://arxiv.org/abs/2206.05975)

### Inferences
- Why "keep" wins ties: in a frame where ~80% of cells are unchanged, "same as previous" is the most frequent token at most positions. When a changing cell's distribution splits across several "moved" tokens (different exact positions/appearances) but "keep" is a single token, "keep" can be the argmax even when P(change) > 0.5 in total. This is a mode-splitting effect of argmax, not just a 50/50 tie; checking P(keep) against the summed mass of all non-keep tokens per cell would test it.
- The thin, fast forearm link is the image analogue of NAT's low-frequency words: few cells, large per-step displacement, so its "moved" tokens are rare and spread over many codes. Expect it to be hurt most by any frequency-biased fix (distillation-style) and helped most by iterative decoding that commits confident body cells first and then re-predicts the forearm cells conditioned on them.
- Coordinated decoding removing the freeze but leaving motion inaccurate matches the literature split: decoding fixes coherence (which mode), but accuracy within the chosen mode depends on the per-token model quality and conditioning (cf. distillation gap shrinking but not closing with iterations).

### Gaps
- No primary source found that measures a "copy previous frame token" bias in MaskGIT-style or one-shot discrete video world models (searched; nothing direct).
- Ding et al. full tables not read; the high-frequency bias claim rests on abstracts and summaries.
