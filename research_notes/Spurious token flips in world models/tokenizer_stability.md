# Tokenizer-level methods for reducing spurious token flips

Scope: per-frame FSQ autoencoder, 64x64 flat-palette MuJoCo frames, 512 codes, GroupNorm encoder, ~15x15 px receptive field per token. Spurious flips = token changes on cells whose pixels did not change (7.4 of 13.3 pp of all frame-to-frame changes). Constraints: >=30 dB PSNR, >=70% token entropy, bit-identical tokens at batch 1 vs large batch, one RTX 5060 Laptop (8 GB).

Research budget note: ~16 searches/fetches. Several primary PDFs could not be fully read (Genie PDF too large; MAGVIT-v2 and Cosmos read only via abstracts/secondary pages). Claims marked accordingly.

## Why do per-frame quantizers flip on unchanged input, and has anyone measured token temporal consistency?

### Takeaway
No paper found that directly measures spurious flip rate of an image/video VQ or FSQ tokenizer on unchanged regions; the closest are a speech tokenizer stability paper (StableToken, ICLR 2026) and a 2026 world-model streaming paper that defines a per-position "token ID changed vs previous frame" indicator. Mechanistically, the strongest candidates for this project are (a) latents sitting near FSQ rounding boundaries, and (b) spatial leakage — from the receptive field and, likely more importantly, from per-frame GroupNorm statistics, which let arm motion anywhere in the frame shift every cell's latent.

### Cited Findings
- FSQ (Mentzer et al., Sept 2023) bounds each latent channel and rounds each scalar independently to the nearest level, with a straight-through estimator for gradients; there is no commitment loss or codebook. — [FSQ paper, arXiv 2309.15505](https://arxiv.org/pdf/2309.15505)
- A secondary overview notes FSQ training often uses a differentiable proxy (additive noise, stochastic surrogates or STE) while inference uses hard rounding — a train/test mismatch at bin edges. — [EmergentMind FSQ topic page](https://www.emergentmind.com/topics/finite-scalar-quantization-fsq) (aggregator, not primary)
- In learned compression, straight-through training can pile latent density up at quantization boundaries (the "twin tower effect"); reported for a different quantizer setup, so only an analogy for FSQ. — [EmergentMind FSLQ topic page](https://www.emergentmind.com/topics/finite-scalar-latent-quantization-fslq) (aggregator)
- iFSQ (Jan 2026) argues equal-interval FSQ leaves edge bins underused, i.e. the latent distribution across bins is uneven. Bears on where latents concentrate, not directly on flips. — [iFSQ, arXiv 2601.17124](https://arxiv.org/html/2601.17124)
- Quantization-aware training of network weights shows the same phenomenon in a different setting: quantized weights "oscillate between two grid-points" during training, hurting accuracy (Nagel et al., ICML 2022). — [Overcoming Oscillations in QAT, PMLR](https://proceedings.mlr.press/v162/nagel22a.html)
- StableToken (Song et al., arXiv Sept 2025, ICLR 2026): semantic speech tokenizers change output tokens drastically under meaning-irrelevant perturbations even at high SNR. Authors blame "a brittle single-path quantization architecture" and a training signal that ignores intermediate token stability; they measure stability with Unit Edit Distance. Numbers not available from the abstract. — [StableToken, arXiv 2509.22220](https://arxiv.org/abs/2509.22220)
- A 2026 world-model streaming paper defines a per-position indicator of whether a token ID differs from the previous ground-truth frame and evaluates only on changing positions, because static regions would dominate the metric (stride-16 VQ-U-Net, 8192 codes). — [Network-Efficient World Model Token Streaming, arXiv 2605.09886](https://arxiv.org/html/2605.09886)
- Identifiable Token Correspondence (2026) reports temporal inconsistency in token-based transformer world models over long rollouts and addresses it by reusing earlier-frame tokens via optimal transport. — [ITC, arXiv 2605.16457](https://arxiv.org/html/2605.16457v3)
- General statement in video-tokenizer literature: frame-by-frame image tokenizers cause flicker, motivating temporal (3D) tokenizers. — [ARSS, arXiv 2509.23008](https://arxiv.org/pdf/2509.23008); [VFRTok, arXiv 2505.12053](https://arxiv.org/pdf/2505.12053)

### Inferences
- Boundary sensitivity: with FSQ, a cell flips if any one of its d channels crosses a half-level boundary. With 512 codes (e.g. levels 8x8x8, d=3) a cell has 3 chances to flip per frame. A tiny latent change (from rendering noise, neighbor motion, or float nondeterminism) only flips cells whose latents sit within that change of a boundary. So flip rate ~ (density of latents near boundaries) x (size of the perturbation). Both levers are addressable (margin loss reduces the first; locality/normalization fixes reduce the second).
- GroupNorm leakage (likely the single most important project-specific hypothesis): GroupNorm computes mean/variance per sample over all spatial positions in each channel group. When the arm moves, those statistics change, which shifts the normalized activation of every cell in the frame — including cells whose 15x15 receptive field is entirely static. This is a global path from motion to every latent, independent of receptive-field size. It is also compatible with the batch-1 vs large-batch invariance requirement (GroupNorm is per-sample), which is probably why it was chosen. Test: for flipped static cells, check whether flip probability depends on distance from the nearest moving pixel. If it is roughly flat with distance, the leak is global (normalization); if it decays beyond ~7 px, it is receptive-field leakage.
- Rendering itself: the dataset is deterministic flat-palette rendering on one GPU, so pixel noise on static regions is probably zero; spurious flips are then caused by the encoder, not the input. Worth confirming by counting cells whose pixels are bit-identical yet flip.
- Nondeterminism: cuDNN algorithm choice can differ by batch size, giving last-bit float differences; near-boundary latents then round differently. The project already requires batch-invariant tokens, so this is presumably controlled, but a margin loss would make it robust rather than lucky.
- Twice-the-flips variant: consistent with either a larger receptive field / more global context, or a latent distribution denser near boundaries. Comparing the distance-to-boundary histogram of the two tokenizers would discriminate.

### Gaps
- No published flip-rate measurement for an image VQ/FSQ tokenizer on static regions was found.
- Could not read StableToken's numbers or loss definition; abstract only.
- No paper found on GroupNorm-induced spatial leakage in tokenizers specifically; this is inference from how GroupNorm works.

## Temporal-consistency losses and regularizers for VQ/FSQ tokenizers

### Takeaway
Explicit token-consistency losses for image tokenizers are rare in the literature; most video work gets consistency from architecture (conditioning on previous frames) instead. The cheap, well-motivated options for this project are a pre-quantization latent consistency loss on static cells between adjacent frames, plus StableToken-style training on perturbed inputs.

### Cited Findings
- A 2026 adaptive-tokenization paper attributes flicker to the same region being reconstructed by different mechanisms on consecutive frames, and adds a pixel-space temporal L1 loss on consecutive-frame differences. — [Adaptive Tokenisation via Temporal Redundancy Masking, arXiv 2606.06158](https://arxiv.org/pdf/2606.06158)
- TECO (Yan et al., ICML 2023) gets temporal consistency from architecture: a CNN encoder encodes each frame conditioned on the previous frame and quantizes with an l2-normalized codebook; it is not an explicit consistency loss. — [TECO, PMLR v202](https://proceedings.mlr.press/v202/yan23b/yan23b.pdf)
- DicFace (ICCV 2025) targets flicker in video face restoration by reformulating the codebook latent space (Dirichlet-constrained) and adding a temporal transformer to stabilize discrete codes across frames. — [DicFace, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Dirichlet-Constrained_Variational_Codebook_Learning_for_Temporally_Coherent_Video_Face_Restoration_ICCV_2025_paper.pdf)
- StableToken uses parallel branches merged by "bit-wise voting" to produce one stable token sequence, and trains with attention to intermediate token stability. — [StableToken, arXiv 2509.22220](https://arxiv.org/abs/2509.22220)
- Temporal regularization for AR video generators measures consistency via angular differences between consecutive frames (generator-side, not tokenizer). — [Temporal Regularization Makes Your Video Generator Stronger, arXiv 2503.15417](https://arxiv.org/html/2503.15417v1)

### Inferences
- Static-cell latent consistency loss (recommended first experiment, low cost): sample adjacent frame pairs (t, t+1); compute a mask of cells whose receptive field pixels are identical (pixel diff == 0, dilated by the receptive field); add lambda * ||z_t - z_{t+1}||^2 over masked cells on the *pre-rounding* bounded latent. Cost: one extra encoder pass per pair, ~2x encoder compute, no architecture change, batch invariance preserved. Risk: if the leak is GroupNorm, this loss must fight the normalization and may push the encoder toward ignoring global stats — which is the desired effect but may cost PSNR; watch the 30 dB floor.
- Combine with a margin term (next section): consistency alone shrinks the perturbation; margin moves latents away from edges. Together they attack both factors of the flip-rate product.
- Augmentation-consistency variant (no pairs needed): encode x and a copy where pixels outside a random patch are replaced by another frame's pixels (simulated motion elsewhere); penalize latent change on cells far from the edited region. Directly trains locality.
- The StableToken voting idea maps to FSQ cheaply at inference: run K encoder variants (or the same encoder on K slightly shifted/jittered inputs) and take per-channel majority of rounded levels. But it multiplies encode cost by K and does not help if the perturbation is systematic (GroupNorm), so lower priority.

### Gaps
- Could not find a published FSQ/VQ image tokenizer trained with an explicit adjacent-frame latent consistency loss plus a reported flip-rate reduction; effect size for this project is unknown until measured.
- Pixel-diff temporal loss (2606.06158) operates on reconstructions, not tokens; whether it reduces token flips was not established.

## Quantization tricks: hysteresis, dead zones, margin losses, noise injection, commitment-like terms

### Takeaway
The best-evidenced idea transfers from quantization-aware training: Nagel et al.'s "oscillation dampening" regularizer pulls values toward bin centers, which for FSQ is exactly a margin/commitment-to-center loss on the bounded latent. Inference-time hysteresis (stateful rounding that keeps the previous level unless the latent clearly crosses) is not found in the literature for tokenizers but is trivially implementable; it breaks per-frame statelessness, so it fits only if tokens may depend on the previous frame's tokens.

### Cited Findings
- Nagel et al. (ICML 2022) propose two fixes for grid-point oscillation: (1) oscillation dampening, a regularizer pulling latent weights toward bin centers; (2) iterative freezing, which tracks oscillation frequency and freezes highly oscillating weights at their most frequent integer. They report state-of-the-art 3- and 4-bit accuracy on MobileNetV2/V3 and EfficientNet-lite; code from Qualcomm AI Research on GitHub. — [PMLR](https://proceedings.mlr.press/v162/nagel22a.html); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2203.11086)
- Dead-zone quantizers with a tunable rounding offset (0 to 0.5, where 0.5 = ordinary rounding) are used in learned image compression. — [Variable Rate Image Compression with Dead-zone Quantizer, arXiv 2004.05855](https://arxiv.org/pdf/2004.05855)
- A video-coding patent proposes setting larger quantization step sizes where content is temporally predicted, using a temporal correlation map between preceding and current frames, and ties the quantization offset to dead-zone width. — [US12231636](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12231636)
- RobusTok ("Image Tokenizer Needs Post-Training", Qiu et al., Sept 2025) trains with latent perturbation: random tokens are swapped for nearby codebook entries, with an annealing schedule, then post-trains only the decoder. Goal is decoder robustness to generator errors (gFID 1.60 -> 1.36 with post-training at ~400M generator), not encoder stability. — [arXiv 2509.12474](https://arxiv.org/html/2509.12474v1)
- No source found describing temporal hysteresis (thresholds depending on the previous frame's level) for VQ/FSQ tokenizers or learned video codecs. — search negative; see [search-adjacent result arXiv 2605.18378](https://arxiv.org/pdf/2605.18378), which reports only that aggressive quantization causes flicker and that temporal consistency degrades nonlinearly with compression.

### Inferences
- Margin loss for FSQ (low cost, keeps per-frame statelessness and batch invariance): on the bounded, scaled latent u (where levels sit at integers), the fractional part f = u - round(u) lies in [-0.5, 0.5]; boundaries are at |f| = 0.5. Add lambda * mean(f^2) (Nagel-style dampening) or a hinge lambda * relu(|f| - (0.5 - m)) to keep latents at least m from edges. Effect: any latent perturbation smaller than m cannot flip a token. Risk: pulling to centers reduces the effective information the STE passes to the decoder and can collapse rarely-used levels, lowering entropy; anneal lambda in late training and track the 70% entropy floor. Could be applied as fine-tuning of the existing Phase 1 checkpoint (cheap: a few epochs).
- Noise injection (low cost): during training, add uniform noise of half-width delta to u before rounding (still STE), or randomly substitute the neighboring level when |f| > 0.5 - delta. The decoder learns that edge latents are ambiguous, and the reconstruction loss pushes the encoder to move informative latents away from edges. Closely related to RobusTok's latent perturbation, but aimed at the encoder side.
- Diagnostic before choosing: histogram |f| over all latents, and separately over channels of cells that spuriously flip. If flipping cells concentrate at |f| near 0.5 while the global histogram is not edge-heavy, a margin loss will help a lot; if flipping cells' latents move by large amounts (larger than a plausible margin), the cause is leakage and margin alone will not fix it.
- Hysteresis at tokenization time (zero training cost): keep previous frame's level l_prev per channel; emit l_prev unless |u - l_prev| > 0.5 + h. This is a deterministic, batch-invariant function of (u_t, tokens_{t-1}) and suppresses small oscillations. Caveats: (1) tokens are no longer a pure function of the frame, so the decoder sees tokens that may be up to h off-center — measure PSNR; (2) the world model must then be trained on hysteresis tokens, and at rollout time the model predicts tokens directly so hysteresis does not apply there; (3) first frame of each episode is stateless. A dead-zone (offset < 0.5) is the stateless cousin but does not address flipping between two adjacent levels, so it is less useful here.
- Iterative-freezing analogue: at dataset tokenization time, track per-cell flip-back patterns (A->B->A within k frames with static pixels) and replace them with the majority level. This is a post-hoc smoother, see last section.

### Gaps
- No direct evidence that a margin loss reduces token flips in a tokenizer (evidence is from weight QAT only).
- Effect of margin loss on token entropy for FSQ is unmeasured in the literature found.

## Video tokenizers that encode temporally (MAGVIT-v2, causal 3D CNN, Cosmos, Genie ST-ViViT, iVideoGPT, Delta-IRIS) — do they reduce flip noise?

### Takeaway
Temporal tokenizers are motivated by flicker, and Genie's ablation shows a causal spatio-temporal tokenizer beats a spatial-only one on both FVD and controllability, but none of these papers reports a token flip rate. The designs most relevant to a small world model are conditional/delta tokenizers (iVideoGPT, Delta-IRIS, TECO), which by construction spend tokens only on what changed — the direct structural fix for spurious flips — at the cost of tokens depending on past frames.

### Cited Findings
- Genie (Bruce et al., ICML 2024) tokenizer ablation (patch size 10 for all): ViT spatial-only 230M params, 0.3 GB, FVD 114.5, delta-t PSNR 1.39; C-ViViT 225M, 1.6 GB, FVD 272.7, delta-t PSNR 1.37; ST-ViViT 205M, 0.9 GB, FVD 81.4, delta-t PSNR 1.66. Temporal layers use a causal mask so each z_t contains information from all previous frames; final Platformers tokenizer: 1024 codes, latent dim 32, patch size 4. C-ViViT was reported to overfit. — [Genie, arXiv 2402.15391](https://arxiv.org/html/2402.15391)
- GenieRedux replication uses ST-ViViT for tokenizer and LAM and reports tokenizer PSNR 38.25 vs Genie's 35.7 (not a controlled ablation). — [GenieRedux, arXiv 2409.06445](https://arxiv.org/html/2409.06445v1)
- MAGVIT-v2 (Yu et al., Oct 2023) uses a temporally causal 3D CNN so the first frame is encoded independently (shared image/video vocabulary), and lookup-free quantization (LFQ) for large vocabularies. Details from secondary sources only. — [arXiv 2310.05737](https://arxiv.org/pdf/2310.05737v1); [seminar slides](https://pml4sc.github.io/assets/seminar/2025_07_08_Language%20model%20beats%20diffusion.pdf); Open-MAGVIT2 replication [arXiv 2409.04410](https://arxiv.org/html/2409.04410v2)
- Cosmos tokenizer (NVIDIA, Jan 2025): temporally causal architecture with 3D Haar wavelet front end, causal temporal convolution and factorized causal spatio-temporal attention; discrete variants use FSQ with levels (8,8,8,5,5,5), vocabulary 64,000; e.g. DV8x16x16 = 8x temporal, 16x16 spatial compression. Large-compute models. — [Cosmos WFM, arXiv 2501.03575](https://arxiv.org/pdf/2501.03575); [Cosmos tokenizer page](https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/)
- iVideoGPT (Wu et al., NeurIPS 2024): context frames tokenized independently with VQGAN; future frames tokenized by a conditional encoder that cross-attends to context features, so only "essential dynamics" (moving object position/pose) need encoding; reports an asymptotic 16x reduction in token sequence length. — [iVideoGPT, arXiv 2405.15223](https://arxiv.org/html/2405.15223v3)
- Delta-IRIS (Micheli, Alonso, Fleuret, ICML 2024): CNN autoencoder on 64x64 frames, encoder and decoder conditioned on the previous frame and action (concatenated channel-wise), K=4 tokens per frame, 1024-entry factorized normalized VQ codebook with EMA; channels [64,64,128,128,256]. Reconstruction L2 0.000185 with 4 tokens vs IRIS 0.001715 with 64 tokens. Random delta-tokens leave deterministic dynamics plausible while stochastic elements go erratic — evidence tokens carry only the stochastic change. Limitations: comparison confounds conditioning with token count; disentanglement shown qualitatively; decoder depends on past (possibly generated) frames, error accumulation unmeasured. — [Delta-IRIS, arXiv 2406.19320](https://arxiv.org/html/2406.19320v1); [code](https://github.com/vmicheli/delta-iris)
- TECO encodes each frame conditioned on the previous one with a VQ codebook and reports long-horizon temporal consistency better than FitVid and Perceiver AR. — [TECO](https://proceedings.mlr.press/v202/yan23b/yan23b.pdf)
- SweetTok (ICCV 2025) and VTok (2026) encode a first/key frame spatially and quantize consecutive-frame residuals for temporal information. DeltaTok (CVPR 2026) compresses the feature change between steps into one delta token, noting backgrounds stay static and only a small part changes. — [SweetTok](https://openaccess.thecvf.com/content/ICCV2025/papers/Tan_SweetTok_Semantic-Aware_Spatial-Temporal_Tokenizer_for_Compact_Video_Discretization_ICCV_2025_paper.pdf); [VTok](https://www.alphaxiv.org/abs/2602.04202); [DeltaTok, arXiv 2604.04913](https://arxiv.org/html/2604.04913v1)

### Inferences
- Large models (Cosmos, MAGVIT-v2, Genie's 200M tokenizer) are out of scope for an 8 GB laptop GPU to train; their ideas are what transfer. Delta-IRIS is the closest match in scale and resolution (64x64 CNN) and is practical on this GPU.
- Temporal compression (Cosmos 8x, MAGVIT 4x) changes the token-per-frame contract of the world model; likely too disruptive for this project's Phase 2 unless the dynamics model is being redesigned anyway.
- A conditional tokenizer (Delta-IRIS / TECO style) removes spurious flips by construction only if static content is carried by the previous-frame pathway. But: (1) tokens are no longer per-frame, so they cannot be compared with the current flip metric directly; (2) batch-1 vs large-batch invariance must still be checked (CNN conditioning keeps it, attention-based conditioning usually does too); (3) "token entropy >= 70%" becomes hard to interpret, because delta tokens are deliberately low-information on quiet frames; (4) the world model inherits error accumulation through the decoder's past-frame dependence. Medium-to-high implementation cost (new tokenizer + retrain dynamics).
- A cheaper middle ground: keep the per-frame FSQ tokenizer but add a causal temporal path only to the encoder's normalization/feature stats, or feed the previous frame's latent as extra encoder input with a consistency loss. Unproven; no source tests it.

### Gaps
- None of these papers reports a token flip rate on static regions; their consistency claims are FVD/PSNR or qualitative.
- Could not read MAGVIT-v2's or Cosmos's full tokenizer sections; architectural claims rest on abstracts, project pages and slides.
- No evidence on whether LFQ (binary per channel, boundary at 0) flips more or less than FSQ.

## Receptive-field limits: does a smaller/local encoder reduce flips from nearby motion? Locality vs entropy

### Takeaway
No external paper measures flips versus receptive field. The project's own Phase 2 handoff already holds a locality-versus-entropy curve (docs/handoff_tokenizer_decision.md), which is the best evidence available; the literature adds only that more global/temporal context improves generation quality (Genie), which pulls the other way. Normalization scope may matter more than kernel receptive field.

### Cited Findings
- Genie's tokenizer comparison shows that adding (causal) temporal context improves FVD and controllability, and that full spatio-temporal C-ViViT overfits and is more memory-hungry (1.6 GB vs 0.9 GB). — [Genie](https://arxiv.org/html/2402.15391)
- Delta-IRIS chose a CNN autoencoder over a transformer one; the transformer reconstructed static data better but learned more slowly in continual learning; authors say the best choice is likely environment-dependent. — [Delta-IRIS](https://arxiv.org/html/2406.19320v1)
- I found no source quantifying token flips as a function of encoder receptive field.

### Inferences
- With a 15x15 receptive field, motion can only leak into cells within ~7 px of the moving region via convolution. If spurious flips are spread across the whole static background, receptive field is not the main cause; GroupNorm's frame-global statistics (or any global pooling/attention) is the likelier path. Cheap fixes for that path, all batch-invariant: replace GroupNorm with a per-pixel normalization (LayerNorm over channels only, i.e. GroupNorm with spatial stats removed / "PixelNorm"), with weight standardization, or with no normalization plus careful init; or freeze the GroupNorm statistics to dataset-level running averages after training (turns it into a fixed affine layer — fully local and batch-invariant). The last option is a fine-tune-only experiment, low cost.
- Shrinking the receptive field trades token entropy and PSNR for stability: smaller context means each token sees less and must encode raw texture, which tends to raise entropy (good for the >=70% floor) but can lower PSNR. The project's own curve should answer where 30 dB is lost.
- Zero/reflect padding at frame edges is a second fixed (not motion-dependent) leak and does not cause temporal flips; ignore.

### Gaps
- Flip-versus-distance-from-motion has not been measured (in the literature or, as far as these notes know, in the project); that single measurement would tell whether receptive field or normalization is the culprit.

## Post-hoc fixes: re-tokenizing with previous-frame tokens as reference, change-detection masks

### Takeaway
Freezing tokens on cells whose receptive-field pixels are bit-identical to the previous frame is the cheapest possible fix and is exact in this setting (deterministic flat-palette renderer), but it only cleans the *training targets*; at rollout time the world model must still learn to copy static tokens. The project has already done a version of this (`--clean-flips`, r69 per recent commits). Token reuse across frames is also what recent world-model work does at the model level.

### Cited Findings
- Identifiable Token Correspondence (2026) improves long-rollout temporal consistency in token world models by selectively reusing tokens from preceding frames via optimal transport. — [ITC, arXiv 2605.16457](https://arxiv.org/html/2605.16457v3)
- Run-length tokenization (NeurIPS 2024) drops repeated patches across frames in video transformers, exploiting static content. — [Don't Look Twice, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/3181db351fd3ced43cd589b0b572675d-Paper-Conference.pdf)
- Streaming work explicitly scores only positions whose token ID changed, treating static positions as uninformative. — [arXiv 2605.09886](https://arxiv.org/html/2605.09886)
- A video-coding patent uses a temporal correlation map between frames to quantize predicted regions more coarsely. — [US12231636](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12231636)

### Inferences
- Pixel-diff freeze mask: for each cell, if all pixels in its receptive field (dilated, not just its 4x4/8x8 footprint) are identical to frame t-1, copy token_{t-1}. Exact for this renderer; zero training cost; decoder fidelity is unchanged because the frozen token decodes the same static content. Breaks pure per-frame tokenization, and whether the decoder's output for a frozen cell is still consistent depends on neighbors (decoder receptive field too). Interaction with `data_hash`/reproducibility: deterministic, so fine.
- Hysteresis (section 3) is the soft version of this that also handles cells with tiny pixel changes.
- Post-hoc methods remove noise from targets but leave the tokenizer itself noisy, so any frame encoded outside the pipeline (e.g. observations fed into a rollout) still shows flips. A tokenizer-level fix (margin loss + normalization fix) is needed if real frames are re-encoded during evaluation or control.

### Gaps
- No literature quantifies the downstream dynamics-model benefit of cleaning flip noise from targets; the project's r69 run is the only evidence here.

## Suggested priority for this project (inference, ranked by cost vs expected effect)
1. Diagnose (hours): histogram of |distance to nearest FSQ boundary| for flipping vs non-flipping channels; flip probability vs distance from nearest moving pixel; count flips on cells with bit-identical receptive fields.
2. If flips are global: freeze GroupNorm statistics or switch to per-pixel normalization; fine-tune. Low cost.
3. If flips are near-boundary: Nagel-style dampening/margin loss on the bounded FSQ latent, plus training noise injection; fine-tune existing checkpoint. Low cost. Watch the entropy floor.
4. Add a static-cell adjacent-frame latent consistency loss. Low-medium cost (~2x encoder compute).
5. Inference-time hysteresis or pixel-diff freezing for dataset tokenization. Near-zero cost, but no longer per-frame.
6. Conditional (Delta-IRIS/TECO-style) tokenizer. Medium-high cost; changes the whole token contract.
