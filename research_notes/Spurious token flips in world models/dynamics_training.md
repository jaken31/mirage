# Training a token dynamics model to stop chasing unpredictable flips (and beat persistence)

Scope: small transformer dynamics model (192-dim, 4 layers) over 512-code FSQ tokens, gate = top-1 accuracy vs persistence. Model wins on moving cells (54.8% vs 50.9%) but loses overall because it changes 4-8% of quiet-cell tokens, betting on tokenizer "spurious flips". Notes below are organised by key question. Citations are to primary papers where possible; anything without a citation is in Inferences or Gaps.

## 1. Label-noise learning applied to token targets (target cleaning, reweighting, robust losses, smoothing, noise-aware targets)

### Takeaway
The label-noise literature has two families that fit here: (a) fix the targets or loss when you know where/how the noise lives (target cleaning, forward loss correction with a transition matrix, bootstrapping), and (b) robust losses (GCE, SCE) that limit how hard the model fits any one noisy label. Because this project *knows* which cells are quiet (pixel diff) and the noise is structured (flip away from the previous code), family (a) is the better fit; "clean flips" targets are the strongest, simplest version of it, and forward correction is the principled alternative if cleaning turns out to throw away real signal.

### Cited Findings
- Deep nets fit clean structure first and memorise noise later; they *can* fit pure noise, and optimisation behaves qualitatively differently on noise vs real data. Tuned explicit regularisation (e.g. dropout) can slow fitting of noise without hurting real-data generalisation (Arpit et al., ICML 2017). — [A Closer Look at Memorization in Deep Networks](https://proceedings.mlr.press/v70/arpit17a.html)
- Forward loss correction (Patrini et al., CVPR 2017): given a noise transition matrix T (P(observed label | true label)), multiply the model's softmax by T before the CE loss, so the network learns the clean distribution while being scored against noisy labels. T can be estimated in a first phase from a model trained on noisy data, then a second model trained with the corrected loss. Forward correction consistently beat plain CE on synthetic noise across MNIST/IMDB/CIFAR; errors in estimating T reduce the gains under strong asymmetric noise. — [Patrini et al. summary (emergentmind)](https://www.emergentmind.com/papers/1609.03683); [Biquality learning review describing forward correction](https://arxiv.org/pdf/2012.09632)
- Bootstrapping (Reed et al., arXiv Dec 2014 / ICLR 2015 workshop): target = convex combination of the noisy label and the model's own prediction, either soft (predicted probabilities) or hard (predicted argmax one-hot). Bootstrap-hard and bootstrap-recon did best on corrupted MNIST. — [Reed et al. (Google Research)](https://research.google/pubs/training-deep-neural-networks-on-noisy-labels-with-bootstrapping/); [arXiv 1412.6596](https://www.arxiv.org/pdf/1412.6596)
- Arazo et al. (ICML 2019) found the original fixed bootstrap weights (0.2 hard, 0.05 soft) do *not* stop the network from fitting noisy samples, and proposed per-sample dynamic weights from a two-component mixture model fitted to per-sample loss. — [Unsupervised Label Noise Modeling and Loss Correction](https://arxiv.org/pdf/1904.11238)
- Generalized Cross Entropy (GCE, Zhang & Sabuncu 2018): loss (1 - p_y^q)/q; q -> 0 recovers CE, q = 1 is MAE. Designed to keep CE's fast early learning with MAE's noise tolerance. — [Adaptive loss paper describing GCE](https://arxiv.org/pdf/2002.06482); [GCE original, arXiv 1805.07836](https://arxiv.org/abs/1805.07836)
- Symmetric Cross Entropy (Wang et al., ICCV 2019): CE + reverse CE (RCE). Motivation: CE overfits noisy labels on easy classes and under-learns hard classes; adding RCE addresses both. — [SCE, arXiv 1908.06112](https://arxiv.org/pdf/1908.06112)
- Symmetric/robust losses tend to underfit; later work proposes asymmetric losses, and normalisation of any loss to make it robust. — [Adaptive loss for noisy labels](https://arxiv.org/pdf/2002.06482); [Asymmetric loss paper](https://arxiv.org/pdf/2106.03110.pdf)
- Label smoothing improves calibration (and beam search) and generalisation, but a label-smoothed teacher distils much worse because it erases inter-class similarity information in the logits (Müller, Kornblith, Hinton, NeurIPS 2019). — [When does label smoothing help?](https://arxiv.org/abs/1906.02629v2)

### Inferences
- **Target cleaning ("clean flips") is the right first move.** The noise is not class-symmetric: it is "true token = previous token, observed token = some neighbour code". That is the exact case where you know T's structure, so you don't need a robust loss that guesses. Cleaning sets T to the identity on quiet cells by construction.
- **Forward correction is the principled fallback** if cleaning is suspected of erasing real changes (e.g. sub-pixel motion that the pixel mask calls "quiet"). Concretely: estimate, on quiet cells, the empirical flip matrix T[prev -> observed] from the tokenizer alone (re-encode identical frames / count flips on cells with zero pixel diff). Train with loss = -log((T^T softmax)[target]) on quiet cells and plain CE on moving cells. At eval, use the raw (uncorrected) softmax argmax, which estimates the clean token, i.e. the previous one on quiet cells. Risk: a 512x512 T is sparse and noisy; restricting it to "stay with prob 1-ε, else uniform over FSQ-lattice neighbours of prev" (one or two scalars) is far more stable.
- **Label smoothing toward the previous token** (target = (1-α)·onehot(y) + α·onehot(prev)) is a soft version of cleaning. On quiet cells where y ≠ prev, argmax of the target stays y until α > 0.5, so it will not by itself stop greedy flips unless α ≥ 0.5 on quiet cells — at which point it is just cleaning. Its real value is calibration on moving cells. Standard (uniform) label smoothing does nothing specific about flips.
- **GCE/SCE are a weak fit.** They reduce the gradient from low-probability (likely noisy) labels everywhere, including on genuine moving-cell changes, which are also "hard". This risks giving back the 54.8% vs 50.9% moving-cell win. If tried, apply only to quiet cells, which again reduces to "you already know which cells are noisy, so clean them".
- **Hard bootstrapping is risky early**: the model's own argmax already flips 4-8% of quiet cells, so bootstrapping on its own predictions would entrench those flips. Arazo's per-sample mixture weighting (loss-based noisy/clean split) would mostly re-discover the quiet-cell mask you already have.
- Arpit's result explains why early stopping didn't help: if the flips are partly predictable from fine latent detail (see section 4), the model is not "memorising noise late" but learning a real-but-useless correlation early, so there is no clean early-stopping point.

### Gaps
- Found no paper applying label-noise methods specifically to tokenizer-induced target noise in next-frame token prediction; the transfer is inference.
- Did not find benchmark numbers comparing forward correction vs target cleaning when the noise location is known exactly.

## 2. Copy / residual / delta prediction (gated copy, change-vs-no-change heads, delta tokens)

### Takeaway
Two distinct ideas: (a) **output-side copy** — add an explicit "copy previous token" path with a learned gate (pointer-generator style), so "no change" is one cheap decision instead of a 512-way argmax that must beat every alternative; (b) **representation-side deltas** — retokenize so tokens encode only what changed (Δ-IRIS, iVideoGPT). (a) fits the current pipeline and is cheap; (b) attacks the root cause (flips exist because each frame is tokenized independently) but means rebuilding the tokenizer.

### Cited Findings
- Pointer-generator (See, Liu, Manning, ACL 2017): final distribution = p_gen · P_vocab + (1 - p_gen) · P_copy, with p_gen a sigmoid of context vector, decoder state and decoder input at every step. Beat the prior abstractive SOTA on CNN/DailyMail by ≥2 ROUGE. — [arXiv 1704.04368](https://arxiv.org/abs/1704.04368); [ACL anthology](https://preview.aclanthology.org/setup/P17-1099)
- Analysis of pointer-generators: the model uses the explicit switch to trade off extraction (copy) vs abstraction (generate); the mixture coefficient's statistics differ significantly between training and testing (i.e. models copy more at test time than training suggests). — [To Point or Not to Point (arXiv 2106.01581)](https://arxiv.org/pdf/2106.01581); [Controlling Decoding... Copy-Based Networks (arXiv 1803.07038)](https://arxiv.org/pdf/1803.07038)
- CopyNet (Gu et al., ACL 2016) is the earlier gated copy/generate mixture for seq2seq. — [arXiv 1603.06393](https://arxiv.org/abs/1603.06393)
- Δ-IRIS (Micheli, Alonso, Fleuret, ICML 2024): the discrete autoencoder's encoder *and* decoder are conditioned on past frames and actions, so its K=4 tokens (vocab 1024) per frame only encode the stochastic change that cannot be inferred from history + action. A causal GPT predicts Δ-tokens with CE; continuous "I-tokens" (frame embeddings) are interleaved because predicting deltas alone forces the transformer to integrate many past steps to locate entities. 25M params, 3-layer 512-dim transformer; ~10x faster than IRIS; SOTA on Crafter at several frame budgets. The paper does not discuss the all-static case and does not compare to a copy baseline. — [Δ-IRIS (arXiv 2406.19320)](https://arxiv.org/html/2406.19320v1); [code](https://github.com/vmicheli/delta-iris)
- iVideoGPT (Wu et al., NeurIPS 2024): context frames tokenized in full, future frames tokenized *conditioned on* context, so only dynamics info (position/pose of moving objects) is encoded; up to 16x fewer tokens for future frames; conditioning also helps temporal consistency. Matches/beats DreamerV3 on six Meta-World manipulation tasks. — [arXiv 2405.15223](https://arxiv.org/pdf/2405.15223)
- Static/dynamic decomposition in tokenizers: GVT stores static content once and reuses it rather than recomputing it per frame; VidTok notes that per-frame (image) tokenization ignores temporal redundancy and consistency, and adopts FSQ to avoid VQ's instability. — [GVT (arXiv 2508.11183)](https://arxiv.org/html/2508.11183v1); [VidTok blog (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/vidtok-introduces-compact-efficient-tokenization-to-enhance-ai-video-processing/)

### Inferences
- **How to add a copy gate here (cheapest version):** per output cell, a scalar head g = σ(w·h) next to the 512-way logits. Output p = g·onehot(prev) + (1-g)·softmax(logits). Train with NLL on p. Optional auxiliary BCE on g against the pixel-motion mask (quiet = 1), which uses the mask as privileged training-only supervision without touching inputs. Greedy decode: emit prev unless (1-g)·q_k > g + (1-g)·q_prev for some k. A cruder variant is a learned (or fixed) additive bias on the prev-token logit; this is basically what confidence-threshold copying does post hoc, so it is less likely to add much on its own.
- **Interaction with persistence:** at g → 1 the model *is* persistence; the gate makes persistence the default and forces the network to "pay" to deviate. That is the right inductive bias when ~88% of cells are quiet. The See et al. train/test mismatch in p_gen is a warning that the gate's behaviour must be measured on held-out data, not inferred from training loss.
- **The gate does not by itself remove the incentive to bet on flips:** if quiet-cell targets still contain flips, NLL still rewards lowering g on cells where a flip is predictable from fine context. Gate + clean targets together is the coherent combination: clean targets make "copy" the correct quiet-cell answer, the gate makes it cheap to express. The auxiliary BCE on the pixel mask gives g a clean signal even with raw targets.
- **Delta-token tokenizer is the root-cause fix** but is a Phase-2-level change: retraining the tokenizer conditioned on previous frames removes most spurious flips because the encoder can see the previous frame and has no reason to re-quantise unchanged content differently. Δ-IRIS's own lesson is that pure deltas make dynamics *harder* (need I-tokens/frame embeddings), so this trades one problem for another. Note also that persistence on delta tokens becomes "predict the null delta", which changes the gate's meaning.
- Δ-IRIS-scale models (25M params, A100) are larger than this project's model but the tokenizer idea itself is cheap at 64x64; fits 8 GB in principle.

### Gaps
- Found no paper that adds a pointer/copy gate to a video or world-model token predictor and reports results vs a copy baseline.
- Found no measurement of how often a per-frame VQ/FSQ tokenizer flips codes on unchanged pixels in published world models; Δ-IRIS and iVideoGPT motivate conditioning by token count and consistency, not by flip rate.

## 3. Loss masking / change-weighted CE / focal loss and their failure modes

### Takeaway
Up-weighting changed cells (or focal loss) increases the pressure to predict change, which is the opposite of what quiet cells need; it can help moving-cell accuracy but will tend to raise the quiet-cell flip rate unless paired with clean quiet targets or a copy default. Masking quiet cells out of the loss entirely is worse: the model gets no signal that copying is correct there.

### Cited Findings
- Focal loss (Lin et al., 2017) down-weights well-classified examples. — [arXiv 1708.02002](https://arxiv.org/abs/1708.02002)
- Mukhoti et al. (NeurIPS 2020): focal loss produces better-calibrated models than CE and, with temperature scaling, SOTA calibration while preserving accuracy; they give a sample-dependent γ schedule (FLSD-53). Secondary summary: the mechanism is lower gradient norms on confident samples, an implicit weight regulariser. — [Calibrating DNNs using Focal Loss (Oxford OATML)](https://oatml.cs.ox.ac.uk/publications/202009_Mukhoti2020Calibration.html); [code](https://github.com/torrvision/focal_calibration)
- In precipitation nowcasting, intensity-weighted losses are used to fix the tendency of average-error losses to underestimate rare heavy rain, and do improve extreme-rain accuracy. — [EGU25-19416](https://meetingorganizer.copernicus.org/EGU25/EGU25-19416.html); [Nowcasting survey (arXiv 2406.04867)](https://arxiv.org/pdf/2406.04867)
- Nowcasting studies report persistence is most competitive at the shortest horizon, with learned models only pulling ahead at longer lead times — the same "copy is hard to beat one step ahead" regime as here. — [Storm-scale evolution (arXiv 2601.17268)](https://arxiv.org/pdf/2601.17268)

### Inferences
- Focal loss pushes effort toward hard examples. On quiet cells, the "hard" examples are precisely the spurious flips, so focal loss on raw targets would likely *increase* flip-chasing. On clean targets it is harmless on quiet cells (they become easy) and might help moving cells.
- Change-weighted CE (weight w > 1 on moving cells) is a reasonable way to spend more capacity where the model already wins, but only after quiet targets are cleaned; otherwise the weighting doesn't touch the quiet-cell problem. Weighting by the pixel mask also leaks nothing into inputs.
- Known failure mode, by analogy to nowcasting: upweighting the rare "event" class buys recall at the cost of false alarms. Here a "false alarm" is a quiet-cell change, which is exactly the metric that is failing.
- Masking quiet cells out of the loss is a trap: with no quiet-cell loss, nothing teaches the model to copy there, and at test it predicts change wherever it is uncertain.

### Gaps
- Could not find a source that directly measures false-alarm inflation from weighted losses against a persistence baseline; the nowcasting link is an analogy.

## 4. Training on cleaned targets but evaluating on raw targets; Bayes-optimal prediction under random flips

### Takeaway
If flips are truly unpredictable and each flip lands on a spread of codes, the Bayes-optimal top-1 prediction on a quiet cell is the previous token, so a correct model should already copy. The fact that it doesn't means either (a) the flips are partly predictable from what the model sees (FSQ boundary proximity, flip history), so betting is locally rational on train data but doesn't generalise, or (b) overconfident/miscalibrated fitting of noise. Training on clean targets and scoring on raw targets is safe for the gate: on quiet cells it makes the model tie persistence by construction, so the whole margin must come from moving cells, where it already wins.

### Cited Findings
- Networks can fit random labels but fit patterns first; how much they memorise depends on the data. — [Arpit et al. 2017](https://proceedings.mlr.press/v70/arpit17a.html)
- CE-trained networks are often poorly calibrated (overconfident); focal loss and label smoothing improve calibration. — [Mukhoti et al. 2020](https://oatml.cs.ox.ac.uk/publications/202009_Mukhoti2020Calibration.html); [Müller et al. 2019](https://arxiv.org/abs/1906.02629v2)
- Forward correction's logic: train against noisy labels through T, evaluate the uncorrected "clean" prediction — i.e. training/eval target mismatch is the intended design, not a bug. — [Patrini et al. (summary)](https://www.emergentmind.com/papers/1609.03683)
- Teacher forcing vs autoregressive sampling mismatch causes error accumulation; GameNGen fixes it by adding Gaussian noise to context frames at training time with the noise level given as input. Without it, quality degrades after 20-30 steps. — [GameNGen (arXiv 2408.14837)](https://arxiv.org/pdf/2408.14837v1.pdf); [ICLR 2025 version](https://proceedings.iclr.cc/paper_files/paper/2025/file/b71ecea210f7159f31e46631fe5c838f-Paper-Conference.pdf)

### Inferences
- **Bayes argument (plain words):** on a quiet cell, let ε be the flip probability and suppose a flip lands on one of m neighbour codes. P(prev) = 1 - ε; P(any specific neighbour) ≤ ε. Copying wins top-1 whenever 1 - ε > max neighbour probability, which holds for any ε < 0.5 if flips are not predictable. With the observed flip rates (single-digit %), a calibrated model with no extra information always copies. So a model that changes 4-8% of quiet tokens is claiming, for those cells, that one specific alternative code is more likely than the old one. That claim is wrong on held-out data (the deficit shows it), so it is either fitting train-specific noise or exploiting a weak real signal that is overestimated.
- **Why flips may be partly predictable (testable):** FSQ rounds each latent dim to a grid; a cell whose latent sits near a rounding boundary will flip under tiny pixel-level noise (rendering, anti-aliasing, the encoder's receptive field spanning moving neighbours). The model sees previous tokens and maybe neighbours that correlate with "near boundary". Also, a cell that oscillates A→B→A gives a real signal ("flip back"). Check: on the val set, condition the flip rate on "this cell flipped last step" and on "a neighbour moved". If flip-back is common, then a learned predictor *could* beat persistence on raw targets for those cells, and clean targets would throw that away — small, but worth measuring before declaring it noise.
- **"Pixels unchanged" vs receptive field:** the tokenizer's encoder sees a neighbourhood; a cell with zero pixel diff inside its patch can still change code because a neighbouring patch moved. Those flips are not pure noise — they're predictable from motion next door. If the quiet mask is defined only from the cell's own pixels, cleaning will also erase these, which raw-target eval will then count as errors. Size of this effect is checkable by splitting quiet cells into "quiet with moving neighbour" vs "quiet with quiet neighbourhood".
- **Clean-train/raw-eval consequence:** on quiet cells a clean-trained model should converge to exact persistence, so its quiet-cell accuracy equals persistence's (including persistence's own losses to flips). Overall accuracy then beats persistence iff moving-cell accuracy beats persistence's moving-cell accuracy, which r69-style numbers already show. This is the expected positive outcome, not a contamination risk. The thing to watch is whether cleaning also degrades moving cells (e.g. the model learns "copy" too strongly near motion boundaries).
- **Noisy inputs:** inputs still contain flips. The model must learn that an input flip at t-1 is not evidence of motion. The previous 30% input-corruption experiment shows input noise isn't the main lever, but clean targets + noisy inputs teaches exactly the invariance needed. For multi-step rollouts, GameNGen-style noise augmentation (with corruption level as input) is the standard drift fix; it doesn't address single-step top-1.
- **Why temperature sampling and threshold copying failed:** both act on the output after the model has already put probability mass on a specific wrong code. If that mass exceeds P(prev), no threshold on max-prob separates good from bad flips cleanly. This points at the training targets, not the decoder.

### Gaps
- No primary source found that formally analyses Bayes-optimal next-token prediction under tokenizer flip noise; the argument above is derivation, not citation.
- The predictability of FSQ flips (boundary proximity, flip-back) is a hypothesis; I found no published measurement.

## 5. Masked / parallel token prediction (MaskGIT, Genie) and discrete diffusion vs autoregressive in a copy-heavy regime

### Takeaway
Genie and TECO use MaskGIT-style parallel decoding of next-frame tokens; none of the sources report it helping specifically with copy-heavy prediction, and parallel decoding typically starts from all-mask, which discards the strongest prior (the previous frame). Discrete diffusion with a transition kernel biased toward the previous state is the conceptually closest "copy-by-default" generative scheme, but it is unproven here and adds inference steps. For a one-step top-1 gate, the existing model type (all next-frame tokens predicted from context in one pass) plus a copy default is a better bet.

### Cited Findings
- Genie (Bruce et al., ICML 2024): video tokenizer + latent action model + dynamics model; the dynamics model is a decoder-only MaskGIT transformer predicting next-frame tokens from past tokens and latent actions. Masking probability is drawn from U(0.5, 1), unlike standard MaskGIT. Trained on a filtered 30k hours of 2D platformer gameplay (some summaries say 200k hours; the paper text says 30k). Official code not released; GenieRedux and Jasmine reimplement it, Jasmine with MaskGIT, fully-causal and diffusion variants. — [Genie (arXiv 2402.15391)](https://arxiv.org/pdf/2402.15391); [GenieRedux (arXiv 2409.06445)](https://www.arxiv.org/pdf/2409.06445v2); [Jasmine (arXiv 2510.27002)](https://arxiv.org/html/2510.27002v1)
- TECO (Yan, Hafner, James, Abbeel, ICML 2023): compresses VQ latents, runs a causal temporal transformer, decodes each frame with a spatial MaskGIT; MaskGIT credited with sharper and faster generation. Better long-horizon consistency than baselines on DMLab, Minecraft, Kinetics-600. — [TECO (arXiv 2210.02396)](https://arxiv.org/pdf/2210.02396v2.pdf); [code](https://github.com/wilson1yan/teco)
- MaskGIT (Chang et al., CVPR 2022): bidirectional transformer predicting masked tokens in parallel with iterative confidence-based unmasking. — [arXiv 2202.04200](https://arxiv.org/abs/2202.04200)
- D3PM (Austin et al., NeurIPS 2021) defines discrete diffusion with structured transition matrices (uniform, absorbing/mask, nearest-neighbour), i.e. the corruption kernel is a design choice. — [arXiv 2107.03006](https://arxiv.org/abs/2107.03006)

### Inferences
- In a copy-heavy regime, MaskGIT's confidence-ordered unmasking would naturally fill confident (quiet, copyable) cells first, which is benign. But at 1-step top-1 evaluation with all cells predicted at once, MaskGIT's iterative refinement only helps by letting cells condition on neighbours' decoded values — useful for spatially coherent motion (moving arm), irrelevant for independent flips.
- A "start from previous frame" variant (discrete diffusion/editing where the initial state is the previous frame's tokens, not all-mask, and the model decides what to rewrite) is the generative analogue of a copy gate. No source here tests it on world models; treat as speculation.
- Swapping AR/MaskGIT does not change the core issue: whatever the decoder, CE on raw targets rewards predicting flips where they look predictable. Target fix comes first.
- 8 GB budget: MaskGIT at 64x64 with a small tokenizer grid is cheap; inference cost is K refinement passes per frame.

### Gaps
- No source found comparing AR vs MaskGIT vs discrete diffusion dynamics models against a persistence/copy baseline on mostly-static scenes.

## 6. Known world models and how they relate to copy baselines on mostly-static scenes

### Takeaway
None of the major token world models (IRIS, Δ-IRIS, TECO, Genie, iVideoGPT) report a per-token copy baseline. The ones that implicitly handle static content well do it in the tokenizer (Δ-IRIS, iVideoGPT condition on past frames; GVT stores static content once) or by avoiding discrete tokens altogether (DIAMOND, GameNGen use continuous/diffusion). So "beat persistence on top-1 token accuracy" is a stricter, less-studied gate than these papers use.

### Cited Findings
- IRIS (Micheli et al., ICLR 2023): discrete autoencoder + GPT world model over per-frame tokens (16-64 tokens/frame); Δ-IRIS reports IRIS at 16 tokens has poor reconstructions and attention cost grows with token count. — [Δ-IRIS paper comparing to IRIS](https://arxiv.org/html/2406.19320v1); [IRIS (arXiv 2209.00588)](https://arxiv.org/abs/2209.00588)
- Δ-IRIS: conditioning the tokenizer on past frames + actions so tokens encode only unpredictable change; baselines are DreamerV3 and IRIS, not copy. — [arXiv 2406.19320](https://arxiv.org/html/2406.19320v1)
- iVideoGPT: conditional tokenization of future frames given context; only dynamics info encoded. — [arXiv 2405.15223](https://arxiv.org/pdf/2405.15223)
- DIAMOND (Alonso et al., NeurIPS 2024): argues discrete-latent compression can drop visual detail important for RL; uses pixel-space diffusion; mean HNS 1.46 on Atari 100k, best for agents trained fully in a world model. — [arXiv 2405.12399](https://arxiv.org/pdf/2405.12399); [code](https://github.com/eloialonso/diamond)
- GameNGen: diffusion over latent frames with context-noise augmentation; ~20 FPS on one TPU; next-frame PSNR 29.4. — [arXiv 2408.14837](https://arxiv.org/pdf/2408.14837v1.pdf)
- Genie / TECO: MaskGIT dynamics; evaluated on generation quality and consistency, not vs copy. — [Genie](https://arxiv.org/pdf/2402.15391); [TECO](https://arxiv.org/pdf/2210.02396v2.pdf)

### Inferences
- The field's main answer to "most of the frame is static" is architectural (conditional/delta tokenization), not a training trick on the dynamics side. That supports keeping a delta/conditional tokenizer as the explicit fallback if clean targets + copy gate don't clear the gate.
- Continuous-latent / diffusion models sidestep spurious flips entirely (no quantisation boundary to cross), but change the metric — top-1 token accuracy no longer exists — so they're out of scope for the current gate.

### Gaps
- Oasis (Decart/Etched 2024) not researched; I found no primary technical paper in this pass.
- No published world model found that reports top-1 token accuracy against persistence; can't say how big a margin is normal.

## Ranked recommendations for this project (inference, not cited)

1. **Clean quiet-cell targets (planned).** Expected: quiet-cell accuracy = persistence exactly; overall margin = moving-cell margin × ~12%. Before trusting it, split quiet cells by "neighbour moved" and "flipped last step" to measure whether any quiet-cell flips are actually predictable; if so, clean only the "quiet neighbourhood" subset.
2. **Copy gate (pointer-generator style) + auxiliary BCE on the pixel-motion mask.** Cheap (one scalar head), makes copy the default, gives a direct change-probability readout for diagnostics. Combine with (1), not instead of it.
3. **Change-weighted CE (or focal) only on top of (1)**, to push moving-cell accuracy, which is where all the margin comes from.
4. **Forward correction with a 1-2 parameter flip model** if cleaning is shown to discard real signal.
5. **Conditional/delta tokenizer (Δ-IRIS / iVideoGPT style)** as the root-cause fix if 1-4 fail; largest change, and Δ-IRIS warns pure deltas make dynamics harder.
6. Skip GCE/SCE, uniform label smoothing, and hard bootstrapping: they don't use the known noise location and risk the moving-cell win.
