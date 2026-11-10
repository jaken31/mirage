# Evaluating world models against copy/persistence baselines when the tokenizer is noisy

Research date 2026-11-01. Sources mostly 2015-2026; dates given per item. Project context: FSQ 512-code tokenizer, 64x64 MuJoCo arm frames, gate row 1 = top-1 next-token accuracy > persistence; model wins on the ~12% of cells whose pixels changed, loses overall because spurious tokenizer flips on unchanged cells are over half of all token changes.

## Do published video predictors / world models compare against copy-last-frame, and how do they fare?

### Takeaway
Yes, and the recurring result is that copy-last-frame beats or ties learned models on whole-frame per-frame metrics whenever most of the frame is static. The standard response in the literature was not to drop the baseline but to report scores restricted to moving regions, or bucketed by amount of motion, where models then do beat copy. Meteorology formalises the same thing as a skill score relative to persistence.

### Cited Findings
- Mathieu et al. (2015, ICLR 2016), UCF101: because "it is trivial to predict pixel values in static areas", they compute PSNR/SSIM only where EpicFlow optical flow exceeds a fixed threshold. In moving areas, frame-1 PSNR: "Last input" (copy) 28.6, L2 multi-scale 27.6, L1 28.7, best Adv+GDL fine-tuned 32.0; frame 2: copy 26.3 vs best 28.9. Their L2 multi-scale model "does not outperform simple frame copy in the moving areas." Whole-image scores pushed to an appendix. — [Mathieu et al., arXiv 1511.05440](https://arxiv.org/abs/1511.05440)
- Villegas et al. (MCnet, ICLR 2017): "A simple copy/paste operation of the last observed frame ensures very high prediction scores in videos where very small motion occur... causes the simple copy/paste baseline to outperform MCnet in the overall performance on UCF101." They then (a) evaluate on motion areas via DeepFlow, following Mathieu, where MCnet beats copy only for the first few steps, and (b) bucket test videos into motion deciles: low-motion deciles show the net is not just copying, high-motion deciles show it beating copy. — [arXiv 1706.08033](https://arxiv.org/abs/1706.08033)
- Villegas et al. (NeurIPS 2019, large stochastic RNNs): on Human3.6M, where the moving person is a small area, "a baseline where we simply copy the last observed frame through time... significantly outperforms all models... per-frame evaluations are not reliable when a large portion of a video does not move." They used FVD for model selection and kept copy-last-frame as a reference curve on robot-push and Human3.6M plots. — [arXiv 1911.01655](https://arxiv.org/abs/1911.01655) (PDF supplementary, checked directly)
- ContextVP (2017/ECCV 2018), UCF-101: copy-last-frame "outperforms almost all state of the art models except Deep Voxel Flow." — [arXiv 1710.08518](https://arxiv.org/pdf/1710.08518)
- PredNet (2016): on Human3.6M copy-last-frame scored PSNR 32.0, above a feedforward multi-scale model (26.7) but below PredNet (38.9). — [arXiv 1605.08104](https://arxiv.org/pdf/1605.08104)
- 2020 photo-realistic prediction paper uses the copy baseline's LPIPS as a measure of how much a sequence moves and reports its gains as a function of that. — [arXiv 2003.08635](https://arxiv.org/pdf/2003.08635)
- vp-suite (video prediction benchmark toolkit) ships a CopyLastFrame model as a default baseline. — [vp-suite docs](https://ais-bonn.github.io/vp-suite/vp_suite.models.copy_last_frame.html)
- Review of deep video prediction (2020): pixel metrics "are easily fooled by learning to match the background"; recommends evaluating only on dynamic parts (citing Mathieu). — [Oprea et al., arXiv 2004.05214](https://arxiv.org/pdf/2004.05214)
- Weather/solar nowcasting: skill score = 1 - error_model / error_persistence; 0 means no better than persistence; persistence is hard to beat when the signal is mostly flat over short horizons. — [arXiv 2112.13974](https://arxiv.org/pdf/2112.13974); IMERG nowcasting reports MSE 60% below persistence — [arXiv 2108.06868](https://arxiv.org/pdf/2108.06868)
- Verification standard (WMO/JWGFVR page): generic skill score = (score_fc - score_ref) / (score_perfect - score_ref), reference "typically persistence or climatology"; "can be unstable for small sample sizes." — [JWGFVR forecast verification](https://jwgfvr.github.io/forecastverification)
- Token-based world models (IRIS, Delta-IRIS, Genie, iVideoGPT, TECO) do not, in the sources found, gate on token accuracy vs a copy baseline. They report agent return (Atari/Crafter), or FVD/PSNR/SSIM/LPIPS on decoded frames. Delta-IRIS reports Crafter tasks solved (17/22 at 10M frames). — [Delta-IRIS, arXiv 2406.19320](https://arxiv.org/abs/2406.19320); iVideoGPT reports FVD/PSNR/SSIM/LPIPS on BAIR/RoboNet — [arXiv 2405.15223](https://arxiv.org/pdf/2405.15223)

### Inferences
- The project's situation (model wins on moving cells, loses overall to copy) is the same pattern the pixel literature hit in 2015-2019. Their fix kept the copy baseline but changed the region being scored. Nobody found treats "lose to copy on whole frame" as disqualifying when the model wins on moving regions.
- Difference from the literature: there, copy wins on static pixels because they truly do not change. Here, copy loses points on static cells too (when the tokenizer flips), but the model loses more there. That means the model is *worse than copy on quiet cells*, which is a real defect (it predicts flips or changes that the target does not have, or fails to copy), not just a metric artefact. Worth separating the two cases before restating the gate.

### Gaps
- No world-model paper found that reports top-1 token accuracy against a copy-previous-tokens baseline. That gate appears to be project-specific; there is no published number to compare to.
- Moving MNIST / KTH / BAIR copy-baseline numbers were not pulled in a citable form.

## Metrics robust to tokenizer noise

### Takeaway
Options in use: (1) region-restricted pixel metrics on decoded frames (flow/motion mask), (2) persistence-relative skill scores, (3) event-only categorical scores (CSI-style, which drop "correct negatives"), (4) FVD/distributional metrics for long rollouts, and (5) per-horizon curves. Decoded-pixel metrics absorb tokenizer flips only if a flipped code decodes to nearly the same pixels; flip-cleaned targets are the most direct fix but need a definition of "spurious" that does not peek at the model.

### Cited Findings
- Motion-masked pixel metrics: flow-threshold mask (Mathieu 2015, Villegas 2017) — [1511.05440](https://arxiv.org/abs/1511.05440), [1706.08033](https://arxiv.org/abs/1706.08033); foreground/background PSNR computed separately via segmentation masks — [arXiv 2203.10528](https://arxiv.org/pdf/2203.10528); 2026 action-conditioned world-model benchmark uses a Masked-MSE that weights dynamic foreground heavily and static background minimally — [ACWM-Phys, arXiv 2605.08567](https://arxiv.org/pdf/2605.08567). No single standard masking protocol exists.
- CSI/threat score = hits / (hits + misses + false alarms) "can be thought of as the accuracy when correct negatives have been removed from consideration"; depends on event frequency (poorer for rarer events). Equitable threat score adjusts for hits by chance. Heidke skill score is accuracy corrected for chance, but "random chance is usually not the best forecast to compare to." Bias score = (hits+false alarms)/(hits+misses) detects over/under-forecasting. — [JWGFVR](https://jwgfvr.github.io/forecastverification)
- Fractions skill score compares event fractions in spatial windows (neighbourhood tolerance), with a stated "useful" threshold 0.5 + f/2 (f = observed event fraction). — [JWGFVR](https://jwgfvr.github.io/forecastverification)
- Per-frame metrics averaged over frames are "by construction blind" to temporal flicker; a 2026 tokenizer paper measures flicker separately in static vs moving regions (DAVIS) and finds token masking roughly doubles static-region flicker while the frozen backbone remains the main source. — [arXiv 2606.06158](https://arxiv.org/pdf/2606.06158) (single recent preprint, appendix only seen via search snippet; not verified)
- Per-frame 2D VQ tokenizers give "reasonable image quality but fall short of frame consistency" (flicker); MAGVIT's 3D tokenizer was built for this. — [MAGVIT, arXiv 2212.05199](https://arxiv.org/pdf/2212.05199). VidTok (2024) argues per-frame tokenization yields temporally inconsistent tokens and uses FSQ. — [arXiv 2412.13061](https://arxiv.org/html/2412.13061v1)
- Delta-IRIS tokenizer encodes the change between frames conditioned on the past, rather than each frame alone; removing its continuous tokens produced glitchy imagined trajectories. — [arXiv 2406.19320](https://arxiv.org/abs/2406.19320). (Relevant as a design that removes the "re-encode an unchanged cell and get a different code" failure.)
- DIAMOND argues compression to discrete tokens "may ignore visual details that are important for RL" and moves dynamics to pixel space with diffusion; Atari 100k HNS 1.46. — [arXiv 2405.12399](https://arxiv.org/abs/2405.12399)
- FVD is sensitive to high-frequency errors: TECO notes Latent FDM "has consistent predictions, but high FVD most likely due to FVD being sensitive to high frequency errors." — [TECO, arXiv 2210.02396](https://arxiv.org/abs/2210.02396)
- Label noise in test sets: a correct prediction can be scored wrong when the label is the error; on corrected test labels model rankings flip (NASNet vs ResNet-18), and small increases in mislabeled test examples can change which model looks best. — [Northcutt et al., Pervasive Label Errors in Test Sets, arXiv 2103.14749](https://arxiv.org/abs/2103.14749); Bayes error as the irreducible floor and feasibility check — [arXiv 2010.08410](https://arxiv.org/abs/2010.08410)

### Inferences
- Cross-entropy/NLL on the true next tokens is more forgiving of unpredictable flips than top-1 accuracy: if flips are random, the best a model can do is spread a little mass on alternative codes, and NLL rewards that calibration while accuracy gives zero credit. Persistence as a baseline has no probabilities, so to compare NLL to persistence one needs a smoothed persistence (e.g. P(copy)=1-ε, rest uniform or empirical flip distribution, ε fit on train). This is a fair "copy" baseline in probabilistic form. (Inference; no source found that does exactly this for token world models.)
- Flip-cleaned targets are equivalent to test-label correction in the Northcutt sense. Ranking can change; the risk is defining "spurious" using a criterion correlated with the model's own behavior. A pixel-based criterion (pixel diff below threshold → flip is spurious → target = previous token) is independent of the model and is analogous to the optical-flow mask.
- Decoded-pixel scoring is the closest to published practice (PSNR/LPIPS in moving mask, plus copy-last-frame reference curve). It absorbs flips only to the degree that the flipped code decodes to similar pixels; measure that (decoded pixel error of tokenizer flips on unchanged cells) before relying on it.

### Gaps
- No source found that quantifies FSQ-specific code flicker on static content.
- Did not find a paper scoring token-world-models with NLL against a probabilistic persistence baseline.

## Pitfalls of scoring only changed regions; combining changed/unchanged fairly

### Takeaway
Changed-only scoring is exactly CSI-style "drop correct negatives" and inherits its known failure: it rewards over-forecasting change (a model that predicts change everywhere scores well on changed cells). Standard defence is to report a complement that catches false alarms (bias score, false-alarm rate on quiet cells) or to use a skill score that covers both strata against the same reference.

### Cited Findings
- CSI ignores correct negatives and depends on event frequency; bias score exists separately to detect over-forecasting. HK/Peirce score (POD - POFD) is independent of event frequency but for rare events is "unduly weighted toward" POD; HSS and HK can reward conservative forecasting per the Gerrity-score note. — [JWGFVR](https://jwgfvr.github.io/forecastverification)
- Equitable threat score corrects hits by random chance but "does not distinguish the source of forecast error" (misses and false alarms weighted equally). — [JWGFVR](https://jwgfvr.github.io/forecastverification)
- Villegas 2017's motion-decile breakdown is a published way to show both: low-motion bins show the model is not worse than copy, high-motion bins show where it wins. — [arXiv 1706.08033](https://arxiv.org/abs/1706.08033)
- Separate foreground/background PSNR reporting (rather than one combined number). — [arXiv 2203.10528](https://arxiv.org/pdf/2203.10528)

### Inferences
- A defensible restated gate has two parts, both vs persistence: (a) on pixel-changed cells, model accuracy > persistence (the thing a world model exists for); (b) on pixel-unchanged cells, model's rate of predicting a token different from the previous token ("false alarm") must not exceed some bound, or accuracy on flip-cleaned quiet targets must be ≥ persistence (which is 100% there by construction, so this becomes "model copies when nothing moved").
- If one number is needed: a stratified skill score, e.g. SS = 1 - (w_c·err_c + w_q·err_q) / (w_c·err_c,persist + w_q·err_q,persist), with weights fixed in advance (equal weights, or by pixel-change area). Equal weighting avoids the 88% quiet mass swamping the result; fixed weights avoid gaming. The meteorology note about instability with small samples applies to the changed stratum if it is small per batch.
- Note that on flip-cleaned quiet cells persistence error is 0, so a pure ratio skill score blows up there; use a difference or the two-part gate instead.

### Gaps
- No ML source found that gives a standard combined changed/unchanged score for token prediction.

## Tokenizer error propagation into long-horizon rollouts; measuring coherence horizon

### Takeaway
Published practice is per-horizon curves (PSNR/SSIM/LPIPS vs step, FVD on long clips, CSI vs lead time) plus a horizon where the metric falls to baseline or near zero. Single-step teacher-forced numbers say little about drift. Noise-augmented conditioning is the common mitigation, and it is directly relevant: tokenizer flips in the context are a form of input corruption.

### Cited Findings
- TECO (ICML 2023): standard metrics "do not measure long-term consistency"; they plot PSNR/SSIM/LPIPS on 150 predicted frames conditioned on 144 frames; baselines show "per-frame metrics sharply dropping as the future prediction horizon increases." DMLab 64x64, 300-frame trajectories (closest public analogue to this project's scale). — [arXiv 2210.02396](https://arxiv.org/abs/2210.02396), [repo](https://github.com/wilson1yan/teco)
- GameNGen (2024, ICLR 2025): single-frame teacher-forced PSNR 29.4 ("comparable to lossy JPEG"), but autoregressive quality "degrades fast after 20-30 steps" without fixes; training with Gaussian noise added to encoded context frames was "critical for preserving visual stability over long time periods." — [arXiv 2408.14837](https://arxiv.org/abs/2408.14837)
- Radar nowcasting reports CSI at ≥45 dBZ per lead time and a "CSI retention" normalised by the first-step value; most DL baselines decay to near-zero CSI by 48-72 min. — [IMPA-Net, arXiv 2604.24224](https://arxiv.org/pdf/2604.24224) (2026 preprint)
- PAN (2025): consistency falls with horizon for all models but at widely varying rates; errors compound because predictions are fed back as inputs. — [arXiv 2511.09057](https://arxiv.org/pdf/2511.09057)
- Mask2Real-WM (2026): rollout error grows fastest over the first ~10 steps then plateaus. — [arXiv 2607.04546](https://arxiv.org/pdf/2607.04546)
- Copy-last-frame "through time" curves are the standard reference on rollout plots. — [arXiv 1911.01655](https://arxiv.org/abs/1911.01655)

### Inferences
- A coherence horizon for this project: the step k at which the model's moving-region score (decoded pixels or flip-cleaned tokens) drops to the copy-first-frame baseline's score. That is the direct analogue of "skill > persistence at lead time k" in nowcasting.
- Spurious flips hurt rollouts in two ways: as noisy targets (scoring) and as noisy context (the model sees flipped tokens in teacher-forced training but its own clean-ish outputs at rollout, a train/test mismatch). GameNGen's context-noise trick suggests training with flips in context is not necessarily harmful and may help robustness. (Inference.)

### Gaps
- No source found that isolates the tokenizer's own contribution to rollout drift (e.g. rolling out with ground-truth-encoded vs model tokens).

## Work quantifying tokenizer temporal noise as a ceiling on dynamics accuracy

### Takeaway
Nothing found that states this ceiling explicitly for token world models. The closest are: label-noise/Bayes-error literature (noise caps measurable accuracy and can flip rankings), tokenizer flicker measurements (MAGVIT, VidTok, a 2026 flicker appendix), and tokenizer designs that sidestep it (Delta-IRIS change-conditioned tokens, 3D tokenizers, DIAMOND's pixel-space dynamics).

### Cited Findings
- Label noise acts as a hard ceiling on measured accuracy; Bayes error is the irreducible floor. — [arXiv 2010.08410](https://arxiv.org/abs/2010.08410); [Voxel51 glossary](https://voxel51.com/glossary/label-noise) (secondary)
- Test-label error alters model rankings. — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749)
- Treating all disagreement as irreducible overestimates the ceiling, because part of it is noise that is actually predictable or removable; the ceiling estimate depends strongly on the estimator. — [arXiv 2608.15619](https://arxiv.org/pdf/2608.15619) (emotion labels, 2026; analogy only)
- Per-frame VQ flicker and FSQ adoption for stability. — [MAGVIT](https://arxiv.org/pdf/2212.05199), [VidTok](https://arxiv.org/html/2412.13061v1)
- iVideoGPT reports tokenizer reconstruction quality as its own research question (RQ7) separate from prediction quality. — [arXiv 2405.15223](https://arxiv.org/pdf/2405.15223)

### Inferences
- A simple ceiling for this project: if flips on unchanged cells are i.i.d. with rate f, and unpredictable, then max achievable top-1 accuracy on quiet cells is 1 - f, and persistence also gets ≈ 1 - f there. So on quiet cells the best the model can do is *tie* persistence; any loss there is model error, not tokenizer noise. The gate fails because the model loses on quiet cells, which a ceiling argument alone does not excuse.
- The flip rate is measurable directly: encode the same static frame (or identical-pixel consecutive frames) and count code changes. If flips are deterministic functions of tiny pixel differences (e.g. rendering noise near FSQ bin edges), they are partly predictable, so "unpredictable noise" should be tested, not assumed (the 2026 emotion-label caveat in ML form).
- Reporting the tokenizer-only ceiling next to the gate (like iVideoGPT separating reconstruction from prediction) would make the gate interpretable without changing it.

### Gaps
- No paper found that measures token-flip rate on static content and relates it to dynamics accuracy. The ceiling argument above is derived, not sourced.
- Genie's ΔtPSNR controllability metric definition could not be retrieved in this pass.
