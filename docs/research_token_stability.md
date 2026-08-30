# Research: token stability, normalisation, and rollout length

**Doc class: Live.** Measured and derived mirage values are cited by name and
`NUM-` id, never restated. External values are stated inline with the paper
section or the repo file and line that owns them, because those numbers are not
in the register and never will be.

**What this is.** A survey of primary sources on four questions raised by the
Phase 1 token-persistence measurement. It is dated 2026-08-29 and it is a
*survey*, not a decision. The decisions it feeds are at the end, ranked and
costed. Nothing here has been run on mirage except the two probes in section (a),
which are reproduced inline so they can be re-run.

**Evidence rules followed here.** Claims about an external paper carry the
section. Claims about an external implementation carry the file and line in a
named repository at a named branch. Claims I could not verify are written as
unverified. Search-engine summaries and blog posts were not accepted as authority
for any technical claim; where one is used it is marked *secondary* in the source
table.

---

## Status as of 2026-08-30: options 1 and 2 have been run

This survey is kept as written, dated 2026-08-29. What follows is what happened
when its first two recommendations were executed the next day, `runs.jsonl` r46.

**Option 1 ran, and did not falsify the line of work.** The spurious-flip rate
was measured by `bench/token_stability_probe.py` over 12 held-out episodes and
460,032 cell-transitions: **8.86%** on the R1 checkpoint, **18.75%** on R2, with
**53.21%** and **71.06%** of all flips spurious. This document named a rate near
zero as the falsifier. It did not happen, and attention makes it worse.

**Option 2 ran, as rung `r1c`.** Channel-only encoder normalisation drives the
spurious-flip rate to **exactly 0 of 396,013** quiet-field transitions, so the
mechanism this survey identified is not merely real but is the *entire* cause.
The cost is the part no source predicted: **Q-2 token entropy falls 74.1% ->
54.6%**, failing the 70% bar, for only 0.282 dB of Q-1. **Phase 2 therefore stayed
on R1** - see `world_model_architecture.md`, "Phase 2 inherits R1", and
`handoff_tokenizer_decision.md` for the reasoning.

**The companion probe in option 1 is REFUTED.** This document proposes that
encoding a frame and the same frame shifted one pixel "separates the aliasing
mechanism from the global-coupling mechanism". **It does not.** A one-pixel shift
moves the input to *every* receptive field in the frame, so both mechanisms fire
together and the measurement cannot attribute the change to either. The
separating measurement is the one option 1 already specifies - conditioning on
whether the cell's own receptive field changed - and that is what was run.

**The receptive-field discrepancy below is confirmed, and it has a subtlety this
survey did not have the measurement to see.** Both directions are correct, under
different readings of "receptive field":

- The **conv** field is 15x15, as this document derives. Flat share over 15x15 is
  **26.33%** against **20.43%** over 22x22, so on that reading the true ceiling is
  **below** `NUM-TOK-Q2CEIL`, exactly as stated below.
- But the derivation's premise is that two cells with identical receptive fields
  must share a code, and under `GroupNorm` a cell's **effective** field is the
  whole frame. The flat share over the whole frame is **0.00%**, so for the
  encoder actually in use the derivation is **vacuous** and the ceiling is 100%.

So `NUM-TOK-Q2CEIL` is understated for a locality-respecting encoder such as
`r1c`, and meaningless for `R1` and `R2`. Neither reading threatens Q-2 and no
passed gate moves, which is why this is still recorded rather than fixed.
`bench/patch_probe.py:60` still says `RF = 22`.

---

## 0. The situation this answers

Two converged 64x64 rungs exist: R1 without grid attention (`NUM-TOK-R1-60`,
`NUM-TOK-ENT-R1`, `NUM-TOK-PARAMS-R1`) and R2 with it (`NUM-TOK-R2-60`,
`NUM-TOK-ENT-R2`, `NUM-TOK-PARAMS-R2`). Both clear `NUM-BAR-Q1` and
`NUM-BAR-Q2`. Attention buys `NUM-TOK-ATTN` of quality for `NUM-TOK-ATTNPARAM`
parameters and `NUM-TOK-ATTNENT` of entropy.

Frame-to-frame token persistence, the share of the 64 grid cells where
`token[t] == token[t-1]` inside an episode, was measured over 19.2M tokens:
**86.62% for R1, 78.20% for R2**. Those two figures are not in
`canonical_numbers.md` yet, so they are stated here rather than cited; if they
are ever quoted twice they belong in the register.

An autograd probe attributed the coupling: with `GroupNorm` live, one latent
cell's gradient support is the whole 64x64 frame; with the norm's forward
neutralised to identity it is exactly 225 pixels. Section (a) reproduces that
result and extends it across every normalisation option.

The gate this all serves is **Q-3**, coherence horizon of at least 200 frames
before the F-9 validator fails, with **Q-5** (link-length drift over a 200-step
rollout) and **Q-6** (object permanence through occlusion) alongside it. The
dynamics model is a decoder-only transformer over interleaved
`[action, 64 frame tokens]` sequences.

---

## (a) Normalisation that preserves spatial locality in conv encoders

### a.1 What each layer reduces over

Group Normalization's own paper states the reduction sets algebraically for four
layers at once (Wu and He, *Group Normalization*, arXiv 1803.08494, section 3,
equations 3 to 7):

- **BatchNorm**: "for each channel, BN computes mu and sigma along the (N, H, W)
  axes"
- **LayerNorm**: "LN computes mu and sigma along the (C, H, W) axes for each
  sample"
- **InstanceNorm**: "IN computes mu and sigma along the (H, W) axes for each
  sample and each channel"
- **GroupNorm**: "GN computes mu and sigma along the (H, W) axes and along a
  group of C/G channels"

All four reduce over `H` and `W`. The PyTorch 2.9.1 docstrings agree, in the
installed copy at
`C:\Users\nguye\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages\torch\nn\modules`:

| Layer | Docstring wording | File and line |
|---|---|---|
| `nn.GroupNorm` | "The mean and standard-deviation are calculated separately over each group." Also notes `nn.GroupNorm(1, C)` is "equivalent with LayerNorm" and `nn.GroupNorm(C, C)` is "equivalent with InstanceNorm" | `normalization.py:240-284` |
| `nn.LayerNorm` | "calculated over the last `D` dimensions, where `D` is the dimension of `normalized_shape`". Its image example is `nn.LayerNorm([C, H, W])`, i.e. channel **and** spatial | `normalization.py:106-181` |
| `nn.BatchNorm2d` | "calculated per-dimension over the mini-batches"; "Batch Normalization is done over the `C` dimension, computing statistics on `(N, H, W)` slices" | `batchnorm.py:380-419` |
| `nn.InstanceNorm2d` | "calculated per-dimension separately for each object in a mini-batch"; `track_running_stats` defaults to `False` | `instancenorm.py:241-296` |
| `nn.RMSNorm` | "The RMS is taken over the last `D` dimensions, where `D` is the dimension of `normalized_shape`" | `normalization.py:333-372` |

The load-bearing consequence: **`nn.LayerNorm` is not automatically local.** On a
`(B, C, H, W)` tensor, `nn.LayerNorm([C, H, W])` reduces over channels and both
spatial axes, which is *more* global than `nn.GroupNorm(8, C)`, not less. To get
per-position channel-only normalisation you must move the channel axis last:

```python
# per-position, channel-only LayerNorm on (B, C, H, W)
ln = nn.LayerNorm(C)                       # normalized_shape = C, not [C, H, W]
y  = ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# same shape, RMS variant (no mean subtraction)
rn = nn.RMSNorm(C)
y  = rn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# EDM2 "pixel norm": divide by the RMS over channels, no learned affine
y = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-4)
```

### a.2 The measured consequence, for mirage's exact encoder

The claim above is arithmetic. This is the measurement. The script builds
mirage's encoder shape (`mirage/fsq.py:217-222` and `290-322`: three
`Conv2d(k=3, s=2, p=1) -> norm -> SiLU` stages at widths 3 -> 64 -> 128 -> 256,
then `Conv2d(256, 3, 1)`), swaps only the normalisation, and counts the input
pixels with non-zero gradient for the centre latent cell.

```python
import torch, torch.nn as nn
def make(norm):
    ch, L = [3, 64, 128, 256], []
    for a, b in zip(ch[:-1], ch[1:]):
        L += [nn.Conv2d(a, b, 3, 2, 1), norm(b), nn.SiLU()]
    return nn.Sequential(*L, nn.Conv2d(256, 3, 1))
def support(m):
    torch.manual_seed(0); x = torch.randn(1, 3, 64, 64, requires_grad=True)
    m(x)[0, :, 4, 4].sum().backward()
    g = x.grad.abs().sum(dim=(0, 1)); nz = (g > 0).nonzero()
    return int((g > 0).sum()), int(nz[:, 0].max() - nz[:, 0].min() + 1)
```

Run 2026-08-29 on torch 2.9.1+cu130, CPU, module in `train()` unless noted:

| Normalisation | Pixels in support | Bounding box |
|---|---|---|
| `nn.GroupNorm(8, C)` (mirage today) | 4096 | 64x64 |
| `nn.GroupNorm(1, C)`, i.e. LayerNorm over `[C,H,W]` | 4096 | 64x64 |
| `nn.BatchNorm2d(C)`, `train()` | 4096 | 64x64 |
| `nn.InstanceNorm2d(C, affine=True)` | 4096 | 64x64 |
| **`nn.LayerNorm(C)` on channels-last** | **225** | **15x15** |
| **`nn.RMSNorm(C)` on channels-last** | **225** | **15x15** |
| **RMS over channels (EDM2 pixel norm)** | **225** | **15x15** |
| **Weight Standardization, no activation norm** | **225** | **15x15** |
| **`weight_norm` parametrization, no activation norm** | **225** | **15x15** |
| **`nn.Identity()` (no norm)** | **225** | **15x15** |
| `nn.BatchNorm2d(C)`, `eval()` | 225 | 15x15 |

Three findings worth keeping.

1. **Every activation normalisation that touches `H` or `W` gives full-frame
   support, and that includes all four of the standard ones.** There is no
   "milder" spatial norm. `GroupNorm(1)` is not a fix, it is strictly worse.
2. **Channel-only normalisation, in any of its three spellings, leaves the
   support at exactly the arithmetic receptive field.** So does doing nothing.
3. **`BatchNorm2d` is the odd one out: global in `train()`, local in `eval()`.**
   The deployed encoder would be local while the trained one was not. NVAE
   (Vahdat and Kautz, arXiv 2007.03898, section 3, "Batch Normalization") reports
   the same asymmetry from the other side: "we observed that the negative impact
   of BN is during evaluation, not training. Because of using running statistics
   in BN, the output of each BN layer can be slightly shifted during evaluation".
   That train/eval split is a reason to not reach for BN here, independent of
   locality.

`GridAttention` (`mirage/fsq.py:240-288`) is global by construction and this
probe says nothing new about it. It is a second, separate source of coupling on
top of the norm, which is consistent with R2 persisting less than R1.

### a.3 The one paper that names this consequence, and measures removing it

Karras et al., *Analyzing and Improving the Training Dynamics of Diffusion
Models* (EDM2, arXiv 2312.02696), appendix B.6 "Dangers of global normalization",
states the mechanism almost exactly as mirage's autograd probe found it:

> global normalization that operates across the entire image should be used
> cautiously. It is firmly at odds with the desire for the model to behave
> consistently across geometric transformations or when synthesizing objects in
> different contexts. Such consistency is easiest to achieve if the internal
> representations of the image contents are capable of being as localized as they
> need to be, but global normalization entangles the representations of every
> part of the image by eliminating the first-order statistics across the image.
> Notably, while attention allows the representations to communicate with each
> other in a way that best fits the task, global normalization forces
> communication to occur, with no way for individual features to avoid it.

Their Config F is "Remove group normalizations": all GroupNorm layers deleted and
replaced with "1/4 as many pixel normalization layers" on the encoder main path
(section 2.4). Table 1 gives the effect on ImageNet-512 FID without guidance:
Config E 3.02 -> **Config F 2.71** -> Config G 2.56. Appendix B.6 adds "the
network trains successfully without any normalization layers", with pixel norm a
"small benefit" on top.

The reference implementation confirms the shape:
`NVlabs/edm2` `training/networks_edm2.py` has no `GroupNorm` anywhere. Line 153
is `x = normalize(x, dim=1)  # pixel norm`, and `normalize` at lines 20-25
divides by the vector norm along the given dim, so `dim=1` on `(N, C, H, W)` is
channel-only at each position. Line 174 applies the same to attention q/k. All
convolutions are `MPConv` (lines 91-108) with forced weight normalisation.

Two caveats before this is treated as a recipe:

- EDM2 removes GroupNorm only **after** Configs D and E have taken control of
  activation, weight, and update magnitudes. The paper is explicit that this is
  the ordering ("With activation, weight, and update magnitudes under control, we
  are now ready to remove..."). Lifting Config F alone is not what they tested.
- EDM2 is a diffusion UNet, not a discrete tokenizer, and FID is not PSNR. The
  mechanism transfers; the number does not.

Supporting, older: Brock et al., *High-Performance Large-Scale Image Recognition
Without Normalization* (NFNets, arXiv 2102.06171), section 3, removes BatchNorm
entirely using Scaled Weight Standardization plus Adaptive Gradient Clipping and
matches or beats normalised baselines on ImageNet. EDM2 B.6 cites exactly this:
"Brock et al. show that normalization is not necessary for obtaining
high-quality results in image classification. We see no reason why it should be
necessary or even beneficial in diffusion models, either."

Also supporting, for the channel-only spelling specifically: ConvNeXt (Liu et
al., arXiv 2201.03545, section 2, "Substituting BN with LN") replaces BatchNorm
with LayerNorm throughout a pure convnet and reports 81.5% ImageNet top-1,
"slightly better" than the BN version. The official implementation
`facebookresearch/ConvNeXt` `models/convnext.py` shows the axes: the block
forward at lines 37-50 does `x.permute(0,2,3,1)` then `self.norm(x)` then
permutes back, and the `channels_first` branch at lines 138-143 computes
`x.mean(1, keepdim=True)`. **Both paths normalise over channels only, at each
spatial position.** ConvNeXt is the existence proof that a convnet trains fine on
channel-only normalisation.

### a.4 What production VQ / FSQ implementations actually use

Read from source, not from papers:

| System | Encoder normalisation | Evidence |
|---|---|---|
| CompVis **taming-transformers** (VQGAN) | `GroupNorm(num_groups=32, eps=1e-6, affine=True)`, used as `norm1`/`norm2` in every `ResnetBlock` and as `norm_out` | `taming/modules/diffusionmodules/model.py:34-35`, `88`, `96`, `431` |
| google-research **MaskGIT** VQGAN | Flax `nn.GroupNorm`, selected by `norm_type`; the shipped config sets it | `maskgit/nets/layers.py:62-63`; `maskgit/configs/vqgan_config.py:64` is `config.vqvae.norm_type = "GN"` |
| **FSQ** paper (arXiv 2309.15505) | **Inherits MaskGIT's, i.e. GroupNorm.** Appendix A.1: "We refer to the MaskGIT GitHub and the UViM GitHub for the model code used in this paper." Appendix A.4.3: "we base all experiments on the `vqgan_config.py` and `maskgit_class_cond_config.py` configs from the MaskGIT GitHub repo" | FSQ A.1, A.4.3 |
| lucidrains **vector-quantize-pytorch** FSQ | **None.** The module is a quantizer only: `project_in`/`project_out` are `nn.Linear` or `nn.Identity`, and there is no normalisation layer in the file | `vector_quantize_pytorch/finite_scalar_quantization.py:117-118` |
| **Open-MAGVIT2** (TencentARC) | `nn.GroupNorm(32, ..., eps=1e-6)` in every ResBlock and at `norm_out`; `AdaptiveGroupNorm` in the decoder | `src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py:25-26`, `101`, `251-254` |
| **VideoGPT** | `BatchNorm3d`. Notably the encoder's downsampling convolutions have **no** norm at all (`F.relu(conv(h))`), with one `BatchNorm3d + ReLU` at the end | `videogpt/vqvae.py:110-117`, `229-236` |
| **TATS** | **Synced BatchNorm, deliberately replacing GroupNorm.** Appendix 0.A.1: "blob-shaped artifacts often appear in the reconstruction ... can be attributed to the normalization layers. This is especially pronounced in the training video VQGAN due to the small batch sizes. We use Synced Batch Normalization as a replace of Group Normalization used in original VQGAN" | arXiv 2204.03638, appendix 0.A.1 |
| **NVIDIA Cosmos Tokenizer** | Paper says LayerNorm, code says `GroupNorm(num_groups=1)` applied per frame. Section 3 (tokenizer architecture): "We leverage Layer Normalization (LayerNorm) instead of Group Normalization (GroupNorm), which prevents large magnitudes from appearing in specific regions of the latent space or reconstructed outputs" | arXiv 2501.03575; `NVIDIA/Cosmos-Tokenizer` `cosmos_tokenizer/modules/utils.py:67-84` |

The Cosmos entry needs care, because it is the single closest thing in the
literature to "a production tokenizer moved off GroupNorm" and **it does not do
what mirage would need**. `CausalNormalize` is
`torch.nn.GroupNorm(num_groups=num_groups, ...)`, and its forward is:

```python
# cosmos_tokenizer/modules/utils.py:78-84
def forward(self, x):
    # if num_groups !=1, we apply a spatio-temporal groupnorm for backward compatibility purpose.
    # All new models should use num_groups=1, otherwise causality is not guaranteed.
    if self.num_groups == 1:
        x, batch_size = time2batch(x)
        return batch2time(self.norm(x), batch_size)
    return self.norm(x)
```

So Cosmos's "LayerNorm" reduces over `(C, H, W)` **within one frame**. The stated
reason for `num_groups=1` in the code comment is temporal **causality**, not
spatial locality, and the stated reason in the paper is magnitude blow-up in
localised regions, citing StyleGAN2. By the probe in a.2 this is 4096-pixel
support, the same as mirage has now. **Copying Cosmos would not change mirage's
locality at all.** This is the kind of claim that would have propagated wrongly
if it had been taken from the paper's prose alone.

### a.5 Does anyone discuss the locality consequence *for a tokenizer*?

**No, as far as this survey found.** EDM2 B.6 makes the argument for a diffusion
UNet. StyleGAN2 (Karras et al., arXiv 1912.04958) and Alias-Free GAN are the
prior art it cites, both generators. TATS and Cosmos both move off GroupNorm but
for artifact and causality reasons, and neither reports a locality measurement.
No VQ, LFQ, or FSQ tokenizer paper found here measures the gradient support of a
token, or connects normalisation choice to token stability.

### a.6 Training a sub-1M-parameter conv autoencoder with no normalisation

**No primary source found that ablates this directly at this scale.** The
adjacent evidence:

- NFNets (2102.06171) shows norm-free works at ImageNet scale but pairs it with
  Scaled Weight Standardization *and* Adaptive Gradient Clipping, and section 3
  notes plain NF-ResNets "are not stable at large batch sizes and do not match
  the performance of EfficientNets" without AGC. That is a warning about large
  batch and large learning rate, neither of which describes mirage.
- EDM2 B.6 says flatly "the network trains successfully without any normalization
  layers", but only after magnitude control is in place.
- NVAE section 3 records that IAF-VAE "relies on regular residual networks for
  both top-down and bottom-up models without any batch normalization, and it has
  been examined on small images only" (Kingma et al., arXiv 1606.04934). So a
  norm-free conv VAE at small image size exists in the literature; its
  normalisation was not the subject of the paper.

For mirage specifically the risk profile is unusually mild: 745k parameters
(`NUM-TOK-PARAMS-R1`), fp32, plain MSE, no GAN loss, no perceptual loss, no
discriminator, batch 128, cosine schedule. Most of what normalisation is there to
rescue in the cited work (GAN instability, tiny batches, very deep stacks) is
absent. That is an argument, not evidence, and the argument is cheap to settle
with one rung (`NUM-TOK-EPOCH64` x 60).

---

## (b) Temporal consistency and equivariance regularisers for video tokenizers

### b.1 Explicit penalties on token change between adjacent frames

**Nobody found does this.** Searches across MAGVIT, MAGVIT-v2, VideoGPT, TATS,
Open-MAGVIT2, Cosmos, OmniTokenizer, and LARP turned up no loss term of the form
"penalise `token[t] != token[t-1]`", and no continuous relaxation of it either.
The word "flicker" appears in MAGVIT-v2 and iVideoGPT only as a *motivation*, and
neither paper defines a metric for it. Consequently there is **no published
reconstruction-quality cost for such a penalty**, because no one has paid it.

The closest published objects are all in other domains: temporal flow-warping
losses in video style transfer and video restoration, which penalise pixel
differences between optical-flow-warped adjacent outputs. Those are not token
penalties and they need optical flow, which mirage does not have.

### b.2 What the field does instead: four families

**1. Encode a temporal window jointly.**

- MAGVIT-v2 (arXiv 2310.05737, section 3) motivates its whole design with a
  flicker complaint about its predecessor: "MAGVIT struggles to tokenize images
  and often results in noticeable flickering in longer videos." Its fix is a
  temporally causal 3D CNN, "padding kt-1 frames before the input and nothing
  after", so "the first frame is always independent of other frames".
- Cosmos (arXiv 2501.03575, section on video tokenization) uses the same causal
  posture: "The token computation for the current frames is not based on future
  observation", plus a 2-level wavelet transform before the encoder stages.
- VideoGPT (arXiv 2104.10157, "Learning Latent Codes") uses 3D convolutions with
  axial attention and LayerNorm in the attention residual blocks.
- Genie (arXiv 2402.15391, section on the video tokenizer) uses an ST-transformer
  in both encoder and decoder "to incorporate temporal dynamics in the encodings,
  which improves the video generation quality. By the causal nature of the
  ST-transformer, each discrete encoding z_t contains information from all
  previously seen frames".

**2. Condition the encoder on context frames.** iVideoGPT (arXiv 2405.15223,
section 3.1) tokenizes each observation *conditioned on* context observations
rather than independently, and section 4.1 gives the only explicit flicker
attribution found in this literature:

> We highlight that MaskViT, a prior method leveraging per-frame tokenization,
> suffers from temporal inconsistency and flicker artifacts in VQGAN
> reconstructions. Our model, which employs compressive tokenization conditioned
> on consistent contextual information, improves this and significantly
> outperforms MaskViT.

Note what this is and is not. It is a qualitative attribution plus an aggregate
FVD comparison against a differently-architected baseline. It is not a controlled
ablation of conditioning at fixed everything-else, and no flicker number is
reported.

**3. Fix boundary and padding effects.** TATS (arXiv 2204.03638) is the sharpest
methodological parallel to mirage's GroupNorm finding, because it is the same
shape of bug in a different layer. Section 2.2:

> the encoder is not temporally shift-equivariant and encodes x^(d:T-1)
> differently when these frames are positioned at different places

The cause is zero padding: "the zero paddings used in these models corrupts the
latent tokens and results in token sequences at the inference time that are
drastically different from those observed during training when using sliding
window. The amount of corruption depends on the temporal position of the token."
The fix is replicate padding in the temporal dimension, chosen because it "makes
decent approximation to the real frames while bringing no computational
overhead". They call the result a "time-agnostic VQGAN" and show it preserves
quality beyond the training length (their Figure 2).

The transferable lesson: **an encoder that is not equivariant produces tokens the
dynamics model never saw, and the failure shows up at rollout time, not at
reconstruction time.** TATS's reconstruction metrics were not what caught it.

**4. Anti-aliasing.** Zhang, *Making Convolutional Networks Shift-Invariant
Again* (arXiv 1904.11486), inserts a low-pass filter before every subsampling,
giving BlurPool. It also supplies the metric shape that section (d) will want:
"Classification consistency. For classification, we check how often the network
outputs the same classification, given the same image with two different shifts"
(section 4.3). That is a categorical-output stability statistic, structurally the
same object as token persistence, applied to a classifier.

Its conclusion names the untested extension: "Other directions include the
potential benefit to downstream applications, such as ... improving temporal
consistency in video models". **As of this survey that is still future work.**

The one tokenizer that took it up is MAGVIT-v2, and only partly. Section 3: "the
downsampling layer in the discriminator now utilizes 3D blur pooling (Zhang,
2019) to encourage shift invariance." **Discriminator, not encoder.** Its
ablation, Table 5(c) on UCF-101 video tokenization, gives the marginal effect:
"+ deeper model 8.90 FVD / 0.0542 LPIPS" then "+ 3D blur pooling 8.62 / 0.0537".
So roughly 3% FVD, the smallest step in a six-row ladder, from an
anti-aliasing change on the adversarial side only.

### b.3 The causal-encoder option under mirage's constraint

Mirage's stated constraint is that inference encodes one frame at a time from a
live rollout, which rules out a non-causal temporal encoder. A causal one is
usable in principle, and both MAGVIT-v2 and Cosmos are designed exactly so that
the first frame is encodable alone. Cosmos's `CausalConv3d` and Genie's
ST-transformer both make streaming encode possible with a feature cache.

Two things make this expensive for mirage rather than merely different:

- It changes the tokens-per-frame contract. MAGVIT-v2's causal scheme maps
  `1 + s*t` frames to `1 + t` latent frames, so the interleaved
  `[action, 64 tokens]` sequence layout the dynamics model assumes stops holding
  at a fixed 64.
- It requires carrying encoder state across rollout steps, which is new
  machinery in the inference path rather than a swap inside `Tokenizer`.

**Did anyone measure that joint temporal encoding gives more stable tokens?** No
direct measurement found. Genie's Table 3 is the closest and it is about
downstream FVD, not token stability; see (c).

---

## (c) Does token temporal stability actually help autoregressive rollout?

This is the question that decides whether any of the above is worth a rung. The
honest summary is: **there is suggestive evidence in both directions and no
controlled experiment isolating token stability.**

### c.1 Evidence for

**Genie, tokenizer architecture ablation (arXiv 2402.15391, section 3.4,
Table 3).** Same dynamics model and same latent action model trained on three
tokenizers at matched parameter count, patch size 10, batch 128, sequence
length 16:

| Tokenizer | Params | Memory | FVD (lower better) | delta_t PSNR (higher better) |
|---|---|---|---|---|
| ViT (spatial-only) | 230M | 0.3 GB | 114.5 | 1.39 |
| C-ViViT (spatio-temporal) | 225M | 1.6 GB | 272.7 | 1.37 |
| ST-ViViT (spatio-temporal, theirs) | 205M | 0.9 GB | **81.4** | **1.66** |

This is the cleanest "tokenizer choice changes downstream generation at fixed
dynamics model" result found. **But it does not say temporal encoding wins**:
C-ViViT is also temporal and is by far the worst, and Genie attributes that to
overfitting, not to the temporal axis. So the table supports "tokenizer choice
matters a lot" and refutes "any temporal tokenizer is better".

**Identifiable Token Correspondence (arXiv 2605.16457, ICML).** The single most
on-point paper found. Abstract:

> [token-based transformer world models] often suffer from temporal
> inconsistency in long-horizon rollouts, including object duplication,
> disappearance, and transmutation. A key reason is that most existing
> approaches treat next-frame prediction purely as a token generation problem,
> without considering the persistence of tokens across time.

Their fix is a **decoding step only**: an optimal-transport solver between the
transformer's next-token distribution and the final next-state tokens, where
"each next-frame token is explained either by copying a token from the previous
frame or by generating a new one". It "leaves the transformer architecture and
training procedure unchanged". Craftax-classic at 1M steps: return
**72.46 +/- 0.45**, score **35.60 +/- 0.92**, against a previous best of 67.42
and 27.91.

The ablation is the part to read carefully (their Table 3 region): baseline
68.55 / 27.24, "+ 3D RoPE" 71.85 / 33.94, "+ ITC" 72.46 / 35.60. **Most of the
gain is positional encoding, not the correspondence solver.** ITC's own marginal
contribution is about +0.6 return and +1.7 score.

The cost model is also stated: their transport cost caps allowed motion at
`d <= 4` in token-grid squared distance, "The cap encodes a generic prior in
visual RL: entities rarely move by more than a couple of token positions between
consecutive frames." That prior is exactly mirage's scene.

**iVideoGPT** attributes MaskViT's flicker to per-frame tokenization and beats it
(section 4.1, quoted in b.2). Confounded by architecture.

### c.2 Evidence against, and the confounds

**A better-reconstructing tokenizer is routinely a worse generative substrate.**
LARP (arXiv 2410.21264, section 1) states it as received knowledge with citations:
"although the reconstruction fidelity of a visual tokenizer sets an upper bound
on the generation fidelity of AR models, the factors that determine the gap
between them remain unclear. In fact, higher reconstruction quality has been
widely reported to sometimes lead to worse generation fidelity."

VA-VAE (arXiv 2501.01423, abstract) names the same phenomenon in the continuous
case: "while increasing the per-token feature dimension in visual tokenizers
improves reconstruction quality, it requires substantially larger diffusion
models and more training iterations to achieve comparable generation
performance." They call it the optimization dilemma.

**Higher entropy / larger vocabulary is reported as good, not bad.** MAGVIT-v2
(section 3, Figure 1): "both reconstruction and generation consistently improves
as the vocabulary size increases - a property not observed in current VQ-VAE
methods." That is direct counter-pressure on any mirage change that trades token
entropy for stability, and mirage's headroom is thin: `NUM-TOK-ENT-R1` sits above
`NUM-BAR-Q2` by about four points.

**No paper found reports a stability-versus-entropy trade curve.** So the
question "would mirage rather have 90% persistence at 71% entropy or 86.6% at
74.1%" is unanswered by the literature and can only be settled by mirage.

### c.3 The theory that predicts the sign

The compounding-error framing is standard and predicts that per-step token error
is what matters, which is *upstream* of stability rather than the same thing.
Ross, Gordon and Bagnell, *A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning* (arXiv 1011.0686), Theorem 2.1:

> Let E[l(s, pi)] = epsilon under the expert's state distribution, then
> J(pi) <= J(pi*) + T^2 * epsilon.

and the note that "this bound is tight, i.e. there exist problems such that a
policy pi with epsilon 0-1 loss ... can incur extra cost that grows
quadratically in T". They attribute the sequence-prediction version to
Kääriäinen 2006, which "behaves as Theta(T^2 epsilon) for small epsilon".

Applied to mirage: at 200 rollout steps the quadratic factor is 40,000, so the
per-token error rate is what decides the horizon. **The connection to token
stability is a hypothesis, not a theorem.** The plausible mechanism is that a
tokenizer whose tokens change only when their own receptive field changes makes
the conditional next-token distribution lower-entropy and more predictable,
lowering epsilon. The equally plausible counter-mechanism is that it lowers the
information content of each token, so the transformer needs more of them to
represent the same dynamics. Nothing found measures either.

---

## (d) Token flicker as a published tokenizer metric

**Finding: no paper found names or reports the statistic mirage measured.** That
is, the fraction of grid cells where the discrete token id at time t equals the
id at time t-1, over an episode. This appears to be unpublished.

What exists nearby, and why each is a different quantity:

| Nearest published thing | What it actually measures | Source |
|---|---|---|
| **Classification consistency** | How often a classifier gives the same argmax under two different input shifts. Categorical-output stability, but over shifts of one image, not over time | Zhang, arXiv 1904.11486, section 4.3 |
| **Static-patch pruning rate** (EVS) | Percentile threshold on `D_{p,t} = ||p_t - p_{t-1}||_1`, the L1 difference of raw **pixel** patches between consecutive frames. Pixel space, and a pruning knob rather than a reported property | arXiv 2510.14624, section 3 |
| **Time-invariant vs time-variant token split** (TivTok) | An architectural factorisation into tokens that attend to the whole clip and tokens that see one frame, with a qualitative "pixel-level persistence" visualisation. Not a rate | arXiv 2606.17590, sections 3.2 and 4.5 |
| **Token correspondence** (ITC) | A latent assignment variable per token deciding copy-versus-generate at decode time. The copy rate is a model output, not a tokenizer measurement, and no aggregate is reported | arXiv 2605.16457, section 3 |
| **Delta tokens** (DeltaTok) | Encodes the vision-foundation-model feature *difference* between consecutive frames into one continuous token. Motivated by "consecutive frames differ only in structured and typically low-dimensional ways", but the redundancy is exploited rather than quantified | arXiv 2604.04913, sections 1 and 3 |
| **Temporal-consistency losses** | Optical-flow-warped pixel differences between adjacent outputs, in style transfer and restoration | many; not tokenizers |

So the honest statement is: **the field talks about flicker constantly and
measures it never, at the token level.** MAGVIT-v2 and iVideoGPT both use
"flicker" as motivation with no metric attached. That makes mirage's 86.62% /
78.20% pair a small original contribution, and it also means there is no external
number to compare against. To make it comparable if it is ever published, the
definition needs to travel with it: *within an episode, over adjacent frame pairs,
the fraction of the H/8 x W/8 grid cells whose integer token id is unchanged;
first frames excluded; measured on held-out episodes.* Resolution-dependent,
since the grid is.

---

## What this implies for mirage

Ranked by evidence strength times expected effect, divided by cost. One rung is
`NUM-TOK-EPOCH64` x 60 epochs, about 88 minutes at 64x64.

| # | Option | Cost | Evidence |
|---|---|---|---|
| 1 | Measure the spurious-flip rate on the existing R1 checkpoint | minutes, no training | our own probe + EDM2 B.6 mechanism |
| 2 | Swap encoder `GroupNorm` for channel-only normalisation, one rung | 1 rung | strong (EDM2 measured, ConvNeXt measured) |
| 3 | ITC-style copy-biased decoding in the rollout loop | no training | direct (ITC, SOTA) but small marginal effect |
| 4 | Blur-pool the stride-2 downsamples | 1 rung | weak for encoders (MAGVIT-v2 used it in the discriminator only) |
| 5 | LARP-style AR prior loss on the tokenizer | 1 rung + new code | strong in-principle, large change |
| 6 | Explicit token-change penalty | 1 rung + new loss | none published |
| 7 | Causal 3D encoder | phase-sized | strong but breaks the 64-tokens-per-frame contract |

### 1. Measure the spurious-flip rate first. Do this before spending a rung.

**The change.** On the shipped R1 checkpoint, over held-out episodes, for each
adjacent frame pair and each grid cell, record both `token_changed` and
`patch_changed`, where `patch_changed` is whether any pixel in that cell's 15x15
arithmetic receptive field differs between the two frames; report
`P(token_changed | not patch_changed)`.

**Why first.** Everything in section (a) establishes that GroupNorm *can* make a
token depend on the whole frame. It does not establish that the trained network
*uses* that dependence. This measurement is the difference between a mechanism
and a cause, and it is the same discipline that killed `NUM-VAL-PCTL` and the
30 ms readback figure.

**Cost.** One forward pass of the encoder over `NUM-DATA-VALFRAMES` held-out
frames plus a pixel-difference pass. No training, no new checkpoint. Comparable
to an existing eval, so minutes.

**What would falsify the whole line of work.** A spurious-flip rate near zero.
That would say the ~13.4% of R1 cells that flip are flipping because their own
receptive field genuinely changed, the tokenizer is already as local as it can
usefully be, and options 2, 4 and 6 have nothing to fix. Given a 2-link arm and
3 blocks on a flat background, a large fraction of the frame is static, so this
outcome is entirely possible and would be the most valuable result of the set.

**What it risks breaking.** Nothing. It is read-only.

**A useful companion, same cost:** encode a held-out frame and the same frame
translated by one pixel, and report the token change rate. That separates the
aliasing mechanism (option 4's target) from the global-coupling mechanism
(option 2's target). The two are different causes with different fixes and
mirage currently has evidence for only one of them.

### 2. Replace encoder `GroupNorm(8)` with channel-only normalisation

**The change.** In `_stage` (`mirage/fsq.py:217-222`), replace
`nn.GroupNorm(GN_GROUPS, cout)` with per-position channel normalisation, spelled
either `nn.RMSNorm(cout)` on a channels-last permute or EDM2's parameter-free
`x * rsqrt(mean(x^2, dim=1) + eps)`. Encoder only in the first attempt; the
decoder's norms do not affect token identity.

**Cost to try.** One rung, about 88 minutes, plus the persistence and
spurious-flip re-measurement. Measured effect on gradient support is already
known exactly: 4096 pixels to 225.

**What would falsify it.** Any of: persistence does not rise materially above
86.62%; held-out PSNR drops below `NUM-BAR-Q1`; token entropy drops below
`NUM-BAR-Q2`. Run-to-run noise is `NUM-PERF-NOISE` at one epoch, so a persistence
move of a few points is well outside it and a PSNR move of hundredths is not.

**What it risks breaking.** Two things, in order of likelihood.

- **Q-2.** `NUM-TOK-ENT-R1` clears `NUM-BAR-Q2` by roughly four points. Attention
  bought `NUM-TOK-ATTNENT` of entropy specifically by decorrelating the FSQ
  digits, which is a *global mixing* effect. Removing the other global mixing
  mechanism could plausibly cost entropy in the same direction. This is the
  single most likely way the rung fails, and it fails on a gate that is already
  passing.
- **Training stability.** EDM2 removed GroupNorm only after taking control of
  activation and weight magnitudes (section 2.4). Mirage would be lifting Config
  F without Configs D and E. Mitigation if it diverges, in increasing cost:
  channel-only LayerNorm rather than pure RMS, then weight standardization on the
  convolutions (measured local in a.2), then Config-D-style magnitude-preserving
  convolutions.

**A cheaper decomposition.** Because R1 has no attention, R1-with-channel-norm is
the *only* configuration where the encoder is fully local. That is the rung to
run. Running R2-with-channel-norm answers a different and less interesting
question, since `GridAttention` reintroduces full coupling by design.

### 3. Copy-biased decoding at rollout time, ITC-style

**The change.** In the rollout loop only, bias the next-frame token distribution
toward the previous frame's token at the same grid position, either as ITC's
optimal-transport assignment or as a much simpler thresholded rule (keep the
previous token unless the model's top-1 probability for a different token exceeds
a margin).

**Cost to try.** No training at all. Implementation in the inference path plus a
threshold sweep against Q-3, Q-5, Q-6.

**What would falsify it.** No improvement in coherence horizon at any bias
strength, or an improvement that comes with a Q-4 action-following regression
(the arm stops moving because the copy prior is too strong).

**What it risks breaking.** Q-4 directly, and Q-5 indirectly: a copy bias that is
too strong freezes the arm, which is exactly the failure mode that scores well on
kinematic stability and badly on everything else. ITC guards this with a distance
cap (`d <= 4` grid cells) rather than a probability threshold, which is the
better-tested shape.

**Honest weighting.** ITC's own ablation attributes most of its headline gain to
3D RoPE rather than to the correspondence step. Expect a small effect. It is
ranked third only because it costs no training and because it targets Q-6's exact
failure mode by name ("object duplication, disappearance, and transmutation").

### 4. Blur-pool the stride-2 downsamples

**The change.** Replace `Conv2d(k=3, s=2, p=1)` with `Conv2d(k=3, s=1, p=1)`
followed by a fixed binomial low-pass filter at stride 2, per Zhang 1904.11486
section 3.

**Cost to try.** One rung, plus a small compute increase since the convolution
now runs at full resolution before subsampling.

**What would falsify it.** The companion measurement in option 1: if the
one-pixel-shift token change rate is already low, aliasing is not a mechanism
here and this option has no target.

**What it risks breaking.** Blur-pooling **increases** the arithmetic receptive
field, which pushes against option 2's goal. If both are adopted the locality
gain is partly given back. They should be evaluated separately, not stacked.

**Evidence weighting: weak.** The only tokenizer that adopted 3D blur pooling put
it in the discriminator, and the ablation step was the smallest in its table
(FVD 8.90 to 8.62). Zhang's own paper lists video temporal consistency as
untested future work.

### 5. Train the tokenizer with a lightweight AR prior loss, LARP-style

**The change.** Attach a small autoregressive transformer over the token grid
during tokenizer training, weighted by a small alpha, and discard it at inference
(LARP arXiv 2410.21264, section 3, equation 8: "the prior model is used solely to
encourage an AR-friendly discrete latent space for LARP during training. It is
discarded at inference time").

**Cost to try.** One rung plus a new module and a new loss term, so materially
more than a rung of wall clock. Also a new hyperparameter (alpha), and LARP notes
it needs a higher learning rate on the prior's parameters.

**What would falsify it.** No improvement in Q-3 horizon over the plain rung, at
any alpha tried.

**What it risks breaking.** It optimises the tokenizer against a *proxy* for the
real dynamics model, so it can overfit to the proxy. It also makes the tokenizer
depend on the Phase 2 design, which cuts against the current phase separation.

**Why it is ranked below the cheap options despite strong evidence.** LARP's
result is the most direct published evidence that shaping a tokenizer for AR
consumption improves AR generation. But it is a bigger change than mirage needs
to test first, and options 1 and 2 are prerequisites for knowing whether the
problem is real.

### 6. An explicit token-change penalty. Do not do this yet.

**The change.** Add a term penalising `||z_t - z_{t-1}||` on the pre-quantisation
latent for adjacent frames.

**Why it ranks last among the tokenizer changes.** Zero published support
(section b.1). It fights `NUM-BAR-Q2` directly, since the cheapest way to satisfy
a smoothness penalty is to use fewer distinct codes. And it requires the loader
to serve adjacent frame pairs, which the current sampler is not built for. The
mechanism it targets is better addressed by removing the *cause* of spurious
flips (option 2) than by penalising the *symptom*.

### 7. Causal 3D encoder. Out of scope for Phase 1.

Usable in principle under the one-frame-at-a-time constraint, since both
MAGVIT-v2 and Cosmos are designed so the first frame is encodable alone. Ruled
out on cost: it breaks the fixed 64-tokens-per-frame contract the dynamics model
sequence layout assumes, and it needs encoder state carried across rollout steps.
Record the trigger rather than the plan: **if Q-3 fails at 200 frames after
options 1 through 3, and the spurious-flip rate is low (so locality is not the
cause), a causal temporal encoder is the next structural lever.**

---

## A discrepancy found on the way, not fixed here

`bench/patch_probe.py:60` sets `RF = 22  # the encoder's receptive field at one
8x8 cell`, and the phrase "22x22 receptive field" appears in
`world_model_architecture.md`, `phase1_structural_plan.md`,
`phase1_progress_report.md`, `AGENDA.md` and `mathematics_notes.md`.

**The encoder's receptive field is 15x15, not 22x22.** Two independent
derivations:

- Arithmetic, over `mirage/fsq.py:290-322` (three `k=3, s=2` stages then a `1x1`):
  start at 1 for the `1x1`, then `RF = (RF - 1) * stride + kernel` backwards
  gives 3, then 7, then **15**.
- Measured, section a.2 above: 225 pixels in a 15x15 box once the normalisation's
  spatial coupling is removed. This also matches the 225-pixel figure the project
  already measured with GroupNorm neutralised.

**Consequence, and its direction.** `NUM-TOK-FLAT` is the share of interior cells
whose receptive field is one flat colour, and `NUM-TOK-Q2CEIL` is derived from it
by the argument that cells with identical receptive fields must get identical
codes. A **smaller** window is **more** likely to be flat, so the true flat share
is higher than `NUM-TOK-FLAT` and the true entropy ceiling is **below**
`NUM-TOK-Q2CEIL`. By how much is unmeasured; settling it is one edit to
`bench/patch_probe.py` and one rerun.

This is very unlikely to threaten Q-2, since `NUM-TOK-Q2CEIL` sits far above
`NUM-BAR-Q2`. It is recorded because a registered number that no longer matches
its own derivation is exactly what `canonical_numbers.md` exists to catch, and
because the 22 figure is quoted in five places. **Not changed here**, since this
document was scoped to create one file.

---

## Sources

| Source | Type | What it was used for | URL |
|---|---|---|---|
| Wu and He, *Group Normalization*, arXiv 1803.08494, section 3 | paper | Reduction axes for BN, LN, IN, GN | https://arxiv.org/abs/1803.08494 |
| PyTorch 2.9.1 `torch/nn/modules/{normalization,batchnorm,instancenorm}.py` | source code (installed) | Docstring wording for GroupNorm, LayerNorm, RMSNorm, BatchNorm2d, InstanceNorm2d | local install; mirrors https://docs.pytorch.org/docs/2.9/ |
| Karras et al., *Analyzing and Improving the Training Dynamics of Diffusion Models* (EDM2), arXiv 2312.02696, sections 2.4 and B.6, Table 1 | paper | The locality argument against global normalisation; measured FID gain from removing GroupNorm | https://arxiv.org/abs/2312.02696 |
| `NVlabs/edm2` `training/networks_edm2.py` (lines 20-25, 91-108, 153, 174) | source code | Pixel norm is channel-only; forced weight normalisation; no GroupNorm | https://github.com/NVlabs/edm2 |
| Brock et al., *High-Performance Large-Scale Image Recognition Without Normalization*, arXiv 2102.06171, sections 1, 3, 4 | paper | Normalizer-free nets; Scaled Weight Standardization; AGC | https://arxiv.org/abs/2102.06171 |
| Qiao et al., *Weight Standardization*, arXiv 1903.10520 | paper | The weight-reparameterisation named by NFNets | https://arxiv.org/abs/1903.10520 |
| Liu et al., *A ConvNet for the 2020s* (ConvNeXt), arXiv 2201.03545, section 2 | paper | LN replacing BN in a convnet, 81.5% top-1 | https://arxiv.org/abs/2201.03545 |
| `facebookresearch/ConvNeXt` `models/convnext.py` (lines 37-50, 119-143) | source code | ConvNeXt's LayerNorm is channel-only at each position, both data formats | https://github.com/facebookresearch/ConvNeXt |
| Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple*, arXiv 2309.15505, appendices A.1 and A.4.3, Table 1 | paper | FSQ inherits MaskGIT's model code and configs; recommended levels | https://arxiv.org/abs/2309.15505 |
| `google-research/maskgit` `maskgit/nets/layers.py:49-66`, `maskgit/configs/vqgan_config.py:64` | source code | `norm_type = "GN"`; the norm-layer factory | https://github.com/google-research/maskgit |
| `CompVis/taming-transformers` `taming/modules/diffusionmodules/model.py:34-35` | source code | VQGAN's `Normalize` is `GroupNorm(32)` | https://github.com/CompVis/taming-transformers |
| `lucidrains/vector-quantize-pytorch` `finite_scalar_quantization.py` | source code | FSQ module carries no normalisation | https://github.com/lucidrains/vector-quantize-pytorch |
| `TencentARC/Open-MAGVIT2` `src/Open_MAGVIT2/modules/diffusionmodules/improved_model.py` | source code | GroupNorm(32) throughout; AdaptiveGroupNorm in the decoder | https://github.com/TencentARC/Open-MAGVIT2 |
| `wilson1yan/VideoGPT` `videogpt/vqvae.py` | source code | BatchNorm3d; no norm on the downsampling convs | https://github.com/wilson1yan/VideoGPT |
| Yan et al., *VideoGPT*, arXiv 2104.10157 | paper | 3D convs plus axial attention with LayerNorm | https://arxiv.org/abs/2104.10157 |
| Ge et al., *Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer* (TATS), arXiv 2204.03638, sections 2.2 and 0.A.1 | paper | Temporal shift-equivariance; zero-padding corruption; replicate padding; Synced BN replacing GroupNorm | https://arxiv.org/abs/2204.03638 |
| Yu et al., *MAGVIT*, arXiv 2212.05199 | paper | 3D-VQ baseline that MAGVIT-v2 criticises for flicker | https://arxiv.org/abs/2212.05199 |
| Yu et al., *Language Model Beats Diffusion* (MAGVIT-v2), arXiv 2310.05737, section 3, Tables 5(b) and 5(c) | paper | Flicker motivation; causal 3D CNN; LFQ; adaptive group norm; 3D blur pooling in the discriminator | https://arxiv.org/abs/2310.05737 |
| NVIDIA, *Cosmos World Foundation Model Platform*, arXiv 2501.03575 | paper | "LayerNorm instead of GroupNorm"; causal tokenizer rationale; wavelet front end | https://arxiv.org/abs/2501.03575 |
| `NVIDIA/Cosmos-Tokenizer` `cosmos_tokenizer/modules/utils.py:67-84`, `modules/layers3d.py` | source code | `CausalNormalize` is `GroupNorm(num_groups=1)` per frame, so still spatially global | https://github.com/NVIDIA/Cosmos-Tokenizer |
| Wang et al., *OmniTokenizer*, arXiv 2406.09399 | paper | Checked for temporal-consistency losses; none found | https://arxiv.org/abs/2406.09399 |
| Wang et al., *LARP*, arXiv 2410.21264, sections 1 and 3 | paper | AR prior loss during tokenizer training; "higher reconstruction quality ... worse generation fidelity" | https://arxiv.org/abs/2410.21264 |
| Yao and Wang, *Reconstruction vs. Generation* (VA-VAE), arXiv 2501.01423 | paper | The reconstruction-generation optimization dilemma | https://arxiv.org/abs/2501.01423 |
| Bruce et al., *Genie*, arXiv 2402.15391, section 3.4 Table 3 | paper | Tokenizer architecture ablation at fixed dynamics model | https://arxiv.org/abs/2402.15391 |
| Wu et al., *iVideoGPT*, arXiv 2405.15223, sections 3.1 and 4.1 | paper | Conditioned tokenization; MaskViT flicker attribution | https://arxiv.org/abs/2405.15223 |
| Guo et al., *MineWorld*, arXiv 2504.08388 | paper | Per-frame image tokenizer, no flicker metric reported | https://arxiv.org/abs/2504.08388 |
| Kim et al., *Identifiable Token Correspondence for World Models*, arXiv 2605.16457 | paper | Copy-versus-generate decoding; Craftax results and ablation; motion-distance prior | https://arxiv.org/abs/2605.16457 |
| Kerssies et al., *A Frame is Worth One Token* (DeltaTok/DeltaWorld), arXiv 2604.04913 | paper | Inter-frame redundancy exploited rather than measured | https://arxiv.org/abs/2604.04913 |
| Chen et al., *TivTok*, arXiv 2606.17590, sections 3.2 and 4.5 | paper | Time-invariant versus time-variant token factorisation | https://arxiv.org/abs/2606.17590 |
| *Efficient Video Sampling*, arXiv 2510.14624, section 3 | paper | Static-patch metric, defined in pixel space | https://arxiv.org/abs/2510.14624 |
| Zhang, *Making Convolutional Networks Shift-Invariant Again*, arXiv 1904.11486, sections 3, 4.3, 5 | paper | BlurPool; classification consistency metric; video temporal consistency named as future work | https://arxiv.org/abs/1904.11486 |
| Karras et al., *Analyzing and Improving the Image Quality of StyleGAN* (StyleGAN2), arXiv 1912.04958 | paper | The blob-artifact precedent EDM2 and TATS both cite | https://arxiv.org/abs/1912.04958 |
| Ross, Gordon and Bagnell, arXiv 1011.0686, Theorem 2.1 | paper | The `T^2 * epsilon` compounding-error bound and its tightness | https://arxiv.org/abs/1011.0686 |
| Vahdat and Kautz, *NVAE*, arXiv 2007.03898, section 3 | paper | BatchNorm's train/eval asymmetry in a conv VAE; IAF-VAE as a norm-free small-image precedent | https://arxiv.org/abs/2007.03898 |
| Kingma et al., *Improved Variational Inference with Inverse Autoregressive Flow*, arXiv 1606.04934 | paper | Conv VAE without batch normalisation, small images | https://arxiv.org/abs/1606.04934 |
| Yang et al., *AR Forcing*, arXiv 2605.31314 | paper | Train/inference distribution shift in AR world models; context only, no tokenizer claim used | https://arxiv.org/abs/2605.31314 |
| Web search result summaries | secondary | Used only to locate candidate papers. No technical claim in this document rests on one; every claim was re-verified against the paper or the source file | - |
