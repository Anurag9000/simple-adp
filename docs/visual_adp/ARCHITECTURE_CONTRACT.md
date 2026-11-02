# Visual ADP architecture contract

## V01 (first implemented incremental feature)
A caller supplies an actual nn.Module image feature extractor with all classification
layers removed. It outputs [B,D] (pool=vector), [B,T,D] (pool=cls or tokens_mean),
or [B,C,H,W] (pool=spatial_mean). The encoder is made eval-only and frozen; its
parameters and buffers are hashed before and after extraction. ADP trains the
dense TabularMLP head, using exactly the registered controller/model ID.

The feature-input dimension D and supervised output dimension remain fixed.
Encoder changes invalidate the visual identity; image resolutions can be
declared and checked. The caller owns preprocessing, pretrained weight acquisition,
split selection, and JSON-serializable dataset provenance. Never pass final
classification logits as embeddings unintentionally. Do not cache embeddings
after trainable internal backbone layers. The feature-head path deliberately
uses the existing tabular preprocessing on frozen embeddings only.

## V03-V11 requested but not implemented
V03 native ViT FFN; V04 ConvNeXt channel FFN;
V05 PoolFormer channel FFN; V06a S2 spatial shift; V06b AS axial shift;
V07 MLP-Mixer token/channel FFNs; V08 multisite adapters; V09 strict bank
of adapters; V10 ResMLP channel FFN; V11 GFNet Fourier-mixed FFN.
All require fixed external interfaces, authentic composite forward/backward,
local eligible-site expansion, reproducible snapshots, controller/runner integration.
No convolution/attention/filter/channel-interface/token-count mutation is permitted.

## Concrete V01 frozen-source encoder interface

src/simple_adp/vision_encoders.py exposes resnet18, convnext_tiny and
vit_b_16 torchvision feature extractors with the full source model loaded
strictly *before* replacing its classifier with identity. ViT-B/16 fixes
resolution at 224x224. Local full-state checkpoints may be supplied with a
SHA256 expectation. Raw weight files alone establish neither pretrained status
nor source dataset: weights_source is explicitly unverified unless independently
documented. Architecture construction does not modify the global RNG and does
not download weights. All frozen parameters and buffers remain untouched during
head training; a torch.no_grad() boundary is valid here only because the encoder
precedes the sole eligible dense head.

## V02 partial local-growth candidate and training-stage mechanics

src/simple_adp/visual_inner.py exposes a real CNN residual adapter model
with `h + P_out MLP(P_in h)` inserted between two frozen convolutional
stages. `P_in` and `P_out` have fixed dimensions and weights; `P_out` is
nonzero to permit gradients into the local MLP from the first step. The
local `TabularMLP` grows only hidden widths/depth through the repository's
exact RR, all-layer, dedicated DEN RR, serial D1 or Dcur implementations.
The source StageTrainer trains the whole image model and validation scores
the whole composite through TaskAdapter; frozen downstream operations
participate in autograd, while frozen params/BN buffers remain unchanged.
Candidate snapshots reconstruct the composite architecture before strict
weight loading and do not consume source RNG draws.

V02 has since gained standalone six-method dedicated controllers with signed
batch-level resume and an opt-in coordinated runner. The eight generic
phase-search controllers and their faithful/corrected profiles have passed
selected CPU CI (259 tests). V02 remains partial because the concrete source
backbone here is a frozen random residual CNN control, not a complete
pretrained ResNet/ViT experiment. V03–V11 remain unimplemented.

## V02 current controller evidence
All 14 canonical IDs run through the V02 full-image composite controller.
Dedicated D1 includes intermediate optimization-only stages. Generic phase
search retains its faithful and corrected source profiles. Signed checkpoints
include model snapshots, architecture/history, optimizer/epoch/batch/RNG,
and the exact method/profile/data identity. The opt-in runner enumerates
22 method/profile routes plus 21 variance counterparts per seed/split/repeat.
Synthetic CPU integration passed at
https://github.com/Anurag9000/simple-adp/actions/runs/41394281464.
V02 uses a frozen random CNN control, not a verified pretrained ResNet-like
adapter or empirical model-quality claim.

## Experimental V03 and V04 native FFN site contracts
`NativeFFNSpec` enforces fixed NCHW resolution, patch size and site ID;
V03 uses `ffn:block0` and V04 uses `channel:stage0.block0`.
V03 is a small train-from-scratch frozen patch stem + CLS/position embeddings
+ self-attention + FFN with GELU on all tokens + CLS classifier. V04 is a
small frozen strided stem + 7x7 depthwise spatial mixer + channels-last
LayerNorm/GELU FFN + fixed layer scale + pooled classifier. These are genuine
inside-backbone FFN sites, not the V01 pooled-head route or a residual
projection adapter; they are *not* pretrained production torchvision models.
All native growth operators use local source TabularMLP overlap copying,
while stateless ReLU layers produced by those operators are converted back to
GELU to retain FFN activation semantics. Interface d_in=d_out=d_model,
token count, depthwise channels, patch count and attention heads never grow.
All fixed modules have requires_grad=False, eval mode and a hash checking
weights/buffers. Frozen downstream paths remain in autograd. Full-image
candidate search/restoration uses VisualResidualSearch's additive composite
model factory, with a separate checkpoint namespace for V03/V04. Standalone
constructor/search tests pending CI; no real-data or pretrained weight claim.
