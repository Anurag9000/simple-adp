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

## Outstanding V07–V11 architectures
V07 MLP-Mixer token/channel FFNs; V08 multisite adapters; V09 strict bank
of adapters; V10 ResMLP channel FFN; V11 GFNet Fourier-mixed FFN. V03–V06b
now have small random-initialized internal-FFN implementations and opt-in
worker routes (selected CPU CI verified through V05; V06a/b pending CI).
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

## Experimental V05 PoolFormer channel FFN contract
A fixed 3x3 average pooling spatial mixer computes `Pool(x)-x` and the
residual `x + (Pool(x)-x)` before the channels-last expandable GELU MLP.
The frozen patch stem, pool geometry, normalization, layer-scale and
classifier remain immutable. The only eligible site is
`channel:poolformer.block0`. V05 inherits the same signed whole-image
search controller, source RR/all/D1/Dcur operations, fixed external channel
dimension, and strict frozen-state restoration as V03/V04. It is a tiny
from-scratch experimental control, not a pretrained PoolFormer checkpoint.
Tests and opt-in runner integration pending CI.

## V06a and V06b parameter-free spatial-shift contracts
V06a `S2SpatialShift` divides frozen channels into four equal groups and
shifts up, down, left and right by one position with zero boundaries.
V06b `AxialSpatialShift` forms separate vertical and horizontal signed
shift branches and combines their outputs equally; it is intentionally
distinct from V06a. Both preserve image/token geometry and have no trainable
spatial mixer parameters. No circular wrap, convolutional-channel change or
attention operation is permitted. Their selected local channels-last GELU
FFN is the only trainable/expandable site. They reuse the whole-image
composite controller/checkpoint, an independently signed architecture spec,
source growth/method profile identities and paired original/variance routes.
The small models are architecture-shaped controls, not the exact pretrained
published network or scientific demonstration. New CI evidence pending.

## Verified selected CPU evidence through V06b
The V06a/V06b native shift-MLP implementation and paired coordinated runner
passed 288 selected CPU tests at
https://github.com/Anurag9000/simple-adp/actions/runs/41399092726.
V01/V02 remain supported along with V03/V04/V05. V07–V11 remain missing.

## V07 MLP-Mixer single-site control contract (pending CI)
The fixed patch embedding creates T tokens with D channels. The first Mixer
residual branch transposes [B,T,D] to [B,D,T] and applies a token-mixing
MLP of fixed T input/output; the second branch applies a channel-mixing MLP
of fixed D input/output. A real frozen opposing MLP remains in each job, so
there is no no_grad boundary in the composite forward.
`MixerSpec(site_id="token:block0")` selects internal hidden layers in
only the token MLP; `MixerSpec(site_id="channel:block0")` selects hidden
layers in only the channel MLP. Different sites have different checkpoint
signatures and independent coordinated-runner route identities.
The frozen opposite site, stem, norms and classifier are hashed and checked.
No token count, channel interface, patch size, embeddings or frozen output
operator may expand. A joint token/channel adaptation on one candidate is
not yet implemented and must not be counted. Tiny random-initialized
MLP-Mixer-style control only; no pretrained Mixer checkpoints validated.

## V07 synchronous two-site ablation (pending CPU CI)
Alongside the two canonical single-site controls, a distinctly labeled
`MixerSpec(site_id="both:block0")` trains both real residual Mixer MLPs.
Each ADP width/depth proposal is applied to both hidden stacks, keeping
fixed distinct token-interface T and channel-interface D unchanged.
Architecture selection still uses one full-image validation result per
joint proposal, but this **explicit synchronous-both schedule is an
additional site policy**, not an implicit redefinition of `ADP-W-All`.
It has a separately named checkpoint/search format, a joint two-site neuron
budget, per-site RR cursor persistence and a separate opt-in suite identity.
This is not V08 arbitrary multistage scheduling or adaptive site selection.

## V10 frozen ResMLP token mixing and channel growth (pending CI)
The patch stem produces fixed T-by-C tokens. A frozen ordinary dense
`Linear(T,T)` crosses spatial tokens after fixed learned-affine-like scale/
shift, and the channel MLP is the only ADP-eligible target. Its input/output
channel dimension C never expands. The fixed token-linear and affine state
are hashed, restored strictly and excluded from the optimizer.

## V11 frozen Fourier global filter and real channel growth (pending CI)
A fixed non-unit complex64 spectral filter with exact shape
`[patch_H, patch_W//2+1, channels]` acts only on the spatial FFT axes.
`rfft2 -> complex multiplication -> irfft2` remains in the differentiable
image forward; no FFT or complex filter coefficients are ADP growth targets.
A real channel `TabularMLP` (GELU hidden layers) is the sole expandable
FFN. Frozen filter and resolution contracts are signed and strict. Both
V10/V11 use separate opt-in runner names and the native method/criterion
ledger. Not a published pretrained-model reproduction.

## V08 fixed three-stage residual adapter contracts (pending CPU CI)
Three preinstalled stages preserve a frozen stem and three spatial 3x3
convolutions, with local residual channel MLPs inserted immediately after
stage-specific convolutions. Site IDs are `adapter:early`,
`adapter:middle`, and `adapter:late`. Each job chooses **one** eligible
local MLP with fixed C-to-C interfaces. The other two real adapter MLPs
remain frozen and in the differentiable forward; neither cached embeddings
nor hidden channel-count mutation is permitted. Frozen state hashing covers
all CNN modules, classifier and both nonselected adapters.
The local controller and signed batch checkpoint are unchanged; the V08 model
factory and restore dispatcher use a distinct kind/name. No adaptive
multistage proposal allocation or joint-site policy is claimed yet.

## V09 fixed hard-routing adapter-bank contract (pending CI)
A fixed frozen convolutional backbone produces [B,C,H,W] features.
The fixed projected feature mean passes through a hard argmax K-way router,
selecting exactly one adapter member per image independent of labels and
other images in its batch. K=3 in the built-in worker. All candidate members
share the unchanged d->d local interface and frozen project-in/out operators.
Only one job-declared `bank:0`/`bank:1`/`bank:2` member is trainable:
nonselected adapters, routing projections and CNN remain eval/frozen and are
included in the immutable SHA-256 state identity. The selected MLP can grow
width or depth; the bank size, output logits and router scores cannot.

The signature and serialized snapshot declare the selected member, bank size,
router kind and exact fixed state. Recording training/test routing occupancy
per member does not make it a learned router, and is not a guarantee of
balanced routing; a selected member may receive few or no routed samples.
The explicit frozen hard router is a controlled baseline, not a task-id router
or an adaptive/generative posterior gate. Tests pending CI.

## V08 noncanonical fixed round-robin multisite scheduling (pending CI)
`ScheduledMultistageSpec` has exactly three fixed local growth targets,
`adapter:early`, `adapter:middle`, `adapter:late`, and immutable outer
order `early -> middle -> late`. The site controller uses the *proposal
index*, not hidden-layer RR, to choose the next target. At each optimization
stage only that target MLP is trainable. All three MLP weights and per-site
layer cursors are saved and restored independently; nonactive MLPs and fixed
CNN operations remain in the differentiable forward.
The fixed-state hash covers backbone/head only; per-site weights are tracked
in the signed candidate snapshot and verified by stage-resume comparisons.
Aggregate neuron budget sums all three local MLPs. This route has a separate
opt-in runner name, snapshot kind, source-version signature and site schedule
history. It is not a learned site selector or a change to the canonical
within-MLP method taxonomy. Tests pending CI.

## Verified V08 schedule execution (2026-11-02)
The predeclared round-robin early/middle/late policy and its independently
signed per-site state passed 354 selected CPU tests at
https://github.com/Anurag9000/simple-adp/actions/runs/41415527242.
This validates real fixed-stage local MLP training/mutations, not learned
site selection or a pretrained three-stage source backbone.


## V01 official-dataset and frozen source transform identity

\`VisionInputAdapter\` accepts raw float32 [N,1|3,H,W] official dataset
partitions, repeats grayscale to RGB, and applies deterministic bilinear
resizing to source-compatible 32×32 (ResNet18/ConvNeXt) or 224×224 (ViT).
All input/output resolutions, channel policies, normalization and frozen
encoder state SHA256 are signed with the run; there is no fitted transform,
stochastic augmentation or test-label-dependent preprocessing. The source
checkpoint (when supplied) must be a complete strict torchvision state_dict.
Loading an arbitrary local checkpoint does **not** prove pretrained origin.
Official test is only materialized after ADP train/validation architecture
selection. The V01 head is the only trainable part; this does not implement
pretrained inner-FFN injection for V03–V11.
