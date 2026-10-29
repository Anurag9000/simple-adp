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

## V02-V11 requested but not implemented
V02 residual adapters; V03 native ViT FFN; V04 ConvNeXt channel FFN;
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

This establishes *candidate-level* source training/mutation, not search
controller parity, multi-site selection, serialized mid-candidate resume or
coordinated runner reachability. V02 remains partial until all those paths
are working. Other requested architectures V03–V11 remain unimplemented.
