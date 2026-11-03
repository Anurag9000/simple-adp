# Live source audit: visual ADP

## Re-audit of current main (2026-11-02)

The original baseline paragraphs below are a chronological record, **not the
present implementation boundary**. At the inspected main
`9b1d7fb96a4a2e71e056a7bd8eaef01130fc897e`, the tree already includes
`visual.py`, `visual_search.py`, `visual_inner.py`,
`visual_native_ffn.py`, `visual_shift.py`, `visual_mixer.py`,
`visual_multistage.py`, `visual_multistage_schedule.py`,
`visual_adapter_bank.py`, `visual_source_convnext.py` and
`visual_source_vit.py`. All original 14 identities remain in
`registry.py`; the tiny controlled vision models are explicitly distinct
from source-faithful torchvision FFNs.

Actual controller path for V02-V11:
`VisionInnerModelAdapter.fit_predict` -> `VisualResidualSearch.fit`
-> `build_visual_composite` -> `StageTrainer.train_stage` ->
`VisualResidualSearch._finish_stage` -> signed
`search.model_.snapshot` inference artifact. Growth mutates only the
local eligible dense stack. `VisionSourceConvNeXtModelAdapter` and
`VisionSourceViTModelAdapter` use distinct explicit source gates and
full-file SHA256 identity; V01 remains
`VisionModelAdapter` -> `VisualADPEstimator` -> `TabularADP`.
This is a dependency/data-flow map, not a claim that all jobs have been
scientifically executed.

The latest inspected baseline complete CPU suite passed:
https://github.com/Anurag9000/simple-adp/actions/runs/41488650402.
Selected real pretrained internal source-FFN micro-smokes also exist for
V03 and V04, but are not full multi-seed scientific comparisons.
No physical CUDA full-suite pass or source-faithful coverage for all
controlled V05-V11 designs was established.

On top of that baseline, commits `a07034219e7f4eac0f8fb8f39409e66687352663`
and `41e7410405adf8bcba5028499079e5880db8bd98` defer held-out
image **and label** indexing in both V01 and V02-V11 coordinated adapters
until after `fit()` returns and add a guarded-access regression for both.
The dedicated source V03/V04 runner already deferred test indexing.
The canonical runner AST navigation map was updated in
`fe8447b478289d19fec42574bbf7a07eec204e18`.
CI validation of these new changes must be recorded separately before
marking the regression runtime-tested.

Unclosed science/engineering issues: CUDA hardware coverage, per-candidate
memory/FLOP accounting and OOM rejection, real source-faithful V05-V11
backbones, bounded official-image experiments for all controlled families,
matched computation-budget analyses and larger multi-seed evaluations.

---

Baseline main: a2e6b6325a80ca6da41b1898ebf71c6ec8829a62 (2026-10-29).
This is an incremental evidence ledger, not a claim that the full vision suite is finished.

## Existing algorithm and API anchors
- src/simple_adp/models.py: TabularMLP, snapshot/from_snapshot, rebuild_mlp. Hidden layer stack is nn.Linear + optional BatchNorm + ReLU, followed by fixed-dimensional output. rebuild_mlp copies overlapping state tensor coordinates.
- src/simple_adp/growth.py: RR, synchronized all-layer width, DEN-specific RR, width-one and matched-width depth, can_* capacity predicates. These accept TabularMLP rather than arbitrary CNN/ViT composite models.
- src/simple_adp/search.py: SearchController._arrays converts feature rows to float tensors; _initial_model creates TabularMLP; run dispatches the 14 configurations and validates TabularMLP-specific resume payloads.
- src/simple_adp/registry.py: exactly 14 registered configurations. They must not be redefined by the vision extension.
- src/simple_adp/tasks.py: TaskAdapter classification/regression objective, target encoding, validation selection, and reporting.
- src/simple_adp/estimator.py: TabularADP preprocessing, native search controller, checkpoints, fit/predict/save/load. The V01 frozen-feature route delegates to this API without changing its semantics.
- src/simple_adp/runner/models.py: ModelRegistry plus SimpleADPModelAdapter enumerate_routes/fit_predict; assumes TabularTaskData and TabularADP.
- src/simple_adp/runner/datasets.py: DatasetProvider, DatasetRegistry, PreparedTaskData. Explicit opt-in V01 provider and adapter live in runner/vision.py.
- src/simple_adp/runner/plan.py: SuitePlan and RunIdentity with provider/suite/task/split/route signatures.
- src/simple_adp/config.py, generic_search.py, dedicated.py: native ADP method settings, generic/dedicated controllers and variance-stagnation criteria.

## Current data flow
Raw image [N,C,H,W] -> user-provided frozen classifier-free encoder (no-grad, eval)
-> pooled [N,D] feature embeddings -> original TabularADP TabularPreprocessor
-> original SearchController -> original TabularMLP growth and TaskAdapter validation.
Inference repeats the same frozen embedding path. Cache equivalence depends on
identical encoder and preprocessing; cache contents must be identified independently.

## Historical unresolved integration from the initial V01 audit (superseded)
For V02-V11, inserting a local MLP inside a composite requires passing the whole
image model through candidate train/evaluation, with fixed downstream differentiable
layers kept in autograd. SearchController, StageTrainer, model snapshots, candidate
mutation, and runner remain TabularMLP-specific. Replacing a module in place without
those integrations is **not** a complete ADP route. A V01 opt-in runner provider, signed splits, and matched method/criterion jobs
now have static code and a worker smoke test awaiting CI. Real CIFAR training,
pretrained backbones, inner-network growth, whole-repo regression execution,
and full serial-depth parity are not yet implemented in this increment.

## Additional opt-in image preparation surface
src/simple_adp/vision_data.py adds torchvision dataset loading and deterministic
classwise source-train/validation partitioning, preserving official test
isolation and recording raw checksums and original indices. This is an installed package API. Separately, runner/vision.py adds an
experimental opt-in DatasetRegistry provider and model adapter for the V01
frozen random tiny-conv control, with data-isolated runner jobs. The optional torchvision extra does not affect existing dependencies.

## Concrete torchvision V01 loaders
src/simple_adp/vision_encoders.py adds frozen resnet18, convnext_tiny and
vit_b_16 constructors. User-provided local torchvision source state_dicts are
hashed and loaded strictly before removing classifier heads. A provided local
checkpoint is not automatically certified as pretrained. The coordinated runner
continues to use its tiny random frozen CNN control: these explicit factories
are available through the package API, not yet native runner configurations.

## Experimental V02 residual inner-MLP candidate
src/simple_adp/visual_inner.py introduces VisualResidualAdapter with a
frozen-CNN/adapter/frozen-CNN composite forward and source-local TabularMLP
mutation (RR, all-layer, DEN-RR, D1, Dcur). Unlike V01 cached embeddings,
all candidate training is on full image tensors, with frozen *downstream*
operations left in autograd. StageTrainer is already duck-typed and accepts
this composite model; canonical DedicatedController/GenericCandidateSearch
still assume TabularMLP restoration and have **not** been generalized to V02.
This is a partial candidate-level implementation; no V02 runner job is yet
allowed by the official compatibility matrix.

## Updated V02 native composite search implementation
The earlier audit above describes the initial unimplemented state, not the
current V02 code. visual_search.py now implements whole-image stage optimization,
architecture-candidate lifecycle, signed exact mid-batch restart, dedicated D1
warmup, and generic faithful/corrected phase search. runner/vision.py registers
`vision_inner` only under SIMPLE_ADP_ENABLE_VISION=1. Existing V01 and
tabular execution remain separate. Last validated V02-D1 CPU CI: 241 passed;
generic all-method extension passed 259 selected CPU tests. No source-pretrained CNN, real CIFAR
scientific run, or V03–V11 local FFN implementation has been verified.


## Measured visual reporting path (2026-11-03)

The opt-in scientific reporting extension follows existing identities,
controllers and worker contracts:

```text
VisionModelAdapter (V01)
  -> original VisualADPEstimator / TabularADP + frozen encoder
  -> selected native head
  -> visual_parameter_accounting
  -> visual_linear_conv_macs on encoder -> head (one validation image)
  -> ClassificationTask + post-selection visual_calibration_metrics
  -> runner.worker.run_spec normal signed result.json

VisionInnerModelAdapter (V02-V11) and VisionSource* (V03/V04)
  -> native VisualResidualSearch
  -> StageTrainer stage actual optimizer steps/epochs
  -> per-completed-stage candidate costs + D1 warmup aggregate budget
  -> selected composite full-image logits, with frozen downstream graph
  -> visual_parameter_accounting + visual_linear_conv_macs
  -> ClassificationTask + post-selection visual_calibration_metrics
  -> runner.worker.run_spec normal signed result.json
```

New anchors: `visual_compute.visual_linear_conv_macs`,
`visual_metrics.visual_calibration_metrics`,
`visual_search.VisualResidualSearch._finish_stage`,
`runner.vision.VisionModelAdapter.fit_predict`,
`runner.vision.VisionInnerModelAdapter.fit_predict`,
`runner.vision_source.VisionSourceConvNeXtModelAdapter.fit_predict`
(shared V03/V04 runner), and `runner.worker.run_spec`.

This addition does **not** alter `TabularMLP`,
`rebuild_mlp`, source ADP growth semantics, the original
`MODEL_REGISTRY`, `TabularADP.save/load`, or the original
TaskAdapter cross-entropy candidate-scoring contract. It provides a
partial measured Linear/Conv2d inference MAC count rather than
whole-model FLOPs and does not claim genuine calibration with tiny
test samples. Physical CUDA and source-faithful V05–V11 remain
unvalidated/unfinished respectively.


## Additional inference evidence and scientific comparability caveat

The `visual_compute` path now estimates dense
`nn.MultiheadAttention` projection and QKᵀ/AV MACs using actual
executed module input geometry; this measures an analytical operation
count rather than instrumenting CUDA kernels. The parent MHA's
`out_proj` Linear submodule is excluded from the directly observed
Linear count, preventing duplicate costs. Other functional attention
and Fourier kernels remain explicitly unmeasured. Source V03's
true torchvision MultiheadAttention submodules are included in the
estimator; V11 global FFT is not converted to a fake dense MAC count.

The official controlled model matrix now invokes
`audit_visual_worker_pairs` after all normal signed workers finish,
giving a provenance-checked 19-pair budget audit. This audit never
modifies a method, checkpoint, model, training loop, job plan or
validation objective; missing legacy V01 optimizer-work records
remain unquantified. Since equal optimizer steps need not mean equal
full-model FLOPs, the audit does not label any pair compute-matched
or make scientific performance claims.


## Source dataset extension without duplicating training controllers

The bounded official-image driver keeps all real data loading
in torchvision's existing `FashionMNIST`, `CIFAR10`, or
`CIFAR100`, then passes only the selected dataset identifier
to `VisionDatasetProvider` via `SuitePlan.build`. That provider
retains `download=False`, class-balanced separate fit/
validation/official-test source indices, raw uint8 checksums,
and no test access until after `fit()`. No new model builder,
controller or optimizer was introduced for CIFAR.

CIFAR10/RGB input is 32x32x3 and stays 3-channel after
`_raw_nchw`; Fashion-MNIST is grayscale and remains 28x28x1.
The script's one-seed, 40/20/20 smoke configuration is
**not** full-source or source-network fidelity evidence for
V05–V11. CIFAR100 requires at least 100 class-balanced examples
in every partition; its full official-source worker CI has
not been run as part of this incremental code addition.


## Native parameter/buffer footprint and safe candidate admission

The memory audit reads the *actual* native/source model's parameter
objects and registered buffers without moving tensor data, changing
grad flags or running another forward pass. Dtype-based logical
payload accounts for complex GFNet filters and source-model buffers.
The opt-in guard checks the constructed initial model before native
training and each grown candidate before StageTrainer begins; it
does not alter `TabularMLP` growth, source copy semantics, backbones,
class heads, checkpoint structure of pre-existing uncapped jobs, or
the accepted-model validation rule.

`RunnerSettings` serializes a positive byte cap only when
explicitly set. `VisualResidualSearch._signature` likewise
includes it only when active, preserving uncapped checkpoints and
preventing capped resumes from switching policies. The V01 route
continues its original tabular head estimator and fails closed
if a per-candidate *native* byte cap is requested. Rejection records,
and logical model memory for all trained stages including D1 warmup,
are kept in the native search's signed state and serialized into
coordinated worker results. The cap is not a complete OOM guard
because training activations, optimizer allocations and temporary
proposal copies are unmeasured.


## Pre-construction local-MLP memory admission and source preservation

The new `visual_memory` projectors independently derive
source `TabularMLP` width/depth shapes, exact unaliased
parameter and optional BatchNorm buffer byte counts,
then replace the affected local-site footprint in the full
composite's logical payload. This calculation occurs before
native `grow_width`, `grow_depth`, fixed-candidate construction
or D1 warmup deep-copy; it never widens convolutions, patches,
attention, frozen residual modules or classifier dimensions.
The existing exact source can/grow functions still make final
admissibility decisions and remain unchanged.

The synchronous V07-both scope predicts both MLP expansions;
V08 scheduled growth predicts only the chosen active site.
The forecast returns unknown for nonstandard dtype/layout
and then retains the signed postconstruction cap. Initial
source backbone restoration/allocation remains necessary to
know model payload; optimizer/activation VRAM peaks are not
forecast. New optional cap semantics are separate from
the original uncapped 14 native model configurations.


## Native multi-seed experimental identity closure

The separate `audit_visual_multiseed_worker_pairs` function
subdivides the authoritative 19-site × 2-controller
official-image worker matrix by the real
`SuitePlan` `RunIdentity.seed`, not a manually invented
reporting key. The fixed and growth jobs must agree on
dataset ID, split source checksum, seed, architecture ID and
growth site before any optimizer budget comparison.
The existing zero-budget V01 legacy uncertainty remains
explicit; no historical model or method is modified.
The new CLI flag to the existing dataset driver defaults
to one seed and leaves all prior output names and formats
unchanged.

Complete provenance-based multi-seed *execution* is
not a scientific evaluation unless the official full
training dataset, matched resources, multiple seeds
and confidence intervals are also available.
