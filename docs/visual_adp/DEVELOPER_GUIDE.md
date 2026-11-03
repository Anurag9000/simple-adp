# Developer guide: V01 frozen encoder head

Available after installation: from simple_adp import VisualADPEstimator

```python
import torch
from torch import nn
from simple_adp import VisualADPEstimator, SearchSettings

encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten())
trainer = VisualADPEstimator(
    encoder, encoder_id="toy-conv", weights_id="random-seed-1",
    dataset_manifest={"source": "my_image_data", "split_sha256": "replace-me",
                      "preprocessing": "normalized-32x32"},
    model_id="ADP-W-RR", pool="vector", expected_resolution=(32,32),
    settings=SearchSettings(init_width=4, init_depth=1, max_width=6,
                            max_candidates=1, max_epochs=1, batch_size=8),
)
# Supply distinct training and validation image tensors and labels:
# trainer.fit(train_images, train_labels, validation_data=(valid_images, valid_labels))
```

For real experiments, replace the toy encoder with a correctly classifier-stripped
pretrained encoder and record the exact weight checksum. This module does not
download pretrained weights. For a CNN returning [B,C,H,W], use pool=spatial_mean.
For ViT token features use pool=cls or tokens_mean. All images must be [N,C,H,W].
Do not pass raw flat image vectors to the module.

Run: python -m pytest -q tests/test_visual_v01.py tests/test_core.py

A checkpoint path has a sibling .vision.json identity sidecar; exact resume
requires both artifacts. Saved models also require the same external encoder
weights and provenance upon VisualADPEstimator.load. Load only trusted head
checkpoint files because legacy TabularADP.load currently uses torch.load
with weights_only=False. The coordinated runner is opt-in for the separate vision families; V01–V05
have selected CPU CI and V06a/b await new CI (see family-specific sections).

## Real dataset preparation (opt-in, torchvision installed)

```python
from simple_adp import load_torchvision_partitions
partitions = load_torchvision_partitions(
    "cifar10", root="./data", download=False, seed=7,
    train_limit=128, validation_limit=64, test_limit=64,
)
x_train, y_train = partitions.materialize("train")
x_valid, y_valid = partitions.materialize("validation")
x_test, y_test = partitions.materialize("test")  # evaluate only after search
# Supply partitions.manifest to VisualADPEstimator(dataset_manifest=...).
```

Use pip install -e '.[vision]' to enable torchvision; downloads are disabled
unless explicitly requested. Validation is drawn only from the official train
group, with deterministic classwise partitioning; the official test group
remains disjoint and is never supplied to candidate selection. Data materialization
returns [N,C,H,W] float32 in [0,1]; apply any backbone-specific mean/std
normalization explicitly and record it in a revised dataset manifest.

## Opt-in coordinated V01 runner (experimental)

Installed provider `vision` and model adapter `vision_adp` are registered only
if `SIMPLE_ADP_ENABLE_VISION=1`. The existing default tabular suite has not
been changed. The built-in benchmark vision backbone is a deterministic
**frozen randomly initialized tiny CNN**; it is NOT a pretrained ResNet/ViT
benchmark, and cannot establish vision ADP accuracy gains.

A CPU-only bounded smoke experiment (requires no torchvision/data downloads):

```bash
SIMPLE_ADP_ENABLE_VISION=1 simple-adp-suite \
  --dataset-suites vision:toy \
  --set 'tasks=["vision"]' \
  --set 'max_epochs=1' \
  --set 'max_candidates=1' \
  --set 'repeat_count=1' \
  --set 'missing_present_task_repeats=1' \
  --set 'model_seeds=[0]' \
  --set 'run_hardness_screen=false' \
  --set 'cpu_only=true'
```

The model IDs are the original 14 native configs, with source generic search
profiles `faithful` and `corrected` when applicable. Each nonfixed route has
a matched variance-stagnation gate job with the same split and RNG seed.
`MLP-Fixed` does not grow. Vision and tabular providers and adapters do not
cross-run. Run `pytest -q tests/test_runner_vision.py` for opt-in worker tests.
To use downloaded real datasets, install `simple-adp[vision]`, set
`SIMPLE_ADP_VISION_ROOT` to an existing torchvision dataset cache and select
`vision:cifar10`, `vision:cifar100` or `vision:fashion_mnist`. Dataset
downloads remain disabled by the runner. `SIMPLE_ADP_VISION_{TRAIN,VALID,TEST}_LIMIT`
can bound per-partition samples; full metadata/checksums are signed into jobs.

**Scope caution:** This runner route is V01 fixed-encoder random-weight
experimental control only; V02–V11 interior MLP growth is not yet wired.

## Concrete torchvision frozen backbones (no model downloads)

```python
from simple_adp import create_frozen_vision_encoder, VisualADPEstimator

encoder, info = create_frozen_vision_encoder(
    "resnet18", weights_path="./my_exact_source_state_dict.pth",
    expected_weights_sha256="the sha256 of the local file",
    normalization="imagenet",
)
# info carries the weights-file digest and frozen-encoder state digest.
# Load full torchvision-source model state_dicts before its classifier is
# removed. A checkpoint file alone is *not* proof of actual pretraining:
# the caller must independently document training data and provenance.
# Pass encoder_id=info["architecture_id"], weights_id=info["weights_id"],
# pool=info["pool"], and a validated dataset manifest to VisualADPEstimator.
```

Supported architecture identifiers: `resnet18`, `convnext_tiny`,
`vit_b_16`. The ViT-B/16 factory rejects any non-224x224 resolution.
ResNet18/ConvNeXt-tiny accept >=32-pixel spatial dimensions under their
native model constraints. No downloaded weights are fetched; if no
`weights_path` is supplied, initialization is deterministic random and is
explicitly labeled as such. Pretrained models using the `imagenet`
normalization expect input pixel values [0,1] and normalize internally.
V01 still trains/grows only the dense classifier head, not source CNN/ViT
blocks or native FFNs.

## Research-only V02 inner-CNN residual adapter candidate

```python
from simple_adp import ResidualAdapterSpec, VisualResidualAdapter
spec = ResidualAdapterSpec(input_channels=3, backbone_channels=6,
                           local_dim=4, classes=2,
                           image_height=8, image_width=9)
model = VisualResidualAdapter(spec, (3, 3))
# Full-image forward: model(image_tensor_NCHW) -> [N, 2]
# Only local MLP coordinates grow; the CNN/post modules stay exactly frozen.
expanded = model.grow_width(site_id="adapter:block1",
                             operator="round_robin", increment=1,
                             max_width=5)
# Source StageTrainer(task, settings).train_stage(...) supports this model;
# full canonical 14-method search and coordinated V02 jobs are NOT yet wired.
```

See `tests/test_visual_inner_v02.py` for full-image gradient and local
mutation, frozen-state, depth phase and StageTrainer checks.

## V02 dedicated composite controller (experimental; five configurations)

```python
from simple_adp import VisualResidualSearch, ResidualAdapterSpec, SearchSettings
spec = ResidualAdapterSpec(input_channels=3, backbone_channels=4,
                           local_dim=3, classes=2, image_height=8, image_width=8)
search = VisualResidualSearch('ADP-W-RR', spec,
    settings=SearchSettings(init_width=2, init_depth=1, max_width=4,
                            max_candidates=1, max_epochs=1, device='cpu'))
# search.fit(train_nchw, train_labels,
#            validation_data=(valid_nchw, valid_labels),
#            checkpoint_path='run-v02.ckpt')
# predictions = search.predict(test_nchw)
```

Supported: LGND-W-Greedy, ADP-W-RR, ADP-W-All, ADP-D-Dcur,
MLP-Fixed (requires `fixed_grid`). Refuses other methods. Its CheckpointStore
signature covers exact train/validation bytes, method defaults, spec,
settings and target encoding. Do not reuse checkpoints for changed image data
or settings. Untrusted torch checkpoint files must never be loaded.

## Opt-in central runner V02 (experimental)
`SIMPLE_ADP_ENABLE_VISION=1` now registers `vision_inner` alongside V01. The
existing opt-in `--dataset-suites vision:toy --set 'tasks=["vision"]'` command
selects both families; V02 routes are `V02:LGND-W-Greedy`,
`V02:ADP-W-RR`, `V02:ADP-W-All`, `V02:ADP-D-Dcur`,
`V02:MLP-Fixed`, plus 4 original-vs-variance pairs. No other inner-site
method is scheduled. Tests cover split isolation and runner execution; CI status
must be verified before treating this path as production-ready. Both this
experimental route and V01's default tiny frozen CNN are controlled baselines,
not a pretrained-ResNet/CIFAR benchmark.

## V02 progressive depth-only D1
`VisualResidualSearch('ADP-D-D1', spec, settings=...)` uses true progressive
width-one depth insertion rather than Dcur insertion. Intermediates
`[3,1]`, `[3,2]` are trained on full images before evaluating `[3,3]`;
they are not separately scored for global-best architecture selection. The
per-batch authoritative checkpoint captures intermediate-stage position,
optimizer/scheduler and RNG for interruption safety. V02 `ADP-D-D1` is also
available as an opt-in `vision_inner` coordinated runner route.

## V02 full 14-configuration source taxonomy (selected CPU CI verified)
All 14 native method IDs are available in the opt-in V02 controller with source
`faithful` and `corrected` profiles for the eight phase-search variants:
```python
search = VisualResidualSearch('ADP-WD-RR-Dcur', spec,
    settings=SearchSettings(init_width=2, init_depth=1, max_candidates=2,
                            max_epochs=1), generic_profile='corrected')
```
The explicit `generic_profile` is part of the signed checkpoint identity.
The `faithful` profile deliberately reproduces the source score-update behavior,
including setting the reported score to the last candidate even when the
selected snapshot remains earlier. For interpretable best-score comparisons,
prefer `corrected`. With `SIMPLE_ADP_ENABLE_VISION=1`, the coordinated
suite enumerates the method/profile jobs and matched variance gates. These generic routes passed selected synthetic CPU CI (259 tests), but
not real-image benchmarks.

### Latest verification
The complete 14-method V02 controller and opt-in runner passed 259 selected
CPU tests: https://github.com/Anurag9000/simple-adp/actions/runs/41394281464.
The `MLP-Fixed` route never grows; dedicated D1 performs real intermediate
optimization; the eight generic methods have separately signed profile routes.

## Experimental native V03/V04 coordinated jobs (pending CI)
Enable the existing vision adapter registration with
`SIMPLE_ADP_ENABLE_VISION=1` and explicitly request `tasks=["vision"]`.
V03 routes use `V03:ADP-W-RR` / `V03:ADP-WD-RR-Dcur@corrected`;
V04 uses `V04:` counterparts. They share the same `vision:toy`,
`vision:cifar10`, `vision:cifar100` and `vision:fashion_mnist`
provider contracts, train/validation/test isolation and native search
settings. Concrete runner variants are small random-initialized ViT-style
(token FFN site `ffn:block0`) or ConvNeXt-style (channel FFN site
`channel:stage0.block0`). They are **not pretrained torchvision source
models** and no accuracy improvement is claimed. Two new worker CI tests
exercise fixed tensors, matched variance gate and generic route execution.

## V05 PoolFormer opt-in image jobs (pending CI)
The `vision_v05` adapter enumerates `V05:<source method>` IDs, with separate
`@faithful`/`@corrected` profiles on eight generic methods and a native
loss-vs-shadow-BGD variance pair per adaptive route. Use the same documented
`SIMPLE_ADP_ENABLE_VISION=1`, `tasks=["vision"]`, and `vision:toy`/
CIFAR/Fashion-MNIST suite selectors as V01–V04. The model is a tiny
random-initialized PoolFormer-style structure; no pretrained PoolFormer or
validated accuracy improvement is claimed. CI status pending.

## V06a and V06b shift-MLP jobs (pending CPU CI)
The installed opt-in image job runner exposes distinct `V06a:` / `V06b:`
route labels and fixed sites `channel:s2.block0` / `channel:axial.block0`.
All 14 canonical source methods, two profiles on eight generic variants,
and matched shadow variance counterparts follow the V03–V05 identity
protocol. `S2SpatialShift` and `AxialSpatialShift` are exported standalone
for inspecting boundary/grouping semantics. Their concrete frozen random
tiny backbones are not pretrained or faithful end-to-end published source
models. New CPU tests and real-dataset benchmarks remain unverified.

### Verified V06 family tests
V06a/V06b native shift/growth/restart and worker tests passed 288 selected
CPU checks in https://github.com/Anurag9000/simple-adp/actions/runs/41399092726.
Only synthetic controlled architectures were tested. Real-data/pretraining
benchmark claims are expressly excluded.

## V07 independently selectable MLP-Mixer sites (pending CI)
The opt-in worker registers `vision_v07_token` and `vision_v07_channel`
under `SIMPLE_ADP_ENABLE_VISION=1`, with 22 method/profile routes and 21
matched native variance criteria per site/split/seed/repeat.
`MixerSpec` accepts exactly `token:block0` or `channel:block0`, and
`VisualMixer` grows the selected local MLP while the other remains frozen.
The full image composite includes both residual MLP branches. A single
joint-site candidate and pretrained weights are **not** implemented. Run
`pytest -q tests/test_visual_mixer.py tests/test_runner_vision.py` for
bounded CPU coverage once installed. CI results remain pending.

## V07 synchronous-both scope (pending CI)
`MixerSpec(site_id="both:block0")` selects a separately named synchronous
site policy. It trains and grows both local token and channel FFNs together;
one proposal means a simultaneous operation on two distinct MLP stacks,
with a joint neuron cap and two saved RR cursors. The opt-in adapter is
`vision_v07_both`. Match `vision_v07_token` and `vision_v07_channel`
for controlled site-scope comparisons; each has 22 source/profile jobs and
21 matched variance counterparts. Do not call synchronous-both a source
single-site ADP method or an implementation of V08 multi-stage search.

## Experimental V10 ResMLP and V11 GFNet native FFNs (pending CI)
Under `SIMPLE_ADP_ENABLE_VISION=1`, separately registered adapters
`vision_v10` / `vision_v11` enumerate the source method taxonomy and
matched native variance gate. Sites are `channel:resmlp.block0` and
`channel:gfnet.block0` respectively. Both accept fixed NCHW inputs from
the existing `vision:toy` and optional CIFAR/Fashion-MNIST providers.
ResMLP's token-mixing matrix and affine are fixed; GFNet's spectral filter
is fixed complex64 with exact spatial FFT shape. Only the real channel
MLP grows. New bounded tests cover actual math, exact restoration, source
methods and worker pairs. No pretrained published checkpoint is provided.

## V08 independent early/middle/late adapter jobs (pending CI)
Opt into vision to register the separate `vision_v08_early`,
`vision_v08_middle`, and `vision_v08_late` model adapters.
Source methods/profile and paired BGD-shadow variance jobs are identical
to the earlier single-site families. The full image always flows through all
three fixed CNN stages and three real residual MLPs; only the declared
stage's `C -> hidden -> C` MLP is trainable and expandable. Use
`MultistageSpec(..., site_id="adapter:early")` and
`VisualMultistage` for the standalone path. No joint or adaptive site
policy has been installed as part of V08 yet.

## V09 strict bank of local residual adapters (pending CI)
Opt into `vision:toy` and `tasks=["vision"]` to register
`vision_v09_bank0`, `vision_v09_bank1`, `vision_v09_bank2`.
The standalone API is `BankSpec(...,site_id="bank:0",bank_size=3)` with
`VisualAdapterBank` through `VisualResidualSearch`. The fixed hard router
receives only frozen backbone features, selects exactly one bank member per
image, and has no trainable parameters. The candidate only changes its
selected local MLP. Result JSON retains `router_kind` and the train/test
member-count vectors (they should be compared with class labels only on
permitted training/validation data, not used to leak test labels into routing).
The three bank jobs have 43 original/profile/variance variants apiece; none
are real pretrained or learned-router science experiments. Tests pending CI.

## V08 predefined early/middle/late site-rotation ablation (pending CI)
The opt-in `vision_v08_schedule` adapter declares one separate family of
22 original source/profile and 21 matched native variance jobs.
`ScheduledMultistageSpec` fixes both the backbone interface and stage order.
`VisualMultistageScheduled` rotates *site selection* by the completed growth
proposal index; its underlying width/depth source method still operates only
on the active site. The worker's result.json includes
`site_schedule_kind="predeclared-early-middle-late-rr-v1"` and an explicit
per-candidate `site_schedule_trace`. Per-site hidden widths/weights and
within-local-MPL RR cursors are signed. This is **not** the canonical ADP-W-RR
within-layer selection, and no adaptive learned controller is implied.

### Fixed V08 schedule verification
The `vision_v08_schedule` family passed bounded full-image synthetic CPU
CI: https://github.com/Anurag9000/simple-adp/actions/runs/41415527242.
The run restored each local MLP's weights/cursor after a middle-site eviction
and exactly reproduced the baseline trajectory. This is a fixed site policy,
not a validation-selected/adaptive site router.

## Inspect the complete opt-in vision job matrix without training
Run `pytest -q tests/test_runner_vision.py -k exact_817_paired_jobs`.
This is a no-training manifest dry-run, with 19 distinct site-policy
families, 22 original source/profile routes per family, 21 matched variance
counterparts per family, and a total of 817 job identities per selected
single split/seed/repeat on `vision:toy`. Real dataset providers are
supported separately and are not automatically executed by the test.

### Route manifest verification
The complete synthetic opt-in plan count was validated by the selected
`355 passed` Actions run:
https://github.com/Anurag9000/simple-adp/actions/runs/41416422457.
This checks job identities and pair provenance without training 817 models.


## Local official-image V01 experiment command (experimental)

Install \`pip install -e '.[vision]'\`. Place CIFAR-10, CIFAR-100 or
Fashion-MNIST official torchvision data under \`--root\`; downloads are disabled
unless you explicitly set \`--download\`. Provide a *complete* local
torchvision model state_dict and preferably its file SHA256. For example:

\`\`\`bash
sha256sum /models/resnet18_full_state_dict.pt
simple-adp-vision-experiment \
  --dataset cifar10 --root /data/vision \
  --out results/cifar10/resnet18_adp_w_rr_seed11 \
  --architecture resnet18 --weights /models/resnet18_full_state_dict.pt \
  --weights-sha256 REPLACE_WITH_ACTUAL_SHA256 \
  --model-id ADP-W-RR --seed 11 \
  --train-limit 500 --valid-limit 100 --test-limit 100 \
  --max-epochs 5 --max-candidates 2 --batch-size 64
\`\`\`

Pretraining is *not* inferred from an arbitrary local checkpoint. Such weights
are recorded as \`unverified_local_state_dict\` with \`pretrained=null\`.
For an intentional random feature control, omit \`--weights\` and supply
\`--allow-random-backbone\`; random source weights are never labeled pretrained.

The deterministic input bridge repeats grayscale into three channels and
bilinearly resizes Fashion-MNIST to 32×32 for ResNet18/ConvNeXt-Tiny.
ViT-B/16 receives 224×224 bilinearly resized images. ImageNet normalization
is handled by the frozen backbone wrapper and is recorded separately.
Candidate selection uses **only** fit/validation subsets of the official
training source; official test is materialized and measured after fitting.

Each isolated \`--out\` directory contains \`manifest.json\`,
\`estimator.checkpoint.pt\`, \`estimator.model.pt\`, its \`.vision.json\`
identity sidecar, and \`result.json\`. The manifest includes source sample
indices, split signatures, raw data hashes, encoder weight/state hashes,
preprocessing, method/profile and settings. A mismatch rejects resumption;
completed results are reused only when model and identity SHA256 match.
The V01 ADP head alone is trained/expanded; backbone weights remain frozen.

The CLI is independent of the existing 817-job opt-in planner and does not
automatically execute 817 jobs. The initial tests use **synthetic image
fixtures**, not actual CIFAR/Fashion-MNIST files or verified pretrained weights.

### Per-job candidate history and confusion evidence

New V01 experiment runs also write `search_history.json` containing the
native candidate-by-candidate ADP architecture-search records, protected by
a SHA256 in `result.json`. The result includes `class_order`,
`confusion_matrix` (rows: true classes; columns: predicted classes),
and parameter counts for the frozen source encoder versus selected ADP head.
This evidence is computed strictly after fit for the official test partition
and is not fed back to candidate selection. Reopening a completed result
verifies both model/identity checksums and candidate history checksum.
Runs created by the earlier CLI version may not contain the new artifact;
use a fresh output directory to regenerate reporting without altering a
previously signed result.

## Signed fixed/loss/variance V01 real-image comparisons

Use the separately installed `simple-adp-vision-compare` command for a
reproducible **matched three-way comparison** of the frozen fixed-width MLP
head, canonical `ADP-W-RR` loss trigger, and `ADP-W-RR` shadow-variance
stagnation trigger:

```bash
simple-adp-vision-compare \
  --dataset fashion_mnist --root /data/vision --out results/v01_paired \
  --architecture resnet18 --weights /models/resnet18_full_state_dict.pt \
  --weights-sha256 YOUR_ACTUAL_SHA256 \
  --seeds 11,23,37 --fixed-width 16 \
  --train-limit 1000 --valid-limit 200 --test-limit 200 \
  --epochs 4 --candidates 2 --init-width 16 --max-width 24 \
  --batch-size 32 --device cuda
```

The CUDA argument is intended for a machine with working CUDA; the current
GitHub Actions evidence is CPU-only. Downloads are disabled unless
`--download` is explicitly supplied. Random-source studies must explicitly
set `--allow-random-backbone` and cannot be described as pretrained.

The comparison predeclares all seed-specific partitions and creates three
separately signed result/checkpoint folders per seed:

```
results/v01_paired/
  comparison_manifest.json
  comparison_summary.json
  runs/
    seed11/
      fixed/
      adp_loss/
      adp_variance/
    seed23/...
    seed37/...
```

Every per-job directory contains `manifest.json`, `result.json`,
`search_history.json`, `estimator.checkpoint.pt`, and the selected
model/identity sidecar. Both ADP criteria share identical source image
indices, weights checksum, frozen encoder SHA256, normalized input mapping,
and settings other than the criterion. The fixed control shares the same
source and training budget but has no growth operator. Each per-run result
checksum is pinned in the final comparison summary.

A killed comparison can be restarted with the **same command and output
directory**. Completed jobs are checksum-checked and reused; missing jobs
continue with their own signed mid-batch checkpoints. Any mismatch in
seeds, limits, source data, weights, settings or methods is rejected.
Comparison results report all official-test metrics descriptively and do
**not** choose a winning method based on the official test set.

An actual single-seed **official pretrained-source Fashion-MNIST**
comparison completed successfully:
https://github.com/Anurag9000/simple-adp/actions/runs/41462840053.
Its train/validation/test sample sizes were only 300/100/100. These numbers
are evidence of real-data execution, not benchmark-quality conclusions.

## V04 real source ConvNeXt inner-FFN experiments

The optional strict source module is `simple_adp.visual_source_convnext`.
It needs local **complete** torchvision ConvNeXt-Tiny weights (not a stripped
feature-only checkpoint). With explicit fit/validation image subsets:

```python
from simple_adp.config import SearchSettings
from simple_adp.visual_source_convnext import SourceConvNeXtSpec, file_sha256
from simple_adp.visual_search import VisualResidualSearch

source_path = "/models/convnext_tiny_full_state_dict.pt"
spec = SourceConvNeXtSpec(
    classes=10, image_height=28, image_width=28, input_channels=1,
    weights_path=source_path, weights_sha256=file_sha256(__import__("pathlib").Path(source_path)),
    fixed_seed=11,
)
search = VisualResidualSearch(
    "ADP-W-RR", spec,
    settings=SearchSettings(
        init_width=384, init_depth=1, max_width=385, max_depth=2,
        max_epochs=1, max_candidates=1, batch_size=8, device="cpu",
    ),
)
search.fit(train_images, train_labels,
           validation_data=(valid_images, valid_labels),
           checkpoint_path="results/v04/estimator.checkpoint.pt")
```

`train_images` and `valid_images` must be raw float32 NCHW pixels in
[0,1] from **disjoint official-training-source indices**, with class
labels encoded from training data only. For grayscale Fashion-MNIST the
wrapper repeats channels and bilinearly resizes to at least 32x32 before
ImageNet normalization. The model *does not download* checkpoints.

The initial hidden width **must be exactly 384** for a mathematically exact
source FFN transplant. The source feature network is frozen, while the
original in-block FFN and new task head are trainable. Subsequent native
width/depth search and checkpoint resume are source-site scoped. The
pretrained-source real-data example is
`scripts/vision_pretrained_convnext_smoke.py`; CI execution succeeded in the separate real-pretrained source run
https://github.com/Anurag9000/simple-adp/actions/runs/41466060451.
The selected in-block FFN grew from 384 to 385 on 40/20/20 official
Fashion-MNIST images; this is a real integration smoke, not a benchmark. No source-full 14-method
scientific benchmark claim is made.

## Optional coordinated source-faithful V04 job family

The coordinated runner can now additionally opt in to the **original
pretrained-model-compatible ConvNeXt-Tiny internal FFN** in a separately
labeled `vision_v04_source` family (distinct from the 19 synthetic source
controls). This requires two opt-in flags plus the exact local full
torchvision checkpoint and its SHA256:

```bash
export SIMPLE_ADP_ENABLE_VISION=1
export SIMPLE_ADP_ENABLE_V04_SOURCE=1
export SIMPLE_ADP_V04_SOURCE_WEIGHTS=/models/convnext_tiny_full_state_dict.pt
export SIMPLE_ADP_V04_SOURCE_SHA256=REPLACE_WITH_REAL_SHA256
```

The saved source path must also be mounted at the same absolute location
in every worker. The planner verifies file checksum and includes its prefix
in the route identity; workers verify the full checksum again before
training. The 43 source V04 routes per selected task/split/seed consist
of 22 original model/profile combinations and 21 variance counterparts,
with no variance clone of `MLP-Fixed`. No source route is registered
unless `SIMPLE_ADP_ENABLE_V04_SOURCE=1`; the previous 817-planned-job
matrix is preserved by default.

Set **finite nonzero** `max_candidates` and `max_epochs`, and
`max_width >= 384` and `max_neurons >= 384` in central runner settings.
For a genuine widening experiment, use `max_width >= 385`. All source
models begin with width 384/depth 1; do not substitute the general
runner's small width-to-depth grid for a pretrained inner FFN. Full
checkpoints can exceed 100 MB per job, so account for disk and queue pressure.
Standalone published-pretrained weights execution and central runner
planning/execution have independent verification records.

## Original ViT-B/16 internal FFN (V03 source) central runner

Source V03 is an **additional** opt-in route, not the existing tiny V03
control. A complete original torchvision ViT-B/16 source checkpoint must
already exist locally; random checkpoints must never be labeled pretrained.

```bash
export SIMPLE_ADP_ENABLE_VISION=1
export SIMPLE_ADP_ENABLE_V03_SOURCE=1
export SIMPLE_ADP_V03_SOURCE_WEIGHTS=/models/vit_b16_full_state_dict.pt
export SIMPLE_ADP_V03_SOURCE_SHA256=REPLACE_WITH_ACTUAL_64_CHAR_SHA256
```

The coordinated family is `vision_v03_source`; all planned route labels
contain the exact source SHA256 prefix. It predeclares 22 source method
and generic-profile original jobs and 21 matched variance counterparts
per split/seed. Each worker rechecks the complete file SHA256, preserves
the initial **3072-neuron** first transformer FFN, performs only native
ADP local growth at the source site, and saves source-hashed selected
inference artifacts. The target classifier is trainable but not growable.

Choose finite `max_candidates` and `max_epochs`, with
`max_width >= 3072` and `max_neurons >= 3072` in the coordinator.
Real widening needs `max_width >= 3073`. The complete 86M-parameter
source model plus checkpoint snapshots are memory/disk intensive;
avoid blindly executing every registered job. The existing **817** tiny
visual jobs remain unchanged unless a source gate is explicitly enabled.

Dedicated source-V03 CI:
https://github.com/Anurag9000/simple-adp/actions/runs/41472976307,
and opt-in source-runner CI:
https://github.com/Anurag9000/simple-adp/actions/runs/41473641002.
Both used full original torchvision models with **random** source
weights. The published-pretrained source smoke is a separate workflow
`scripts/vision_pretrained_vit_smoke.py`.

## Official real-image all-controlled-family CPU matrix

The canonical worker-plan integration smoke is available via
`scripts/visual_official_fashion_matrix.py`. It explicitly downloads
**official Fashion-MNIST** into a local isolated directory, opts in to the
`vision` dataset provider, builds the authoritative `SuitePlan`, selects
each of the 19 controlled architecture/site adapters with one `MLP-Fixed`
and one `ADP-W-RR` route, then invokes the saved native
`runner.worker.run_spec` for each. No new ADP optimizer or shadow
search controller is introduced. Source ViT/ConvNeXt pretrained checkpoints
remain separate, explicitly gated workflows.

```bash
python -m pip install -e '.[vision,dev]'
python scripts/visual_official_fashion_matrix.py \
  --workdir /tmp/simple-adp-official-fashion-matrix \
  --train-limit 40 --valid-limit 20 --test-limit 20
```

For unattended GitHub Actions execution, see
`.github/workflows/visual-official-family-matrix.yml`.
`official_fashion_vision_matrix.json` records each **completed or failed**
worker, the exact split SHA256, selected architecture, per-site parameter
partition and actual validation/test metrics. `results/` contains the
independently signed normal worker reports and inference/checkpoint evidence.
A failed run remains incomplete and the script exits nonzero; do not interpret
mere job selection as execution. The one-epoch setup is deliberately unfit for
accuracy or budget-parity claims. Use independently matched training budgets
and multi-seed evaluations for any scientific comparisons.

The core model-level read-only accounting also ships as an installed-package
lazy export `from simple_adp import visual_parameter_accounting`.
For each whole-image model, pass its actual `growth_targets`, or for a
frozen V01 encoder include the encoder and native head together in a
temporary `nn.ModuleDict`. The returned `linear_macs_per_mlp_invocation`
is **not** whole-model FLOPs.


## Inspect partial full-composite inference cost and native search budget

```python
from simple_adp import visual_linear_conv_macs

# model: selected composite image model, never just its growth target
# image: one float NCHW image already placed on the model's device
probe = visual_linear_conv_macs(model, image[:1])
print(probe["observed_linear_conv_macs_per_image"])
print(probe["whole_model_flops"])  # None: intentionally NOT total FLOPs
```

This read-only diagnostic executes the actual model with hooks for
`nn.Linear` and `nn.Conv2d`, restores all training flags, RNG and
parameter trainability, and does not change model weights or buffers
for the supported standard PyTorch operations. It is a **partial** MAC
trace. For transformer attention, FFT and pooling models, the excluded
operation list must be read alongside it; no claim that these operations
were measured is justified.

The coordinated visual worker results include
`inference_compute`. Native V02–V11 and source V03/V04 results also
include `training_budget`; candidate history records measured
optimizer steps/epochs, candidate parameter counts and excludes D1
warmup stages from acceptance decisions while counting their cost.
V01's original tabular head controller remains untouched and does not
claim a native composite search-budget field.


## Visual validation/test calibration

The opt-in vision worker outputs add the fields
`top_label_ece_10` and `multiclass_brier` inside both
`validation_metrics` and `test_metrics`. They are post-selection,
not replacement architecture-scoring criteria. The source
`ClassificationTask` still owns the training label mapping and
cross-entropy validation objective.

For stand-alone, explicitly fitted classification task experiments:

```python
from simple_adp import visual_calibration_metrics
metrics = visual_calibration_metrics(task, labels, predicted_probabilities, bins=10)
```

`predicted_probabilities` must be [N,C] in the fitted
`ClassificationTask` order and sum to one. Small-image smoke ECE/Brier
figures are merely integration checks. No claim of calibrated confidence
or model advantage is supported by one seed and 20 held-out images.

The native composite search now stores `training_budget` in its
signed checkpoint and worker summary, and per-candidate
`trained_epochs`, `optimizer_steps` and parameter counts in its
history. It counts D1 catch-up stages in total computation without
treating them as independent accepted candidates. A resumed historical
checkpoint without prior budget provenance is marked
`complete=false`; this prevents pretending unobserved training
steps have been reconstructed.


## Attention MACs and conservative fixed/RR job-budget audits

Visual native V03 or source ViT models additionally report
`inference_compute.analytical_multihead_attention_macs_per_image`
and `inference_compute.analytical_multihead_attention_modules`,
which include forward call count and separate dense projections
versus score/value MACs. The analytically estimated stream is
kept separate from observed module-hook Linear/Conv2d MACs,
preventing double counting. Both are partial inference arithmetic
only; even the combined total is **not model FLOPs**.

The real-image matrix script now produces
`official_fashion_vision_matrix.json` with
`fixed_vs_growth_budget_audit`. To analyze a different complete
set of worker rows without retraining:

```python
from simple_adp.visual_budget_audit import audit_visual_worker_pairs

audit = audit_visual_worker_pairs(rows, expected_adapters=all_visual_adapters)
for pair in audit["paired_budget_evidence"]:
    print(pair["adapter"], pair["same_optimizer_step_count"],
          pair["growth_to_fixed_step_ratio"])
```

`same_optimizer_step_count=None` indicates unknown/partial or
mismatched provenance, not equality. In particular the historical
V01 `VisualADPEstimator` head has no native
`VisualResidualSearch.training_budget`, and remains unknown.
`full_training_flops_matched=None` always: an optimizer step does
not have identical work across changed architectures.


## Original official RGB CIFAR image worker matrix

The single authoritative `SuitePlan`/`run_spec` 38-job controlled
vision smoke driver supports three official torchvision inputs without
forking the ADP training implementation:

```bash
# Historical default and historical artifact/schema preserved
python scripts/visual_official_fashion_matrix.py \
  --dataset fashion_mnist --workdir /tmp/visual-fashion \
  --train-limit 40 --valid-limit 20 --test-limit 20

# Actual RGB 32x32 CIFAR-10; 19 controlled families x fixed/RR
python scripts/visual_official_fashion_matrix.py \
  --dataset cifar10 --workdir /tmp/visual-cifar10 \
  --train-limit 40 --valid-limit 20 --test-limit 20

# CIFAR-100; every one of its 100 classes requires >=1 example in
# EACH of the three partitions (prefer substantially more in science)
python scripts/visual_official_fashion_matrix.py \
  --dataset cifar100 --workdir /tmp/visual-cifar100 \
  --train-limit 200 --valid-limit 100 --test-limit 100
```

Dataset download is an explicit step in this standalone integration
driver; the underlying historical `VisionDatasetProvider` retains
`download=False` and never begins implicit network access. The
official test group stays inaccessible until fitting and validation
selection complete. The output file follows dataset identity:
`official_fashion_vision_matrix.json`,
`official_cifar10_vision_matrix.json` or
`official_cifar100_vision_matrix.json`.
The **original Fashion-MNIST summary format** is preserved for
existing consumers. Each summary includes 38 individual signed worker
results and the conservative `fixed_vs_growth_budget_audit`.

This driver is a smoke test with one seed, tiny class-balanced
subsets, only one epoch per candidate and one adaptive proposal.
It does not compare scientific performance under matched budgets;
full datasets, repeated seeds and source-faithful V05–V11 remain
separate uncompleted work. Real CIFAR-10 workflow:
`.github/workflows/visual-official-cifar-matrix.yml`.


## Model tensor memory accounting and opt-in admission

```python
from simple_adp import visual_model_tensor_payload
from simple_adp.visual_search import VisualResidualSearch

memory = visual_model_tensor_payload(selected_image_model)
print(memory["total_model_tensor_payload_bytes"])
print(memory["logical_bytes_by_dtype"])
assert memory["peak_training_memory_bytes"] is None

# An explicitly signed guard, native local-growth models only.
search = VisualResidualSearch(
    "ADP-W-RR", native_spec, settings=search_settings,
    max_model_tensor_payload_bytes=256 * 1024 * 1024,
)
search.fit(train_x, train_y, validation_data=(valid_x, valid_y))
print(search.search_state_["memory_admission_rejections"])
print(search.search_state_["trained_model_memory_trace"])
```

For coordinated jobs, supply
`RunnerSettings(visual_model_tensor_payload_cap_bytes=...)`.
The field is absent from legacy uncapped run identities and present
in the job signature only when configured. Every normal native worker
returns `model_memory_accounting`; capped jobs also report the
limit, rejected candidate records and complete per-stage memory
traces. The strict admission policy deliberately does **not**
apply to V01's original tabular-head model: a capped V01 worker
fails with a clear error instead of silently skipping the guard.

Caveats: this cap limits only *logical tensor payload after candidate
construction* and before model stage optimization. It does not
prevent proposal construction OOM, autograd activation peaks, source
checkpoint load peaks, GPU caching fragmentation or optimizer
allocation peaks. Do not use this value as a GPU VRAM/whole-training
RAM requirement; include additional headroom for all those
unmeasured costs.


## Zero-candidate-allocation proposal preflight

`visual_memory.projected_tabular_mlp_payload_bytes`,
`visual_memory.prospective_tabular_growth_widths` and
`visual_memory.forecast_visual_growth_payload_bytes` provide
pure structural projections. These calculate standard MLP
parameter and buffer bytes without allocating proposed model
parameters or constructing another full visual backbone.

The opt-in native `VisualResidualSearch` cap now calls these
before `grow_width`, `grow_depth`, the D1 warmup
`rebuild_mlp` or a new fixed-grid model is built.
`memory_admission_rejections` records
`before_candidate_construction=true` only if the structural
formula was sufficient to prove an over-budget proposal.
Unsupported/non-default dtype or unusual shared memory may
fall back to the postconstruction check. Source growth
controllers and their neuron/patience rules remain authoritative;
the forecast is an additional memory gate, not an architecture
search policy.

`tests/test_visual_memory.py` compares projected MLP bytes to
actual rebuilt models (with and without batch norm), covers source
width/depth geometry operators, all 19 controlled visual
architecture/site variants, and monkeypatches growth constructors
to make sure over-cap operations are never called. CI evidence
must be consulted separately; adding tests alone is not a pass.


## Independent official-image model seeds without new controllers

The original `scripts/visual_official_fashion_matrix.py` supports
`--model-seeds N` (positive integer). The default **remains 1** and
keeps the original Fashion/CIFAR summary filename/format. An explicit
larger N passes `model_seeds=(0,...,N-1)` to the existing
`RunnerSettings` and `SuitePlan.build`, which derives
independent signed `RunIdentity.seed` values for the official
partition. Each seed is applied consistently to the fixed and growth
member of every controlled architecture/site pair.

```bash
# 2 seeds x (19 adapter/site variants x 2 controllers) = 76 jobs
python scripts/visual_official_fashion_matrix.py \
  --dataset fashion_mnist --model-seeds 2 \
  --workdir /tmp/visual-multiseed \
  --train-limit 40 --valid-limit 20 --test-limit 20

# 3 seeds x 38 = 114 jobs; still not a full benchmark
python scripts/visual_official_fashion_matrix.py \
  --dataset cifar10 --model-seeds 3 \
  --workdir /tmp/visual-cifar-multiseed \
  --train-limit 100 --valid-limit 50 --test-limit 50
```

The second command is a **usage example**, not a verified completed
run. The two-seed Fashion output is
`official_fashion_vision_matrix_2seeds.json`, leaving the historical
one-seed artifact untouched. Each row includes the actual derived
source runner seed; the summary records the requested seed count,
derived seed IDs, the status of every signed job and a grouped
`fixed_vs_growth_budget_audit`. Rejected/missing workers, duplicate
identities and cross-seed mismatches fail closed.

`audit_visual_multiseed_worker_pairs` independently audits each
run seed before adding budget-evidence counts. Matching optimizer
step numbers within a seed do not guarantee matched training FLOPs,
and two or three tiny smoke runs do not justify confidence intervals
or scientific performance claims. A genuine benchmark requires
the complete official source partition, matched compute and multiple
seeds with separate held-out evaluation.


## Bounded optimizer-step-balancing attempt (fixed vs adaptive)

A new opt-in `--step-balanced` mode uses **existing** native
runner per-route epoch caps without changing an ADP controller:

```bash
python scripts/visual_official_fashion_matrix.py \
  --dataset fashion_mnist --step-balanced --batch-size 16 \
  --workdir /tmp/fashion-stepbalanced \
  --train-limit 40 --valid-limit 20 --test-limit 20

# Independent seeds and optimizer-step attempt may be combined:
python scripts/visual_official_fashion_matrix.py \
  --dataset cifar10 --model-seeds 3 --step-balanced --batch-size 32 \
  --workdir /tmp/cifar-multiseed-stepbalanced \
  --train-limit 300 --valid-limit 100 --test-limit 100
```

The second command is **illustrative only**, not a recorded
completed execution. The mode configures
`RunnerSettings(max_epochs=None, max_epochs_vanilla=2,
max_epochs_adp=1, max_candidates=1, smoke_test=False)`.
Fixed controls train their initial model for at most two epochs.
Native RR trains one initial stage plus a single growth
candidate for at most one epoch each. `--batch-size` is
effective because this mode disables the worker's default
one-batch smoke override. The native `training_budget`
records **actual optimizer steps**, not intended steps, and
the fixed/RR auditor reports any difference.

The single-seed output gets the suffix
`_stepbalanced.json`; combining this with N seeds produces
`_Nseeds_stepbalanced.json`. Both remain separate from
the historical one-seed one-epoch artifacts. V01's original
tabular head has no comparable native stage-budget record
and must still report optimizer work as unknown.

This is only an **attempt to equalize optimizer-step counts**.
Gradient cost per step differs by architecture and model
size; no full-training FLOPs, wall-clock or peak memory
parity is implied. The default one-epoch-per-candidate
one-seed matrix is unchanged.


## Non-invasive real fit process-resource observations

```python
from simple_adp import (
    visual_process_resource_snapshot,
    visual_fit_resource_window,
)

before = visual_process_resource_snapshot("cpu")
# Run native training/architecture search here, without touching test labels.
after = visual_process_resource_snapshot("cpu")
window = visual_fit_resource_window(before, after)
print(window["rss_current_before_bytes"])
print(window["rss_current_after_bytes"])
print(window["new_lifetime_peak_rss_during_fit_bytes"])
assert window["exact_fit_window_peak_rss_bytes"] is None
```

A real CUDA job must pass the model's actual `cuda:N`
device. The memory snapshot reports PyTorch's per-device
`allocated/reserved` counters and the *process-lifetime*
high-water marks, **without resetting them**. The all-CPU
official smoke returns `cuda_pytorch_allocator=None`.
On non-Linux platforms with no `/proc/self/status`,
RSS fields are null. The implementation intentionally
requires no new dependency and preserves checkpoint,
stage-selection, and RNG behavior.

All coordinated V01–V11 and explicit source V03/V04
visual adapter results now include `fit_process_resources`.
`run_spec` copies the report into its signed
`result.json` record. The official image matrix validates
its schema and stores it alongside `training_budget`,
`trained_model_memory_trace` and
`model_memory_accounting`.
This is *fit-call* resource evidence, **not** a precisely
sampled CPU/GPU peak-memory profile and not an OOM cap.
Use short-lived, one-job-per-process runs for the most
meaningful process-lifetime high-water comparisons.


## Fresh-process official visual resource evidence

For a true **per-job process lifetime** RSS high-water comparison,
enable the original signed worker CLI in a new subprocess for
each matrix route:

```bash
python scripts/visual_official_fashion_matrix.py \
  --dataset fashion_mnist \
  --isolated-workers --step-balanced --batch-size 16 \
  --workdir /tmp/visual-process-memory \
  --train-limit 40 --valid-limit 20 --test-limit 20
```

This still uses `SuitePlan.build`, original
`runner.worker.main`/`run_spec`, original signed
per-job JSON specifications, the official dataset
provider, and the native ADP controllers. It does not
replace any optimizer or create a second training
backend. Every route runs
`python -m simple_adp.runner.worker --spec ...`
in a fresh Python process. The parent reads each
`result_dir/result.json` only after the child exits
successfully, verifies the result's signed run identity,
and retains a separate child stdout/stderr log.
Failed child exit or tampered/mismatched identity
fails the matrix job explicitly.

The independent output
`official_fashion_vision_matrix_stepbalanced_isolated.json`
retains the same fixed/RR budget audit but additionally
labels `worker_process_scope=one-process-per-signed-run`.
The per-job `fit_process_resources.pid` is unique for
fresh-process workers. A process VmHWM then pertains to
that child worker rather than earlier matrix jobs. It
still includes interpreter startup, model construction
and data reading in the **same** child, so it is not
an exact fit-stage peak or a model-specific allocation.
The parent may hold official dataset objects; OS
memory of the parent is **not included** in the
child worker's RSS.

This opt-in mode is heavier because it initializes
Python/torch for every job. The existing historical
one-process-per-matrix fast mode is unchanged.
See CI:
`.github/workflows/visual-official-isolated-memory.yml`.


## Paired descriptive CPU RSS audit for isolated jobs

When `--isolated-workers` completes all signed worker
jobs, the matrix now adds
`paired_isolated_process_rss_audit` alongside
`fixed_vs_growth_budget_audit`. It uses the existing
source-identity gate to check that each fixed/growth
pair shares the dataset, official raw split, derived
seed and growth site. For each independently executed
*child process* it then reports endpoint RSS before
and after fit, child lifetime highwater after fit,
and descriptive growth-minus-fixed differences.

Use `visual_resource_audit.summarize_isolated_visual_resource_pairs`
to recalculate the report on an externally stored
matrix. The API **rejects** data not explicitly
recorded with `worker_process_scope=
"one-process-per-signed-run"`, mismatched identities,
unavailable RSS, fake exact fit peaks, or CUDA data
incorrectly treated as CPU RSS. It does not compute
confidence intervals, treat process VmHWM as a
model-specific peak or claim statistical significance.
V01's unknown native optimizer work remains unknown
even when its CPU RSS is measured.


## Manual official CIFAR CPU process-memory runs

The isolated-memory GitHub Actions workflow
`.github/workflows/visual-official-isolated-memory.yml`
also accepts a manual `workflow_dispatch` choice of
`fashion_mnist`, `cifar10` or `cifar100`.
A normal push continues to exercise Fashion-MNIST.
The manual workflow adjusts class-complete subset
limits for CIFAR-100 (200 train/100 validation/100
official test); the other datasets use 40/20/20.
All three use the original native 19-site fixed/RR
matrix under `--step-balanced --isolated-workers`,
retaining signed jobs, nested child logs, process
RSS evidence and the descriptive paired resource audit.

A workflow choice does **not** itself establish
completed CIFAR isolated-memory execution. Cite
dataset-specific results only after verifying its
actual GitHub Actions run and saved artifact.


## Read-only paired accuracy and calibration analysis

Once a complete official worker matrix exists, use:

```bash
python scripts/visual_paired_metrics_analysis.py \
  --matrix /path/to/official_fashion_vision_matrix_2seeds.json
```

The generated sibling `*_paired_metrics.json` contains a
separate descriptive paired report for **each of the 19**
controlled visual sites; it does not pool them. Source artifacts
are never modified and no model trains during analysis.
See `docs/visual_adp/PAIRED_RESULTS_ANALYSIS.md` for the
full metric-direction conventions, exact provenance
validation and why no statistical-significance/accuracy
gain claim is permitted.

The two-seed official Fashion CI now runs that CLI after
the original 76 signed jobs and uploads both the original
worker matrix and the paired report. It has been verified
at GitHub run 41616572651, artifact 13222170910.


### Strong posthoc original result-file agreement

Use `--results-root /path/to/workdir/results` with
`scripts/visual_paired_metrics_analysis.py` to re-read
all original per-worker result JSON files before
producing a paired accuracy/calibration report.
All workers from one signed SuitePlan **share**
a run signature; only worker job IDs are unique.
The analyzer validates the plan signature format,
cross-file uniformity, original data/split/seed/site,
selected architecture, validation/test metrics and
optimizer steps. A deterministic bundle SHA-256
fingerprint is emitted.

This is not an external public-key verification,
and it is not an endorsement of tiny smoke-test
accuracy. The original worker matrix and its
temporary staging path are protected from being
overwritten by the report.


## Trace actual native training forwards, without another forward pass

```python
from simple_adp.visual_search import VisualResidualSearch
from simple_adp.runner.config import RunnerSettings

search = VisualResidualSearch(
    "ADP-W-RR", native_visual_spec,
    settings=your_search_settings,
    track_training_forward_macs=True,  # opt-in and checkpoint-signed
)
search.fit(train_images, train_labels, validation_data=(valid_images, valid_labels))
print(search.search_state_["training_forward_compute"])
print(search.search_state_["trained_forward_mac_trace"])

# Coordinator: only native visual and separate V03/V04 source routes are
# instrumented; V01 original tabular-head training stays unmeasured.
settings = RunnerSettings(
    tasks=("vision",), visual_training_forward_macs=True,
    # ... provide all source dataset/growth/epoch constraints as usual
)
```

For real official-image 19-family fixed/RR integration:

```bash
python scripts/visual_official_fashion_matrix.py \
  --dataset fashion_mnist \
  --step-balanced --observe-training-forwards --batch-size 16 \
  --train-limit 40 --valid-limit 20 --test-limit 20 \
  --workdir /tmp/actual-train-forward
python scripts/visual_paired_metrics_analysis.py \
  --matrix /tmp/actual-train-forward/official_fashion_vision_matrix_stepbalanced_observedtrain.json \
  --results-root /tmp/actual-train-forward/results
```

The flag is **not enabled by default**. False is omitted
from `RunnerSettings.as_dict` and from the visual search
checkpoint signature, preserving historical run and restart
identities. True is carried in signed individual worker
specifications and exact-resume stage checkpoints. The
post-hoc report gains `training_forward_work_audit` only
when native observed work was supplied, while preserving
original classification/calibration metrics.

The observer's context spans only the real `model(xb)`
training call and is detached before gradient backward.
This avoids counting validation or auxiliary shadow
passes, but it also deliberately means backward and
Adam/AdamW work is **not counted**. Comparing the ratio of
these partial forward MACs is stronger than comparing
optimizer steps alone, but cannot establish equal
full-training FLOPs or wall-clock parity.
