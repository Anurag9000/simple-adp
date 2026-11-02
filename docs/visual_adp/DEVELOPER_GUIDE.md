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
with weights_only=False. The coordinated runner is opt-in for V01 only (see below).

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
