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
with weights_only=False. The coordinated runner has not yet been wired to V01.

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
