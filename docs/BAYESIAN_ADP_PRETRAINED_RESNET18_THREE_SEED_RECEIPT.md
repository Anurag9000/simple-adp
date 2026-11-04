# Published ImageNet-pretrained ResNet18: three matched real Fashion-MNIST seeds

**Executed source revision:** `dd3b1f84318cd32681756fbb5eb461e314dfef19`.

**Completed GitHub Actions run:**
https://github.com/Anurag9000/simple-adp/actions/runs/41853189052

**Artifact:** `13335508598`,
`real-fashion-resnet18-v01-3seed-loss-vs-variance`.

**Provenance and source-model authentication:**

- Official torchvision `ResNet18_Weights.IMAGENET1K_V1`.
- Published source weights URL:
  `https://download.pytorch.org/models/resnet18-f37072fd.pth`.
- Locally serialized complete pretrained state dict SHA-256:
  `441cd3eb2e23b5693285560605a585c85bd54f89f208179c4d38369abfcd01f9`.
- Frozen encoder-state SHA-256 across **all nine jobs**:
  `c747d20ed87f67d619438f6f072235eb69db114df73bb89481f23e66bacd9601`.
- Model/library versions in the source receipt:
  Torch `2.15.0+cu132`, torchvision `0.29.2+cu132`,
  Python `3.11.17` on a CPU runner.
- Verified comparison signature:
  `65d2016bb38e5bd1cf07b5c122d788ae7a606435afae2e2d31998ee74c11`.
- Official Fashion-MNIST source, 300 training, 100 validation and
  100 held-out test observations per derived seed.
- Three treatment routes per seed: fixed width 8, original loss-trigger
  ADP initialized at width 8, and shadow-BGD-variance ADP initialized
  at width 8. The optimizer/model/source implementation is shared.
- These were actual pretrained-source models, **unlike** the separate tiny
  frozen-random V02/V03/V04/V07 visual worker proof.

## Real held-out test measurements (descriptive only)

| Seed | Fixed width / accuracy | Original ADP width / accuracy | Shadow-BGD width / accuracy |
|---|---|---|---|
| 11 | 8 / 0.28 | 10 / 0.54 | 8 / 0.30 |
| 23 | 8 / 0.38 | 10 / 0.57 | 8 / 0.38 |
| 37 | 8 / 0.35 | 10 / 0.57 | 8 / 0.34 |
| **Mean** | **0.3367** | **0.5600** | **0.3400** |

Mean paired accuracy difference, shadow-BGD minus original loss ADP:
`−0.2200` (−22 percentage points), conditional on the three
seed-specific data partitions.

All nine `result.json` members have been independently SHA-256
verified against the comparison summary. Every seed's three arms share
the same official source split digest, while split hashes differ across
seeds (expected due to different partitions).

| Seed | Matched official-split SHA-256 |
|---|---|
| 11 | `254e778699fba412b97528f9fae8b9abc1f940481f903235c4114a187d63ce54` |
| 23 | `9c94dbb67a260a8813543e60d01b4fd012c5cf29898e96936d3398e9943d823c` |
| 37 | `c4cb04c9f31954e5cd81ebe45c0a955327591ae06b971284196f9f819fbdf464` |

## Scope and nonclaims

The original loss-triggered method selected width 10 in each seed;
the shadow-BGD variant did not expand beyond width 8. Its lower
observed accuracy does not prove Bayesian uncertainty-based growth is
inherently inferior. This was a bounded source-faithful smoke study,
not a large-data search with matched capacity, matched total
floating-point compute, posterior calibration or cross-dataset statistics.

The result is source-faithful for the ResNet18 V01 encoder only; it does
**not** certify published-source V03 ViT or V04 ConvNeXt matched Bayesian
pairs. Their pretrained-source one-arm workflows, where present, remain
separate evidence.
