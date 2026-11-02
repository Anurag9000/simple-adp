# Verified real-image V01 experiment evidence

This record distinguishes a **completed real-data integration smoke** from
benchmark-quality scientific evaluation. The frozen source feature extractor
uses the official torchvision pretrained ResNet18 weight enum. The V01
adaptive dense head is trained on official dataset imagery; no inner
ResNet layer grows. The recorded micro-sample metrics should not be used to
infer generalization performance.

## Run A — Fashion-MNIST, official ResNet18

- **Status:** Verified successful (GitHub Actions).
- **CI run:** https://github.com/Anurag9000/simple-adp/actions/runs/41451103405
- **Source commit:** `cefd74e3b0bf861835437ab830b2101303e8ea1a`.
- **Weights:** torchvision `ResNet18_Weights.IMAGENET1K_V1`,
  `https://download.pytorch.org/models/resnet18-f37072fd.pth`;
  full source state_dict SHA256
  `eff99529cc4a2ad4edac486be5a9e9724e7f4e0ccc0e3b319dd07b7e48306592`.
- **Source:** `torchvision.datasets.FashionMNIST` official train/test.
- **Preprocessing:** uint8/255, grayscale channel repeat, 28×28 to 32×32
  bilinear, ResNet18's ImageNet normalization.
- **Architecture search:** V01 `ADP-W-RR`, loss criterion, maximum 1
  growth proposal, 1 epoch per stage; source encoder completely frozen.
- **Source sample counts:** 40 train, 20 architecture validation (from
  official training partition), 20 official-test images. Stratified
  class-balanced limits, seed 11.
- **Validation:** accuracy 0.15; balanced accuracy 0.15; macro-F1 0.10;
  log loss 2.400973100436234.
- **Official test:** accuracy 0.15; balanced accuracy 0.15; macro-F1 0.095;
  log loss 4.293031086048549.
- **Test provenance:** `external_source_provenance.json` records the
  runtime torchvision weight enum and download URL; local model loader
  separately preserves `pretrained: null` because a local state_dict by
  itself cannot prove training origin.
- **Run signature:** `9da96df9c007cc712de7edbbcae880912f9258a5392476d0a5dfe1cd6dd6cbae`.
- **Artifact:** GitHub Actions uploaded
  `fashion-mnist-official-resnet18-v01-evidence`, artifact ID 13144598582,
  containing `manifest.json`, `result.json`,
  `external_source_provenance.json`, and the signed inference model
  plus identity sidecar (30-day artifact retention at upload).

This was a true **real-image data flow and pretrained-source integration**
run, not a reliable statistical performance estimate. There was no repeated
seed distribution, baseline, confidence interval, hyperparameter selection
study, full source architecture ADP, or GPU comparison.

## Run B — attempted CIFAR-10, not completed

- Run: https://github.com/Anurag9000/simple-adp/actions/runs/41451620431
- **Cancelled** after 30 minutes, at approximately 60.1% of the
  official CIFAR-10 archive download. No CIFAR-10 training or test metrics
  were produced. The run's partial artifact includes only the already
  completed Fashion-MNIST data.
- Remediation: the source-data workflow now runs Fashion-MNIST as an
  independent automatic job, and supports CIFAR-10/100 only through an
  explicitly requested, timeout-bounded on-demand job.
- The new separated Fashion-MNIST job succeeded:
  https://github.com/Anurag9000/simple-adp/actions/runs/41462613451.
  Artifact `fashion-mnist-official-source-v01`, ID 13150322032,
  includes the newer signed `search_history.json` as well.

## Run C — real Fashion-MNIST matched frozen/loss/variance V01 comparison

- Code: `abbefac35c6644717a0202143705916d4e082d7a`, external
  workflow commit `a1987748510c7f460773f20f62cc94b7181262f8`.
- **Verified success:**
  https://github.com/Anurag9000/simple-adp/actions/runs/41462840053
- Pretrained encoder: published torchvision
  `ResNet18_Weights.IMAGENET1K_V1`, same strict local source and frozen
  encoder SHA256 `c747d20ed87f67d619438f6f072235eb69db114df73bb89481f23e66bacd9601`
  across **all three** runs.
- Data: official Fashion-MNIST, seed 11, identical class-stratified
  300 fit / 100 validation / 100 official-test images across every method.
  All routes use two maximum training epochs per stage, at most one
  candidate expansion, and matched common hyperparameters.
- Comparison manifest signature:
  `b60357431f19c0645a030ed384a61a64a7489bbe3717c9799ce22ba4766b5183`.

| V01 route | Selected hidden width | Validation accuracy | Official test accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|
| Fixed `MLP-Fixed` | 8 | 0.30 | 0.28 | 0.184284 |
| `ADP-W-RR`, loss trigger | 10 | 0.47 | 0.54 | 0.460163 |
| `ADP-W-RR`, variance-stagnation trigger | 8 | 0.28 | 0.30 | 0.210817 |

- Artifact `real-fashion-resnet18-v01-loss-vs-variance`, ID
  **13149472698**, contains **18 files** (signed summary, source
  provenance, per-method manifests, candidate search histories,
  inference model sidecars and outputs).
- All three method/criterion jobs were defined in advance. The experiment
  does not use the official test outcomes to choose a method; no comparative
  superiority claim is warranted from a single seed and small subset.
- A separate follow-up with three predefined seeds 11/23/37 was committed
  as `315f9cd0510f3191378efaa5e866a06a9f5781a4` and is **verified** as Run D below; it has a distinct signed
  comparison summary and separate artifacts.

## Run D — matched three-seed real Fashion-MNIST comparison

- **Verified success:** https://github.com/Anurag9000/simple-adp/actions/runs/41463589627
- Workflow commit `315f9cd0510f3191378efaa5e866a06a9f5781a4`.
- Official real Fashion-MNIST, published torchvision
  `ResNet18_Weights.IMAGENET1K_V1` frozen source; seeds **11, 23, 37**.
- Predeclared **nine independent jobs** (fixed/loss/variance per seed).
  Within each seed, all methods share the exact source, class-stratified
  official train/validation/test indices and training settings except
  expansion criterion.
- Per seed: **300 train, 100 validation, 100 official test**;
  two training epochs per stage, one maximum growth proposal.
- Comparison signature:
  `65d2016bb38e5bd1cf07b5c122d788ae7a606435afae2e9e2d31998ee74c11`.

| V01 route | Seed 11 test | Seed 23 test | Seed 37 test | Mean test accuracy |
|---|---:|---:|---:|---:|
| Fixed MLP | 0.28 | 0.38 | 0.35 | 0.3367 |
| `ADP-W-RR` loss trigger | 0.54 | 0.57 | 0.57 | 0.5600 |
| `ADP-W-RR` variance-stagnation trigger | 0.30 | 0.38 | 0.34 | 0.3400 |

- Artifact `real-fashion-resnet18-v01-3seed-loss-vs-variance`,
  GitHub Actions artifact ID **13149638400**, approximately **384 KB**.
  Includes all model outputs, manifests, candidate histories, and signed
  comparison summary.
- Full selected regression https://github.com/Anurag9000/simple-adp/actions/runs/41463404204:
  **365 tests passed in 73.32s**. Dedicated comparison tests
  https://github.com/Anurag9000/simple-adp/actions/runs/41463404061:
  **15 passed**.
- Earlier failed contract run `41462977800` only had an incorrect
  synthetic fixture expectation: 32 samples with a 20% per-class
  validation split give six validation examples, not eight. The corrected
  fixture was committed `5a1d09e31958026277b6c5428e61f3d229869b8b`;
  production split logic was unchanged.
- The means are descriptive for **only three small seeds**. No official-test
  outcomes were used to select candidate architectures or a winning route.
  These are *not* benchmark-quality superiority claims.

## Remaining scientific and technical work

1. Complete CIFAR-10 and CIFAR-100 actual dataset/split smoke runs.
2. Larger pre-registered multisplit/multiseed comparisons with uncertainty
   estimates and meaningful training budgets; three small runs are now verified.
3. Verified pretrained **inner-FFN** replacement (V03–V11) rather than only
   tiny from-scratch architecture-shaped controls.
4. CUDA execution and full-repository regression; current selected CI is CPU.
5. Extended weight provenance and durable experiment artifacts beyond
   GitHub Actions' finite retention window.

## Run E — published pretrained ConvNeXt-Tiny **internal FFN** growth

- **Verified success:** https://github.com/Anurag9000/simple-adp/actions/runs/41466060451
- Source-model code: `af64f94f90571400f63a331e038a9a0d72b052b6`;
  workflow script commit `7d380260066318dabc83ccd7695dcba754f2d43f`.
- Exact published full torchvision checkpoint:
  `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`,
  https://download.pytorch.org/models/convnext_tiny-983f1562.pth;
  local full state_dict file SHA256
  `48f10ea8765c304f933e395d3676017eb388634ff1a2808b8e4db08f6c91c272`.
- Growth target: the real first CNBlock source channel FFN,
  `convnext:features.1.0.channel_ffn`, initially transplanted *exactly*
  as `96→384→96` GELU. Native `ADP-W-RR` widened its hidden layer to
  **385 neurons** while keeping its input/output interface C=96.
- Frozen source state SHA256:
  `afb456139a1d70961457bbd2ceec406e3ac9216bfea1c7cd872184944ca133af`.
  The target-specific classifier and in-block FFN are trainable; all
  other pretrained source stages are frozen and included in the hash.
- Official Fashion-MNIST: **40 fit / 20 validation / 20 official test**,
  seed 11; one epoch per stage, at most one growth proposal. Official test
  is evaluated only after architecture selection.
- Validation accuracy **0.40**, validation macro-F1 **0.285714**,
  validation log loss **1.8915664164**.
- Official-test accuracy **0.45**, macro-F1 **0.422222**,
  test log loss **1.7877205028**.
- Uploaded `real-v04-convnext-source-inner-ffn` artifact ID
  **13151101792**, 103,225,380 bytes; includes source manifest,
  result JSON, local growth-history JSON and selected complete model
  snapshot (SHA256
  `a80346091aa5c6298c891ab6cf21d1f6cff6ceb5d32fbf7589264ca024b17752`).
- Source-fidelity tests against the actual torchvision **architecture**
  passed 4/4 in 9.97s:
  https://github.com/Anurag9000/simple-adp/actions/runs/41465492673.
  Those tests had random source weights; this independent Run E
  explicitly downloaded the official published pretrained weights.
- This verifies a **single real published-pretrained source FFN route**,
  not all V04 source methods or V03/V05–V11 pretrained internal FFNs.
  Tiny data, one seed and a one-proposal budget mean it is not evidence of
  benchmark superiority or trustworthy generalization.

## Run F — published pretrained ViT-B/16 *internal FFN* expansion

- **Verified successful** https://github.com/Anurag9000/simple-adp/actions/runs/41474035681
  with source script commit `d3464fba354f44f5961e586e1ddf6fe490a4227f`.
- Original source `torchvision.models.vit_b_16`, published
  `ViT_B_16_Weights.IMAGENET1K_V1` (ImageNet-1K):
  https://download.pytorch.org/models/vit_b_16-c867db91.pth.
  Complete saved strict source file SHA256:
  `335d1dfe0a7aac5787ffdc111e46843f7e2223218f49ab989333b880f77a427e`.
- Real first transformer FFN growth target
  `vit:encoder_layer_0.channel_ffn` starts at **768→3072→768** GELU
  with original pretrained weight tensors transplanted exactly.
  Source ADP-W-RR chose widened **3073 hidden neurons** after one
  proposal; no patch/attention/downstream source dimensions changed.
- Selected source-frozen SHA256:
  `71dae45b55b257d03809417ea7960c265365b8931656979dab07169807ca8196`.
  Source FFN/tiny task classifier are the only trainable units;
  entire remaining pretrained ViT is frozen.
- Official Fashion-MNIST seed 11, **20 fit, 10 validation, 10 official
  test** images; one epoch per stage, two history entries, no test
  materialization until training/selection completed.
- Validation accuracy **0.40**, macro-F1 **0.28**, log-loss
  **1.8708838387**. Official-test accuracy **0.40**, macro-F1
  **0.30**, log-loss **1.9859184127**.
- Artifact `official-fashion-vitb16-v03-source-ffn`, ID
  **13157600366**, size **319,053,611 bytes**. Four files: source
  manifest, `result.json`, `search_history.json`, complete selected
  model snapshot, SHA256
  `2c18fac8f1fc1b94523b1bbaea027532a20cd1de67a60434be16d1df7db46390`.
  GitHub Actions artifacts have finite retention (30 days).
- Prior original ViT source-fidelity CI
  https://github.com/Anurag9000/simple-adp/actions/runs/41472976307
  **4/4 passed** on randomly initialized full torchvision source
  architecture (initial feature equality), and coordinated source
  runner CI https://github.com/Anurag9000/simple-adp/actions/runs/41473641002
  **3/3 passed** with actual full-source CPU worker and 43 source
  route pairs. Both tests are separate from this real published-weight run.
- This is strictly an architecture/pretrained-data **integration smoke**,
  not benchmark-quality generalization or comparative scientific evidence.
