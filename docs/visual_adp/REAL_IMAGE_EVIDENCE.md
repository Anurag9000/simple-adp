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

## Run B — CIFAR-10

The same bounded official ResNet18 source and V01 procedure were added in
`22f3cee27b5f4a362047c9dd1efa1535bd902d10` to test native RGB data.
**Pending** the new network-enabled GitHub Actions run; do not record
CIFAR-10 metrics until the separate workflow succeeds.

## Remaining scientific and technical work

1. Complete CIFAR-10 and CIFAR-100 actual dataset/split smoke runs.
2. Real multisplit/multiseed comparisons on meaningful sample sizes,
   including a frozen-head baseline and loss-versus-variance matched routes.
3. Verified pretrained **inner-FFN** replacement (V03–V11) rather than only
   tiny from-scratch architecture-shaped controls.
4. CUDA execution and full-repository regression; current selected CI is CPU.
5. Extended weight provenance and durable experiment artifacts beyond
   GitHub Actions' finite retention window.
