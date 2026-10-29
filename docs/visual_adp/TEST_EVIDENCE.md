# Test evidence — vision ADP
Baseline GitHub main inspected: a2e6b6325a80ca6da41b1898ebf71c6ec8829a62.

Before publishing this increment, source/contract audit was performed against
live source. Locally, an isolated substitute-head V01 shape/freeze/manifest
smoke passed 2 tests on CPU (torch 2.10.0), but it is **not** evidence that
the actual repository's TabularADP search or runner has passed.

Authoritative repository tests are in tests/test_visual_v01.py.
The .github/workflows/visual-adp-v01.yml workflow requests these tests and
the existing core regression tests on a fresh Python installation.
Do not mark as executed until a matching GitHub Actions run is inspected.
No CIFAR-10/100, Fashion-MNIST, CUDA, pretrained-backbone or multi-seed
scientific result has been executed in this increment.

## Verified GitHub Actions on main
- Commit: 1be6b214d4edcc78fbe9eb1246bf10ffe2d80b57
- Visual V01 CPU smoke + legacy core workflow: https://github.com/Anurag9000/simple-adp/actions/runs/40918821989 — **success**, 27 passed in 3.57s (tests/test_visual_v01.py and tests/test_core.py).
- Original-versus-variance controller contract workflow: https://github.com/Anurag9000/simple-adp/actions/runs/40918821988 — **success**.
- This proves bounded CPU unit/integration coverage; no real-image CIFAR, pretrained backbone, runner, full CUDA or multi-seed experiment has been executed.

## Verified second CPU test pass (image partitioning)
- Commit: 51322fca736591ffa8ea9b35f23745bb10427e17
- Visual ADP V01, new image split tests, and legacy core: https://github.com/Anurag9000/simple-adp/actions/runs/40919599860 — **success**, 30 passed in 5.67s.
- Original-versus-variance contracts: https://github.com/Anurag9000/simple-adp/actions/runs/40919599981 — **success**.
- Evidence level: offline synthetic image tests, native ADP head training, V01 save/load, deterministic partition tests, core regressions. Real torchvision dataset files and pretrained weight downloads were not used.

## Second opt-in coordinated worker CI (verified)
- Commit: 94be53aef5d7a96da88ea2942bb85069a9b2023a
- Run: https://github.com/Anurag9000/simple-adp/actions/runs/40921830940
- **Success, 226 passed in 24.48s** across V01 runner, plugin, model,
  visual estimator, image split, and original core tests.
- Failed prior attempts revealed systemd re-exec in pytest and a scientifically
  invalid variance counterpart for third-party adapters; both were fixed.
- Concrete pretrained-model factory tests remain pending CI on the next commit.
