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

## Concrete frozen torchvision encoder CI (verified)
- Code commit: eca4b61c688d66137a7bf8b52379a491eddf0c89.
- CI run: https://github.com/Anurag9000/simple-adp/actions/runs/40922695070 — **success, 228 passed in 21.02s**.
- Tested local torchvision ResNet18 state-dict reconstruction, classifier removal, frozen parameter and inference equivalence, SHA256 rejection, deterministic construction, malformed input rejection; previous V01 runner and core tests rerun.
- Standalone ADP original-versus-variance contract: https://github.com/Anurag9000/simple-adp/actions/runs/40922695059 — **success**.
- ConvNeXt-Tiny/ViT-B16 source constructors are statically implemented but not yet exercised by this workflow; actual pretrained checkpoint provenance, real CIFAR training, scientific evaluation and CUDA are not verified.

## V02 residual adapter stage evidence (verified)
- Code commit: 77fa4f62b487183db044c5effeccb961c293ca1b.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/40923983465 — **success, 231 passed in 19.74s**.
- Native variance/source-contract CI: https://github.com/Anurag9000/simple-adp/actions/runs/40923983499 — **success**.
- V02 specific executed tests: differentiable frozen downstream CNN, nonzero
  local MLP gradients, frozen parameter/buffer bitwise digest invariance,
  source RR/all/DEN-RR width and D1/Dcur depth operators, real local structural
  parameter transfer, whole-image native StageTrainer and exact architecture
  snapshot reconstruction without consuming RNG.
- Scope **only** candidate training/mutation. V02 full canonical search,
  acceptance/rollback across 14 methods, exact interruption resume and central
  runner are not implemented or tested. No real-image scientific validation.

## V02 dedicated-search standalone validation
- Commit: c2ef9657ba0eb7734d5f8716d062e095c461af33
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41393058629
- **239 passed in 27.07s**, including five dedicated V02 controllers,
  complete candidate search/selection and mid-stage exact batch resume.
- Standalone native variance-contract workflow for the same commit succeeded.
- Subsequent coordinated runner V02 adapter is pending a separate test run.

## V02 D1 progressive test addition (pending CI)
- Added a synthetic full-image trace for width-one insertion, widths 1->2->3
  optimization-only ladder, no intermediate best selection and exact
  interruption/restart inside a warmup stage. No numerical claim until
  corresponding workflow passes.

## Progressive D1 CI (verified)
- Commit: 230bdd4f7d0c5b484d19d90fe3ac3add4e118c2c
- Workflow: https://github.com/Anurag9000/simple-adp/actions/runs/41393792798
- **241 passed in 35.75s**, including interrupted progressive D1 width-one
  insertion, intermediate optimization-only stages and exact warmup replay.
- Standalone original-versus-variance contract workflow succeeded.

## Full generic V02 phase-search increment (pending CI)
- Eight generic method configurations x two profiles, phase transition,
  signed profile identity, exact mid-batch interruption, generic V02 worker
  execution and paired variance worker have been added. Do not call runtime
  verified until the corresponding Actions test run is inspected.

## All 14 canonical V02 IDs and generic runner profiles verified (2026-11-02)
- Commit: b51eac1ea9555e2398d05afc8e5384e32e5603ec.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41394281464.
- **259 passing tests in 35.61s** on selected synthetic CPU suite.
- Tested generic phase transitions, both source profiles, signed profile-specific
  stage checkpoints, interruption/recovery, V02 full-image worker and native
  variance-shadow counterpart, while retaining V01/tabular regression checks.
- Original ADP native loss/variance contract workflow also succeeded.
- No real pretrained source backbones, CIFAR scientific benchmarks, CUDA or
  complete repository-wide test suite were executed in this increment.

## V03/V04 standalone native FFN tests (pending CI)
- New tests: tests/test_visual_native_ffn.py covers ViT attention/CLS token FFN,
  ConvNeXt-style depthwise/channel FFN, fixed-state hash, actual gradients,
  native RR/all/D1/Dcur, GELU/weights transfer, exact checkpoint restoration,
  fixed model and generic search, and mid-batch fault/restart.
- Do not count new routes or scientific jobs as executed until verified by CI.

## V03/V04 coordinated runner test addition (pending CI)
- Test: test_native_ffn_runner_executes_and_pairs_variance, for V03/V04.
- Checks source job registration, distinct family identity, genuine worker
  completion, fixed growth site, signed model exports, matched split/seed,
  variance shadow and corrected generic source route.
- No real dataset benchmark or CUDA coverage claimed.

## V03/V04 full runner integration verified (2026-11-02)
- Commit: 02dba9579030e946f768a4bd18ec426a7517b0c7.
- Workflow: https://github.com/Anurag9000/simple-adp/actions/runs/41397841490.
- **269 passed in 41.31 seconds** on the selected synthetic CPU suite,
  including V03 and V04 worker execution, matched variance and generic route.

## V05 PoolFormer new tests (pending CI)
- Enlarged V03/V04 native FFN test matrix to V05 for gradient flow,
  mutation, signed replay, fixed model, generic controller and runner job.
- Explicitly asserts parameter-free `Pool(x)-x` and checks that changes to
  the pooling kernel invalidate the frozen-state verification.

## V06a/V06b shift operators, models and paired jobs (pending CI)
- tests/test_visual_native_ffn.py now includes V06a/b in the frozen backbone,
  native growth, snapshot, interrupted resume, fixed control and generic
  method suite. Explicit mathematical shift geometry/zero boundary tests
  compare distinct S2 versus axial mixing, prove no learnable mixer weights
  and verify backprop through both spatial permutations.
- tests/test_runner_vision.py extends native worker/variance route tests to
  both V06a and V06b, preserving the same split and seed provenance.
- Do not claim CI success or a real pretrained backbone here yet.

## V05 PoolFormer native FFN verified (2026-11-02)
- Commit: debf351a6fa5edb280abeb5ea1534697cbead81e.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41398371660.
- **275 passed in 21.92s**, selected synthetic CPU suite with V05
  pooling-operator frozen identity, native growth, restarts, worker routes,
  generic method and criterion counterpart.
- No real-image scientific runs or pretrained source weights.

## V06a/V06b zero-padded shift and paired runner CI verified (2026-11-02)
- Code commit: 3bff656fceeaa62e0feef6398124a261366d7273.
- Workflow: https://github.com/Anurag9000/simple-adp/actions/runs/41399092726.
- **288 passed in 40.71s** on selected CPU tests, including both shift
  geometries/no-wrap boundaries, no-trainable-spatial-weights check,
  local MLP growth, signed exact resume, generic/fixed methods and V06a/b
  coordinated worker original-versus-shadow-BGD variance counterpart.
- Standalone original-versus-variance controller contract CI also succeeded.
- No pretrained published architecture, real CIFAR training or CUDA validation.
