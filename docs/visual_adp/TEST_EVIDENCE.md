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

## V07 Mixer dual site-scoped search and worker tests (pending CI)
- tests/test_visual_mixer.py covers both independent site contracts,
  end-to-end gradients through frozen opposite Mixer MLP, RR/all/DEN RR/D1/
  Dcur growth, fixed dimensional interfaces, copied weights, frozen hash,
  strict state reconstruction, signed mid-batch exact resume and generic/fixed
  controllers. tests/test_runner_vision.py covers both route families,
  22+21 route inventory, matched split/seed, criterion and worker execution.
- CI and real-world pretrained scientific jobs not yet verified.

## V07 two independent site CI (verified)
- Commit c0c6bfae7c3f6bd37c0a70f471f77f440dd6f41b.
- Workflow https://github.com/Anurag9000/simple-adp/actions/runs/41407812970
- **299 selected CPU tests passed in 42.75s**, including both V07
  token/channel scopes and end-to-end paired workers; native original/variance
  contract workflow also passed.

## V07 synchronous-both proposed tests (pending CI)
- Added dual local-MLP gradient/frozen-backbone checks, synchronized RR/all/
  D1/Dcur, cursor-resolved snapshot, exact mid-batch restart, joint capacity
  ceiling and source D1 serial warmup, plus coordinated-both source/variance
  worker. Explicitly separate from V08 adaptive multi-site scheduling.

## V10/V11 native spatial mixer tests (pending CI)
- tests/test_visual_native_ffn.py now includes V10/V11 in growth, frozen
  operator, signed resume and generic/fixed matrix. Dedicated V10 checks
  dense fixed token matrix/affine hash and local width expansion. V11 checks
  complex64 nontrivial Fourier filter, rfft2/irfft2 axes/shapes, gradients,
  frozen-filter strictness and depth expansion.
- tests/test_runner_vision.py extends the native source/variance and generic
  worker smoke matrix for both new architecture IDs.
- CI and real dataset scientific execution pending; do not infer pretrained
  source behavior from these bounded controls.

## V07 both site and V10/V11 verified CI (2026-11-02)
- V07 303 passed: https://github.com/Anurag9000/simple-adp/actions/runs/41409201374
- V10/V11 315 passed: https://github.com/Anurag9000/simple-adp/actions/runs/41409409751
- The earlier V07 neuron-cap fixture was corrected to count the fixed
  local output dimensions per source total_neurons(). Both new runs passed.

## V08 three fixed sites implementation tests (pending CPU CI)
- tests/test_visual_multistage.py covers early/middle/late model placement,
  source growth and warm start, exact frozen backbone and unselected adapter
  hashes, signed stage eviction/restart, generic search and fixed controls.
- tests/test_runner_vision.py extends independent-site method/variance
  matched jobs and end-to-end worker execution for each V08 site.
- No simultaneous-growth or adaptive-site scheduling claim.

## V08 independent early/middle/late stage CI verified (2026-11-02)
- Commit 916c560c94a2d1d474f6e5e7c356fa4786ecb1a5.
- https://github.com/Anurag9000/simple-adp/actions/runs/41410224145:
  **331 passed in 53.35s**. Original/variance controller contract also passed.

## V09 fixed hard-routing bank tests (pending CPU CI)
- New tests/test_visual_adapter_bank.py covers immutable hard gating,
  label-free/batch-order-independent member routing, local-only backprop,
  trainable/immutable bank isolation, source RR/all/DEN/D1/Dcur,
  snapshot and RNG identity, exact signed batch recovery and generic/fixed
  controllers across all three members.
- tests/test_runner_vision.py exercises all three bank adapter routes with
  matching original/variance seeds/split identities, immutable routing-count
  JSON diagnostics, artifact outputs and corrected generic controller.
- Fresh Actions status pending. No pretrained source or real-image benchmark.

## V09 frozen hard-router bank selected CI verified (2026-11-02)
- Code commit `1d5c1ae3472671fe162b47a97ac73df4a6ab2b5e`.
- Workflow https://github.com/Anurag9000/simple-adp/actions/runs/41413752172
- **347 passed in 67.91 seconds** selected CPU suite; nonselected bank
  isolation, frozen router, accurate route count export, source growth,
  exact interruption recovery and loss/variance matched worker were tested.
- Original-vs-variance method contract workflow also succeeded.
- No learned routing or pretrained bank experiment performed.

## V08 predefined multi-site source policy tests (pending CI)
- New tests/test_visual_multistage_schedule.py covers real separate local
  MLPs and active-only gradients, frozen CNN invariance, source RR/all/D1/Dcur
  growth within the selected site, per-site cursor and snapshot exactness,
  explicit early/middle/late trace, mid-middle-site crash/replay, aggregate
  capacity ceiling, generic phase/fixed controls and invalid site order.
- tests/test_runner_vision.py adds the fourth V08 family, signed per-job
  schedule trace, 22 original/21 variance route matching and source workers.
- No pretrained scientific run; CI pending.

## V08 scheduled multisite ablation selected CI — verified (2026-11-02)
- Initial code `53f7d7a33b6a78439363bb9c7419aedacf3d0e99`.
- First workflow https://github.com/Anurag9000/simple-adp/actions/runs/41415150793
  failed test collection due to a Python dictionary-comprehension syntax
  error in snapshot restore; resolved by follow-up fixes.
- Successful follow-up workflow:
  https://github.com/Anurag9000/simple-adp/actions/runs/41415527242.
- **354 selected CPU tests passed in 69.35s**, including all prior V01–V11
  controlled families, V09 fixed bank, V08 scheduled three-site real local
  growth, aggregate capacity, exact middle-site interrupted resume, and
  original/variance matched worker. Variance contract workflow also passed.
- These are synthetic CPU implementation tests, not full pretrained real
  CIFAR-10/100, Fashion-MNIST, full-suite, or GPU scientific validation.

## 817-job full opt-in vision manifest closure (verified CI)
- New test: test_opt_in_full_visual_route_inventory_has_exact_817_paired_jobs.
- Tests all 19 architecture/site-policy families enumerate 22 original
  method/profile jobs and exactly 21 source-matched variance counterparts;
  no fixed-control duplicates, identity collisions, split/fold/seed/profile
  changes or loss/variance pairing gaps. This test creates a plan only,
  and executes **zero** training jobs. CI verified at
  https://github.com/Anurag9000/simple-adp/actions/runs/41416422457.

## 817-job opt-in plan check succeeded (2026-11-02)
- CI https://github.com/Anurag9000/simple-adp/actions/runs/41416422457
- **355 selected CPU tests passed in 50.96s**, including the new no-training
  method/profile/architecture/scope original-versus-variance manifest closure.
- No 817-job execution or real pretrained dataset was performed.


## V01 real-data experiment runner tests (pending selected CI)

Added \`tests/test_vision_experiment.py\` to test the explicit grayscale
repeat/bilinear resize bridge, frozen model/geometry contract, random
backbone opt-in, deterministic split/manifest, test-partition materialization
only after fit, signed experiment result and model checksum checks.
All samples in these tests are **offline synthetic fixtures**, not real
CIFAR-10/CIFAR-100/Fashion-MNIST or published pretrained weights.
A real source-model checkpoint or a functioning experiment CLI alone is not
evidence of accuracy or reproducible real-data science. CI pending.

## Verified V01 official Fashion-MNIST + pretrained ResNet18 smoke (2026-11-02)
- 360 selected CPU offline tests passed:
  https://github.com/Anurag9000/simple-adp/actions/runs/41450351931.
- A separate network-enabled workflow completed on real official images
  and published torchvision pretrained weights:
  https://github.com/Anurag9000/simple-adp/actions/runs/41451103405.
- Data: 40 train / 20 validation / 20 official test; source model
  `ResNet18_Weights.IMAGENET1K_V1`, SHA256
  `eff99529cc4a2ad4edac486be5a9e9724e7f4e0ccc0e3b319dd07b7e48306592`.
- Validation accuracy 0.15; test accuracy 0.15; test macro-F1 0.095;
  test log-loss 4.2930310860. Small-scale **integration smoke only**;
  no statistical generalization or comparative scientific claim.
- Artifacts uploaded and verified: Fashion-MNIST V01 manifest,
  result JSON, published source provenance and inference model.

## V01 exact experiment resume verified (2026-11-02)

- New offline fixture test interrupted the native V01 head after a batch,
  preserved signed manifest/checkpoint and resumed identically.
- GitHub Actions https://github.com/Anurag9000/simple-adp/actions/runs/41452649913:
  **361 selected CPU tests passed in 47.32 seconds**.
- New reporting fields (candidate search history, confusion matrix,
  parameter counts) are committed in `8e2d618b67d2db27fb0de298202c8ce31aa6d714`;
  the extended integrity tests are pending fresh CI.

## V01 per-job full-search history and result integrity verified (2026-11-02)
- https://github.com/Anurag9000/simple-adp/actions/runs/41453060345:
  **361 selected CPU tests passed in 64.78s**, including source ADP candidate
  history JSON, class-ordered confusion matrices, selected/frozen model
  parameter counts and candidate-history SHA256 tamper rejection.
- Original loss-vs-variance controller contract workflow also succeeded.
- Tests use offline synthetic fixtures; earlier official Fashion-MNIST smoke
  remains valid but was executed before per-candidate search_history.json was
  added. The richer artifacts require a new real-data run.

## Matched loss/variance/fixed V01 suite — verified (2026-11-02)
- Source code: `abbefac35c6644717a0202143705916d4e082d7a`.
  One-seed published pretrained ResNet18 + official Fashion-MNIST:
  https://github.com/Anurag9000/simple-adp/actions/runs/41462840053.
  3 completed jobs, identical source weights and split indices,
  300/100/100 official image subsets, signed source/model/result JSON.
- Three-seed extended source/model comparison:
  https://github.com/Anurag9000/simple-adp/actions/runs/41463589627.
  **9 completed jobs** on seeds 11/23/37, all matched source and split
  within each seed. Official-test accuracy fixed=[.28,.38,.35],
  loss=[.54,.57,.57], variance=[.30,.38,.34]. Not a statistical benchmark.
- Offline signed comparison contract originally failed one test fixture
  asserting validation count 8 for 32 samples and 20% stratification;
  actual correct source validation size is 6. Corrected test in
  `5a1d09e31958026277b6c5428e61f3d229869b8b`.
- Dedicated CI:
  https://github.com/Anurag9000/simple-adp/actions/runs/41463404061
  **15 passed**. Full selected image/core CI:
  https://github.com/Anurag9000/simple-adp/actions/runs/41463404204
  **365 passed in 73.32 seconds**.
- Official CIFAR-10 source download in run 41451620431 was cancelled
  after ~30 min at ~60%; no CIFAR metrics. Independent workflows keep
  successful Fashion evidence and CIFAR download failures separate.

## V04 exact torchvision ConvNeXt FFN source transfer — verified (2026-11-02)

- Source+tests: `af64f94f90571400f63a331e038a9a0d72b052b6`.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41465492673
  **4/4 passed in 9.97s** against a complete, randomly initialized
  torchvision ConvNeXt-Tiny source state_dict.
- Tests establish exact original-source CNN feature tensor equality
  before mutation, byte-identical original FFN tensor transfer,
  GELU nonlinearities, strict frozen tensor and source-file hashes,
  local/head-only gradients, real internal source width/depth expansion,
  exact frozen-source snapshots and one native full-image trained stage.
- This does **not** prove published pretrained model state was used in the
  test. A distinct published-weight real-image experiment was committed
  as `7d380260066318dabc83ccd7695dcba754f2d43f`,
  https://github.com/Anurag9000/simple-adp/actions/runs/41466060451;
  **verified completed successfully** with published pretrained weights.

## Published pretrained ConvNeXt-Tiny real internal FFN growth verified

- Real official-weight/official-image workflow:
  https://github.com/Anurag9000/simple-adp/actions/runs/41466060451,
  commit `7d380260066318dabc83ccd7695dcba754f2d43f`.
- **Success**: full strict original torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
  state, real `features[1][0]` internal channel FFN transfer and true
  width growth from **384 to 385**, trained with full-image backprop.
  All other pretrained source parameters frozen and SHA256 verified.
- Official Fashion-MNIST fit/validation/test: 40/20/20.
  Validation accuracy 0.40, official-test accuracy 0.45, test macro-F1
  0.422222, with one seed and one epoch per stage.
- Artifact ID 13151101792, selected source model ~103MB with
  provenance, exact source/selected-model checksums, stage history,
  evaluation metrics and official split manifest.
- This is a real pretrained internal FFN **micro-smoke only**. No large
  pretrained V04 experiment grid, V03/V05–V11 source proof, or CUDA
  execution has been claimed.

## Source-faithful coordinated V04 worker / 43-job identity test — verified
- First run 41469833279 rejected invalid synthetic global max_width=385
  because the default runner architecture grid includes wider widths.
  Fixture corrected to max_width=512, without changing the source model
  or its required 384 initial FFN width.
- Corrected test CI https://github.com/Anurag9000/simple-adp/actions/runs/41470335656:
  **3/3 passed in 21.36s**. Assertions include original 817-job plan
  unchanged; gated 43 source additions, exact 22/21 source/variance pairing,
  distinct SHA-bound identities, rejection of source checksum/path mutation,
  and an actual full original-torchvision source worker producing inference
  artifact + frozen-source hash. Source weights in fixture were *random*;
  this test is not a pretrained benchmark.
- Complete selected CPU suite is separately queued/running at
  https://github.com/Anurag9000/simple-adp/actions/runs/41470335650:
  **373 passing selected CPU tests in 112.18s**.

## Verified source V03 ViT-B16 architecture and central runner (2026-11-02)

- Exact full architecture source test CI:
  https://github.com/Anurag9000/simple-adp/actions/runs/41472976307,
  **4 passed in 23.28s**. Original first source FFN tensor values and
  whole source encoder features are equal before mutation. Width, frozen
  source SHA256, strict restore and native training tested with a complete
  random initialization from torchvision, not published pretrained weights.
- Strict coordinated source V03 test CI:
  https://github.com/Anurag9000/simple-adp/actions/runs/41473641002,
  **3 passed in 51.52s**. Default 817-job plan unchanged; explicit V03
  gate adds 43 source-bound jobs (22/21 original/variance), matches
  split/seed/model, refuses SHA or file drift, and executes one full-source
  ViT CPU worker saving a model artifact.
- Published pretrained experiment https://github.com/Anurag9000/simple-adp/actions/runs/41474035681
  is a separate real-data workflow **pending**. The full selected CPU
  regression after source V03 runner integration is also separately pending.
