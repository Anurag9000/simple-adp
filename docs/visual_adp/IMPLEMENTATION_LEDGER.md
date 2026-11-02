# Visual ADP implementation ledger

States distinguish source/static implementation from executed tests and science.

| Item | State | Evidence / missing |
|---|---|---|
| Legacy 14 model identities | preserved | registry.py unchanged |
| V01 frozen image feature -> native ADP MLP head | runtime-tested (bounded CPU) | src/simple_adp/visual.py |
| V01 actual candidate growth | runtime-tested (bounded CPU) | delegates to native TabularADP; tests/test_visual_v01.py |
| V01 fixed non-growing control | runtime-tested (bounded CPU) | MLP-Fixed native route, tested in new test |
| V01 concrete torchvision ResNet18/ConvNeXt-tiny/ViT-B16 frozen encoder construction | runtime-tested ResNet18; other two static-complete | vision_encoders.py; local-only optional weights, trained origins not asserted |
| V01 packaged estimator export | static-complete | src/simple_adp/__init__.py |
| CIFAR-10/CIFAR-100/Fashion-MNIST optional loader | runtime-tested API on offline synthetic arrays (not real datasets) | vision_data.py, torchvision optional; no real dataset execution | 
| V01 opt-in coordinated runner and signed split jobs | runtime-tested (toy CPU) | runner/vision.py, plan.py, plugins.py, tests/test_runner_vision.py; toy and offline image provider only |
| V02 residual CNN adapter local MLP stage + native RR/all/D1/Dcur mutation | runtime-tested (bounded CPU); dedicated controller and opt-in runner verified | visual_inner.py; direct StageTrainer with whole-image gradient, site-specific growth, frozen-state snapshots |
| V03/V04 | missing | requires composite controller and source FFN integration |
| V05/V06a/V06b/V07 | missing | operator/model integration, mutation and resume |
| V08/V09 | missing | multistage/bank routing, isolation and candidate scheduling |
| V10/V11 | missing | ResMLP/GFNet operators and full integration |
| Real image scientific mini-runs and multi-seed benchmarks | missing | do not infer from synthetic CI |
| CPU/CUDA complete regression | partial | 241 selected CPU tests passed including V01 worker and torchvision ResNet18, plus variance contract CI; CUDA/full suite not executed |

Next: validate the eight generic V02 controller profiles through the coordinated runner, then integrate V03/V04. Original tabular job identities remain untouched.

**Next:** finish the V02 generic-profile CI; then extend eligible local FFN sites to V03/V04 without modifying the source taxonomy.

## V02 dedicated composite architecture-search increment (pending CI)

`src/simple_adp/visual_search.py` executes the original dedicated loss/variance
stage trainer over full-image V02 candidates for LGND-W-Greedy, ADP-W-RR,
ADP-W-All, ADP-D-Dcur and MLP-Fixed. Search covers candidate proposal,
validation-selected best architecture, no mutation of frozen CNN parameters,
per-batch optimizer/RNG/permutation checkpoints, restart and completed-run
idempotence. Other methods including eight generic phase-search configurations
and progressive ADP-D-D1 are explicitly rejected, not silently approximated.
This first controller is a standalone library API, not yet a signed V02
coordinated runner job. CPU execution and interruption tests pending CI.

## V02 coordinated suite integration (pending new CI)
A separately registered opt-in `vision_inner` ModelAdapter enumerates only the
five standalone supported V02 controllers, with one additional variance-shadow
counterpart per growing method. The original V01 routes remain independent.
The V02 worker trains on full images, uses the signed native per-batch resume
checkpoint, reports architecture ID/growth site, and writes a separately signed
best-model artifact. All vision paths are disabled by default. A bounded
worker end-to-end regression was added to tests/test_runner_vision.py.

## V02 ADP-D-D1 progressive insertion (pending CI)
The composite search now supports native width-one depth insertion and an
optimization-only ladder through widths 1..Dcur. Intermediate stages retain
whole-image gradients and signed mid-stage checkpoint/resume but never enter
architecture-selection history. The final matched-width depth candidate is
evaluated once; one logical insertion counts as one proposal. New synthetic
CPU tests interrupt exactly at width-two warmup and compare bitwise to an
uninterrupted run. The V02 coordinated runner enumerates this sixth method and
its variance counterpart without affecting legacy tabular algorithms.

## V02 generic composite phase-search (latest increment; pending CI)
`VisualResidualSearch` now supports all 14 canonical IDs. Eight generic
width-to-depth/depth-to-width algorithms run through real full-image composite
training with the source's `faithful` and `corrected` profiles. The stage
optimizer is native AdamW, preserving the original generic penalty objective,
RR/all-layer width and D1/Dcur depth operators. The faithful profile preserves
its documented source candidate score update semantics (including the known
unusual non-global-best score reporting). Signed mid-batch checkpoints hash the
profile as well as run data/config. The existing six dedicated checkpoints
retain their old identity signatures and can be resumed unchanged.

The opt-in V02 runner now enumerates 22 method/profile routes and their 21
variance-stagnation counterparts per seed/split/repeat, with isolation from V01.
The new 8-method x 2-profile controller tests, source profile identity/restart,
phase transition tests and V02 generic/variance worker execution are **pending
CI**. V03–V11 remain unimplemented.
