# Visual ADP implementation ledger

**The top table is the current summary.** Later chronological sections preserve
historical development milestones, including past pending-CI statements that
were subsequently verified by newer test evidence.

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
| V02 residual CNN adapter with all 14 canonical methods and generic profiles | runtime-tested on synthetic CPU; 259 tests passed | visual_inner.py; direct StageTrainer with whole-image gradient, site-specific growth, frozen-state snapshots |
| V03/V04 tiny FFN controls | runtime-tested (selected CPU CI) | 269 selected tests passed; native internal FFNs with opt-in runner, not pretrained torchvision models |
| V05 PoolFormer channel FFN | runtime-tested (selected CPU CI) | 275 selected tests passed, synthetic toy worker and native growth |
| V06a S2 / V06b axial shift FFN | runtime-tested (selected CPU CI) | 288 passing selected tests, real zero-padded shift operators, source growth and paired workers |
| V07 MLP-Mixer | implemented; CPU CI pending | independent token-MLP and channel-MLP routes, full-image controller, signed restart; simultaneous joint-site search not implemented |
| V08/V09 | missing | multistage/bank routing, isolation and candidate scheduling |
| V10/V11 | missing | ResMLP/GFNet operators and full integration |
| Real image scientific mini-runs and multi-seed benchmarks | missing | do not infer from synthetic CI |
| CPU/CUDA complete regression | partial | 288 selected CPU tests passed through V06a/b; CUDA and full suite not executed |

**Next:** implement V07 token/channel mixer scopes, V08–V11, pretrained source FFNs and
real-data science. Original tabular identities remain unchanged.

## V02 dedicated composite architecture-search increment (historical superseded phase)

`src/simple_adp/visual_search.py` executes the original dedicated loss/variance
stage trainer over full-image V02 candidates for LGND-W-Greedy, ADP-W-RR,
ADP-W-All, ADP-D-Dcur and MLP-Fixed. Search covers candidate proposal,
validation-selected best architecture, no mutation of frozen CNN parameters,
per-batch optimizer/RNG/permutation checkpoints, restart and completed-run
idempotence. Other methods including eight generic phase-search configurations
and progressive ADP-D-D1 are explicitly rejected, not silently approximated.
This first controller is a standalone library API, not yet a signed V02
coordinated runner job. CPU execution and interruption tests pending CI.

## V02 coordinated suite integration (historical superseded phase)
A separately registered opt-in `vision_inner` ModelAdapter enumerates only the
five standalone supported V02 controllers, with one additional variance-shadow
counterpart per growing method. The original V01 routes remain independent.
The V02 worker trains on full images, uses the signed native per-batch resume
checkpoint, reports architecture ID/growth site, and writes a separately signed
best-model artifact. All vision paths are disabled by default. A bounded
worker end-to-end regression was added to tests/test_runner_vision.py.

## V02 ADP-D-D1 progressive insertion (historical superseded phase)
The composite search now supports native width-one depth insertion and an
optimization-only ladder through widths 1..Dcur. Intermediate stages retain
whole-image gradients and signed mid-stage checkpoint/resume but never enter
architecture-selection history. The final matched-width depth candidate is
evaluated once; one logical insertion counts as one proposal. New synthetic
CPU tests interrupt exactly at width-two warmup and compare bitwise to an
uninterrupted run. The V02 coordinated runner enumerates this sixth method and
its variance counterpart without affecting legacy tabular algorithms.

## V02 generic composite phase-search (selected CPU CI verified)
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
The 8-method x 2-profile controller tests, profile identity/restart, phase
transitions and V02 generic/variance worker execution all passed selected CPU
CI (259 tests). V03–V11 remain unimplemented.

## Current verified V02 summary — 2026-11-02
- Commit: b51eac1ea9555e2398d05afc8e5384e32e5603ec
- CPU CI: https://github.com/Anurag9000/simple-adp/actions/runs/41394281464
- 259 selected tests passed, covering all 14 V02 canonical method IDs,
  faithful/corrected generic profiles, signed mid-batch resume, V01/V02
  coordinated workers, progressive D1 and shadow-BGD variance counterpart.
- Original tabular method/variance workflow also passed.
- Still missing: V03–V11 (with V06a/V06b distinct), real pretrained internal
  image backbones, actual CIFAR experiments, multi-seed science, full-suite
  regression and CUDA behavior verification.

## V03/V04 native inner-FFN composite controllers — verified selected CPU CI
- `src/simple_adp/visual_native_ffn.py` introduces **V03** patch/tokens/CLS,
  fixed multihead attention and a native token-wise expandable GELU FFN, and
  **V04** fixed depthwise spatial convolution and channels-last GELU channel FFN.
- Both own immutable patch/channel/attention interfaces and strict frozen
  state hashes. Only FFN linear hidden layers grow; all other modules and
  buffers are frozen. They are small random-initialized *architecture-shaped*
  controls, not pretrained torchvision ViT/ConvNeXt models.
- The V02 full-image candidate controller has an additive, explicitly typed
  composite factory and snapshot dispatcher, retaining the exact V02 checkpoint
  namespace. The V03/V04 signatures use distinct new namespaces.
- Tests added for frozen gradients, width/depth operators, GELU restoration,
  local weight transfer, safe snapshot, fixed control, generic phase search,
  actual training/growth and mid-stage signed replay. **Pending CI.**
- This pass implements the standalone library search path; central runner
  route registration and full real-data experiments are still missing.

## V03/V04 opt-in coordinated runner extension (verified selected CPU CI)
`VisionNativeFFNModelAdapter` registers independent adapter IDs
`vision_v03` / `vision_v04` under `SIMPLE_ADP_ENABLE_VISION=1`.
Each enumerates 14 canonical IDs / 22 method-profile combinations, with 21
matched shadow-BGD variance counterparts per split, seed and repeat through
the existing SuitePlan. The worker trains full images with true internal FFN
gradients, selected-candidate exports, frozen SHA checks and signed exact
checkpointing. V01/V02 and all nonvisual adapters remain unchanged by default.
Real pretrained ViT-B16/ConvNeXt-Tiny FFN replacement is **not** implemented.
New bounded CPU worker tests cover V03/V04 source jobs, criterion pairs and
corrected generic methods. Evidence remains pending CI.

## V03/V04 combined coordinated runner verified — 2026-11-02
- Code integration: 02dba9579030e946f768a4bd18ec426a7517b0c7.
- Selected CI: https://github.com/Anurag9000/simple-adp/actions/runs/41397841490.
- **269 passing CPU tests in 41.31 seconds**, including V03/V04 signed
  standalone search and opt-in worker training, corrected generic search,
  loss/variance counterpart and preserved earlier core/V01/V02 behavior.
- Native method/profile and matched job generation: **43 runner jobs per
  selected seed/split/repeat per V03 and per V04**, subject to suite settings.
- No pretrained FFN injection, real CIFAR benchmark or GPU scientific run.

## V05 PoolFormer native channel FFN (verified selected CPU CI)
- Same frozen-interface architecture factory now implements a distinct
  `V05` family: fixed strided stem, parameter-free 3x3 average pooling minus
  identity token mixer, residual update, channels-last LayerNorm and local
  GELU FFN, fixed layer scale and pooled frozen classifier.
- No spatial kernels, positions, channels, patch counts, normalization or
  fixed layer scales are eligible for growth. Exact frozen hashing includes
  the parameter-free pooling operator representation as well as modules.
- V05 has a separate installed opt-in `vision_v05` runner adapter, 22 method/
  profile routes plus 21 matched shadow variance gates per seed/split/repeat,
  signed full-image model training and resume through the common controller.
- Unit, frozen-pool, native growth, exact replay and worker tests added.
  **CI pending**. These are random-initialized controls, not pretrained PoolFormer
  or validated real-image science.

## V06a and V06b independent shift-MLP channel FFN routes (verified selected CPU CI)
- `src/simple_adp/visual_shift.py` defines separate zero-padded,
  parameter-free **S2SpatialShift** (four directional channel groups) and
  **AxialSpatialShift** (separate height/width branches, averaged).
  Neither circular wraps boundaries nor changes channels/tokens. Both
  are differentiable without trainable shift weights.
- `NativeFFNSpec` identifies distinct `V06a` / `V06b`, sites
  `channel:s2.block0` / `channel:axial.block0`, fixed spatial dimensions,
  groupability checks and immutable shift operator representation. The
  channel FFN alone uses original source RR/all/D1/Dcur architecture growth.
- The installed opt-in `vision_v06a` and `vision_v06b` model adapters
  enumerate separately 22 method/profile routes plus 21 matched variance
  counterparts per seed/split/repeat through the existing worker/ledger.
- Tests cover exact no-wrap geometry and gradients, model frozen-shift
  hashing, source growth, whole-image exact stage restart and criterion
  paired worker execution. **Pending CI.** These are tiny from-scratch
  shift-MLP controls, not verified published S²-MLP/AS-MLP checkpoints.

## Verified V05 checkpoint and coordinated worker — 2026-11-02
- Commit: debf351a6fa5edb280abeb5ea1534697cbead81e.
- CPU CI: https://github.com/Anurag9000/simple-adp/actions/runs/41398371660
- **275 selected tests passed in 21.92 seconds**, including native V05
  pooling-operator verification, source ADP growth, snapshot/mid-batch replay,
  fixed/generic routes, central worker and matched variance-gated job.
- Only synthetic toy images and randomly initialized feature backbones tested.

## V06a/V06b selected CI confirmation — 2026-11-02
- Code commit: 3bff656fceeaa62e0feef6398124a261366d7273.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41399092726.
- **288 passing selected CPU tests in 40.71s**; both V06 families run through
  source growth, signed restart, fixed/generic controllers, opt-in worker,
  matched variance gate and frozen spatial-operator checks.
- Original ADP original-vs-variance contract CI separately succeeded.
- These are tiny experimental fixed-shift backbones; no pretrained model,
  published-architecture numerical equivalence, real dataset benchmarks or
  CUDA-wide scientific tests are claimed.

## V07 MLP-Mixer independent token/channel sites (pending CI)
- `src/simple_adp/visual_mixer.py` implements a genuine residual one-block
  Mixer with fixed patch embedding, separate token-mixing and channel-mixing
  GELU MLPs, two LayerNorms, and frozen final classifier.
- `MixerSpec` allows exactly one eligible site per *job*: `token:block0`
  expands hidden layers of the [B,D,T] token MLP with fixed T; `channel:block0`
  expands hidden layers of the [B,T,D] channel MLP with fixed D. The opposite
  real Mixer MLP stays frozen and participates in forward/backward.
- Both modes use all canonical 14 ADP method IDs, 8 generic methods with
  faithful/corrected profiles, a separately matched variance-stagnation
  counterpart for each growing job. `vision_v07_token` and
  `vision_v07_channel` are independently identified opt-in model adapters.
- Complete source growth, overlap-copy, frozen-module hashing, exact signed
  mid-batch resume, fixed models and generic phases reuse the prior visual
  controller without touching tabular semantics.
- 43 route/criterion jobs per selected split/seed/repeat/site, 86 across
  both sites. **Not yet a joint two-site growth route**: simultaneous
  token+channel growth is deferred to V08 multisite search and not recorded
  as implemented. The control is random-initialized, not pretrained.
- New tests for site contract, gradients, all growth operators, snapshot
  restoration, exact mid-batch replay and worker criterion pairing. CI pending.
