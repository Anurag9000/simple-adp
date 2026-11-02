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
| V07 MLP-Mixer | all 3 token/channel/synchronous-both scopes CPU CI verified (303 tests) | true residual token/channel MLPs, signed resumed search, matched variance jobs and joint budget |
| V08 fixed three-stage residual adapter | independent sites + fixed multisite schedule verified (354 tests) | early/middle/late local growth, saved site order/weights and exact middle-site resume |
| V09 strict adapter bank | selected CPU CI verified: 347 passed | hard frozen router, three strictly isolated adapter targets, routing counts and matched workers |
| V10 ResMLP / V11 GFNet | selected CPU CI verified (315 tests) | fixed dense token/affine mixer and complex FFT filter, native channel ADP and paired worker |
| Local official-image V01 experiment CLI and deterministic grayscale/RGB transform | selected CPU CI verified; 360 passed | signed source/run artifacts and official test after fit; synthetic unit fixtures |
| Official Fashion-MNIST + published pretrained ResNet18 micro-smoke | real-data integration verified | 40/20/20 official image partitions, GitHub Actions run 41451103405, uploaded model and provenance; **not** a performance benchmark |
| Real image scientific mini-runs and multi-seed benchmarks | missing | do not infer from synthetic CI |
| CPU/CUDA complete regression | partial | 354 selected CPU tests passed including V08 schedule and V09 bank; CUDA/full suite not executed |

**Next:** adaptive multi-site selection and optional bank creation/learned routing; then
strict pretrained-source FFN integration, real-image benchmarking and GPU/full-suite checks.
Original tabular identities remain unchanged.

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

## V07 independently selectable sites verified — 2026-11-02
- Commit: c0c6bfae7c3f6bd37c0a70f471f77f440dd6f41b
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41407812970
- **299 selected CPU tests passed in 42.75s**, including both independently
  trainable Mixer scopes, all local growth operators, frozen-state hashing,
  exact mid-batch restart and two separate coordinated worker variants.
- Original method/variance contract workflow succeeded.

## Explicit V07 synchronous-both token+channel ablation (pending CI)
- New scope `both:block0`, route family `vision_v07_both`, is a
  **separately labeled noncanonical site policy**. Every candidate proposes
  the *same source operator* synchronously to the token-Mixer and channel-
  Mixer hidden MLPs (not to token count, embedding dimension, stem or norms).
  Both site parameters train and both participate in shadow-BGD variance
  observation. The controller's original within-local-MLP semantics remain
  unchanged for token-only/channel-only jobs.
- Signed snapshot records both hidden-width vectors and individual RR
  cursors; frozen-state verification excludes the two trainable MLPs but
  includes all other modules. Candidate neuron budget accounts for both
  stacks. Progressive D1 warmup grows both new layers in lockstep.
- Separate search signature and checkpoint kind prohibit accidental resume
  between single-site and synchronous-both experiments.
- Model adapter enumerates 22 method/profile routes + 21 matched variance
  counterparts per split/seed/repeat, in addition to the 86 independent
  token/channel jobs (129 V07 jobs total for all three explicit scopes).
- New tests cover true dual gradients, native RR/all/depth expansion,
  independent cursor restoration, exactly resumed mid-batch trajectory,
  synchronized D1 intermediate widths and signed budget enforcement.
  **CI pending**. This is a synchronous two-site ablation, NOT the V08
  adaptive multi-stage scheduling algorithm.

## V10 ResMLP native token-mixer/channel-FFN route (pending CI)
- Native V10 keeps a fixed token-mixing `Linear(T,T)` and fixed per-channel
  affine scale/shift, plus immutable patch embedding and classifier. The
  expandable GELU MLP has invariant `Linear(C,H)...Linear(H,C)` interfaces.
- `channel:resmlp.block0` identifies its sole eligible site. The dense
  spatial-token mixing matrix remains frozen and included in the frozen
  backbone hash. Source width/depth methods, signed resume, and opt-in
  `vision_v10` worker routes are integrated without touching V01–V07 jobs.
- One fixed + 21 adaptive method/profile routes with one matched BGD-shadow
  variance counterpart each: 43 jobs per seed/split/repeat, pending CI.

## V11 GFNet fixed complex-frequency filter and real ADP FFN (pending CI)
- Native V11 applies `torch.fft.rfft2(..., dim=(1,2))` and `irfft2`
  to patch-grid spatial axes, using a reproducible nontrivial **complex64**
  frozen `[H,W//2+1,C]` global filter. Spectral dimensions, patch geometry,
  output channels, filter coefficients, norms and classifier are immutable.
- Only the real-valued channels-last GELU MLP at
  `channel:gfnet.block0` grows. True full-image FFT/IFFT forward/backward,
  strict complex-buffer/weight frozen hashing, signed mid-batch restart,
  14 canonical ADP IDs and paired native variance gates share the prior
  controller and opt-in `vision_v11` worker.
- Same 43 route/criterion jobs per seed/split/repeat; pending CI.
- Both are **tiny randomly initialized architecture controls**, not published
  pretrained ResMLP or GFNet checkpoints. No CIFAR benchmark results claimed.

## V07 and V10/V11 selected CPU verification (2026-11-02)
- V07 explicit synchronous both code `dafc5c8944af91339bc76c1f6a830b4949fb906a`,
  corrected cap test `dc4d91d1a9a33769171ea9467d779f7f57a5414f`.
  CI https://github.com/Anurag9000/simple-adp/actions/runs/41409201374:
  **303 passed in 28.75 seconds**. Synchronous growth of both Mixer MLPs
  is a separate named site-policy ablation, not a canonical method change.
- V10/V11 code `71b0011a31ad7bc87c8e87f898f9a7a7da5a7307`.
  CI https://github.com/Anurag9000/simple-adp/actions/runs/41409409751:
  **315 passed in 42.53 seconds**, including actual ResMLP fixed token-linear
  and GFNet complex FFT filtering, native local growth, runner and resume.
- All are bounded synthetic CPU tests, not real image scientific experiments.

## V08 independently selected multistage residual adapters (pending CI)
- New `src/simple_adp/visual_multistage.py` freezes a fixed three-stage CNN,
  inserting a real residual MLP at each predeclared early/middle/late stage.
  Exactly one is ADP-eligible and trained per job; the other two remain frozen
  but participate in the forward/autograd graph. No in-place CNN mutation.
- Distinct `adapter:early`, `adapter:middle`, `adapter:late` growth sites,
  stored with exact stage/selected-MPL weights, site-specific signed
  checkpoint namespace and model architecture. All growth operators remain
  local to the selected real MLP.
- Three independent opt-in runner adapters `vision_v08_early`,
  `vision_v08_middle`, `vision_v08_late` enumerate source 14 method IDs,
  faithful/corrected generic profiles and paired loss/variance candidates:
  43 per selected site/split/seed/repeat, 129 total across three sites.
- New tests cover gradient flow through frozen later CNN stages, exact
  frozen weights, RR/all/D1/Dcur, saved/restored architectures, eviction
  replay, fixed/generic controls and paired worker. **CI pending**.
- Simultaneous multistage growth, learned site selection, pretrained
  backbone weights, real-image scientific validation and V09 bank are not yet
  implemented.

## V08 fixed three-stage site search verified — 2026-11-02
- Code commit: `916c560c94a2d1d474f6e5e7c356fa4786ecb1a5`.
- Workflow: https://github.com/Anurag9000/simple-adp/actions/runs/41410224145.
- **331 selected CPU tests passed in 53.35s**, including three independently
  selected real residual sites, exact source architecture growth/restart,
  coordinated worker and original-versus-variance counterparts. Separate
  loss/variance contract CI also succeeded.
- No adaptive multi-site selection or real pretrained CNN experiment.

## V09 strict hard-routed adapter bank (selected CPU CI verified)
- `visual_adapter_bank.py` implements a frozen CNN, shared frozen in/out
  projections and classifier, **three real residual local MLP members** and
  immutable hard argmax K-way router based solely on CNN features. No label
  or test-time task-id leakage into routing. Zero learnable gate parameters.
- Each job selects one eligible member `bank:0`, `bank:1`, `bank:2`;
  all other member weights, gate parameters and fixed CNN state are frozen,
  unchanged under growth and signed into the checkpoint. Source RR/all/D1/
  Dcur operators mutate only that member's hidden units, preserving bank
  count, routing behavior and fixed external interface.
- Three independently registered opt-in worker adapters
  `vision_v09_bank0/1/2` enumerate all 14 source ADP IDs plus the eight
  additional generic faithful/corrected profiles and 21 matched variance
  counterparts (43 jobs each, 129 total per split/seed/repeat).
- Per-run `router_kind`, `router_histogram_train` and
  `router_histogram_test` are persisted to result.json with the original
  split, architecture and frozen-state identity. These are **occupancy
  diagnostics**, not learned routing accuracy or head-selection metrics.
- Signed model/checkpoint snapshots carry the router semantics and bank
  site; mid-batch interrupted resume rejects changed member/settings/data.
  Unit+runner tests are **pending CI**. This is a small random-initialized
  hard-routed adapter-bank control, not a trained mixture-of-experts gate,
  pretrained architecture or real-image benchmark.

## V09 verified results — 2026-11-02
- Code: `1d5c1ae3472671fe162b47a97ac73df4a6ab2b5e`.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41413752172
- **347 selected CPU tests passed in 67.91s**, including all three
  strict bank members, hard-routing immutability, no nonselected gradients,
  local ADP growth, signed interruption/restart, and coordinated original/
  variance plus generic worker jobs with route occupancy JSON.
- Standalone ADP original-versus-variance contract workflow also passed.
- This is a frozen hard router; no learned gate, new-member creation,
  pretrained backbone, real-image scientific experiment or CUDA validation.

## V08 predefined multisite stage schedule (verified selected CPU CI)
- New `visual_multistage_schedule.py` holds three genuine trainable local
  C-to-C residual MLPs inside a fixed three-convolutional-stage CNN, with an
  explicit immutable site order `early -> middle -> late`.
- The outer policy rotates the **site** by completed growth proposal index.
  The original 14 source ADP operators still choose hidden layers *within*
  the selected MLP; they never silently grow all eligible backbone sites.
  Per-candidate only the active MLP has trainable parameters; earlier selected
  site weights stay committed and frozen until the schedule selects them again.
- New `vision_v08_schedule` source family has 22 original method/profile
  jobs and 21 paired variance counterparts per seed/split/repeat, in addition
  to V08's 129 separately signed fixed-site jobs (172 total V08 jobs).
- Every signed snapshot stores the three distinct per-site hidden widths,
  weights, within-MLP cursors, active site and fixed backbone checksum.
  The controller records the actual site in every result history entry, uses
  whole-bank (3 local MLP) neuron capacity and supports exact mid-stage
  interruption/replay. Worker exports `site_schedule_kind` and
  `site_schedule_trace` to result.json.
- This is a **predeclared round-robin site policy**, not learned site routing,
  validation-based adaptive site choice or the canonical ADP-W-All definition.
  Synthetic CPU tests, including a mid-middle-site eviction and exact resume,
  are added but **CI is pending**. Pretrained source and real-image science
  remain absent.

## Verified V08 predefined site schedule — 2026-11-02
- Implemented: `53f7d7a33b6a78439363bb9c7419aedacf3d0e99`.
- Snapshot syntax corrected by `73db80bfd43229762f0a0158d356f6f00e639716`
  and `98bfd9f78339cb77b16c9214a2dba944ebac546f`.
- Selected workflow: https://github.com/Anurag9000/simple-adp/actions/runs/41415527242
- **354 CPU tests passed in 69.35 seconds**; early/middle/late
  fixed schedule, exact per-site weights/cursors, signed mid-middle-site
  interrupted recovery, source variants, coordinated runner and matching
  variance jobs. The separate source original-versus-variance workflow
  also succeeded.
- A prior CI run at `53f7d7a3` had failed at collection due to a
  malformed Python dict comprehension in the new snapshot restore.
  The correction is in current main; the above test run confirms success.
- Scientific image training, pretrained source implementations, full
  repo-wide regression and CUDA correctness remain unverified.

## Complete opt-in source/variance route-identity closure (verified CI)
- Added a **no-training** manifest regression over all 12 architectures and
  19 explicit architecture/site-policy families, including V07 three scopes,
  V08 four scopes and V09 three bank members.
- Expected for one vision toy task/split/seed/repeat: **19 × (22 source
  method/profile routes + 21 matched shadow-BGD variance counterparts) =
  817 distinct job identities**. Fixed MLP controls never get a fake variance
  companion. IDs, split hashes, fold, source method/profile and seed must
  match each paired criterion. Training/test datasets must stay isolated.
- These are planned / enumerated job descriptors only, **not** 817 completed
  training runs or scientific results. test_runner_vision.py verifies closure;
  Selected CI passed. This does not yet establish the larger dimensions of task
  sequences, real CIFAR datasets, multi-seed/repeat scientific validation or
  GPU execution.

## Verified exhaustive vision routing manifest — 2026-11-02
- Source test commit: `0ec34b7324a7b60a01dec70be45065298f229ae2`.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41416422457
- **355 selected CPU tests passed in 50.96s**, including the precise
  `19 scopes × (22 original routes + 21 variance counterparts) = 817`
  opt-in `vision:toy` per-one-split/seed/repeat job identities.
- Test verifies uniqueness, canonical method/profile, matching split SHA,
  seed/fold/repeat, no dummy fixed-model variance jobs and correct paired
  source method/criterion provenance.
- This is **not 817 completed training jobs**. No complete real dataset,
  GPU or repository-wide experimental validation is claimed.


## Official image source V01 experiment integration (pending selected CPU CI)

- \`src/simple_adp/vision_experiment.py\` provides a bounded one-job CLI over
  locally available official torchvision CIFAR-10/CIFAR-100/Fashion-MNIST files.
  A complete local pretrained-claim-*unverified* source model state_dict is
  supported via the existing frozen ResNet18/ConvNeXt-Tiny/ViT-B16 factory.
  Random backbones require \`--allow-random-backbone\`; no hidden downloads.
- \`VisionInputAdapter\` is a frozen deterministic grayscale→RGB repeat and
  32×32 / 224×224 bilinear resize with declared geometry and no augmentation.
- Only the official training source enters candidate fit/validation. Official
  test materialization/evaluation happens **after** architecture fit, with
  separate validation and test metrics. The V01 frozen-feature head alone
  trains; no pretrained internal FFN mutation is claimed.
- Per-job manifest stores raw sample/split indices, source data SHA256,
  image transform and encoder SHA256, local checkpoint origin/checksum,
  method/profile/settings and explicit run signature. A signed estimator
  checkpoint, inference artifact, model checksum and atomic result JSON
  support idempotent resumption and reject cross-job identity collisions.
- New synthetic fixture unit tests cover preprocessing, official-test order,
  random source opt-in, artifact identity, checksum tamper detection.
  **Selected CPU CI pending**; no real dataset/pretrained experiment executed.

## Verified real-data source integration (2026-11-02)

- V01 experiment CLI code `78b2a0d12e7c9902f0e8d1f41548f8a7797f422e`:
  **360 selected CPU tests passed in 44.73s**,
  https://github.com/Anurag9000/simple-adp/actions/runs/41450351931.
- External network smoke `cefd74e3b0bf861835437ab830b2101303e8ea1a`:
  https://github.com/Anurag9000/simple-adp/actions/runs/41451103405
  completed on official Fashion-MNIST and published
  `ResNet18_Weights.IMAGENET1K_V1` with full source-state SHA256.
- Data counts: 40 fit, 20 validation and 20 official test. Validation and
  test accuracy each **0.15**, test macro-F1 **0.095**, test log-loss
  **4.2930310860**. This is a tiny smoke test, **not** benchmark accuracy.
- GitHub Actions artifact `fashion-mnist-official-resnet18-v01-evidence`
  (ID 13144598582) contains JSON manifest/results, external provenance,
  and the model plus identity sidecar. Detailed record:
  `docs/visual_adp/REAL_IMAGE_EVIDENCE.md`.
- Official local checkpoint loader continues to label the source weights
  as unverified local state; the smoke workflow separately records the
  official torchvision pretrained enum, download URL and exact checksum.
- V03–V11 genuine pretrained inner-FFN integration and multi-seed benchmarks
  remain outstanding. A CIFAR-10 RGB smoke extension is underway.
