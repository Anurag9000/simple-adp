# Visual ADP implementation ledger

**The top table is the current summary.** Later chronological sections preserve
historical development milestones, including past pending-CI statements that
were subsequently verified by newer test evidence.

States distinguish source/static implementation from executed tests and science.

| Item | State | Evidence / missing |
|---|---|---|
| Coordinated held-out test-indexing isolation (V01 and V02-V11) | runtime-tested bounded CPU | Held-out X/y reads moved after fit in runner/vision.py; guarded row-access regression in tests/test_runner_vision_test_isolation.py; dedicated GitHub Actions run 41491589807 succeeded; source V03/V04 runner already defers |
| Read-only visual parameter accounting in V01-V11 and source V03/V04 workers | runtime-tested bounded CPU | src/simple_adp/visual_accounting.py, tests/test_visual_accounting.py and coordinated V01/V02 report assertions; dedicated run 41492546181 succeeded; not whole-model FLOPs |
| Real Fashion-MNIST controlled architecture/site matrix | verified bounded official-image CPU execution, not full benchmark | 38/38 paired fixed/ADP-W-RR jobs across 19 architecture/site adapters ran successfully through SuitePlan and signed worker, after fixing global cap settings and worker accounting serialization; run 41493795442; artifact 13166111100 |
| Measured single-image Linear/Conv2d inference MACs | focused CPU runtime-tested; latest integration CI pending | visual_compute.py runs the selected composite with token/spatial repetition and restores mode/RNG; excludes fused attention, FFT, pooling, etc., never claims full FLOPs; focused run 41575556560 and 38-job run 41575563341 passed before latest budget/calibration changes |
| Native visual training/search budget and D1 warmup cost | full CPU test suite verified; actual official-image worker tested | visual_search.py signed optimizer steps, epochs, candidate parameter totals and D1 catch-up costs; 38/38 official-image native worker run 41575726025 passed; focused V02 worker budget assertion run 41575803153 passed; extended D1 exact restart test awaits full-suite result; original V01 head controller unchanged |
| Dense multi-head attention arithmetic reporting | focused CPU + 38-route official-image verified | visual_compute.py separates observed nn.Linear/nn.Conv2d MACs from analytic nn.MultiheadAttention projection and QK/AV MACs; attention unit and 38/38 Fashion run 41582544787 passed; incomplete full FLOPs |
| Paired fixed/RR scientific budget audit | actual official-image CPU integration verified | visual_budget_audit.py rejects missing/mismatched identities, flags unequal or unknown optimizer work, and refuses full-compute parity claims; official 38/38 run 41582755349 passed with this audited summary |
| Official RGB CIFAR-10 and CIFAR-100 controlled worker matrices | verified bounded official-image CPU execution, NOT full science | 38/38 official CIFAR-10 jobs passed (run 41583545865) and 38/38 full 100-class CIFAR-100 jobs passed (run 41583684320); original 19 visual sites × native Fixed/RR; tiny one-seed, one-epoch-per-candidate source data; not pretrained V05–V11 parity |
| Dtype-aware parameter/buffer logical payload accounting | focused 129-test CPU suite verified, actual official-image workers verified | src/simple_adp/visual_memory.py, tests/test_visual_memory.py; frozen/trainable/buffer dtype and device byte partitions; explicitly not peak RAM/VRAM |
| Signed optional model payload candidate admission | focused 127/129 CPU tests verified; no physical CUDA/OOM proof | Native VisualResidualSearch option and RunnerSettings visual_model_tensor_payload_cap_bytes, checkpoint policy signature, max initial/candidate bytes before StageTrainer; V01 legacy cap fails closed; candidate cloning/activation/optimizer memory excluded |
| Per-stage candidate model payload timeline including D1 warmups | focused CPU and bounded Fashion signed worker verified | trained_model_memory_trace signed for native source/controlled searches, full history of *trained stages* with completeness flag, new exact-resume tests; distinct from acceptance history |


| Post-selection multiclass calibration on visual jobs | implemented, latest CI pending | visual_metrics.py adds 10-bin top-label ECE and multiclass Brier using ClassificationTask label map; never changes training or architecture-validation objectives |
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
| V03 original torchvision ViT-B16 source FFN | real published-pretrained in-block width growth verified | 3072→3073 hidden, official Fashion-MNIST 20/10/10, real run 41474035681; 43 separate opt-in source planner routes verified |
| V04 original torchvision ConvNeXt source FFN | exact source-tensor equality (4 tests) AND published-pretrained real-data growth verified | first true pretrained in-block FFN 384→385 on Fashion-MNIST; distinct from tiny vision_v04 route |
| V05 PoolFormer channel FFN | runtime-tested (selected CPU CI) | 275 selected tests passed, synthetic toy worker and native growth |
| V06a S2 / V06b axial shift FFN | runtime-tested (selected CPU CI) | 288 passing selected tests, real zero-padded shift operators, source growth and paired workers |
| V07 MLP-Mixer | all 3 token/channel/synchronous-both scopes CPU CI verified (303 tests) | true residual token/channel MLPs, signed resumed search, matched variance jobs and joint budget |
| V08 fixed three-stage residual adapter | independent sites + fixed multisite schedule verified (354 tests) | early/middle/late local growth, saved site order/weights and exact middle-site resume |
| V09 strict adapter bank | selected CPU CI verified: 347 passed | hard frozen router, three strictly isolated adapter targets, routing counts and matched workers |
| V10 ResMLP / V11 GFNet | selected CPU CI verified (315 tests) | fixed dense token/affine mixer and complex FFT filter, native channel ADP and paired worker |
| Local official-image V01 experiment CLI and deterministic grayscale/RGB transform | selected CPU CI verified; 360 passed | signed source/run artifacts and official test after fit; synthetic unit fixtures |
| Official Fashion-MNIST + published pretrained ResNet18 micro-smoke | real-data integration verified | 40/20/20 official image partitions, GitHub Actions run 41451103405, uploaded model and provenance; **not** a performance benchmark |
| V01 matched loss/variance vs fixed real-image comparison | three-seed CPU real-data runs verified | 9 predeclared jobs on official Fashion-MNIST, pretrained ResNet18, 300/100/100 per seed; tiny controlled study, not benchmark-quality |
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

## V01 exact interruption recovery and detailed result records (2026-11-02)

- `c183aed986190f00cb68822a4f4bea4f9596b140`: adds deliberate
  mid-training eviction and exact restart to the real-image experiment
  wrapper; selected GitHub Actions
  https://github.com/Anurag9000/simple-adp/actions/runs/41452649913
  passed **361 CPU tests in 47.32s**. Selected architecture, validation/test
  metrics and search-history count matched the uninterrupted baseline.
- `8e2d618b67d2db27fb0de298202c8ce31aa6d714`: stores full
  candidate search history in checksummed `search_history.json`,
  class-ordered confusion matrix and source/head parameter counts.
  New offline tests verify generated files and reject history tampering.
  Selected CPU CI passed (361 tests); pre-existing Fashion smoke artifact predates
  the separate history file and is not retroactively modified.

## Verified checksum-guarded V01 experiment search-history artifacts
- Code: `8e2d618b67d2db27fb0de298202c8ce31aa6d714`.
- CI: https://github.com/Anurag9000/simple-adp/actions/runs/41453060345
  **361 selected CPU tests passed in 64.78 seconds**, including added
  search_history.json candidate trace, class-ordered confusion matrix,
  source/head parameter counts, tamper rejection and exact interruption
  recovery. The original loss/variance contract CI also passed.
- The earlier real Fashion-MNIST artifact predates the richer result JSON.
  Do not fabricate the missing historical stage trace; only new jobs produce
  the expanded schema automatically.

## Verified paired V01 real-data and split provenance closure — 2026-11-02
- New `vision_comparison.py` + `simple-adp-vision-compare` CLI registers a
  fixed MLP head and matched loss/variance `ADP-W-RR` jobs for each seed,
  with independent signed checkpoint/model/result paths.
- Comparison manifest signs all seed-specific source partitions, source weight
  identity, settings, method/criterion routes and data limits. Every result
  is checksum verified and must preserve the same frozen source and split
  across the three methods within each seed. Official test results are
  reported *descriptively*, never used to choose the best architecture.
- First verified one-seed Fashion-MNIST pretrained ResNet18 run:
  https://github.com/Anurag9000/simple-adp/actions/runs/41462840053,
  3 real jobs, 300 train/100 validation/100 test. Fixed/loss/variance
  official-test accuracy respectively 0.28/0.54/0.30.
- Expanded **3-seed, 9-job** run verified:
  https://github.com/Anurag9000/simple-adp/actions/runs/41463589627,
  seeds 11/23/37, per-method test accuracy fixed [0.28,0.38,0.35],
  loss [0.54,0.57,0.57], variance [0.30,0.38,0.34]. Predeclared
  source and criterion. No inference of statistical superiority from
  three small seeds. Artifacts retained on GitHub Actions (30-day retention).
- Signed offline paired contract: 15 passing tests at
  https://github.com/Anurag9000/simple-adp/actions/runs/41463404061;
  full selected CPU suite: **365 passed** at
  https://github.com/Anurag9000/simple-adp/actions/runs/41463404204.
  Earlier `41462977800` had a fixture validation-count error only, corrected
  by `5a1d09e31958026277b6c5428e61f3d229869b8b`.
- Attempted official CIFAR-10 download in `41451620431` timed out at about
  60%; model training never ran. Automatic Fashion job and on-demand
  timeout-bounded CIFAR jobs are now independent. The separated Fashion
  workflow succeeded at https://github.com/Anurag9000/simple-adp/actions/runs/41462613451.
- Still missing: pretrained source internal FFN replacement (V03–V11),
  larger real-data benchmark, full CUDA and repository-wide regression.

## Source-exact first ConvNeXt-Tiny FFN architecture growth (2026-11-02)

- New `visual_source_convnext.py` loads a **complete strict local source**
  `torchvision.models.convnext_tiny` state_dict, enforces file SHA256 and
  replaces exactly the first CNBlock's original
  `Linear(96,384) -> GELU -> Linear(384,96)` with a trainable,
  source-weight-transplanted `TabularMLP(96,[384],96)`.
- The initial `source.features(...)` tensor is provably **exactly equal**
  to the original complete torchvision source model on deterministic images,
  before local ADP mutation. All other source parameters/stochastic-depth
  behaviors are frozen, while a separate fixed-interface task head trains.
  Only the selected FFN hidden units are ADP growth targets.
- Full source tensor hash + loader checkpoint SHA256, site, width cursor,
  model/trainer state and target head are recorded for strict recovery.
  The existing `VisualResidualSearch` now dispatches a separate source-V04
  model kind; it requires starting from full source hidden width **384**.
- Dedicated actual-torchvision source tests passed **4/4 in 9.97s**:
  https://github.com/Anurag9000/simple-adp/actions/runs/41465492673.
  Evidence: exact source feature equality, trainable local/head gradients,
  frozen-source isolation, native width/depth growth, strict signed snapshot
  and one full-image StageTrainer run. Their checkpoint was full but
  **randomly initialized**, so this is source-architecture fidelity, *not
  published pretrained experiment evidence*.
- Explicit network-enabled experiment on official pretrained
  `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` and Fashion-MNIST is committed at
  `7d380260066318dabc83ccd7695dcba754f2d43f`, **verified successful**.
- Source V04 is not the older synthetic `vision_v04` job adapter. The
  former's actual full-source search and local pretrained weight use must
  be verified independently before replacing or adding source V04 to the
  central 817-job manifest. Source-faithful V03/V05–V11 remain unfinished.

## First actual published-pretrained V04 in-block source FFN growth — verified

- https://github.com/Anurag9000/simple-adp/actions/runs/41466060451
  completed real official Fashion-MNIST execution with
  `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` source, strict complete
  state_dict and exact initial first-CNBlock `96→384→96` GELU transplant.
- The native one-proposal `ADP-W-RR` controller trained the original
  internal source FFN and a fixed-interface target-specific head, selected
  width **385**, and strictly preserved all other ConvNeXt source weights.
- 40 training, 20 validation, 20 official-test images, one epoch per
  stage, seed 11. Validation accuracy **0.40**; official-test accuracy
  **0.45**, macro-F1 **0.4222**. A small **integration smoke**, not a
  benchmark or multi-seed scientific ranking.
- Published weights provenance URL:
  https://download.pytorch.org/models/convnext_tiny-983f1562.pth.
  Full source saved state SHA256:
  `48f10ea8765c304f933e395d3676017eb388634ff1a2808b8e4db08f6c91c272`.
- Selected source-model snapshot, search history, manifest and result JSON
  uploaded as artifact `real-v04-convnext-source-inner-ffn`, ID
  **13151101792** (103MB). Full result and caveats are in
  `docs/visual_adp/REAL_IMAGE_EVIDENCE.md`.
- This is a **separate** source-faithful research path, not the tiny V04
  option currently counted among the original 817 coordinated jobs.
  Remaining: complete source-V04 central runner matrix, other architectures'
  pretrained inner FFN transplants, GPU/full-suite checks and larger
  pre-registered experiments.

## V04 source-faithful central job planner and worker — verified selected CPU source tests

- `src/simple_adp/runner/vision_source.py` introduces a **separate**
  `vision_v04_source` identity. It is registered only when both
  `SIMPLE_ADP_ENABLE_VISION=1` and `SIMPLE_ADP_ENABLE_V04_SOURCE=1`.
  It requires `SIMPLE_ADP_V04_SOURCE_WEIGHTS` (complete original full
  ConvNeXt-Tiny state_dict) and `SIMPLE_ADP_V04_SOURCE_SHA256`.
- The exact 20-character prefix of the verified complete source SHA256
  is embedded in each `ModelRoute` label and therefore in planned
  run identities. A worker refuses a changed or missing source file.
  Training resumes only from a matching checkpoint/data/settings signature.
- The source runner enumerates 22 original native source-method/profile
  routes and 21 native variance-stagnation counterparts per one split/seed.
  The existing source-unaware 817-job matrix remains unchanged when
  source support is disabled. A separately gated plan contains 860 jobs
  (817 earlier + 43 new source).
- Source route starts at hidden width 384, depth 1 and never substitutes
  a tiny proxy; all 14 native ADP families remain distinct. Nonzero
  finite max_candidates and max_epochs are mandatory for this expensive
  28M-parameter source model. Inference artifacts and worker JSON include
  source checksum, frozen SHA256, site, architecture and growth criterion.
- A new direct worker test exercises a complete original **random
  torchvision state_dict** (no published pretrained claim), signed
  planning and one real full-source stage, with separate source pairs.
  This integration passed 3/3 source-specific CPU tests at
  https://github.com/Anurag9000/simple-adp/actions/runs/41470335656.
  The expanded selected visual CPU suite passed **373 tests** at
  https://github.com/Anurag9000/simple-adp/actions/runs/41470335650.
  The already
  verified published-pretrained single V04 full-image experiment is a
  separate successful standalone execution, not proof these 43 planned
  source runner jobs have been trained.

### Verified source V04 coordinated 43-job route closure — 2026-11-02
- Integration code and tests: `c5d4ed0bc4340af1680cd9079304c0038cc275e5`,
  fixture-limit correction `fa1016bc554233734ac24f64c6ab332f02783544`.
- Dedicated GitHub Actions: https://github.com/Anurag9000/simple-adp/actions/runs/41470335656.
  **3/3 source-runner tests passed in 21.36s**.
- Verified source disabled => exact original **817** planned jobs; explicit
  source enabled => **860** jobs (43 additional; 22 source original and
  21 native variance pairs). One source-V04 model trained and produced a
  source-hashed inference bundle via the actual coordinated CPU worker.
- Worker test used an original full **random-state** torchvision ConvNeXt
  checkpoint. The published pretrained independent real-image experiment is
  separately recorded; neither test executed all 43 source V04 jobs.

### Complete selected visual CPU regression after source-V04 integration
- GitHub Actions https://github.com/Anurag9000/simple-adp/actions/runs/41470335650:
  **373 tests passed in 112.18 seconds** (CPU). The new source runner,
  previously working 817-job opt-in matrix, V01–V11 controlled families,
  V04 original source fidelity and exact resume, paired V01 comparison,
  and existing source/native method tests remained green.
- Dedicated source-specific test workflow:
  https://github.com/Anurag9000/simple-adp/actions/runs/41470335656,
  **3 passed in 21.36 seconds**. This selected test set is not a claim
  of full repository regression or GPU correctness.

## V03 original ViT-B/16 source architecture and coordinated runner (2026-11-02)

- Source `visual_source_vit.py` strict-loads a **complete** torchvision
  ViT-B/16 checkpoint; checks its exact local SHA256, and transplants
  original first transformer block `encoder_layer_0.mlp`'s
  **768→3072→768** GELU FFN into an ADP-expandable `TabularMLP`.
  The rest of the real 12-block ViT, attention, positional embedding,
  normalization, and fixed image-patch topology are preserved and frozen.
  The task head is replaced with a separately trainable fixed-interface
  classifier that never participates in ADP architecture growth.
- The source ViT has fixed 224x224 positional geometry; deterministic
  raw-grayscale repeat, bilinear resize to 224x224 and ImageNet mean/std
  normalization form part of the source model. One local FFN site only.
  Original source initialization must use hidden width **3072**, depth 1.
- Dedicated full-source fidelity test suite:
  https://github.com/Anurag9000/simple-adp/actions/runs/41472976307
  **4 passed in 23.28s** (random official torchvision architecture weights),
  including exact whole-encoder feature equivalence before mutation,
  strict tensor transplant, width growth, signed snapshots, frozen-state
  corruption detection and a native full-image training stage.
- Source runner `vision_v03_source` requires both
  `SIMPLE_ADP_ENABLE_VISION=1` and
  `SIMPLE_ADP_ENABLE_V03_SOURCE=1`, plus a complete existing source
  checkpoint and `SIMPLE_ADP_V03_SOURCE_SHA256`. Source SHA256 prefix is
  embedded in route identity. Finite search candidates and epochs are
  mandatory due ~86M source model parameters.
- With V03 source disabled the **817** original jobs are unchanged; a
  separately enabled single-source V03 run produces **860** planned
  routes (22 original, 21 matched variance counterparts). Dedicated CI
  https://github.com/Anurag9000/simple-adp/actions/runs/41473641002
  **3 tests passed in 51.52s**, including an actual coordinated full ViT
  worker on tiny offline images. The source test weights were **random**;
  43 registered jobs are not 43 training runs.
- Published pretrained `ViT_B_16_Weights.IMAGENET1K_V1` real
  Fashion-MNIST source FFN job launched through network-enabled
  https://github.com/Anurag9000/simple-adp/actions/runs/41474035681,
  **pending** execution verification; no accuracy claim until success.
  This route is separate from tiny randomly initialized `vision_v03`.

## Verified published-pretrained V03 internal transformer FFN expansion

- Official published source `ViT_B_16_Weights.IMAGENET1K_V1`
  loaded strictly by complete state_dict SHA256
  `335d1dfe0a7aac5787ffdc111e46843f7e2223218f49ab989333b880f77a427e`.
- https://github.com/Anurag9000/simple-adp/actions/runs/41474035681
  **succeeded**: genuine first ViT-B16 FFN went from **3072→3073**
  hidden units, all other pretrained ViT stages frozen, source frozen
  state SHA256 recorded.
- Official real Fashion-MNIST 20/10/10, seed 11, 1 epoch per stage:
  validation accuracy 0.40, official-test accuracy 0.40, test macro-F1
  0.30, test log-loss 1.9859184127. One tiny integration run, **not**
  a trustworthy generalization performance estimate.
- Uploaded four files (manifest, result, search history, selected
  complete source model) in artifact `official-fashion-vitb16-v03-source-ffn`
  ID **13157600366**. Details in `REAL_IMAGE_EVIDENCE.md`.
- V03 original-source full-model **mid-batch** interruption/restart
  regression is newly added (commit `43269193141177c9b4661eccf6f5a7d6585912bd`)
  and **passed 5/5 dedicated source-V03 tests** at
  https://github.com/Anurag9000/simple-adp/actions/runs/41474722199
  (94.01 seconds), including exact interruption/relaunch parity.

## Verified signed full-model V03 mid-batch resume (2026-11-02)

- Exact source-V03 resumed checkpoint test added to
  `tests/test_visual_source_vit.py` in
  `43269193141177c9b4661eccf6f5a7d6585912bd`.
- https://github.com/Anurag9000/simple-adp/actions/runs/41474722199:
  **5/5 passed in 94.01 seconds**. Interrupted after the first
  training batch, restored all original ~86M ViT source weights plus
  trainable local FFN, batch cursor, optimizer state and RNG, matching
  the uninterrupted selected architecture, history, search score,
  frozen source hash, and per-example prediction probabilities exactly.
  Changed source/search signatures fail closed on recovery.
- A new **whole repository** CPU pytest workflow was added in
  `7bef111fe251cd6219313dad81ff258dd214e6d3`; it runs
  `pytest -q tests` with every declared optional test dependency
  rather than just the selected visual regression tests. Initial
  workflow https://github.com/Anurag9000/simple-adp/actions/runs/41475398340
  is still pending. No full-suite pass count has been established.

## Selected visual suite after V03 full-source exact-resume coverage

- https://github.com/Anurag9000/simple-adp/actions/runs/41474722290
  **381 passing selected CPU tests in 233.07 seconds**. Includes
  new V03 real ViT original-source FFN, 43-job coordinated ViT
  source integration, and signed mid-batch full-model recovery,
  without breaking older visual architecture tests.
- Full `tests/` regression is distinct and was still active at
  https://github.com/Anurag9000/simple-adp/actions/runs/41475398340
  when this section was written. No full-suite completion is implied.


| Pre-construction local FFN memory payload projection | 127 focused CPU tests + source V03/V04 genuine model CI verified | native source growth geometry exact forecasts before constructing oversized candidates; all 19 controlled sites tested; ViT CI 41592646848 / ConvNeXt CI 41592638223; still not initial-model or peak memory |
| Nominal dense Adam/AdamW gradient and moment byte estimates | 129 focused CPU tests verified; not measured peak | hypothetical model+gradient+2 moments per requires_grad parameter; exact CPU optimizer-state comparisons, excludes activations/step scalars, not VRAM assertion |
| Official multi-seed fixed/RR job matrix | 76/76 genuine official-image worker jobs verified | 19 controlled sites x fixed/RR x 2 independent derived seeds, native SuitePlan worker, per-seed identity and budget audits; CI 41593036221; tiny data/epochs, not science |
| Optional step-balanced fixed/RR epoch policy | 38/38 official-image workers passed, 18/18 native pairs with equal 6-step budgets, V01 unknown | real Fashion run 41593647892, artifact 13211018409, zero mismatches, no full FLOP matching or accuracy superiority claim; fixed two epochs vs RR two single-epoch stages; original controllers unchanged |

| Observed OS RSS and CUDA allocator fit-call resource window | 154 focused tests, 38/38 real Fashion worker reports verified; CUDA hardware pending | visual_resources.py samples Linux current RSS/process-lifetime VmHWM and optional current CUDA PyTorch allocation/reservation counters before/after V01–V11 and source V03/V04 fit; never resets CUDA highwater or claims interval peak; signed worker and three official-image matrices; tests/test_visual_resources.py |


| Fresh-process signed visual memory experiment | 38/38 Fashion jobs passed, all 38 PIDs distinct, all 38 nested worker logs archived | corrected workflow 41606221098 artifact 13217047019; native 18/18 same-step pairs, V01 unknown; VmRSS/VmHWM are whole-process, not exact training peak |
| Failure-path process-resource diagnostic | implemented, focused and full-suite CI pending | visual worker catches original failure, records best-effort RSS/CUDA resource window without overriding original cap/model exception; missing metrics never treated as zero or block the original failure |


| Paired descriptive process RSS evidence | implemented, official isolated CI pending | visual_resource_audit.py refuses shared-process/mismatched dataset split/seed/site and unknown/fabricated RSS, reports signed fixed/growth whole-child highwater and fit-endpoint differences without causal/statistical/VRAM claims |


| Read-only paired visual classification and calibration analysis | 76/76 genuine 2-seed official Fashion workers, 19-site post-run report verified; 201 focused CPU tests passed, 2 CUDA-only skipped | `src/simple_adp/visual_paired_metrics.py` + `scripts/visual_paired_metrics_analysis.py`; validates unique complete signed job metadata, dataset and split identity across seeds, metric direction/range, independent per-site pairing; report artifact 13222170910 in CI 41616572651; no confidence intervals, p-values, pooled families, equal-FLOP or scientific gain claim |
| Official Fashion/CIFAR10/CIFAR100/step-balanced/isolated pair reports | Fashion and step-balanced CI verified; newer RGB/isolated CI in progress | each original worker matrix runs the same read-only posthoc analyzer and archives the original matrix alongside `*_paired_metrics.json`; no controller/model/dataset access at analysis time |
| Full-repository CPU checkpoint disk-pressure diagnostics | committed, latest full-suite CI pending | 41616499845 completed with 19393 passing and 2 PyTorch checkpoint write errors (`basic_ios::clear`), consistent with insufficient ephemeral runner storage; subsequent `.github/workflows/full-repository-cpu-regression.yml` releases unrelated hosted-VM SDK caches and captures `df/du` on failure without skipping any full-source ViT tests; no disk diagnosis claimed proven until new CI |


| Cross-file original worker result provenance for paired visual metrics | implemented, prior 76-worker raw archive manually inspected; new CI pending | `visual_paired_metrics.verify_visual_worker_result_artifacts` and `--results-root` cross-check original signed result.json metrics, dataset/split/seed/site, unique job IDs, **shared suite-plan** run_signature and SHA256 bundle fingerprint; path traversal/metric tampering/source overwrite tests; all six existing official-image workflows now enforce it |
| Full CPU suite ephemeral checkpoint disk-pressure recovery | CI workflow committed; unfiltered full-run outcome pending | failure 41616499845 was 19393 passed / 2 skipped / 2 PyTorch checkpoint I/O errors, likely hosted-runner storage but not explicitly ENOSPC; all tests preserved, free unused SDKs and add df/du diagnostics in `.github/workflows/full-repository-cpu-regression.yml` |


| Actual native StageTrainer minibatch forward MAC ledger (opt-in) | code implemented; newest focused and full-repository CI pending | `visual_train_compute.py` observes executed nn.Linear/nn.Conv2d forward operations and separately analytically estimates declared MHA dense operations; signed per-batch checkpoint counts, no extra forward, validation/shadow/test excluded; backward/optimizer/full FLOPs explicitly unavailable |
| Per-stage signed native search forward-work timeline | implementation and exact-restart/D1 regression tests committed, verification pending | `VisualResidualSearch(track_training_forward_macs=True)` persists stage totals and aggregate across initial, growth and D1 warmups; signature unchanged when disabled; old unmeasured resumes never counted as zero |
| Original worker paired actual-forward-work audit | implementation and strict provenance tests committed, official-image CI pending | `visual_training_work_audit.py` checks original dataset/split/seed/site, aggregate-stage closure and optimizer-step equality; reports growth/fixed *partial forward* MAC ratios, V01 unknown, full-training FLOP equivalence null |
| Official opt-in training-forward work experiment | code complete, 38-worker official Fashion-MNIST run pending | `.github/workflows/visual-official-forward-work.yml`; `--step-balanced --observe-training-forwards`; archives original signed result.json, staged measured work and metrics; not a full-dataset/matched-FLOP science result |
