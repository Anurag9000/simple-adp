# Results and comparison protocol (future vision jobs)

Use dataset-indexed disjoint train / architecture-validation / test splits.
Record dataset source/revision/license, split indices/checksums, pretraining
dataset and checkpoint digest, normalization, augmentations, seeds, model/site,
method/profile, growth criterion, optimizer budget and training hardware.

Pair each ADP job against same-initialization MLP-Fixed with identical training
data, initialization, encoder and compute budgets. Log candidate sequence,
global-best selection, top-1 and balanced accuracy, NLL/calibration when valid,
frozen/trainable/new parameter counts, FLOPs including shared spatial repetition,
peak VRAM, training steps, train time and candidate rejection reasons.
Never choose architecture on held-out test metrics. Report actual measured
outcomes separately from pipeline conformance. 

## Machine-readable parameter accounting (2026-11-02 integration)

New coordinated visual worker results include `parameter_accounting`, constructed
by `src/simple_adp/visual_accounting.py::visual_parameter_accounting` from
the **selected model instance**, not guessed from the route name:

- `total_parameters`: unique parameter tensors across the composite model,
  including frozen encoder/backbone, any classifier and eligible MLP sites.
- `frozen_parameters` / `trainable_parameters`: mutually exclusive counts
  by current `requires_grad` state; their sum equals total.
- `eligible_mlp_parameters`: sum over distinct actual `TabularMLP`
  submodules advertised by `growth_targets`. In V08's scheduled multi-site
  model, this includes previously committed currently frozen MLPs.
- `currently_trainable_eligible_mlp_parameters`: eligible-site subset whose
  parameter tensors are **currently** trainable. Other trainable parameters
  such as an allowed fixed-dimension supervised source task head are counted
  separately as `other_trainable_parameters`.
- `eligible_growth_sites[site_id]`: fixed local interfaces, current hidden
  widths, MLP parameter counts, active-trainable flag, and Linear matrix MACs
  per **one actual MLP invocation**.

MAC counts are not full-image FLOPs and do **not** account for repeated
token/spatial application, interpolation, attention, convolution, FFT,
normalization or entire-backbone inference. They must not be advertised as
whole-model compute. Neither these selected-state counts nor real
frozen-source micro-smokes establish fairness across **search training
budgets** or estimate newly trained FLOPs. Future scientific result exports
must include measured per-candidate training steps, exact geometry-specific
forward compute and hardware peak memory in separately identified fields.

For the frozen V01 head, a temporary composite ModuleDict includes both
the original frozen encoder and the selected native ADP head **only for
read-only accounting**, without changing either serialized model.


## Observed image inference MACs and native candidate budgets (2026-11-03)

The new `visual_linear_conv_macs(model, image)` executes the **selected
whole image composite** with a single [1,C,H,W] validation image and
counts calls through registered `nn.Linear` and `nn.Conv2d` modules:

```text
Linear MACs  = numel(actual_linear_output) * input_features
Conv2d MACs  = numel(actual_conv_output) * (input_channels / groups)
               * kernel_height * kernel_width
```

Thus a local channel FFN's weights are counted once in parameter
accounting, but its actual repeated token/spatial invocations are counted
in the inference MAC trace. The report is named
`inference_compute.observed_linear_conv_macs_per_image`, with per-module
counts and calls; `inference_compute.whole_model_flops` is explicitly null.
Functional/fused attention projections, matrix attention products,
complex FFT/IFFT, normalization, pooling, activation, residual arithmetic,
interpolation, memory movement and training/backward are excluded. These
results are **partial operator-instrumented inference MACs**, not total
model FLOPs, and must not be mixed with training search cost.

For V02–V11 and source V03/V04, each completed candidate history entry
now adds `trained_epochs`, `optimizer_steps`,
`candidate_total_parameters`, and
`candidate_eligible_mlp_parameters`. A signed
`training_budget` accumulates optimizer steps, epochs, trained
stages and `depth_warmup` catch-up stages that are not eligible for
architecture selection. `complete=false` means an old checkpoint
was resumed without a history of all prior budget measurements.
These are **observed counts from StageTrainer**, not extrapolated
timings or hypothetical FLOPs.

V01 uses the original tabular head controller, so the new native
composite-stage `training_budget` field is not fabricated for it.
The fixed and growth routes must still be compared using actual
optimizer-step and image-example budgets, not just equal epochs per
stage. No numerical claim of ADP gains follows from this accounting.


## Post-selection calibration diagnostics

The opt-in vision runner reports `top_label_ece_10` and
`multiclass_brier` alongside existing source `ClassificationTask`
accuracy, balanced accuracy, macro-F1 and log-loss. These are computed
**only after** `fit()` has completed; no calibration metric enters
the architecture-validation objective or any controller decision.

For sample (i), predicted distribution (p_{ic}), encoded class
(y_i), maximum confidence (q_i=\max_c p_{ic}), predicted class
(\hat y_i=\arg\max_c p_{ic}), and (N) held-out samples:

```text
ECE(10) = sum_b [ count(bin_b) / N ] *
          abs(mean_correct(bin_b) - mean_confidence(bin_b))

Multiclass Brier = (1/N) * sum_i sum_c
                   (p[i,c] - 1[c==y_i])**2
```

The first ten confidence intervals are half-open except the last
interval, which includes confidence 1.0. Source class labels may be
nonconsecutive; `ClassificationTask.encode_targets` preserves the
original training label map. Probabilities must be finite, lie in
[0,1], and sum to one. For a ten-class classifier the multiclass
Brier score lies in [0,2]; lower is better. Small smoke partitions
yield unstable calibration estimates: do not interpret a low ECE on
20 images as scientific calibration evidence. Future multi-seed
calibration comparisons must provide confidence intervals and matched
training budgets.


## Dense self-attention cost and budget-pair audit (2026-11-03, second pass)

`visual_linear_conv_macs` now returns two **separately labeled** MAC
accounting streams for the selected full visual image model:

- `observed_linear_conv_macs_per_image`: directly counted
  `nn.Linear` and `nn.Conv2d` forward calls, including repeated
  spatial/token use. The PyTorch `nn.MultiheadAttention.out_proj`
  child is explicitly excluded here to avoid double counting.
- `analytical_multihead_attention_macs_per_image`: **shape-based
  dense MAC estimate** for each executed `nn.MultiheadAttention`
  invocation, including Q/K/V/output projections and the QKᵀ
  and attention-value dense products. It uses the actual query,
  key, and value lengths, distinct key/value dimensions, batch
  dimensions and supported batch-first/sequence-first/unbatched
  layouts; repeated invocation is counted repeatedly.
- `partial_dense_attention_inclusive_macs_per_image` is the
  sum of these two streams. It is **not** a whole-model FLOP
  count. Attention softmax, masks and sparse implementation
  savings, normalization, complex FFT, pooling and backward
  computation remain unmeasured. Different sources/kernels
  may fuse operations; analytical MHA is not a profiler of
  actual hardware instructions.

For a batch of one, transformer embedding size D, query length Q,
key/value length K, key dimension Dk and value dimension Dv,
the standard dense attention accounting is:

```text
QKV/out projections: Q*D*D + K*Dk*D + K*Dv*D + Q*D*D
QK^T plus attention-value: 2*Q*K*D
```

The official Fashion-MNIST smoke matrix now also records a conservative
`fixed_vs_growth_budget_audit` with 19 fixed/RR pairs. It rejects
duplicate/missing method identities, separates split/site mismatches,
and shows the **observed optimizer-step ratio** only when both native
budgets are known and complete. The V01 legacy head budget is
`unknown`; its step count is never silently set to zero. Even
matching optimizer step counts do not prove matching full training
FLOPs or scientific validity. Matched compute, full dataset, more
seeds and confidence intervals remain future requirements.


## Additional official-image dataset coverage

The original `scripts/visual_official_fashion_matrix.py` optionally
selects `--dataset cifar10` or `--dataset cifar100`, preserving
the historical Fashion-MNIST default, artifact filename and format.
This uses the **existing** opt-in `VisionDatasetProvider`,
`SuitePlan.build`, fixed/ADP native controller and
`runner.worker.run_spec`; it is not a parallel CIFAR training path.

The script requires the class-balanced train, architecture-validation
and official-test subset limits each contain at least 10 items for
Fashion-MNIST/CIFAR-10 or 100 items for CIFAR-100.
All splits retain the source image and source-index hashes, and the
official test remains isolated throughout architecture selection.
Summary files have dataset-specific names and include the
fixed-versus-growth optimizer-step audit.

Completion of 38 tiny jobs for a dataset is strictly an integration
smoke, **not** a matched-budget, multi-seed full CIFAR benchmark.


## Logical model memory and opt-in candidate admission (2026-11-03)

The CPU/CUDA-independent `visual_model_tensor_payload(model)`
reports *logical*, dtype-aware parameter and registered-buffer payload
bytes. It partitions frozen parameter, trainable parameter and buffer
bytes, with per-device and per-dtype totals. Complex64 GFNet filter
weights are **8 bytes per element**, not an assumed float32 4.
Distinct tensor views with shared underlying storage may be
overcounted; repeated references to the **same Parameter object**
are counted once. The returned
`peak_training_memory_bytes` and
`unique_allocator_storage_bytes` are explicitly null.

`VisualResidualSearch(max_model_tensor_payload_bytes=N)` optionally
refuses an initial model whose parameter+buffer payload exceeds N and
refuses over-budget growth candidates *after building the proposed
model but before StageTrainer allocates optimizer state or begins
training*. Rejections are signed, include proposed architecture and
site, and do not consume a successfully admitted proposal slot.
The limit is a **pre-training payload gate**, not a guarantee against
OOM: it does not bound temporary cloning at proposal construction,
input batches, activations, gradients, optimizer moments, checkpoint
copies, attention workspaces or the CUDA caching allocator.

Every native stage, including progressive D1 width-1 catch-up,
is recorded in `trained_model_memory_trace` with its logical
payload and optimizer steps. Only selection-eligible stages appear
in the ordinary architecture-selection history. The completeness
flag is false when resuming from an older checkpoint that never
saved prior-stage memory records; no nonexistent historical values
are fabricated. Results expose both the selected-model
`model_memory_accounting` and the signed native history.

The coordinated suite exposes
`RunnerSettings(visual_model_tensor_payload_cap_bytes=N)`.
It is omitted from the default plan signature and worker specs
when None, preserving pre-existing uncapped plan identities.
With a positive integer, it becomes a signed control-plane
setting and is forwarded to native V02–V11 and source-faithful
V03/V04 search. Changing it invalidates a checkpoint signature.
V01 uses a different legacy tabular-head estimator and **rejects
the optional cap** instead of falsely claiming per-candidate memory
admission. The user must disable the cap or select native routes
for capped experiments.

The verified official-image Fashion/CIFAR10/CIFAR100 matrices
are unbounded-by-byte-cap smoke tests; their new memory reporting
must not be described as actual GPU memory measurements. Physical
CUDA OOM testing and full activation/optimizer accounting remain
outstanding.


## Exact local-MLP growth payload preflight (2026-11-03)

Before building a new native visual FFN candidate, the optional memory
admission policy now applies a **zero-candidate-allocation forecast**
to standard `TabularMLP` parameters and registered BatchNorm buffers.
For widths (w_1,\ldots,w_L), input dimension (d), output dimension
(c), floating dtype itemsize (b) and optional BatchNorm, the
modeled logical payload is:

```text
Linear parameter elements =
    (d+1)*w1 + sum_(i=2..L) (w[i-1]+1)*w[i]
    + (w[L]+1)*c

Per BatchNorm1d hidden layer =
    4*w[i]*b + 8 bytes (two affine vectors, two running-stat vectors,
                         one int64 num_batches_tracked)
```

The preflight first verifies that this source model's **existing**
payload matches the analytic formula; if its structure/dtype is
different or cannot be proven, it safely leaves the existing
postconstruction check in place and marks preallocation prediction
unavailable. For supported structures, it replaces only the eligible
MLP's predicted bytes in the whole-model logical payload. Synchronous
V07 token/channel growth includes the second MLP delta; V08 scheduled
growth uses its active site. Fixed grid, width-RR, all-layer width,
source DEN RR, D1/Dcur depth and intermediate D1 warmup are covered.

Over-cap candidates are refused *before allocating new growth
weights or deep-copying the model*, with
`memory_admission_rejections[].before_candidate_construction=true`
and a signed `prediction_kind`. The local model still must first be
restored from its snapshot, and the initial model/source checkpoint
must be loaded before memory accounting; this is not a global
no-allocation or OOM guarantee. The existing **postconstruction**
cap remains the final defense for unsupported modules or mismatches,
and the existing accepted-candidate source optimizer/training logic
is unchanged.

No forecast is assumed to measure optimizer moments, activations,
CUDA graph/kernel workspaces or allocator fragmentation. Neither
this cap nor the whole-model inference MAC sum is equivalent to
matching the full training-memory or training-compute budget.


## Multiple signed runner seeds and independent pair provenance

The official image matrix can now execute more than one *actual*
`SuitePlan` model seed. The selected job key is
(`model_adapter`, `model_id`, `RunIdentity.seed`),
rather than conflating separate seeds in one fixed/RR pair.
For N seeds and all 19 controlled architecture/site adapters,
the planned bounded worker matrix contains `38*N` jobs.

The multi-seed summary reports every derived runner seed
and a grouped `fixed_vs_growth_budget_audit`. For each
seed, the existing auditor requires identical dataset ID,
split checksum, seed and architecture/site identity before
examining optimizer-step counts. A missing, duplicate or
cross-seed worker produces an explicit error/unknown comparison,
not a new benchmark datapoint. Per-seed budget comparisons
are aggregated as **counts of known, unknown or unequal work**,
not as pooled model accuracy or statistically significant gains.

By default N=1, preserving the historical one-seed output schema
and file path. The two-seed first-run workflow is
`.github/workflows/visual-official-multiseed.yml`.
It uses tiny official Fashion images and 1 epoch per stage:
success proves execution and identity isolation only, not
training-compute parity or scientific superiority.


## Separate fixed and RR epochs with observed step parity

The opt-in official-image driver `--step-balanced` bypasses
the standard `smoke_test=True` full-training-batch,
one-epoch override and uses the existing worker
`max_epochs_vanilla=2`, `max_epochs_adp=1`,
`max_epochs=None`, a bounded single ADP structural
proposal and explicit `batch_size`. This typically
offers fixed 2 epochs versus 2 RR stages x 1 epoch.
No new training loop or ADP growth semantics are introduced.

The signed output records
`requested_step_balance.fixed_epoch_cap`,
`requested_step_balance.adp_per_candidate_epoch_cap`,
`requested_step_balance.batch_size` and the
worker-reported **actual** steps. A fixed-vs-growth pair
is not considered equal-step unless both complete
native stage budgets agree, belong to the same
dataset/split/seed/site and have positive counts.
Even then `full_training_flops_matched=null`
and `scientific_performance_conclusion=null`.
Optional N independent seeds remain independently audited.

The bounded official Fashion-MNIST workflow
`.github/workflows/visual-official-stepbalanced.yml`
exercises 38 native worker jobs and saves the
resulting conditional comparison artifact. This is
not a scientific benchmark until full data and
actual matched compute/uncertainty reporting exist.


## Verified step-count balance without equal FLOPs

The first explicit native optimizer-step balancing experiment
(official Fashion-MNIST 40 train/20 validation/20 test,
one derived seed, 19 controlled architecture/sites × two
controllers) completed 38/38 workers:
https://github.com/Anurag9000/simple-adp/actions/runs/41593647892.
The signed audit recorded **18 native pairs with exactly
six optimizer steps for each fixed/growth job**, zero
native step mismatches, and one V01 pair with unknown
legacy optimizer count. It explicitly preserved
`full_training_flops_matched=null`.

For example, one native V02 selected model's observed
partial inference arithmetic was 178,792 MACs per image
on the fixed side versus 185,064 on the growth side,
despite equal six-step training counts. These are
partial Linear/Conv2d + analytical declared
MultiheadAttention MACs, not all-model FLOPs.
The **training** operation counts remain unknown.
Validation-chosen architectures, source class identities,
budget completeness, and the fixed seed all remain part
of the scientific provenance. A one-seed micro-smoke
cannot justify claiming a general ADP improvement.


## Observed process resources around native visual fit (2026-11-03)

The signed visual adapter/worker result now includes
`fit_process_resources`, independently of the existing
`model_memory_accounting`. Three different kinds of memory
information must **never** be conflated:

1. **Logical model tensor payload** — exact dtype-aware sum of
   registered model parameter and buffer *elements*; no activations,
   optimizer allocations or independent backing-storage proof.
2. **Nominal dense Adam/AdamW payload** — hypothetical model +
   dense-gradient + two same-dtype moment tensors for every
   trainable parameter. It is an analytical calculation, not a
   peak training memory measurement.
3. **Observed process-resource snapshots** — Linux
   `/proc/self/status` `VmRSS` and `VmHWM` before/after
   `fit()`; on actual CUDA jobs, PyTorch's *current-device*
   `memory_allocated`, `memory_reserved`, and unreset
   *process-lifetime* allocation/reservation maxima.

The fit window covers `VisualADPEstimator.fit()` (V01)
or `VisualResidualSearch.fit()` (V02–V11/source-V03/V04),
including training, native growth, architectural validation,
checkpoint construction and recovery. Test-set inference
is conducted only after the end snapshot. The resource
probes do not inspect held-out data, move models, alter RNG,
reset CUDA peak counters or change selected architectures.

The `rss_current_delta_bytes` is the difference of two
*point measurements*, not the memory allocated by the
algorithm. The `rss_window_peak_lower_bound_bytes` is a
lower bound based on those endpoint samples. When the
**process-lifetime** VmHWM increased, the new high-water
mark was achieved during the fit interval; if VmHWM did
not increase, this method cannot recover a distinct fit
peak. Both `exact_fit_window_peak_rss_bytes` and
`cuda_pytorch_allocator.exact_fit_window_peak_cuda_bytes`
are **always null**. The PyTorch CUDA allocator counters
do not cover other processes, driver allocations or all
operator workspaces. In long-lived official matrix
processes, peak counters include **previous workers**,
so they must not be compared as per-job peaks.

Unavailable Linux /proc or physical CUDA metrics remain
null, never zero. The result's presence is a truthful
observation, **not physical CUDA benchmark evidence**:
CUDA hardware remains necessary to establish real
CUDA allocator values. Full per-step GPU profiler traces,
device-wide VRAM and actual training FLOP counts remain
unimplemented.


## Independent job process scope for memory comparison

The official smoke driver can add
`--isolated-workers` to run each **original** signed
worker specification in a fresh subprocess, verifying
the saved `run_signature` on return. The independent
worker's `VmHWM` is process-local and not contaminated
by previous matrix jobs. Child logs and signed results
are retained as evidence. Normal in-process execution
remains the default for compatibility.

The combined opt-in
`--step-balanced --isolated-workers` uses the same
fixed two-epoch and ADP one-epoch-per-stage contracts
and the same 19 source-site fixed/RR pairs; it adds
actual current RSS and lifetime process highwater to
the comparison. No per-stage exact peak is reported,
and CPU RSS does not measure VRAM or full training FLOPs.
A job may still allocate memory before the first fit
snapshot. This setup is a *better controlled measured
memory baseline*, not a scientific comparison by
itself.


## Verified RSS schema in official real-image worker results

The official 38-worker Fashion step-balanced run
https://github.com/Anurag9000/simple-adp/actions/runs/41605309264
passed with measured `fit_process_resources` present
in every result, in addition to model tensor estimates.
The run's 38 workers were called in one **shared**
Python process; their process-lifetime VmHWM values
are therefore *not* independent per-worker peaks.
The subsequent isolated-process workflow addresses
that comparability gap, but its results need independent
verification. No observed CPU RSS measure should be
presented as a physical GPU/VRAM memory benchmark.


## Verified independent per-job CPU RSS observations

Official source evidence:
https://github.com/Anurag9000/simple-adp/actions/runs/41605486116.
All 38 fixed/growth controlled jobs completed through
`runner.worker.main` in their **own** Python subprocess.
All 38 reported distinct process IDs and valid CPU RSS
snapshots, demonstrating that their **process-lifetime**
high-water counters are not inherited from prior
matrix jobs. The audit also retained 18 native pairs
with matching actual optimizer steps and one honest
unknown V01 pair. The results do **not** demonstrate
matched optimizer memory, CUDA VRAM, backward FLOPs,
statistically reliable performance gains or exact
fit-stage CPU peaks.


## Descriptive independent-process memory pair comparison

For completed `--isolated-workers` matrices only,
`paired_isolated_process_rss_audit` records the
fixed and growth *child process* RSS before/after
fit, their post-fit process-lifetime VmHWM, and
growth-minus-fixed descriptive differences by site
and source runner seed. It independently rejects
mismatched dataset/split/seed/site identities before
forming any pair. It **never** asserts equal backward
FLOPs, exact fit peak, model-only RSS, GPU VRAM,
or statistically significant superiority. In-process
multiworker matrices do not receive this audit.

Raw byte differences reflect differences in whole
child processes and possible OS/shared-library
effects, not an isolated causal ADP architecture
memory effect. A multi-seed, full-source
research comparison will require suitable hardware,
data scale, and uncertainty characterization.
