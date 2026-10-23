# OPF Suite Audit

## Scope

The current Case 14 manifest contains `179` jobs for one repeat with vanilla
enabled:

- width-only: `45` jobs (`3` families x `5` seed depths x `3` heads)
- width-to-depth: `12` jobs (`4` families x `3` heads)
- depth-to-width: `12` jobs (`4` families x `3` heads)
- depth-only: `60` jobs (`2` families x `10` seed widths x `3` heads)
- vanilla: `50` fixed-architecture jobs (`5` depths x `10` widths)

The adaptive head variants are `single_head`, `two_heads`, and `four_heads`.
The architecture limits are width `512`, depth `5`, and total neurons
`1_000_000_000` for neuron safety; epoch limits are separate and unlimited by
default (`0` internal sentinel, with `inf`/`unlimited` accepted at the CLI and
environment boundary).

The active `DNN/VANILLA/main.py` registry contains `41` importable entry
points: the `39` adaptive family/head aliases plus the two vanilla aliases.
Unavailable historical or optional module references are pruned from the active
registry rather than exposed as broken choices. A subprocess import audit found
`0` failures across all `41` advertised entries.

## Model-Path Verification

The existing 2026-08-29 smoke matrix launched every manifest job through the
exact child command stored by the central suite runner. It used the real
prepared Case 14 cache, one training epoch, and small train/validation/test
windows so the test remained finite while still exercising the complete child
path. That matrix was an executable child-path test, not a production
convergence run and not a pressure-termination/relaunch test.

All `179/179` jobs passed these checks:

- zero nonzero child exit codes
- finite metric values
- at least one optimizer step after backpropagation
- nonempty checkpoint output
- CSV telemetry or vanilla metric-log output
- progress-state output for adaptive jobs; vanilla jobs use their native
  metric CSV and checkpoint contract instead

One representative job was also run with normal diagnostics enabled. The
remaining exhaustive pass used `OPF_ADP_SKIP_DIAGNOSTICS=1` to avoid spending
the audit time on plots; this flag skips only plotting, not training,
validation, telemetry, checkpointing, or progress-state writes.

The smoke matrix verifies each requested architecture's forward/backward path
and basic checkpoint output. It does not claim production-scale convergence, a
complete expansion trajectory for every adaptive job, or correct recovery from
a killed child; those require separate tests.

After the checkpoint and registry corrections, the complete current-code Case
14 matrix was rerun on 2026-08-29 using four disjoint verifier ranges. The
results were `45/45`, `45/45`, `45/45`, and `44/44`: aggregate `179/179`, zero
nonzero child exits, zero training-verification failures, and zero timeouts.
The 129 adaptive jobs all produced both a stage-local and telemetry
`checkpoint_last.pt`; the 50 vanilla jobs produced their native checkpoint and
metric artifacts. Reports are preserved under:

- `/tmp/opf_adp_current_audit_part0.5XNFcl/verification_report.json`
- `/tmp/opf_adp_current_audit_part1.Zsu8dp/verification_report.json`
- `/tmp/opf_adp_current_audit_part2.sOyL2y/verification_report.json`
- `/tmp/opf_adp_current_audit_part3.UqYgmu/verification_report.json`

This is current-code one-epoch execution evidence for every suite job. It is
not a claim that production early stopping converges identically, and it does
not destructively force a live hardware-pressure event for every job.

A bounded expansion follow-up selected all `40` unique family/head variants
(the `39` adaptive variants plus one vanilla representative), capped growth at
width/depth `2`, and passed `40/40`. During that follow-up, the four-head
width-only implementation was found to write valid telemetry and checkpoints
but omit its stage-state records. `ADP_DEN_4_head_width_only` now persists and
restores initial, expansion, and final stage state; both affected four-head
variants were rerun and passed `2/2` with the stage-state records present.

The expansion audit also verifies two continuation invariants. A stage
checkpoint is accepted only when its recorded hidden-width vector matches the
live architecture; a checkpoint from the preceding growth stage is ignored so
it cannot restore the old architecture or reset the growth loop. A matching
checkpoint still restores model, optimizer, scheduler, patience, epoch, and
batch-cursor state. Round-robin width growth uses the minimum hidden-layer
width as its target and advances exactly one layer per expansion for the
single-, two-, and four-head engines; all-layers growth remains simultaneous.

Completed stage state is a terminal fast path: the saved best checkpoint is
restored and no stage is retrained. Every width-growth controller writes a
durable `stage_state.json` beside its telemetry before entering a stage. It
records the stage name/index, exact hidden-width vector, round-robin cursor,
global-best checkpoint, and that stage's `checkpoint_last.pt`. On relaunch, an
active stage restores that architecture and continues the same
optimizer/model/epoch/batch checkpoint; it does not fall back to `initial` or
replay an already completed growth transition. A completed non-final stage
restores its latest stage weights as the source for the next expansion while
retaining the independent global-best snapshot for final output.

At normal termination every width-only engine restores the global-best
architecture and weights before diagnostics and writes that architecture in
the final stage state. This applies to single-, two-, and four-head
patience-based growth and all three LG-ND head variants. Growth patience
remains separate from epoch early-stopping patience: patience-based families
count failed expansions, while LG-ND still stops after its first
non-improving expansion.

The stage checkpoint's architecture metadata must match the live hidden-width
vector. A stale checkpoint is skipped rather than partially applied. The
four-head depth-only engine also persists its round-robin cursor, so its
per-layer growth order survives checkpoint restoration.

## Progress Accounting

### Corrective Finding: Pressure-Recovery Checkpoint Mismatch

The previous pressure-aware run is invalid for model or algorithm comparison.
The massive runner's pressure handler terminates the child process tree and
later relaunches it; it does not preserve the Python process in memory. The
training code loaded `stages/<stage>/checkpoint_last.pt`, while telemetry had
been saving the checkpoint only under its task telemetry directory. After a
pressure relaunch, adaptive jobs therefore started from fresh initialization
even though telemetry counters and logs accumulated across attempts. Repeated
early loss sequences and inflated epoch-record totals were the direct symptoms.

The corrected contract is implemented in the native adaptive engines. Every
last, best, and best-batch checkpoint write atomically mirrors the payload to
both telemetry and stage-local paths. The stage-local training load path
restores the latest model, optimizer, scheduler, epoch, batch cursor, and
patience state; it does not replace the latest model with the older `best_snap`
before continuation. Outer stage-state recovery restores the global-best
architecture only for terminal output, while an active width-growth stage
restores its recorded architecture and latest stage checkpoint exactly.

The old result tree must not be resumed or used as scientific evidence. The
checkpoint fix is forward-looking; it cannot reconstruct optimizer history that
was lost during the earlier fresh relaunches.

The old status line reported `batches_done` and `epochs_done` by summing every
telemetry-bearing job in the run root. It did not restrict those values to the
currently active jobs. Failed/retried jobs and completed jobs with preserved
telemetry therefore contributed to the totals. In addition, an ADP expansion
stage emits its own epoch records, so `epochs_done` is a count of telemetry
records across stages, not the current stage's displayed epoch number.

The status line now reports both scopes:

- `batches_done`, `epochs_done`: all-job totals
- `live_batches_done`, `live_epochs_done`: jobs currently `running`, `paused`,
  or `suspended`
- `telemetry_jobs`: jobs with a progress sidecar, job log, or saved progress
  counters; stale job state alone is not enough

The pasted historical status showed `active=6` but `telemetry_jobs=16`, which
proves that its `epochs_done=75954` was not the sum of six current models. The
old counter included accumulated telemetry from non-live failed/retried jobs
and from earlier expansion-stage records. The exact Aug 28 runtime tree was
deleted during repository cleanup, so that specific 75,954 value cannot be
decomposed per job now. An older July snapshot remains recoverable from Git
commit `5f803df`, but it is not evidence for the Aug 28 six-job state.

The scheduler audit also exercised both admission lanes. GPU-first mode admits
CPU work when GPU admission is unavailable and host RAM gates permit it. Swap occupancy is recorded for diagnostics but does not gate launches or pauses.
Retry accounting remains separate from pressure pause/resume attempts. Current
policy treats `--max-retries 0` as unbounded retries; positive values impose a
limit. The runner's default remains one retry; the current ADP campaign passes
zero explicitly for unbounded retries.

The live launch loop calls the same scheduler-selection function covered by
the unit tests. `--scheduler`, `--cpu-only`, active-job caps, and gate state
affect actual admission rather than being metadata-only controls. CUDA-OOM
recovery retries on the job's original device after pressure recovery; CPU
admission remains governed by ordinary GPU and host pressure gates.
Simultaneous pressure is recorded as a combined reason, and a suspended GPU
job resumes only after every pressure domain that paused it has reopened.

Launch-time `Popen` failures are persisted and retried while retries remain
enabled; positive retry limits can still finalize a terminal failure. A failed
process creation cannot silently drop one job or abort the rest of the suite.
All adaptive growth APIs also enforce both `max_width` and the aggregate
`max_neurons` limit before mutation, including multi-layer and round-robin
increments.

If the orchestrator itself is restarted while a child remains alive, the child
is represented by a recovered PID handle because its original exit status is
not observable. When that child later disappears, the runner requeues it for
checkpointed continuation instead of inventing a failure or consuming retry
budget. Terminal job states are not resurrected merely because a matching
command or PID is found.

## Dataset Concurrency And Diagnostics

Case cache preparation is protected by a per-case inter-process file lock.
Download, extraction, generated processed-cache construction, and forced
reloads are serialized for a shared `OPF_ADP_DATA_ROOT`; concurrent children
reuse the completed manifest instead of deleting or rebuilding each other's
groups. The lock regression test is in
`tests/test_opf_dataset_cache_lock.py`.

The cache builder normalizes sample filenames before applying manifest split
cursors. Current GridOpt archives use global names (`group_1/example_15000.json`),
while local fixtures may restart at `example_0.json` in each extracted 15,000
sample group. The normalizer supports both layouts and never offsets an
already-global filename twice. Cache format versioning forces older manifests
to rebuild, and the global-index regression is covered by the cache tests.

Stage diagnostics first prefer an existing `sample_case<N>.json`. On a clean
checkout, when that legacy file is absent, the resolver uses the canonical
`example_0.json` from the downloaded raw Case cache, including a configured
`OPF_ADP_DATA_ROOT`.

## Launcher Verification

A central-launcher smoke run admitted the first manifest job on GPU, completed
it with no retry, and wrote the expected run-root state. With tiny controlled
windows its final summary reported three batch records and three epoch records:
one record for each executed ADP stage. After completion, the live counters
were zero while the all-job counters remained three, confirming the separation
between the two scopes.

The progress-accounting contract and corrected checkpoint mirror contract are
covered by
`tests/test_suite_progress_accounting.py`, which verifies all-job versus live
counts, failed telemetry inclusion, legacy log fallback, ignored stale state,
byte-identical telemetry/stage checkpoint mirrors, all six native adaptive
engines using the mirrors, the latest-model load-path rule, and the massive
runner's real terminate/persist/relaunch state transition. The test passes with
Python `unittest`.

The final current-code Case 14 child audit produced `179/179` passes across
`/tmp/opf_adp_current_audit/verify0` through `verify3`: zero nonzero exits,
zero timeouts, zero finite-metric failures, minimum optimizer step `1`, and
minimum checkpoint count `1`. A diagnostics-enabled vanilla child also passed
against the clean raw-cache layout.

The complete diagnostics-enabled unique-variant pass covered `40/40` jobs and
produced finite metric summaries, per-sample metric files, checkpoints, and
diagnostic outputs for every selected variant. Nine compatibility plots were
generated by the newer width-growth implementations. The stagewise depth
implementations write their native metric CSV schema; the legacy plotter skips
only the incompatible train/validation chart when its `Train Loss` and
`Val Loss` columns are absent. Stage metrics, telemetry, checkpoints, and
diagnostic summaries remain present.

## Shutdown Safety

The runner and shell watchdog now terminate only launcher-owned child process
trees on `Ctrl+C`/termination signals, rewrite interrupted jobs as resumable,
and do not issue broad `pkill` patterns that could terminate unrelated Python
workloads. The emergency cleanup scripts use the same repository-scoped
identity check. PowerShell massive launches explicitly pass vanilla inclusion
and track the Python PID before starting the watchdog. Compatibility plots are
written below the job's `OPF_ADP_RESULTS_DIR` rather than the process working
directory.

## Automated Regression Matrix

Run the repository tests with Python's standard library test runner:

```bash
.venv_opf_adp/bin/python -m unittest discover -s tests -v
```

On 2026-08-30, all `42` tests in the full test suite passed. Coverage includes case-path resolution,
results-directory plot scoping, concurrent cache locking, manifest counts and
head/task filters, deterministic repeat seeds, scheduler fallback and caps,
argument validation, all/live progress accounting, retry budgets, cleanup
ownership, POSIX pressure pause/resume, tar traversal rejection, regular and
chunked resume bundles, sharded telemetry rehydration/checksums, the four-head
stage-state regression, strict checkpoint metadata validation, combined
pressure reopening, concurrency-file validation, the pressure-recovery
checkpoint mirror, and the massive-runner pressure relaunch path.

Additional executable checks completed in the same audit:

- all Bash scripts passed `bash -n`;
- all repository Python packages passed `compileall`;
- the central Python launcher and shell wrapper each produced an `895`-job
  five-case dry-run manifest;
- a real Case 14 wrapper launch completed one GPU child with three batch/epoch
  records, checkpoint, telemetry, diagnostics, and exit code `0`;
- the normal pressure-aware wrapper completed a child successfully;
- GPU-visible and GPU-hidden backend probes selected the expected accelerator
  state.

The child matrix uses one epoch and tiny data windows by design. It verifies
executable training, backpropagation, optimizer updates, checkpointing,
telemetry, and diagnostics, but does not claim that all `895` production jobs
have converged or that every adaptive job reaches its maximum width/depth under
production early-stopping settings. Before treating a production run as valid,
run a controlled interruption test that confirms a second launch reports a
nonzero saved epoch/batch cursor and does not repeat the initial loss sequence.
A live hardware OOM was not forced destructively; the process-group pressure
path and checkpoint contract are covered by unit/regression tests.

## Clean Case 14 Start

The prior local `Results/` tree and any local `resume_snapshots/` bundle were
deleted on 2026-08-29 after the audit. The repository was checked for tracked
result artifacts and none exist. The ignored `data/` cache and `.venv_opf_adp/`
environment were preserved because they are inputs/runtime dependencies, not
results. No suite runner process remains active.

After removing any prior local outputs, start a new Case 14 suite without
restoring an old result bundle:

```bash
cd /home/anurag/Projects/OPF_ADP
INCLUDE_VANILLA=1 \
REPEAT_COUNT=0 \
OPF_ADP_GLOBAL_SEED=42 \
OPF_ADP_DATA_FRESH=1 \
OPF_ADP_AUTO_RESUME_BUNDLE=0 \
./run_case_14.sh
```

`REPEAT_COUNT=0` is the one-pass override; the repository default remains five
repeats. `INCLUDE_VANILLA=1` includes the 50 fixed-architecture jobs. The
wrapper's zero concurrency and zero max-active values mean unbounded admission
subject to the pressure-aware RAM/VRAM gates.

## Terminal Truth Contract

The runner does not invent progress counters. Each status line is assembled
from the scheduler's in-memory job list plus the persisted `job_state.json`,
adaptive `progress_state.json`/telemetry, and legacy job logs. `queued`,
`active`, `suspended`, `completed`, retry, and failure fields describe the
actual scheduler states. `batches_done` and `epochs_done` are cumulative
records found across all jobs; `live_batches_done` and `live_epochs_done` are
the same calculation restricted to jobs currently running, paused, or
suspended. Consequently, all-job totals can exceed the sum of current models,
especially after completed or retried jobs and multiple adaptive stages.

RAM and VRAM fields are sampled resource observations at the timestamp of the
line, not predictions. The live process counts and gate fields are updated from
the same active/suspended job state used for admission. A status line is
therefore internally consistent with the persisted run state at its sampling
instant, subject to normal operating-system and `nvidia-smi` sampling latency.

## 2026-10-04 exact-resume state audit

New training checkpoints use schema 2. Adaptive and native trainers record model/optimizer/scheduler state, early-stop state, epoch and batch position, best-selection metadata, Python/NumPy/Torch CPU/CUDA RNG, active shuffled-sample permutation, backend determinism flags, runtime versions, and a strict loader contract. Fresh OPF shuffles match PyTorch's default DataLoader/RandomSampler RNG consumption, and mid-epoch restore avoids consuming the DataLoader base seed twice. Writes are atomic. Native training additionally saves partial-epoch metric accumulators and writes a boundary checkpoint after validation and early-stop updates. A saved loader contract mismatch now stops restore rather than silently applying an incompatible cursor.

Historical checkpoints without RNG or sampler state are rejected when selected. The runner marks the job failed without retrying it, because no implementation can reconstruct the missing historical stochastic state. Exact floating-point replay still requires matching input data, compatible software/hardware, and deterministic kernels. Focused interruption equivalence tests cover shuffled order, dropout RNG, optimizer, scheduler, and resumed model weights.
The external loss-guided ADP adapter has no scheduler by design; its exact checkpoint records AdamW, early-stop state, partial-epoch metrics, RNG and loader order. The method controller restores the completed candidate random stream before applying its next architecture-growth transition.

Stage-growth sidecars also reference an atomic companion RNG checkpoint. Loading the stage sidecar restores the random stream before the controller makes its next growth transition, covering restart windows between stages.
