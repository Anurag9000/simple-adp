# OPF_ADP Orchestration and Model Mechanisms

This document is the technical appendix for the current OPF_ADP suite. It describes the shared launcher, the pressure-aware admission controller, the resumability layout, the runtime tuning layer, and the width/depth ADP model families that are wired into the case runners.

Scope:
- Linux shell case runners and the shared Python launcher contract
- Massive suite orchestration and case-specific execution
- Dataset preparation and resumable cache building
- Process watchdog and runtime tuning
- GPU/host admission and gate locking
- Width-only and depth-only model growth policies

## 1. Control Plane Overview

The active stack is:

1. `run_case_*.sh`, `run_massive_opf_suite.sh`, `run_search_suite.sh`, and `run_adp_w2d_suite_parallel.sh`
2. `utils/opf_suite_launcher.sh`
3. `utils/opf_suite_launcher.py`
4. `utils/opf_massive_suite_runner.py`
5. `DNN/VANILLA/Dyn_DNN4OPF/models/*.py`

The case runner shells are thin wrappers. They only select a case, source the shared defaults, and forward to the centralized launcher.

The launcher is responsible for:
- setting `RESULTS_DIR`, `RUN_ROOT`, and `SOURCE_RUN_ROOT`
- prefetching the case dataset in case mode
- wiring the shared runtime defaults into the massive runner
- exporting the launcher-level knobs as environment variables
- emitting the launch manifest

The massive suite runner is responsible for:
- building the job graph
- persisting run metadata and per-job state
- launching child training jobs
- pausing, resuming, and retrying jobs
- enforcing launch admission thresholds
- enforcing the child-footprint gate lock logic
- writing progress, summary, and replay artifacts

The model code is responsible for:
- training
- early stopping
- checkpointing
- width/depth expansion
- restoring the best snapshot once a search phase is done

## 2. Shared Runtime Defaults

The central defaults are defined in:

- `utils/opf_shared_defaults.py`
- `DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py`
- `utils/opf_run_defaults.sh`
- `utils/opf_run_defaults.ps1`

Current shared values:

- cases: `pglib_opf_case14_ieee`, `pglib_opf_case30_ieee`, `pglib_opf_case57_ieee`, `pglib_opf_case118_ieee`, `pglib_opf_case500_goc`
- dataset groups: `20`
- samples per group: `15000`
- split ratios: `90% train`, `5% val`, `5% test`
- dataset chunk size: `512`
- batch size: `65536`
- diagnostic sample cap: `256`
- training patience: `10`
- width expansion patience: `10`
- depth expansion patience: `10`
- width expansion step: `1`
- max width: `512`
- max depth: `5`
- max neurons: `1_000_000_000`
- max epochs: unlimited by default (`0` internal sentinel; CLI/environment also accept `inf` or `unlimited`)
- finite limits remain available only when explicitly supplied for a test or controlled experiment
- width learning rate: `5e-4`
- depth/common learning rate: `1e-3`
- post-launch sleep before the next fresh resource poll: `30 seconds`
- pressure poll interval in the main orchestrator loop: `0.5 seconds`
- runtime stop/resume thresholds:
  - host RAM: `1024 / 1280 MiB`
  - GPU VRAM: `100 / 500 MiB`
- launch reserves:
  - host: `0 MiB`
  - GPU: `0 MiB`
- child gate-close buffers:
  - GPU child buffer: `100 MiB`
  - host child buffer: `500 MiB`

## 3. Runtime Tuning Layer

The runtime tuning code lives in `utils/runtime_tuning.py` and is applied by the shared launcher bootstrap.

### 3.0 Optional GPU Python Accelerators

When a CUDA-capable GPU is present, the interpreter bootstrap also tries to enable the optional GPU Python accelerators in `sitecustomize.py`:

- `cudf.pandas` for pandas-compatible code paths
- `cuml.accel` for scikit-learn-compatible estimators and metrics
- `cupy-cuda12x` for CuPy-backed array helpers where the code explicitly opts in

If those wheels are missing, the repo keeps using the CPU-backed libraries without aborting startup.

This is intentionally limited to safe accelerator hooks. It does not rewrite every `numpy` or `scipy` call site to CuPy, because many utilities rely on CPU-native behavior or plotting/I/O libraries that do not have strict drop-in GPU equivalents.
The shared shell launcher exports the repository root into `PYTHONPATH` so the bootstrap is active for all case and suite runner entrypoints.
The shared seed helper also mirrors `np.random.seed(...)` into CuPy when available, which covers the repo's many existing `set_seed` helpers without a file-by-file rewrite.
The current repository scan is stricter than the old helper-routing model: there are now zero direct `numpy`, `pandas`, or `sklearn` imports anywhere outside the central backend shim, and zero remaining call sites reported by `utils/verify_backend_coverage.py`.
Backend policy is centralized in `utils/ml_backends.py`, which owns optional accelerator activation, backend module selection, and the local accuracy helper used by the graph runners.

### 3.1 CPU thread and affinity controls

The bootstrap derives a CPU budget from the visible cores and the active concurrency hint, then exports:

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `GOTO_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `TORCH_NUM_THREADS`
- `TORCH_INTEROP_THREADS=1`
- `OMP_DYNAMIC=FALSE`
- `MKL_DYNAMIC=FALSE`
- `OMP_WAIT_POLICY=ACTIVE`
- `OMP_PROC_BIND=spread`
- `OMP_PLACES=cores`
- `KMP_AFFINITY=granularity=fine,scatter`
- `TABULAR_CHILD_SHARED_CPU=1`
- `TABULAR_CPU_WORKERS=0`
- `TABULAR_CPU_THREADS=<derived budget>`
- `TABULAR_CPU_CORES=<detected cores>`
- `TABULAR_CPU_JOB_CONCURRENCY=1` for shared CPU child launches

The shell bootstrap also tries:
- `renice -20`
- `ionice -c2 -n0`
- `chrt -b`

On Linux, the bootstrap attempts `mlockall(3)` to discourage swapping.

If `systemd-run` is available, the launcher can re-exec under a user scope with `MemorySwapMax=0`, `TasksMax=infinity`, high CPU/I/O weights, and the current allowed-CPU set. The ADP campaign launch wrapper skips this optional scope so host swap remains available to the pressure-aware runner.

### 3.2 Backend Audit

`utils/verify_backend_coverage.py` is the repo audit tool used to prove the import surface is clean.

Current audited state:

- direct imports: `0`
- `np` call sites: `0`
- `pd` call sites: `0`
- `sklearn` call sites: `0`

### 3.3 Windows-specific runtime knobs

The Python helper also supports:
- `TABULAR_WINDOWS_PRIORITY_CLASS`
- `TABULAR_WINDOWS_MEMORY_PRIORITY`

These are best-effort and do not change the Linux launch path.

## 4. Watchdog and Orphan Cleanup

The shell helper `./_orphan_watchdog.sh` wraps the Python orchestrator and installs traps for:

- `INT`
- `TERM`
- `HUP`
- `QUIT`
- `EXIT`

Behavior:
- the main Python orchestrator is launched in the background
- the watchdog monitors the parent shell
- if the parent shell disappears, it sends `TERM` to the python tree, waits a grace period, then sends `KILL`
- on terminal interrupts, it kills the Python orchestrator tree and the watchdog itself
- it also removes the temporary watchdog log on normal exit

This is the repo-level defense against orphaned child processes.

## 5. Dataset Preparation

Dataset preparation is handled by `utils/opf_dataset_cache.py` and triggered through the shared launcher in case mode.

Important behaviors:
- case datasets are prefetched before training begins
- `--fresh` removes the generated processed cache and extracted chunk cache for the case, but keeps the raw downloads
- the cache builder is resumable by chunk
- case `500` is not rebuilt monolithically; it is sliced into chunked cache files

The default case cache size is governed by:
- `OPF_ADP_DATA_CHUNK_SIZE=512`

The case launcher resolves `data/sample_case*.json` for stage diagnostics from the configured local generated dataset cache. If the legacy sample JSON is missing, the resolver uses the canonical `example_0.json` in the downloaded raw case cache. These dataset artifacts are not tracked in Git. Cache download, extraction, processed-cache construction, and forced reloads are serialized by a per-case inter-process lock so concurrently launched children cannot corrupt the shared cache.

## 6. Shared Launcher Contract

`utils/opf_suite_launcher.py` is the shared contract for both case mode and massive mode.

### 6.1 Inputs it accepts

Launcher knobs:
- `--mode case|massive`
- `--case-name`
- `--case-tag`
- `--results-dir`
- `--run-root`
- `--source-run-root`
- `--concurrency`
- `--concurrency-file`
- `--job-start-index`
- `--job-limit`
- `--max-active-jobs`
- `--max-active-gpu-jobs`
- `--launch-host-reserve-mib`
- `--launch-gpu-reserve-mib`
- `--gpu-child-gate-close-buffer-mib`
- `--host-child-gate-close-buffer-mib`
- `--external-memory-drop-reopen-mib`
- `--max-epochs-adp`
- `--gpu-device-index`
- `--cpu-only`

- `--data-root`
- `--chunk-size`
- `--fresh`
- `--surface`
- `--tasks`
- `--param-band`
- `--legacy-architecture-grid`
- `--include-small-stl`
- `--include-vanilla`
- `--width-only-depths`
- `--width-to-depth-widths`
- `--depth-only-widths`
- `--missing-present-task-repeats`
- `--prediction-repeats`

The shared dynamic-model checkpoint restore path uses partial tensor loading so
widened width-only and depth-only stages can resume from earlier checkpoint
files without crashing on size mismatches. During pressure recovery, the
massive runner terminates and relaunches the child rather than suspending it in
place. Adaptive engines therefore write identical checkpoint payloads to both
the telemetry directory and the stage-local
`stages/<stage>/checkpoint_{last,best,best_batch}.pt` paths. The stage-local
`checkpoint_last.pt` is authoritative for model, optimizer, scheduler, epoch,
batch cursor, and patience continuation; `best_snap` is not applied over the
latest model during that load.

Growth-stage safety is explicit: before a stage checkpoint is loaded, its
hidden-width vector must match the live architecture. A stale checkpoint is
skipped. Completed stage state restores the global-best architecture and
returns without retraining. Non-final stage state is replayed through the
numbered stage directories, where already-completed stages return immediately
from their checkpoints and an interrupted stage resumes its own checkpoint.
This prevents a pressure relaunch from advancing the growth cursor past an
interrupted architecture. Round-robin growth persists its per-layer cursor;
all-layers growth increments every current hidden width and both paths enforce
`max_width` and aggregate `max_neurons` before mutation.

### 6.2 What it writes

The launcher creates:
- `suite_launch_manifest.json`
- `RESULTS_DIR`
- `RUN_ROOT`
- `SOURCE_RUN_ROOT`

In case mode it also triggers dataset preparation before the suite starts.

## 7. Massive Suite Orchestrator

`utils/opf_massive_suite_runner.py` is the core scheduler.

### 7.1 Scheduling model

The runner uses:
- a queue of `JobSpec`
- active job tracking
- suspended job tracking
- per-job `job_state.json`
- batch backoff state in `batch_backoff_state.json`
- run metadata and manifest files

The default scheduler mode is `gpu_first`.

The current launch policy is:
- prefer GPU if GPU and host RAM are both admissible
- fall back to CPU if GPU is not admissible but host RAM is
- honor the current gate-lock state
- honor the post-launch hold timer
- honor concurrency limits from the launcher and from any `concurrency_file`

The live launch loop invokes `select_launch_device` with the current gate and
hold admission flags. Thus `pressure_aware`, `gpu_first`, `fixed`,
`--cpu-only`, and active-job caps are execution controls, not merely recorded
metadata. CUDA-OOM recovery resumes a job on its original device after
pressure recovery; normal CPU admissions remain available when GPU admission
is closed. A negative, malformed, or unreadable concurrency file is rejected
rather than silently converted to unbounded concurrency.

### 7.2 Resource sampling

Each control-loop iteration samples:
- host memory via `psutil.virtual_memory()`
- swap via `psutil.swap_memory()`
- GPU memory via `nvidia-smi --query-gpu=memory.total,memory.used`
- active job RSS using the live process tree
- per-child GPU VRAM using `nvidia-smi --query-compute-apps=pid,process_name,used_memory`

### 7.3 Gate thresholds

The admission and pause thresholds are:
- host stop at `1024 MiB`
- host resume at `1280 MiB`
- GPU stop at `100 MiB`
- GPU resume at `500 MiB`

Swap use is sampled for diagnostics only. It does not close admission gates,
pause jobs, or block resume; host RAM and child-footprint pressure remain the
host-side controls.

### 7.4 Child-footprint gate lock logic

The runner now persists and reuses:
- `gpu_gate_closed`
- `host_gate_closed`
- `gpu_gate_lock_job_id`
- `host_gate_lock_job_id`
- `last_child_job_id`
- `last_child_host_rss_mib`
- `last_child_gpu_vram_mib`
- `last_child_host_remaining_mib`
- `last_child_gpu_remaining_mib`

Gate-lock algorithm:

1. Identify the most recent live active child.
2. Sample its process-tree RSS and, if GPU-backed, its GPU VRAM footprint.
3. Compute remaining host RAM and remaining GPU VRAM.
4. If the machine is already under real GPU pressure and `last_child_gpu_vram_mib + 100 > remaining_gpu_vram`, close the GPU admission gate for that child.
5. If the machine is already under host RAM pressure and `last_child_host_rss_mib + 500 > remaining_host_ram`, close host admissions for that child. Swap occupancy is observational only.
6. Keep the gate locked until a genuine process completion event or a qualifying non-Python process exit releases it.

Release rule:
- GPU completion reopens both the host and GPU admission gates
- host/CPU completion reopens the host admission gate
- a qualifying non-Python RAM/GPU process exit can also reopen the matching gate after the configured `external-memory-drop-reopen-mib` threshold (default `500 MiB`) from the last observed external footprint
- these reopen checks are driven by the observed resource class, not by the original lock-owner job id

If a pause reason contains multiple domains, every named domain must satisfy
its own resume threshold or qualifying external-drop condition. For example,
`gpu_memory_pressure+host_ram_pressure` cannot resume on VRAM recovery while
host RAM remains below its resume threshold.

The standalone `pressure_aware_wrapper.py` applies the same configured drop
threshold, but it tracks one process tree rather than separate scheduler jobs.
It resumes that tree only when both the host (including swap) and GPU release
conditions are satisfied; a RAM drop alone cannot resume through active GPU
pressure, and a GPU drop alone cannot resume through active host pressure.

The intent is to avoid admitting a new child only when the current child and the machine pressure together indicate that the next launch would be unsafe. Healthy machines should continue admitting jobs.

### 7.5 Post-launch hold and polling

After each launch:
- the runner sleeps for `30 seconds`
- the next fresh RAM/VRAM resource poll happens after that hold

The main loop then continues to poll at `0.5 second` intervals.

### 7.6 Backoff and retries

If a child exits nonzero:
- the runner records the exit
- it writes retry state when retry budget remains
- it flips the relevant launch gate off temporarily
- it preserves the run-root state so the next invocation can resume

Retry accounting is separate from pressure pausing. `attempt` increments for
launches and pause/resume cycles; `retry_count` increments only when a failed
attempt is re-queued. The default retry budget is `1`, controlled by
`MAX_RETRIES` or `--max-retries`; `0` enables unbounded retries.

If a child exits with an OOM-class failure:
- `batch_scale` is reduced by `batch_backoff_factor`
- the emergency backoff state is persisted
- ordinary queue pressure does not reduce `batch_scale`

If a pressure recovery relaunch itself fails, the job remains persisted as
`paused` with `reason=resume_launch_error`, and the runner logs the exception.
It is not counted as resumed and is not silently dropped from the scheduler.

If initial process creation (`Popen`) fails, the same retry contract is used.
The job is not lost from the queue: it is persisted as `retrying` and requeued
while retries remain enabled; a positive retry limit can still finalize a
terminal failure.

When the orchestrator is restarted with a still-live child, the child is
tracked by a recovered PID handle. Because that handle cannot know the child's
actual exit code, a later disappearance is requeued for checkpointed
continuation rather than recorded as a synthetic failure. Completed or failed
terminal states are not resurrected solely from a matching live command.

On every new invocation:
- `batch_scale` is reset to `1.0`
- transient backoff flags and gate locks are cleared
- job-level resumability remains intact

In `gpu_first` scheduling, GPU is preferred when both lanes are admissible, but
CPU remains eligible as a fallback when GPU admission is unavailable. This is a
preference, not a CPU starvation condition.

Current batch backoff factor:
- `0.5`

### 7.7 End-to-End Control Flow

The live execution path is:

1. A case or massive shell wrapper sources the shared defaults from `utils/opf_run_defaults.sh` or `utils/opf_run_defaults.ps1`.
2. The wrapper exports the central run defaults, including `REPEAT_COUNT`, `OPF_ADP_GLOBAL_SEED`, `INCLUDE_VANILLA`, the pressure thresholds, and the width/depth caps.
3. `utils/opf_suite_launcher.sh` turns those environment values into a Python launcher invocation, preserving explicit CLI overrides when present.
4. `utils/opf_suite_launcher.py` restores any local resume bundle, rehydrates sharded telemetry, prepares the case dataset cache, and writes the launch manifest before the job graph is built.
5. `utils/opf_massive_suite_runner.py` validates the repeat and architecture caps, expands the per-case job list, and deterministically derives each job seed from `sha256(base_seed|mode|case|repeat_index|phase|family|head_variant|depth|width)`.
6. The runner launches jobs through the pressure-aware scheduler, using the current RAM/VRAM/swap samples to decide whether a new child may start and whether a live child should pause.
7. Each child writes its job log and checkpoint files into the case-scoped `Results/` tree; adaptive children additionally write telemetry shards and progress sidecars, while vanilla children write their native metric CSV.
8. When a child completes, the suite reopens the matching admission gates based on the observed resource class and the live pressure samples.
9. When `repeat_count=0` is requested, the suite still executes a single canonical pass for the case graph and suppresses extra repeat cycles. This is the fresh-run override used for the command in this thread.

## 8. Persistent State Layout

Per run root, the suite writes:
- `manifest.json`
- `job_manifest.json`
- `run_metadata.json`
- `batch_backoff_state.json`
- `run_progress.csv`
- `task_summary.csv`
- `final_report.json`
- `suite_summary.json`
- `suite_summary.md`

On clean exit, the shared launcher can export a chunked portable resume bundle under `resume_snapshots/` for the resumability-critical state and latest checkpoints. That bundle is a local artifact; if you copy it alongside the matching `Results/...` tree to another machine, the next launcher invocation can restore it before the same case or suite starts if the results tree is empty or missing.

Per job, it writes:
- `jobs/<job_id>/job_state.json`
- `jobs/<job_id>/job_metadata.json`
- `jobs/<job_id>/command.json`
- `jobs/<job_id>/launch_context.json`
- `jobs/<job_id>/candidate_summary.json`
- `jobs/<job_id>/job.log`

The live per-batch and per-epoch trace is therefore not hidden in a generic
results file. It is anchored in the job log and the per-task telemetry tree:

- `Results/cases/<case>/run_root/jobs/<family>/<case>/repeat_01/<job_id>/job.log`
- `Results/cases/<case>/.../telemetry/.../task_<id>/telemetry.csv`
- `Results/cases/<case>/.../telemetry/.../task_<id>/telemetry.csv.partNNN`
- `Results/cases/<case>/.../telemetry/.../task_<id>/telemetry.csv.manifest.json`

Per model/task telemetry, it writes:
- `telemetry/<model>/<case>/task_<id>/telemetry.csv`
- `telemetry/<model>/<case>/task_<id>/progress_state.json`
- `telemetry/<model>/<case>/task_<id>/checkpoint_last.pt`
- `telemetry/<model>/<case>/task_<id>/checkpoint_best.pt`
- `telemetry/<model>/<case>/task_<id>/checkpoint_best_batch.pt`

Adaptive jobs also mirror the same payloads under each active stage:

- `stages/<stage>/checkpoint_last.pt`
- `stages/<stage>/checkpoint_best.pt`
- `stages/<stage>/checkpoint_best_batch.pt`

The telemetry copies are useful for reporting and bundle inspection; the
stage-local copies are the paths consumed by the native training engines when a
pressure pause is recovered by termination and relaunch.

If a telemetry CSV exceeds GitHub's per-file limit, the local run tree stores it as exact `telemetry.csv.partNNN` shards with a `.manifest.json` sidecar alongside the original path, and the launcher rehydrates those shards back to the original `telemetry.csv` before resume.

Compatibility views:
- `candidate_state.json`
- `method_state.json`
- `task_state.json`
- `replay_manifest.json`

The compatibility views let the suite reuse older resume/reporting expectations while still using one centralized run root. The live `[STATUS]` line uses the per-model `progress_state.json` sidecars to report all-job cumulative batches and epoch records when available, and falls back to the child `job.log` for older already-running jobs so live progress still shows up before sidecars exist. `telemetry_jobs` counts jobs with a progress sidecar, job log, or saved progress counters rather than trusting stale job state alone. `live_batches_done` and `live_epochs_done` separately report only jobs currently marked `running`, `paused`, or `suspended`; all-job totals include failed/retried and completed jobs with preserved telemetry. Because ADP expansion stages emit separate epoch records, `epochs_done` is not the current stage's epoch number.

## 9. Exact Case Runner Surface

Case scripts:
- `run_case_14.sh`
- `run_case_30.sh`
- `run_case_57.sh`
- `run_case_118.sh`
- `run_case_500.sh`

All case scripts forward into the shared launcher.

Case runner behavior:
- prefetch train/val/test splits
- run isolated per-case results under `Results/cases/<case>/...`
- resume by rerunning the same script
- reuse the same dataset root unless the cache is removed

Default case sweep:
- `width_only`
- `width_to_depth`
- `depth_to_width`
- `depth_only`

Current case-suite graph:
 - 13 canonical model families
- 129 adaptive jobs per supported case, plus 50 vanilla jobs when vanilla is enabled
- identical job graph for `pglib_opf_case14_ieee`, `pglib_opf_case30_ieee`, `pglib_opf_case57_ieee`, `pglib_opf_case118_ieee`, and `pglib_opf_case500_goc`

Default repetitions:
- `5` (`--repeat-count 0` requests one canonical pass)

## 10. Width-Only Model Algorithms

The width-only families are:
- `loss_guided_neural_densification_width_growth_stop_on_first_non_improvement` with strict stop-on-first-non-improvement control flow
- `patience_based_width_growth_all_layers_together` with `width_growth_strategy=all_layers`
- `patience_based_width_growth_round_robin_layer_wise` with `width_growth_strategy=round_robin`

The inverse depth-to-width families are:
- `depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1` with `loop_order=depth_to_width`
- `depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width` with `loop_order=depth_to_width` and `seed_new_depth_to_current_width=1`
- `depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width` with `loop_order=depth_to_width`, `width_growth_strategy=round_robin`, and `seed_new_depth_to_current_width=1`
- `depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1` with `loop_order=depth_to_width`, `width_growth_strategy=all_layers`, and `seed_new_depth_to_current_width=0`

These are width-growth-only architectures. They do not add depth.

### 10.1 Common width-only training constants

From shared defaults and runner wiring:
- inner early stopping patience: `10`
- width expansion patience: `10`
- expansion step: `1`
- width delta threshold: default `0.0` unless overridden
- max width: `512`
- max depth: `5`
- max neurons: `1_000_000_000`
- max epochs: unlimited by default (`0` internal sentinel; CLI/environment also accept `inf` or `unlimited`)

### 10.2 Width-only forward pass

- The OPF case-suite width-only path now runs the `single_head`, `two_heads`, and `four_heads` variants and returns the direct regression head output
- Hidden layers are ReLU-activated

### 10.3 Width-only expansion algorithms

LG-ND width-only:

1. Train the current architecture with early stopping.
2. Keep the best validation snapshot.
3. Expand every hidden layer width by `1` neuron.
4. Re-train with early stopping.
5. If validation improves, accept the new best snapshot and continue.
6. Otherwise stop immediately on the first non-improvement.

Simultaneous width growth variant:

1. Train the current architecture with early stopping.
2. Remember the best validation snapshot.
3. Expand every hidden layer width by `1` neuron.
4. Re-train with early stopping.
5. If validation improves by more than `delta_width`, accept the new best snapshot and reset width-failure count.
6. Otherwise increment width-failure count.
7. Stop when width-failure count reaches `10`, or width/max-neuron limits are reached.
8. Restore the best snapshot once at the end.

Round-robin width growth variant:

1. Train the current architecture with early stopping.
2. Remember the best validation snapshot.
3. Set the next target width to `current_width + 1`.
4. Grow one hidden layer by `1` neuron at a time, rotating through the hidden stack.
5. Re-train after each growth step.
6. If validation improves by more than `delta_width`, accept the new best snapshot and reset width-failure count.
7. Otherwise increment width-failure count.
8. Stop when width-failure count reaches `10`, or width/max-neuron limits are reached.
9. Restore the best snapshot once at the end.

Important detail:
- the model does not roll back the architecture after a failed width expansion
- it keeps the widened architecture in place and continues until expansion patience is exhausted

### 10.4 Head variants in the case suite

The OPF case suite launches the `single_head`, `two_heads`, and `four_heads` variants together. The shared launcher emits all three head variants for each supported family by default.

## 11. Depth-Only and Width-to-Depth Algorithms

The depth-growth families are:
- `stagewise_depth_growth_start_new_stage_at_width_1`

The same model class is also used for the modern width-to-depth families through runner flags.

### 11.1 Common depth-growth constants

- early stopping patience: `10`
- depth expansion patience: `10`
- expansion step: `1`
- depth delta threshold: default `0.0` unless overridden
- max width: `512`
- max depth: `5`
- max neurons: `1_000_000_000`

### 11.2 New depth stage seeding

When a new depth stage is added:
- if `seed_new_depth_to_current_width=False`, the new hidden layer starts at width `1`
- if `seed_new_depth_to_current_width=True`, the new hidden layer starts at the current width

The suite wires:
- neuron-by-neuron modern width-to-depth: `seed_new_depth_to_current_width=False`
- match-width width-to-depth: `seed_new_depth_to_current_width=True`

The suite also wires the width-growth policy:
- neuron-by-neuron modern width-to-depth: `width_growth_strategy=round_robin`
- match-width width-to-depth: `width_growth_strategy=all_layers`

### 11.3 Uniform-width gate for adding depth

The model can add a new depth stage only when:
- there is at least one hidden layer
- all hidden layer widths are uniform
- depth + 1 does not exceed max depth
- the total neuron budget is still under `max_neurons`

This is the exact uniform-width gate the user asked about.

### 11.4 Terminal-layer widening

After a depth stage is introduced:
- the neuron-by-neuron family widens one hidden layer at a time in round-robin order until all hidden widths reach the stage target
- the match-width family widens all hidden layers together by one neuron per step
- after each widening step, the stage is retrained with early stopping
- the best snapshot is tracked separately from the current architecture

### 11.5 Outer depth-expansion loop

The outer loop behavior is:

1. Train the current stage to early stopping with patience `10`.
2. If the validation improves by more than `delta_depth`, update the best snapshot.
3. Otherwise increment depth-failure count.
4. If the stage is the neuron-by-neuron variant, widen one hidden layer at a time in round-robin order until all widths are uniform at the stage target.
5. If the stage is the match-width variant, widen all hidden layers together until the next uniform width is reached.
6. Once the stage target is satisfied and the architecture is uniform, the runner may add a new depth stage.
7. Stop when depth-failure count reaches `10` or the depth/neuron budget is exhausted.
8. Restore the best snapshot once at the end.

### 11.6 Modern width-to-depth vs modern depth-only

The shared model class is the same.

The difference is in the runner schedule:
- width-to-depth families sweep width values `1,2,4,8,16,32,64,128,256,512`
- depth-only families use the same expansion mechanics but are scheduled as the dedicated depth-only case family

In both paths:
- the new layer begins at width `1` unless the match-width flag is set
- the neuron-by-neuron family uses round-robin hidden-layer widening
- the match-width family uses all-layers widening
- depth growth is gated by uniform hidden widths

### 11.7 Head variants in the case suite

The OPF case suite launches the `single_head`, `two_heads`, and `four_heads` depth-growth variants together. They remain in-tree and are wired into the current case runners by default.

## 12. Exact Runner Values Used in the Suite

Launcher and suite defaults:
- `CONCURRENCY=0` meaning unbounded-by-default, then gated by resources
- `MAX_ACTIVE_JOBS=0`
- `MAX_ACTIVE_GPU_JOBS=0`
- `LAUNCH_HOST_RESERVE_MIB=0`
- `LAUNCH_GPU_RESERVE_MIB=0`
- `GPU_CHILD_GATE_CLOSE_BUFFER_MIB=100`
- `HOST_CHILD_GATE_CLOSE_BUFFER_MIB=500`
- `HOST_RAM_STOP_AVAIL_MIB=1024`
- `HOST_RAM_RESUME_AVAIL_MIB=1280`
- `GPU_STOP_AVAIL_MIB=100`
- `GPU_RESUME_AVAIL_MIB=500`
- `PRESSURE_POLL_INTERVAL_SEC=0.5`
- `POST_LAUNCH_SAMPLE_DELAY_SEC=30`
- `BATCH_BACKOFF_FACTOR=0.5`
- `SWAP_PRESSURE_LIMIT_PCT=70`
- `SWAP_RESUME_PCT=70`
- `OPF_ADP_DATA_CHUNK_SIZE=512`
- `repeat_count=5` (`--repeat-count 0` is the single-pass override used for the fresh-run command in this thread)
- `MISSING_PRESENT_TASK_REPEATS=1`
- `PREDICTION_REPEATS=1`

Model defaults:
- `patience=10`
- `trials_width=10`
- `trials_depth=10`
- `ex_k=1`
- `delta=0`
- `init_width=1`
- `max_width=512`
- `max_depth=5`
- `max_neurons=1_000_000_000`

## 13. Practical Efficiency Knobs

To push the runtime on a CPU-heavy machine without editing code, the main knobs are:

- `TABULAR_CPU_AFFINITY_CPUS`
- `TABULAR_CPU_JOB_CONCURRENCY`
- `TABULAR_CHILD_SHARED_CPU`
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `TORCH_NUM_THREADS`
- `TORCH_INTEROP_THREADS`
- `OMP_PROC_BIND`
- `OMP_PLACES`
- `KMP_AFFINITY`
- `CUDA_VISIBLE_DEVICES`
- `PYTORCH_CUDA_ALLOC_CONF`

For memory discipline:
- keep swap disabled
- keep the host and GPU stop/resume thresholds unchanged unless the machine class changes
- keep the child gate-close buffers enabled so the next admission only opens after the last child truly releases memory

## 14. Confirmed Behavior Summary

Confirmed in the current implementation:
- width-only models in the case suite run as `single_head`, `two_heads`, and `four_heads` variants
- loss-guided neural densification width-only stops on the first non-improvement instead of keeping a patience loop
- the width-only suite includes both simultaneous widening and round-robin widening variants
- depth-to-width models mirror the width-to-depth loop order with the same patience `10`
- stagewise depth-growth and width-to-depth use the depth-growth engine with patience `10`
- new depth stages start at width `1` unless the match-width flag is enabled
- a new depth stage is only allowed once the widths are uniform across the hidden stack
- width-first round-robin variants use round-robin hidden-layer widening
- match-width width-to-depth widens all hidden layers together
- modern depth-to-width uses the same growth policies with the loop order inverted
- the shared launcher and suite runner now persist the child-footprint gate-lock state
- the suite writes resumable run-root manifests and per-job state files
- the suite is wired through a single shared launcher contract
- direct Python runner use includes vanilla jobs by default; `--no-include-vanilla` is the explicit opt-out
- direct adaptive example calls apply the resolved configuration seed after overrides are merged

## 15. Additional In-Tree Model Analogs

The repository also contains explicit `two_heads` and `four_heads` analogs of the width-only, width-to-depth, depth-to-width, and depth-only families.
They follow the same growth rules as the base `single_head` variants and are wired into the current OPF case-suite launcher path by default.

## Exact training resume state (schema 2)

Batch checkpoints now include Python/NumPy/Torch CPU/CUDA RNG state, the active train sampler permutation, model/optimizer/scheduler state, early-stop counters and best-state metadata, batch/epoch cursor, backend determinism flags, runtime versions, and the loader contract. Fresh OPF shuffles match PyTorch's default DataLoader/RandomSampler RNG consumption; mid-epoch restore avoids replaying the iterator's base-seed draw. Checkpoint replacement is atomic. Native jobs also save partial-epoch loss accumulators and a boundary checkpoint after scheduler, validation, and early-stopping updates. Resume rejects a loader contract mismatch. Checkpoints predating schema 2 are rejected for continuation when selected. Missing historical random state or sample order cannot be reconstructed, so the runner marks that job failed without retrying it. Numerical bitwise equivalence remains conditional on matching data and deterministic operations on a compatible software/hardware stack.
The external loss-guided adapter has no scheduler in its training algorithm. It saves AdamW, early-stop and partial-epoch metric state, RNG, sampler order, and loader state, then restores the completed candidate random stream before a growth transition.

Stage-growth sidecars also reference an atomic companion RNG checkpoint. Loading the stage sidecar restores the random stream before the controller makes its next growth transition, covering restart windows between stages.
