# Case Runners

The repository now ships one GPU-first runner per OPF case.

Cases covered:

- `pglib_opf_case14_ieee`
- `pglib_opf_case30_ieee`
- `pglib_opf_case57_ieee`
- `pglib_opf_case118_ieee`
- `pglib_opf_case500_goc`

Each case runner:

- Prefetches the train/val/test splits for that case before launching jobs.
- Uses the same pressure-aware GPU-first scheduler as the main suite.
- Routes through the shared launcher contract in `utils/opf_suite_launcher.sh` and `utils/opf_suite_launcher.py`.
- Pulls common run defaults from `DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py`, `utils/opf_shared_defaults.py`, and the shared launcher defaults in `utils/opf_run_defaults.sh` / `utils/opf_run_defaults.ps1`.
- Keeps its own `Results/cases/<case>/run_root` tree.
- Writes non-overlapping results under `Results/cases/<case>/...`.
- Is resumable by rerunning the same script after a `git pull` on another machine, or by restoring the matching local `Results/` tree and optional `resume_snapshots/` bundle.
- Exports a chunked portable bundle under `resume_snapshots/` on clean exit as an extra fallback.
- Large telemetry CSVs inside the local run tree are sharded into exact `telemetry.csv.partNNN` files with a `.manifest.json` sidecar so the repository stays within GitHub's file-size limit, and the launcher rehydrates them back to `telemetry.csv` before the run resumes.
- Persists per-stage checkpoints under `Results/cases/<case>/stages/<stage>/`.
- Applies the same per-stage checkpoint and callback contract to all single-, two-, and four-head depth-only paths; every depth stage emits stage metadata, metrics, and a resumable checkpoint before the next stage is admitted.
- Appends adaptive stage state and metrics into a single per-task telemetry stream under `Results/cases/<case>/.../telemetry/.../task_<id>/telemetry.csv` and, when the file becomes large, splits it into exact `telemetry.csv.partNNN` shards with a `.manifest.json` sidecar. Vanilla jobs use their native metric CSV and checkpoint output.
- Writes the authoritative live batch/epoch trace in `Results/cases/<case>/run_root/jobs/<family>/<case>/repeat_01/<job_id>/job.log`.
- Still writes `checkpoint_last.pt` / `checkpoint_best.pt` / `checkpoint_best_batch.pt` for stage resume.
- Defaults to resource-gated launch concurrency from the runner CLI. The shell wrapper no longer hardcodes a parallel cap.
- Enforces the shared architecture ceiling of width `512` and depth `5` for all OPF ADP case-suite jobs.
- Uses a shared base seed from `OPF_ADP_GLOBAL_SEED` / `--seed` and derives a deterministic per-job seed from the case, repeat, family, head variant, depth, and width so repeated runs replay the same seed schedule.
- Exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the launcher shell so PyTorch's deterministic mode has the matching cuBLAS workspace setting.
- Distinguishes terminal failures from retry attempts in the live `[STATUS]` line: `failed_attempts` counts nonzero exits that were retried, while `terminal_failed` counts jobs that were not retried.
- Bounds failure retries with `MAX_RETRIES` / `--max-retries` (default `1`); pressure pause/resume attempts do not consume this budget, and `0` disables retries.
- On `SIGINT`, `SIGTERM`, `SIGHUP`, or `SIGQUIT`, terminates child process trees and immediately rewrites active jobs as `queued` (or preserves `paused`), clearing stale PIDs without marking interrupted work as failed or completed; the next identical launch resumes from the latest checkpoint/telemetry.

## Single-Case Launch Recipe

1. Choose the case runner for the machine.
2. Set the data root if you do not want to use the local `data/` folder:

```bash
export OPF_ADP_DATA_ROOT=/absolute/path/to/opf_data
```

3. Run the case script:

```bash
./run_case_14.sh
```

4. If the run is interrupted, rerun the same command. The existing `run_root` is reused and completed jobs are skipped.

For the exact remote-machine resume contract, see [Agent Resume Protocol](./docs/AGENT_RESUME_PROTOCOL.md).

## Scripts

- `./run_case_14.sh`
- `./run_case_30.sh`
- `./run_case_57.sh`
- `./run_case_118.sh`
- `./run_case_500.sh`

## What each case runner executes

The case runner schedules 13 OPF ADP families across 3 head variants by default: `single_head`, `two_heads`, and `four_heads`, and also adds the fixed one-shot vanilla grid when `INCLUDE_VANILLA=1`.

The canonical launcher model names use the exact family-plus-head form shown above. For example, the live runner resolves `loss_guided_neural_densification_width_growth_stop_on_first_non_improvement_single_head` to `Dyn_DNN4OPF.examples.run_adp_lg_nd_width_only`.

## Complete Model Smoke Audit

To exercise every job in an existing suite manifest without changing its real results, run:

```bash
.venv_opf_adp/bin/python utils/verify_opf_suite_models.py \
  --manifest Results/cases/case14/run_root/job_manifest.json
```

The verifier reuses each manifest child command, redirects results to a temporary isolated directory, runs one small-data epoch, and checks the exit code, finite metrics, optimizer step, and checkpoint. Adaptive jobs additionally emit progress state; vanilla jobs use their native metric log. Adaptive smoke jobs cap both growth dimensions at each job's requested initial architecture by default because this audit verifies each individual model/configuration's training path; use `--allow-adp-expansion` for a slower expansion-path audit. The report is written as `verification_report.json` under the selected temporary output root. Use `--start` and `--limit` to split a large manifest across independent processes.

Validation completed on 2026-08-28 for a Case 14 manifest with vanilla enabled: all `179/179` jobs passed with zero nonzero exits, finite-metric failures, or unverified training paths. Every job produced at least one optimizer step, checkpoint, and metric CSV row; adaptive jobs also produced progress sidecars. The audit used the real prepared Case 14 cache, one epoch, and small train/validation/test windows; it verifies each configuration's end-to-end forward/backward/checkpoint path, not production-scale convergence or full ADP expansion behavior.

To exercise one bounded growth path for every adaptive family/head variant, select the unique variants and cap both dimensions at 2:

```bash
.venv_opf_adp/bin/python utils/verify_opf_suite_models.py \
  --manifest Results/cases/case14/run_root/job_manifest.json \
  --unique-model-variants \
  --expansion-max-width 2 \
  --expansion-max-depth 2 \
  --output-root /tmp/opf_adp_suite_expansion_verify
```

Width-only families:

- `loss_guided_neural_densification_width_growth_stop_on_first_non_improvement`: all-layers width growth, strict stop-on-first-non-improvement control flow, depths `1,2,3,4,5`
- `patience_based_width_growth_round_robin_layer_wise`: round-robin width growth, patience-based widening, depths `1,2,3,4,5`
- `patience_based_width_growth_all_layers_together`: all-layers width growth, patience-based widening, depths `1,2,3,4,5`

Width-to-depth families:

- `width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1`: round-robin width growth, then add depth seeded at width `1`
- `width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width`: all-layers width growth, then add depth seeded at the current width
- `width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width`: round-robin width growth, then add depth seeded at the current width
- `width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1`: all-layers width growth, then add depth seeded at width `1`

Depth-to-width families:

- `depth_first_round_robin_widen_then_add_width_seed_new_width_at_width_1`: round-robin width growth with depth growth as the outer loop
- `depth_first_all_layers_widen_then_add_width_seed_new_width_at_current_width`: all-layers width growth with depth growth as the outer loop and matched seeding
- `depth_first_round_robin_widen_then_add_width_seed_new_width_at_current_width`: round-robin width growth with depth growth as the outer loop and matched seeding
- `depth_first_all_layers_widen_then_add_width_seed_new_width_at_width_1`: all-layers width growth with depth growth as the outer loop

Depth-only families:

- `stagewise_depth_growth_start_new_stage_at_width_1`: new depth stages start at width `1`, then the terminal layer widens one neuron at a time until it matches the previous uniform width
- `stagewise_depth_growth_start_new_stage_at_current_width`: new depth stages start at the current width, then are compared globally after each stage
- vanilla one-shot: fixed `MLP` jobs over the depth grid `1,2,3,4,5` and width grid `1,2,4,8,16,32,64,128,256,512`

Per case, that gives 129 adaptive jobs total:

- width-only: `3` families × `5` seed depths × `3` head variants = `45`
- width-to-depth: `4` families × `1` seed configuration × `3` head variants = `12`
- depth-to-width: `4` families × `1` seed configuration × `3` head variants = `12`
- depth-only: `2` families × `10` seed widths × `3` head variants = `60`

That is `43 × 3 = 129` adaptive jobs per supported case.

Vanilla one-shot jobs add `5 × 10 = 50` more jobs per case:

- depths: `1,2,3,4,5`
- widths: `1,2,4,8,16,32,64,128,256,512`
- hidden widths: repeat the chosen width across all hidden layers

With vanilla enabled, the total is `179` jobs per repeat, or `895` jobs per supported case at the default `repeat_count=5`.

Concurrency is controlled from the runner CLI, not hardcoded in the shell wrappers.

Default trial grids:

- Width-only depth sweep: `1,2,3,4,5`
- Width-to-depth width sweep: `1,2,4,8,16,32,64,128,256,512`
- Depth-only width sweep: `1,2,4,8,16,32,64,128,256,512`

Width-to-depth runs use the same depth sweep as width-only and now differ in both their width-growth policy and their new-layer seed:

- `width_first_round_robin_widen_then_add_depth_seed_new_depth_at_width_1`: round-robin width growth, one hidden layer at a time, then a new depth layer seeded at width `1`
- `width_first_all_layers_widen_then_add_depth_seed_new_depth_at_current_width`: all-layers width growth, all hidden layers grow together, then a new depth layer seeded at the current width
- `width_first_round_robin_widen_then_add_depth_seed_new_depth_at_current_width`: round-robin width growth, then a new depth layer seeded at the current width
- `width_first_all_layers_widen_then_add_depth_seed_new_depth_at_width_1`: all-layers width growth, then a new depth layer seeded at width `1`

All width-to-depth variants still use:

- inner early stopping patience: `10`
- outer expansion patience: `10`

The case runner schedules 13 adaptive families across the default head variants:

- width-only: 3 families over the depth seed grid `1,2,3,4,5`
- width-to-depth: 4 families seeded at `depth=1, width=1`
- depth-to-width: 4 families seeded at `depth=1, width=1`
- depth-only: 2 families over the width seed grid `1,2,4,8,16,32,64,128,256,512`
- vanilla one-shot: fixed `MLP` jobs over the depth grid `1,2,3,4,5` and width grid `1,2,4,8,16,32,64,128,256,512`

The launcher keeps the same inner and outer patience settings used elsewhere in the repo:

- Inner early stopping patience: `10`
- Outer expansion patience: `10`

Shared run defaults:

- Shared data/training defaults: `utils/opf_shared_defaults.py`
- Python runner defaults: `DNN/VANILLA/Dyn_DNN4OPF/utils/run_defaults.py`
- Central launcher contract: `utils/opf_suite_launcher.py`
- Central shell entrypoint: `utils/opf_suite_launcher.sh`
- Shell defaults: `utils/opf_run_defaults.sh`
- PowerShell defaults: `utils/opf_run_defaults.ps1`
- Global seed default: `OPF_ADP_GLOBAL_SEED=42`
- Repeat default: `REPEAT_COUNT=5` in the wrapper environment; `--repeat-count 0` collapses to a single pass for a fresh run
- Failure retry default: `MAX_RETRIES=1`; set `MAX_RETRIES=0` or `--max-retries 0` for no requeue after a nonzero exit
- Diagnostic sample budget: `256` per stage unless overridden
- Dataset prep chunk size: `512`
- Training batch size: `65536`
- DataLoader workers: `0`
- Case groups: `20`
- Samples per group: `15000`
- Split ratios: `90% train / 5% val / 5% test`
- Early stopping patience: `10`
- Runtime stop/resume thresholds: host `1024 / 1280 MiB`, GPU `100 / 500 MiB`
- Launch-admission reserves: host `0 MiB`, GPU `0 MiB`
- Child gate-close buffers: GPU `100 MiB`, RAM `500 MiB`
- Width learning rate: `5e-4`
- Depth/common learning rate: `1e-3`
- Architecture caps: width `512`, depth `5`

The launch-admission reserves are not the same as the runtime thresholds:

- Launch reserves are disabled by default here (`0` means no upfront reserve gate).
- Runtime thresholds decide when a running job is paused or resumed.
- Child gate-close buffers are advisory headroom checks; the shared launcher only locks a gate when the live child footprint is already colliding with real RAM/VRAM pressure.

## Dataset Root

The dataset root is controlled by `OPF_ADP_DATA_ROOT`.

Default:

```bash
export OPF_ADP_DATA_ROOT=data
```

The case runner prefetches the OPF dataset splits into that root before starting training.
The first download/extraction/processed-cache build for a case is serialized by a per-case inter-process lock, so concurrently launched children cannot rebuild or delete each other's cache groups. The stage diagnostics resolve `data/sample_case*.json` automatically from the configured dataset root or repo root. If the legacy sample file is absent, they use `example_0.json` from the downloaded raw case cache. These files are local generated dataset artifacts and are not tracked.

The live status counters have two scopes. `batches_done` and `epochs_done` are all-job totals across every job with telemetry or log progress evidence in the run root, including failed/retried and already completed jobs. `live_batches_done` and `live_epochs_done` count only jobs currently marked `running`, `paused`, or `suspended`. `epochs_done` counts telemetry epoch records across stages, so an ADP job that trains several expansion stages contributes one record per completed epoch in each stage; it is not only the current stage's epoch number. `telemetry_jobs` counts jobs with a progress sidecar, job log, or saved progress counters; stale job state alone does not count. A high all-job total therefore cannot be inferred from the number of currently live models.

## Results Layout

Each case writes to its own result tree:

```text
Results/cases/<case_tag>/
  run_root/
  source_run_root/
  jobs/
  <case-specific training outputs>
```

Optional local bundle:

- `resume_snapshots/`

Inside the run tree, the first places to inspect are:

- `Results/cases/<case>/run_root/run_progress.csv`
- `Results/cases/<case>/run_root/suite_launch_manifest.json`
- `Results/cases/<case>/run_root/jobs/<family>/<case>/repeat_01/<job_id>/job.log`
- `Results/cases/<case>/.../telemetry/.../task_<id>/telemetry.csv`
- `Results/cases/<case>/.../telemetry/.../task_<id>/telemetry.csv.partNNN`
- `Results/cases/<case>/.../telemetry/.../task_<id>/telemetry.csv.manifest.json`
- `Results/cases/<case>/.../stages/<stage>/logs/{train,validation,test}_metrics_summary.csv`
- `Results/cases/<case>/.../stages/<stage>/logs/{train,validation,test}_per_sample_metrics.csv`

Each run writes detailed metric CSVs under the stage logs directory, including per-sample objective gap, equality residual, inequality violation, and thermal-limit summaries for Train / Validation / Test splits.

Examples:

- `Results/cases/case14`
- `Results/cases/case30`
- `Results/cases/case57`
- `Results/cases/case118`
- `Results/cases/case500`

## Resume Behavior

To resume a case, rerun the same script with the same `RESULTS_DIR` and `RUN_ROOT`.

Example:

```bash
./run_case_14.sh
```

The job manifest and per-job state files are reused, so already completed jobs are skipped and incomplete jobs continue from their saved state.
Within each job, the latest stage checkpoint is reused from the corresponding `stages/` directory.
For adaptive width growth, the telemetry task directory also contains
`stage_state.json`. It is written before a stage starts and records the exact
hidden-width vector, round-robin cursor, global-best checkpoint, and current
stage checkpoint. If the orchestrator terminates during that stage, the next
launch restores that stage and its optimizer/model/epoch/batch state directly;
it does not replay the initial stage or advance the growth cursor twice. At
normal completion, the terminal state and diagnostics use the global-best
validation architecture, not merely the last widened architecture.
If the `Results/cases/<case_tag>` tree is missing on the new machine, the launcher restores it from the matching local bundle in `resume_snapshots/` before the suite starts.

The case runner also reuses the same dataset root, so the already downloaded train/val/test splits are not re-fetched unless the local dataset cache is removed.

If you need to rebuild just the generated dataset cache for a case while keeping the raw downloads, run:

`--fresh` removes the generated processed cache and extracted group cache for that case, but keeps the raw dataset tarballs.

```bash
.venv_opf_adp/bin/python utils/prepare_opf_case_dataset.py \
  --case-name pglib_opf_case500_goc \
  --data-root data \
  --fresh
```

The default chunk size is now `512`. Use `--chunk-size` only if you need to override it for a different machine.

## Distributed Use

Assign one case to one laptop:

- Laptop A: `./run_case_14.sh`
- Laptop B: `./run_case_30.sh`
- Laptop C: `./run_case_57.sh`
- Laptop D: `./run_case_118.sh`
- Laptop E: `./run_case_500.sh`

When the runs are finished, merge the `Results/cases/<case_tag>` directories from the laptops. Because each case has its own directory, the outputs do not overlap.
If you want to move a live run to another machine, copy the matching `Results/` tree and `resume_snapshots/` bundle directory alongside the repository and rerun the same case script there. The bundle carries the resumability-critical state and latest checkpoints; the full per-batch telemetry remains in the original `Results/` tree.

## Case-to-Script Map

- `pglib_opf_case14_ieee` -> `./run_case_14.sh`
- `pglib_opf_case30_ieee` -> `./run_case_30.sh`
- `pglib_opf_case57_ieee` -> `./run_case_57.sh`
- `pglib_opf_case118_ieee` -> `./run_case_118.sh`
- `pglib_opf_case500_goc` -> `./run_case_500.sh`
