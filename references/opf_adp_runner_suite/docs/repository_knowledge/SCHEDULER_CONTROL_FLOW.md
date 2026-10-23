# OPF Suite Scheduler: Current Static Control-Flow Map

## Snapshot and reuse boundary

Inspected worktree: `main` at HEAD `6461b8ee8d378ef6e5c7eacee9bbc6abde63f20c`, inspected 2026-10-18. `utils/opf_massive_suite_runner.py`, `utils/opf_shared_defaults.py`, `utils/pressure_aware_wrapper.py`, shell/PowerShell defaults, selected DNN resume sources, and several existing docs are modified relative to HEAD. This document describes the current worktree where those differences matter. It is a static source trace; no application code, tests, or workloads were run.

## Entrypoint and run-plan construction

`utils/opf_massive_suite_runner.py:main` performs these steps:

1. Parse arguments and resolve `include_vanilla` from `INCLUDE_VANILLA` when the CLI value is unset; the environment fallback defaults true (`main`, lines 2515-2518).
2. Validate argument ranges, initialize repository runtime bootstrap, and enable optional GPU accelerators only when neither explicit nor inherited CPU-only admission applies (lines 2515-2535).
3. Resolve/create `run_root`, `results_root`, and the job directory; seed from an optional source run root.
4. Build the `JobSpec` list from cases, task/surface filters, heads, family grids, repeat controls, and optional vanilla depth-width grid (`build_suite_jobs`, lines 1091-1215).
5. Write a signed manifest. If an existing plan differs, the runner writes `manifest_mismatch.json` and exits rather than replacing it. The current worktree additionally treats absolute interpreter symlink aliases as equivalent while requiring other plan fields to match (`write_manifest`, lines 1375-1485).
6. Write run metadata. `--dry-run` emits the manifest and returns before scheduling.
7. For a real launch, reset transient gate/backoff/hold state to a clean controller baseline; this does not erase job checkpoints or per-job state (lines 2580-2660).
8. Load/reconcile persisted job states into completed, terminal-inexact-failure, active/recovered, suspended, or queued collections (lines 2660-2690; `load_existing_runtime`, lines 1765-1791).

`build_child_command` maps each job to `universal_main.py`. Vanilla jobs use the fixed one-shot model with depth and width; adaptive jobs resolve a family/head alias and pass initial architecture, patience, growth policy, and any family-specific loop/seeding flags (`build_child_command`, lines 981-1090). The full family grid and CLI filter surface still require a dedicated algorithm/manifest pass.

## Persisted job identity and recovered processes

- Job identity includes phase, case, repeat, and architecture label (`build_job_id`, lines 977-979).
- State files are JSON-written through the atomic helpers (`load_job_state`/`save_job_state`, lines 1217-1224).
- If a saved PID is absent or stale, the runner searches for a process whose normalized full command exactly matches the saved command; a found process is represented by `RecoveredProcessHandle` (`_find_live_pid_for_command`, lines 1251-1276).
- A running/launching/retrying state with a live process is reconciled to running. A stale running state without a live process is queued. A paused state stays paused. If a recovered process later disappears, its actual return code is unknowable, so the runner persists and requeues it for checkpointed continuation rather than fabricating an exit status (`_reconcile_stale_job_state`, lines 1278-1343; `_requeue_recovered_runtime_after_exit`, lines 1345-1373).
- Saved completed state with zero exit is skipped at initialization. A failed exact-resume state is treated as terminal across controller restart. Other failed states enter the queue for a new launch attempt (lines 2660-2690).

## Admission and resource loop

Every controller cycle samples host available memory, swap use, GPU memory, active process-tree RSS, per-child GPU footprint, and external non-Python memory. The effective host availability is `psutil.virtual_memory().available`; active RSS is diagnostic and is not subtracted a second time. GPU free memory comes from `nvidia-smi`. Swap is recorded in resource snapshots but is not an admission or pause condition in this worktree (`sample_host_memory`, lines 422-439; main loop around lines 2940-3035).

The scheduler closes admission when free VRAM or available host RAM crosses stop thresholds, or when the most recent child footprint plus its configured headroom would collide with remaining resources. External non-Python memory release can reopen a corresponding gate. Explicit active-job caps and reserve settings can further limit launches; zero active caps and zero reserve values mean those extra caps/reserves are disabled, but pressure thresholds and gate logic remain operative (`resolve_launch_capacity`, lines 949-975; main loop around lines 3040-3120).

`select_launch_device` enforces CUDA visibility/CPU-only parent constraints and chooses an eligible lane. In `gpu_first`, it selects GPU when GPU and host admission are both open, otherwise CPU if host admission is open. Therefore an OOM retry is not permanently marked CPU-only, but it can naturally be admitted on CPU if the next GPU admission check is closed and CPU admission is open (`select_launch_device`, lines 1814-1855). This distinction corrects wording in older docs that says a CUDA-OOM recovery always resumes on the original device.

After an ordinary pressure pause, `_suspend_runtime_for_pressure` terminates the owned process tree, persists `paused`, clears the PID, and moves the runtime from active to suspended (`lines 2072-2118`). The resume path calls `launch_job` with the saved runtime's current `device_mode`; the main loop also requires that same lane's resume threshold before relaunch (`_resume_runtime_from_pause`, lines 2121-2174; main loop around lines 3130-3190). A relaunch is therefore a new child process and depends on the model checkpoint implementation to recover training state.

After each launch, the controller sets a per-lane hold-until timestamp and sleeps for the configured delay before the next fresh resource poll. Defaults are 30 seconds for this worktree (`utils/opf_shared_defaults.py:118`, `utils/opf_run_defaults.sh:37`); the CLI allows a different value. The main loop polls between resource decisions at the configured pressure interval.

## Exit, retry, completion, and failure semantics

`finalize_job` sets `status=completed` when the child exit code is zero and `status=failed` otherwise, then persists job state, compatibility views, a candidate summary, and a progress CSV row (`finalize_job`, lines 2225-2328). The scheduler itself does not independently prove convergence, minimum epoch count, finite metrics, constraint feasibility, or a family-specific scientific stopping criterion before assigning `completed`; those properties must be checked in the child implementation and artifacts. A runner-level completed label is thus a zero-exit outcome, not a standalone scientific-validity certificate.

For nonzero child exits, the runner inspects the latest `=== LAUNCH` attempt block (not stale prior retry text) to classify OOM and unrecoverable exact-resume errors (`_last_job_attempt_log`, lines 641-695). A nonzero exit increments failed-attempt count. Inexact checkpoint recovery becomes terminal, because retry cannot recreate missing state. Other failures are requeued while retry allowance remains; `--max-retries 0` means no fixed retry cap. OOM retries reduce the transient `batch_scale` by the configured factor and persist backoff state. A new controller invocation resets this transient scale to `1.0`; per-job resumability remains (`should_retry`, lines 1793-1801; OOM path lines 3228-3298; initialization lines 2580-2630).

The legacy `force_cpu_after_cuda_oom` state field is retained for compatibility but set false when loading or writing views, and it no longer forces the next device selection. The CUDA-OOM retry is queued back through ordinary scheduler selection. A separate still-running GPU peer may be pressure-paused after a CUDA OOM; that peer's pressure-pause recovery retains its current device mode. This is different from migrating the OOM-failed job to CPU.

Launch-time `Popen` failures are counted and use the same retry budget, with `reason=launch_error`; exhausted failures finalize as failed (main loop lines 3498-3541). On signal or keyboard interruption, the owned runtime process trees are terminated and jobs are persisted for restart; the detailed handler still needs a separate full read (`_shutdown_all_runtimes`, definition near main initialization).

## Important documentation discrepancies found

1. `docs/opf_orchestration_report.md:355-359` says the standalone wrapper's host release condition includes swap. In current `utils/pressure_aware_wrapper.py:212-244`, host readiness uses host RAM percentage and external RAM/GPU drop conditions; swap is sampled only for the status message. The runner parser likewise labels swap options legacy and non-gating (`utils/opf_massive_suite_runner.py:790-791`).
2. `docs/OPF_SUITE_AUDIT.md` and `docs/opf_orchestration_report.md` say CUDA-OOM recovery retries on the original device. Current retry flow enqueues the failed runtime and later applies normal launch-device selection; under `gpu_first`, this can be CPU if GPU admission is closed and host admission is open. Pressure-paused job resume is the path that explicitly retains its original device mode.
3. Existing docs describe a completed runner job as finished, but the scheduler's terminal decision is based on child exit code `0`. Any stronger completion criteria have to be traced to the child training/evaluation path and actual run artifacts; controller status alone does not verify them.

## Remaining inspection

This map is deliberately scoped to the central controller. Still required: full CLI/default/precedence matrix, complete transition/event audit, child worker and checkpoint call graphs per DNN family, dataset/metric pathway, manifest arithmetic under all task filters, shutdown signal handling details, and historical comparison of these mechanisms against relevant commits. Other model-family workstreams are documented separately.
