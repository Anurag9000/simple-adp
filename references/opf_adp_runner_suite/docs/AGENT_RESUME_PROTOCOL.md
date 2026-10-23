# Agent Resume Protocol

This document is the canonical resume contract for OPF_ADP runs.
It exists so an automated agent on another machine can restore the exact same
case suite state, verify that restoration is correct, and repair the setup if
anything is missing or inconsistent.

## Scope

This protocol applies to every local case-suite run under `Results/cases/<case>/`,
especially `case14`, `case30`, `case57`, `case118`, and `case500`.

It covers:

- the local `Results/` tree
- `run_root/` manifests and scheduler state
- model checkpoints and optimizer / scheduler state
- per-batch telemetry CSVs
- sharded telemetry CSV recovery
- batch-level resume cursors
- the optional local `resume_snapshots/` bundle
- resumability verification on a different machine

## Canonical Resume Rule

The repository `main` branch is the source of code.
On a new machine, the correct resume procedure is:

```bash
git pull origin main
./run_case_14.sh
```

If the venv is not active:

```bash
source .venv_opf_adp/bin/activate
git pull origin main
./run_case_14.sh
```

The launcher must then:

1. Restore the saved `Results/cases/<case>/...` tree or the matching
   `resume_snapshots/` bundle that was copied from the source machine.
2. Rehydrate any `telemetry.csv.partNNN` shard sets back into the original
   `telemetry.csv` files.
3. Load the latest stage-local checkpoint state for each job. For adaptive jobs,
   this is `stages/<stage>/checkpoint_last.pt`, not the telemetry copy alone.
4. Resume from the saved epoch / batch cursor.
5. Append new telemetry rows to the existing CSV rather than replacing it.

Repository policy for new run outputs:

- `Results/` and `resume_snapshots/` are generated locally and ignored by
  default.
- top-level plot artifacts are also treated as generated outputs.
- a normal `git add .` should not stage those generated outputs unless they are
  intentionally copied elsewhere.

## What Must Be Present

For each running adaptive model/task, the following files are the minimum
expected resume artifacts:

- `stages/<stage>/checkpoint_last.pt`
- `stages/<stage>/checkpoint_best.pt`
- `stages/<stage>/checkpoint_best_batch.pt`
- telemetry copies of those three files under the task telemetry directory
- `progress_state.json`
- `telemetry.csv` or the equivalent sharded set:
  - `telemetry.csv.manifest.json`
  - `telemetry.csv.part000`
  - `telemetry.csv.part001`
  - ...
- if the CSV is too large for GitHub, the matching:
  - `telemetry.csv.manifest.json`
  - `telemetry.csv.part000`
  - `telemetry.csv.part001`
  - ...

Vanilla one-shot jobs use their native metric CSV and model checkpoint files;
they do not require the adaptive `progress_state.json` sidecar.

The massive runner does not suspend adaptive children in place. Its pressure
path terminates the launcher-owned child process tree, records a paused state,
and later relaunches the child. Therefore the stage-local checkpoint mirror is
required for correctness. Each native single-, two-, and four-head width-only
and depth-only engine writes the same payload atomically to both locations.

The case runner also maintains:

- `run_root/suite_launch_manifest.json`
- `run_root/batch_backoff_state.json`
- `jobs/<job_id>/job_state.json`
- `jobs/<job_id>/job_metadata.json`
- `jobs/<job_id>/command.json`
- `jobs/<job_id>/launch_context.json`
- `jobs/<job_id>/candidate_summary.json`
- `jobs/<job_id>/job.log`

The exact place to find the live per-batch / per-epoch trace is:

```bash
find Results/cases/case14/run_root/jobs -name job.log
find Results/cases/case14/pglib_opf_case14_ieee/repeat_01 -path '*/telemetry/*/task_*/telemetry.csv*'
```

If a reviewer cannot find the trace, these commands must be the first step before assuming the run never recorded it.

## Exact Verification Sequence

After `git pull origin main`, a remote machine should run these checks before
starting the suite:

```bash
git status -sb
git rev-parse HEAD
find Results -type f -size +100M
```

Expected outcomes:

- `git status -sb` should be clean unless the user intentionally added new local
  changes.
- `git rev-parse HEAD` should match the same commit the source machine pushed.
- `find Results -type f -size +100M` should print nothing.

Then confirm that the launcher can see the rehydration hook:

```bash
rg -n "rehydrate_sharded_telemetry" utils/opf_suite_launcher.py
```

Expected outcome:

- the launcher imports `rehydrate_sharded_telemetry`
- the helper is called before the suite launches

Then start the case:

```bash
./run_case_14.sh
```

Expected launch evidence:

- a line stating that one or more sharded telemetry files were rehydrated
- a `Launch command:` line that points to `utils/opf_massive_suite_runner.py`
- live status lines showing `queued`, `active`, `completed`, `failed`, `retried`,
  `batches_done`, and `epochs_done`

## How Resume Must Behave

The resume semantics are strict:

- `telemetry.csv` must be appended to, not truncated.
- The latest stage-local checkpoint must restore:
  - model weights
  - optimizer state
  - scheduler state
  - patience
  - epoch counter
  - batch cursor
  - best snapshot
  - best batch snapshot
  - best batch loss
  - Python, NumPy, CPU Torch, and all CUDA RNG states
  - active shuffled sampler permutation and sampler/global RNG state
  - deterministic backend flags and loader contract (dataset type/length, batch size, sampler)
  - schema/runtime version metadata and native epoch-loss accumulators
- If a job was interrupted in the middle of a stage, it must resume from the
  stored batch cursor rather than restarting the stage from batch 0.
- Loading a checkpoint must leave the latest `model_state` active. The older
  `best_snap` and `best_batch_snap` are selection metadata used for patience and
  final model selection; they must not overwrite the live model during resume.
- If a telemetry CSV was sharded, the launcher must reassemble the original CSV
  before the run begins.

The relevant implementation points are:

- [`utils/opf_suite_launcher.py`](../utils/opf_suite_launcher.py)
- [`utils/opf_resume_bundle.py`](../utils/opf_resume_bundle.py)
- [`DNN/VANILLA/Dyn_DNN4OPF/models/adp_den_width_only.py`](../DNN/VANILLA/Dyn_DNN4OPF/models/adp_den_width_only.py)
- [`DNN/VANILLA/Dyn_DNN4OPF/models/adp_den_4head_depth_only.py`](../DNN/VANILLA/Dyn_DNN4OPF/models/adp_den_4head_depth_only.py)

## What The Logs Mean

When the suite is healthy, the on-screen status must reflect the actual job
state and the saved counters.

Important fields:

- `queued`: jobs waiting to launch
- `active`: jobs currently running or admitted
- `completed`: jobs that have finished
- `failed_attempts`: nonzero exits that were retried
- `terminal_failed`: jobs that exited nonzero and were not retried
- `retried`: jobs that were re-queued after failure
- `attempt`: launch and pressure pause/resume attempts for a job
- `retry_count`: failed-attempt requeues consumed from `MAX_RETRIES` / `--max-retries`; default `1`, `0` is unbounded, and positive values impose that retry limit
- `batches_done`: cumulative batch records across every telemetry-bearing job in the run root
- `epochs_done`: cumulative epoch records across every telemetry-bearing job in the run root
- `live_batches_done`: batch records belonging only to jobs currently `running`, `paused`, or `suspended`
- `live_epochs_done`: epoch records belonging only to jobs currently `running`, `paused`, or `suspended`
- `telemetry_jobs`: jobs with a progress sidecar, job log, or saved progress counters
- `telemetry_dirs`: directories containing telemetry outputs

The all-job counters intentionally include telemetry left by failed/retried and
already completed jobs in the same run root. For ADP jobs, each expansion stage
emits its own epoch records, so these counters are not equivalent to the current
stage's displayed epoch number. Use the `live_*` counters when assessing only
the currently admitted models.

If `batches_done` or `epochs_done` are still zero after a resumed run has already
written telemetry and checkpoints, that is a bug in the status aggregation path,
not proof that the model did not train.

## Repair Steps If Resume Is Not Exact

If the remote machine does not resume exactly, fix it in this order:

1. Confirm the machine is on the pushed commit:

   ```bash
   git pull origin main
   git rev-parse HEAD
   ```

2. Confirm the local `Results/` tree exists.

3. Confirm the sharded telemetry manifests and parts exist if the original CSV
   was too large.

4. Confirm the launcher includes the rehydration hook:

   ```bash
   rg -n "rehydrate_sharded_telemetry" utils/opf_suite_launcher.py
   ```

5. Confirm the resume state files exist in the case directory:

   - `run_root/suite_launch_manifest.json`
   - `jobs/<job_id>/job_state.json`
   - `stages/<stage>/checkpoint_last.pt`
   - the matching telemetry checkpoint copies
   - `progress_state.json`

6. If a `telemetry.csv` file is missing but the shard parts exist, the launcher
   should recreate it. If it does not, the fix is to restore the launcher hook
   from `utils/opf_suite_launcher.py` and rerun.

7. If stage-local checkpoints are missing, restore from the matching bundle under
   `resume_snapshots/` or from the previously saved `Results/` tree.

8. Only once the files are present and the helper is wired in should you rerun:

   ```bash
   ./run_case_14.sh
   ```

## Non-Goals

This protocol does not change the experiment graph, model family definitions,
or the training hyperparameters. It only defines how a run is restored,
verified, and repaired.


## Checkpoint schema 2 and exact continuation

Adaptive and native batch checkpoints use schema 2. They are written atomically after an optimizer step and include model, optimizer, scheduler, early-stop state, current epoch/batch, best-selection state, random number generator states (Python, NumPy, Torch CPU, and CUDA), and the training sampler permutation. OPF training loaders use a checkpointable sampler that matches PyTorch's default `RandomSampler(generator=None)` seed and permutation behavior. On mid-epoch restore, the replacement iterator's duplicate base-seed draw is undone before the next batch. Native jobs also save a post-validation epoch-boundary checkpoint so an interruption there cannot apply the scheduler step twice. The loader contract is checked on restore; a mismatch fails closed instead of silently continuing with a different batch shape or sample count.

Older training checkpoints that lack RNG or sampler persistence are rejected for continuation. For compatibility, a legacy stage-state JSON without its separate RNG sidecar may use the active incomplete stage's canonical stage-local schema 2 `checkpoint_last.pt` only after validating the full training/RNG/sampler state and matching the saved architecture. RNG restore is deferred until that training checkpoint loads after architecture reconstruction. Completed stages without the companion remain rejected because their next architecture transition has no proven RNG source. Exact numerical replay also depends on the same dataset contents, software and hardware stack, worker behavior, and deterministic kernels; schema metadata records versions and backend settings, but cannot make nondeterministic kernels deterministic retroactively.

Depth-growth controllers can widen the model several times while keeping one
stage checkpoint path. When a grown architecture makes the previous
`checkpoint_last.pt` stale, the trainer must atomically replace it with a full
schema 2 stage-start checkpoint before the next optimizer update, then refresh
the active sidecar's width vector. If interruption lands between those two
writes, resume may reconcile the sidecar from that same-stage checkpoint only
when its full RNG/sampler state is present, its `model_state` tensor shapes
match its architecture snapshot, and the saved widths form a valid monotonic
transition within the sidecar's recorded stage start/target bounds. A legacy
recovered depth sidecar without a target is eligible only for the known
one-step all-layers seed transition from every layer at the recorded start
width to every layer at start width plus one. Other architecture disagreements
remain fail-closed.

The vanilla/native trainer now fails closed if an existing checkpoint cannot be loaded or a batch checkpoint cannot be written. It does not silently restart from initial weights or continue after a failed save; the runner can retry from the last atomic checkpoint.

The pressure-aware runner classifies OOM and exact-resume failures from the
latest `=== LAUNCH` attempt block in each job log. Older failed attempts remain
in the append-only log and must not determine the outcome of a later retry.
Retries use normal pressure-aware device selection; the legacy forced-CPU flag
is ignored. Independent queued jobs can use CPU when the host pressure gate
admits them.

When a run root is reused after a runner restart, manifest comparison resolves
absolute `argv[0]` symlinks before deciding whether the plan changed. For
example, `python` and `python3` remain equivalent when they resolve to the
same interpreter; any other command or suite-plan difference still fails
closed.

Stage growth sidecars now point to an atomic companion RNG checkpoint. The telemetry layer restores that state when loading a valid stage sidecar, so a restart between completed-stage bookkeeping and the next growth operation can reproduce the layer initialization stream. For a legacy active stage with no companion, the verified stage-local per-batch checkpoint remains authoritative and supplies RNG when loaded for training.
