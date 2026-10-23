#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-}"
shift || true

if [[ -z "$MODE" ]]; then
  echo "usage: $0 <case|massive> [runner args...]" >&2
  exit 1
fi

source "utils/runtime_tuning.sh"
tabular_runtime_bootstrap
source "./_orphan_watchdog.sh"
source "utils/opf_run_defaults.sh"

# Make CUDA GEMM reproducible when the runner turns on deterministic PyTorch
# algorithms. PyTorch warns about this at runtime if it is left unset.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON_BIN=".venv_opf_adp/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN=".venv_opf_adp/Scripts/python.exe"
fi
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(tabular_runtime_python)"
fi

case "${MODE}" in
  case)
    : "${CASE_NAME:?CASE_NAME is required for case mode}"
    ;;
  massive)
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac

launcher_args=(--mode "$MODE")
if [ -n "${CASE_NAME:-}" ]; then launcher_args+=(--case-name "$CASE_NAME"); fi
if [ -n "${CASE_TAG:-}" ]; then launcher_args+=(--case-tag "$CASE_TAG"); fi
if [ -n "${RESULTS_DIR:-}" ]; then launcher_args+=(--results-dir "$RESULTS_DIR"); fi
if [ -n "${RUN_ROOT:-}" ]; then launcher_args+=(--run-root "$RUN_ROOT"); fi
if [ -n "${SOURCE_RUN_ROOT:-}" ]; then launcher_args+=(--source-run-root "$SOURCE_RUN_ROOT"); fi
if [ -n "${CONCURRENCY:-}" ]; then launcher_args+=(--concurrency "$CONCURRENCY"); fi
if [ -n "${CONCURRENCY_FILE:-}" ]; then launcher_args+=(--concurrency-file "$CONCURRENCY_FILE"); fi
if [ -n "${JOB_START_INDEX:-}" ]; then launcher_args+=(--job-start-index "$JOB_START_INDEX"); fi
if [ -n "${JOB_LIMIT:-}" ]; then launcher_args+=(--job-limit "$JOB_LIMIT"); fi
if [ -n "${MAX_ACTIVE_JOBS:-}" ]; then launcher_args+=(--max-active-jobs "$MAX_ACTIVE_JOBS"); fi
if [ -n "${MAX_ACTIVE_GPU_JOBS:-}" ]; then launcher_args+=(--max-active-gpu-jobs "$MAX_ACTIVE_GPU_JOBS"); fi
if [ -n "${REPEAT_COUNT:-}" ]; then launcher_args+=(--repeat-count "$REPEAT_COUNT"); fi
if [ -n "${MAX_RETRIES:-}" ]; then launcher_args+=(--max-retries "$MAX_RETRIES"); fi
if [ -n "${LAUNCH_HOST_RESERVE_MIB:-}" ]; then launcher_args+=(--launch-host-reserve-mib "$LAUNCH_HOST_RESERVE_MIB"); fi
if [ -n "${LAUNCH_GPU_RESERVE_MIB:-}" ]; then launcher_args+=(--launch-gpu-reserve-mib "$LAUNCH_GPU_RESERVE_MIB"); fi
if [ -n "${GPU_CHILD_GATE_CLOSE_BUFFER_MIB:-}" ]; then launcher_args+=(--gpu-child-gate-close-buffer-mib "$GPU_CHILD_GATE_CLOSE_BUFFER_MIB"); fi
if [ -n "${HOST_CHILD_GATE_CLOSE_BUFFER_MIB:-}" ]; then launcher_args+=(--host-child-gate-close-buffer-mib "$HOST_CHILD_GATE_CLOSE_BUFFER_MIB"); fi
if [ -n "${GPU_DEVICE_INDEX:-}" ]; then launcher_args+=(--gpu-device-index "$GPU_DEVICE_INDEX"); fi
if [ -n "${OPF_ADP_GLOBAL_SEED:-}" ]; then launcher_args+=(--seed "$OPF_ADP_GLOBAL_SEED"); fi
if [ -n "${WIDTH_TO_DEPTH_WIDTH:-}" ]; then launcher_args+=(--width-to-depth-width "$WIDTH_TO_DEPTH_WIDTH"); fi
if [ -n "${WIDTH_TO_DEPTH_DEPTH:-}" ]; then launcher_args+=(--width-to-depth-depth "$WIDTH_TO_DEPTH_DEPTH"); fi
if [ -n "${HEAD_VARIANTS:-}" ]; then launcher_args+=(--head-variants "$HEAD_VARIANTS"); fi
case "${CPU_ONLY:-0}" in
  1|true|yes|on|y|Y|TRUE|YES|ON) launcher_args+=(--cpu-only) ;;
esac
if [ -n "${OPF_ADP_DATA_ROOT:-}" ]; then launcher_args+=(--data-root "$OPF_ADP_DATA_ROOT"); fi
if [ -n "${OPF_ADP_DATA_CHUNK_SIZE:-}" ]; then launcher_args+=(--chunk-size "$OPF_ADP_DATA_CHUNK_SIZE"); fi
case "${OPF_ADP_DATA_FRESH:-0}" in
  1|true|yes|on|y|Y|TRUE|YES|ON) launcher_args+=(--fresh) ;;
esac
if [ "$#" -gt 0 ]; then
  launcher_args+=(--)
  launcher_args+=("$@")
fi

case "${OPF_ADP_DETACH:-0}" in
  1|true|yes|on|y|Y|TRUE|YES|ON)
    results_dir="${RESULTS_DIR:-}"
    if [ -z "$results_dir" ]; then
      if [ -n "${CASE_TAG:-}" ]; then
        results_dir="Results/cases/${CASE_TAG}"
      elif [ -n "${CASE_NAME:-}" ]; then
        case_tag="${CASE_NAME#pglib_opf_case}"
        case_tag="${case_tag%%_*}"
        results_dir="Results/cases/case${case_tag}"
      else
        results_dir="Results/massive_opf_suite"
      fi
    fi
    mkdir -p "${results_dir}/run_root"
    log_file="${results_dir}/run_root/launcher.log"
    pid_file="${results_dir}/run_root/launcher.pid"
    "$PYTHON_BIN" - "$log_file" "$pid_file" "${launcher_args[@]}" <<'PY'
import os
import subprocess
import sys

log_file = sys.argv[1]
pid_file = sys.argv[2]
launcher_args = sys.argv[3:]
command = [sys.executable, "-u", "utils/opf_suite_launcher.py", *launcher_args]

try:
    with open(log_file, "ab", buffering=0) as log_handle, open(os.devnull, "rb") as devnull:
        proc = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            stdin=devnull,
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
            close_fds=True,
        )
    with open(pid_file, "w", encoding="utf-8") as handle:
        handle.write(f"{proc.pid}\n")
        handle.flush()
        os.fsync(handle.fileno())
except Exception as exc:
    print(f"[OPF_ADP] Failed to detach launcher: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"[OPF_ADP] Detached launcher started: PID={proc.pid} LOG={log_file}")
PY
    exit 0
    ;;
esac

_run_with_watchdog "$PYTHON_BIN" -u utils/opf_suite_launcher.py "${launcher_args[@]}"
