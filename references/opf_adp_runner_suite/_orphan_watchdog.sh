#!/usr/bin/env bash

# Best-effort orphan watchdog for OPF_ADP launchers.
# Source this file before invoking the long-running Python orchestrator and
# wrap the command with _run_with_watchdog.

_run_with_watchdog() {
  local MASTER_PID=$$
  local GRACE_SEC=15
  local WATCHDOG_PID=""
  local PYTHON_PID=""
  local _OPF_WATCHDOG_CLEANED=0

  _watchdog_kill_process_tree() {
    local parent_pid="$1"
    local signal_name="${2:-TERM}"
    local child_pid
    local child_pids

    child_pids="$(pgrep -P "$parent_pid" 2>/dev/null || true)"
    for child_pid in $child_pids; do
      _watchdog_kill_process_tree "$child_pid" "$signal_name"
    done
    kill -"${signal_name}" "$parent_pid" 2>/dev/null || true
  }

  _watchdog_kill_python_tree() {
    local signal_name="${1:-TERM}"
    if [ -n "${PYTHON_PID:-}" ] && kill -0 "$PYTHON_PID" 2>/dev/null; then
      _watchdog_kill_process_tree "$PYTHON_PID" "$signal_name"
    fi
  }

  _watchdog_cleanup() {
    local reason="${1:-signal}"
    if [ "${_OPF_WATCHDOG_CLEANED:-0}" -ne 0 ]; then
      return 0
    fi
    _OPF_WATCHDOG_CLEANED=1
    echo "[RUNNER] $reason received. Stopping Python orchestrator (PID ${PYTHON_PID:-unknown})..." >&2
    _watchdog_kill_python_tree TERM
    sleep 2
    _watchdog_kill_python_tree KILL
    if [ -n "${WATCHDOG_PID:-}" ]; then
      kill -KILL "$WATCHDOG_PID" 2>/dev/null || true
    fi
    trap - INT TERM HUP QUIT EXIT
    return 0
  }

  "$@" &
  PYTHON_PID=$!

  (
    sleep 2
    while true; do
      sleep 3
      if ! kill -0 "$MASTER_PID" 2>/dev/null; then
        echo "[WATCHDOG] master $MASTER_PID gone; cleaning python children" >> "/tmp/opf_watchdog_${MASTER_PID}.log" 2>&1
        _watchdog_kill_python_tree TERM
        sleep "$GRACE_SEC"
        _watchdog_kill_python_tree KILL
        exit 0
      fi
    done
  ) </dev/null >>"/tmp/opf_watchdog_${MASTER_PID}.log" 2>&1 &

  WATCHDOG_PID=$!
  disown "$WATCHDOG_PID" 2>/dev/null || true
  trap '_watchdog_cleanup INT' INT
  trap '_watchdog_cleanup TERM' TERM
  trap '_watchdog_cleanup HUP' HUP
  trap '_watchdog_cleanup QUIT' QUIT
  trap '_watchdog_cleanup EXIT' EXIT

  wait "$PYTHON_PID"
  local EXIT_CODE=$?
  _OPF_WATCHDOG_CLEANED=1
  trap - INT TERM HUP QUIT EXIT
  if [ -n "${WATCHDOG_PID:-}" ]; then
    kill -KILL "$WATCHDOG_PID" 2>/dev/null || true
  fi
  rm -f "/tmp/opf_watchdog_${MASTER_PID}.log" 2>/dev/null || true
  return "$EXIT_CODE"
}
