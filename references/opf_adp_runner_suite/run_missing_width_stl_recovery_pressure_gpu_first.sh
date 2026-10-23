#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TASKS="${TASKS:-width_only,width_to_depth}"
export SURFACE="${SURFACE:-active}"
export RESULTS_DIR="${RESULTS_DIR:-Results/massive_opf_suite_recovery}"

exec ./run_massive_opf_suite.sh "$@"
