#!/usr/bin/env bash
set -euo pipefail
export CASE_NAME="pglib_opf_case57_ieee"
export CASE_TAG="case57"
export INCLUDE_VANILLA="${INCLUDE_VANILLA:-1}"
exec "$(dirname "$0")/run_case_suite.sh" "$@"
