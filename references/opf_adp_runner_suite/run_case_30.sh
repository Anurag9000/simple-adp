#!/usr/bin/env bash
set -euo pipefail
export CASE_NAME="pglib_opf_case30_ieee"
export CASE_TAG="case30"
export INCLUDE_VANILLA="${INCLUDE_VANILLA:-1}"
exec "$(dirname "$0")/run_case_suite.sh" "$@"
