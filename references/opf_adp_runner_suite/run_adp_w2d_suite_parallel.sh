#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TASKS="${TASKS:-width_only,width_to_depth}"
export SURFACE="${SURFACE:-active}"

exec ./utils/opf_suite_launcher.sh massive "$@"
