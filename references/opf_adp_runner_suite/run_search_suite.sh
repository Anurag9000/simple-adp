#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export TASKS="${TASKS:-vanilla,width_only,width_to_depth}"
export SURFACE="${SURFACE:-all}"

exec ./utils/opf_suite_launcher.sh massive "$@"
