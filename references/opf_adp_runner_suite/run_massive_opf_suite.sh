#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export INCLUDE_VANILLA="${INCLUDE_VANILLA:-1}"
exec ./utils/opf_suite_launcher.sh massive "$@"
