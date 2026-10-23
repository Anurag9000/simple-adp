#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CASE_NAME="${CASE_NAME:-}"
if [ -z "$CASE_NAME" ]; then
    echo "CASE_NAME is required" >&2
    exit 1
fi
exec ./utils/opf_suite_launcher.sh case "$@"
