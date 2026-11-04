"""Export an evidence-honest one-to-one inventory from a full ADP manifest.

Usage: python scripts/audit_bayesian_pairs.py RUN_ROOT/manifest.json --output OUT.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simple_adp.runner.pair_audit import audit_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--permit-slice", action="store_true",
                        help="Audit only selected rows, never claim full-plan closure")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = audit_manifest(manifest, require_full=not args.permit_slice,
                            source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
