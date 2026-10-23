"""Run an isolated one-epoch smoke test for every materialized suite job.

This deliberately reuses the child commands written by the suite manifest. It
does not alter the real run tree; each child receives a private temporary
results directory and a tiny sample window so the test covers model loading,
forward/backward, optimizer state, and checkpoint persistence quickly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Results/cases/case14/run_root/job_manifest.json"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip expensive plots during the isolated audit; training, validation, telemetry, and checkpoints still run.",
    )
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--val-samples", type=int, default=4)
    parser.add_argument("--test-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--unique-model-variants",
        action="store_true",
        help="Select one manifest job for each family/head variant instead of every architecture configuration.",
    )
    parser.add_argument(
        "--expansion-max-width",
        type=int,
        default=None,
        help="Override adaptive max_width for a bounded expansion-path audit.",
    )
    parser.add_argument(
        "--expansion-max-depth",
        type=int,
        default=None,
        help="Override adaptive max_depth for a bounded expansion-path audit.",
    )
    parser.add_argument(
        "--allow-adp-expansion",
        action="store_true",
        help="Keep the suite max width/depth for ADP jobs; default smoke tests stop at each initial architecture.",
    )
    return parser.parse_args()


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError:
        command.extend([option, value])
        return
    if index + 1 >= len(command):
        command.append(value)
    else:
        command[index + 1] = value


def _child_command(job: dict[str, Any], result_dir: Path, args: argparse.Namespace) -> list[str]:
    command = [str(part) for part in job["command"]]
    _replace_option(command, "--results-dir", str(result_dir))
    _replace_option(command, "--max-epochs", str(args.max_epochs))
    if job.get("family") != "vanilla" and (
        args.expansion_max_width is not None
        or args.expansion_max_depth is not None
        or not args.allow_adp_expansion
    ):
        # Keep every requested initial architecture legal. Stagewise jobs
        # intentionally start at widths 1..512; only suppress later growth.
        smoke_width = max(1, int(args.expansion_max_width or job.get("width", 1)))
        smoke_depth = max(1, int(args.expansion_max_depth or job.get("depth", 1)))
        for index, value in enumerate(command):
            if value.startswith("max_width="):
                command[index] = f"max_width={smoke_width}"
            elif value.startswith("max_depth="):
                command[index] = f"max_depth={smoke_depth}"
    command.extend(
        [
            f"train_samples={int(args.train_samples)}",
            f"val_samples={int(args.val_samples)}",
            f"test_samples={int(args.test_samples)}",
            f"batch_size={int(args.batch_size)}",
            "diagnostic_samples=1",
        ]
    )
    return command


def _finite_metric_rows(output_dir: Path) -> tuple[int, bool]:
    rows = 0
    finite = True
    for path in output_dir.rglob("*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows += 1
                    for key, value in row.items():
                        if value is None or value == "":
                            continue
                        # Adaptive telemetry initializes best_val to +inf
                        # before the first validation pass. That is a control
                        # sentinel, not a non-finite training metric.
                        if "best_val" in key.lower():
                            continue
                        try:
                            number = float(value)
                        except ValueError:
                            continue
                        if number != number or number in (float("inf"), float("-inf")):
                            finite = False
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows, finite


def _optimizer_step(checkpoint: Path) -> int:
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("opt_state", {}) if isinstance(payload, dict) else {}
        steps: list[int] = []
        for item in state.get("state", {}).values() if isinstance(state, dict) else []:
            if not isinstance(item, dict):
                continue
            value = item.get("step")
            try:
                steps.append(int(value.item() if hasattr(value, "item") else value))
            except (TypeError, ValueError):
                pass
        return max(steps, default=0)
    except Exception:
        return 0


def _verify_artifacts(output_dir: Path, family: str) -> dict[str, Any]:
    checkpoints = list(output_dir.rglob("checkpoint_last.pt"))
    checkpoints.extend(output_dir.rglob("checkpoint_Native_Training.pt"))
    checkpoints.extend(output_dir.rglob("best_model.pt"))
    checkpoints = [path for path in checkpoints if path.is_file() and path.stat().st_size > 0]
    csv_rows, finite_metrics = _finite_metric_rows(output_dir)
    step = max((_optimizer_step(path) for path in checkpoints), default=0)
    progress_files = [path for path in output_dir.rglob("progress_state.json") if path.is_file()]
    return {
        "checkpoint_count": len(checkpoints),
        "csv_rows": csv_rows,
        "finite_metrics": finite_metrics,
        "optimizer_step": step,
        "progress_state_count": len(progress_files),
        "training_verified": bool(checkpoints and csv_rows > 0 and finite_metrics and step > 0),
        "expected_family": family,
    }


def main() -> int:
    args = _parse_args()
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if isinstance(manifest_payload, list):
        # The runner's dry-run output is a bare job list; the persisted job
        # manifest is a metadata object containing the same list.
        manifest = {}
        all_jobs = list(manifest_payload)
    elif isinstance(manifest_payload, dict):
        manifest = manifest_payload
        all_jobs = list(manifest.get("jobs", []))
    else:
        raise SystemExit(f"Unsupported manifest JSON shape: {type(manifest_payload).__name__}")
    jobs = all_jobs
    if args.unique_model_variants:
        seen_variants: set[tuple[str, str]] = set()
        jobs = []
        for job in all_jobs:
            variant = (str(job.get("family", "")), str(job.get("head_variant", "")))
            if variant in seen_variants:
                continue
            seen_variants.add(variant)
            jobs.append(job)
    start = max(0, int(args.start))
    end = len(jobs) if int(args.limit) <= 0 else min(len(jobs), start + int(args.limit))
    if start >= end:
        raise SystemExit(f"No jobs selected: start={start} end={end} total={len(jobs)}")

    if args.output_root is None:
        output_root = Path(tempfile.mkdtemp(prefix="opf_adp_suite_verify_"))
    else:
        output_root = args.output_root
        output_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": 1,
        "manifest": str(args.manifest),
        "manifest_job_count": len(all_jobs),
        "selected_job_count": end - start,
        "selected_start": start,
        "selected_end": end,
        "output_root": str(output_root),
        "jobs": [],
    }
    for index, job in enumerate(jobs[start:end], start=start):
        job_dir = output_root / f"job_{index:04d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        command = _child_command(job, job_dir / "results", args)
        env = os.environ.copy()
        seed = str(job.get("seed", manifest.get("seed", 42)))
        env.update(
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OPF_ADP_GLOBAL_SEED": seed,
                "PYTHONHASHSEED": seed,
            }
        )
        if args.skip_diagnostics:
            env["OPF_ADP_SKIP_DIAGNOSTICS"] = "1"
        started = time.monotonic()
        timed_out = False
        stdout_path = job_dir / "stdout.log"
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    stdout=stdout_handle,
                    stderr=subprocess.STDOUT,
                    timeout=max(1.0, float(args.timeout_sec)),
                    check=False,
                )
            return_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
        elapsed = time.monotonic() - started
        artifact_report = _verify_artifacts(job_dir / "results", str(job.get("family", "")))
        row = {
            "index": index,
            "job_id": job.get("job_id"),
            "family": job.get("family"),
            "head_variant": job.get("head_variant"),
            "depth": job.get("depth"),
            "width": job.get("width"),
            "seed": job.get("seed"),
            "return_code": return_code,
            "timed_out": timed_out,
            "elapsed_sec": round(elapsed, 3),
            **artifact_report,
        }
        report["jobs"].append(row)
        (output_root / "verification_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[{index + 1}/{len(jobs)}] rc={return_code} "
            f"training={artifact_report['training_verified']} "
            f"family={job.get('family')} head={job.get('head_variant')} "
            f"d={job.get('depth')} w={job.get('width')} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    selected = report["jobs"]
    report["passed"] = sum(bool(row["return_code"] == 0 and row["training_verified"]) for row in selected)
    report["failed"] = len(selected) - int(report["passed"])
    (output_root / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ("manifest_job_count", "selected_start", "selected_end", "passed", "failed", "output_root")}, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
