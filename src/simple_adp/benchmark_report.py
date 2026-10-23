"""Aggregate benchmark JSONL records into per-task and macro summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


LOWER_IS_BETTER = {"log_loss", "rmse", "mae", "validation_loss"}
STANDARD_METRICS = {
    "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "rmse", "mae", "r2",
    "precision", "recall", "f1", "pr_auc", "roc_auc", "ndcg@5", "ndcg@10",
    "recall@5", "recall@10", "event_precision", "event_recall", "event_f1",
}


def summarize(results_dir: str | Path) -> dict[str, Path]:
    root = Path(results_dir)
    source = root / "results.jsonl"
    selected_signature = None
    active_path = root / "active-run.json"
    config_files = sorted(root.glob("run-config-*.json"), key=lambda path: path.stat().st_mtime)
    if active_path.exists():
        selected_signature = json.loads(active_path.read_text(encoding="utf-8")).get("run_signature")
    elif config_files:
        selected_signature = json.loads(config_files[-1].read_text(encoding="utf-8")).get("run_signature")
    screening_path = root / "dataset-decisions.jsonl"
    screening_records = []
    if screening_path.exists():
        latest_by_task = {}
        with screening_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if selected_signature is not None and record.get("run_signature") != selected_signature:
                    continue
                latest_by_task[(record.get("suite"), record.get("task_id"))] = record
        screening_records = list(latest_by_task.values())
    screen_summary_path = root / "dataset-screening.csv"
    if screening_records:
        pd.DataFrame.from_records(screening_records).to_csv(screen_summary_path, index=False)

    records = []
    if source.is_file():
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {source}:{line_number}") from exc
                if record.get("status") == "completed":
                    if selected_signature is not None and record.get("run_signature") != selected_signature:
                        continue
                    flat = {key: value for key, value in record.items() if key != "test_metrics"}
                    task_metrics = record.get("test_metrics", {})
                    declared = record.get("metric_directions") or {}
                    # Current adapters explicitly declare every metric. Keep
                    # the old report format readable for legacy records that
                    # predate metric declarations, but never surface an
                    # undeclared metric from a task-aware result.
                    allowed = set(declared) if declared else STANDARD_METRICS
                    flat.update({name: value for name, value in task_metrics.items() if name in allowed})
                    flat["task_key"] = f"{record.get('provider', 'openml')}:{record['suite']}:{record['task_id']}"
                    records.append(flat)
    paths = {"dataset_screening": screen_summary_path} if screening_records else {}
    metrics: list[str] = []
    frame = pd.DataFrame.from_records(records)
    if records:
        declared_metrics = {
            metric
            for record in records
            for metric in (record.get("metric_directions") or {})
        }
        metrics = sorted((STANDARD_METRICS | declared_metrics).intersection(frame.columns))
        task_keys = [
            key for key in (
                "provider", "suite", "task_id", "dataset_id", "dataset_name",
                "task_family", "task_variant", "task_key", "model", "model_id", "generic_profile",
            ) if key in frame.columns
        ]
        per_task = frame.groupby(task_keys, dropna=False, as_index=False)[metrics].mean(numeric_only=True)
        counts_by_task = frame.groupby(task_keys, dropna=False).size().rename("n_runs").reset_index()
        per_task = per_task.merge(counts_by_task, on=task_keys, how="left", validate="one_to_one")
        directions_by_task: dict[tuple[Any, ...], dict[str, str]] = {}
        direction_keys = [key for key in ("provider", "suite", "task_id", "task_family", "task_variant") if key in frame.columns]
        for _, record in frame.iterrows():
            key = tuple(record[name] for name in direction_keys)
            directions_by_task.setdefault(key, {}).update(record.get("metric_directions") or {})
        for metric in metrics:
            # Metric direction can vary by adapter; rank only after applying the
            # direction declared by that task's adapter.
            for _, indices in per_task.groupby([key for key in ("provider", "suite", "task_id", "task_family", "task_variant") if key in per_task.columns], dropna=False).groups.items():
                index_list = list(indices)
                direction = directions_by_task.get(tuple(per_task.loc[index_list[0], name] for name in direction_keys), {}).get(metric)
                ascending = direction == "min" if direction in {"min", "max"} else metric in LOWER_IS_BETTER
                per_task.loc[index_list, f"rank_{metric}"] = per_task.loc[index_list].groupby(["suite", "task_key"])[metric].rank(ascending=ascending, method="average")
        summary_rows: list[dict[str, Any]] = []
        summary_keys = [key for key in ("provider", "suite", "task_family", "task_variant", "model", "model_id", "generic_profile") if key in per_task.columns]
        for group_key, group in per_task.groupby(
            summary_keys, dropna=False
        ):
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            row: dict[str, Any] = {
                **dict(zip(summary_keys, values)),
                "n_datasets": int(group["task_key"].nunique()),
            }
            for metric in metrics:
                row[f"macro_mean_{metric}"] = float(group[metric].mean()) if group[metric].notna().any() else None
                row[f"between_dataset_std_{metric}"] = float(group[metric].std(ddof=1)) if group[metric].count() > 1 else None
                ranks = group[f"rank_{metric}"]
                row[f"mean_dataset_rank_{metric}"] = float(ranks.mean()) if ranks.notna().any() else None
            summary_rows.append(row)
        task_path = root / "dataset-summary.csv"
        model_path = root / "model-summary.csv"
        per_task.to_csv(task_path, index=False)
        pd.DataFrame(summary_rows).sort_values(["suite", "model"], kind="stable").to_csv(model_path, index=False)
        paths.update({"dataset_summary": task_path, "model_summary": model_path})

    counts: dict[str, int] = {"completed_runs": len(frame), "screened_datasets": len(screening_records)}
    for filename in ("screening.jsonl", "dataset-decisions.jsonl", "results.jsonl"):
        path = root / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if selected_signature is not None and record.get("run_signature") not in {None, selected_signature}:
                    continue
                status = record.get("decision", record.get("status", "unknown"))
                key = f"{filename.removesuffix('.jsonl')}:{status}"
                counts[key] = counts.get(key, 0) + 1
    summary_path = root / "benchmark-summary.json"
    summary_path.write_text(json.dumps({"run_signature": selected_signature, "counts": counts, "metrics": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["summary"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize simple-adp benchmark results")
    parser.add_argument("results_dir", help="benchmark output directory")
    args = parser.parse_args(argv)
    outputs = summarize(args.results_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
