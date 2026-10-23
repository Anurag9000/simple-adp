"""CLI for the optional OpenML benchmark driver."""

from __future__ import annotations

import argparse
import json

from .benchmark import OPENML_SUITES, OpenMLBenchmarkRunner, load_suite_tasks
from .config import SearchSettings


def _csv_values(value: str, cast):
    try:
        return tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation-screened benchmark for ADP and vanilla tabular MLPs")
    parser.add_argument("--suites", default="cc18,tabzilla,ctr23", help="comma-separated suite keys: cc18,tabzilla,ctr23 (default: all three pools)")
    parser.add_argument("--output", default="benchmark-results/hard-tabular", help="output directory for manifests, splits, screening, and scores")
    parser.add_argument("--profiles", default="faithful,corrected", help="generic ADP profiles to run")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated full-run model seeds")
    parser.add_argument("--seed", type=int, default=0, help="base seed for validation split and screening")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task-limit", type=int, help="limit tasks per suite for a pilot run")
    parser.add_argument("--fold-limit", type=int, help="limit folds per task for a pilot run")
    parser.add_argument("--screen-only", action="store_true", help="run/cache the shallow-MLP screen without launching full comparisons")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--screen-seed-count", type=int, default=2)
    parser.add_argument("--classification-ceiling", type=float, default=1.0)
    parser.add_argument("--regression-ceiling-r2", type=float, default=0.999)
    parser.add_argument("--min-complexity-gain", type=float, default=0.02,
                        help="minimum validation balanced-accuracy/R-squared gain from adding depth")
    parser.add_argument("--min-hard-fold-fraction", type=float, default=0.8,
                        help="fraction of a dataset's folds that must show a depth gain to include the dataset")
    parser.add_argument("--list-suites", action="store_true", help="fetch and print suite task counts, then exit")
    args = parser.parse_args(argv)

    suites = _csv_values(args.suites, str)
    unknown = sorted(set(suites) - set(OPENML_SUITES))
    if unknown:
        parser.error(f"unknown suite(s): {', '.join(unknown)}")
    if args.list_suites:
        print(json.dumps({key: {"openml_id": OPENML_SUITES[key].openml_id,
                                "description": OPENML_SUITES[key].description,
                                "task_count": len(load_suite_tasks(key))}
                          for key in suites}, indent=2))
        return 0
    profiles = _csv_values(args.profiles, str)
    seeds = _csv_values(args.seeds, int)
    settings = SearchSettings(max_epochs=args.max_epochs, batch_size=args.batch_size, seed=args.seed, device=args.device)
    runner = OpenMLBenchmarkRunner(
        args.output,
        settings=settings,
        suites=suites,
        profiles=profiles,
        seeds=seeds,
        validation_fraction=args.validation_fraction,
        screen_seed_count=args.screen_seed_count,
        classification_ceiling=args.classification_ceiling,
        regression_ceiling_r2=args.regression_ceiling_r2,
        min_complexity_gain=args.min_complexity_gain,
        min_hard_fold_fraction=args.min_hard_fold_fraction,
        task_limit=args.task_limit,
        fold_limit=args.fold_limit,
        screen_only=args.screen_only,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
