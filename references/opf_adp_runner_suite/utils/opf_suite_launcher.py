from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VANILLA_ROOT = ROOT / "DNN" / "VANILLA"
if str(VANILLA_ROOT) not in sys.path:
    sys.path.insert(0, str(VANILLA_ROOT))

from utils.logging_utils import configure_root_logging
from utils.opf_dataset_cache import _default_data_root, prepare_case_cache
from utils.opf_resume_bundle import (
    bundle_path_for_results_dir,
    export_results_bundle,
    rehydrate_sharded_telemetry,
    restore_results_bundle,
)
from utils.opf_shared_defaults import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEPTH_ONLY_WIDTHS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_NEURONS,
    DEFAULT_MAX_WIDTH,
    DEFAULT_VANILLA_DEPTHS,
    DEFAULT_VANILLA_WIDTHS,
    DEFAULT_WIDTH_ONLY_DEPTHS,
    DEFAULT_WIDTH_TO_DEPTH_WIDTHS,
    parse_epoch_limit,
)
from utils.runtime_tuning import bootstrap_runtime
from DNN.VANILLA.Dyn_DNN4OPF.utils.run_defaults import RUNNER_DEFAULTS


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    if lowered in {"0", "false", "no", "off", "n", ""}:
        return False
    return default


def _arg_or_env(value: Any, env_name: str, default: Any = None) -> Any:
    """Prefer an explicitly supplied value, including explicit zero."""
    if value is not None:
        return value
    return os.environ.get(env_name, default)


def _epoch_limit(value: Any, env_name: str, default: Any) -> int:
    """Resolve CLI/environment epoch limits; ``inf`` becomes the 0 sentinel."""
    return parse_epoch_limit(_arg_or_env(value, env_name, default))


def _case_tag(case_name: str) -> str:
    tag = case_name.replace("pglib_opf_case", "")
    tag = tag.split("_", 1)[0]
    return f"case{tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central OPF_ADP experiment launcher.")
    parser.add_argument("--mode", choices=["case", "massive"], required=True)
    parser.add_argument("--case-name", default=None)
    parser.add_argument("--case-tag", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--source-run-root", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--concurrency-file", default=None)
    parser.add_argument("--job-start-index", type=int, default=None)
    parser.add_argument("--job-limit", type=int, default=None)
    parser.add_argument("--max-active-jobs", type=int, default=None)
    parser.add_argument("--max-active-gpu-jobs", type=int, default=None)
    parser.add_argument("--repeat-count", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--launch-host-reserve-mib", type=float, default=None)
    parser.add_argument("--launch-gpu-reserve-mib", type=float, default=None)
    parser.add_argument("--gpu-child-gate-close-buffer-mib", type=float, default=None)
    parser.add_argument("--host-child-gate-close-buffer-mib", type=float, default=None)
    parser.add_argument("--external-memory-drop-reopen-mib", type=float, default=None)
    parser.add_argument("--gpu-device-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--surface", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--param-band", default=None)
    parser.add_argument("--legacy-architecture-grid", action="store_true")
    parser.add_argument("--include-small-stl", action="store_true")
    parser.add_argument("--include-vanilla", dest="include_vanilla", action="store_true")
    parser.add_argument("--no-include-vanilla", dest="include_vanilla", action="store_false")
    parser.set_defaults(include_vanilla=None)
    parser.add_argument("--width-only-depths", default=None)
    parser.add_argument("--width-to-depth-width", type=int, default=None)
    parser.add_argument("--width-to-depth-depth", type=int, default=None)
    parser.add_argument("--width-to-depth-widths", default=None)
    parser.add_argument("--depth-only-widths", default=None)
    parser.add_argument("--vanilla-depths", default=None)
    parser.add_argument("--vanilla-widths", default=None)
    parser.add_argument("--vanilla-model", default=None)
    parser.add_argument("--max-epochs-vanilla", type=parse_epoch_limit, default=None)
    parser.add_argument("--max-epochs-adp", type=parse_epoch_limit, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-neurons", type=int, default=None)
    parser.add_argument("--head-variants", default=None)
    parser.add_argument("--missing-present-task-repeats", type=int, default=None)
    parser.add_argument("--prediction-repeats", type=int, default=None)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _normalize_runner_args(runner_args: List[str]) -> List[str]:
    args = list(runner_args)
    if args and args[0] == "--":
        args = args[1:]
    return args


def _default_results_dir(mode: str, case_tag: Optional[str]) -> str:
    if mode == "case":
        return f"Results/cases/{case_tag or 'case'}"
    return "Results/massive_opf_suite"


def _runner_base_args(mode: str, args: argparse.Namespace, results_dir: str, run_root: str, source_run_root: str) -> List[str]:
    include_vanilla = args.include_vanilla
    if include_vanilla is None:
        include_vanilla = _truthy(os.environ.get("INCLUDE_VANILLA"), default=True)
    base = [
        "--mode",
        mode,
        "--results-dir",
        results_dir,
        "--run-root",
        run_root,
        "--source-run-root",
        source_run_root,
        "--scheduler",
        "gpu_first",
        "--concurrency",
        str(args.concurrency if args.concurrency is not None else os.environ.get("CONCURRENCY", RUNNER_DEFAULTS["concurrency"])),
        "--host-ram-stop-avail-mib",
        str(os.environ.get("HOST_RAM_STOP_AVAIL_MIB", RUNNER_DEFAULTS["host_ram_stop_avail_mib"])),
        "--host-ram-resume-avail-mib",
        str(os.environ.get("HOST_RAM_RESUME_AVAIL_MIB", RUNNER_DEFAULTS["host_ram_resume_avail_mib"])),
        "--gpu-stop-avail-mib",
        str(os.environ.get("GPU_STOP_AVAIL_MIB", RUNNER_DEFAULTS["gpu_stop_avail_mib"])),
        "--gpu-resume-avail-mib",
        str(os.environ.get("GPU_RESUME_AVAIL_MIB", RUNNER_DEFAULTS["gpu_resume_avail_mib"])),
        "--swap-pressure-limit-pct",
        str(os.environ.get("SWAP_PRESSURE_LIMIT_PCT", RUNNER_DEFAULTS["swap_pressure_limit_pct"])),
        "--swap-resume-pct",
        str(os.environ.get("SWAP_RESUME_PCT", RUNNER_DEFAULTS["swap_resume_pct"])),
        "--pressure-poll-interval-sec",
        str(os.environ.get("PRESSURE_POLL_INTERVAL_SEC", RUNNER_DEFAULTS["pressure_poll_interval_sec"])),
        "--post-launch-sample-delay-sec",
        str(os.environ.get("POST_LAUNCH_SAMPLE_DELAY_SEC", RUNNER_DEFAULTS["post_launch_sample_delay_sec"])),
        "--batch-backoff-factor",
        str(os.environ.get("BATCH_BACKOFF_FACTOR", RUNNER_DEFAULTS["batch_backoff_factor"])),
        "--launch-host-reserve-mib",
        str(args.launch_host_reserve_mib if args.launch_host_reserve_mib is not None else os.environ.get("LAUNCH_HOST_RESERVE_MIB", RUNNER_DEFAULTS["launch_host_reserve_mib"])),
        "--launch-gpu-reserve-mib",
        str(args.launch_gpu_reserve_mib if args.launch_gpu_reserve_mib is not None else os.environ.get("LAUNCH_GPU_RESERVE_MIB", RUNNER_DEFAULTS["launch_gpu_reserve_mib"])),
        "--gpu-child-gate-close-buffer-mib",
        str(args.gpu_child_gate_close_buffer_mib if args.gpu_child_gate_close_buffer_mib is not None else os.environ.get("GPU_CHILD_GATE_CLOSE_BUFFER_MIB", RUNNER_DEFAULTS["gpu_child_gate_close_buffer_mib"])),
        "--host-child-gate-close-buffer-mib",
        str(args.host_child_gate_close_buffer_mib if args.host_child_gate_close_buffer_mib is not None else os.environ.get("HOST_CHILD_GATE_CLOSE_BUFFER_MIB", RUNNER_DEFAULTS["host_child_gate_close_buffer_mib"])),
        "--external-memory-drop-reopen-mib",
        str(args.external_memory_drop_reopen_mib if args.external_memory_drop_reopen_mib is not None else os.environ.get("EXTERNAL_MEMORY_DROP_REOPEN_MIB", RUNNER_DEFAULTS["external_memory_drop_reopen_mib"])),
    ]
    if args.concurrency_file or os.environ.get("CONCURRENCY_FILE"):
        base.extend(["--concurrency-file", args.concurrency_file or os.environ["CONCURRENCY_FILE"]])
    if args.job_start_index is not None or os.environ.get("JOB_START_INDEX"):
        base.extend(["--job-start-index", str(args.job_start_index if args.job_start_index is not None else os.environ["JOB_START_INDEX"])])
    if args.job_limit is not None or os.environ.get("JOB_LIMIT"):
        base.extend(["--job-limit", str(args.job_limit if args.job_limit is not None else os.environ["JOB_LIMIT"])])
    if args.max_active_jobs is not None or os.environ.get("MAX_ACTIVE_JOBS"):
        base.extend(["--max-active-jobs", str(args.max_active_jobs if args.max_active_jobs is not None else os.environ["MAX_ACTIVE_JOBS"])])
    if args.max_active_gpu_jobs is not None or os.environ.get("MAX_ACTIVE_GPU_JOBS"):
        base.extend(["--max-active-gpu-jobs", str(args.max_active_gpu_jobs if args.max_active_gpu_jobs is not None else os.environ["MAX_ACTIVE_GPU_JOBS"])])
    if args.gpu_device_index is not None or os.environ.get("GPU_DEVICE_INDEX"):
        base.extend(["--gpu-device-index", str(args.gpu_device_index if args.gpu_device_index is not None else os.environ["GPU_DEVICE_INDEX"])])
    if args.repeat_count is not None or os.environ.get("REPEAT_COUNT"):
        base.extend(["--repeat-count", str(args.repeat_count if args.repeat_count is not None else os.environ.get("REPEAT_COUNT", RUNNER_DEFAULTS["repeat_count"]))])
    if args.max_retries is not None or os.environ.get("MAX_RETRIES"):
        base.extend(["--max-retries", str(args.max_retries if args.max_retries is not None else os.environ.get("MAX_RETRIES", RUNNER_DEFAULTS["max_retries"]))])
    if args.seed is not None or os.environ.get("OPF_ADP_GLOBAL_SEED"):
        base.extend(["--seed", str(args.seed if args.seed is not None else os.environ.get("OPF_ADP_GLOBAL_SEED", RUNNER_DEFAULTS["seed"]))])
    base.extend([
        "--width-to-depth-width",
        str(args.width_to_depth_width if args.width_to_depth_width is not None else os.environ.get("WIDTH_TO_DEPTH_WIDTH", 1)),
        "--width-to-depth-depth",
        str(args.width_to_depth_depth if args.width_to_depth_depth is not None else os.environ.get("WIDTH_TO_DEPTH_DEPTH", 1)),
    ])
    if args.missing_present_task_repeats is not None or os.environ.get("MISSING_PRESENT_TASK_REPEATS"):
        base.extend(["--missing-present-task-repeats", str(_arg_or_env(args.missing_present_task_repeats, "MISSING_PRESENT_TASK_REPEATS", RUNNER_DEFAULTS["missing_present_task_repeats"]))])
    else:
        base.extend(["--missing-present-task-repeats", str(RUNNER_DEFAULTS["missing_present_task_repeats"])])
    if args.prediction_repeats is not None or os.environ.get("PREDICTION_REPEATS"):
        base.extend(["--prediction-repeats", str(_arg_or_env(args.prediction_repeats, "PREDICTION_REPEATS", RUNNER_DEFAULTS["prediction_repeats"]))])
    else:
        base.extend(["--prediction-repeats", str(RUNNER_DEFAULTS["prediction_repeats"])])
    if args.max_width is not None or os.environ.get("MAX_WIDTH"):
        base.extend(["--max-width", str(args.max_width if args.max_width is not None else os.environ.get("MAX_WIDTH", DEFAULT_MAX_WIDTH))])
    if args.max_depth is not None or os.environ.get("MAX_DEPTH"):
        base.extend(["--max-depth", str(args.max_depth if args.max_depth is not None else os.environ.get("MAX_DEPTH", DEFAULT_MAX_DEPTH))])
    base.extend(["--max-neurons", str(args.max_neurons if args.max_neurons is not None else os.environ.get("MAX_NEURONS", DEFAULT_MAX_NEURONS))])
    if args.head_variants or os.environ.get("HEAD_VARIANTS"):
        base.extend(["--head-variants", args.head_variants or os.environ["HEAD_VARIANTS"]])
    if args.cpu_only or _truthy(os.environ.get("CPU_ONLY")):
        base.append("--cpu-only")
    if mode == "case":
        base.extend(
            [
                "--cases",
                args.case_name,
                "--surface",
                "all",
                "--tasks",
                "width_only,width_to_depth,depth_to_width,depth_only",
                "--width-only-depths",
                args.width_only_depths or os.environ.get("WIDTH_ONLY_DEPTHS", ",".join(str(item) for item in DEFAULT_WIDTH_ONLY_DEPTHS)),
                "--depth-only-widths",
                args.depth_only_widths or os.environ.get("DEPTH_ONLY_WIDTHS", ",".join(str(item) for item in DEFAULT_DEPTH_ONLY_WIDTHS)),
            ]
        )
        if include_vanilla:
            base.append("--include-vanilla")
        else:
            base.append("--no-include-vanilla")
        if args.vanilla_depths or os.environ.get("VANILLA_DEPTHS"):
            base.extend(["--vanilla-depths", args.vanilla_depths or os.environ["VANILLA_DEPTHS"]])
        if args.vanilla_widths or os.environ.get("VANILLA_WIDTHS"):
            base.extend(["--vanilla-widths", args.vanilla_widths or os.environ["VANILLA_WIDTHS"]])
        if args.vanilla_model or os.environ.get("VANILLA_MODEL"):
            base.extend(["--vanilla-model", args.vanilla_model or os.environ["VANILLA_MODEL"]])
        if args.max_epochs_vanilla is not None or os.environ.get("MAX_EPOCHS_VANILLA"):
            base.extend(["--max-epochs-vanilla", str(_epoch_limit(args.max_epochs_vanilla, "MAX_EPOCHS_VANILLA", RUNNER_DEFAULTS["max_epochs_vanilla"]))])
        base.extend(["--max-epochs-adp", str(_epoch_limit(args.max_epochs_adp, "MAX_EPOCHS_ADP", RUNNER_DEFAULTS["max_epochs_adp"]))])
        return base

    surface = args.surface or os.environ.get("SURFACE", "all")
    tasks = args.tasks or os.environ.get("TASKS", "all")
    base.extend(
        [
            "--surface",
            surface,
            "--tasks",
            tasks,
        ]
    )
    if args.param_band or os.environ.get("PARAM_BAND"):
        base.extend(["--param-band", args.param_band or os.environ["PARAM_BAND"]])
    if args.legacy_architecture_grid or _truthy(os.environ.get("LEGACY_ARCHITECTURE_GRID")):
        base.append("--legacy-architecture-grid")
    if args.include_small_stl or _truthy(os.environ.get("INCLUDE_SMALL_STL")):
        base.append("--include-small-stl")
    if include_vanilla:
        base.append("--include-vanilla")
    else:
        base.append("--no-include-vanilla")
    if args.vanilla_depths or os.environ.get("VANILLA_DEPTHS"):
        base.extend(["--vanilla-depths", args.vanilla_depths or os.environ["VANILLA_DEPTHS"]])
    if args.vanilla_widths or os.environ.get("VANILLA_WIDTHS"):
        base.extend(["--vanilla-widths", args.vanilla_widths or os.environ["VANILLA_WIDTHS"]])
    if args.vanilla_model or os.environ.get("VANILLA_MODEL"):
        base.extend(["--vanilla-model", args.vanilla_model or os.environ["VANILLA_MODEL"]])
    if args.max_epochs_vanilla is not None or os.environ.get("MAX_EPOCHS_VANILLA"):
        base.extend(["--max-epochs-vanilla", str(_epoch_limit(args.max_epochs_vanilla, "MAX_EPOCHS_VANILLA", RUNNER_DEFAULTS["max_epochs_vanilla"]))])
    base.extend(["--max-epochs-adp", str(_epoch_limit(args.max_epochs_adp, "MAX_EPOCHS_ADP", RUNNER_DEFAULTS["max_epochs_adp"]))])
    if args.width_only_depths or os.environ.get("WIDTH_ONLY_DEPTHS"):
        base.extend(["--width-only-depths", args.width_only_depths or os.environ["WIDTH_ONLY_DEPTHS"]])
    if args.width_to_depth_widths or os.environ.get("WIDTH_TO_DEPTH_WIDTHS"):
        base.extend(["--width-to-depth-widths", args.width_to_depth_widths or os.environ["WIDTH_TO_DEPTH_WIDTHS"]])
    if args.depth_only_widths or os.environ.get("DEPTH_ONLY_WIDTHS"):
        base.extend(["--depth-only-widths", args.depth_only_widths or os.environ["DEPTH_ONLY_WIDTHS"]])
    return base


def main() -> int:
    configure_root_logging()
    bootstrap_runtime("opf_suite_launcher")
    args = parse_args()
    runner_args = _normalize_runner_args(args.runner_args)
    include_vanilla = args.include_vanilla if args.include_vanilla is not None else _truthy(
        os.environ.get("INCLUDE_VANILLA"), default=True
    )
    args.include_vanilla = include_vanilla

    case_name = args.case_name or os.environ.get("CASE_NAME")
    case_tag = args.case_tag or os.environ.get("CASE_TAG")
    if args.mode == "case":
        if not case_name:
            raise SystemExit("--case-name (or CASE_NAME) is required in case mode")
        if not case_tag:
            case_tag = _case_tag(case_name)
    else:
        if case_name and not case_tag:
            case_tag = _case_tag(case_name)

    results_dir = args.results_dir or os.environ.get("RESULTS_DIR") or _default_results_dir(args.mode, case_tag)
    run_root = args.run_root or os.environ.get("RUN_ROOT") or f"{results_dir}/run_root"
    source_run_root = args.source_run_root or os.environ.get("SOURCE_RUN_ROOT") or f"{results_dir}/source_run_root"
    data_root = Path(args.data_root or os.environ.get("OPF_ADP_DATA_ROOT") or _default_data_root())
    chunk_size = int(args.chunk_size or os.environ.get("OPF_ADP_DATA_CHUNK_SIZE") or DEFAULT_CHUNK_SIZE)
    fresh = bool(args.fresh or _truthy(os.environ.get("OPF_ADP_DATA_FRESH")))
    auto_resume_bundle = _truthy(os.environ.get("OPF_ADP_AUTO_RESUME_BUNDLE"), default=True)

    run_root_path = Path(run_root)
    run_root_path.mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    results_path = Path(results_dir)
    if auto_resume_bundle:
        restore_results_bundle(results_path, bundle_path_for_results_dir(results_path))
    rehydrated_parts = rehydrate_sharded_telemetry(results_path)
    if rehydrated_parts:
        print(f"[OPF_ADP] Rehydrated {rehydrated_parts} sharded telemetry file(s) before launch")

    launch_manifest: Dict[str, Any] = {
        "mode": args.mode,
        "case_name": case_name,
        "case_tag": case_tag,
        "results_dir": results_dir,
        "run_root": run_root,
        "source_run_root": source_run_root,
        "data_root": str(data_root),
        "chunk_size": chunk_size,
        "fresh": fresh,
        "include_vanilla": bool(include_vanilla),
        "vanilla_depths": args.vanilla_depths or os.environ.get("VANILLA_DEPTHS"),
        "vanilla_widths": args.vanilla_widths or os.environ.get("VANILLA_WIDTHS"),
        "vanilla_model": args.vanilla_model or os.environ.get("VANILLA_MODEL"),
        "max_epochs_vanilla": _epoch_limit(args.max_epochs_vanilla, "MAX_EPOCHS_VANILLA", RUNNER_DEFAULTS["max_epochs_vanilla"]),
        "max_epochs_adp": _epoch_limit(args.max_epochs_adp, "MAX_EPOCHS_ADP", RUNNER_DEFAULTS["max_epochs_adp"]),
        "max_width": int(args.max_width if args.max_width is not None else os.environ.get("MAX_WIDTH", DEFAULT_MAX_WIDTH)),
        "max_depth": int(args.max_depth if args.max_depth is not None else os.environ.get("MAX_DEPTH", DEFAULT_MAX_DEPTH)),
        "max_neurons": int(args.max_neurons if args.max_neurons is not None else os.environ.get("MAX_NEURONS", DEFAULT_MAX_NEURONS)),
        "external_memory_drop_reopen_mib": float(args.external_memory_drop_reopen_mib if args.external_memory_drop_reopen_mib is not None else os.environ.get("EXTERNAL_MEMORY_DROP_REOPEN_MIB", RUNNER_DEFAULTS["external_memory_drop_reopen_mib"])),
        "runner_args": runner_args,
    }
    (run_root_path / "suite_launch_manifest.json").write_text(json.dumps(launch_manifest, indent=2, sort_keys=True), encoding="utf-8")

    env_updates: Dict[str, str] = {
        "RESULTS_DIR": results_dir,
        "RUN_ROOT": run_root,
        "SOURCE_RUN_ROOT": source_run_root,
        "OPF_ADP_DATA_ROOT": str(data_root),
        "OPF_ADP_DATA_CHUNK_SIZE": str(chunk_size),
        "OPF_ADP_ENABLE_FLAT_MATERIALIZATION": os.environ.get(
            "OPF_ADP_ENABLE_FLAT_MATERIALIZATION",
            "1" if case_name == "pglib_opf_case14_ieee" else "0",
        ),
        "HOST_RAM_STOP_AVAIL_MIB": str(os.environ.get("HOST_RAM_STOP_AVAIL_MIB", RUNNER_DEFAULTS["host_ram_stop_avail_mib"])),
        "HOST_RAM_RESUME_AVAIL_MIB": str(os.environ.get("HOST_RAM_RESUME_AVAIL_MIB", RUNNER_DEFAULTS["host_ram_resume_avail_mib"])),
        "GPU_STOP_AVAIL_MIB": str(os.environ.get("GPU_STOP_AVAIL_MIB", RUNNER_DEFAULTS["gpu_stop_avail_mib"])),
        "GPU_RESUME_AVAIL_MIB": str(os.environ.get("GPU_RESUME_AVAIL_MIB", RUNNER_DEFAULTS["gpu_resume_avail_mib"])),
        "SWAP_PRESSURE_LIMIT_PCT": str(os.environ.get("SWAP_PRESSURE_LIMIT_PCT", RUNNER_DEFAULTS["swap_pressure_limit_pct"])),
        "SWAP_RESUME_PCT": str(os.environ.get("SWAP_RESUME_PCT", RUNNER_DEFAULTS["swap_resume_pct"])),
        "PRESSURE_POLL_INTERVAL_SEC": str(os.environ.get("PRESSURE_POLL_INTERVAL_SEC", RUNNER_DEFAULTS["pressure_poll_interval_sec"])),
        "POST_LAUNCH_SAMPLE_DELAY_SEC": str(os.environ.get("POST_LAUNCH_SAMPLE_DELAY_SEC", RUNNER_DEFAULTS["post_launch_sample_delay_sec"])),
        "BATCH_BACKOFF_FACTOR": str(os.environ.get("BATCH_BACKOFF_FACTOR", RUNNER_DEFAULTS["batch_backoff_factor"])),
        "LAUNCH_HOST_RESERVE_MIB": str(args.launch_host_reserve_mib if args.launch_host_reserve_mib is not None else os.environ.get("LAUNCH_HOST_RESERVE_MIB", RUNNER_DEFAULTS["launch_host_reserve_mib"])),
        "LAUNCH_GPU_RESERVE_MIB": str(args.launch_gpu_reserve_mib if args.launch_gpu_reserve_mib is not None else os.environ.get("LAUNCH_GPU_RESERVE_MIB", RUNNER_DEFAULTS["launch_gpu_reserve_mib"])),
        "GPU_CHILD_GATE_CLOSE_BUFFER_MIB": str(args.gpu_child_gate_close_buffer_mib if args.gpu_child_gate_close_buffer_mib is not None else os.environ.get("GPU_CHILD_GATE_CLOSE_BUFFER_MIB", RUNNER_DEFAULTS["gpu_child_gate_close_buffer_mib"])),
        "HOST_CHILD_GATE_CLOSE_BUFFER_MIB": str(args.host_child_gate_close_buffer_mib if args.host_child_gate_close_buffer_mib is not None else os.environ.get("HOST_CHILD_GATE_CLOSE_BUFFER_MIB", RUNNER_DEFAULTS["host_child_gate_close_buffer_mib"])),
        "EXTERNAL_MEMORY_DROP_REOPEN_MIB": str(args.external_memory_drop_reopen_mib if args.external_memory_drop_reopen_mib is not None else os.environ.get("EXTERNAL_MEMORY_DROP_REOPEN_MIB", RUNNER_DEFAULTS["external_memory_drop_reopen_mib"])),
        "MAX_EPOCHS_ADP": str(_epoch_limit(args.max_epochs_adp, "MAX_EPOCHS_ADP", RUNNER_DEFAULTS["max_epochs_adp"])),
    }
    if case_name:
        env_updates["CASE_NAME"] = str(case_name)
    if case_tag:
        env_updates["CASE_TAG"] = str(case_tag)
    if args.concurrency is not None or os.environ.get("CONCURRENCY"):
        env_updates["CONCURRENCY"] = str(args.concurrency if args.concurrency is not None else os.environ["CONCURRENCY"])
    if args.concurrency_file or os.environ.get("CONCURRENCY_FILE"):
        env_updates["CONCURRENCY_FILE"] = str(args.concurrency_file or os.environ["CONCURRENCY_FILE"])
    if args.job_start_index is not None or os.environ.get("JOB_START_INDEX"):
        env_updates["JOB_START_INDEX"] = str(args.job_start_index if args.job_start_index is not None else os.environ["JOB_START_INDEX"])
    if args.job_limit is not None or os.environ.get("JOB_LIMIT"):
        env_updates["JOB_LIMIT"] = str(args.job_limit if args.job_limit is not None else os.environ["JOB_LIMIT"])
    if args.max_active_jobs is not None or os.environ.get("MAX_ACTIVE_JOBS"):
        env_updates["MAX_ACTIVE_JOBS"] = str(args.max_active_jobs if args.max_active_jobs is not None else os.environ["MAX_ACTIVE_JOBS"])
    if args.max_active_gpu_jobs is not None or os.environ.get("MAX_ACTIVE_GPU_JOBS"):
        env_updates["MAX_ACTIVE_GPU_JOBS"] = str(args.max_active_gpu_jobs if args.max_active_gpu_jobs is not None else os.environ["MAX_ACTIVE_GPU_JOBS"])
    if args.gpu_device_index is not None or os.environ.get("GPU_DEVICE_INDEX"):
        env_updates["GPU_DEVICE_INDEX"] = str(args.gpu_device_index if args.gpu_device_index is not None else os.environ["GPU_DEVICE_INDEX"])
    if args.repeat_count is not None or os.environ.get("REPEAT_COUNT"):
        env_updates["REPEAT_COUNT"] = str(args.repeat_count if args.repeat_count is not None else os.environ.get("REPEAT_COUNT", RUNNER_DEFAULTS["repeat_count"]))
    if args.max_retries is not None or os.environ.get("MAX_RETRIES"):
        env_updates["MAX_RETRIES"] = str(args.max_retries if args.max_retries is not None else os.environ.get("MAX_RETRIES", RUNNER_DEFAULTS["max_retries"]))
    if args.cpu_only or _truthy(os.environ.get("CPU_ONLY")):
        env_updates["CPU_ONLY"] = "1"
    if args.surface or os.environ.get("SURFACE"):
        env_updates["SURFACE"] = str(args.surface or os.environ["SURFACE"])
    if args.tasks or os.environ.get("TASKS"):
        env_updates["TASKS"] = str(args.tasks or os.environ["TASKS"])
    if args.param_band or os.environ.get("PARAM_BAND"):
        env_updates["PARAM_BAND"] = str(args.param_band or os.environ["PARAM_BAND"])
    if args.legacy_architecture_grid or _truthy(os.environ.get("LEGACY_ARCHITECTURE_GRID")):
        env_updates["LEGACY_ARCHITECTURE_GRID"] = "1"
    if args.include_small_stl or _truthy(os.environ.get("INCLUDE_SMALL_STL")):
        env_updates["INCLUDE_SMALL_STL"] = "1"
    env_updates["INCLUDE_VANILLA"] = "1" if include_vanilla else "0"
    if args.vanilla_depths or os.environ.get("VANILLA_DEPTHS"):
        env_updates["VANILLA_DEPTHS"] = str(args.vanilla_depths or os.environ["VANILLA_DEPTHS"])
    if args.vanilla_widths or os.environ.get("VANILLA_WIDTHS"):
        env_updates["VANILLA_WIDTHS"] = str(args.vanilla_widths or os.environ["VANILLA_WIDTHS"])
    if args.vanilla_model or os.environ.get("VANILLA_MODEL"):
        env_updates["VANILLA_MODEL"] = str(args.vanilla_model or os.environ["VANILLA_MODEL"])
    if args.max_epochs_vanilla is not None or os.environ.get("MAX_EPOCHS_VANILLA"):
        env_updates["MAX_EPOCHS_VANILLA"] = str(_epoch_limit(args.max_epochs_vanilla, "MAX_EPOCHS_VANILLA", RUNNER_DEFAULTS["max_epochs_vanilla"]))
    if args.width_only_depths or os.environ.get("WIDTH_ONLY_DEPTHS"):
        env_updates["WIDTH_ONLY_DEPTHS"] = str(args.width_only_depths or os.environ["WIDTH_ONLY_DEPTHS"])
    if args.width_to_depth_width is not None or os.environ.get("WIDTH_TO_DEPTH_WIDTH"):
        env_updates["WIDTH_TO_DEPTH_WIDTH"] = str(args.width_to_depth_width if args.width_to_depth_width is not None else os.environ["WIDTH_TO_DEPTH_WIDTH"])
    if args.width_to_depth_depth is not None or os.environ.get("WIDTH_TO_DEPTH_DEPTH"):
        env_updates["WIDTH_TO_DEPTH_DEPTH"] = str(args.width_to_depth_depth if args.width_to_depth_depth is not None else os.environ["WIDTH_TO_DEPTH_DEPTH"])
    if args.width_to_depth_widths or os.environ.get("WIDTH_TO_DEPTH_WIDTHS"):
        env_updates["WIDTH_TO_DEPTH_WIDTHS"] = str(args.width_to_depth_widths or os.environ["WIDTH_TO_DEPTH_WIDTHS"])
    if args.depth_only_widths or os.environ.get("DEPTH_ONLY_WIDTHS"):
        env_updates["DEPTH_ONLY_WIDTHS"] = str(args.depth_only_widths or os.environ["DEPTH_ONLY_WIDTHS"])
    if args.max_width is not None or os.environ.get("MAX_WIDTH"):
        env_updates["MAX_WIDTH"] = str(args.max_width if args.max_width is not None else os.environ.get("MAX_WIDTH", DEFAULT_MAX_WIDTH))
    if args.max_depth is not None or os.environ.get("MAX_DEPTH"):
        env_updates["MAX_DEPTH"] = str(args.max_depth if args.max_depth is not None else os.environ.get("MAX_DEPTH", DEFAULT_MAX_DEPTH))
    if args.max_neurons is not None or os.environ.get("MAX_NEURONS"):
        env_updates["MAX_NEURONS"] = str(args.max_neurons if args.max_neurons is not None else os.environ.get("MAX_NEURONS", DEFAULT_MAX_NEURONS))
    if args.head_variants or os.environ.get("HEAD_VARIANTS"):
        env_updates["HEAD_VARIANTS"] = str(args.head_variants or os.environ["HEAD_VARIANTS"])
    if args.missing_present_task_repeats is not None or os.environ.get("MISSING_PRESENT_TASK_REPEATS"):
        env_updates["MISSING_PRESENT_TASK_REPEATS"] = str(args.missing_present_task_repeats if args.missing_present_task_repeats is not None else os.environ["MISSING_PRESENT_TASK_REPEATS"])
    if args.prediction_repeats is not None or os.environ.get("PREDICTION_REPEATS"):
        env_updates["PREDICTION_REPEATS"] = str(args.prediction_repeats if args.prediction_repeats is not None else os.environ["PREDICTION_REPEATS"])
    if fresh:
        env_updates["OPF_ADP_DATA_FRESH"] = "1"
    os.environ.update(env_updates)

    banner_label = "CASE SUITE" if args.mode == "case" else "MASSIVE EXPERIMENT SUITE"
    print("============================================================")
    print(f" [OPF_ADP] {banner_label} INITIATED ")
    print("============================================================")
    print(
        " [OPF_ADP] launch concurrency=unbounded-by-default "
        f"host_stop={os.environ.get('HOST_RAM_STOP_AVAIL_MIB', RUNNER_DEFAULTS['host_ram_stop_avail_mib'])} "
        f"host_resume={os.environ.get('HOST_RAM_RESUME_AVAIL_MIB', RUNNER_DEFAULTS['host_ram_resume_avail_mib'])} "
        f"gpu_stop={os.environ.get('GPU_STOP_AVAIL_MIB', RUNNER_DEFAULTS['gpu_stop_avail_mib'])} "
        f"gpu_resume={os.environ.get('GPU_RESUME_AVAIL_MIB', RUNNER_DEFAULTS['gpu_resume_avail_mib'])}"
    )
    if args.mode == "case":
        print(f" [OPF_ADP] case={case_name} tag={case_tag} data_root={data_root}")
    else:
        print(
            f" [OPF_ADP] surface={args.surface or os.environ.get('SURFACE', 'all')} "
            f"tasks={args.tasks or os.environ.get('TASKS', 'all')} "
            f"head_variants={args.head_variants or os.environ.get('HEAD_VARIANTS', 'single_head,two_heads,four_heads')}"
        )

    if args.mode == "case":
        print(f"[INFO] Preparing case {case_name} at {data_root} with chunk_size={chunk_size}")
        manifest = prepare_case_cache(
            root=data_root,
            case_name=str(case_name),
            num_groups=int(os.environ.get("OPF_ADP_NUM_GROUPS", "20")),
            chunk_size=chunk_size,
            force_reload=fresh,
        )
        print(
            f"Prepared OPF dataset for {case_name} at {data_root} "
            f"(splits=train, val, test, chunksize={manifest['chunk_size']})"
        )

    runner_script = Path(__file__).resolve().parent / "opf_massive_suite_runner.py"
    command = [sys.executable, "-u", str(runner_script), *_runner_base_args(args.mode, args, results_dir, run_root, source_run_root), *runner_args]
    print("[OPF_ADP] Launch command:", " ".join(command))
    completed = None
    try:
        completed = subprocess.run(command, cwd=str(ROOT))
    finally:
        if auto_resume_bundle:
            try:
                export_results_bundle(results_path, bundle_path_for_results_dir(results_path))
            except Exception as exc:
                print(f"[OPF_ADP] Resume bundle export skipped: {exc}")
    print("============================================================")
    print(f" [OPF_ADP] {banner_label} COMPLETED ")
    print("============================================================")
    return int(completed.returncode if completed is not None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
