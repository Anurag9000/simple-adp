from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from utils.opf_shared_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONSTRAINT_THRESHOLDS,
    DEFAULT_DEPTH_ONLY_WIDTHS,
    DEFAULT_DIAGNOSTIC_SAMPLES,
    DEFAULT_DELTA,
    DEFAULT_EQ_TOL,
    DEFAULT_GAP_THR,
    DEFAULT_EPOCHS,
    DEFAULT_INIT_WIDTH,
    DEFAULT_LR_COMMON,
    DEFAULT_LR_DEPTH,
    DEFAULT_LR_FOUR_HEAD,
    DEFAULT_LR_WIDTH,
    DEFAULT_GLOBAL_SEED,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_WIDTH,
    DEFAULT_MAX_NEURONS,
    epoch_indices,
    parse_epoch_limit,
    DEFAULT_PATIENCE,
    DEFAULT_POST_LAUNCH_SAMPLE_DELAY_SEC,
    DEFAULT_SPL_THRESHOLD,
    DEFAULT_SUPPORTED_CASES,
    DEFAULT_TRIALS_DEPTH,
    DEFAULT_TRIALS_WIDTH,
    DEFAULT_WARMUP_EPOCHS,
    DEFAULT_VANILLA_DEPTHS,
    DEFAULT_VANILLA_WIDTHS,
    DEFAULT_WIDTH_ONLY_DEPTHS,
    DEFAULT_WIDTH_TO_DEPTH_WIDTHS,
)

DEFAULT_CASES = DEFAULT_SUPPORTED_CASES

DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "delta": DEFAULT_DELTA,
    "batch_size": DEFAULT_BATCH_SIZE,
    "patience": DEFAULT_PATIENCE,
    "trials_depth": DEFAULT_TRIALS_DEPTH,
    "trials_width": DEFAULT_TRIALS_WIDTH,
    "warmup_epochs": DEFAULT_WARMUP_EPOCHS,
    "spl_thr": DEFAULT_SPL_THRESHOLD,
    "ex_k": 1,
    "init_width": DEFAULT_INIT_WIDTH,
    "init_depth": 1,
    "max_width": DEFAULT_MAX_WIDTH,
    "max_depth": DEFAULT_MAX_DEPTH,
    "max_neurons": DEFAULT_MAX_NEURONS,
    "max_epochs": DEFAULT_MAX_EPOCHS,
    "epochs": DEFAULT_EPOCHS,
    "diagnostic_samples": DEFAULT_DIAGNOSTIC_SAMPLES,
    "clip_test": False,
    "pg_thr": 1e-2,
    "qg_thr": 1e-2,
    "vm_thr": 1e-2,
    "gap_thr": DEFAULT_GAP_THR,
    "constraint_thresholds": deepcopy(DEFAULT_CONSTRAINT_THRESHOLDS),
    "train_samples": None,
    "val_samples": None,
    "test_samples": None,
    "batches": None,
    "p_tol": DEFAULT_EQ_TOL,
    "q_tol": DEFAULT_EQ_TOL,
    "seed": DEFAULT_GLOBAL_SEED,
}

RUNNER_DEFAULTS: Dict[str, Any] = {
    "concurrency": 0,
    "pytorch_cuda_alloc_conf": "expandable_segments:True,max_split_size_mb:128",
    "cuda_visible_devices": "0",
    "tabular_child_shared_cpu": "0",
    "tabular_cpu_workers": "0",
    "host_ram_stop_avail_mib": 1024,
    "host_ram_resume_avail_mib": 1280,
    "gpu_stop_avail_mib": 100,
    "gpu_resume_avail_mib": 500,
    "launch_host_reserve_mib": 0,
    "launch_gpu_reserve_mib": 0,
    "gpu_child_gate_close_buffer_mib": 100,
    "host_child_gate_close_buffer_mib": 500,
    "external_memory_drop_reopen_mib": 500,
    "pressure_poll_interval_sec": 0.5,
    "post_launch_sample_delay_sec": DEFAULT_POST_LAUNCH_SAMPLE_DELAY_SEC,
    "batch_backoff_factor": 0.5,
    "swap_pressure_limit_pct": 70.0,
    "swap_resume_pct": 70.0,
    "diagnostic_samples": DEFAULT_DIAGNOSTIC_SAMPLES,
    "repeat_count": 5,
    "max_retries": 1,
    "missing_present_task_repeats": 1,
    "prediction_repeats": 1,
    "seed": DEFAULT_GLOBAL_SEED,
    "vanilla_min_band": 1,
    "vanilla_max_band": 6,
    "vanilla_samples": 3,
    "max_epochs_vanilla": DEFAULT_MAX_EPOCHS,
    "max_epochs_adp": DEFAULT_MAX_EPOCHS,
    "width_only_depths": ",".join(str(item) for item in DEFAULT_WIDTH_ONLY_DEPTHS),
    "width_to_depth_widths": ",".join(str(item) for item in DEFAULT_WIDTH_TO_DEPTH_WIDTHS),
    "depth_only_widths": ",".join(str(item) for item in DEFAULT_DEPTH_ONLY_WIDTHS),
    "vanilla_depths": ",".join(str(item) for item in DEFAULT_VANILLA_DEPTHS),
    "vanilla_widths": ",".join(str(item) for item in DEFAULT_VANILLA_WIDTHS),
    "max_width": DEFAULT_MAX_WIDTH,
    "max_depth": DEFAULT_MAX_DEPTH,
    "max_neurons": DEFAULT_MAX_NEURONS,
}

PRESSURE_WRAPPER_DEFAULTS: Dict[str, Any] = {
    "host_ram_pressure_limit_pct": 90.0,
    "host_ram_resume_pct": 85.0,
    "swap_pressure_limit_pct": 70.0,
    "swap_resume_pct": 70.0,
    "gpu_memory_pressure_limit_pct": 90.0,
    "gpu_memory_resume_pct": 85.0,
    "external_memory_drop_reopen_mib": 500.0,
    "pressure_poll_interval_sec": 0.5,
}


def make_adp_run_config(
    *,
    model: str,
    case_name: str,
    log_file: str,
    init_depth: int = 1,
    init_width: int = DEFAULT_INIT_WIDTH,
    lr: float = DEFAULT_LR_COMMON,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_RUN_CONFIG)
    config.update(
        {
            "model": model,
            "case_name": case_name,
            "log_file": log_file,
            "init_depth": init_depth,
            "init_width": init_width,
            "lr": lr,
        }
    )
    if extra:
        extra = dict(extra)
        thresholds = extra.pop("constraint_thresholds", None)
        if thresholds:
            merged = deepcopy(config["constraint_thresholds"])
            merged.update(thresholds)
            config["constraint_thresholds"] = merged
        config.update(extra)
    return config


def build_stage_datasets(
    train_loader,
    val_loader,
    test_loader,
    *,
    diagnostic_samples: int | None = None,
):
    from Dyn_DNN4OPF.utils.config import materialize_loader_tensors

    diag_limit = diagnostic_samples if diagnostic_samples is not None else DEFAULT_DIAGNOSTIC_SAMPLES
    X_tr, Y_tr, _ = materialize_loader_tensors(train_loader, limit=diag_limit)
    X_va, Y_va, _ = materialize_loader_tensors(val_loader, limit=diag_limit)
    X_te, Y_te, _ = materialize_loader_tensors(test_loader, limit=diag_limit)
    return {"Train": (X_tr, Y_tr), "Validation": (X_va, Y_va), "Test": (X_te, Y_te)}


__all__ = [
    "DEFAULT_DIAGNOSTIC_SAMPLES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_PATIENCE",
    "DEFAULT_TRIALS_DEPTH",
    "DEFAULT_TRIALS_WIDTH",
    "DEFAULT_INIT_WIDTH",
    "DEFAULT_MAX_NEURONS",
    "DEFAULT_MAX_EPOCHS",
    "epoch_indices",
    "parse_epoch_limit",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_LR_WIDTH",
    "DEFAULT_LR_DEPTH",
    "DEFAULT_LR_COMMON",
    "DEFAULT_LR_FOUR_HEAD",
    "DEFAULT_CONSTRAINT_THRESHOLDS",
    "DEFAULT_CASES",
    "DEFAULT_VANILLA_DEPTHS",
    "DEFAULT_VANILLA_WIDTHS",
    "DEFAULT_RUN_CONFIG",
    "DEFAULT_WIDTH_ONLY_DEPTHS",
    "DEFAULT_WIDTH_TO_DEPTH_WIDTHS",
    "DEFAULT_DEPTH_ONLY_WIDTHS",
    "RUNNER_DEFAULTS",
    "PRESSURE_WRAPPER_DEFAULTS",
    "make_adp_run_config",
    "build_stage_datasets",
]
