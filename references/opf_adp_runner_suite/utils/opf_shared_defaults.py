from __future__ import annotations

import hashlib
import itertools
import os
from typing import Dict, Iterator, Tuple

try:
    import torch
except Exception:  # pragma: no cover - torch is expected, but keep import resilient
    torch = None

DEFAULT_SUPPORTED_CASES: Tuple[str, ...] = (
    "pglib_opf_case14_ieee",
    "pglib_opf_case30_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case118_ieee",
    "pglib_opf_case500_goc",
)

DEFAULT_NUM_GROUPS = 20
DEFAULT_GROUP_SAMPLE_COUNT = 15_000
DEFAULT_TRAIN_RATIO = 0.90
DEFAULT_VAL_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.05
DEFAULT_CHUNK_SIZE = 512
DEFAULT_BATCH_SIZE = int(os.environ.get("OPF_ADP_BATCH_SIZE", "65536"))
DEFAULT_FLAT_SPLIT_CACHE_MAX_MIB = int(os.environ.get("OPF_ADP_FLAT_SPLIT_CACHE_MAX_MIB", "512"))
DEFAULT_FLAT_SPLIT_MATERIALIZE_MAX_SAMPLES = int(
    os.environ.get("OPF_ADP_FLAT_SPLIT_MATERIALIZE_MAX_SAMPLES", "50000")
)
DEFAULT_DIAGNOSTIC_SAMPLES = 256
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_TRIALS_DEPTH = 10
DEFAULT_TRIALS_WIDTH = 10
DEFAULT_INIT_WIDTH = 1
DEFAULT_MAX_WIDTH = 512
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_NEURONS = 1_000_000_000
# Zero is the explicit unlimited sentinel.  It is used instead of infinity so
# the value remains serializable and can safely cross the CLI/process boundary.
DEFAULT_MAX_EPOCHS = 0
DEFAULT_WARMUP_EPOCHS = 100
DEFAULT_SPL_THRESHOLD = 0.25
DEFAULT_DELTA = 0
DEFAULT_LR_WIDTH = 5e-4
DEFAULT_LR_DEPTH = 1e-3
DEFAULT_LR_FOUR_HEAD = 1e-3
DEFAULT_LR_COMMON = 1e-3
DEFAULT_GAP_THR = 5e-2
DEFAULT_EQ_TOL = 1e-1
DEFAULT_CONSTRAINT_THRESHOLDS = {
    "voltage_upper": 1e-3,
    "voltage_lower": 1e-3,
    "gen_real_upper": 1e-3,
    "gen_real_lower": 1e-3,
    "gen_reac_upper": 1e-3,
    "gen_reac_lower": 1e-3,
}

def _cuda_available() -> bool:
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except Exception:
        return False


DEFAULT_LAZY_DATALOADER_WORKERS = int(
    os.environ.get("OPF_ADP_LAZY_DATALOADER_WORKERS", "2" if _cuda_available() else "0")
)


DEFAULT_DATALOADER_NUM_WORKERS = int(
    os.environ.get(
        "OPF_ADP_DATALOADER_WORKERS",
        "0" if DEFAULT_BATCH_SIZE >= 65536 else ("2" if _cuda_available() else "0"),
    )
)
DEFAULT_DATALOADER_PIN_MEMORY = os.environ.get(
    "OPF_ADP_DATALOADER_PIN_MEMORY",
    "0" if DEFAULT_BATCH_SIZE >= 65536 else ("1" if _cuda_available() else "0"),
) not in {"0", "false", "False", ""}
DEFAULT_DATALOADER_PERSISTENT_WORKERS = os.environ.get(
    "OPF_ADP_DATALOADER_PERSISTENT_WORKERS",
    "0" if DEFAULT_DATALOADER_NUM_WORKERS <= 0 else "1",
) not in {"0", "false", "False", ""}
DEFAULT_DATALOADER_PREFETCH_FACTOR = int(os.environ.get("OPF_ADP_DATALOADER_PREFETCH_FACTOR", "2"))
DEFAULT_DATALOADER_SHUFFLE_TRAIN = True
DEFAULT_DATALOADER_SHUFFLE_EVAL = False

DEFAULT_ADP_MAX_EPOCHS = 0


def parse_epoch_limit(value: object, *, default: int = DEFAULT_MAX_EPOCHS) -> int:
    """Normalize an epoch limit; ``inf`` and zero mean no hard ceiling."""
    if value is None:
        return int(default)
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity", "unlimited", "none", "0"}:
        return 0
    try:
        limit = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid epoch limit {value!r}; use a non-negative integer or inf") from exc
    if limit < 0:
        raise ValueError(f"epoch limit must be non-negative or inf, got {value!r}")
    return limit


def epoch_indices(max_epochs: int, start: int = 0) -> Iterator[int]:
    """Iterate epochs with ``0`` meaning unlimited."""
    limit = parse_epoch_limit(max_epochs)
    return itertools.count(start) if limit == 0 else iter(range(start, limit))
DEFAULT_ADP_WEIGHT_DECAY = 1e-4
DEFAULT_ADP_GRAD_CLIP = 1.0
DEFAULT_ADP_DELTA = 1e-3
DEFAULT_POST_LAUNCH_SAMPLE_DELAY_SEC = 30

DEFAULT_TRAIN_SAMPLES_PER_GROUP = int(DEFAULT_GROUP_SAMPLE_COUNT * DEFAULT_TRAIN_RATIO)
DEFAULT_VAL_SAMPLES_PER_GROUP = int(DEFAULT_GROUP_SAMPLE_COUNT * DEFAULT_VAL_RATIO)
DEFAULT_TEST_SAMPLES_PER_GROUP = DEFAULT_GROUP_SAMPLE_COUNT - DEFAULT_TRAIN_SAMPLES_PER_GROUP - DEFAULT_VAL_SAMPLES_PER_GROUP

DEFAULT_WIDTH_ONLY_DEPTHS: Tuple[int, ...] = (1, 2, 3, 4, 5)
DEFAULT_WIDTH_TO_DEPTH_WIDTHS: Tuple[int, ...] = (1,)
DEFAULT_DEPTH_ONLY_WIDTHS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
DEFAULT_VANILLA_DEPTHS: Tuple[int, ...] = tuple(range(1, 6))
DEFAULT_VANILLA_WIDTHS: Tuple[int, ...] = DEFAULT_DEPTH_ONLY_WIDTHS
DEFAULT_GLOBAL_SEED = int(os.environ.get("OPF_ADP_GLOBAL_SEED", "42"))

DEFAULT_DATASET_DEFAULTS: Dict[str, object] = {
    "case_names": DEFAULT_SUPPORTED_CASES,
    "num_groups": DEFAULT_NUM_GROUPS,
    "group_sample_count": DEFAULT_GROUP_SAMPLE_COUNT,
    "train_ratio": DEFAULT_TRAIN_RATIO,
    "val_ratio": DEFAULT_VAL_RATIO,
    "test_ratio": DEFAULT_TEST_RATIO,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "batch_size": DEFAULT_BATCH_SIZE,
    "flat_split_cache_max_mib": DEFAULT_FLAT_SPLIT_CACHE_MAX_MIB,
    "flat_split_materialize_max_samples": DEFAULT_FLAT_SPLIT_MATERIALIZE_MAX_SAMPLES,
    "lazy_dataloader_workers": DEFAULT_LAZY_DATALOADER_WORKERS,
    "diagnostic_samples": DEFAULT_DIAGNOSTIC_SAMPLES,
    "epochs": DEFAULT_EPOCHS,
    "patience": DEFAULT_PATIENCE,
    "trials_depth": DEFAULT_TRIALS_DEPTH,
    "trials_width": DEFAULT_TRIALS_WIDTH,
    "init_width": DEFAULT_INIT_WIDTH,
    "max_width": DEFAULT_MAX_WIDTH,
    "max_depth": DEFAULT_MAX_DEPTH,
    "max_neurons": DEFAULT_MAX_NEURONS,
    "max_epochs": DEFAULT_MAX_EPOCHS,
    "warmup_epochs": DEFAULT_WARMUP_EPOCHS,
    "spl_thr": DEFAULT_SPL_THRESHOLD,
    "delta": DEFAULT_DELTA,
    "lr_width": DEFAULT_LR_WIDTH,
    "lr_depth": DEFAULT_LR_DEPTH,
    "lr_four_head": DEFAULT_LR_FOUR_HEAD,
    "lr_common": DEFAULT_LR_COMMON,
    "gap_thr": DEFAULT_GAP_THR,
    "eq_tol": DEFAULT_EQ_TOL,
    "constraint_thresholds": DEFAULT_CONSTRAINT_THRESHOLDS,
    "dataloader_num_workers": DEFAULT_DATALOADER_NUM_WORKERS,
    "dataloader_pin_memory": DEFAULT_DATALOADER_PIN_MEMORY,
    "dataloader_persistent_workers": DEFAULT_DATALOADER_PERSISTENT_WORKERS,
    "dataloader_prefetch_factor": DEFAULT_DATALOADER_PREFETCH_FACTOR,
    "dataloader_shuffle_train": DEFAULT_DATALOADER_SHUFFLE_TRAIN,
    "dataloader_shuffle_eval": DEFAULT_DATALOADER_SHUFFLE_EVAL,
    "adp_max_epochs": DEFAULT_ADP_MAX_EPOCHS,
    "adp_weight_decay": DEFAULT_ADP_WEIGHT_DECAY,
    "adp_grad_clip": DEFAULT_ADP_GRAD_CLIP,
    "adp_delta": DEFAULT_ADP_DELTA,
    "post_launch_sample_delay_sec": DEFAULT_POST_LAUNCH_SAMPLE_DELAY_SEC,
    "train_samples_per_group": DEFAULT_TRAIN_SAMPLES_PER_GROUP,
    "val_samples_per_group": DEFAULT_VAL_SAMPLES_PER_GROUP,
    "test_samples_per_group": DEFAULT_TEST_SAMPLES_PER_GROUP,
    "width_only_depths": DEFAULT_WIDTH_ONLY_DEPTHS,
    "width_to_depth_widths": DEFAULT_WIDTH_TO_DEPTH_WIDTHS,
    "depth_only_widths": DEFAULT_DEPTH_ONLY_WIDTHS,
    "vanilla_depths": DEFAULT_VANILLA_DEPTHS,
}


def derive_suite_seed(
    base_seed: int,
    *,
    mode: str,
    case_name: str,
    repeat_index: int,
    phase: str,
    family: str,
    head_variant: str,
    depth: int,
    width: int,
) -> int:
    payload = "|".join(
        [
            str(int(base_seed)),
            str(mode),
            str(case_name),
            str(int(repeat_index)),
            str(phase),
            str(family),
            str(head_variant),
            str(int(depth)),
            str(int(width)),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**31 - 1)
    return seed if seed > 0 else max(1, int(base_seed) % (2**31 - 1))

__all__ = [
    "DEFAULT_SUPPORTED_CASES",
    "DEFAULT_NUM_GROUPS",
    "DEFAULT_GROUP_SAMPLE_COUNT",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_FLAT_SPLIT_CACHE_MAX_MIB",
    "DEFAULT_FLAT_SPLIT_MATERIALIZE_MAX_SAMPLES",
    "DEFAULT_LAZY_DATALOADER_WORKERS",
    "DEFAULT_DIAGNOSTIC_SAMPLES",
    "DEFAULT_EPOCHS",
    "DEFAULT_PATIENCE",
    "DEFAULT_TRIALS_DEPTH",
    "DEFAULT_TRIALS_WIDTH",
    "DEFAULT_INIT_WIDTH",
    "DEFAULT_MAX_WIDTH",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NEURONS",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_WARMUP_EPOCHS",
    "DEFAULT_SPL_THRESHOLD",
    "DEFAULT_DELTA",
    "DEFAULT_LR_WIDTH",
    "DEFAULT_LR_DEPTH",
    "DEFAULT_LR_FOUR_HEAD",
    "DEFAULT_LR_COMMON",
    "DEFAULT_GAP_THR",
    "DEFAULT_EQ_TOL",
    "DEFAULT_CONSTRAINT_THRESHOLDS",
    "DEFAULT_DATALOADER_NUM_WORKERS",
    "DEFAULT_DATALOADER_PIN_MEMORY",
    "DEFAULT_DATALOADER_PERSISTENT_WORKERS",
    "DEFAULT_DATALOADER_PREFETCH_FACTOR",
    "DEFAULT_DATALOADER_SHUFFLE_TRAIN",
    "DEFAULT_DATALOADER_SHUFFLE_EVAL",
    "DEFAULT_ADP_MAX_EPOCHS",
    "DEFAULT_ADP_WEIGHT_DECAY",
    "DEFAULT_ADP_GRAD_CLIP",
    "DEFAULT_ADP_DELTA",
    "DEFAULT_POST_LAUNCH_SAMPLE_DELAY_SEC",
    "DEFAULT_TRAIN_SAMPLES_PER_GROUP",
    "DEFAULT_VAL_SAMPLES_PER_GROUP",
    "DEFAULT_TEST_SAMPLES_PER_GROUP",
    "DEFAULT_WIDTH_ONLY_DEPTHS",
    "DEFAULT_WIDTH_TO_DEPTH_WIDTHS",
    "DEFAULT_DEPTH_ONLY_WIDTHS",
    "DEFAULT_VANILLA_DEPTHS",
    "DEFAULT_VANILLA_WIDTHS",
    "DEFAULT_GLOBAL_SEED",
    "DEFAULT_DATASET_DEFAULTS",
    "derive_suite_seed",
]
