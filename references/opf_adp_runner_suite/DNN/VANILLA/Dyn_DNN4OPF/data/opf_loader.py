"""
opf_loader.py
=============

Purpose:
    Loads and preprocesses Optimal Power Flow (OPF) datasets from PGLib using PyTorch Geometric.
    Outputs are flattened tensors suitable for regression tasks.

Workflow:
    - Uses `OPFDataset` from torch_geometric to load Batches 1–5
    - Splits data into train/val/test sets
    - Flattens hetero-graph node features and generator outputs into tensors

Relevance to Papers:
    This file defines the *input-output interface* for all models.
    It supports continual learning by allowing batch-wise loading (for B1–B5),
    crucial for task-by-task training in EWC, DEN, Progressive networks.
"""

from torch import Tensor
import gc
import json
import logging
import math
import os
import fcntl
import tarfile
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import psutil
import torch
from torch_geometric.data import HeteroData
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset, TensorDataset

from config import VALID_CASES
from utils.opf_shared_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATALOADER_NUM_WORKERS,
    DEFAULT_DATALOADER_PIN_MEMORY,
    DEFAULT_DATALOADER_PERSISTENT_WORKERS,
    DEFAULT_DATALOADER_PREFETCH_FACTOR,
    DEFAULT_GROUP_SAMPLE_COUNT,
    DEFAULT_FLAT_SPLIT_CACHE_MAX_MIB,
    DEFAULT_FLAT_SPLIT_MATERIALIZE_MAX_SAMPLES,
    DEFAULT_LAZY_DATALOADER_WORKERS,
    DEFAULT_NUM_GROUPS,
    DEFAULT_DATALOADER_SHUFFLE_EVAL,
    DEFAULT_DATALOADER_SHUFFLE_TRAIN,
    DEFAULT_TEST_SAMPLES_PER_GROUP,
    DEFAULT_TRAIN_SAMPLES_PER_GROUP,
    DEFAULT_VAL_SAMPLES_PER_GROUP,
)
from utils.opf_dataset_cache import OPFCaseSplitDataset
from Dyn_DNN4OPF.utils.ybus_builder import construct_admittance_matrix


def _resolve_dataset_root() -> Path:
    env_root = os.environ.get("OPF_ADP_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "data"


DATASET_ROOT = _resolve_dataset_root()


class CheckpointableRandomSampler(Sampler[int]):
    """Random permutation sampler whose current epoch order can be checkpointed."""
    def __init__(self, data_source: Dataset) -> None:
        self.data_source = data_source
        self._order: list[int] | None = None
        self._exhausted = False
        self._resume_pending = False

    def __iter__(self):
        if self._order is None or (self._exhausted and not self._resume_pending):
            size = len(self.data_source)
            if size:
                # Match torch.utils.data.RandomSampler(generator=None): it
                # draws a seed from the global RNG, then creates the
                # permutation with a private generator. DataLoader itself has
                # already consumed its base seed before the sampler is read.
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(seed)
                self._order = torch.randperm(size, generator=generator).tolist()
            else:
                self._order = []
            self._exhausted = False
        self._resume_pending = False
        order = self._order
        for index in order:
            yield index
        self._exhausted = True

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict[str, Any]:
        return {
            "order": list(self._order) if self._order is not None else None,
            "exhausted": self._exhausted,
        }

    def load_state_dict(self, state: dict[str, Any], *, replay_current_epoch: bool = True) -> None:
        order = state.get("order")
        if order is not None:
            if len(order) != len(self.data_source) or sorted(order) != list(range(len(self.data_source))):
                raise ValueError("checkpoint sampler order does not match dataset length")
            self._order = [int(i) for i in order]
            self._exhausted = bool(state.get("exhausted", False))
            self._resume_pending = bool(replay_current_epoch)
        else:
            self._order = None
            self._exhausted = False
            self._resume_pending = False


class CheckpointableDataLoader(DataLoader):
    """DataLoader that avoids replaying the base-seed draw after mid-epoch resume."""

    _resume_rng_state: dict[str, Any] | None = None

    def defer_rng_restore(self, state: dict[str, Any] | None) -> None:
        self._resume_rng_state = state

    def __iter__(self):
        iterator = super().__iter__()
        state = self._resume_rng_state
        if state is not None:
            from Dyn_DNN4OPF.utils.resume_state import restore_rng_state

            if not restore_rng_state(state):
                raise ValueError("checkpoint RNG state could not be restored")
            self._resume_rng_state = None
        return iterator


def _make_loader(dataset: Dataset, *, batch_size: int, shuffle: bool, lazy: bool = False) -> DataLoader:
    kwargs = (_lazy_loader_kwargs if lazy else _loader_kwargs)(batch_size=batch_size, shuffle=False)
    kwargs.pop("shuffle", None)
    if shuffle:
        kwargs["sampler"] = CheckpointableRandomSampler(dataset)
    return CheckpointableDataLoader(dataset, **kwargs)

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust level as needed


def _memory_snapshot() -> str:
    try:
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        rss_mib = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        return (
            f"rss={rss_mib:.1f}MiB "
            f"host_avail={vm.available / (1024 * 1024):.1f}MiB "
            f"swap_used={swap.used / (1024 * 1024):.1f}MiB "
            f"swap_pct={swap.percent:.1f}%"
        )
    except Exception:
        return "rss=unknown host_avail=unknown swap=unknown"


def _flat_split_cache_root() -> Path:
    env_root = os.environ.get("OPF_ADP_FLAT_SPLIT_CACHE_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DATASET_ROOT / "flat_split_cache"


def _sample_case_json_path(case_name: str) -> Path:
    case_text = str(case_name)
    digits = "".join(ch for ch in case_text if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not infer case number from case_name={case_name!r}")
    return DATASET_ROOT / f"sample_case{digits}.json"


def _materialize_sample_case_json(case_name: str) -> Path:
    sample_path = _sample_case_json_path(case_name)
    if sample_path.exists():
        return sample_path

    release_root = DATASET_ROOT / "dataset_release_1" / case_name / "raw"
    tarballs = sorted(release_root.glob("*.tar.gz"))
    if not tarballs:
        raise FileNotFoundError(f"Missing sample file for '{case_name}': {sample_path}")

    target_member_suffix = f"{case_name}/group_0/example_0.json"
    fallback_member_suffix = "group_0/example_0.json"
    for tarball in tarballs:
        with tarfile.open(tarball, "r:gz") as archive:
            chosen_name = None
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                name = member.name
                if name.endswith(target_member_suffix) or name.endswith(fallback_member_suffix):
                    chosen_name = name
                    break
            if chosen_name is None:
                continue
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(chosen_name) as src, sample_path.open("wb") as dst:
                if src is None:
                    continue
                dst.write(src.read())
            logger.info("Materialized missing sample JSON for %s at %s from %s", case_name, sample_path, tarball)
            return sample_path

    raise FileNotFoundError(f"Missing sample file for '{case_name}': {sample_path}")


class LazyFlattenedOPFDataset(Dataset):
    """Flatten OPF samples on demand instead of pre-materializing the full split."""

    def __init__(self, base_dataset: Dataset, label: str) -> None:
        self.base_dataset = base_dataset
        self.label = label
        self._len = len(base_dataset)
        self._request_count = 0
        logger.info(
            "Prepared lazy flattened %s set with %d samples (%s)",
            label,
            self._len,
            _memory_snapshot(),
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += self._len
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)

        self._request_count += 1
        if self._request_count <= 5 or self._request_count % 250 == 0 or self._request_count == self._len:
            logger.info(
                "[%s] lazy flatten request %d/%d (%s)",
                self.label,
                self._request_count,
                self._len,
                _memory_snapshot(),
            )

        data = self.base_dataset[idx]
        return flatten_heterodata(data)


def _flatten_sample(sample: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(sample, HeteroData):
        return flatten_heterodata(sample)
    if isinstance(sample, (list, tuple)) and len(sample) >= 3:
        return torch.as_tensor(sample[0]), torch.as_tensor(sample[1]), torch.as_tensor(sample[2])
    raise TypeError(f"Unsupported sample type for flattening: {type(sample)!r}")


def _estimate_flattened_sample_bytes(sample: object) -> int:
    x, y, obj = _flatten_sample(sample)
    x = x.detach().cpu()
    y = y.detach().cpu()
    obj = obj.detach().cpu()
    total = x.numel() * x.element_size()
    total += y.numel() * y.element_size()
    total += obj.numel() * obj.element_size()
    return int(total)


def _materialize_flattened_dataset(dataset: Dataset, label: str) -> TensorDataset:
    length = len(dataset)
    if length == 0:
        empty = torch.empty((0,))
        return TensorDataset(empty, empty, empty)

    start = time.time()
    first_x, first_y, first_obj = _flatten_sample(dataset[0])
    first_x = first_x.detach().cpu().reshape(-1)
    first_y = first_y.detach().cpu().reshape(-1)
    first_obj = first_obj.detach().cpu().reshape(-1)
    first_sample = dataset[0]

    xs = torch.empty((length, first_x.numel()), dtype=first_x.dtype)
    ys = torch.empty((length, first_y.numel()), dtype=first_y.dtype)
    objs = torch.empty((length, first_obj.numel()), dtype=first_obj.dtype)

    def _store_row(index: int, sample: object) -> None:
        x, y, obj = _flatten_sample(sample)
        xs[index].copy_(x.detach().cpu().reshape(-1))
        ys[index].copy_(y.detach().cpu().reshape(-1))
        objs[index].copy_(obj.detach().cpu().reshape(-1))

    logger.info(
        "Materializing flat %s split with %d samples (%s)",
        label,
        length,
        _memory_snapshot(),
    )
    with torch.no_grad():
        _store_row(0, first_sample)
        for idx in range(1, length):
            if idx == 1 or idx % 1000 == 0 or idx == length - 1:
                logger.info(
                    "[%s] materialize progress %d/%d elapsed=%.1fs (%s)",
                    label,
                    idx,
                    length,
                    time.time() - start,
                    _memory_snapshot(),
                )
            _store_row(idx, dataset[idx])

    logger.info(
        "Materialized flat %s split in %.2f seconds (%s, x=%s, y=%s, obj=%s)",
        label,
        time.time() - start,
        _memory_snapshot(),
        tuple(xs.shape),
        tuple(ys.shape),
        tuple(objs.shape),
    )
    return TensorDataset(xs, ys, objs)


def _dataset_identity(dataset: Dataset, label: str) -> dict[str, object]:
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    case_name = getattr(base_dataset, "case_name", getattr(dataset, "case_name", None))
    split_name = getattr(base_dataset, "split", getattr(dataset, "split", None))
    num_groups = getattr(base_dataset, "num_groups", getattr(dataset, "num_groups", None))

    identity: dict[str, object] = {
        "label": label,
        "dataset_type": type(base_dataset).__name__,
        "case_name": case_name,
        "split": split_name,
        "num_groups": int(num_groups) if num_groups is not None else None,
        "length": len(dataset),
    }

    if isinstance(dataset, Subset):
        indices = dataset.indices
        if isinstance(indices, range):
            identity["subset"] = {
                "kind": "range",
                "start": int(indices.start),
                "stop": int(indices.stop),
                "step": int(indices.step),
            }
        else:
            indices_list = list(indices)
            if not indices_list:
                identity["subset"] = {"kind": "empty"}
            else:
                contiguous = all(
                    int(value) == int(indices_list[0]) + offset for offset, value in enumerate(indices_list)
                )
                if contiguous:
                    identity["subset"] = {
                        "kind": "range",
                        "start": int(indices_list[0]),
                        "stop": int(indices_list[-1]) + 1,
                        "step": 1,
                    }
                else:
                    digest = hashlib.sha256(
                        ",".join(str(int(value)) for value in indices_list).encode("utf-8")
                    ).hexdigest()[:16]
                    identity["subset"] = {
                        "kind": "hash",
                        "digest": digest,
                        "count": len(indices_list),
                    }

    return identity


def _flat_split_cache_path(dataset: Dataset, label: str) -> Path:
    identity = _dataset_identity(dataset, label)
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    case_name = str(identity.get("case_name") or "unknown_case")
    split_name = str(identity.get("split") or label.lower())
    return _flat_split_cache_root() / case_name / f"{split_name}_{digest}.pt"


def _save_flat_split_cache(cache_path: Path, dataset: TensorDataset, identity: dict[str, object]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": identity,
        "tensors": tuple(t.detach().cpu() for t in dataset.tensors),
    }
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def _load_flat_split_cache_payload(cache_path: Path) -> dict[str, object] | None:
    load_attempts = (
        {"map_location": "cpu", "weights_only": True, "mmap": True},
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu", "mmap": True},
        {"map_location": "cpu"},
    )
    for load_kwargs in load_attempts:
        try:
            payload = torch.load(cache_path, **load_kwargs)
            if load_kwargs.get("mmap"):
                logger.info("Loaded flat split cache %s with mmap=%s", cache_path, load_kwargs.get("mmap"))
            return payload
        except TypeError:
            continue
        except Exception:
            logger.exception("Failed to load flat split cache %s with kwargs=%s", cache_path, load_kwargs)
    return None


def _load_flat_split_cache(cache_path: Path, identity: dict[str, object]) -> TensorDataset | None:
    if not cache_path.exists():
        return None
    payload = _load_flat_split_cache_payload(cache_path)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("identity") != identity:
        return None
    tensors = payload.get("tensors")
    if not isinstance(tensors, (list, tuple)) or len(tensors) != 3:
        return None
    return TensorDataset(*tensors)


def _load_or_build_flat_split(
    dataset: Dataset,
    label: str,
    cache_path: Path,
    identity: dict[str, object],
) -> TensorDataset:
    cached = _load_flat_split_cache(cache_path, identity)
    if cached is not None:
        logger.info("Loaded cached flat %s split from %s", label, cache_path)
        return cached

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Acquiring flat %s split build lock at %s", label, lock_path)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached = _load_flat_split_cache(cache_path, identity)
        if cached is not None:
            logger.info("Loaded cached flat %s split from %s after lock wait", label, cache_path)
            return cached
        logger.info("Building cached flat %s split at %s", label, cache_path)
        built = _materialize_flattened_dataset(dataset, label)
        _save_flat_split_cache(cache_path, built, identity)
        return built


def _subset_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    limit_i = max(0, min(int(limit), len(dataset)))
    return Subset(dataset, range(limit_i))


def _materialized_flat_loaders_enabled() -> bool:
    value = os.environ.get("OPF_ADP_ENABLE_FLAT_MATERIALIZATION")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _loader_kwargs(*, batch_size: int, shuffle: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": DEFAULT_DATALOADER_NUM_WORKERS,
        "pin_memory": DEFAULT_DATALOADER_PIN_MEMORY,
    }
    if DEFAULT_DATALOADER_NUM_WORKERS > 0:
        kwargs["persistent_workers"] = DEFAULT_DATALOADER_PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = DEFAULT_DATALOADER_PREFETCH_FACTOR
    return kwargs


def _lazy_loader_kwargs(*, batch_size: int, shuffle: bool) -> dict[str, object]:
    kwargs = _loader_kwargs(batch_size=batch_size, shuffle=shuffle)
    if DEFAULT_LAZY_DATALOADER_WORKERS > 0:
        kwargs["num_workers"] = DEFAULT_LAZY_DATALOADER_WORKERS
        kwargs["persistent_workers"] = DEFAULT_DATALOADER_PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = DEFAULT_DATALOADER_PREFETCH_FACTOR
    return kwargs


def _collect_objectives(dataset: Dataset, label: str) -> torch.Tensor:
    start = time.time()
    objs: list[torch.Tensor] = []
    logger.info("Collecting %s objectives from %d samples (%s)", label, len(dataset), _memory_snapshot())
    for idx in range(len(dataset)):
        if idx == 0 or idx % 1000 == 0 or idx == len(dataset) - 1:
            logger.info(
                "[%s] objective progress %d/%d elapsed=%.1fs (%s)",
                label,
                idx,
                len(dataset),
                time.time() - start,
                _memory_snapshot(),
            )
        sample = dataset[idx]
        obj = sample["objective"] if isinstance(sample, HeteroData) else sample[2]
        objs.append(torch.as_tensor(obj))
    if not objs:
        return torch.empty(0)
    return torch.stack(objs)

def flatten_heterodata(data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten features and targets for OPF prediction.

    Input:
        - [PD, QD] per bus (shape: [n_buses, 2]); zero for buses without loads.

    Output:
        - [PG, QG] from `generator.y` (shape: [n_gens, 2])
        - [VA, VM] from `bus.y`       (shape: [n_buses, 2])

    Returns:
        Tuple[Tensor, Tensor]: (input x, target y)
        - x: shape [2 * n_buses]
        - y: shape [2 * (n_gens + n_buses)]
        - objective : shape[1]
    """
    # Number of buses in this sample
    n_buses = data['bus'].num_nodes

    # Build a [n_buses × 2] PD/QD matrix (zero if no load on that bus)
    pd_qd_per_bus = torch.zeros(
        (n_buses, data['load'].x.size(1)),
        dtype=data['load'].x.dtype,
        device=data['load'].x.device
    )

    # Map each load node to its bus via the load_link edge_index.
    # Use a vectorized scatter path instead of a Python loop so the first
    # training batch does not stall on large cases like case500.
    load_to_bus = data['load', 'load_link', 'bus'].edge_index
    if load_to_bus.numel() > 0:
        load_bus_idx = load_to_bus[1].to(torch.long)
        pd_qd_per_bus.index_add_(0, load_bus_idx, data['load'].x)
    # Flatten into [2 * n_buses]
    pd = pd_qd_per_bus[:, 0]  # Shape: [n_buses]
    qd = pd_qd_per_bus[:, 1]  # Shape: [n_buses]

    # Concatenate PD and QD in sequence
    x = torch.cat([pd, qd])

    # Output vector: generator power outputs and bus voltages
    gen_y = data['generator'].y
    y_gen = gen_y.transpose(0, 1).flatten()

    bus_y = data['bus'].y 
    y_bus = bus_y.transpose(0, 1).flatten() 

    # Concatenate to form final target vector
    y = torch.cat([y_gen, y_bus])

    #get objective function
    obj = data['objective']
    return x, y, obj

allowed_cases= [
    "pglib_opf_case14_ieee", "pglib_opf_case30_ieee", "pglib_opf_case57_ieee",
    "pglib_opf_case118_ieee", "pglib_opf_case500_goc", "pglib_opf_case2000_goc",
    "pglib_opf_case4661_sdet", "pglib_opf_case6470_rte",
    "pglib_opf_case10000_goc", "pglib_opf_case13659_pegase",
]

_DATASET_CACHE = {}

def get_cached_dataset(root, case_name, split, num_groups=None):
    key = (root, case_name, split, num_groups)
    if key not in _DATASET_CACHE:
        logger.info(f"Loading OPF cache for {case_name} ({split}) at {root}...")
        effective_num_groups = int(num_groups) if num_groups else DEFAULT_NUM_GROUPS
        if split == "all":
            _DATASET_CACHE[key] = ConcatDataset(
                [
                    OPFCaseSplitDataset(root=root, case_name=case_name, split="train", num_groups=effective_num_groups),
                    OPFCaseSplitDataset(root=root, case_name=case_name, split="val", num_groups=effective_num_groups),
                    OPFCaseSplitDataset(root=root, case_name=case_name, split="test", num_groups=effective_num_groups),
                ]
            )
        else:
            _DATASET_CACHE[key] = OPFCaseSplitDataset(
                root=root,
                case_name=case_name,
                split=split,
                num_groups=effective_num_groups,
            )
    return _DATASET_CACHE[key]

def load_opf_custom_split(
    *,
    case_name: str,
    train_samples: int | None = None,
    val_samples: int | None = None,
    test_samples: int | None = None,
    batches: list[int] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess OPF data for a given case with optional control over the number of samples and batches.
    """
    if case_name not in VALID_CASES:
        raise ValueError(f"Invalid case_name: {case_name}. Must be one of {VALID_CASES}")

    root_str = str(DATASET_ROOT)
    logger.info(f"Loading OPF data for case: {case_name} (Dataset Root: {root_str})")
    if batches is not None and isinstance(batches, int):
        batches = [batches]
    elif batches is not None and isinstance(batches, str):
        batches = [int(item) for item in batches.split(",") if item.strip()]
    batch_set = set(int(batch) for batch in batches) if batches is not None else None
    if batch_set:
        logger.info("Filtering custom split to batches=%s", sorted(batch_set))

    def tensorify(subset: list, label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        start = time.time()
        logger.info(f"Flattening {label} set with {len(subset)} samples... {_memory_snapshot()}")
        X, Y, OBJ= [], [], []
        last_report = start
        with torch.no_grad():
            i = -1
            try:
                for i, d in enumerate(subset):
                    if i == 0 or i % 1000 == 0 or i == len(subset) - 1:
                        logger.info(
                            "[%s] flatten progress %d/%d %s elapsed=%.1fs",
                            label,
                            i,
                            len(subset),
                            _memory_snapshot(),
                            time.time() - start,
                        )
                    elif i % 100 == 0 and (time.time() - last_report) >= 10.0:
                        logger.info(
                            "[%s] flatten heartbeat %d/%d %s elapsed=%.1fs",
                            label,
                            i,
                            len(subset),
                            _memory_snapshot(),
                            time.time() - start,
                        )
                        last_report = time.time()
                    x, y, obj= flatten_heterodata(d)
                    X.append(x)
                    Y.append(y)
                    OBJ.append(obj)
            except KeyboardInterrupt:
                logger.error(
                    "[%s] interrupted during flattening at sample %d/%d (%s elapsed=%.1fs)",
                    label,
                    max(i, 0),
                    len(subset),
                    _memory_snapshot(),
                    time.time() - start,
                )
                raise
        x_stack = torch.stack(X)
        y_stack = torch.stack(Y)
        obj_stack = torch.stack(OBJ)
        logger.info(
            "Finished flattening %s in %.2f seconds (%s, x=%s, y=%s, obj=%s)",
            label,
            time.time() - start,
            _memory_snapshot(),
            tuple(x_stack.shape),
            tuple(y_stack.shape),
            tuple(obj_stack.shape),
        )
        return x_stack, y_stack, obj_stack

    if batches is not None:
        logger.info(f"Using custom batches {batches} (overrides default split)")
        full_data = get_cached_dataset(root_str, case_name, "all")
        filtered = []
        for idx, d in enumerate(full_data):
            meta = getattr(d, "meta", {}) or {}
            batch_idx = meta.get("batch")
            if batch_idx is None:
                batch_idx = idx // DEFAULT_GROUP_SAMPLE_COUNT + 1
            try:
                batch_idx = int(batch_idx)
            except (TypeError, ValueError):
                batch_idx = idx // DEFAULT_GROUP_SAMPLE_COUNT + 1
            if batch_set is None or batch_idx in batch_set:
                filtered.append(d)
        total = len(filtered)
        if total == 0:
            raise ValueError(f"No samples found for case={case_name} in batches={batches}")
        
        n_train = min(train_samples or int(0.8 * total), int(0.8 * total))
        n_val = min(val_samples or int(0.1 * total), total - n_train)
        n_test = min(test_samples or (total - n_train - n_val), total - n_train - n_val)
        
        train_set = filtered[:n_train]
        val_set = filtered[n_train:n_train+n_val]
        test_set = filtered[n_train+n_val:n_train+n_val+n_test]
    else:
        if train_samples is None:
            train_set = get_cached_dataset(root_str, case_name, "train")
        else:
            batches_req = math.ceil(train_samples / DEFAULT_TRAIN_SAMPLES_PER_GROUP)
            train_raw = get_cached_dataset(root_str, case_name, "train", batches_req)
            train_set = [train_raw[i] for i in range(min(train_samples, len(train_raw)))]

        if val_samples is None:
            val_set = get_cached_dataset(root_str, case_name, "val")
        else:
            batches_req = math.ceil(val_samples / DEFAULT_VAL_SAMPLES_PER_GROUP)
            val_raw = get_cached_dataset(root_str, case_name, "val", batches_req)
            val_set = [val_raw[i] for i in range(min(val_samples, len(val_raw)))]

        if test_samples is None:
            test_set = get_cached_dataset(root_str, case_name, "test")
        else:
            batches_req = math.ceil(test_samples / DEFAULT_TEST_SAMPLES_PER_GROUP)
            test_raw = get_cached_dataset(root_str, case_name, "test", batches_req)
            test_set = [test_raw[i] for i in range(min(test_samples, len(test_raw)))]

    X_train, Y_train, OBJ_train = tensorify(train_set, "Train")
    X_val, Y_val, OBJ_val = tensorify(val_set, "Validation")
    X_test, Y_test , OBJ_test = tensorify(test_set, "Test")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, OBJ_train, OBJ_val, OBJ_test

def extract_opf_constraints(data: Dict) -> Dict[str, torch.Tensor]:
    """
    Extracts all physical constraint bounds and grid structure from a single OPFData JSON sample.

    Returns:
        dict[str, torch.Tensor] with keys:
            - v_min, v_max: voltage bounds (n_bus,)
            - p_min, p_max: active power bounds (n_gen,)
            - q_min, q_max: reactive power bounds (n_gen,)
            - thermal_limits: thermal capacity bounds (n_branch,)
            - y_bus: complex admittance matrix (sparse format)
            - bus_map: str-to-int map of bus indices
            - gen_buses, load_buses: int lists of bus IDs hosting generators/loads
            - from_bus, to_bus: int arrays defining network topology
    """
    # ── 1. Voltage limits at each bus ─────────────────────────────
    try:
        bus_data = data['grid']['nodes']['bus']
        v_min = torch.tensor([b[2] for b in bus_data], dtype=torch.float64)

        v_max = torch.tensor([b[3] for b in bus_data], dtype=torch.float64)
    except (KeyError, IndexError) as e:
        raise ValueError("Missing or malformed bus voltage bounds in data['grid']['nodes']['bus']") from e

    # ── 2. Generator limits ──────────────────────────────────────
    try:
        gen_data = data['grid']['nodes']['generator']
        p_min = torch.tensor([g[2] for g in gen_data], dtype=torch.float64)
        p_max = torch.tensor([g[3] for g in gen_data], dtype=torch.float64)
        q_min = torch.tensor([g[5] for g in gen_data], dtype=torch.float64)
        q_max = torch.tensor([g[6] for g in gen_data], dtype=torch.float64)
    except (KeyError, IndexError) as e:
        raise ValueError("Missing or malformed generator bounds in data['grid']['nodes']['generator']") from e

    # ── 3. Line and transformer thermal limits ───────────────────
    try:
        ac_lines = data['grid']['edges']['ac_line']['features']
        trafos   = data['grid']['edges']['transformer']['features']
        thermal_limits = torch.tensor(
            [line[6] for line in ac_lines] + [t[4] for t in trafos],
            dtype=torch.float64
        )
        line_features = torch.tensor(ac_lines, dtype=torch.float64)
        transformer_features = torch.tensor(trafos, dtype=torch.float64)
    except (KeyError, IndexError) as e:
        raise ValueError("Missing or malformed thermal limits in AC lines/transformers") from e

    # ── 4. Network topology and admittance matrix ────────────────
    y_bus = construct_admittance_matrix(data)
    n_bus = len(bus_data)
    bus_map = {str(i): i for i in range(n_bus)}

    # ── 5. Generator and load bus mappings ───────────────────────
    try:
        gen_buses  = data['grid']['edges']['generator_link']['receivers']
        load_buses = data['grid']['edges']['load_link']['receivers']
    except KeyError as e:
        raise ValueError("Missing generator_link or load_link receivers") from e

    # ── 6. Power flow topology (from → to) ───────────────────────
    try:
        from_bus = torch.tensor(
            data['grid']['edges']['ac_line']['senders'] +
            data['grid']['edges']['transformer']['senders'],
            dtype=torch.long
        )
        to_bus = torch.tensor(
            data['grid']['edges']['ac_line']['receivers'] +
            data['grid']['edges']['transformer']['receivers'],
            dtype=torch.long
        )
    except KeyError as e:
        raise ValueError("Missing sender/receiver edge lists in AC lines or transformers") from e

    # �"?�"? 7. Bus metadata (types and shunts) �"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?�"?
    bus_types = torch.tensor([int(b[1]) if len(b) > 1 else 1 for b in bus_data], dtype=torch.long)

    shunt_b = torch.zeros(n_bus, dtype=torch.float64)
    shunt_g = torch.zeros(n_bus, dtype=torch.float64)
    try:
        shunt_nodes = data["grid"]["nodes"].get("shunt", [])
        shunt_link = data["grid"]["edges"].get("shunt_link", {})
        shunt_receivers = shunt_link.get("receivers", [])
        for idx, bus_idx in enumerate(shunt_receivers):
            if idx < len(shunt_nodes):
                sh = shunt_nodes[idx]
                if len(sh) >= 2:
                    shunt_b[bus_idx] += float(sh[0])
                    shunt_g[bus_idx] += float(sh[1])
    except (KeyError, AttributeError, TypeError):
        pass

    return {
        'v_max': v_max,
        'v_min': v_min,
        'p_max': p_max,
        'p_min': p_min,
        'q_max': q_max,
        'q_min': q_min,
        'thermal_limits': thermal_limits,
        'y_bus': y_bus,
        'bus_map': bus_map,
        'gen_buses': gen_buses,
        'load_buses': load_buses,
        'from_bus': from_bus,
        'to_bus': to_bus,
        'bus_types': bus_types,
        'shunt_b': shunt_b,
        'shunt_g': shunt_g,
        'line_features': line_features,
        'transformer_features': transformer_features,
    }

def load_case_bounds(case_name: str) -> Dict[str, torch.Tensor]:
    """
    Loads the physical constraints for a given OPF case from a JSON file located in `data/`.

    Args:
        case_name (str): must be one of VALID_CASES.

    Returns:
        Dictionary of bounds and grid properties as torch.Tensors.
    """
    if case_name not in VALID_CASES:
        raise ValueError(f"Invalid case_name: '{case_name}'. Must be one of {VALID_CASES}")
    file_path = _materialize_sample_case_json(case_name)
    print(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)

    return extract_opf_constraints(raw_json)

def load_output_bounds(case_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs (lo, hi) output constraint vectors in PG | QG | VA | VM order.

    Args:
        case_name (str): which case to load (default: case14)

    Returns:
        Tuple (lower_bounds, upper_bounds) as 1D tensors.
    """

    constraints = load_case_bounds(case_name)

    lo_pg, hi_pg = constraints["p_min"], constraints["p_max"]
    lo_qg, hi_qg = constraints["q_min"], constraints["q_max"]
    lo_vm, hi_vm = constraints["v_min"], constraints["v_max"]
    n_bus = lo_vm.shape[0]

    lo_va = torch.full((n_bus,), -math.pi, dtype=torch.float64)
    hi_va = torch.full((n_bus,),  math.pi, dtype=torch.float64)

    bounds_lo = torch.cat([lo_pg, lo_qg, lo_va, lo_vm])
    bounds_hi = torch.cat([hi_pg, hi_qg, hi_va, hi_vm])
    return bounds_lo, bounds_hi

def load_cost_coeff(data: Dict) -> Dict[str, Tensor]:
    """
    Extracts generator cost coefficients from a single OPFData JSON entry.

    Returns:
        Dict with keys: 'cost_q', 'cost_l', 'cost_c' (quadratic, linear, constant)
    """
    gen_features = data['grid']['nodes']['generator']
    cost_coeffs_q = torch.tensor([gen[8] for gen in gen_features], dtype=torch.float64)
    cost_coeffs_l = torch.tensor([gen[9] for gen in gen_features], dtype=torch.float64)
    cost_coeffs_c = torch.tensor([gen[10] for gen in gen_features], dtype=torch.float64)

    return {
        'cost_q': cost_coeffs_q,
        'cost_l': cost_coeffs_l,
        'cost_c': cost_coeffs_c
    }

def get_data_loaders(
    batch_size: int | None,
    case_name: str,
    train_samples: int | None,
    val_samples: int | None,
    test_samples: int | None,
    batches: list[int] | None,
    *,
    collect_test_objectives: bool = True,
):
    resolved_batch_size = int(batch_size or DEFAULT_BATCH_SIZE)
    if batches is not None:
        X_train, Y_train, X_val, Y_val, X_test, Y_test, OBJ_train, OBJ_val, OBJ_test = \
            load_opf_custom_split(
                case_name=case_name,
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                batches=batches
            )
        train_loader = _make_loader(TensorDataset(X_train, Y_train, OBJ_train), batch_size=resolved_batch_size, shuffle=DEFAULT_DATALOADER_SHUFFLE_TRAIN)
        val_loader = _make_loader(TensorDataset(X_val, Y_val, OBJ_val), batch_size=resolved_batch_size, shuffle=DEFAULT_DATALOADER_SHUFFLE_EVAL)
        test_loader = _make_loader(TensorDataset(X_test, Y_test, OBJ_test), batch_size=resolved_batch_size, shuffle=DEFAULT_DATALOADER_SHUFFLE_EVAL)
        logging.getLogger(__name__).info(
            "Loaded materialized data loaders (case=%s) with batch_size=%s workers=%s pin_memory=%s",
            case_name,
            resolved_batch_size,
            DEFAULT_DATALOADER_NUM_WORKERS,
            DEFAULT_DATALOADER_PIN_MEMORY,
        )
        if not collect_test_objectives:
            OBJ_test = None
        return train_loader, val_loader, test_loader, OBJ_test

    root_str = str(DATASET_ROOT)
    logger.info(
        "Loading lazy OPF data loaders for case=%s batch_size=%s train_samples=%s val_samples=%s test_samples=%s (%s)",
        case_name,
        resolved_batch_size,
        train_samples,
        val_samples,
        test_samples,
        _memory_snapshot(),
    )
    train_base = get_cached_dataset(root_str, case_name, "train")
    val_base = get_cached_dataset(root_str, case_name, "val")
    test_base = get_cached_dataset(root_str, case_name, "test")

    flat_cache_limit = int(os.environ.get("OPF_ADP_FLAT_SPLIT_CACHE_MAX_MIB", DEFAULT_FLAT_SPLIT_CACHE_MAX_MIB))
    train_split = _subset_dataset(train_base, train_samples)
    val_split = _subset_dataset(val_base, val_samples)
    test_split = _subset_dataset(test_base, test_samples)

    train_identity = _dataset_identity(train_split, "Train")
    val_identity = _dataset_identity(val_split, "Validation")
    test_identity = _dataset_identity(test_split, "Test")
    train_cache_path = _flat_split_cache_path(train_split, "Train")
    val_cache_path = _flat_split_cache_path(val_split, "Validation")
    test_cache_path = _flat_split_cache_path(test_split, "Test")

    if _materialized_flat_loaders_enabled():
        logger.info(
            "Using shared cached flat OPF data loaders for case=%s batch_size=%s limit=%dMiB",
            case_name,
            resolved_batch_size,
            flat_cache_limit,
        )
        train_dataset = _load_or_build_flat_split(train_split, "Train", train_cache_path, train_identity)
        val_dataset = _load_or_build_flat_split(val_split, "Validation", val_cache_path, val_identity)
        test_dataset = _load_or_build_flat_split(test_split, "Test", test_cache_path, test_identity)
        del train_base, val_base, test_base, train_split, val_split, test_split
        gc.collect()
    else:
        logger.info(
            "Using lazy flattened OPF data loaders for case=%s batch_size=%s (set OPF_ADP_ENABLE_FLAT_MATERIALIZATION=1 to enable shared flat caches)",
            case_name,
            resolved_batch_size,
        )
        train_dataset = LazyFlattenedOPFDataset(train_split, "Train")
        val_dataset = LazyFlattenedOPFDataset(val_split, "Validation")
        test_dataset = LazyFlattenedOPFDataset(test_split, "Test")

    use_lazy_loader = isinstance(train_dataset, LazyFlattenedOPFDataset)
    train_loader = _make_loader(train_dataset, batch_size=resolved_batch_size, shuffle=DEFAULT_DATALOADER_SHUFFLE_TRAIN, lazy=use_lazy_loader)
    val_loader = _make_loader(val_dataset, batch_size=resolved_batch_size, shuffle=DEFAULT_DATALOADER_SHUFFLE_EVAL, lazy=use_lazy_loader)
    test_loader = _make_loader(test_dataset, batch_size=resolved_batch_size, shuffle=DEFAULT_DATALOADER_SHUFFLE_EVAL, lazy=use_lazy_loader)

    OBJ_test = None
    if collect_test_objectives:
        OBJ_test = _collect_objectives(test_base if test_samples is None else _subset_dataset(test_base, test_samples), "Test")
    logging.getLogger(__name__).info(
        "Loaded lazy data loaders (case=%s) with batch_size=%s workers=%s pin_memory=%s (%s)",
        case_name,
        resolved_batch_size,
        DEFAULT_DATALOADER_NUM_WORKERS,
        DEFAULT_DATALOADER_PIN_MEMORY,
        _memory_snapshot(),
    )
    return train_loader, val_loader, test_loader, OBJ_test

def load_optuna_best_params(filename: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if not os.path.exists(filename):
        logger.warning(f"Optuna hyperparameter file '{filename}' not found. Using default config.")
        return params

    with open(filename, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            key, val = key.strip(), val.strip()
            if val.isdigit():
                params[key] = int(val)
            else:
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    logger.info(f"Loaded Optuna best params: {params}")
    return params
