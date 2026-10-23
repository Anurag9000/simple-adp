from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import tarfile
import time
from bisect import bisect_left
from contextlib import contextmanager
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.data.separate import separate
from torch_geometric.datasets.opf import extract_edge_index, extract_edge_index_rev
from torch_geometric.data import download_url, extract_tar
from torch_geometric.io import fs

from utils.opf_shared_defaults import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GROUP_SAMPLE_COUNT,
    DEFAULT_NUM_GROUPS,
    DEFAULT_SUPPORTED_CASES,
    DEFAULT_TEST_SAMPLES_PER_GROUP,
    DEFAULT_TRAIN_SAMPLES_PER_GROUP,
    DEFAULT_VAL_SAMPLES_PER_GROUP,
)

logger = logging.getLogger(__name__)

VALID_CASES: tuple[str, ...] = (
    *DEFAULT_SUPPORTED_CASES,
    "pglib_opf_case2000_goc",
    "pglib_opf_case6470_rte",
    "pglib_opf_case4661_sdet",
    "pglib_opf_case10000_goc",
    "pglib_opf_case13659_pegase",
)

_SAMPLE_RE = re.compile(r"^example_(\d+)\.json$")
CACHE_FORMAT_VERSION = 2


def _default_data_root() -> Path:
    env_root = os.environ.get("OPF_ADP_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    return Path(__file__).resolve().parents[1] / "data"


def _default_chunk_size() -> int:
    raw_value = os.environ.get("OPF_ADP_DATA_CHUNK_SIZE")
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            pass
    return DEFAULT_CHUNK_SIZE


def _default_chunk_cache_size() -> int:
    raw_value = os.environ.get("OPF_ADP_DATA_CHUNK_CACHE_SIZE")
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            pass
    # Keep the cache intentionally small so a worker does not retain the full
    # split in memory. This is the main lever that keeps many parallel jobs
    # from ballooning past the host RAM ceiling.
    return 4


def _release_name(topological_perturbations: bool) -> str:
    return "dataset_release_1_nminusone" if topological_perturbations else "dataset_release_1"


def _parse_sample_index(name: str) -> int:
    match = _SAMPLE_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected OPF sample filename: {name}")
    return int(match.group(1))


def _group_dir(raw_dir: Path, release: str, case_name: str, group_idx: int) -> Path:
    return raw_dir / "gridopt-dataset-tmp" / release / case_name / f"group_{group_idx}"


def _group_complete(group_path: Path) -> bool:
    return group_path.exists() and group_path.is_dir() and sum(1 for p in group_path.iterdir() if p.is_file()) == DEFAULT_GROUP_SAMPLE_COUNT


def _global_sample_index(group_idx: int, sample_index: int) -> int:
    """Return the global index for either supported archive naming layout.

    GridOpt archives use global filenames (for example, group 1 starts at
    ``example_15000.json``), while local fixtures may restart filenames at
    zero for every group. Detect the former before applying the group offset
    so global indices are never offset twice.
    """
    group_idx = int(group_idx)
    sample_index = int(sample_index)
    group_start = group_idx * DEFAULT_GROUP_SAMPLE_COUNT
    if group_start <= sample_index < group_start + DEFAULT_GROUP_SAMPLE_COUNT:
        return sample_index
    if 0 <= sample_index < DEFAULT_GROUP_SAMPLE_COUNT:
        return group_start + sample_index
    return sample_index


@contextmanager
def _case_cache_lock(root: Path, release: str, case_name: str):
    """Serialize shared download/extract/cache construction across workers."""
    lock_path = root / ".locks" / f"{release}_{case_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            handle.write("0")
            handle.flush()
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _download_tarball(raw_dir: Path, release: str, tar_name: str) -> Path:
    tar_path = raw_dir / tar_name
    if tar_path.exists():
        tar_path.unlink()
    url = f"https://storage.googleapis.com/gridopt-dataset/{release}/{tar_name}"
    logger.info("Downloading %s", tar_name)
    download_url(url, str(raw_dir))
    return tar_path


def _extract_tarball_with_retry(raw_dir: Path, release: str, tar_name: str, tar_path: Path) -> None:
    try:
        extract_tar(str(tar_path), str(raw_dir))
        return
    except (EOFError, tarfile.ReadError, OSError) as exc:
        logger.warning(
            "Failed to extract %s (%s); removing and retrying download once",
            tar_name,
            exc,
        )

    if tar_path.exists():
        tar_path.unlink()
    url = f"https://storage.googleapis.com/gridopt-dataset/{release}/{tar_name}"
    download_url(url, str(raw_dir))
    extract_tar(str(tar_path), str(raw_dir))


def _ensure_raw_tarballs(
    *,
    raw_dir: Path,
    release: str,
    case_name: str,
    num_groups: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for group_idx in range(num_groups):
        tar_name = f"{case_name}_{group_idx}.tar.gz"
        tar_path = raw_dir / tar_name
        if not tar_path.exists() or not tarfile.is_tarfile(tar_path):
            tar_path = _download_tarball(raw_dir, release, tar_name)

        group_path = _group_dir(raw_dir, release, case_name, group_idx)
        if _group_complete(group_path):
            logger.info("Group %s already extracted", group_idx)
            continue

        if group_path.exists():
            shutil.rmtree(group_path)
        logger.info("Extracting %s", tar_name)
        _extract_tarball_with_retry(raw_dir, release, tar_name, tar_path)


def _purge_case_generated_state(
    *,
    root: Path,
    release: str,
    case_name: str,
) -> None:
    case_dir = root / release / case_name
    if not case_dir.exists():
        return

    for processed_dir in case_dir.glob("processed_*"):
        if processed_dir.is_dir():
            shutil.rmtree(processed_dir)

    raw_tmp_dir = case_dir / "raw" / "gridopt-dataset-tmp"
    if raw_tmp_dir.exists():
        shutil.rmtree(raw_tmp_dir)


def _build_heterodata(obj: Dict[str, Any]) -> HeteroData:
    grid = obj["grid"]
    solution = obj["solution"]
    metadata = obj["metadata"]

    data = HeteroData()
    data.x = torch.tensor(grid["context"]).view(-1)
    data.objective = torch.tensor(metadata["objective"])
    data.meta = metadata

    data["bus"].x = torch.tensor(grid["nodes"]["bus"])
    data["bus"].y = torch.tensor(solution["nodes"]["bus"])

    data["generator"].x = torch.tensor(grid["nodes"]["generator"])
    data["generator"].y = torch.tensor(solution["nodes"]["generator"])

    data["load"].x = torch.tensor(grid["nodes"]["load"])
    data["shunt"].x = torch.tensor(grid["nodes"]["shunt"])

    data["bus", "ac_line", "bus"].edge_index = extract_edge_index(obj, "ac_line")
    data["bus", "ac_line", "bus"].edge_attr = torch.tensor(grid["edges"]["ac_line"]["features"])
    data["bus", "ac_line", "bus"].edge_label = torch.tensor(solution["edges"]["ac_line"]["features"])

    data["bus", "transformer", "bus"].edge_index = extract_edge_index(obj, "transformer")
    data["bus", "transformer", "bus"].edge_attr = torch.tensor(grid["edges"]["transformer"]["features"])
    data["bus", "transformer", "bus"].edge_label = torch.tensor(solution["edges"]["transformer"]["features"])

    data["generator", "generator_link", "bus"].edge_index = extract_edge_index(obj, "generator_link")
    data["bus", "generator_link", "generator"].edge_index = extract_edge_index_rev(obj, "generator_link")

    data["load", "load_link", "bus"].edge_index = extract_edge_index(obj, "load_link")
    data["bus", "load_link", "load"].edge_index = extract_edge_index_rev(obj, "load_link")

    data["shunt", "shunt_link", "bus"].edge_index = extract_edge_index(obj, "shunt_link")
    data["bus", "shunt_link", "shunt"].edge_index = extract_edge_index_rev(obj, "shunt_link")
    return data


def _prepare_case_cache_unlocked(
    *,
    root: Path,
    case_name: str,
    num_groups: int = DEFAULT_NUM_GROUPS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    topological_perturbations: bool = False,
    force_reload: bool = False,
) -> Dict[str, Any]:
    release = _release_name(topological_perturbations)
    raw_dir = root / release / case_name / "raw"
    processed_dir = root / release / case_name / f"processed_{num_groups}"
    manifest_path = processed_dir / "manifest.json"

    if force_reload:
        _purge_case_generated_state(root=root, release=release, case_name=case_name)
    elif processed_dir.exists() and not manifest_path.exists():
        shutil.rmtree(processed_dir)

    if manifest_path.exists() and not force_reload:
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        if (
            manifest.get("version") == CACHE_FORMAT_VERSION
            and manifest.get("case_name") == case_name
            and int(manifest.get("num_groups", -1)) == num_groups
            and int(manifest.get("chunk_size", -1)) == chunk_size
            and manifest.get("release") == release
            and "splits" in manifest
        ):
            valid = True
            for split_name in ("train", "val", "test"):
                split_info = manifest["splits"].get(split_name, {})
                for chunk in split_info.get("chunks", []):
                    if not (processed_dir / chunk["path"]).exists():
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                logger.info(
                    "Reusing prepared cache for case=%s split=%s groups=%s chunk_size=%s",
                    case_name,
                    ",".join(("train", "val", "test")),
                    num_groups,
                    chunk_size,
                )
                return manifest

    _ensure_raw_tarballs(
        raw_dir=raw_dir,
        release=release,
        case_name=case_name,
        num_groups=num_groups,
    )

    train_limit = int(DEFAULT_TRAIN_SAMPLES_PER_GROUP * num_groups)
    val_limit = train_limit + int(DEFAULT_VAL_SAMPLES_PER_GROUP * num_groups)

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "version": CACHE_FORMAT_VERSION,
        "case_name": case_name,
        "release": release,
        "num_groups": num_groups,
        "chunk_size": chunk_size,
        "splits": {
            "train": {"chunks": [], "next_chunk_id": 0, "next_sample_index": 0, "count": 0},
            "val": {"chunks": [], "next_chunk_id": 0, "next_sample_index": 0, "count": 0},
            "test": {"chunks": [], "next_chunk_id": 0, "next_sample_index": 0, "count": 0},
        },
    }

    def save_manifest() -> None:
        tmp_path = manifest_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, manifest_path)

    buffers: dict[str, list[HeteroData]] = {"train": [], "val": [], "test": []}
    starts: dict[str, Optional[int]] = {"train": None, "val": None, "test": None}
    split_dirs = {
        "train": processed_dir / "train" / "chunks",
        "val": processed_dir / "val" / "chunks",
        "test": processed_dir / "test" / "chunks",
    }
    for split_dir in split_dirs.values():
        split_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building cache for case=%s groups=%s chunk_size=%s",
        case_name,
        num_groups,
        chunk_size,
    )

    def flush(split_name: str, end_index: int) -> None:
        buffer = buffers[split_name]
        if not buffer:
            return
        state = manifest["splits"][split_name]
        chunk_id = int(state["next_chunk_id"])
        chunk_name = f"chunk_{chunk_id:06d}.pt"
        final_path = split_dirs[split_name] / chunk_name
        tmp_path = split_dirs[split_name] / f"{chunk_name}.tmp"
        start_index = starts[split_name] if starts[split_name] is not None else end_index - len(buffer) + 1
        InMemoryDataset.save(buffer, str(tmp_path))
        os.replace(tmp_path, final_path)
        logger.info(
            "Wrote %s chunk=%s samples=%s [%s..%s]",
            split_name,
            chunk_id,
            len(buffer),
            start_index,
            end_index,
        )
        state["chunks"].append(
            {
                "path": f"{split_name}/chunks/{chunk_name}",
                "length": len(buffer),
                "start_index": int(start_index),
                "end_index": int(end_index),
            }
        )
        state["next_chunk_id"] = chunk_id + 1
        state["next_sample_index"] = end_index + 1
        state["count"] = int(state["count"]) + len(buffer)
        save_manifest()
        buffer.clear()
        starts[split_name] = None

    for group_idx in range(num_groups):
        group_path = _group_dir(raw_dir, release, case_name, group_idx)
        if not _group_complete(group_path):
            raise RuntimeError(f"Incomplete extracted group directory: {group_path}")

        for name in sorted((p.name for p in group_path.iterdir() if p.is_file() and p.suffix == ".json"), key=_parse_sample_index):
            sample_index = _global_sample_index(group_idx, _parse_sample_index(name))
            if sample_index < manifest["splits"]["train"]["next_sample_index"]:
                continue

            if sample_index < train_limit:
                split_name = "train"
            elif sample_index < val_limit:
                split_name = "val"
            else:
                split_name = "test"

            state = manifest["splits"][split_name]
            if sample_index < int(state["next_sample_index"]):
                continue

            if starts[split_name] is None:
                starts[split_name] = sample_index

            with (group_path / name).open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
            buffers[split_name].append(_build_heterodata(obj))

            if len(buffers[split_name]) >= chunk_size:
                flush(split_name, sample_index)
                gc.collect()

    for split_name in ("train", "val", "test"):
        if buffers[split_name]:
            last_end = int(manifest["splits"][split_name]["next_sample_index"]) + len(buffers[split_name]) - 1
            flush(split_name, last_end)

    save_manifest()
    logger.info(
        "Finished cache build for case=%s split_counts=%s",
        case_name,
        {name: info["count"] for name, info in manifest["splits"].items()},
    )
    return manifest


def prepare_case_cache(
    *,
    root: Path,
    case_name: str,
    num_groups: int = DEFAULT_NUM_GROUPS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    topological_perturbations: bool = False,
    force_reload: bool = False,
) -> Dict[str, Any]:
    release = _release_name(topological_perturbations)
    with _case_cache_lock(root, release, case_name):
        return _prepare_case_cache_unlocked(
            root=root,
            case_name=case_name,
            num_groups=num_groups,
            chunk_size=chunk_size,
            topological_perturbations=topological_perturbations,
            force_reload=force_reload,
        )


@dataclass
class _ChunkRecord:
    path: str
    length: int
    start_index: int
    end_index: int


class OPFCaseSplitDataset(TorchDataset):
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        case_name: str,
        split: str = "train",
        num_groups: int = DEFAULT_NUM_GROUPS,
        chunk_size: Optional[int] = None,
        topological_perturbations: bool = False,
        force_reload: bool = False,
    ) -> None:
        if case_name not in VALID_CASES:
            raise ValueError(f"Invalid case_name: {case_name}. Must be one of {VALID_CASES}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.root = Path(root).expanduser() if root is not None else _default_data_root()
        self.case_name = case_name
        self.split = split
        self.num_groups = int(num_groups)
        self.chunk_size = int(chunk_size or _default_chunk_size())
        self.topological_perturbations = topological_perturbations
        self.force_reload = force_reload
        self._release = _release_name(topological_perturbations)
        self._raw_dir = self.root / self._release / self.case_name / "raw"
        self._processed_dir = self.root / self._release / self.case_name / f"processed_{self.num_groups}"
        self._manifest_path = self._processed_dir / "manifest.json"
        self._manifest: Dict[str, Any] = {}
        self._chunk_cache_size = _default_chunk_cache_size()
        self._chunk_cache: "OrderedDict[int, tuple[Any, Any, type[Any], int]]" = OrderedDict()
        self._chunk_starts: list[int] = []
        self._chunk_ends: list[int] = []

        self._ensure_ready()

    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @property
    def processed_dir(self) -> Path:
        return self._processed_dir

    def _ensure_ready(self) -> None:
        if self.force_reload:
            self._manifest = prepare_case_cache(
                root=self.root,
                case_name=self.case_name,
                num_groups=self.num_groups,
                chunk_size=self.chunk_size,
                topological_perturbations=self.topological_perturbations,
                force_reload=True,
            )
            self._refresh_chunk_index()
            return
        if self._manifest_path.exists():
            with self._manifest_path.open("r", encoding="utf-8") as fh:
                self._manifest = json.load(fh)
            if not self._manifest_matches():
                self._manifest = prepare_case_cache(
                    root=self.root,
                    case_name=self.case_name,
                    num_groups=self.num_groups,
                    chunk_size=self.chunk_size,
                    topological_perturbations=self.topological_perturbations,
                    force_reload=True,
                )
            else:
                self._validate_manifest()
                self._refresh_chunk_index()
        else:
            self._manifest = prepare_case_cache(
                root=self.root,
                case_name=self.case_name,
                num_groups=self.num_groups,
                chunk_size=self.chunk_size,
                topological_perturbations=self.topological_perturbations,
                force_reload=self.force_reload,
            )
            self._refresh_chunk_index()

    def _manifest_matches(self) -> bool:
        return (
            self._manifest.get("version") == CACHE_FORMAT_VERSION
            and self._manifest.get("case_name") == self.case_name
            and int(self._manifest.get("num_groups", -1)) == self.num_groups
            and int(self._manifest.get("chunk_size", -1)) == self.chunk_size
            and self._manifest.get("release") == self._release
        )

    def _validate_manifest(self) -> None:
        split_info = self._manifest.get("splits", {}).get(self.split, {})
        for chunk in split_info.get("chunks", []):
            if not (self._processed_dir / chunk["path"]).exists():
                self._manifest = prepare_case_cache(
                    root=self.root,
                    case_name=self.case_name,
                    num_groups=self.num_groups,
                    chunk_size=self.chunk_size,
                    topological_perturbations=self.topological_perturbations,
                    force_reload=True,
                )
                self._refresh_chunk_index()
                return

    def _refresh_chunk_index(self) -> None:
        split_info = self._manifest.get("splits", {}).get(self.split, {})
        starts = [int(chunk["start_index"]) for chunk in split_info.get("chunks", [])]
        ends = [int(chunk["end_index"]) for chunk in split_info.get("chunks", [])]
        self._split_offset = starts[0] if starts else 0
        self._chunk_starts = [start - self._split_offset for start in starts]
        self._chunk_ends = [end - self._split_offset for end in ends]

    def _split_bounds(self) -> tuple[int, int, int]:
        train_limit = int(15_000 * self.num_groups * 0.9)
        val_limit = train_limit + int(DEFAULT_VAL_SAMPLES_PER_GROUP * self.num_groups)
        return train_limit, val_limit, int(15_000 * self.num_groups)

    def _split_for_index(self, sample_index: int) -> str:
        train_limit, val_limit, _ = self._split_bounds()
        if sample_index < train_limit:
            return "train"
        if sample_index < val_limit:
            return "val"
        return "test"

    def _split_state(self) -> dict[str, Any]:
        splits = self._manifest.setdefault("splits", {})
        state = splits.setdefault(
            self.split,
            {
                "chunks": [],
                "next_chunk_id": 0,
                "next_sample_index": 0,
                "count": 0,
            },
        )
        return state

    def _save_manifest(self) -> None:
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._manifest_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, self._manifest_path)

    def _flush_buffer(
        self,
        *,
        buffer: list[HeteroData],
        split_state: dict[str, Any],
        chunk_start_index: int,
        chunk_end_index: int,
    ) -> None:
        if not buffer:
            return

        split_dir = self._processed_dir / self.split / "chunks"
        split_dir.mkdir(parents=True, exist_ok=True)
        chunk_id = int(split_state["next_chunk_id"])
        final_name = f"chunk_{chunk_id:06d}.pt"
        final_path = split_dir / final_name
        tmp_path = split_dir / f"{final_name}.tmp"

        InMemoryDataset.save(buffer, str(tmp_path))
        os.replace(tmp_path, final_path)

        split_state["chunks"].append(
            {
                "path": f"{self.split}/chunks/{final_name}",
                "length": len(buffer),
                "start_index": chunk_start_index,
                "end_index": chunk_end_index,
            }
        )
        split_state["next_chunk_id"] = chunk_id + 1
        split_state["next_sample_index"] = chunk_end_index + 1
        split_state["count"] = int(split_state["count"]) + len(buffer)
        self._save_manifest()
        buffer.clear()

    def prepare(self) -> None:
        self._manifest = prepare_case_cache(
            root=self.root,
            case_name=self.case_name,
            num_groups=self.num_groups,
            chunk_size=self.chunk_size,
            topological_perturbations=self.topological_perturbations,
            force_reload=True,
        )

    def _load_chunk(self, chunk_index: int) -> tuple[Any, Any, type[Any], int]:
        if chunk_index in self._chunk_cache:
            self._chunk_cache.move_to_end(chunk_index)
            return self._chunk_cache[chunk_index]

        split_info = self._manifest["splits"][self.split]
        chunk_info = split_info["chunks"][chunk_index]
        chunk_path = self._processed_dir / chunk_info["path"]
        out = fs.torch_load(str(chunk_path))
        if len(out) == 2:
            data, slices = out
            data_cls = HeteroData
        else:
            data, slices, data_cls = out
        if isinstance(data, dict):
            data = data_cls.from_dict(data)
        length = int(chunk_info["length"])
        self._chunk_cache[chunk_index] = (data, slices, data_cls, length)
        self._chunk_cache.move_to_end(chunk_index)
        while len(self._chunk_cache) > self._chunk_cache_size:
            self._chunk_cache.popitem(last=False)
            gc.collect()
        return self._chunk_cache[chunk_index]

    def __len__(self) -> int:
        return int(self._manifest.get("splits", {}).get(self.split, {}).get("count", 0))

    def __getitem__(self, idx: int) -> HeteroData:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if not self._chunk_ends:
            raise IndexError(idx)

        chunk_index = bisect_left(self._chunk_ends, idx)
        data, slices, data_cls, _ = self._load_chunk(chunk_index)
        local_idx = idx - self._chunk_starts[chunk_index]
        return separate(
            cls=data_cls,
            batch=data,
            idx=local_idx,
            slice_dict=slices,
            decrement=False,
        )

    def __iter__(self) -> Iterator[HeteroData]:
        for idx in range(len(self)):
            yield self[idx]


def ensure_case_dataset(
    case_name: str,
    data_root: Path,
    splits: Iterable[str],
    *,
    chunk_size: Optional[int] = None,
    num_groups: int = DEFAULT_NUM_GROUPS,
    fresh: bool = False,
    topological_perturbations: bool = False,
) -> None:
    prepare_case_cache(
        root=data_root,
        case_name=case_name,
        num_groups=num_groups,
        chunk_size=int(chunk_size or _default_chunk_size()),
        topological_perturbations=topological_perturbations,
        force_reload=fresh,
    )


def load_case_split(
    *,
    case_name: str,
    split: str,
    data_root: Optional[Path] = None,
    chunk_size: Optional[int] = None,
    num_groups: int = DEFAULT_NUM_GROUPS,
    topological_perturbations: bool = False,
    force_reload: bool = False,
) -> OPFCaseSplitDataset:
    return OPFCaseSplitDataset(
        root=data_root or _default_data_root(),
        case_name=case_name,
        split=split,
        num_groups=num_groups,
        chunk_size=chunk_size,
        topological_perturbations=topological_perturbations,
        force_reload=force_reload,
    )


def load_case_sample(
    *,
    case_name: str,
    data_root: Optional[Path] = None,
    chunk_size: Optional[int] = None,
    num_groups: int = DEFAULT_NUM_GROUPS,
    topological_perturbations: bool = False,
) -> HeteroData:
    return load_case_split(
        case_name=case_name,
        split="train",
        data_root=data_root,
        chunk_size=chunk_size,
        num_groups=num_groups,
        topological_perturbations=topological_perturbations,
    )[0]
