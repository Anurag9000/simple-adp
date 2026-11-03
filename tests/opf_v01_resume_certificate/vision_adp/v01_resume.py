"""Fail-closed, atomic V01 optimizer-step checkpoint storage and RNG snapshots.

Checkpointing is opt-in. It is not a replacement for completion receipts.
Exact batch cursor replay is currently supported only with num_workers=0;
CPU training has a bitwise-resume contract in the accompanying tests.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch


SCHEMA = "vision-adp-v01-optimizer-step-resume/v1"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng() -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else []
        ),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((
        value["name"], np.asarray(value["keys"], dtype=np.uint32),
        int(value["pos"]), int(value["has_gauss"]),
        float(value["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint needs CUDA RNG but the device is unavailable")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA generator count differs")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def loader_generator_state(loader):
    generator = getattr(loader, "generator", None)
    return None if generator is None else generator.get_state().clone()


def restore_loader_generator(loader, state) -> None:
    generator = getattr(loader, "generator", None)
    if state is None:
        if generator is not None:
            raise ValueError("checkpoint lacks the train-loader generator")
        return
    if generator is None:
        raise ValueError("resume requires the saved train-loader generator")
    generator.set_state(state.clone())


class V01CheckpointStore:
    def __init__(self, output_dir: str | Path, science_id: str):
        output = Path(output_dir)
        if output.is_symlink():
            raise ValueError("checkpoint output root cannot be a symlink")
        self.root = output.parent / ("." + output.name + ".training_resume")
        self.payload = self.root / "checkpoint.pt"
        self.receipt = self.root / "checkpoint_receipt.json"
        self.previous_payload = self.root / "previous_checkpoint.pt"
        self.previous_receipt = self.root / "previous_checkpoint_receipt.json"
        if not isinstance(science_id, str) or len(science_id) != 64:
            raise ValueError("checkpoint science identity must be a SHA-256")
        self.science_id = science_id
        self.restored_generation: str | None = None

    def _validate_root(self):
        if self.root.is_symlink():
            raise ValueError("checkpoint directory cannot be a symlink")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("checkpoint root must be a directory")
        allowed = {
            self.payload.name, self.receipt.name,
            self.previous_payload.name, self.previous_receipt.name,
        }
        if self.root.exists():
            for item in self.root.iterdir():
                if item.is_symlink() or not item.is_file():
                    raise ValueError("checkpoint contains symlink or unsupported file type")
                if item.name not in allowed and not (
                    (item.name.startswith(".pending-") and item.name.endswith(".pt"))
                    or (item.name.startswith(".receipt-") and item.name.endswith(".tmp"))
                    or (item.name.startswith(".rollback-") and item.name.endswith(".tmp"))
                ):
                    raise ValueError(
                        "interrupted V01 checkpoint transaction has unexpected files: "
                        + item.name
                    )

    def _has_committed_files(self) -> bool:
        return any(path.exists() for path in (
            self.payload, self.receipt,
            self.previous_payload, self.previous_receipt,
        ))

    def _read_pair(self, payload: Path, receipt: Path) -> dict:
        if not payload.is_file() or not receipt.is_file():
            raise ValueError("incomplete V01 checkpoint pair")
        record = json.loads(receipt.read_text(encoding="utf-8"))
        if record.get("schema") != SCHEMA or record.get("science_id") != self.science_id:
            raise ValueError("V01 checkpoint science identity mismatch")
        if record.get("sha256") != _sha_file(payload):
            raise ValueError("V01 checkpoint checksum mismatch")
        saved = torch.load(payload, map_location="cpu", weights_only=True)
        if (
            not isinstance(saved, dict)
            or saved.get("schema") != SCHEMA
            or saved.get("science_id") != self.science_id
            or saved.get("kind") not in {"stage", "controller"}
            or saved["kind"] != record.get("kind")
        ):
            raise ValueError("invalid V01 training checkpoint payload")
        return saved

    def exists(self):
        self._validate_root()
        return self._has_committed_files()

    def load(self) -> dict | None:
        self._validate_root()
        self.restored_generation = None
        if not self._has_committed_files():
            if self.root.exists() and any(self.root.iterdir()):
                raise ValueError(
                    "interrupted V01 checkpoint transaction has unexpected files"
                )
            return None
        errors = []
        for generation, payload, receipt in (
            ("current", self.payload, self.receipt),
            ("previous", self.previous_payload, self.previous_receipt),
        ):
            try:
                saved = self._read_pair(payload, receipt)
                self.restored_generation = generation
                return saved
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                errors.append(str(exc))
            except Exception as exc:
                # torch.load(weights_only=True) may reject corrupt data with
                # UnpicklingError. The rollback is still checksum-validated.
                errors.append(type(exc).__name__)
        raise ValueError(
            "V01 checkpoint has no valid committed generation (checksum,"
            " identity or incomplete receipt): " + "; ".join(errors)
        )

    def _sync_directory(self):
        if os.name == "posix":
            directory = os.open(
                self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def _rollback_link(self, source: Path, target: Path) -> None:
        # A hard link is a constant-time, same-filesystem snapshot, not
        # a multi-gigabyte model copy. Replacing the active checkpoint
        # creates a fresh inode, leaving this committed inode intact.
        fd, tmp = tempfile.mkstemp(
            dir=self.root, prefix=".rollback-", suffix=".tmp"
        )
        os.close(fd)
        temporary = Path(tmp)
        try:
            temporary.unlink()
            os.link(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _clean_finished_temporaries(self) -> None:
        for item in self.root.iterdir():
            if item.name.startswith((".pending-", ".receipt-", ".rollback-")):
                if item.is_symlink() or not item.is_file():
                    raise ValueError("unsafe pending checkpoint file")
                item.unlink()

    def save(self, checkpoint: dict) -> None:
        self._validate_root()
        if (
            checkpoint.get("schema") != SCHEMA
            or checkpoint.get("science_id") != self.science_id
            or checkpoint.get("kind") not in {"stage", "controller"}
        ):
            raise ValueError("cannot publish incompatible V01 checkpoint")
        self.root.mkdir(parents=True, exist_ok=True)
        # If a prior interrupted publication damaged the active generation,
        # leave an intact rollback untouched until a newer generation commits.
        if self._has_committed_files():
            prior = self.load()
            assert prior is not None

        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".pending-", suffix=".pt")
        temp = Path(tmp)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            digest = _sha_file(temp)
            # First snapshot an intact active payload/receipt in the alternate
            # generation. Every step before replacing active payload leaves
            # active intact, and every step afterwards leaves rollback intact.
            if self.payload.exists() and self.receipt.exists():
                try:
                    self._read_pair(self.payload, self.receipt)
                except (OSError, ValueError, RuntimeError, TypeError, KeyError):
                    # The current active payload may be a failed interrupted
                    # write; load() already checked the rollback is valid.
                    pass
                else:
                    self._rollback_link(self.payload, self.previous_payload)
                    self._rollback_link(self.receipt, self.previous_receipt)
                    self._sync_directory()

            os.replace(temp, self.payload)
            record = {
                "schema": SCHEMA,
                "science_id": self.science_id,
                "sha256": digest,
                "kind": checkpoint["kind"],
            }
            with tempfile.NamedTemporaryFile(
                dir=self.root, prefix=".receipt-", suffix=".tmp",
                mode="w", encoding="utf-8", delete=False,
            ) as stream:
                receipt_tmp = Path(stream.name)
                json.dump(record, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(receipt_tmp, self.receipt)
            finally:
                receipt_tmp.unlink(missing_ok=True)
            self._sync_directory()
            self._clean_finished_temporaries()
        finally:
            temp.unlink(missing_ok=True)

    def clear(self) -> None:
        self._validate_root()
        if self.root.is_dir():
            for item in self.root.iterdir():
                if item.is_symlink() or not item.is_file():
                    raise ValueError("refusing to remove unsafe checkpoint data")
                item.unlink()
            self._sync_directory()


def science_checkpoint_identity(
    *, matched_digest: str, criterion: str, device: torch.device,
    source_digests: dict[str, str],
    library_versions: dict[str, str] | None = None,
) -> str:
    """Bind a checkpoint to data science AND numerical execution context.

    Changing installed dataset generators, Python/NumPy versions, CPU
    threading, or physical CUDA hardware can change replayed minibatches
    even when the V01 Python source and frozen backbone stay unchanged.
    """
    device = torch.device(device)
    gpu_execution = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(index)
        gpu_execution = {
            "logical_device_index": int(index),
            "name": str(prop.name),
            "capability": tuple(torch.cuda.get_device_capability(index)),
            "total_memory": int(prop.total_memory),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_runtime": str(torch.version.cuda),
            "cudnn_version": torch.backends.cudnn.version(),
        }
    payload = {
        "schema": SCHEMA,
        "matched_training_sha256": matched_digest,
        "criterion": criterion,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "python_runtime": [sys.version_info.major, sys.version_info.minor,
                           sys.version_info.micro],
        "numpy_version": str(np.__version__),
        "library_versions": library_versions or {},
        "gpu_execution": gpu_execution,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "source_sha256": source_digests,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
