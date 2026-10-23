from __future__ import annotations

import json
import tarfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RESUME_ROOT = REPO_ROOT / "resume_snapshots"
BUNDLE_PART_SIZE_BYTES = 64 * 1024 * 1024


def _safe_name(path: Path) -> str:
    return "__".join(path.parts)


def bundle_path_for_results_dir(results_dir: Path) -> Path:
    resolved = results_dir.resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        rel = Path(*resolved.parts[1:]) if resolved.is_absolute() else resolved
    return RESUME_ROOT / f"{_safe_name(rel)}.tar.gz"


def manifest_path_for_bundle(bundle_path: Path) -> Path:
    return bundle_path.with_suffix("").with_suffix(".json")


def _bundle_part_paths(bundle_path: Path) -> list[Path]:
    return sorted(bundle_path.parent.glob(f"{bundle_path.name}.part*"))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_manifest(results_dir: Path, bundle_path: Path, *, kind: str = "results") -> Dict[str, Any]:
    return {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(results_dir),
        "bundle_path": str(bundle_path),
        "bundle_mode": "chunked",
        "part_size_bytes": BUNDLE_PART_SIZE_BYTES,
        "source_exists": results_dir.exists(),
        "source_size_bytes": _dir_size(results_dir),
    }


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except Exception:
            continue
    return total


def _bundle_filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
    if tarinfo.name.endswith("/telemetry.csv") or tarinfo.name == "telemetry.csv":
        return None
    return tarinfo


def _safe_extract_tar(tar: tarfile.TarFile, destination: Path, expected_root: str) -> None:
    """Extract a generated bundle without allowing path traversal or links."""
    destination = destination.resolve()
    root_name = str(expected_root).strip()
    if not root_name or Path(root_name).name != root_name:
        raise ValueError(f"Invalid bundle root: {expected_root!r}")

    members = tar.getmembers()
    for member in members:
        normalized_name = str(member.name).replace("\\", "/")
        member_path = PurePosixPath(normalized_name)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe bundle member path: {member.name!r}")
        if not member_path.parts or member_path.parts[0] != root_name:
            raise ValueError(f"Unexpected bundle root: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"Links are not permitted in result bundles: {member.name!r}")
        target = (destination / Path(*member_path.parts)).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"Unsafe bundle member path: {member.name!r}") from exc

    tar.extractall(path=destination)


def export_results_bundle(results_dir: Path, bundle_path: Optional[Path] = None) -> Optional[Path]:
    results_dir = results_dir.resolve()
    if not results_dir.exists():
        return None
    bundle_path = bundle_path or bundle_path_for_results_dir(results_dir)
    _ensure_dir(bundle_path.parent)
    manifest = build_manifest(results_dir, bundle_path)
    manifest["source_dir"] = str(results_dir)
    _write_json(manifest_path_for_bundle(bundle_path), manifest)

    tmp_path = bundle_path.with_suffix(bundle_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    for existing_part in _bundle_part_paths(bundle_path):
        existing_part.unlink()
    if bundle_path.exists():
        bundle_path.unlink()

    with tarfile.open(tmp_path, mode="w:gz", compresslevel=1) as tar:
        tar.add(results_dir, arcname=results_dir.name, recursive=True, filter=_bundle_filter)

    if tmp_path.stat().st_size <= BUNDLE_PART_SIZE_BYTES:
        tmp_path.replace(bundle_path)
        return bundle_path

    part_index = 0
    with tmp_path.open("rb") as src:
        while True:
            chunk = src.read(BUNDLE_PART_SIZE_BYTES)
            if not chunk:
                break
            part_path = bundle_path.with_suffix(bundle_path.suffix + f".part{part_index:03d}")
            with part_path.open("wb") as dst:
                dst.write(chunk)
            part_index += 1

    tmp_path.unlink()
    return bundle_path


def restore_results_bundle(results_dir: Path, bundle_path: Optional[Path] = None) -> bool:
    results_dir = results_dir.resolve()
    bundle_path = bundle_path or bundle_path_for_results_dir(results_dir)
    if results_dir.exists():
        try:
            next(results_dir.iterdir())
            return False
        except StopIteration:
            pass
        except Exception:
            return False
    if not bundle_path.exists():
        parts = _bundle_part_paths(bundle_path)
        if not parts:
            return False
        _ensure_dir(results_dir.parent)
        tmp_path = bundle_path.with_suffix(bundle_path.suffix + ".restore")
        with tmp_path.open("wb") as dst:
            for part_path in parts:
                with part_path.open("rb") as src:
                    while True:
                        chunk = src.read(BUNDLE_PART_SIZE_BYTES)
                        if not chunk:
                            break
                        dst.write(chunk)
        with tarfile.open(tmp_path, mode="r:gz") as tar:
            _safe_extract_tar(tar, results_dir.parent, results_dir.name)
        tmp_path.unlink()
        return True
    _ensure_dir(results_dir.parent)
    with tarfile.open(bundle_path, mode="r:gz") as tar:
        _safe_extract_tar(tar, results_dir.parent, results_dir.name)
    return True


def _manifest_target_path(manifest_path: Path, original_path: str) -> Path:
    target = Path(original_path)
    if target.is_absolute():
        resolved = target.resolve()
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"Sharded telemetry target escapes repository: {original_path!r}") from exc
        return resolved
    repo_root = REPO_ROOT.resolve()
    resolved = (repo_root / target).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Sharded telemetry target escapes repository: {original_path!r}") from exc
    return resolved


def _shard_manifest_paths(root: Path):
    for manifest_path in root.rglob("*.manifest.json"):
        if ".csv.manifest.json" in manifest_path.name:
            yield manifest_path


def rehydrate_sharded_telemetry(root: Path) -> int:
    restored = 0
    root = root.resolve()
    for manifest_path in _shard_manifest_paths(root):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        original_path = payload.get("original_path")
        parts = payload.get("parts") or []
        if not original_path or not parts:
            continue
        target_path = _manifest_target_path(manifest_path, str(original_path))
        if target_path.exists():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(target_path.suffix + ".rehydrate")
        digest = hashlib.sha256()
        with tmp_path.open("wb") as dst:
            for part_name in parts:
                part_path = (manifest_path.parent / str(part_name)).resolve()
                try:
                    part_path.relative_to(manifest_path.parent.resolve())
                except ValueError as exc:
                    raise ValueError(f"Telemetry shard escapes manifest directory: {part_name!r}") from exc
                if not part_path.exists():
                    raise FileNotFoundError(part_path)
                with part_path.open("rb") as src:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        dst.write(chunk)
        expected_sha256 = payload.get("sha256")
        if expected_sha256 and digest.hexdigest() != expected_sha256:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Checksum mismatch for {target_path}")
        tmp_path.replace(target_path)
        restored += 1
    return restored
