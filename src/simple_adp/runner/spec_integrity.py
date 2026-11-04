"""Race-safe, fail-closed publication of immutable ADP worker run specifications.

This module deliberately has no import from the suite planner, so both native
and visual planner paths can use it without circular imports. Source and
variance arms must never reuse a file with different effective settings.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def publish_immutable_run_spec(path: str | Path, document: str) -> None:
    """Publish *document* once, or verify an existing logically identical spec.

    The temporary file is durable before a hard-link publishes it without
    replacement. Competing writers cannot silently overwrite one another;
    the loser checks the winner's complete document. On filesystems without
    hard-link support, fail closed rather than reverting to unsafe replace.
    """
    path = Path(path)
    try:
        expected = json.loads(document)
    except (TypeError, ValueError) as exc:
        raise ValueError("run specification payload is not valid JSON") from exc
    if not isinstance(expected, dict):
        raise ValueError("run specification payload must be a JSON object")

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def check_published() -> bool:
        if path.is_symlink():
            raise ValueError(f"run specification path must not be a symlink: {path}")
        if not path.exists():
            return False
        if not path.is_file():
            raise ValueError(f"run specification is not a regular file: {path}")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise ValueError(f"existing run specification is unreadable or corrupt: {path}") from exc
        if canonical(existing) != canonical(expected):
            raise ValueError(f"existing run specification differs from planned science: {path}")
        return True

    if check_published():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # The original runner used `os.replace`, which could silently replace a
    # competing worker's different job spec after a check-then-write race.
    # A hard link atomically publishes only if the final path is absent.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".run-spec-", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        if not check_published():
            raise RuntimeError(f"run specification was not published: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
