from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_adp_guide_block_crosswalk_covers_all_indexed_blocks():
    repository = Path(__file__).parents[1]
    crosswalk = repository / "docs" / "ADP_ALGORITHM_BLOCK_CROSSWALK.md"

    subprocess.run(
        [sys.executable, "scripts/generate_adp_block_crosswalk.py", "--check"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    text = crosswalk.read_text(encoding="utf-8")
    assert text.count("<summary>Source text</summary>") == 458
    assert "### Block 000" in text
    assert "### Block 457" in text
