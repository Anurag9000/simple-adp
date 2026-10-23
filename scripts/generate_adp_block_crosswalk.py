#!/usr/bin/env python3
"""Generate a numbered index of every nonempty paragraph/table row in the ADP guide."""

from __future__ import annotations

import argparse
import html
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "ADP_algorithms.docx"
OUTPUT = ROOT / "docs" / "ADP_ALGORITHM_BLOCK_CROSSWALK.md"
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

RANGES = (
    (0, 66, "Taxonomy, notation, and controller definitions", "registry.py; config.py; dedicated.py; generic_search.py", "taxonomy and notation"),
    (67, 144, "Shared architecture and all-layer/round-robin width DEN", "models.py; growth.py; training.py; dedicated.py", "shared mechanics and width DEN"),
    (145, 207, "Depth growth, nested DEN, and LG-ND", "growth.py; dedicated.py; search.py", "depth, nested, and LG-ND"),
    (208, 262, "Generic candidate search and mutation primitives", "generic_search.py; growth.py; training.py; artifacts.py", "generic search and mutations"),
    (263, 457, "Checkpoint recovery, traces, reproducibility, taxonomy, and appendices", "checkpoint.py; estimator.py; training.py; dedicated.py; generic_search.py; registry.py", "checkpoint and taxonomy"),
)


def _element_text(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def extract_blocks(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    body = document.find("w:body", NS)
    if body is None:
        raise ValueError("DOCX has no document body")
    blocks: list[str] = []
    for child in body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = _element_text(child)
            if text:
                blocks.append(text)
        elif tag == "tbl":
            for row in child.findall("w:tr", NS):
                cells = [_element_text(cell) for cell in row.findall("w:tc", NS)]
                text = " | ".join(cells).strip()
                if text:
                    blocks.append(text)
    return blocks


def render(blocks: list[str]) -> str:
    if len(blocks) != 458:
        raise ValueError(f"Expected 458 source blocks, found {len(blocks)}")
    lines = [
        "# ADP guide block crosswalk",
        "",
        "This generated index preserves the source document's body order: each",
        "nonempty paragraph or table row is assigned a stable zero-based block ID.",
        "It indexes the paragraph-level review recorded in",
        "[the ADP implementation audit](ADP_ALGORITHM_IMPLEMENTATION_AUDIT.md);",
        "it is not a claim of testing every possible numeric setting, custom",
        "callback, optimizer, runtime, or interruption point.",
        "",
        "| Block IDs | Review area | Implementation sources | Audit record |",
        "|---:|---|---|---|",
    ]
    for start, end, area, sources, audit in RANGES:
        lines.append(f"| {start:03d}–{end:03d} | {area} | `{sources}` | {audit} |")
    lines.extend(["", "## Indexed source blocks", ""])
    for index, block in enumerate(blocks):
        start, end, area, sources, audit = next(item for item in RANGES if item[0] <= index <= item[1])
        lines.extend([
            f"### Block {index:03d}",
            "",
            f"**Review area:** {area}  ",
            f"**Code crosswalk:** `{sources}`  ",
            f"**Audit section:** {audit}",
            "",
            "<details>",
            "<summary>Source text</summary>",
            "",
            "<pre>",
            html.escape(block),
            "</pre>",
            "</details>",
            "",
        ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated output is stale")
    args = parser.parse_args()
    content = render(extract_blocks(SOURCE))
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            print(f"stale generated crosswalk: {OUTPUT}", file=sys.stderr)
            return 1
        print("ADP guide block crosswalk is current")
        return 0
    OUTPUT.write_text(content, encoding="utf-8")
    print(f"updated {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
