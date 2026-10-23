import ast
import hashlib
import json
from pathlib import Path


def test_copied_opf_runner_reference_matches_its_46_file_manifest():
    repository = Path(__file__).parents[1]
    snapshot = repository / "references" / "opf_adp_runner_suite"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["files"]

    assert len(entries) == 46
    listed_paths = set()
    for entry in entries:
        relative = entry["path"]
        assert relative not in listed_paths
        listed_paths.add(relative)
        path = snapshot / relative
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        assert len(content) == entry["bytes"], relative
        assert digest == entry["copied_sha256"] == entry["source_sha256"], relative

    actual_paths = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    assert actual_paths == listed_paths | {"README.md", "manifest.json"}


def test_runtime_package_does_not_depend_on_copied_opf_runner_modules():
    repository = Path(__file__).parents[1]
    source_root = repository / "src" / "simple_adp"
    forbidden_module = "opf_adp_runner_suite"
    forbidden_path = "references/opf_adp_runner_suite"

    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                assert all(forbidden_module not in name for name in names), path
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert forbidden_module not in module, path
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert forbidden_path not in node.value, path
