"""Official-image matrix CLI validates CIFAR class coverage before downloads."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _matrix_main():
    file = Path(__file__).resolve().parents[1] / "scripts" / "visual_official_fashion_matrix.py"
    spec = importlib.util.spec_from_file_location("official_visual_matrix_cli", file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


@pytest.mark.parametrize(
    ("dataset", "train", "valid", "test", "minimum"),
    [
        ("fashion_mnist", 9, 20, 20, 10),
        ("cifar10", 40, 9, 20, 10),
        ("cifar100", 99, 100, 100, 100),
        ("cifar100", 100, 99, 100, 100),
        ("cifar100", 100, 100, 99, 100),
    ],
)
def test_class_coverage_guard_runs_before_any_official_download(
    dataset, train, valid, test, minimum, tmp_path, capsys,
):
    main = _matrix_main()
    with pytest.raises(SystemExit) as exc:
        main([
            "--dataset", dataset, "--workdir", str(tmp_path),
            "--train-limit", str(train), "--valid-limit", str(valid),
            "--test-limit", str(test),
        ])
    assert exc.value.code == 2
    assert f"at least {minimum}" in capsys.readouterr().err
    assert not (tmp_path / "dataset").exists()


def test_rejects_unknown_dataset_before_download(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _matrix_main()([
            "--dataset", "not-a-dataset", "--workdir", str(tmp_path)
        ])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
