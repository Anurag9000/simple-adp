"""Explicitly network-enabled official Fashion-MNIST paired V01 smoke.

This is only a bounded one-seed experiment; it does not report a benchmark
winner. All three jobs share the exact official data indices, pretrained
torchvision ResNet18 weights and preprocessing. External source provenance is
independently recorded rather than inferred by the local weight loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from simple_adp.config import SearchSettings
from simple_adp.vision_comparison import run_vision_comparison


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official source matched V01 experiment")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-limit", type=int, default=300)
    parser.add_argument("--valid-limit", type=int, default=100)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--seeds", default="11")
    args = parser.parse_args(argv)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    source_enum = ResNet18_Weights.IMAGENET1K_V1
    source = resnet18(weights=source_enum).eval()
    source_weights = workdir / "resnet18_official_full_state_dict.pt"
    torch.save(source.state_dict(), source_weights)
    source_sha256 = sha256_file(source_weights)

    comparison_dir = workdir / "comparison" / "fashion_mnist" / "resnet18_v01"
    seeds = tuple(int(s) for s in args.seeds.split(","))
    summary = run_vision_comparison(
        dataset="fashion_mnist",
        root=workdir / "official_data",
        output_dir=comparison_dir,
        architecture_id="resnet18",
        weights_path=source_weights,
        expected_weights_sha256=source_sha256,
        allow_random_backbone=False,
        download=True, seeds=seeds,
        settings=SearchSettings(
            device=args.device, seed=0,
            init_width=8, init_depth=1,
            max_width=10, max_depth=2, max_neurons=128,
            max_epochs=2, max_candidates=1,
            batch_size=32, patience_es=2,
        ),
        fixed_width=8, train_limit=args.train_limit,
        validation_limit=args.valid_limit,
        test_limit=args.test_limit,
        encoder_batch_size=32,
    )
    assert summary["status"] == "completed"
    assert summary["completed_jobs"] == 3 * len(seeds)
    for seed in seeds:
        rows = [x for x in summary["rows"] if x["seed"] == seed]
        assert [x["route"] for x in rows] == [
            "fixed", "adp_loss", "adp_variance",
        ]
        assert len({x["official_split_sha256"] for x in rows}) == 1
        assert len({x["source_encoder_state_sha256"] for x in rows}) == 1
    external = {
        "format": "simple-adp-real-paired-pretraining-v1",
        "dataset": "torchvision.datasets.FashionMNIST official",
        "source_model": "torchvision.models.resnet18",
        "pretrained_weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "pretrained_weights_url": source_enum.url,
        "complete_source_state_dict_sha256": source_sha256,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "comparison_signature": summary["comparison_signature"],
        "real_data": True,
        "benchmark_quality": False,
        "note": "Small matched-source smoke; official test metrics are descriptive, not method selection",
    }
    (comparison_dir / "external_source_provenance.json").write_text(
        json.dumps(external, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"provenance": external, "summary": summary},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
