"""Bounded network-enabled *real* Fashion-MNIST + official ResNet18 V01 smoke.

Not run by offline unit tests. Must be explicitly invoked through the
dedicated GitHub Actions workflow or CLI. Published weights are acquired
from the torchvision enum and then passed as a complete strict source
state_dict to the local-only experiment API. Never relabel arbitrary local
weights as pretrained.
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
from simple_adp.vision_experiment import run_vision_experiment


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Network-enabled official vision data smoke")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--dataset", choices=("fashion_mnist", "cifar10", "cifar100"), default="fashion_mnist")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    weights = ResNet18_Weights.IMAGENET1K_V1
    # torchvision is the declared official pretrained checkpoint provider;
    # this call may download and caches only on the CI runner.
    source = resnet18(weights=weights).eval()
    source_path = workdir / "resnet18_full_official_source.pt"
    torch.save(source.state_dict(), source_path)
    source_digest = file_sha256(source_path)

    result_dir = workdir / args.dataset / "resnet18" / "ADP-W-RR" / "seed11"
    result = run_vision_experiment(
        dataset=args.dataset,
        root=workdir / "official_datasets",
        output_dir=result_dir,
        architecture_id="resnet18",
        model_id="ADP-W-RR",
        weights_path=source_path,
        expected_weights_sha256=source_digest,
        allow_random_backbone=False,
        download=True,  # Explicitly requested only by this network smoke.
        settings=SearchSettings(
            device=args.device, seed=11,
            init_width=8, init_depth=1, max_width=9, max_depth=2,
            max_epochs=1, max_candidates=1, batch_size=16,
            max_neurons=128, patience_es=1,
        ),
        train_limit=40, validation_limit=20, test_limit=20,
        split_seed=11, encoder_seed=0, encoder_batch_size=16,
    )
    manifest = json.loads((result_dir / "manifest.json").read_text())
    assert result["status"] == "completed"
    assert result["source"] == "torchvision:" + args.dataset
    assert result["sample_counts"] == {
        "train": 40, "validation": 20, "official_test": 20,
    }
    assert result["weights_source"] == "unverified_local_state_dict"
    assert result["pretrained"] is None  # local loader never infers training origin
    assert manifest["source_encoder"]["weights_file_sha256"] == source_digest
    assert manifest["input_transform"]["raw_channels"] == (1 if args.dataset == "fashion_mnist" else 3)
    assert manifest["input_transform"]["output_resolution"] == [32, 32]
    assert manifest["partitions"]["source"] == "torchvision:" + args.dataset
    assert len(manifest["partitions"]["train_raw_sha256"]) == 64
    assert result["selected_architecture"]
    assert result["test_metrics"]["accuracy"] >= 0
    source_provenance = {
        "format": "simple-adp-official-pretraining-source-v1",
        "source_library": "torchvision",
        "source_library_version": torchvision.__version__,
        "source_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "source_url": weights.url,
        "strict_loaded_source_model": "torchvision.models.resnet18",
        "complete_local_state_dict_sha256": source_digest,
        "source_dataset": "ImageNet-1K (as named by torchvision enum)",
        "model_local_loader_pretrained_field": None,
        "data_source": "torchvision.datasets." + ("FashionMNIST" if args.dataset == "fashion_mnist" else "CIFAR10" if args.dataset == "cifar10" else "CIFAR100") + " official train/test",
        "network_enabled": True,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "run_signature": result["run_signature"],
        "result_file": str(result_dir / "result.json"),
        "scientific_scope": "tiny bounded real-data smoke; not a multi-seed benchmark",
        "dataset_name": args.dataset,
    }
    (result_dir / "external_source_provenance.json").write_text(
        json.dumps(source_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "test_metrics": result["test_metrics"],
        "validation_metrics": result["validation_metrics"],
        "source_provenance": source_provenance,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
