"""Network-enabled real Fashion-MNIST + published ViT-B/16 source FFN smoke.

Strict complete torchvision weights, exact first ViT FFN transplant and
bounded native width-growth. This micro-run is *not* a scientific benchmark.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import ViT_B_16_Weights, vit_b_16

from simple_adp.config import SearchSettings
from simple_adp.vision_data import load_torchvision_partitions
from simple_adp.vision_experiment import _atomic_json
from simple_adp.visual_search import VisualResidualSearch
from simple_adp.visual_source_convnext import file_sha256
from simple_adp.visual_source_vit import SourceViTSpec


def main() -> int:
    parser = argparse.ArgumentParser(description="Published pretrained ViT V03 FFN real-image smoke")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    source = vit_b_16(weights=weights).eval()  # explicit network download
    source_file = workdir / "vit_b16_official_full_source.pt"
    torch.save(source.state_dict(), source_file)
    source_sha = file_sha256(source_file)
    del source

    parts = load_torchvision_partitions(
        "fashion_mnist", workdir / "official_data",
        download=True, seed=11, train_limit=20,
        validation_limit=10, test_limit=10,
    )
    train_x, train_y = parts.materialize("train")
    valid_x, valid_y = parts.materialize("validation")
    target = workdir / "experiments" / "fashion_mnist" / "V03-source" / "ADP-W-RR" / "seed11"
    target.mkdir(parents=True, exist_ok=True)
    spec = SourceViTSpec(
        classes=int(len(np.unique(train_y))),
        image_height=int(train_x.shape[2]),
        image_width=int(train_x.shape[3]),
        input_channels=int(train_x.shape[1]),
        weights_path=str(source_file),
        weights_sha256=source_sha,
        fixed_seed=11,
    )
    manifest = {
        "format": "simple-adp-pretrained-vit-source-smoke-v1",
        "official_dataset": parts.manifest,
        "torchvision_architecture": "vit_b_16",
        "published_pretrained_enum": "ViT_B_16_Weights.IMAGENET1K_V1",
        "published_weights_url": weights.url,
        "complete_local_source_sha256": source_sha,
        "source_site": spec.site_id,
        "source_hidden_width": 3072,
        "target_head_classes": spec.classes,
        "preprocessing": "grayscale repeat, deterministic bilinear 224x224, ImageNet mean/std",
        "split_policy": "train/validation source only; official test measured after fit",
        "scope": "tiny one-seed source integration only; no benchmark conclusions",
    }
    _atomic_json(target / "manifest.json", manifest)

    search = VisualResidualSearch(
        "ADP-W-RR", spec,
        settings=SearchSettings(
            init_width=3072, init_depth=1, max_width=3073,
            max_depth=2, max_neurons=5000, max_candidates=1,
            max_epochs=1, batch_size=5, patience_es=1,
            patience_width_exp=1, device=args.device, seed=11,
        ),
    )
    search.fit(
        train_x, train_y, validation_data=(valid_x, valid_y),
        checkpoint_path=target / "estimator.checkpoint.pt",
    )
    search.model_.verify_frozen()
    assert search.model_.source_ffn_exact_at_initialization
    valid_predictions = search.predict(valid_x)
    valid_probabilities = search.predict_proba(valid_x)
    valid_metrics = search.task.evaluation_metrics(
        valid_y, valid_predictions, valid_probabilities,
    )
    # Only after architecture search is completed.
    test_x, test_y = parts.materialize("test")
    predictions = search.predict(test_x)
    probabilities = search.predict_proba(test_x)
    test_metrics = search.task.evaluation_metrics(
        test_y, predictions, probabilities,
    )
    history = {
        "format": "simple-adp-v03-source-history-v1",
        "method": "ADP-W-RR", "entries": search.history_,
        "selected_architecture": search.model_.hidden_widths,
        "source_initialization_exact": search.model_.source_ffn_exact_at_initialization,
    }
    _atomic_json(target / "search_history.json", history)
    out_file = target / "selected_source_vit_model.pt"
    torch.save(search.model_.snapshot(cursor=search.model_._width_cursor), out_file)
    result = {
        "format": "simple-adp-pretrained-vit-source-result-v1",
        "status": "completed",
        "dataset": "torchvision:FashionMNIST",
        "method": "ADP-W-RR",
        "source": "torchvision.models.vit_b_16",
        "pretraining_enum": "ViT_B_16_Weights.IMAGENET1K_V1",
        "pretrained_weights_url": weights.url,
        "source_weights_sha256": source_sha,
        "frozen_source_sha256": search.model_.frozen_state_sha256(),
        "source_site": spec.site_id,
        "source_initialization_exact": search.model_.source_ffn_exact_at_initialization,
        "selected_architecture": search.model_.hidden_widths,
        "history_entries": len(search.history_),
        "validation_metrics": valid_metrics,
        "official_test_metrics": test_metrics,
        "sample_counts": {
            "train": len(train_y), "validation": len(valid_y), "official_test": len(test_y),
        },
        "model_artifact": out_file.name,
        "model_file_sha256": file_sha256(out_file),
        "search_history_file_sha256": file_sha256(target / "search_history.json"),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "scope": "one-seed 20/10/10 tiny real-data smoke, one epoch/stage; no benchmark claims",
    }
    _atomic_json(target / "result.json", result)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
