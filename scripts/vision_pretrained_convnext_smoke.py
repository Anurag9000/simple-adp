"""Official pretrained ConvNeXt-Tiny inner-FFN V04 bounded real-data smoke.

This explicitly network-enabled CI script verifies full-image training of
the original first CNBlock source FFN after an exact weight transplant.
The official test labels are materialized only after fit; the limited sample
count is not benchmark-quality evidence.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from simple_adp.config import SearchSettings
from simple_adp.vision_data import load_torchvision_partitions
from simple_adp.visual_source_convnext import SourceConvNeXtSpec, file_sha256
from simple_adp.visual_search import VisualResidualSearch
from simple_adp.vision_experiment import _atomic_json


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="official pretrained V04 FFN full-image experiment")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = Path(args.workdir)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)

    source_enum = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    source = convnext_tiny(weights=source_enum).eval()
    source_file = root / "convnext_tiny_complete_source.pt"
    torch.save(source.state_dict(), source_file)
    sha = file_sha256(source_file)
    dataset = load_torchvision_partitions(
        "fashion_mnist", root / "official_dataset",
        download=True, seed=11,
        train_limit=40, validation_limit=20, test_limit=20,
    )
    x_fit, y_fit = dataset.materialize("train")
    x_valid, y_valid = dataset.materialize("validation")
    spec = SourceConvNeXtSpec(
        classes=int(len(np.unique(y_fit))),
        image_height=int(x_fit.shape[2]), image_width=int(x_fit.shape[3]),
        input_channels=int(x_fit.shape[1]),
        weights_path=str(source_file), weights_sha256=sha,
        fixed_seed=11,
    )
    experiment_dir = root / "experiments" / "fashion_mnist" / "V04-source" / "ADP-W-RR" / "seed11"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "simple-adp-pretrained-v04-real-source-v1",
        "dataset": dataset.manifest,
        "architecture": "torchvision.models.convnext_tiny",
        "source_weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "source_weights_url": source_enum.url,
        "full_local_source_sha256": sha,
        "site": spec.site_id,
        "source_hidden_width": 384,
        "classification_classes": spec.classes,
        "target_head_trainable": True,
        "rest_of_source_frozen": True,
        "test_policy": "official test materialized strictly after fit",
        "scope": "tiny one-seed integration smoke; not a scientific benchmark",
    }
    _atomic_json(experiment_dir / "manifest.json", manifest)
    search = VisualResidualSearch(
        "ADP-W-RR", spec,
        settings=SearchSettings(
            init_width=384, init_depth=1,
            max_width=385, max_depth=2, max_neurons=1000,
            max_epochs=1, max_candidates=1, batch_size=8,
            patience_es=1, patience_width_exp=1,
            device=args.device, seed=11,
        ),
    )
    search.fit(
        x_fit, y_fit, validation_data=(x_valid, y_valid),
        checkpoint_path=experiment_dir / "estimator.checkpoint.pt",
    )
    search.model_.verify_frozen()
    assert search.model_.source_ffn_exact_at_initialization
    validation_probs = search.predict_proba(x_valid)
    validation_pred = search.predict(x_valid)
    validation_metrics = search.task.evaluation_metrics(
        y_valid, validation_pred, validation_probs,
    )
    # No official test materialization until the architecture is frozen.
    x_test, y_test = dataset.materialize("test")
    probabilities = search.predict_proba(x_test)
    predictions = search.predict(x_test)
    test_metrics = search.task.evaluation_metrics(
        y_test, predictions, probabilities,
    )
    history = {
        "format": "simple-adp-v04-source-search-history-v1",
        "entries": search.history_,
        "selected_architecture": search.model_.hidden_widths,
        "birth_source_ffn_exact": search.model_.source_ffn_exact_at_initialization,
    }
    _atomic_json(experiment_dir / "search_history.json", history)
    model_file = experiment_dir / "selected_source_ffn_model.pt"
    torch.save(search.model_.snapshot(cursor=search.model_._width_cursor), model_file)
    out = {
        "format": "simple-adp-v04-source-real-result-v1",
        "status": "completed",
        "source": "torchvision:FashionMNIST",
        "model_source": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "source_url": source_enum.url,
        "source_checkpoint_sha256": sha,
        "frozen_source_sha256": search.model_.frozen_state_sha256(),
        "site": spec.site_id,
        "method": "ADP-W-RR",
        "selected_architecture": search.model_.hidden_widths,
        "initial_source_ffn_exact": search.model_.source_ffn_exact_at_initialization,
        "validation_metrics": validation_metrics,
        "official_test_metrics": test_metrics,
        "sample_counts": {
            "train": len(y_fit), "validation": len(y_valid), "official_test": len(y_test),
        },
        "model_artifact": model_file.name,
        "model_sha256": file_sha256(model_file),
        "history_artifact": "search_history.json",
        "history_sha256": file_sha256(experiment_dir / "search_history.json"),
        "source_provenance": {
            "torchvision_version": torchvision.__version__,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "pretrained_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        },
        "scientific_scope": "single seed, 40/20/20 official images, 1 epoch per stage",
    }
    _atomic_json(experiment_dir / "result.json", out)
    print(json.dumps(out, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
