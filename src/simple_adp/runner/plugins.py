"""Load optional provider/model plugins through Python package entry points."""

from __future__ import annotations

import os
from importlib.metadata import entry_points
from typing import Any

from .datasets import DatasetRegistry, OpenMLDatasetProvider
from .models import ModelRegistry, SimpleADPModelAdapter
from .public_datasets import MovieLensProvider
from .temporal_datasets import UCIElectricityProvider
from .sequence_datasets import UCIHAPTProvider
from .nab_datasets import NABProvider
from .monash_datasets import MonashForecastingProvider
from .physionet_datasets import PhysioNetECGProvider
from .odds_datasets import ODDSProvider


def _instantiate(value: Any) -> Any:
    return value() if isinstance(value, type) else value


def load_registries() -> tuple[DatasetRegistry, ModelRegistry]:
    datasets = DatasetRegistry()
    models = ModelRegistry()
    datasets.register(OpenMLDatasetProvider())
    datasets.register(MovieLensProvider())
    datasets.register(UCIElectricityProvider())
    datasets.register(UCIHAPTProvider())
    datasets.register(NABProvider())
    datasets.register(MonashForecastingProvider())
    datasets.register(PhysioNetECGProvider())
    datasets.register(ODDSProvider())
    models.register(SimpleADPModelAdapter())
    # Explicit opt-in retains byte-for-byte legacy model/provider inventories.
    if os.environ.get("SIMPLE_ADP_ENABLE_VISION") == "1":
        from .vision import VisionDatasetProvider, VisionModelAdapter, VisionInnerModelAdapter, VisionNativeFFNModelAdapter, VisionMixerModelAdapter, VisionMultistageModelAdapter, VisionAdapterBankModelAdapter, VisionMultistageScheduleAdapter
        datasets.register(VisionDatasetProvider())
        models.register(VisionModelAdapter())
        models.register(VisionInnerModelAdapter())
        models.register(VisionNativeFFNModelAdapter("V03"))
        models.register(VisionNativeFFNModelAdapter("V04"))
        models.register(VisionNativeFFNModelAdapter("V05"))
        models.register(VisionNativeFFNModelAdapter("V06a"))
        models.register(VisionNativeFFNModelAdapter("V06b"))
        models.register(VisionNativeFFNModelAdapter("V10"))
        models.register(VisionNativeFFNModelAdapter("V11"))
        models.register(VisionMixerModelAdapter("token"))
        models.register(VisionMixerModelAdapter("channel"))
        models.register(VisionMixerModelAdapter("both"))
        models.register(VisionMultistageModelAdapter("early"))
        models.register(VisionMultistageModelAdapter("middle"))
        models.register(VisionMultistageModelAdapter("late"))
        models.register(VisionMultistageScheduleAdapter())
        for member in range(3):
            models.register(VisionAdapterBankModelAdapter(member))
        # Independent explicit gate; never change the existing 817 toy jobs.
        if os.environ.get("SIMPLE_ADP_ENABLE_V04_SOURCE") == "1":
            from .vision_source import VisionSourceConvNeXtModelAdapter
            models.register(VisionSourceConvNeXtModelAdapter())
    points = entry_points()
    for group, registry in (
        ("simple_adp.dataset_providers", datasets),
        ("simple_adp.model_adapters", models),
    ):
        for point in sorted(points.select(group=group), key=lambda entry: entry.name):
            registry.register(_instantiate(point.load()))
    return datasets, models
