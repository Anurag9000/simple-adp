"""Load optional provider/model plugins through Python package entry points."""

from __future__ import annotations

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
    points = entry_points()
    for group, registry in (
        ("simple_adp.dataset_providers", datasets),
        ("simple_adp.model_adapters", models),
    ):
        for point in sorted(points.select(group=group), key=lambda entry: entry.name):
            registry.register(_instantiate(point.load()))
    return datasets, models
