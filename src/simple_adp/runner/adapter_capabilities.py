"""Explicit built-in supervised-ADP adapter capabilities.

Single authority for the core suite planner and its static Bayesian pair
auditor. Being named here does not prove that a job ran or that a published
source checkpoint/data asset is available. External plugin names never gain
automatic scientific eligibility merely by appearing in a RunSpec.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True)
class AdapterGrowthCapability:
    name: str
    data_modality: Literal["tabular", "vision"]
    loss_growth: bool
    shadow_bgd_variance: bool
    source_weights_required: bool = False


_BUILTIN_CAPABILITIES = (
    AdapterGrowthCapability("simple_adp", "tabular", True, True),
    AdapterGrowthCapability("vision_adp", "vision", True, True),
    AdapterGrowthCapability("vision_inner", "vision", True, True),
    AdapterGrowthCapability("vision_v03", "vision", True, True),
    AdapterGrowthCapability("vision_v04", "vision", True, True),
    AdapterGrowthCapability("vision_v05", "vision", True, True),
    AdapterGrowthCapability("vision_v06a", "vision", True, True),
    AdapterGrowthCapability("vision_v06b", "vision", True, True),
    AdapterGrowthCapability("vision_v10", "vision", True, True),
    AdapterGrowthCapability("vision_v11", "vision", True, True),
    AdapterGrowthCapability("vision_v07_token", "vision", True, True),
    AdapterGrowthCapability("vision_v07_channel", "vision", True, True),
    AdapterGrowthCapability("vision_v07_both", "vision", True, True),
    AdapterGrowthCapability("vision_v08_early", "vision", True, True),
    AdapterGrowthCapability("vision_v08_middle", "vision", True, True),
    AdapterGrowthCapability("vision_v08_late", "vision", True, True),
    AdapterGrowthCapability("vision_v08_schedule", "vision", True, True),
    AdapterGrowthCapability("vision_v09_bank0", "vision", True, True),
    AdapterGrowthCapability("vision_v09_bank1", "vision", True, True),
    AdapterGrowthCapability("vision_v09_bank2", "vision", True, True),
    AdapterGrowthCapability("vision_v04_source", "vision", True, True, True),
    AdapterGrowthCapability("vision_v03_source", "vision", True, True, True),
)

if len({entry.name for entry in _BUILTIN_CAPABILITIES}) != len(_BUILTIN_CAPABILITIES):
    raise RuntimeError("duplicate built-in adapter capability")

ADAPTER_CAPABILITIES = MappingProxyType({
    capability.name: capability for capability in _BUILTIN_CAPABILITIES
})
VISION_ADAPTERS = frozenset(
    item.name for item in _BUILTIN_CAPABILITIES if item.data_modality == "vision"
)
SHADOW_VARIANCE_ADAPTERS = frozenset(
    item.name for item in _BUILTIN_CAPABILITIES
    if item.loss_growth and item.shadow_bgd_variance
)
