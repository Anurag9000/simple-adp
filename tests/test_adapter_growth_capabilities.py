"""Cross-module capability contract for Bayesian ADP pairing."""
import pytest

from simple_adp.runner import plan, pair_audit
from simple_adp.runner.adapter_capabilities import (
    ADAPTER_CAPABILITIES, SHADOW_VARIANCE_ADAPTERS, VISION_ADAPTERS,
)


def test_planner_and_auditor_share_one_capability_set():
    assert plan.SHADOW_VARIANCE_ADAPTERS is SHADOW_VARIANCE_ADAPTERS
    assert pair_audit.KNOWN_GROWTH_ADAPTERS is SHADOW_VARIANCE_ADAPTERS
    assert plan.VISION_ADAPTERS is VISION_ADAPTERS


def test_every_registered_builtin_is_classified_without_aliases():
    assert len(ADAPTER_CAPABILITIES) == 22
    assert len(VISION_ADAPTERS) == 21
    assert SHADOW_VARIANCE_ADAPTERS == frozenset(ADAPTER_CAPABILITIES)
    assert "simple_adp" not in VISION_ADAPTERS
    assert ADAPTER_CAPABILITIES["simple_adp"].data_modality == "tabular"
    for name, capability in ADAPTER_CAPABILITIES.items():
        assert name == capability.name
        assert capability.loss_growth and capability.shadow_bgd_variance
        assert capability.data_modality in {"tabular", "vision"}


@pytest.mark.parametrize("family", ["vision_v03_source", "vision_v04_source"])
def test_source_weight_requirement_is_not_silently_dropped(family):
    item = ADAPTER_CAPABILITIES[family]
    assert item.source_weights_required
    assert item.data_modality == "vision"


def test_external_model_plugins_are_not_automatically_claimed_as_paired():
    assert "custom_growth" not in SHADOW_VARIANCE_ADAPTERS
    assert "custom_growth" not in VISION_ADAPTERS
    with pytest.raises(TypeError):
        ADAPTER_CAPABILITIES["custom_growth"] = ADAPTER_CAPABILITIES["simple_adp"]


def test_v06a_v06b_and_v07_variants_remain_distinct():
    for group in (
        {"vision_v06a", "vision_v06b"},
        {"vision_v07_token", "vision_v07_channel", "vision_v07_both"},
        {"vision_v08_early", "vision_v08_middle", "vision_v08_late", "vision_v08_schedule"},
        {"vision_v09_bank0", "vision_v09_bank1", "vision_v09_bank2"},
    ):
        assert group <= VISION_ADAPTERS


def test_registered_optional_vision_and_source_adapters_match_shared_capabilities(monkeypatch):
    """Instantiate actual built-ins; do not mistake a hard-coded list for registration."""
    from simple_adp.runner import plugins

    class _NoThirdPartyEntryPoints:
        def select(self, *, group):
            assert group in {"simple_adp.dataset_providers", "simple_adp.model_adapters"}
            return []

    monkeypatch.setattr(plugins, "entry_points", lambda: _NoThirdPartyEntryPoints())
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_V03_SOURCE", "1")
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_V04_SOURCE", "1")
    datasets, models = plugins.load_registries()
    assert "vision" in datasets.providers
    assert set(models.adapters) == set(ADAPTER_CAPABILITIES)
    assert {name for name in models.adapters if name != "simple_adp"} == set(VISION_ADAPTERS)
