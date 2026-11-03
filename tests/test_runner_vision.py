"""End-to-end opt-in coordinated V01 vision suite tests."""
from __future__ import annotations

import json

import pytest

from simple_adp.config import SearchSettings
from simple_adp.runner.config import RunnerSettings
from simple_adp.runner.plan import SuitePlan
from simple_adp.runner.plugins import load_registries
from simple_adp.runner.worker import run_spec


def test_vision_is_opt_in_and_does_not_change_legacy_registries(monkeypatch):
    monkeypatch.delenv("SIMPLE_ADP_ENABLE_VISION", raising=False)
    datasets, models = load_registries()
    assert "vision" not in datasets.providers
    assert "vision_adp" not in models.adapters
    assert "vision_inner" not in models.adapters
    assert "vision_v03" not in models.adapters
    assert "vision_v04" not in models.adapters
    assert "vision_v05" not in models.adapters
    assert "vision_v06a" not in models.adapters
    assert "vision_v06b" not in models.adapters
    assert "vision_v10" not in models.adapters
    assert "vision_v11" not in models.adapters
    assert "vision_v07_token" not in models.adapters
    assert "vision_v07_channel" not in models.adapters
    assert "vision_v07_both" not in models.adapters
    assert "vision_v08_early" not in models.adapters
    assert "vision_v08_middle" not in models.adapters
    assert "vision_v08_late" not in models.adapters
    assert "vision_v08_schedule" not in models.adapters
    assert "vision_v09_bank0" not in models.adapters
    assert "vision_v09_bank1" not in models.adapters
    assert "vision_v09_bank2" not in models.adapters


def test_vision_provider_manifest_and_split_isolation(monkeypatch):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    datasets, models = load_registries()
    provider = datasets.get("vision")
    data = provider.load_task(provider.list_tasks("toy")[0])
    assert data.X.ndim == 4 and data.X.shape[1] == 1
    assert data.task_family == "vision"
    split = provider.list_splits(provider.list_tasks("toy")[0], data, RunnerSettings())[0]
    a, b, c = (set(split.train_indices), set(split.validation_indices),
               set(split.test_indices))
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)
    assert len(a) + len(b) + len(c) == len(data.X)
    assert set(data.context["official_test_group"][list(split.test_indices)]) == {1}
    assert set(data.context["official_test_group"][list(split.validation_indices)]) == {0}


def test_coordinated_vision_job_is_routed_and_executed(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    # In-process worker tests must not exec into a systemd-run user scope.
    # The production worker still keeps its original systemd behavior.
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    # In normal worker processes the scheduler explicitly admits a backend.
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings,
        results_dir=tmp_path / "results", run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    assert plan.runs
    assert {run.model_adapter for run in plan.runs} == {"vision_adp", "vision_inner", "vision_v03", "vision_v04", "vision_v05", "vision_v06a", "vision_v06b", "vision_v10", "vision_v11", "vision_v07_token", "vision_v07_channel", "vision_v07_both", "vision_v08_early", "vision_v08_middle", "vision_v08_late", "vision_v08_schedule", "vision_v09_bank0", "vision_v09_bank1", "vision_v09_bank2"}
    assert all(run.task_ref.provider == "vision" for run in plan.runs)
    assert any(run.route.model_id == "MLP-Fixed" for run in plan.runs)
    assert not any(
        run.route.model_id == "MLP-Fixed"
        and run.worker_settings.get("expansion_criterion") == "variance_stagnation"
        for run in plan.runs
    )
    target = next(
        run for run in plan.runs
        if run.route.model_id == "ADP-W-RR"
        and run.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    counterpart = next(
        run for run in plan.runs
        if run.worker_settings.get("counterpart_of") == target.identity.run_id
    )
    assert target.identity.seed == counterpart.identity.seed
    assert target.identity.split_sha256 == counterpart.identity.split_sha256
    assert counterpart.worker_settings["expansion_criterion"] == "variance_stagnation"
    spec_path = target.command[-1]
    output = run_spec(spec_path)
    assert output["status"] == "completed"
    assert output["parameter_accounting"]["total_parameters"] == (
        output["parameter_accounting"]["frozen_parameters"]
        + output["parameter_accounting"]["trainable_parameters"]
    )
    assert "head" in output["parameter_accounting"]["eligible_growth_sites"]
    assert output["inference_compute"]["observed_linear_conv_macs_per_image"] > 0
    assert output["inference_compute"]["whole_model_flops"] is None
    assert output["task_family"] == "vision"
    assert output["test_metrics"]["accuracy"] >= 0
    assert output["model_id"] == "ADP-W-RR"
    assert (tmp_path / "results").exists()

    inner = next(
        run for run in plan.runs
        if run.model_adapter == "vision_inner"
        and run.route.model_id == "ADP-W-RR"
        and run.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    inner_pair = next(
        run for run in plan.runs
        if run.worker_settings.get("counterpart_of") == inner.identity.run_id
    )
    assert inner_pair.model_adapter == "vision_inner"
    assert inner_pair.identity.seed == inner.identity.seed
    assert inner_pair.identity.split_sha256 == inner.identity.split_sha256
    assert inner_pair.worker_settings["expansion_criterion"] == "variance_stagnation"
    inner_result = run_spec(inner.command[-1])
    assert inner_result["status"] == "completed"
    assert "adapter:block1" in inner_result["parameter_accounting"]["eligible_growth_sites"]
    assert inner_result["architecture_id"] == "V02:toy-frozen-residual-adapter"
    assert inner_result["growth_site"] == "adapter:block1"
    assert len(inner_result["selected_architecture"]) >= 1
    assert inner_result["test_metrics"]["accuracy"] >= 0
    assert inner_result["artifact_path"] is not None

    generic = next(
        run for run in plan.runs
        if run.model_adapter == "vision_inner"
        and run.route.model_id == "ADP-WD-RR-Dcur"
        and run.route.generic_profile == "corrected"
        and run.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    generic_result = run_spec(generic.command[-1])
    assert generic_result["status"] == "completed"
    assert generic_result["architecture_id"] == "V02:toy-frozen-residual-adapter"
    assert generic_result["test_metrics"]["accuracy"] >= 0
    assert any(run.model_adapter == "vision_inner"
               and run.route.generic_profile == "faithful" for run in plan.runs)

    variance_result = run_spec(inner_pair.command[-1])
    assert variance_result["status"] == "completed"
    assert variance_result["expansion_criterion"] == "variance_stagnation"
    assert variance_result["selected_architecture"]





@pytest.mark.parametrize(
    ("family", "adapter_name", "expected_site"),
    [
        ("V03", "vision_v03", "ffn:block0"),
        ("V04", "vision_v04", "channel:stage0.block0"),
        ("V05", "vision_v05", "channel:poolformer.block0"),
        ("V06a", "vision_v06a", "channel:s2.block0"),
        ("V06b", "vision_v06b", "channel:axial.block0"),
        ("V10", "vision_v10", "channel:resmlp.block0"),
        ("V11", "vision_v11", "channel:gfnet.block0"),
    ],
)
def test_native_ffn_runner_executes_and_pairs_variance(
    monkeypatch, tmp_path, family, adapter_name, expected_site,
):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    native = next(
        row for row in plan.runs
        if row.model_adapter == adapter_name
        and row.route.model_id == "ADP-W-RR"
        and row.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    alternate = next(
        row for row in plan.runs
        if row.worker_settings.get("counterpart_of") == native.identity.run_id
    )
    assert alternate.model_adapter == adapter_name
    assert alternate.identity.seed == native.identity.seed
    assert alternate.identity.split_sha256 == native.identity.split_sha256
    assert alternate.worker_settings["expansion_criterion"] == "variance_stagnation"
    native_result = run_spec(native.command[-1])
    assert native_result["status"] == "completed"
    assert native_result["growth_site"] == expected_site
    assert native_result["architecture_id"].startswith(family + ":tiny-")
    assert len(native_result["selected_architecture"]) == 1
    assert native_result["test_metrics"]["accuracy"] >= 0
    assert native_result["artifact_path"] is not None
    assert len(native_result["frozen_sha256"]) == 64
    alternate_result = run_spec(alternate.command[-1])
    assert alternate_result["status"] == "completed"
    assert alternate_result["expansion_criterion"] == "variance_stagnation"
    generic = next(
        row for row in plan.runs
        if row.model_adapter == adapter_name
        and row.route.model_id == "ADP-WD-RR-Dcur"
        and row.route.generic_profile == "corrected"
        and row.worker_settings.get("expansion_criterion") != "variance_stagnation"
    )
    assert generic.route.route_label.startswith(family + ":")
    generic_result = run_spec(generic.command[-1])
    assert generic_result["status"] == "completed"
    assert generic_result["growth_site"] == expected_site
    assert any(
        row.model_adapter == adapter_name and row.route.generic_profile == "faithful"
        for row in plan.runs
    )



@pytest.mark.parametrize(
    ("site", "adapter_name", "expected_target"),
    [
        ("token:block0", "vision_v07_token", 4),
        ("channel:block0", "vision_v07_channel", 4),
        ("both:block0", "vision_v07_both", 4),
    ],
)
def test_coordinated_v07_mixer_sites_with_matched_variance(
    monkeypatch, tmp_path, site, adapter_name, expected_target,
):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False, repeat_count=1,
        missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    originals = [
        r for r in plan.runs
        if r.model_adapter == adapter_name
        and r.worker_settings.get("expansion_criterion") != "variance_stagnation"
    ]
    variances = [r for r in plan.runs if r.model_adapter == adapter_name
                 and r.worker_settings.get("expansion_criterion") == "variance_stagnation"]
    assert len(originals) == 22
    assert len(variances) == 21
    route = next(r for r in originals if r.route.model_id == "ADP-W-RR")
    alternate = next(r for r in variances
                     if r.worker_settings.get("counterpart_of") == route.identity.run_id)
    assert alternate.identity.seed == route.identity.seed
    assert alternate.identity.split_sha256 == route.identity.split_sha256
    assert alternate.route.model_id == route.route.model_id
    assert route.route.route_label.startswith("V07-")
    result = run_spec(route.command[-1])
    assert result["status"] == "completed"
    assert result["growth_site"] == site
    assert result["architecture_id"].startswith("V07:tiny-mlp-mixer-")
    assert len(result["selected_architecture"]) == 1
    assert result["selected_architecture"][0] >= 2
    assert len(result["frozen_sha256"]) == 64
    assert result["artifact_path"] is not None
    assert result["test_metrics"]["accuracy"] >= 0
    alt_result = run_spec(alternate.command[-1])
    assert alt_result["status"] == "completed"
    assert alt_result["expansion_criterion"] == "variance_stagnation"
    generic = next(r for r in originals
                   if r.route.model_id == "ADP-WD-RR-Dcur"
                   and r.route.generic_profile == "corrected")
    assert run_spec(generic.command[-1])["status"] == "completed"



@pytest.mark.parametrize("site", ["early", "middle", "late"])
def test_v08_coordinated_fixed_stage_and_matched_criterion(monkeypatch, tmp_path, site):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    options = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=options, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    name = "vision_v08_" + site
    originals = [
        j for j in plan.runs if j.model_adapter == name
        and j.worker_settings.get("expansion_criterion") != "variance_stagnation"
    ]
    alternatives = [
        j for j in plan.runs if j.model_adapter == name
        and j.worker_settings.get("expansion_criterion") == "variance_stagnation"
    ]
    assert len(originals) == 22
    assert len(alternatives) == 21
    original = next(j for j in originals if j.route.model_id == "ADP-W-RR")
    paired = next(j for j in alternatives
                  if j.worker_settings.get("counterpart_of") == original.identity.run_id)
    assert original.identity.seed == paired.identity.seed
    assert original.identity.split_sha256 == paired.identity.split_sha256
    result = run_spec(original.command[-1])
    assert result["status"] == "completed"
    assert result["growth_site"] == "adapter:" + site
    assert result["architecture_id"] == "V08:tiny-3stage-residual-adapter-" + site
    assert len(result["frozen_sha256"]) == 64
    assert result["artifact_path"] is not None
    assert result["test_metrics"]["accuracy"] >= 0
    alt = run_spec(paired.command[-1])
    assert alt["status"] == "completed"
    assert alt["expansion_criterion"] == "variance_stagnation"
    generic = next(j for j in originals if j.route.model_id == "ADP-WD-RR-Dcur"
                   and j.route.generic_profile == "corrected")
    assert run_spec(generic.command[-1])["status"] == "completed"



@pytest.mark.parametrize("member", [0, 1, 2])
def test_v09_coordinated_bank_member_and_immutable_routing_histograms(
    monkeypatch, tmp_path, member,
):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    settings = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=settings, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    name = f"vision_v09_bank{member}"
    originals = [x for x in plan.runs if x.model_adapter == name
                 and x.worker_settings.get("expansion_criterion") != "variance_stagnation"]
    alternate = [x for x in plan.runs if x.model_adapter == name
                 and x.worker_settings.get("expansion_criterion") == "variance_stagnation"]
    assert len(originals) == 22
    assert len(alternate) == 21
    native = next(x for x in originals if x.route.model_id == "ADP-W-RR")
    counterpart = next(
        x for x in alternate
        if x.worker_settings.get("counterpart_of") == native.identity.run_id
    )
    assert counterpart.identity.seed == native.identity.seed
    assert counterpart.identity.split_sha256 == native.identity.split_sha256
    result = run_spec(native.command[-1])
    assert result["status"] == "completed"
    assert result["growth_site"] == f"bank:{member}"
    assert result["architecture_id"] == f"V09:tiny-frozen-hard-router-bank{member}"
    assert result["router_kind"] == "hard-argmax-cosine-v1"
    assert len(result["router_histogram_train"]) == 3
    assert len(result["router_histogram_test"]) == 3
    assert sum(result["router_histogram_train"]) > 0
    assert sum(result["router_histogram_test"]) > 0
    assert len(result["frozen_sha256"]) == 64
    assert result["artifact_path"] is not None
    assert result["test_metrics"]["accuracy"] >= 0
    other = run_spec(counterpart.command[-1])
    assert other["status"] == "completed"
    assert other["expansion_criterion"] == "variance_stagnation"
    assert other["router_histogram_train"] == result["router_histogram_train"]
    assert other["router_histogram_test"] == result["router_histogram_test"]
    generic = next(
        x for x in originals if x.route.model_id == "ADP-WD-RR-Dcur"
        and x.route.generic_profile == "corrected"
    )
    assert run_spec(generic.command[-1])["status"] == "completed"



def test_v08_predeclared_site_rr_scheduled_worker_pairs_and_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    monkeypatch.setenv("TABULAR_SKIP_SYSTEMD_SCOPE", "1")
    monkeypatch.setenv("TABULAR_BACKEND", "cpu")
    monkeypatch.setenv("SIMPLE_ADP_DISABLE_GPU_ACCELERATORS", "1")
    monkeypatch.setenv("TRAINING_CONTROL_CPU_ONLY", "1")
    datasets, models = load_registries()
    options = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1, model_seeds=(0,),
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
        smoke_test=True, cpu_only=True,
    )
    plan = SuitePlan.build(
        settings=options, results_dir=tmp_path / "results",
        run_root=tmp_path / "runs",
        dataset_providers=tuple(datasets.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(models.adapters.values()),
    )
    name = "vision_v08_schedule"
    originals = [x for x in plan.runs if x.model_adapter == name
                 and x.worker_settings.get("expansion_criterion") != "variance_stagnation"]
    alternatives = [x for x in plan.runs if x.model_adapter == name
                    and x.worker_settings.get("expansion_criterion") == "variance_stagnation"]
    assert len(originals) == 22 and len(alternatives) == 21
    native = next(x for x in originals if x.route.model_id == "ADP-W-RR")
    paired = next(x for x in alternatives
                  if x.worker_settings.get("counterpart_of") == native.identity.run_id)
    assert paired.identity.split_sha256 == native.identity.split_sha256
    assert paired.identity.seed == native.identity.seed
    result = run_spec(native.command[-1])
    assert result["status"] == "completed"
    assert result["growth_site"] == "adapter:round_robin_sites"
    assert result["site_schedule_kind"] == "predeclared-early-middle-late-rr-v1"
    assert result["site_schedule_trace"]
    assert all(x["site"] in {"adapter:early", "adapter:middle", "adapter:late"}
               for x in result["site_schedule_trace"])
    assert result["architecture_id"] == "V08:tiny-3stage-predeclared-rr-sites"
    assert len(result["frozen_sha256"]) == 64
    assert result["artifact_path"] is not None
    assert result["test_metrics"]["accuracy"] >= 0
    other = run_spec(paired.command[-1])
    assert other["status"] == "completed"
    assert other["expansion_criterion"] == "variance_stagnation"
    assert other["site_schedule_kind"] == result["site_schedule_kind"]
    generic = next(x for x in originals if x.route.model_id == "ADP-WD-RR-Dcur"
                   and x.route.generic_profile == "corrected")
    assert run_spec(generic.command[-1])["status"] == "completed"



def test_opt_in_full_visual_route_inventory_has_exact_817_paired_jobs(monkeypatch, tmp_path):
    """No-training manifest closure over every *registered* visual scope.

    There are 19 explicit model/site-policy families (12 architecture IDs).
    Each has 22 original source-method/profile routes and 21 corresponding
    variance-gated adaptive routes. The fixed non-growing control has none.
    This does NOT mean these 817 synthetic/real dataset jobs were trained.
    """
    monkeypatch.setenv("SIMPLE_ADP_ENABLE_VISION", "1")
    providers, adapters = load_registries()
    configured = RunnerSettings(
        tasks=("vision",), run_hardness_screen=False,
        repeat_count=1, missing_present_task_repeats=1,
        model_seeds=(0,), smoke_test=True, cpu_only=True,
        width_to_depth_width=2, width_to_depth_depth=1,
        max_candidates=1, max_epochs=1, batch_size=16,
    )
    plan = SuitePlan.build(
        settings=configured,
        results_dir=tmp_path / "result",
        run_root=tmp_path / "run",
        dataset_providers=tuple(providers.providers.values()),
        dataset_suites=(("vision", "toy"),),
        model_adapters=tuple(adapters.adapters.values()),
    )
    expected = {
        "vision_adp", "vision_inner",
        "vision_v03", "vision_v04", "vision_v05",
        "vision_v06a", "vision_v06b",
        "vision_v07_token", "vision_v07_channel", "vision_v07_both",
        "vision_v08_early", "vision_v08_middle", "vision_v08_late",
        "vision_v08_schedule",
        "vision_v09_bank0", "vision_v09_bank1", "vision_v09_bank2",
        "vision_v10", "vision_v11",
    }
    assert len(expected) == 19
    assert {r.model_adapter for r in plan.runs} == expected
    assert len(plan.runs) == 817
    all_job_ids = [r.identity.run_id for r in plan.runs]
    assert len(all_job_ids) == len(set(all_job_ids))
    for family in sorted(expected):
        jobs = [r for r in plan.runs if r.model_adapter == family]
        originals = {
            r.identity.run_id: r for r in jobs
            if r.worker_settings.get("expansion_criterion") != "variance_stagnation"
        }
        alternates = [
            r for r in jobs
            if r.worker_settings.get("expansion_criterion") == "variance_stagnation"
        ]
        assert len(originals) == 22, family
        assert len(alternates) == 21, family
        assert len(jobs) == 43, family
        assert len({r.route.route_label for r in originals.values()}) == 22, family
        assert sum(r.route.model_id == "MLP-Fixed" for r in originals.values()) == 1
        claimed = set()
        for variant in alternates:
            parent_id = variant.worker_settings.get("counterpart_of")
            assert parent_id in originals, (family, parent_id)
            assert parent_id not in claimed
            claimed.add(parent_id)
            parent = originals[parent_id]
            assert parent.route.model_id != "MLP-Fixed"
            assert variant.route.model_id == parent.route.model_id
            assert variant.route.generic_profile == parent.route.generic_profile
            assert variant.identity.seed == parent.identity.seed
            assert variant.identity.split_sha256 == parent.identity.split_sha256
            assert variant.identity.fold == parent.identity.fold
            assert variant.identity.split_repeat == parent.identity.split_repeat
        assert len(claimed) == 21
