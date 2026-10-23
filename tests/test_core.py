import numpy as np
import pandas as pd
import pytest
import torch
from itertools import product
from dataclasses import replace
import simple_adp.estimator as estimator_module

from simple_adp import (
    MODEL_REGISTRY, CallableTask, ClassificationTask, RegressionTask,
    SearchSettings, TabularADP, TabularMLP, TabularPreprocessor,
    can_add_depth, expand_depth, expand_depth_match_width, expand_width_den_round_robin,
    can_grow_width_all_layers, can_grow_width_round_robin, expand_width_all_layers,
    expand_width_round_robin, rebuild_mlp,
)


EXPECTED = [
    "LGND-W-Greedy", "ADP-W-RR", "ADP-W-All", "ADP-WD-RR-D1",
    "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
    "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR",
    "ADP-DW-D1-All", "ADP-D-D1", "ADP-D-Dcur", "MLP-Fixed",
]


def tiny_settings(**updates):
    values = dict(init_width=4, init_depth=2, max_epochs=2, patience_es=2,
                  patience_width_exp=1, patience_depth_exp=1, max_width=8,
                  max_depth=4, max_neurons=100, batch_size=8, seed=7)
    values.update(updates)
    return SearchSettings(**values)


def test_registry_has_all_documented_ids_and_custom_fixed_grid():
    assert list(MODEL_REGISTRY) == EXPECTED
    assert MODEL_REGISTRY["LGND-W-Greedy"].controller == "lgnd"
    assert MODEL_REGISTRY["ADP-W-RR"].width_strategy == "round_robin"
    assert MODEL_REGISTRY["ADP-W-All"].width_strategy == "all_layers"
    assert all(MODEL_REGISTRY[name].generic for name in EXPECTED[3:11])
    assert MODEL_REGISTRY["ADP-D-D1"].progressive_depth
    assert MODEL_REGISTRY["ADP-D-Dcur"].depth_seed == "current"
    assert MODEL_REGISTRY["MLP-Fixed"].controller == "fixed"
    with pytest.raises(ValueError, match="explicitly supplied fixed_grid"):
        TabularADP("MLP-Fixed", task=RegressionTask(), settings=tiny_settings()).fit(
            np.arange(30, dtype=np.float32).reshape(10, 3)[:8],
            np.arange(8, dtype=np.float32),
            validation_data=(np.arange(30, dtype=np.float32).reshape(10, 3)[8:], np.arange(8, 10, dtype=np.float32)),
        )
    expected = {
        "ADP-WD-RR-D1": ("width_then_depth", "round_robin", "one"),
        "ADP-WD-All-Dcur": ("width_then_depth", "all_layers", "current"),
        "ADP-WD-RR-Dcur": ("width_then_depth", "round_robin", "current"),
        "ADP-WD-All-D1": ("width_then_depth", "all_layers", "one"),
        "ADP-DW-D1-RR": ("depth_then_width", "round_robin", "one"),
        "ADP-DW-Dcur-All": ("depth_then_width", "all_layers", "current"),
        "ADP-DW-Dcur-RR": ("depth_then_width", "round_robin", "current"),
        "ADP-DW-D1-All": ("depth_then_width", "all_layers", "one"),
    }
    assert {name: (MODEL_REGISTRY[name].order, MODEL_REGISTRY[name].width_strategy,
                   MODEL_REGISTRY[name].depth_seed) for name in expected} == expected
    assert MODEL_REGISTRY["ADP-W-All"].optimizer_profile == "adam_cosine"
    assert MODEL_REGISTRY["ADP-WD-RR-D1"].optimizer_profile == "adamw"
    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(10, dtype=np.float32)
    est = TabularADP("MLP-Fixed", task=RegressionTask(), settings=tiny_settings())
    est.fit(x[:8], y[:8], validation_data=(x[8:], y[8:]), fixed_grid=[[3, 2]])
    assert est.model_.hidden_widths == [3, 2]


def test_default_model_and_search_neuron_caps_match_documented_runner_value():
    expected = 1_000_000_000
    assert SearchSettings().max_neurons == expected
    assert all(config.max_neurons == expected for config in MODEL_REGISTRY.values())
    assert TabularADP("ADP-W-All").settings.max_neurons == expected


def test_estimator_uses_model_config_search_defaults_and_explicit_settings_override(monkeypatch):
    configured = replace(
        MODEL_REGISTRY["ADP-W-All"], patience_width_exp=3,
        patience_depth_exp=4, delta_width=0.125, delta_depth=0.25,
        max_width=19, max_depth=6, max_neurons=1234,
    )
    monkeypatch.setattr(estimator_module, "get_model_config", lambda _model_id: configured)

    estimator = TabularADP("ADP-W-All", task=RegressionTask())
    assert estimator.settings.patience_width_exp == 3
    assert estimator.settings.patience_depth_exp == 4
    assert estimator.settings.delta_width == 0.125
    assert estimator.settings.delta_depth == 0.25
    assert estimator.settings.max_width == 19
    assert estimator.settings.max_depth == 6
    assert estimator.settings.max_neurons == 1234

    supplied = tiny_settings(max_width=7, max_depth=3, max_neurons=500)
    override = TabularADP("ADP-W-All", task=RegressionTask(), settings=supplied)
    assert override.settings is supplied
    assert override.settings.max_width == 7


def test_saved_estimator_rejects_same_id_with_changed_registered_mechanism(tmp_path, monkeypatch):
    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = X[:, 0] - X[:, 1]
    settings = tiny_settings(init_width=2, init_depth=1, max_width=2, max_depth=1,
                             max_neurons=16, max_epochs=1, patience_width_exp=0)
    estimator = TabularADP("ADP-W-All", task=RegressionTask(), settings=settings)
    estimator.fit(X[:6], y[:6], validation_data=(X[6:], y[6:]))
    artifact = tmp_path / "estimator.pt"
    estimator.save(artifact)

    changed_mechanism = replace(MODEL_REGISTRY["ADP-W-All"], width_strategy="round_robin")
    monkeypatch.setattr(estimator_module, "get_model_config", lambda _model_id: changed_mechanism)
    with pytest.raises(ValueError, match="registered mechanism"):
        TabularADP.load(artifact)


def test_saved_estimator_rejects_legacy_artifact_without_mechanism_identity(tmp_path):
    artifact = tmp_path / "legacy-v1.pt"
    torch.save({"format_version": 1, "model_id": "ADP-W-All"}, artifact)
    with pytest.raises(ValueError, match="unsupported estimator artifact version"):
        TabularADP.load(artifact)


def test_growth_mutations_cursor_depth_and_overlap_copy():
    torch.manual_seed(3)
    model = TabularMLP(3, [2, 2], 1)
    rr1 = expand_width_round_robin(model, 1, 5)
    assert rr1.hidden_widths == [3, 2]
    assert rr1._width_cursor == 1
    rr2 = expand_width_round_robin(rr1, 1, 5)
    assert rr2.hidden_widths == [3, 3]
    assert rr2._width_cursor == 0
    restored = TabularMLP.from_snapshot(rr1.snapshot(cursor=rr1._width_cursor))
    assert restored._width_cursor == 1
    uneven = TabularMLP(3, [5, 4, 4], 1)
    den_next = expand_width_den_round_robin(uneven, 1, 8)
    generic_next = expand_width_round_robin(uneven, 1, 8)
    assert den_next.hidden_widths == [5, 5, 4]
    assert generic_next.hidden_widths == [6, 4, 4]
    all_layers = expand_width_all_layers(model, 1, 5)
    assert all_layers.hidden_widths == [3, 3]
    # Overlapping old parameters survive widening exactly.
    torch.testing.assert_close(all_layers.hidden_layers[0].weight[:2, :], model.hidden_layers[0].weight)
    assert expand_depth(model, 4).hidden_widths == [2, 2, 1]
    assert tiny_settings(ex_k_depth=3).ex_k_depth == 3
    assert expand_depth(model, 4).hidden_widths == [2, 2, 1]  # one layer per dedicated call
    assert expand_depth_match_width(model, 4).hidden_widths == [2, 2, 2]
    assert can_add_depth(model, 4, 100, 1)
    assert not can_add_depth(model, 2, 100, 1)
    assert not can_add_depth(model, 4, 5, 1)
    assert can_grow_width_all_layers(model, 1, 5, 10)
    assert not can_grow_width_all_layers(model, 1, 5, 6)
    assert can_grow_width_round_robin(model, 1, 5, 10)
    assert not can_grow_width_round_robin(model, 1, 5, 5)
    nonuniform = rebuild_mlp(model, [2, 3])
    assert expand_depth(nonuniform, 4) is None
    assert expand_depth_match_width(nonuniform, 4).hidden_widths == [2, 3, 3]


def test_generic_candidate_rebuild_copies_batchnorm_overlap_and_snapshot_buffers():
    model = TabularMLP(2, [2, 2], 1, batch_norm=True)
    with torch.no_grad():
        for index, layer in enumerate(model.hidden_layers):
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.copy_(torch.tensor([2.0, 3.0]))
                layer.bias.copy_(torch.tensor([4.0, 5.0]))
                layer.running_mean.copy_(torch.tensor([6.0, 7.0]))
                layer.running_var.copy_(torch.tensor([8.0, 9.0]))
                layer.num_batches_tracked.fill_(10 + index)
        model.output_layer.weight.copy_(torch.tensor([[11.0, 12.0]]))

    widened = rebuild_mlp(model, [2, 3])

    assert widened.batch_norm is True
    assert widened.hidden_widths == [2, 3]
    new_norm = widened.hidden_layers[4]
    assert isinstance(new_norm, torch.nn.BatchNorm1d)
    torch.testing.assert_close(new_norm.weight[:2], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(new_norm.bias[:2], torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(new_norm.running_mean[:2], torch.tensor([6.0, 7.0]))
    torch.testing.assert_close(new_norm.running_var[:2], torch.tensor([8.0, 9.0]))
    # Newly added BatchNorm entries retain the fresh module initialization.
    torch.testing.assert_close(new_norm.weight[2:], torch.ones(1))
    torch.testing.assert_close(new_norm.bias[2:], torch.zeros(1))
    torch.testing.assert_close(new_norm.running_mean[2:], torch.zeros(1))
    torch.testing.assert_close(new_norm.running_var[2:], torch.ones(1))
    torch.testing.assert_close(new_norm.num_batches_tracked, torch.tensor(14))
    torch.testing.assert_close(widened.output_layer.weight[:, :2], model.output_layer.weight)

    restored = TabularMLP.from_snapshot(widened.snapshot(cursor=1))
    assert restored.batch_norm is True
    assert restored._width_cursor == 1
    for key, value in widened.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_architecture_growth_keeps_new_linear_cells_at_fresh_initialization():
    seed = 9182
    source = TabularMLP(2, [2], 1)

    torch.manual_seed(seed)
    fresh_target = TabularMLP(2, [3], 1)
    fresh_hidden_weight = fresh_target.hidden_layers[0].weight.detach().clone()
    fresh_hidden_bias = fresh_target.hidden_layers[0].bias.detach().clone()
    fresh_output_weight = fresh_target.output_layer.weight.detach().clone()

    torch.manual_seed(seed)
    grown = rebuild_mlp(source, [3])

    torch.testing.assert_close(
        grown.hidden_layers[0].weight[:2], source.hidden_layers[0].weight,
    )
    torch.testing.assert_close(
        grown.hidden_layers[0].bias[:2], source.hidden_layers[0].bias,
    )
    torch.testing.assert_close(
        grown.output_layer.weight[:, :2], source.output_layer.weight,
    )
    torch.testing.assert_close(
        grown.hidden_layers[0].weight[2:], fresh_hidden_weight[2:],
    )
    torch.testing.assert_close(
        grown.hidden_layers[0].bias[2:], fresh_hidden_bias[2:],
    )
    torch.testing.assert_close(
        grown.output_layer.weight[:, 2:], fresh_output_weight[:, 2:],
    )


@pytest.mark.parametrize("increment", [0, -1])
def test_generic_growth_primitives_clamp_nonpositive_size_arguments(increment):
    """The documented generic helpers use max(1, value) at their boundaries."""
    model = TabularMLP(2, [2, 2], 1)
    assert expand_width_round_robin(model, increment, 5).hidden_widths == [3, 2]
    assert expand_width_all_layers(model, increment, 5).hidden_widths == [3, 3]
    assert expand_depth(model, 4, increment).hidden_widths == [2, 2, 1]


def test_width_mutations_exhaust_small_architecture_state_space():
    """Exhaustively check the guide's three width-transition rules on small states."""
    for depth in range(1, 4):
        for widths in product(range(1, 4), repeat=depth):
            model = TabularMLP(2, widths, 2)
            for max_width in range(1, 5):
                for increment in (1, 2):
                    expected_all = [
                        min(max_width, width + increment) for width in widths
                    ]
                    all_layers = expand_width_all_layers(
                        model, increment, max_width,
                    )
                    if expected_all == list(widths):
                        assert all_layers is None
                    else:
                        assert all_layers.hidden_widths == expected_all

                    for cursor in range(depth):
                        model._width_cursor = cursor

                        # Generic adp_search RR selects the first layer below
                        # max_width, regardless of the other layer widths.
                        generic_index = next(
                            (cursor + offset) % depth
                            for offset in range(depth)
                            if widths[(cursor + offset) % depth] < max_width
                        ) if any(width < max_width for width in widths) else None
                        generic = expand_width_round_robin(
                            model, increment, max_width,
                        )
                        if generic_index is None:
                            assert generic is None
                        else:
                            expected = list(widths)
                            expected[generic_index] = min(
                                max_width, expected[generic_index] + increment,
                            )
                            assert generic.hidden_widths == expected
                            assert generic._width_cursor == (generic_index + 1) % depth

                        # Dedicated DEN RR instead targets minimum(widths)+k.
                        target = min(widths) + increment
                        den_index = next(
                            (cursor + offset) % depth
                            for offset in range(depth)
                            if widths[(cursor + offset) % depth] < target
                            and widths[(cursor + offset) % depth] < max_width
                        ) if any(width < target and width < max_width for width in widths) else None
                        den = expand_width_den_round_robin(
                            model, increment, max_width,
                        )
                        if den_index is not None and widths[den_index] + increment > max_width:
                            den_index = None
                        if den_index is None:
                            assert den is None
                        else:
                            expected = list(widths)
                            expected[den_index] += increment
                            assert den.hidden_widths == expected
                            assert den._width_cursor == (den_index + 1) % depth


def test_depth_mutations_exhaust_small_architecture_state_space():
    """Exhaustively check depth eligibility and both documented seed rules."""
    for depth in range(1, 4):
        for widths in product(range(1, 4), repeat=depth):
            model = TabularMLP(2, widths, 2)
            uniform = len(set(widths)) == 1
            current_width = widths[-1]
            current_units = sum(widths) + model.output_dim
            for max_depth in range(1, 5):
                for max_neurons in range(1, 13):
                    for seed_width in (1, current_width):
                        expected_allowed = (
                            uniform
                            and depth + 1 <= max_depth
                            and current_units + seed_width <= max_neurons
                        )
                        assert can_add_depth(
                            model, max_depth, max_neurons, seed_width,
                        ) is expected_allowed

                    progressive = expand_depth(model, max_depth, 1)
                    expected_progressive = (
                        [*widths, 1]
                        if uniform and depth < max_depth
                        else None
                    )
                    assert (progressive.hidden_widths if progressive else None) == expected_progressive

                    matched = expand_depth_match_width(model, max_depth)
                    expected_matched = (
                        [*widths, current_width]
                        if depth < max_depth
                        else None
                    )
                    assert (matched.hidden_widths if matched else None) == expected_matched


def test_width_capacity_guards_exhaust_small_architecture_state_space():
    """Check dedicated all-layer/RR capacity predicates at every small boundary."""
    for depth in range(1, 4):
        for widths in product(range(1, 5), repeat=depth):
            model = TabularMLP(2, widths, 2)
            hidden_and_output_units = sum(widths) + model.output_dim
            for max_width in range(1, 6):
                for increment in range(1, 4):
                    all_widths_fit = all(
                        width + increment <= max_width for width in widths
                    )
                    for max_neurons in range(
                        0, hidden_and_output_units + depth * increment + 2,
                    ):
                        expected_all = (
                            all_widths_fit
                            and hidden_and_output_units + depth * increment <= max_neurons
                        )
                        cursor = int(getattr(model, "_width_cursor", 0)) % depth
                        first_rr_layer = next(
                            (
                                widths[(cursor + offset) % depth]
                                for offset in range(depth)
                                if widths[(cursor + offset) % depth] < min(widths) + increment
                                and widths[(cursor + offset) % depth] < max_width
                            ),
                            None,
                        )
                        expected_rr = (
                            first_rr_layer is not None
                            and first_rr_layer + increment <= max_width
                            and hidden_and_output_units + increment <= max_neurons
                        )
                        assert can_grow_width_all_layers(
                            model, increment, max_width, max_neurons,
                        ) is expected_all
                        assert can_grow_width_round_robin(
                            model, increment, max_width, max_neurons,
                        ) is expected_rr


def test_dedicated_round_robin_does_not_clip_configured_increment_at_width_cap():
    model = TabularMLP(2, [4, 4], 1)
    assert not can_grow_width_round_robin(model, 3, 5, 100)
    assert expand_width_den_round_robin(model, 3, 5) is None

    # Generic candidate search has a separately documented clamping rule.
    generic = expand_width_round_robin(model, 3, 5)
    assert generic is not None
    assert generic.hidden_widths == [5, 4]


def test_preprocessor_train_only_missing_unknown_and_schema():
    train = pd.DataFrame({"n": [1.0, np.nan, 3.0], "c": ["a", "b", None]})
    prep = TabularPreprocessor().fit(train)
    transformed = prep.transform(pd.DataFrame({"c": ["unseen", None], "n": [1000.0, np.nan]}))
    assert transformed.shape == (2, prep.output_dim)
    assert transformed[0, 1] == 1  # unseen category uses explicit unknown column
    assert transformed[1, 2] == 1  # missing category has its own learned indicator
    assert transformed[1, 0] == pytest.approx(0.0)  # training median is 2, training mean is 2
    with pytest.raises(ValueError, match="schema mismatch"):
        prep.transform(pd.DataFrame({"n": [1.0], "other": ["x"]}))
    array_prep = TabularPreprocessor(feature_schema=["numeric", "categorical"])
    array_prep.fit(np.array([[1, "x"], [2, "y"]], dtype=object))
    assert array_prep.output_dim == 4
    assert all(col.kind == "numeric" for col in TabularPreprocessor().fit(np.array([[1, 2], [3, 4]])).columns_)
    with pytest.raises((ValueError, TypeError)):
        TabularPreprocessor().fit(np.array([[1, "x"], [2, "y"]], dtype=object))


def test_classification_regression_custom_task_and_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"a": rng.normal(size=40), "kind": np.where(np.arange(40) % 2, "odd", "even")})
    y = np.where(X.a.to_numpy() > 0, "yes", "no")
    model = TabularADP("ADP-W-All", task=ClassificationTask(), settings=tiny_settings())
    stage_checkpoint = tmp_path / "stage.pt"
    model.fit(X[:32], y[:32], validation_data=(X[32:], y[32:]), checkpoint_path=stage_checkpoint)
    assert model.task.output_spec.output_dim == 2
    assert model.task.output_spec.semantics == "ClassificationTask"
    assert model.predict(X[32:]).shape == (8,)
    assert model.predict_proba(X[32:]).shape == (8, 2)
    import torch
    checkpoint_record = torch.load(stage_checkpoint, weights_only=False)
    checkpoint = checkpoint_record["payload"]
    assert checkpoint["configuration_id"] == "ADP-W-All"
    assert checkpoint["search_status"] == "completed"
    assert checkpoint["current_architecture"] and checkpoint["best_architecture"]
    assert checkpoint["global_best_model_snapshot"]
    assert checkpoint["counters"] and checkpoint["stage_state"]
    assert checkpoint["optimizer_state"] and checkpoint["scheduler_state"] and checkpoint["rng_state"]
    assert "permutation" in checkpoint["data_position"]
    path = tmp_path / "classifier.pt"
    model.save(path)
    saved_artifact = torch.load(path, weights_only=False)
    assert saved_artifact["format_version"] == 2
    assert saved_artifact["model_configuration"] == model.config.signature_dict()
    loaded = TabularADP.load(path)
    np.testing.assert_array_equal(model.predict(X[32:]), loaded.predict(X[32:]))

    regression = TabularADP("ADP-D-Dcur", task=RegressionTask(), settings=tiny_settings(patience_width_exp=0))
    regression.fit(X.a.to_numpy()[:, None][:32], X.a.to_numpy()[:32],
                   validation_data=(X.a.to_numpy()[:, None][32:], X.a.to_numpy()[32:]))
    assert regression.predict(X.a.to_numpy()[:, None][32:]).shape == (8,)

    custom = CallableTask(1, lambda o, t: torch.nn.functional.l1_loss(o, t.reshape(-1, 1)),
                          lambda o, t: float(torch.nn.functional.l1_loss(o, t.reshape(-1, 1))),
                          lambda o: o[:, 0], task_name="mae")
    custom_model = TabularADP(task=custom, settings=tiny_settings(patience_width_exp=0))
    numeric = np.arange(20, dtype=np.float32)[:, None]
    custom_model.fit(numeric[:16], numeric[:16, 0], validation_data=(numeric[16:], numeric[16:, 0]))
    assert custom_model.predict(numeric[16:]).shape == (4,)
    custom_model.save(tmp_path / "custom.pt")
    with pytest.raises(ValueError, match="custom TaskAdapter"):
        TabularADP.load(tmp_path / "custom.pt")
    assert TabularADP.load(tmp_path / "custom.pt", task=CallableTask(
        1, custom.loss_fn, custom.score_fn, custom.prediction_fn, task_name="mae"
    )).predict(numeric[16:]).shape == (4,)

    class PenalizedTask(CallableTask):
        def __init__(self):
            super().__init__(1, custom.loss_fn, custom.score_fn, custom.prediction_fn, task_name="penalized")
            self.penalty_calls = 0

        def auxiliary_loss(self, outputs, targets):
            self.penalty_calls += 1
            return outputs.new_tensor(0.01)

    penalized = PenalizedTask()
    generic = TabularADP("ADP-WD-RR-D1", task=penalized,
                        settings=tiny_settings(patience_width_exp=0, patience_depth_exp=0,
                                               max_epochs=1, lambda_eq=1.0))
    generic.fit(numeric[:16], numeric[:16, 0], validation_data=(numeric[16:], numeric[16:, 0]))
    assert penalized.penalty_calls > 0


def test_generic_search_penalty_weights_follow_documented_gate_and_components():
    class ComponentTask(RegressionTask):
        def __init__(self):
            super().__init__()
            self.component_calls = 0

        def auxiliary_loss_components(self, outputs, targets):
            self.component_calls += 1
            return outputs.new_tensor(2.0), outputs.new_tensor(3.0)

    task = ComponentTask()
    task.fit_targets(np.array([0.0]))
    outputs = torch.tensor([[2.0]])
    targets = torch.tensor([[0.0]])

    prediction_only = task.adp_search_loss(
        outputs, targets, lambda_loss=7.0, lambda_eq=0.0, lambda_ineq=0.0,
    )
    assert prediction_only.item() == pytest.approx(4.0)
    assert task.component_calls == 0  # source bypasses auxiliary terms at zero weights

    weighted = task.adp_search_loss(
        outputs, targets, lambda_loss=0.5, lambda_eq=2.0, lambda_ineq=3.0,
    )
    assert weighted.item() == pytest.approx(15.0)
    assert task.component_calls == 1

    with pytest.raises(ValueError, match="invalid optimizer settings"):
        tiny_settings(lambda_eq=-0.1)


def test_checkpoint_signature_rejection(tmp_path):
    from simple_adp import CheckpointStore, stable_signature
    store = CheckpointStore(tmp_path / "checkpoint.pt")
    signature = stable_signature({"model_id": "ADP-W-All", "settings": {"max_width": 20}})
    store.save({"stage_state": {"epoch": 1}}, signature=signature)
    assert store.load(signature=signature)["stage_state"]["epoch"] == 1
    incompatible = stable_signature({"model_id": "ADP-W-RR", "settings": {"max_width": 20}})
    with pytest.raises(ValueError, match="exact-resume failure: checkpoint configuration/data signature"):
        store.load(signature=incompatible)


def test_estimator_custom_generic_mutation_is_checkpoint_identity(tmp_path):
    X = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = X[:, 0] * 0.25

    def custom_depth(model, _settings):
        return TabularMLP(model.input_dim, [*model.hidden_widths, 3], model.output_dim)

    settings = tiny_settings(
        max_width=4, max_depth=4, max_candidates=1,
        patience_depth_exp=1, max_epochs=1,
    )
    checkpoint = tmp_path / "custom-mutation.pt"
    estimator = TabularADP(
        "ADP-WD-RR-D1", task=RegressionTask(), settings=settings,
        generic_deepen_fn=custom_depth, generic_mutation_id="depth-policy-v1",
    )
    estimator.fit(X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint)
    assert estimator.history_[1]["architecture"] == [4, 4, 3]

    changed_policy = TabularADP(
        "ADP-WD-RR-D1", task=RegressionTask(), settings=settings,
        generic_deepen_fn=custom_depth, generic_mutation_id="depth-policy-v2",
    )
    with pytest.raises(ValueError, match="exact-resume failure: checkpoint configuration/data signature"):
        changed_policy.fit(
            X[:8], y[:8], validation_data=(X[8:], y[8:]), checkpoint_path=checkpoint,
        )


@pytest.mark.parametrize(
    ("saved_record", "expected_message"),
    [
        (b"not a torch checkpoint", "checkpoint could not be loaded"),
        (["not", "a", "mapping"], "checkpoint envelope is not a mapping"),
        ({"format_version": 999, "signature": "sig", "payload": {}}, "unsupported checkpoint format"),
        ({"format_version": 1, "signature": "sig", "payload": []}, "checkpoint payload is not a mapping"),
    ],
)
def test_invalid_estimator_checkpoints_are_tagged_exact_resume_failures(
    tmp_path, saved_record, expected_message,
):
    from simple_adp.checkpoint import CheckpointStore

    path = tmp_path / "invalid.checkpoint.pt"
    if isinstance(saved_record, bytes):
        path.write_bytes(saved_record)
    else:
        torch.save(saved_record, path)

    with pytest.raises(ValueError, match=f"exact-resume failure: {expected_message}"):
        CheckpointStore(path).load(signature="sig")


def test_checkpoint_store_fsyncs_file_and_directory(tmp_path, monkeypatch):
    import os
    import stat
    import simple_adp.checkpoint as checkpoint_module

    fsync_targets = []
    original_fsync = os.fsync

    def observe_fsync(fd):
        fsync_targets.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        return original_fsync(fd)

    monkeypatch.setattr(checkpoint_module.os, "fsync", observe_fsync)
    store = checkpoint_module.CheckpointStore(tmp_path / "atomic.pt")
    store.save({"stage": 1}, signature="sig")
    assert fsync_targets == [False, True]


def test_rng_capture_does_not_initialize_cuda_for_uninitialized_runtime(monkeypatch):
    import simple_adp.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(checkpoint_module.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        checkpoint_module.torch.cuda, "get_rng_state_all",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA RNG should not be initialized")),
    )
    state = checkpoint_module.capture_rng_state()
    assert state["cuda"] == []
