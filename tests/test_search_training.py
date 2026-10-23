import copy
from itertools import product

import numpy as np
import pytest
import torch

from simple_adp import (
    MODEL_REGISTRY, DedicatedController, ModelConfig, SearchController, SearchSettings, StageResult,
    StageTrainer, TabularMLP, rebuild_mlp,
)


class ScalarTask:
    """Small protocol-compatible task for controller-only tests."""
    def encode_targets(self, targets, *, device):
        return torch.as_tensor(targets, dtype=torch.float32, device=device)

    def loss(self, output, target):
        return torch.nn.functional.mse_loss(output, target)

    def validation_score(self, output, target):
        return float(torch.nn.functional.mse_loss(output, target))

    def auxiliary_loss(self, output, target):
        return output.new_zeros(())


def settings(**kwargs):
    values = dict(init_width=4, init_depth=2, max_epochs=2, patience_es=2,
                  patience_width_exp=2, patience_depth_exp=0, max_width=8,
                  max_depth=4, max_neurons=100, batch_size=4)
    values.update(kwargs)
    return SearchSettings(**values)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("all_layers", 17), ("round_robin", 16)],
)
def test_scalar_width_summary_depends_on_growth_strategy(strategy, expected):
    from simple_adp.growth import scalar_width

    model = TabularMLP(1, [17, 16, 16], 1)
    assert scalar_width(model, strategy) == expected


def test_lgnd_stops_at_first_failure_and_advances_forward(monkeypatch):
    controller = SearchController(ScalarTask(), settings(patience_width_exp=0, delta_width=0.5))
    scores = iter([1.0, 0.9, 0.9, 0.1])

    def stage(model, arrays, **kwargs):
        score = next(scores)
        return StageResult(score, model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(MODEL_REGISTRY["LGND-W-Greedy"], x, y, x, y,
                            input_dim=1, output_dim=1)
    # Positive DEN delta is ignored: .9 is accepted. A later flat score is the
    # first failure; score .1 is never visited.
    assert [entry["score"] for entry in result.history] == [1.0, 0.9, 0.9]
    assert result.model.hidden_widths == [5, 5]


def test_lgnd_matches_bounded_independent_trajectory_oracle(monkeypatch):
    """Cross-check LG-ND strict descent, capacity, and retained forward state."""
    score_patterns = ((), (9.0,), (10.0,), (9.0, 10.0), (9.0, 9.0), (9.0, 11.0))
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)

    for init_width, depth, increment, width_steps, budget_steps, patience, delta, pattern in product(
        (1, 2), (1, 2), (1, 2), (0, 1, 2), (0, 1, 2), (0, 2),
        (0.0, 0.5, 2.0), score_patterns,
    ):
        config = MODEL_REGISTRY["LGND-W-Greedy"]
        settings_value = settings(
            init_width=init_width, init_depth=depth, ex_k_width=increment,
            max_width=init_width + width_steps * increment,
            max_depth=depth,
            max_neurons=init_width * depth + 1 + budget_steps * depth * increment,
            patience_width_exp=patience, delta_width=delta,
        )
        controller = SearchController(ScalarTask(), settings_value)
        observed: list[tuple[list[int], float]] = []

        def stage(model, _arrays, **kwargs):
            index = len(observed)
            score = 10.0 if index == 0 else (
                pattern[index - 1] if index - 1 < len(pattern) else 10.0
            )
            observed.append((list(model.hidden_widths), score))
            return StageResult(score, model.snapshot(), 1, 1, {})

        monkeypatch.setattr(controller, "_stage", stage)
        result = controller.run(
            config, x, y, x, y, input_dim=1, output_dim=1,
        )

        current_width = init_width
        best_width = init_width
        best_score = 10.0
        expected = [([init_width] * depth, best_score)]
        candidate_scores = (*pattern, *(10.0 for _ in range(2 - len(pattern))))
        for score in candidate_scores:
            proposed_width = current_width + increment
            if proposed_width > settings_value.max_width:
                break
            if proposed_width * depth + 1 > settings_value.max_neurons:
                break
            current_width = proposed_width
            expected.append(([current_width] * depth, score))
            if score < best_score:
                best_score = score
                best_width = current_width
            else:
                break

        assert observed == expected
        assert result.model.hidden_widths == [best_width] * depth
        assert result.score == best_score
        assert result.state["current_architecture"] == [current_width] * depth


def test_lgnd_round_robin_override_keeps_all_layer_capacity_precheck():
    config = ModelConfig("LGND-W-RR-compat", "lgnd", "width", "round_robin")
    settings_value = settings(init_width=4, max_width=4, ex_k_width=1)
    seen = []

    def stage(model, arrays, **kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(0.5, model.snapshot(), 1, 1, {})

    controller = DedicatedController(settings_value, stage)
    arrays = tuple(torch.zeros((2, 1)) for _ in range(4))
    result = controller.run(TabularMLP(1, [4, 3], 1), arrays, config)
    # RR could legally widen [4,3] to [4,4], but the LG-ND source check assumes
    # both layers need room for +1 and therefore stops before proposing it.
    assert seen == [[4, 3]]
    assert result.state["current_architecture"] == [4, 3]


def test_lgnd_resumed_failed_expansion_stops_before_next_growth():
    config = MODEL_REGISTRY["LGND-W-Greedy"]
    cfg = settings(init_width=4, init_depth=2, max_width=7, max_neurons=100)
    best = TabularMLP(1, [4, 4], 1)
    resumed_candidate = TabularMLP(1, [5, 5], 1)
    arrays = tuple(torch.zeros((2, 1)) for _ in range(4))
    seen = []

    def stage(model, _arrays, *, generic, resume_state=None, search_context):
        seen.append((list(model.hidden_widths), resume_state))
        return StageResult(0.6, model.snapshot(), 1, 1, {})

    resume_payload = {
        "search_context": {
            "controller": config.id,
            "phase": "width",
            "current_architecture": [5, 5],
            "global_best_architecture": [4, 4],
            "global_best_score": 0.5,
            "global_best_snapshot": best.snapshot(),
            "counters": {"width_failures": 0},
            "growth_cursor": 0,
            "history": [{"architecture": [4, 4], "score": 0.5,
                          "accepted": True, "stage": "initial"}],
        },
        "stage_state": {"resume_marker": "interrupted-width-stage"},
    }
    controller = DedicatedController(cfg, stage)
    result = controller.run(
        resumed_candidate, arrays, config, resume_payload=resume_payload,
    )

    assert seen == [([5, 5], {"resume_marker": "interrupted-width-stage"})]
    assert [entry["architecture"] for entry in result.history] == [[4, 4], [5, 5]]
    assert result.history[-1]["accepted"] is False
    assert result.model.hidden_widths == [4, 4]
    assert result.score == pytest.approx(0.5)
    assert result.state["current_architecture"] == [5, 5]


def test_completed_search_checkpoint_restores_global_best_without_training(monkeypatch):
    config = MODEL_REGISTRY["ADP-W-All"]
    cfg = settings(init_width=4, init_depth=2)
    best = TabularMLP(1, [3, 2], 1)
    with torch.no_grad():
        best.hidden_layers[0].weight.fill_(0.125)
        best.hidden_layers[0].bias.fill_(-0.25)
    history = [{"architecture": [3, 2], "score": 0.375, "accepted": True, "stage": "width"}]
    state = {"best_architecture": [3, 2], "current_architecture": [4, 2]}
    payload = {
        "search_status": "completed",
        "global_best_model_snapshot": best.snapshot(cursor=1),
        "global_best_score": 0.375,
        "search_history": history,
        "search_state": state,
    }
    controller = SearchController(ScalarTask(), cfg)
    monkeypatch.setattr(
        controller, "_stage",
        lambda *args, **kwargs: pytest.fail("completed search checkpoint started a new stage"),
    )
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        config, x, x, x, x, input_dim=1, output_dim=1,
        resume_payload=payload,
    )

    assert result.model.hidden_widths == [3, 2]
    assert result.model._width_cursor == 1
    assert result.score == pytest.approx(0.375)
    assert result.history == history
    assert result.state == state
    torch.testing.assert_close(
        result.model.hidden_layers[0].weight,
        best.hidden_layers[0].weight,
    )


def test_architecture_improvement_equal_to_delta_is_a_failure(monkeypatch):
    controller = SearchController(ScalarTask(), settings(
        patience_width_exp=1, delta_width=0.2, max_width=8,
    ))
    scores = iter([1.0, 0.8])
    monkeypatch.setattr(controller, "_stage", lambda model, arrays, **kwargs:
                        StageResult(next(scores), model.snapshot(), 1, 1, {}))
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(MODEL_REGISTRY["ADP-W-All"], x, x, x, x,
                            input_dim=1, output_dim=1)
    assert result.history[-1]["accepted"] is False  # strict comparison: 0.8 is not < 1.0 - 0.2
    assert result.score == pytest.approx(1.0)
    assert result.model.hidden_widths == [4, 4]


@pytest.mark.parametrize(
    ("order", "expected_stages", "expected_architectures"),
    [
        ("width", ["initial", "width"], [[4, 4], [5, 4]]),
        ("depth", ["initial", "depth"], [[4, 4], [4, 4, 1]]),
    ],
)
def test_generic_search_supports_standalone_width_and_depth_modes(
    monkeypatch, order, expected_stages, expected_architectures,
):
    cfg = ModelConfig(
        f"Generic-{order.title()}", "phase_search", order,
        width_strategy="round_robin", generic=True,
    )
    controller = SearchController(
        ScalarTask(), settings(
            init_width=4, init_depth=2, max_width=5, max_depth=3,
            patience_width_exp=1, patience_depth_exp=1, max_candidates=1,
        ), generic_profile="corrected",
    )
    seen = []

    def stage(model, arrays, **kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(1.0 if len(seen) == 1 else 0.5, model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(cfg, x, x, x, x, input_dim=1, output_dim=1)

    assert seen == expected_architectures
    assert [row["stage"] for row in result.history] == expected_stages


def test_round_robin_den_traverses_failed_shape_from_cursor(monkeypatch):
    controller = SearchController(ScalarTask(), settings(
        init_width=4, init_depth=2, ex_k_width=1, max_width=6,
        patience_width_exp=2, max_neurons=100,
    ))
    scores = iter([1.0, 0.9, 1.1, 0.8, 1.2])
    seen = []

    def stage(model, arrays, **kwargs):
        seen.append((list(model.hidden_widths), int(getattr(model, "_width_cursor", 0))))
        return StageResult(next(scores), model.snapshot(cursor=int(getattr(model, "_width_cursor", 0))), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(MODEL_REGISTRY["ADP-W-RR"], x, x, x, x,
                            input_dim=1, output_dim=1)
    assert seen == [([4, 4], 0), ([5, 4], 1), ([5, 5], 0), ([6, 5], 1), ([6, 6], 0)]
    assert result.state["current_architecture"] == [6, 6]
    assert result.model.hidden_widths == [6, 5]
    assert result.score == pytest.approx(0.8)


def test_patience_delta_failure_traversal_and_final_global_best(monkeypatch):
    controller = SearchController(ScalarTask(), settings(
        patience_width_exp=2, delta_width=0.05, max_width=8,
    ))
    scores = iter([1.0, 0.8, 0.79, 0.78])

    def stage(model, arrays, **kwargs):
        return StageResult(next(scores), model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(MODEL_REGISTRY["ADP-W-All"], x, y, x, y,
                            input_dim=1, output_dim=1)
    # 0.79 fails the 0.05 margin, but traversal continues to 0.78 and then stops
    # after the second consecutive failure; accepted best remains the 0.8 model.
    assert [row["score"] for row in result.history] == [1.0, 0.8, 0.79, 0.78]
    assert result.history[2]["accepted"] is False
    assert result.history[3]["accepted"] is False
    assert result.model.hidden_widths == [5, 5]
    assert result.score == pytest.approx(0.8)
    assert result.state["current_architecture"] == [7, 7]


@pytest.mark.parametrize("model_id", ["ADP-W-All", "ADP-D-Dcur"])
@pytest.mark.parametrize("patience", range(4))
@pytest.mark.parametrize("delta", [0.0, 0.1, 0.2])
def test_dedicated_patience_margin_and_capacity_finite_matrix(
    monkeypatch, model_id, patience, delta,
):
    is_width = model_id == "ADP-W-All"
    cfg = settings(
        init_width=2, init_depth=1, ex_k_width=1,
        max_width=5, max_depth=4, max_neurons=16,
        patience_width_exp=patience if is_width else 0,
        patience_depth_exp=patience if not is_width else 0,
        delta_width=delta if is_width else 0.0,
        delta_depth=delta if not is_width else 0.0,
    )
    controller = SearchController(ScalarTask(), cfg)
    scores = (1.0, 0.9, 0.8, 0.75)
    seen = []

    def stage(model, _arrays, **_kwargs):
        index = len(seen)
        seen.append(list(model.hidden_widths))
        return StageResult(scores[index], model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        MODEL_REGISTRY[model_id], x, x, x, x,
        input_dim=1, output_dim=1,
    )

    # Small independent oracle for the documented strict global-best margin,
    # consecutive-failure counter, retained forward traversal, and hard cap.
    initial_widths = [2]
    expected_history = [{
        "architecture": initial_widths,
        "score": scores[0],
        "accepted": True,
        "stage": "initial",
    }]
    best_score = scores[0]
    best_widths = initial_widths
    current_widths = initial_widths
    failures = 0
    candidate_architectures = (
        [[width] for width in (3, 4, 5)] if is_width
        else [[2] * depth for depth in (2, 3, 4)]
    )
    for candidate_index, architecture in enumerate(candidate_architectures, start=1):
        if failures >= patience:
            break
        score = scores[candidate_index]
        accepted = score < best_score - delta
        expected_history.append({
            "architecture": architecture,
            "score": score,
            "accepted": accepted,
            "stage": "width" if is_width else "depth",
        })
        current_widths = architecture
        if accepted:
            best_score = score
            best_widths = architecture
            failures = 0
        else:
            failures += 1

    assert result.history == expected_history
    assert seen == [entry["architecture"] for entry in expected_history]
    assert result.score == pytest.approx(best_score)
    assert result.model.hidden_widths == best_widths
    assert result.state["current_architecture"] == current_widths
    counter = "width_failures" if is_width else "depth_failures"
    assert result.state[counter] == failures


@pytest.mark.parametrize("patience", range(4))
@pytest.mark.parametrize("delta", [0.0, 0.05, 0.1])
def test_round_robin_den_controller_matches_margin_patience_and_cursor_oracle(
    monkeypatch, patience, delta,
):
    """Cross the RR controller's proposal cursor with strict-best patience rules."""
    cfg = settings(
        init_width=2, init_depth=2, ex_k_width=1,
        max_width=4, max_neurons=32,
        patience_width_exp=patience, delta_width=delta,
    )
    controller = SearchController(ScalarTask(), cfg)
    scores = (1.0, 0.9, 0.86, 0.8, 0.79)
    proposals = (([3, 2], 1), ([3, 3], 0), ([4, 3], 1), ([4, 4], 0))
    seen = []

    def stage(model, _arrays, **_kwargs):
        index = len(seen)
        seen.append((list(model.hidden_widths), int(getattr(model, "_width_cursor", 0))))
        return StageResult(
            scores[index], model.snapshot(cursor=int(getattr(model, "_width_cursor", 0))),
            1, 1, {},
        )

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        MODEL_REGISTRY["ADP-W-RR"], x, x, x, x,
        input_dim=1, output_dim=1,
    )

    expected_seen = [([2, 2], 0)]
    expected_history = [{
        "architecture": [2, 2], "score": scores[0],
        "accepted": True, "stage": "initial",
    }]
    best_architecture, best_score = [2, 2], scores[0]
    current_architecture, failures = [2, 2], 0
    for index, (architecture, cursor) in enumerate(proposals, start=1):
        if failures >= patience:
            break
        score = scores[index]
        accepted = score < best_score - delta
        expected_seen.append((architecture, cursor))
        expected_history.append({
            "architecture": architecture, "score": score,
            "accepted": accepted, "stage": "width",
        })
        current_architecture = architecture
        if accepted:
            best_architecture, best_score, failures = architecture, score, 0
        else:
            failures += 1

    assert seen == expected_seen
    assert result.history == expected_history
    assert result.state["current_architecture"] == current_architecture
    assert result.state["growth_cursor"] == expected_seen[-1][1]
    assert result.model.hidden_widths == best_architecture
    assert result.score == pytest.approx(best_score)
    assert result.state["width_failures"] == failures


@pytest.mark.parametrize(
    ("model_id", "expected_widths", "generic_profile"),
    [
        ("ADP-W-All", [3, 3], "faithful"),
        ("ADP-WD-RR-D1", [3, 2], "corrected"),
    ],
)
def test_global_best_snapshot_restoration_does_not_shift_growth_rng(
    monkeypatch, model_id, expected_widths, generic_profile,
):
    cfg = settings(
        init_width=2, init_depth=2, max_width=3, max_depth=3,
        patience_width_exp=1, patience_depth_exp=1, max_candidates=1,
    )
    torch.manual_seed(cfg.seed)
    initial = TabularMLP(1, [cfg.init_width] * cfg.init_depth, 1)
    rebuild_mlp(initial, expected_widths)
    expected_rng = torch.get_rng_state()

    controller = SearchController(
        ScalarTask(), cfg, generic_profile=generic_profile,
    )
    scores = iter((0.5, 0.4))
    monkeypatch.setattr(
        controller, "_stage",
        lambda model, arrays, **kwargs: StageResult(
            next(scores), model.snapshot(cursor=int(getattr(model, "_width_cursor", 0))), 1, 1, {},
        ),
    )
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        MODEL_REGISTRY[model_id], x, x, x, x,
        input_dim=1, output_dim=1,
    )

    assert result.model.hidden_widths == expected_widths
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_progressive_and_matched_width_depth_proposals(monkeypatch):
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)
    calls = []

    def stage(model, arrays, **kwargs):
        calls.append(list(model.hidden_widths))
        score = 1.0 if len(calls) == 1 else 0.5
        return StageResult(score, model.snapshot(), 1, 1, {})

    progressive = SearchController(ScalarTask(), settings(patience_depth_exp=1, max_depth=3))
    monkeypatch.setattr(progressive, "_stage", stage)
    result = progressive.run(MODEL_REGISTRY["ADP-D-D1"], x, y, x, y,
                             input_dim=1, output_dim=1)
    assert calls == [[4, 4], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]]
    assert result.model.hidden_widths == [4, 4, 4]

    calls.clear()
    matched = SearchController(ScalarTask(), settings(patience_depth_exp=1, max_depth=3))
    monkeypatch.setattr(matched, "_stage", stage)
    matched_result = matched.run(MODEL_REGISTRY["ADP-D-Dcur"], x, y, x, y,
                                 input_dim=1, output_dim=1)
    assert calls == [[4, 4], [4, 4, 4]]
    assert matched_result.model.hidden_widths == [4, 4, 4]


@pytest.mark.parametrize("model_id", ["ADP-D-D1", "ADP-D-Dcur"])
def test_depth_candidate_equal_to_delta_is_failure_and_restores_global_best(monkeypatch, model_id):
    controller = SearchController(ScalarTask(), settings(
        patience_depth_exp=1, max_depth=3, delta_depth=0.2,
    ))
    # Progressive insertion trains three stepping stones before evaluating
    # the completed candidate. Their scores must not alter the architecture
    # level global best; the completed 0.8 score is exactly the 0.2 margin.
    scores = iter([1.0, 0.1, 0.1, 0.1, 0.8] if model_id == "ADP-D-D1" else [1.0, 0.8])
    monkeypatch.setattr(controller, "_stage", lambda model, arrays, **kwargs:
                        StageResult(next(scores), model.snapshot(), 1, 1, {}))
    x = np.zeros((2, 1), dtype=np.float32)

    result = controller.run(MODEL_REGISTRY[model_id], x, x, x, x,
                            input_dim=1, output_dim=1)

    assert result.history[-1]["accepted"] is False
    assert result.score == pytest.approx(1.0)
    assert result.model.hidden_widths == [4, 4]


def test_progressive_depth_stepping_scores_do_not_replace_global_best(monkeypatch):
    controller = SearchController(ScalarTask(), settings(
        patience_depth_exp=1, max_depth=3, delta_depth=0.0,
    ))
    scores = iter([0.5, 0.1, 0.05, 0.03, 0.6])
    seen = []

    def stage(model, arrays, **kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(next(scores), model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(MODEL_REGISTRY["ADP-D-D1"], x, x, x, x,
                            input_dim=1, output_dim=1)
    assert seen == [[4, 4], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]]
    assert [entry["score"] for entry in result.history] == [0.5, 0.6]
    assert result.score == pytest.approx(0.5)
    assert result.model.hidden_widths == [4, 4]
    assert result.state["current_architecture"] == [4, 4, 4]


def test_progressive_depth_capacity_stop_retains_last_forward_stepping_architecture(monkeypatch):
    controller = SearchController(ScalarTask(), settings(
        init_width=2, init_depth=2, max_neurons=6, max_depth=3,
        patience_depth_exp=2,
    ))
    seen = []

    def stage(model, _arrays, **_kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(0.5 if len(seen) == 1 else 0.1, model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        MODEL_REGISTRY["ADP-D-D1"], x, x, x, x,
        input_dim=1, output_dim=1,
    )

    # The appended width-1 layer fits the unit budget and is trained, but the
    # next catch-up increment would exceed it. It remains forward state only;
    # it is neither scored as a completed depth candidate nor selected best.
    assert seen == [[2, 2], [2, 2, 1]]
    assert result.history == [{
        "architecture": [2, 2], "score": 0.5,
        "accepted": True, "stage": "initial",
    }]
    assert result.model.hidden_widths == [2, 2]
    assert result.score == pytest.approx(0.5)
    assert result.state["current_architecture"] == [2, 2, 1]


def test_depth_den_crosses_failed_completed_candidate(monkeypatch):
    controller = SearchController(ScalarTask(), settings(
        patience_depth_exp=2, max_depth=4, delta_depth=0.0,
    ))
    scores = iter([0.5, 0.6, 0.4])
    seen = []

    def stage(model, arrays, **kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(next(scores), model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(MODEL_REGISTRY["ADP-D-Dcur"], x, x, x, x,
                            input_dim=1, output_dim=1)
    assert seen == [[4, 4], [4, 4, 4], [4, 4, 4, 4]]
    assert result.history[1]["accepted"] is False
    assert result.model.hidden_widths == [4, 4, 4, 4]
    assert result.score == pytest.approx(0.4)


def test_dedicated_depth_controllers_match_bounded_trajectory_oracle(monkeypatch):
    """Cross depth seeds, caps, patience and margins against a pure oracle."""
    x = np.zeros((2, 1), dtype=np.float32)
    checked = 0

    def score_for(widths):
        weighted = sum((index + 1) * width for index, width in enumerate(widths))
        return ((weighted * 7 + len(widths) * 5 + sum(w * w for w in widths)) % 37) / 100.0

    for (initial_width, depth_extra, budget_extra, patience, delta, model_id) in product(
        (1, 2, 3), (0, 1, 2), (0, 1, 2, 4), (0, 1, 2, 3),
        (0.0, 0.05, 0.15), ("ADP-D-D1", "ADP-D-Dcur"),
    ):
        initial = [initial_width, initial_width]
        max_depth = len(initial) + depth_extra
        max_neurons = sum(initial) + 1 + budget_extra
        progressive = model_id == "ADP-D-D1"
        cfg = settings(
            init_width=initial_width, init_depth=len(initial), max_depth=max_depth,
            max_neurons=max_neurons, patience_depth_exp=patience,
            delta_depth=delta,
        )
        controller = SearchController(ScalarTask(), cfg)
        seen = []

        def stage(model, _arrays, **_kwargs):
            widths = list(model.hidden_widths)
            seen.append(widths)
            return StageResult(score_for(widths), model.snapshot(), 1, 1, {})

        monkeypatch.setattr(controller, "_stage", stage)
        result = controller.run(
            MODEL_REGISTRY[model_id], x, x, x, x,
            input_dim=1, output_dim=1,
        )

        current = list(initial)
        best = list(initial)
        best_score = score_for(tuple(initial))
        failures = 0
        expected_seen = [list(initial)]
        expected_history = [{
            "architecture": list(initial), "score": best_score,
            "accepted": True, "stage": "initial",
        }]
        while failures < patience:
            seed_width = 1 if progressive else current[-1]
            if (len(current) + 1 > max_depth
                    or len(set(current)) != 1
                    or sum(current) + 1 + seed_width > max_neurons):
                break
            candidate = [*current, seed_width]
            if progressive:
                # Each incomplete terminal width is trained as an
                # optimization-only stepping stone. The target-width stage is
                # separately scored by the architecture-level controller.
                while candidate[-1] < current[-1]:
                    expected_seen.append(list(candidate))
                    if sum(candidate) + 1 + 1 > max_neurons:
                        current = list(candidate)
                        candidate = None
                        break
                    candidate[-1] += 1
                if candidate is None:
                    break

            expected_seen.append(list(candidate))
            candidate_score = score_for(tuple(candidate))
            accepted = candidate_score < best_score - delta
            expected_history.append({
                "architecture": list(candidate), "score": candidate_score,
                "accepted": accepted, "stage": "depth",
            })
            current = list(candidate)
            if accepted:
                best = list(candidate)
                best_score = candidate_score
                failures = 0
            else:
                failures += 1

        assert seen == expected_seen
        assert result.history == expected_history
        assert result.model.hidden_widths == best
        assert result.score == pytest.approx(best_score)
        assert result.state["current_architecture"] == current
        assert result.state["best_architecture"] == best
        assert result.state["depth_failures"] == failures
        checked += 1

    assert checked == 864


def test_nested_outer_width_inner_depth_trajectory(monkeypatch):
    scores_by_shape = {(4, 4): 0.5, (5, 5, 1): 0.4}
    seen = []

    def stage(model, arrays, **kwargs):
        shape = tuple(model.hidden_widths)
        seen.append(shape)
        score = scores_by_shape.get(shape, 0.6)
        return StageResult(score, model.snapshot(), 1, 1, {})

    controller = DedicatedController(settings(
        max_depth=3, patience_width_exp=1, patience_depth_exp=1,
    ), stage)
    initial = TabularMLP(1, [4, 4], 1)
    arrays = tuple(torch.zeros((2, 1)) for _ in range(4))
    result = controller.run_nested_width_depth(initial, arrays)
    assert seen[:2] == [(4, 4), (5, 5, 1)]
    assert result.model.hidden_widths == [5, 5, 1]
    assert result.score == pytest.approx(0.4)


def test_nested_depth_seed_can_match_current_width(monkeypatch):
    seen = []

    def stage(model, arrays, **kwargs):
        shape = tuple(model.hidden_widths)
        seen.append(shape)
        score = 0.5 if shape == (4, 4) else 0.4
        return StageResult(score, model.snapshot(), 1, 1, {})

    controller = SearchController(
        ScalarTask(), settings(
            max_width=5, max_depth=3, patience_width_exp=1,
            patience_depth_exp=2,
        ),
    )
    monkeypatch.setattr(controller, "_stage", stage)
    initial = TabularMLP(1, [4, 4], 1)
    arrays = tuple(torch.zeros((2, 1)) for _ in range(4))
    result = controller.run_nested_width_depth(
        initial, arrays, seed_new_depth_to_current_width=True,
    )

    assert seen == [(4, 4), (5, 5, 5)]
    assert result.model.hidden_widths == [5, 5, 5]


def test_nested_controller_honors_round_robin_outer_width_strategy(monkeypatch):
    seen = []

    def stage(model, arrays, **kwargs):
        widths = tuple(model.hidden_widths)
        seen.append(widths)
        score = 0.5 if widths == (2, 2) else 0.4
        return StageResult(score, model.snapshot(), 1, 1, {})

    controller = SearchController(
        ScalarTask(), settings(
            max_width=3, max_depth=3, max_neurons=100,
            patience_width_exp=3, patience_depth_exp=1,
        ),
    )
    monkeypatch.setattr(controller, "_stage", stage)
    initial = TabularMLP(1, [2, 2], 1)
    arrays = tuple(torch.zeros((2, 1)) for _ in range(4))

    result = controller.run_nested_width_depth(
        initial, arrays, width_growth_strategy="round_robin",
    )

    # RR first produces non-uniform [3,2], so the guide's depth insertion
    # precondition blocks that inner sweep. A complete RR cycle restores
    # uniformity and admits [3,3,1].
    assert seen == [(2, 2), (3, 3, 1)]
    assert result.model.hidden_widths == [3, 3, 1]
    assert result.state["current_architecture"] == [3, 3, 3]
    assert result.state["width_growth_strategy"] == "round_robin"
    assert result.state["growth_cursor"] == 0


def test_nested_resume_rejects_changed_width_growth_strategy():
    cfg = settings(max_epochs=2, patience_width_exp=1, patience_depth_exp=1, max_depth=3)
    x = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
    arrays = (x[:4], x[:4], x[4:], x[4:])
    saved = {}

    class StopAfterNestedStage(Exception):
        pass

    def stop(payload):
        saved.clear()
        saved.update(copy.deepcopy(payload))
        if payload.get("stage_status") == "active":
            raise StopAfterNestedStage

    with pytest.raises(StopAfterNestedStage):
        SearchController(ScalarTask(), cfg, checkpoint_callback=stop).run_nested_width_depth(
            TabularMLP(1, [2, 2], 1), arrays,
            width_growth_strategy="round_robin",
        )

    context = saved["search_context"]
    resumed_model = TabularMLP(1, context["current_architecture"], 1)
    with pytest.raises(ValueError, match="nested search settings/data signature"):
        SearchController(ScalarTask(), cfg).run_nested_width_depth(
            resumed_model, arrays, width_growth_strategy="all_layers",
            resume_payload=saved,
        )


def test_nested_round_robin_resume_restores_growth_cursor():
    cfg = settings(
        max_epochs=2, patience_es=2, patience_width_exp=2,
        patience_depth_exp=1, max_width=3, max_depth=3, batch_size=2,
    )
    x = torch.arange(12, dtype=torch.float32).reshape(-1, 1) / 12
    y = x * 0.5
    arrays = (x[:8], y[:8], x[8:], y[8:])

    torch.manual_seed(83)
    full_initial = TabularMLP(1, [2, 2], 1)
    full_initial._width_cursor = 1
    uninterrupted = SearchController(ScalarTask(), cfg).run_nested_width_depth(
        full_initial, arrays, width_growth_strategy="round_robin",
    )

    saved = {}

    class StopAfterNestedBatch(Exception):
        pass

    def interrupt(payload):
        context = payload.get("search_context", {})
        if (context.get("phase") == "inner_depth"
                and payload.get("stage_status") == "active"
                and payload.get("batch_cursor") == 1):
            saved.clear()
            saved.update(copy.deepcopy(payload))
            raise StopAfterNestedBatch

    torch.manual_seed(83)
    interrupted_initial = TabularMLP(1, [2, 2], 1)
    interrupted_initial._width_cursor = 1
    with pytest.raises(StopAfterNestedBatch):
        SearchController(ScalarTask(), cfg, checkpoint_callback=interrupt).run_nested_width_depth(
            interrupted_initial, arrays, width_growth_strategy="round_robin",
        )

    context = saved["search_context"]
    assert context["growth_cursor"] == 1
    resumed_initial = TabularMLP(1, context["current_architecture"], 1)
    resumed = SearchController(ScalarTask(), cfg).run_nested_width_depth(
        resumed_initial, arrays, width_growth_strategy="round_robin",
        resume_payload=saved,
    )
    assert resumed.history == uninterrupted.history
    assert resumed.model.hidden_widths == uninterrupted.model.hidden_widths
    assert resumed.state == uninterrupted.state


def test_nested_outer_width_patience_resets_when_gain_equals_delta(monkeypatch):
    seen = []

    def stage(model, arrays, **kwargs):
        shape = tuple(model.hidden_widths)
        seen.append(shape)
        score = 0.5 if shape == (1, 1) else 0.375 if shape == (2, 2, 1) else 0.5
        return StageResult(score, model.snapshot(), 1, 1, {})

    controller = SearchController(
        ScalarTask(),
        settings(
            max_width=3, max_depth=3, max_neurons=100,
            patience_width_exp=1, patience_depth_exp=1,
            delta_width=0.125, delta_depth=0.0,
        ),
    )
    monkeypatch.setattr(controller, "_stage", stage)
    initial = TabularMLP(1, [1, 1], 1)
    arrays = tuple(torch.zeros((2, 1)) for _ in range(4))

    result = controller.run_nested_width_depth(initial, arrays)

    # Section 6.1 says an outer-width round resets patience when the global
    # best improves by at least delta_width, so equality admits another round.
    assert seen == [(1, 1), (2, 2, 1)]
    assert result.state["current_architecture"] == [3, 3, 2]
    assert result.state["width_failures"] == 1
    assert result.model.hidden_widths == [2, 2, 1]
    assert result.score == pytest.approx(0.375)


def test_nested_controller_resumes_interrupted_inner_depth_candidate():
    cfg = settings(
        max_epochs=3, patience_es=3, patience_width_exp=2,
        patience_depth_exp=2, max_depth=3, batch_size=2,
    )
    x = torch.arange(12, dtype=torch.float32).reshape(-1, 1) / 12
    y = x * 0.5
    arrays = (x[:8], y[:8], x[8:], y[8:])

    torch.manual_seed(71)
    uninterrupted_controller = SearchController(ScalarTask(), cfg)
    uninterrupted = uninterrupted_controller.run_nested_width_depth(
        TabularMLP(1, [2, 2], 1), arrays,
    )

    saved = {}

    class StopAfterInnerBatch(Exception):
        pass

    def interrupt(payload):
        saved.clear()
        saved.update(copy.deepcopy(payload))
        context = payload.get("search_context", {})
        if (context.get("phase") == "inner_depth"
                and payload.get("stage_status") == "active"
                and payload.get("batch_cursor") == 1):
            raise StopAfterInnerBatch

    torch.manual_seed(71)
    interrupted_controller = SearchController(ScalarTask(), cfg, checkpoint_callback=interrupt)
    with pytest.raises(StopAfterInnerBatch):
        interrupted_controller.run_nested_width_depth(TabularMLP(1, [2, 2], 1), arrays)

    context = saved["search_context"]
    assert context["controller"] == "nested_width_depth"
    assert context["phase"] == "inner_depth"
    resume_model = TabularMLP(1, context["current_architecture"], 1)
    incompatible = SearchController(ScalarTask(), settings(**{
        **cfg.as_dict(), "learning_rate": cfg.learning_rate * 2,
    }))
    with pytest.raises(ValueError, match="nested search settings/data signature"):
        incompatible.run_nested_width_depth(resume_model, arrays, resume_payload=saved)
    with pytest.raises(ValueError, match="nested search settings/data signature"):
        SearchController(ScalarTask(), cfg).run_nested_width_depth(
            resume_model, arrays, seed_new_depth_to_current_width=True,
            resume_payload=saved,
        )

    resumed = SearchController(ScalarTask(), cfg).run_nested_width_depth(
        resume_model, arrays, resume_payload=saved,
    )

    assert resumed.model.hidden_widths == uninterrupted.model.hidden_widths
    assert resumed.score == pytest.approx(uninterrupted.score, abs=1e-9)
    assert resumed.history == uninterrupted.history


def test_nested_controller_exhaustive_small_domain_matches_independent_oracle():
    """Enumerate nested DEN knobs and compare every proposal/counter to a pure oracle."""
    initial_architectures = ((1,), (2, 2))
    x = torch.zeros((2, 1), dtype=torch.float32)
    arrays = (x, x, x, x)

    def validation_score(widths):
        weighted = sum((index + 1) * width for index, width in enumerate(widths))
        quadratic = sum(width * width for width in widths)
        return ((weighted * 7 + quadratic * 3) % 31) / 100.0

    tested = 0
    for (
        initial_widths, ex_k_width, width_extra, depth_extra, neuron_headroom,
        seed_current, patience_width, patience_depth, delta_width, delta_depth,
    ) in product(
        initial_architectures,
        (1, 2),
        (0, 2),
        (0, 1, 2),
        (0, 2),
        (False, True),
        (0, 1, 2),
        (0, 1, 2),
        (0.0, 0.07),
        (0.0, 0.07),
    ):
        initial_widths = list(initial_widths)
        initial_depth = len(initial_widths)
        max_width = max(initial_widths) + width_extra * ex_k_width
        max_depth = initial_depth + depth_extra
        neuron_limit = sum(initial_widths) + 1 + neuron_headroom
        cfg = settings(
            init_width=initial_widths[0], init_depth=initial_depth,
            ex_k_width=ex_k_width, max_width=max_width, max_depth=max_depth,
            max_neurons=neuron_limit, patience_width_exp=patience_width,
            patience_depth_exp=patience_depth, delta_width=delta_width,
            delta_depth=delta_depth,
        )
        controller = SearchController(ScalarTask(), cfg)

        def stage(model, _arrays, **_kwargs):
            widths = tuple(model.hidden_widths)
            return StageResult(
                validation_score(widths), model.snapshot(), 1, 1, {},
            )

        controller._stage = stage
        result = controller.run_nested_width_depth(
            TabularMLP(1, initial_widths, 1), arrays,
            seed_new_depth_to_current_width=seed_current,
        )

        # Independent scalar oracle: the source controller first advances the
        # all-layer width state, then performs narrow or matched depth trials.
        current = list(initial_widths)
        best_architecture = list(initial_widths)
        best_score = validation_score(tuple(initial_widths))
        width_failures = 0
        expected_history = [{
            "architecture": list(initial_widths), "score": best_score,
            "accepted": True, "stage": "nested-initial",
        }]
        while width_failures < patience_width:
            if (
                any(width + ex_k_width > max_width for width in current)
                or sum(current) + 1 + len(current) * ex_k_width > neuron_limit
            ):
                break
            before_round_best = best_score
            current = [width + ex_k_width for width in current]
            depth_failures = 0

            while depth_failures < patience_depth:
                seed_width = current[-1] if seed_current else 1
                if (
                    len(set(current)) != 1
                    or len(current) + 1 > max_depth
                    or sum(current) + 1 + seed_width > neuron_limit
                ):
                    break
                current = [*current, seed_width]
                score = validation_score(tuple(current))
                accepted = score < best_score - delta_depth
                expected_history.append({
                    "architecture": list(current), "score": score,
                    "accepted": accepted, "stage": "nested-depth",
                })
                if accepted:
                    best_score = score
                    best_architecture = list(current)
                    depth_failures = 0
                else:
                    depth_failures += 1

            width_failures = (
                0 if before_round_best - best_score >= delta_width
                else width_failures + 1
            )

        assert result.history == expected_history
        assert result.score == pytest.approx(best_score)
        assert result.model.hidden_widths == best_architecture
        assert result.state["current_architecture"] == current
        assert result.state["best_architecture"] == best_architecture
        assert result.state["width_failures"] == width_failures
        tested += 1

    assert tested == 3_456


@pytest.mark.parametrize(
    ("initial_widths", "seed_current", "interrupt_architecture"),
    [
        ([2, 2], False, [2, 2]),       # interrupted initial architecture
        ([1], True, [2, 2]),           # first inner-depth proposal
        ([1], True, [2, 2, 2]),        # later proposal in the same inner sweep
    ],
)
def test_nested_controller_resume_matches_uninterrupted_at_each_stage_position(
    initial_widths, seed_current, interrupt_architecture,
):
    cfg = settings(
        init_width=initial_widths[0], init_depth=len(initial_widths),
        ex_k_width=1, max_width=3 if max(initial_widths) > 1 else 2,
        max_depth=3, max_neurons=64, max_epochs=3, patience_es=3,
        patience_width_exp=2, patience_depth_exp=2, batch_size=2,
    )
    x = torch.arange(12, dtype=torch.float32).reshape(-1, 1) / 12
    y = x * 0.5
    arrays = (x[:8], y[:8], x[8:], y[8:])

    torch.manual_seed(91)
    uninterrupted = SearchController(ScalarTask(), cfg).run_nested_width_depth(
        TabularMLP(1, initial_widths, 1), arrays,
        seed_new_depth_to_current_width=seed_current,
    )

    saved = {}

    class StopAtStage(Exception):
        pass

    def checkpoint(payload):
        context = payload.get("search_context", {})
        if (
            context.get("current_architecture") == interrupt_architecture
            and payload.get("stage_status") == "active"
            and payload.get("batch_cursor") == 1
        ):
            saved.clear()
            saved.update(copy.deepcopy(payload))
            raise StopAtStage

    torch.manual_seed(91)
    with pytest.raises(StopAtStage):
        SearchController(ScalarTask(), cfg, checkpoint_callback=checkpoint).run_nested_width_depth(
            TabularMLP(1, initial_widths, 1), arrays,
            seed_new_depth_to_current_width=seed_current,
        )

    assert saved["search_context"]["current_architecture"] == interrupt_architecture
    resume_model = TabularMLP(1, interrupt_architecture, 1)
    resumed = SearchController(ScalarTask(), cfg).run_nested_width_depth(
        resume_model, arrays, seed_new_depth_to_current_width=seed_current,
        resume_payload=saved,
    )
    assert resumed.model.hidden_widths == uninterrupted.model.hidden_widths
    assert resumed.score == pytest.approx(uninterrupted.score, abs=1e-9)
    assert resumed.history == uninterrupted.history


@pytest.mark.parametrize("interrupt_architecture", [
    [1], [2, 2], [3, 3, 3],
])
def test_nested_controller_resume_matches_each_outer_and_inner_stage(
    interrupt_architecture,
):
    """Resume each reachable nested stage, including later outer-width rounds."""
    cfg = settings(
        init_width=1, init_depth=1, ex_k_width=1,
        max_width=3, max_depth=3, max_neurons=64,
        max_epochs=2, patience_es=2,
        patience_width_exp=4, patience_depth_exp=1, batch_size=2,
        delta_width=100.0, delta_depth=100.0,
    )
    x = torch.arange(12, dtype=torch.float32).reshape(-1, 1) / 12
    y = x * 0.5
    arrays = (x[:8], y[:8], x[8:], y[8:])

    torch.manual_seed(191)
    uninterrupted = SearchController(ScalarTask(), cfg).run_nested_width_depth(
        TabularMLP(1, [1], 1), arrays,
        seed_new_depth_to_current_width=True,
    )
    assert interrupt_architecture in [
        entry["architecture"] for entry in uninterrupted.history
    ]

    saved = {}

    class StopAtStage(Exception):
        pass

    def checkpoint(payload):
        context = payload.get("search_context", {})
        if (
            context.get("current_architecture") == interrupt_architecture
            and payload.get("stage_status") == "active"
            and payload.get("batch_cursor") == 1
        ):
            saved.clear()
            saved.update(copy.deepcopy(payload))
            raise StopAtStage

    torch.manual_seed(191)
    with pytest.raises(StopAtStage):
        SearchController(
            ScalarTask(), cfg, checkpoint_callback=checkpoint,
        ).run_nested_width_depth(
            TabularMLP(1, [1], 1), arrays,
            seed_new_depth_to_current_width=True,
        )

    assert saved["search_context"]["current_architecture"] == interrupt_architecture
    resumed = SearchController(ScalarTask(), cfg).run_nested_width_depth(
        TabularMLP(1, interrupt_architecture, 1), arrays,
        seed_new_depth_to_current_width=True,
        resume_payload=saved,
    )
    assert resumed.model.hidden_widths == uninterrupted.model.hidden_widths
    assert resumed.score == pytest.approx(uninterrupted.score, abs=1e-9)
    assert resumed.history == uninterrupted.history


def test_generic_faithful_and_corrected_global_best_and_capacity(monkeypatch):
    config = MODEL_REGISTRY["ADP-WD-RR-D1"]
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)

    outputs = {}
    for profile in ("faithful", "corrected"):
        controller = SearchController(ScalarTask(), settings(), generic_profile=profile)
        scores = iter([0.5, 0.6, 0.55])

        def stage(model, arrays, **kwargs):
            score = next(scores)
            return StageResult(score, model.snapshot(), 1, 1, {})

        monkeypatch.setattr(controller, "_stage", stage)
        outputs[profile] = controller.run(config, x, y, x, y, input_dim=1, output_dim=1)
    assert outputs["faithful"].score == pytest.approx(0.55)  # reproduces local-comparison defect
    assert outputs["corrected"].score == pytest.approx(0.5)  # retains global best
    assert outputs["faithful"].state["failure_counts"]["width"] == 2
    assert outputs["corrected"].state["failure_counts"]["width"] == 2

    reset_counts = {}
    for profile in ("faithful", "corrected"):
        controller = SearchController(ScalarTask(), settings(), generic_profile=profile)
        scores = iter([0.5, 0.6, 0.4])
        controller.settings = settings(max_candidates=2)
        monkeypatch.setattr(controller, "_stage", lambda model, arrays, **kwargs:
                            StageResult(next(scores), model.snapshot(), 1, 1, {}))
        reset_counts[profile] = controller.run(config, x, y, x, y,
                                               input_dim=1, output_dim=1).state["failure_counts"]["width"]
    assert reset_counts == {"faithful": 2, "corrected": 0}

    constrained = SearchSettings(**{**settings().as_dict(), "max_neurons": 9})
    blocked = SearchController(ScalarTask(), constrained, generic_profile="corrected")
    monkeypatch.setattr(blocked, "_stage", lambda model, arrays, **kwargs:
                        StageResult(0.5, model.snapshot(), 1, 1, {}))
    result = blocked.run(config, x, y, x, y, input_dim=1, output_dim=1)
    assert len(result.history) == 1
    faithful = SearchController(ScalarTask(), constrained, generic_profile="faithful")
    scores = iter([0.5, 0.6, 0.7])
    monkeypatch.setattr(faithful, "_stage", lambda model, arrays, **kwargs:
                        StageResult(next(scores), model.snapshot(), 1, 1, {}))
    result = faithful.run(config, x, y, x, y, input_dim=1, output_dim=1)
    assert len(result.history) > 1  # compatibility behavior omits the documented cap


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
def test_generic_max_candidates_is_a_shared_cross_phase_ceiling(monkeypatch, profile):
    config = MODEL_REGISTRY["ADP-DW-D1-RR"]
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)

    def run(max_candidates):
        controller = SearchController(
            ScalarTask(), settings(
                max_candidates=max_candidates,
                patience_width_exp=1,
                patience_depth_exp=1,
                max_width=8,
                max_depth=4,
            ),
            generic_profile=profile,
        )
        seen = []

        def stage(model, arrays, **kwargs):
            seen.append(list(model.hidden_widths))
            return StageResult(1.0, model.snapshot(), 1, 1, {})

        monkeypatch.setattr(controller, "_stage", stage)
        result = controller.run(
            config, x, y, x, y, input_dim=1, output_dim=1,
        )
        return seen, result

    capped_stages, capped = run(max_candidates=1)
    unlimited_stages, unlimited = run(max_candidates=0)

    assert capped_stages == [[4, 4], [4, 4, 1]]
    assert capped.state["candidate_count"] == 1
    assert capped.state["failure_counts"] == {"depth": 1}

    assert unlimited_stages == [[4, 4], [4, 4, 1], [5, 4, 1]]
    assert unlimited.state["candidate_count"] == 2
    assert unlimited.state["failure_counts"] == {"depth": 1, "width": 1}


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
@pytest.mark.parametrize("model_id", [
    "ADP-WD-RR-D1", "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
    "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR", "ADP-DW-D1-All",
])
def test_generic_mixed_order_phase_transition_preserves_forward_candidate(
    monkeypatch, profile, model_id,
):
    """A failed first phase feeds its current shape into the next phase."""
    config = MODEL_REGISTRY[model_id]
    cfg = settings(
        init_width=2, init_depth=1, ex_k_width=1,
        max_width=4, max_depth=4, max_neurons=100,
        patience_width_exp=1, patience_depth_exp=1,
        delta_width=0.05, delta_depth=0.05, max_candidates=2,
    )
    controller = SearchController(ScalarTask(), cfg, generic_profile=profile)
    scores = iter((1.0, 1.5, 0.9))
    seen = []

    def stage(model, _arrays, **_kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(next(scores), model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        config, x, x, x, x, input_dim=1, output_dim=1,
    )

    if config.order == "width_then_depth":
        first_phase, second_phase = "width", "depth"
        forward = [3]
        if config.depth_seed == "one":
            next_architecture = [3, 1]
        else:
            next_architecture = [3, 3]
    else:
        first_phase, second_phase = "depth", "width"
        forward = [2, 1] if config.depth_seed == "one" else [2, 2]
        if config.width_strategy == "all_layers":
            next_architecture = [width + 1 for width in forward]
        elif config.depth_seed == "one":
            # Generic RR selects the first layer below max_width from its cursor;
            # unlike dedicated RR it does not target min(widths) + k.
            next_architecture = [3, 1]
        else:
            next_architecture = [3, 2]

    assert seen == [[2], forward, next_architecture]
    assert [entry["stage"] for entry in result.history] == [
        "initial", first_phase, second_phase,
    ]
    assert [entry["accepted"] for entry in result.history] == [
        True,
        False,
        True,
    ]
    assert result.history[1]["architecture"] == forward
    assert result.history[2]["architecture"] == next_architecture
    assert result.state["candidate_count"] == 2
    assert result.state["current_architecture"] == next_architecture
    assert result.model.hidden_widths == next_architecture
    assert result.score == pytest.approx(0.9)
    expected_failures = 1 if profile == "faithful" else 0
    assert result.state["failure_counts"] == {
        first_phase: 1,
        second_phase: expected_failures,
    }


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
@pytest.mark.parametrize("patience", range(4))
@pytest.mark.parametrize("delta", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("model_id", [
    "ADP-WD-RR-D1", "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
    "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR", "ADP-DW-D1-All",
])
def test_generic_profile_patience_margin_finite_matrix(
    monkeypatch, profile, patience, delta, model_id,
):
    config = MODEL_REGISTRY[model_id]
    cfg = settings(
        init_width=2, init_depth=1, max_width=5, max_depth=1, max_neurons=8,
        patience_width_exp=patience, patience_depth_exp=0,
        delta_width=delta, delta_depth=0.0,
    )
    controller = SearchController(ScalarTask(), cfg, generic_profile=profile)
    scores = (1.0, 0.9, 0.8, 0.75)
    seen = []

    def stage(model, _arrays, **_kwargs):
        index = len(seen)
        seen.append(list(model.hidden_widths))
        return StageResult(scores[index], model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        config, x, x, x, x, input_dim=1, output_dim=1,
    )

    best_score = current_score = previous_candidate_score = scores[0]
    best_width = current_width = 2
    failures = 0
    expected_history = [{
        "architecture": [2], "score": scores[0], "accepted": True, "stage": "initial",
    }]
    for candidate_index, width in enumerate((3, 4, 5), start=1):
        if failures >= patience:
            break
        score = scores[candidate_index]
        if profile == "faithful":
            # Preserve the source update order: the local minimum is assigned
            # before comparison, so its margin failure counter cannot reset.
            stage_best = min(current_score, score)
            improved_phase = score < stage_best - delta
            accepted = score < previous_candidate_score
            if accepted:
                best_width = width
            best_score = score
            current_score = score
            previous_candidate_score = score
            failures = 0 if improved_phase else failures + 1
        else:
            improved_phase = score < current_score - delta
            accepted = score < best_score - delta
            if accepted:
                best_width = width
                best_score = score
            if improved_phase:
                current_score = score
                failures = 0
            else:
                failures += 1
            previous_candidate_score = score
        current_width = width
        expected_history.append({
            "architecture": [width], "score": score,
            "accepted": accepted, "stage": "width", "profile": profile,
        })

    assert result.history == expected_history
    assert seen == [row["architecture"] for row in expected_history]
    assert result.score == pytest.approx(best_score)
    assert result.model.hidden_widths == [best_width]
    assert result.state["current_architecture"] == [current_width]
    assert result.state["failure_counts"].get("width", 0) == failures


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
@pytest.mark.parametrize("patience", range(4))
@pytest.mark.parametrize("delta", [0.0, 0.1, 0.2])
@pytest.mark.parametrize("model_id", [
    "ADP-WD-RR-D1", "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
    "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR", "ADP-DW-D1-All",
])
def test_generic_depth_phase_margin_matrix_across_mixed_order_models(
    monkeypatch, profile, patience, delta, model_id,
):
    config = MODEL_REGISTRY[model_id]
    cfg = settings(
        init_width=2, init_depth=1, max_width=2, max_depth=5, max_neurons=100,
        patience_width_exp=0, patience_depth_exp=patience,
        delta_width=0.0, delta_depth=delta,
    )
    controller = SearchController(ScalarTask(), cfg, generic_profile=profile)
    scores = (1.0, 0.9, 0.8, 0.75, 0.7)
    seen = []

    def stage(model, _arrays, **_kwargs):
        index = len(seen)
        seen.append(list(model.hidden_widths))
        return StageResult(scores[index], model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(config, x, x, x, x, input_dim=1, output_dim=1)

    seed_width = 1 if config.depth_seed == "one" else 2
    best_score = current_score = previous_candidate_score = scores[0]
    best_architecture = current_architecture = [2]
    failures = 0
    expected_history = [{
        "architecture": [2], "score": scores[0], "accepted": True, "stage": "initial",
    }]
    # Generic depth growth refuses to deepen a non-uniform stack. A width-1
    # seed therefore produces one candidate when the incoming width is 2;
    # matched-width insertion remains uniform and can continue to the cap.
    candidate_depths = range(2, 3) if config.depth_seed == "one" else range(2, 6)
    for candidate_index, depth in enumerate(candidate_depths, start=1):
        if failures >= patience:
            break
        architecture = [2] + [seed_width] * (depth - 1)
        score = scores[candidate_index]
        if profile == "faithful":
            # The compatibility path updates its local minimum before applying
            # the margin and selects against the preceding candidate.
            stage_best = min(current_score, score)
            improved_phase = score < stage_best - delta
            accepted = score < previous_candidate_score
            if accepted:
                best_architecture = architecture
            best_score = score
            current_score = score
            previous_candidate_score = score
            failures = 0 if improved_phase else failures + 1
        else:
            improved_phase = score < current_score - delta
            accepted = score < best_score - delta
            if accepted:
                best_architecture = architecture
                best_score = score
            if improved_phase:
                current_score = score
                failures = 0
            else:
                failures += 1
            previous_candidate_score = score
        current_architecture = architecture
        expected_history.append({
            "architecture": architecture, "score": score,
            "accepted": accepted, "stage": "depth", "profile": profile,
        })

    assert result.history == expected_history
    assert seen == [row["architecture"] for row in expected_history]
    assert result.score == pytest.approx(best_score)
    assert result.model.hidden_widths == best_architecture
    assert result.state["current_architecture"] == current_architecture
    assert result.state["failure_counts"].get("depth", 0) == failures


@pytest.mark.parametrize("profile", ["faithful", "corrected"])
@pytest.mark.parametrize("model_id", [
    "ADP-WD-RR-D1", "ADP-WD-All-Dcur", "ADP-WD-RR-Dcur", "ADP-WD-All-D1",
    "ADP-DW-D1-RR", "ADP-DW-Dcur-All", "ADP-DW-Dcur-RR", "ADP-DW-D1-All",
])
def test_generic_neuron_capacity_profile_matrix_across_mixed_order_models(
    monkeypatch, profile, model_id,
):
    config = MODEL_REGISTRY[model_id]
    is_width_first = config.order == "width_then_depth"
    cfg = settings(
        init_width=2, init_depth=1,
        max_width=3 if is_width_first else 2,
        max_depth=1 if is_width_first else 2,
        max_neurons=3, patience_width_exp=1, patience_depth_exp=1,
    )
    controller = SearchController(ScalarTask(), cfg, generic_profile=profile)
    seen = []

    def stage(model, _arrays, **_kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(0.5, model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(config, x, x, x, x, input_dim=1, output_dim=1)

    # Initial [2] plus output width 1 exactly consumes the three-unit budget.
    # Faithful adp_search omits that check; corrected search rejects the next
    # width/depth candidate before it reaches the trainer.
    if profile == "faithful":
        expected_stages = (
            ["initial", "width"] if is_width_first
            else ["initial", "depth"] + (["width"] if config.depth_seed == "one" else [])
        )
        assert [row["stage"] for row in result.history] == expected_stages
        assert len(seen) == len(expected_stages)
    else:
        assert len(seen) == len(result.history) == 1
    assert result.model.hidden_widths == [2]
    assert result.state["current_architecture"] == seen[-1]


def test_fixed_grid_preserves_candidate_order_ties_and_global_best(monkeypatch):
    controller = SearchController(ScalarTask(), settings(max_depth=4, max_width=8))
    candidates = [[2], [3, 2], [2, 3], [4, 1, 1]]
    scores = iter((0.5, 0.3, 0.3, 0.4))
    seen = []

    def stage(model, _arrays, **_kwargs):
        seen.append(list(model.hidden_widths))
        return StageResult(next(scores), model.snapshot(), 1, 1, {})

    monkeypatch.setattr(controller, "_stage", stage)
    x = np.zeros((2, 1), dtype=np.float32)
    result = controller.run(
        MODEL_REGISTRY["MLP-Fixed"], x, x, x, x,
        input_dim=1, output_dim=1, fixed_grid=candidates,
    )

    assert seen == candidates
    assert [row["accepted"] for row in result.history] == [True, True, False, False]
    assert result.model.hidden_widths == [3, 2]
    assert result.score == pytest.approx(0.3)
    assert result.state["fixed_grid"] == candidates


def test_all_eight_mixed_order_variants_execute_their_declared_mutations(monkeypatch):
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)
    expected = {
        "ADP-WD-RR-D1": ([5, 4], [4, 4, 1], [5, 4, 1]),
        "ADP-WD-All-Dcur": ([5, 5], [4, 4, 4], [5, 5, 5]),
        "ADP-WD-RR-Dcur": ([5, 4], [4, 4, 4], [5, 4, 4]),
        "ADP-WD-All-D1": ([5, 5], [4, 4, 1], [5, 5, 2]),
        "ADP-DW-D1-RR": ([4, 4, 1], None, [5, 4, 1]),
        "ADP-DW-Dcur-All": ([4, 4, 4], None, [5, 5, 5]),
        "ADP-DW-Dcur-RR": ([4, 4, 4], None, [5, 4, 4]),
        "ADP-DW-D1-All": ([4, 4, 1], None, [5, 5, 2]),
    }
    for model_id, (first_candidate, depth_candidate, next_width_candidate) in expected.items():
        cfg = SearchSettings(**{
            **settings(max_candidates=2, max_width=8, patience_depth_exp=1).as_dict(),
        })
        controller = SearchController(ScalarTask(), cfg, generic_profile="corrected")
        seen = []

        def stage(model, arrays, **kwargs):
            seen.append(list(model.hidden_widths))
            return StageResult(1.0, model.snapshot(), 1, 1, {})

        monkeypatch.setattr(controller, "_stage", stage)
        controller.run(MODEL_REGISTRY[model_id], x, y, x, y, input_dim=1, output_dim=1)
        if model_id.startswith("ADP-WD"):
                # A width-to-depth route proposes width before consulting its
                # depth-seeding policy.
            assert seen[1] == first_candidate

            # Saturating width forces the sequential search to enter depth.
            constrained = SearchController(
                ScalarTask(), settings(max_candidates=1, max_width=4, patience_depth_exp=1),
                generic_profile="corrected",
            )
            depth_seen = []
            monkeypatch.setattr(
                constrained, "_stage",
                lambda model, arrays, **kwargs: (
                    depth_seen.append(list(model.hidden_widths))
                    or StageResult(1.0, model.snapshot(), 1, 1, {})
                ),
            )
            constrained.run(MODEL_REGISTRY[model_id], x, y, x, y, input_dim=1, output_dim=1)
            assert depth_seen[1] == depth_candidate
        else:
            assert seen[1:3] == [first_candidate, next_width_candidate]


def test_generic_mutation_hooks_and_minimum_depth_seed_are_configurable(monkeypatch):
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.zeros((2, 1), dtype=np.float32)
    config = MODEL_REGISTRY["ADP-WD-RR-D1"]

    def run_with_hooks(*, settings_override, widen_fn, deepen_fn, mutation_id):
        controller = SearchController(
            ScalarTask(), settings_override, generic_profile="corrected",
            generic_widen_fn=widen_fn, generic_deepen_fn=deepen_fn,
            generic_mutation_id=mutation_id,
        )
        seen = []

        def stage(model, arrays, **kwargs):
            seen.append(list(model.hidden_widths))
            return StageResult(1.0, model.snapshot(), 1, 1, {})

        monkeypatch.setattr(controller, "_stage", stage)
        controller.run(config, x, y, x, y, input_dim=1, output_dim=1)
        return seen

    widened = run_with_hooks(
        settings_override=settings(max_candidates=1),
        widen_fn=lambda model, _settings: TabularMLP(1, [5, 4], 1),
        deepen_fn=None,
        mutation_id="custom-width-v1",
    )
    assert widened[1] == [5, 4]

    deepened = run_with_hooks(
        settings_override=settings(max_candidates=1, max_width=4, patience_depth_exp=1),
        widen_fn=lambda _model, _settings: None,
        deepen_fn=lambda model, _settings: TabularMLP(
            model.input_dim, [*model.hidden_widths, 3], model.output_dim,
        ),
        mutation_id="custom-depth-v1",
    )
    assert deepened[1] == [4, 4, 3]

    default_depth = run_with_hooks(
        settings_override=settings(
            max_candidates=1, max_width=4, patience_depth_exp=1,
            min_new_layer_width=3,
        ),
        widen_fn=lambda _model, _settings: None,
        deepen_fn=None,
        mutation_id="custom-disabled-width-v1",
    )
    assert default_depth[1] == [4, 4, 3]

    without_identity = SearchController(
        ScalarTask(), settings(), generic_widen_fn=lambda model, _settings: model,
    )
    with pytest.raises(ValueError, match="stable mutation_id"):
        without_identity.run(config, x, y, x, y, input_dim=1, output_dim=1)


@pytest.mark.parametrize(
    ("interrupt_epoch", "interrupt_batch"),
    [(0, 0)]
    + [(epoch, batch) for epoch in range(3) for batch in range(1, 5)]
    + [(epoch, 0) for epoch in (1, 2, 3)],
)
def test_stage_optimizer_boundary_reset_and_same_stage_resume(
    interrupt_epoch, interrupt_batch,
):
    task = ScalarTask()
    # A fixed validation score keeps the stage active through its full epoch
    # budget, so every legal checkpoint position below is reached.
    task.validation_score = lambda _outputs, _targets: 1.0
    cfg = settings(max_epochs=3, patience_es=10, batch_size=4)
    x = torch.arange(20, dtype=torch.float32).reshape(-1, 1) / 20
    y = x * 2

    torch.manual_seed(123)
    initial = TabularMLP(1, [3], 1)
    initial_state = copy.deepcopy(initial.state_dict())
    stage_rng = torch.get_rng_state().clone()
    uninterrupted = TabularMLP(1, [3], 1)
    uninterrupted.load_state_dict(initial_state)
    torch.set_rng_state(stage_rng)
    full = StageTrainer(task, cfg).train_stage(uninterrupted, x[:16], y[:16], x[16:], y[16:])

    interrupted = TabularMLP(1, [3], 1)
    interrupted.load_state_dict(initial_state)
    torch.set_rng_state(stage_rng)
    saved = {}

    class StopAfterBatch(Exception):
        pass

    def stop(payload):
        saved.clear()
        saved.update(copy.deepcopy(payload))
        if (payload["stage_status"] == "active"
                and payload["epoch"] == interrupt_epoch
                and payload["batch_cursor"] == interrupt_batch):
            raise StopAfterBatch

    with pytest.raises(StopAfterBatch):
        StageTrainer(task, cfg, checkpoint_callback=stop).train_stage(
            interrupted, x[:16], y[:16], x[16:], y[16:]
        )
    assert saved["optimizer_state"] and saved["scheduler_state"]
    assert (saved["epoch"], saved["batch_cursor"]) == (
        interrupt_epoch, interrupt_batch,
    )
    resumed = StageTrainer(task, cfg).train_stage(
        interrupted, x[:16], y[:16], x[16:], y[16:], resume_state=saved
    )
    assert full.best_score == pytest.approx(resumed.best_score, abs=1e-9)
    for key, value in full.best_snapshot["state_dict"].items():
        torch.testing.assert_close(value, resumed.best_snapshot["state_dict"][key], atol=1e-7, rtol=1e-6)

    # Each architecture gets a fresh optimizer; no moments cross the stage boundary.
    trainer = StageTrainer(task, cfg)
    first_opt = trainer._optimizer(TabularMLP(1, [3], 1))
    second_opt = trainer._optimizer(TabularMLP(1, [4], 1))
    assert first_opt.state_dict()["state"] == second_opt.state_dict()["state"] == {}


def test_controller_resets_adam_and_cosine_scheduler_at_architecture_boundary():
    cfg = settings(
        init_width=2, init_depth=1, max_width=3, max_depth=2,
        max_epochs=1, patience_es=2, patience_width_exp=1,
        batch_size=4, scheduler_t0=3,
    )
    controller = SearchController(ScalarTask(), cfg)
    completed_stages = []

    def checkpoint(payload):
        if payload.get("stage_status") == "completed":
            completed_stages.append(copy.deepcopy(payload))

    controller.checkpoint_callback = checkpoint
    x = np.arange(8, dtype=np.float32).reshape(-1, 1) / 8
    y = x * 0.5
    controller.run(
        MODEL_REGISTRY["ADP-W-All"], x[:4], y[:4], x[4:], y[4:],
        input_dim=1, output_dim=1,
    )

    assert [stage["hidden_widths"] for stage in completed_stages] == [[2], [3]]
    optimizer_steps = []
    scheduler_states = []
    for stage in completed_stages:
        steps = {
            int(state["step"].item())
            for state in stage["optimizer_state"]["state"].values()
        }
        optimizer_steps.append(steps)
        scheduler_states.append(stage["scheduler_state"])
    # Each architecture sees one Adam update in its own optimizer. Carrying
    # moments across the boundary would make the second candidate's step 2.
    assert optimizer_steps == [{1}, {1}]
    assert scheduler_states[0]["T_cur"] == scheduler_states[1]["T_cur"]
    assert scheduler_states[0]["last_epoch"] == scheduler_states[1]["last_epoch"]


@pytest.mark.parametrize(
    ("optimizer_kind", "expected_type", "expected_weight_decay"),
    [("adam", torch.optim.Adam, 0.0), ("adamw", torch.optim.AdamW, 0.23)],
)
def test_stage_optimizer_profiles_keep_documented_parameters_and_scheduler(
    optimizer_kind, expected_type, expected_weight_decay,
):
    cfg = settings(
        init_width=3, init_depth=1, max_epochs=1, patience_es=1,
        batch_size=4, learning_rate=0.004, weight_decay=0.23,
        scheduler_t0=7, scheduler_t_mult=3, scheduler_eta_min=0.0002,
    )
    trainer = StageTrainer(ScalarTask(), cfg, optimizer_kind=optimizer_kind)
    model = TabularMLP(2, [3], 1)
    captured = []
    trainer.checkpoint_callback = lambda state: captured.append(state)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
    y = torch.tensor([[0.0], [1.0], [2.0], [-1.0]])

    result = trainer.train_stage(model, x, y, x, y)

    assert result.epochs == 1
    assert captured
    optimizer = trainer._optimizer(TabularMLP(2, [3], 1))
    assert isinstance(optimizer, expected_type)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.004)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(expected_weight_decay)
    if optimizer_kind == "adam":
        scheduler_state = captured[-1]["scheduler_state"]
        assert scheduler_state["T_0"] == 7
        assert scheduler_state["T_mult"] == 3
        assert scheduler_state["eta_min"] == pytest.approx(0.0002)
    else:
        assert captured[-1]["scheduler_state"] is None


def test_gradient_clipping_is_generic_adamw_only(monkeypatch):
    cfg = settings(
        max_epochs=1, patience_es=2, batch_size=4, grad_clip=0.01,
    )
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = torch.tensor([[100.0], [-100.0], [200.0], [-200.0]])
    original_clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def record_clip(*args, **kwargs):
        calls.append(kwargs.get("max_norm", args[1] if len(args) > 1 else None))
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    dedicated = StageTrainer(ScalarTask(), cfg, optimizer_kind="adam")
    dedicated.train_stage(TabularMLP(1, [2], 1), x, y, x, y)
    assert calls == []

    generic = StageTrainer(
        ScalarTask(), cfg, optimizer_kind="adamw", use_penalties=False,
    )
    generic.train_stage(TabularMLP(1, [2], 1), x, y, x, y)
    assert calls == [0.01]


def test_epoch_early_stop_uses_strict_one_e_minus_twelve_margin():
    task = ScalarTask()
    scores = iter([1.0, 1.0 - 0.5e-12, 1.0 - 0.7e-12])
    task.validation_score = lambda _outputs, _targets: next(scores)
    cfg = settings(max_epochs=5, patience_es=2, batch_size=4)
    model = TabularMLP(1, [2], 1)
    x = torch.ones((4, 1))
    y = torch.ones((4, 1))
    result = StageTrainer(task, cfg).train_stage(model, x, y, x, y)
    assert result.epochs == 3
    assert result.best_score == pytest.approx(1.0)


def test_zero_epoch_limit_means_train_until_epoch_patience_stops(monkeypatch):
    cfg = settings(
        init_width=2, init_depth=1, max_epochs=0, patience_es=2,
        batch_size=4,
    )
    trainer = StageTrainer(ScalarTask(), cfg, optimizer_kind="adamw")
    monkeypatch.setattr(StageTrainer, "_score", staticmethod(lambda *_args: 1.0))
    x = torch.ones((4, 1), dtype=torch.float32)
    y = torch.zeros((4, 1), dtype=torch.float32)

    result = trainer.train_stage(TabularMLP(1, [2], 1), x, y, x, y)

    # Epoch one establishes the best score; two equal scores exhaust patience.
    assert result.epochs == 3
    assert result.batches == 3
    assert result.best_score == pytest.approx(1.0)


def test_malformed_stage_rng_is_classified_as_exact_resume_failure():
    task = ScalarTask()
    cfg = settings(max_epochs=1)
    model = TabularMLP(1, [3], 1)
    trainer = StageTrainer(task, cfg)
    optimizer = trainer._optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.scheduler_t0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed)
    resume_state = {
        "hidden_widths": [3],
        "model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_model_state": copy.deepcopy(model.state_dict()),
        "best_score": 1.0,
        "patience": 1,
        "epoch": 0,
        "batch_cursor": 0,
        "generator_state": generator.get_state(),
        "rng_state": {"python": object()},
    }
    x = torch.ones((4, 1))
    y = torch.ones((4, 1))
    with pytest.raises(ValueError, match="exact-resume failure: stage checkpoint restore failed"):
        trainer.train_stage(model, x, y, x, y, resume_state=resume_state)


def test_dedicated_controller_rejects_missing_global_best_as_exact_resume_failure():
    config = MODEL_REGISTRY["ADP-W-All"]
    payload = {
        "search_context": {
            "controller": config.id,
            "phase": "width",
            "current_architecture": [3],
            "history": [],
            "counters": {"width_failures": 0},
        },
        "stage_state": {"hidden_widths": [3]},
    }
    x = np.ones((4, 1), dtype=np.float32)
    y = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="exact-resume failure: checkpoint controller is missing its global-best snapshot"):
        SearchController(ScalarTask(), settings()).run(
            config, x, y, x, y, input_dim=1, output_dim=1,
            resume_payload=payload,
        )
    def encode_targets(self, targets, *, device):
        return torch.as_tensor(targets, dtype=torch.float32, device=device)

