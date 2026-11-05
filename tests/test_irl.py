"""Gradient identities, constrained objectives, fitting, and trajectory bootstrap."""

import math

import numpy as np
import pytest
import torch

from wnp.demos import Trajectory, sample
from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.irl import (
    bootstrap_fit, classify_fit, fit, loss, physical_weights, summarize_bootstrap,
)
from wnp.soft_dp import plan, trajectory_log_prob
from wnp import verification


@pytest.fixture(scope="module")
def unit_data():
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[1, 1] = True
    env = GridMDP(occupied, (0, 0, 0), (0, 2), headings=4)
    phi = feature_table(env)
    weights = torch.tensor([1., .7, 1.2], dtype=torch.float64)
    # Deliberately finite low penalty exercises unfinished records in unit tests.
    penalty = 4.0
    records = sample(env, plan(env, phi, weights, 5, penalty), [env.start_state] * 128, 17)
    return env, phi, records, penalty


def test_shared_physical_gradient_check_and_gradcheck():
    report = verification.verify_gradients()
    assert report["max_autograd_finite_difference_error"] < 1e-7
    assert report["max_autograd_analytical_error"] < 1e-7
    assert report["max_finite_difference_analytical_error"] < 1e-7
    assert report["gradcheck"] is True
    assert report["all_state_gradients_finite"]
    assert report["single_legal_action_states"] > 0
    assert report["unreachable_states"] > 0
    assert report["unfinished_demonstrations"] > 0


def test_shared_regularized_raw_gradient_check():
    report = verification.verify_regularized()
    assert report["max_autograd_finite_difference_error"] < 1e-7
    assert report["max_autograd_analytical_error"] < 1e-7
    assert report["gradcheck"] is True
    assert report["l2"] > 0


def test_shared_hessian_check_uses_same_single_action_fixture():
    report = verification.verify_hessian()
    assert report["max_hessian_expected_features_fd_error"] < 1e-7
    assert report["single_legal_action_states"] > 0
    assert report["unreachable_states"] > 0
    assert report["fixture"] == verification.verify_gradients()["fixture"]


def test_softplus_fixes_distance_and_preserves_raw_gradient():
    raw = torch.tensor([-.2, .3], dtype=torch.float64, requires_grad=True)
    weights = physical_weights(raw)
    assert weights.dtype == torch.float64 and weights[0].item() == 1
    torch.testing.assert_close(weights[1:], torch.log1p(raw.exp()))
    torch.testing.assert_close(torch.autograd.grad(weights.sum(), raw)[0], raw.sigmoid())
    large = physical_weights(torch.tensor([-100., 20.], dtype=torch.float64))
    assert 0 < large[1] < 1e-3 and large[2] > 10  # No clipping to fit-success bounds.


@pytest.mark.parametrize("raw", [torch.ones(2), torch.ones(3, dtype=torch.float64),
    torch.tensor([float("nan"), 0.], dtype=torch.float64),
    torch.tensor([-1000., 0.], dtype=torch.float64)])
def test_parameterization_rejects_invalid_or_unrepresentable_raw_parameters(raw):
    with pytest.raises(ValueError):
        physical_weights(raw)


def test_loss_is_mean_trajectory_nll_plus_l2_on_physical_free_weights(unit_data):
    env, phi, records, penalty = unit_data
    assert any(record.arrival_step is None for record in records)
    raw = torch.tensor([.1, -.4], dtype=torch.float64, requires_grad=True)
    weights = physical_weights(raw)
    nll = -trajectory_log_prob(env, phi, weights, records, penalty).mean()
    actual = loss(env, phi, records, raw, l2=.2, terminal_penalty=penalty)
    torch.testing.assert_close(actual, nll + .1 * weights[1:].square().sum(),
                               atol=1e-12, rtol=1e-12)
    repeated = loss(env, phi, records + records, raw, l2=.2, terminal_penalty=penalty)
    torch.testing.assert_close(actual, repeated, atol=1e-12, rtol=1e-12)
    wrong_model = loss(env, phi, records, raw, l2=.2, terminal_penalty=30)
    assert abs((actual - wrong_model).item()) > .1


@pytest.mark.parametrize("l2", [-1, float("nan"), float("inf"), True])
def test_loss_rejects_invalid_regularization(unit_data, l2):
    env, phi, records, penalty = unit_data
    with pytest.raises(ValueError, match="l2"):
        loss(env, phi, records, torch.zeros(2, dtype=torch.float64), l2, penalty)


@pytest.mark.parametrize("objective, gradient, weights, iterations, expected", [
    (1., [0., 1e-6], [1e-3, 10.], 149, None),
    (1., [0., 1.01e-6], [1., 1.], 1, "unconverged"),
    (1., [0., 0.], [1., 1.], 150, "unconverged"),
    (1., [0., 0.], [0.00099, 1.], 1, "boundary"),
    (1., [0., 0.], [1., 10.001], 1, "boundary"),
    (float("inf"), [0., 0.], [0., 20.], 150, "non-finite"),
    (1., [float("nan"), 0.], [1., 1.], 1, "non-finite"),
    (1., [0., 0.], [float("inf"), 1.], 1, "non-finite"),
    (1., [1., 1.], [0., 20.], 150, "boundary"),
])
def test_convergence_classification_exact_gates_and_failure_priority(
    objective, gradient, weights, iterations, expected,
):
    assert classify_fit(objective, gradient, weights, iterations) == expected


def test_fit_is_repeatable_and_reports_actual_regularized_gradients(unit_data):
    env, phi, records, penalty = unit_data
    first = fit(env, phi, records, l2=.01, terminal_penalty=penalty)
    second = fit(env, phi, records, l2=.01, terminal_penalty=penalty)
    assert first == second
    assert first["success"], first
    assert first["failure_reason"] is None
    assert first["raw_gradient_inf_norm"] <= 1e-6
    assert first["iterations"] < 150
    assert first["closure_evaluations"] > 0
    assert first["weights"][0] == 1
    assert all(1e-3 <= w <= 10 for w in first["weights"][1:])
    assert first["terminal_penalty"] == penalty
    assert first["unfinished_count"] == sum(r.arrival_step is None for r in records)
    raw = torch.tensor(first["raw_parameters"], dtype=torch.float64, requires_grad=True)
    objective = loss(env, phi, records, raw, .01, penalty)
    torch.testing.assert_close(objective, torch.tensor(first["objective"], dtype=torch.float64),
                               atol=1e-12, rtol=1e-12)
    gradient = torch.autograd.grad(objective, raw)[0]
    torch.testing.assert_close(gradient, torch.tensor(first["raw_gradient"], dtype=torch.float64),
                               atol=1e-12, rtol=1e-12)
    physical_gradient = torch.tensor(first["physical_free_gradient"], dtype=torch.float64)
    torch.testing.assert_close(gradient, physical_gradient * raw.sigmoid(), atol=1e-12, rtol=1e-12)


def test_finite_objective_is_not_mistaken_for_convergence(unit_data):
    env, phi, records, penalty = unit_data
    result = fit(env, phi, records, max_iter=1, terminal_penalty=penalty)
    assert math.isfinite(result["objective"])
    assert result["raw_gradient_inf_norm"] > 1e-6
    assert not result["success"] and result["failure_reason"] == "unconverged"


def test_boundary_fit_is_reported_without_clipping():
    env = GridMDP(np.zeros((1, 1), dtype=bool), (0, 0, 0), (0, 0), headings=4)
    record = Trajectory((env.start_state,), (), 0)
    result = fit(env, feature_table(env), [record], initial_raw=torch.tensor(
        [-20., 20.], dtype=torch.float64))
    assert result["objective"] == 0
    assert result["raw_gradient_inf_norm"] == 0
    assert not result["success"] and result["failure_reason"] == "boundary"
    assert result["weights"][1] < 1e-3 and result["weights"][2] > 10


def test_nonfinite_optimized_objective_has_priority_and_serializable_diagnostics(unit_data):
    import json

    env, phi, records, penalty = unit_data
    result = fit(env, phi, records, l2=1e308, terminal_penalty=penalty,
                 initial_raw=torch.tensor([20., 20.], dtype=torch.float64))
    assert not result["success"] and result["failure_reason"] == "non-finite"
    assert result["failure_detail"]
    assert result["objective"] is None
    assert result["weights"][1] > 10  # Preserve the boundary values, never clip.
    json.dumps(result, allow_nan=False)


def test_fit_rejects_nontensor_features_with_value_error(unit_data):
    env, phi, records, penalty = unit_data
    with pytest.raises(ValueError, match="features"):
        fit(env, phi.numpy(), records, terminal_penalty=penalty)


@pytest.mark.parametrize("kwargs", [{"max_iter": 0}, {"max_iter": 151},
                                    {"max_iter": True}, {"terminal_penalty": -1},
                                    {"l2": -1}])
def test_fit_rejects_invalid_settings(unit_data, kwargs):
    env, phi, records, _ = unit_data
    with pytest.raises(ValueError):
        fit(env, phi, records, **kwargs)


def test_loss_and_fit_reject_malformed_and_mixed_horizon_data(unit_data):
    env, phi, records, _ = unit_data
    for bad in ([], [records[0], Trajectory((env.start_state,), (), None)],
                [Trajectory((env.start_state,), (0,), None)]):
        with pytest.raises(ValueError):
            loss(env, phi, bad, torch.zeros(2, dtype=torch.float64))
        with pytest.raises(ValueError):
            fit(env, phi, bad)


def test_whole_trajectory_bootstrap_is_repeatable_and_preserves_model(unit_data):
    env, phi, records, penalty = unit_data
    first = bootstrap_fit(env, phi, records, 31, refits=2, l2=.01, terminal_penalty=penalty)
    assert first == bootstrap_fit(env, phi, records, 31, refits=2, l2=.01,
                                  terminal_penalty=penalty)
    assert first["refits"] == 2 and len(first["fits"]) == 2
    assert first["resampling_unit"] == "whole trajectory"
    assert first["terminal_penalty"] == penalty
    assert first["successful_refits"] + first["failed_refits"] == 2
    assert not first["reliable"]  # Fewer than 80 successes cannot support these intervals.
    for index, result in enumerate(first["fits"]):
        indices = np.random.default_rng(result["resample_seed"]).integers(
            len(records), size=len(records))
        selected = [records[int(i)] for i in indices]
        assert result["trajectory_count"] == len(records)
        assert result["horizon"] == 5
        assert result["terminal_penalty"] == penalty
        assert result["unfinished_count"] == sum(r.arrival_step is None for r in selected)
        repeat = fit(env, phi, selected, l2=.01, terminal_penalty=penalty)
        assert result["weights"] == repeat["weights"]
        assert result["bootstrap_index"] == index


def test_bootstrap_explicitly_accounts_for_all_failed_refits(unit_data):
    env, phi, records, penalty = unit_data
    result = bootstrap_fit(env, phi, records, 13, refits=3, max_iter=1,
                           terminal_penalty=penalty)
    assert result["failed_refits"] == 3
    assert result["failure_counts"] == {"unconverged": 3}
    assert result["successful_refits"] == 0
    assert result["percentile_intervals"] is None
    assert not result["reliable"]


def test_bootstrap_summary_uses_successes_only_and_exact_80_of_100_cutoff():
    successes = [{"success": True, "weights": [1., 1., 2.], "failure_reason": None}] * 80
    failures = [{"success": False, "weights": [1., 50., 50.],
                 "failure_reason": "boundary"}] * 20
    summary = summarize_bootstrap(successes + failures)
    assert summary["successful_refits"] == 80 and summary["failed_refits"] == 20
    assert summary["required_successes"] == 80 and summary["reliable"]
    assert summary["percentile_intervals"] == [[1., 1.], [2., 2.]]
    fewer = summarize_bootstrap(successes[:-1] + failures + [failures[0]])
    assert fewer["successful_refits"] == 79 and not fewer["reliable"]


@pytest.mark.parametrize("kwargs", [{"seed": None}, {"seed": True}, {"seed": -1},
                                    {"refits": 0}, {"refits": 1.5}, {"refits": True}])
def test_bootstrap_rejects_invalid_settings(unit_data, kwargs):
    env, phi, records, penalty = unit_data
    options = {"seed": 11, "refits": 2, "terminal_penalty": penalty, **kwargs}
    with pytest.raises(ValueError):
        bootstrap_fit(env, phi, records, **options)


def test_bootstrap_default_is_100_without_running_full_bootstrap():
    import inspect

    assert inspect.signature(bootstrap_fit).parameters["refits"].default == 100


@pytest.mark.parametrize("l2", [0.0, 1e-300])
def test_regularization_does_not_overflow_before_scaling_large_unidentifiable_weight(l2):
    from wnp.demos import enumerate_trajectories

    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (0, 1), headings=4)
    phi = feature_table(env)
    records = enumerate_trajectories(env, env.start_state, 4)
    raw = torch.tensor([0., 1e155], dtype=torch.float64, requires_grad=True)
    weights = physical_weights(raw)
    nll = -trajectory_log_prob(env, phi, weights, records).mean()
    actual = loss(env, phi, records, raw, l2=l2)
    assert torch.isfinite(actual)
    expected_penalty = 0.0 if l2 == 0 else 5e9
    assert actual.item() == pytest.approx(nll.item() + expected_penalty, rel=1e-14)
    assert torch.isfinite(torch.autograd.grad(actual, raw)[0]).all()


def test_large_unregularized_boundary_weight_is_not_mislabeled_nonfinite():
    from wnp.demos import enumerate_trajectories

    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (0, 1), headings=4)
    result = fit(env, feature_table(env), enumerate_trajectories(env, env.start_state, 4),
                 l2=0, initial_raw=torch.tensor([0., 1e155], dtype=torch.float64))
    assert math.isfinite(result["objective"])
    assert result["failure_reason"] == "boundary"
    assert result["weights"][2] == 1e155
