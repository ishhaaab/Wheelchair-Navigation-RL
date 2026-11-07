"""Exact tiny fixtures, never study labels or held-out layouts."""

import json
import math

import numpy as np
import pytest
import torch

from wnp.demos import Trajectory, enumerate_trajectories
from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.irl import NumericalDomainError, fit, loss, physical_weights
from wnp.multi_irl import EnvironmentDemos, fit_multi, loss_multi
from wnp.soft_dp import trajectory_log_prob


@pytest.fixture
def batches():
    first = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (0, 1), 4)
    occupied = np.zeros((2, 3), dtype=bool)
    occupied[1, 1] = True
    second = GridMDP(occupied, (0, 0, 1), (0, 2), 4)
    return [EnvironmentDemos(first, feature_table(first),
                            enumerate_trajectories(first, first.start_state, 3), 4.),
            EnvironmentDemos(second, feature_table(second),
                             enumerate_trajectories(second, second.start_state, 4)[:7], 7.)]


@pytest.mark.parametrize("parameterization", ["fixed_distance", "three_free"])
def test_loss_is_pooled_nll_with_one_regularizer_and_exact_gradients(batches, parameterization):
    size = 2 if parameterization == "fixed_distance" else 3
    raw = torch.linspace(-.3, .4, size, dtype=torch.float64, requires_grad=True)
    weights = physical_weights(raw, parameterization=parameterization)
    counts = [len(b.trajectories) for b in batches]
    explicit = sum(-trajectory_log_prob(b.env, b.features, weights, b.trajectories,
                                       b.terminal_penalty).sum() for b in batches) / sum(counts)
    expected = explicit + .1 * weights[3-size:].square().sum()
    actual = loss_multi(batches, raw, l2=.2, parameterization=parameterization)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual, raw)[0],
                               torch.autograd.grad(expected, raw)[0], atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(lambda r: loss_multi(
        batches, r, l2=.2, parameterization=parameterization), (raw,))


def test_duplicating_all_data_preserves_mean_but_unequal_counts_change_weighting(batches):
    raw = torch.zeros(3, dtype=torch.float64)
    original = loss_multi(batches, raw, .1, parameterization="three_free")
    doubled = [EnvironmentDemos(b.env, b.features, b.trajectories * 2, b.terminal_penalty)
               for b in batches]
    torch.testing.assert_close(original, loss_multi(doubled, raw, .1, parameterization="three_free"))
    naive = sum(loss(b.env, b.features, b.trajectories, raw, .1, b.terminal_penalty,
                     parameterization="three_free") for b in batches) / 2
    assert abs((naive - original).item()) > 1e-3


@pytest.mark.parametrize("parameterization", ["fixed_distance", "three_free"])
def test_single_batch_matches_existing_loss_and_fit(batches, parameterization):
    b = batches[0]
    expected = fit(b.env, b.features, b.trajectories, l2=.01, max_iter=3,
                   terminal_penalty=b.terminal_penalty, parameterization=parameterization)
    actual = fit_multi([b], l2=.01, max_iter=3, parameterization=parameterization)
    for key in ["weights", "objective", "raw_gradient", "iterations", "closure_evaluations",
                "success", "failure_reason", "physical_free_gradient"]:
        assert actual[key] == expected[key]


def test_fit_optimizes_shared_objective_repeatably_and_reports_counts(batches):
    result = fit_multi(batches, l2=.1, max_iter=20, parameterization="three_free")
    assert result == fit_multi(batches, l2=.1, max_iter=20, parameterization="three_free")
    initial = torch.full((3,), math.log(math.expm1(1)), dtype=torch.float64)
    assert result["objective"] < loss_multi(batches, initial, .1, parameterization="three_free")
    raw = torch.tensor(result["raw_parameters"], dtype=torch.float64, requires_grad=True)
    objective = loss_multi(batches, raw, .1, parameterization="three_free")
    assert result["objective"] == pytest.approx(objective.item(), abs=1e-12)
    torch.testing.assert_close(torch.tensor(result["raw_gradient"], dtype=torch.float64),
                               torch.autograd.grad(objective, raw)[0], atol=1e-12, rtol=1e-12)
    assert result["trajectory_count"] == sum(len(b.trajectories) for b in batches)
    assert result["horizons"] == [3, 4]
    assert result["terminal_penalties"] == [4., 7.]
    assert result["total_nll"] == pytest.approx(sum(-trajectory_log_prob(
        b.env, b.features, torch.tensor(result["weights"], dtype=torch.float64),
        b.trajectories, b.terminal_penalty).sum().item() for b in batches))
    json.dumps(result, allow_nan=False)


def test_goal_only_zero_horizon_is_successful():
    env = GridMDP(np.zeros((1, 1), dtype=bool), (0, 0, 0), (0, 0), 4)
    b = EnvironmentDemos(env, feature_table(env), [Trajectory((env.start_state,), (), 0)], 0.)
    result = fit_multi([b], parameterization="three_free")
    assert result["success"] and result["objective"] == 0
    assert result["raw_gradient"] == [0., 0., 0.]


def test_feature_accumulation_domain_failure_is_fatal_during_preparation(batches):
    b = batches[0]
    huge = b.features.clone()
    huge[b.env.start_state, 0, 0] = torch.finfo(torch.float64).max / 8
    with pytest.raises(NumericalDomainError, match="cumulative") as caught:
        fit_multi([EnvironmentDemos(b.env, huge, b.trajectories, 4.)])
    attempt = caught.value.fits[0]
    assert attempt["numerical_domain_failure"]
    assert attempt["phase"] == "preparation"
    assert attempt["environment_index"] == 0
    json.dumps(caught.value.fits, allow_nan=False)


def test_numerical_domain_is_fatal_and_retains_attempt_diagnostics(batches):
    b = batches[0]
    unsupported = EnvironmentDemos(b.env, b.features, b.trajectories, 1001.)
    with pytest.raises(NumericalDomainError, match="numerical domain") as caught:
        fit_multi([unsupported])
    assert caught.value.fits[0]["numerical_domain_failure"]
    assert caught.value.fits[0]["phase"] == "initial"
    json.dumps(caught.value.fits, allow_nan=False)
    # A deliberately near-limit unit fixture tests a fatal line-search trial.
    env = GridMDP(np.zeros((1, 3), dtype=bool), (0, 0, 0), (0, 2), 4)
    phi = feature_table(env)
    records = [Trajectory((env.start_state, int(env.next_state[env.start_state, 0])), (0,), None)]
    initial = torch.tensor([0., 0.], dtype=torch.float64)
    with pytest.raises(NumericalDomainError) as trial:
        fit_multi([EnvironmentDemos(env, phi, records, 997.7)], initial_raw=initial)
    assert trial.value.fits[0]["phase"] == "optimization"
    # See a separate real-domain objective rejection even when loss is called directly.
    with pytest.raises(ValueError, match="numerical domain"):
        loss_multi([EnvironmentDemos(env, phi, records, 1000.)], initial)


@pytest.mark.parametrize("kwargs", [{"l2": -1}, {"l2": True}, {"max_iter": 0},
                                    {"max_iter": True}, {"max_iter": 151},
                                    {"parameterization": "unknown"}])
def test_fit_rejects_invalid_settings(batches, kwargs):
    with pytest.raises(ValueError):
        fit_multi(batches, **kwargs)


def test_empty_malformed_and_learnable_inputs_rejected(batches):
    raw = torch.zeros(2, dtype=torch.float64)
    for bad in ([], ["wrong"], [EnvironmentDemos(batches[0].env, batches[0].features, [], 4.)]):
        with pytest.raises(ValueError):
            loss_multi(bad, raw)
        with pytest.raises(ValueError):
            fit_multi(bad)
    b = batches[0]
    with pytest.raises(ValueError, match="fixed features"):
        fit_multi([EnvironmentDemos(b.env, b.features.clone().requires_grad_(), b.trajectories, 4.)])
    with pytest.raises(ValueError, match="horizon"):
        loss_multi([EnvironmentDemos(b.env, b.features, [b.trajectories[0],
            Trajectory((b.env.start_state,), (), None)], 4.)], raw)
