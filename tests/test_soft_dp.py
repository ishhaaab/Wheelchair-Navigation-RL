"""Exact distribution and all-state autograd requirements for the soft planner."""

import numpy as np
import pytest
import torch

from wnp.demos import Trajectory, enumerate_trajectories
from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.soft_dp import expected_features, plan, trajectory_log_prob
from wnp.verification import (
    exhaustive_fixture,
    terminal_margin_diagnostic,
    verify_distribution,
    write_verification,
)


def tiny():
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (0, 1), headings=4)
    return env, feature_table(env), torch.tensor([1., .7, 1.2], dtype=torch.float64)


@pytest.mark.parametrize("name", ["four_heading", "eight_heading"])
def test_exhaustive_distribution_and_expected_features(name):
    report = verify_distribution(name)
    assert report["trajectory_count"] == report["independent_sequence_count"]
    assert report["completed_count"] > 0
    assert report["unfinished_count"] > 0
    assert report["max_probability_error"] < 1e-10
    assert report["max_likelihood_probability_error"] < 1e-10
    assert report["reference_normalization_error"] < 1e-10
    assert report["policy_normalization_error"] < 1e-10
    assert report["max_expected_features_error"] < 1e-10
    assert report["terminal_margin"]["passed"]
    assert report["terminal_margin"]["unfinishable_reachable_pairs"] > 0
    if name == "eight_heading":
        assert report["legal_diagonal_used"]
        assert report["blocked_move_checks"] == 3


def test_verification_artifact_is_generated_from_shared_checks(tmp_path):
    import json

    path = tmp_path / "verification.json"
    report = write_verification(path)
    saved = json.loads(path.read_text())
    assert saved == report
    assert set(saved["fixtures"]) == {"four_heading", "eight_heading"}
    assert saved["task4_checks"] == "NOT RUN"


def test_time_dependent_policy_is_normalized_and_invalid_actions_have_zero_mass():
    env, phi, weights = tiny()
    result = plan(env, phi, weights, 4)
    assert result.values.shape == (5, len(env.states))
    assert result.log_policy.shape == (4, len(env.states), 3)
    assert result.values.dtype == result.log_policy.dtype == torch.float64
    assert torch.isfinite(result.values).all()
    assert torch.isneginf(result.log_policy[:, ~env.legal]).all()
    assert torch.count_nonzero(result.log_policy.exp()[:, ~env.legal]) == 0
    torch.testing.assert_close(result.log_policy.exp().sum(-1),
                               torch.ones((4, len(env.states)), dtype=torch.float64))
    assert not torch.allclose(result.log_policy[0, env.start_state],
                              result.log_policy[-1, env.start_state])


@pytest.mark.parametrize("horizon", [0, 1, 4])
def test_goal_has_no_entropy_and_retains_gradient_connection(horizon):
    env = GridMDP(np.zeros((1, 1), dtype=bool), (0, 0, 0), (0, 0), headings=4)
    weights = torch.ones(3, dtype=torch.float64, requires_grad=True)
    result = plan(env, feature_table(env), weights, horizon)
    assert result.values.requires_grad
    assert result.log_policy.requires_grad
    assert torch.count_nonzero(result.values) == 0
    assert torch.count_nonzero(result.log_policy[:, :, 0]) == 0
    gradient = torch.autograd.grad(result.values.sum(), weights)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(weights))


def test_zero_horizon_preserves_terminal_value_and_empty_trajectory_likelihood():
    env, phi, weights = tiny()
    result = plan(env, phi, weights, 0, terminal_penalty=7)
    assert result.values.shape == (1, len(env.states))
    assert result.log_policy.shape == (0, len(env.states), 3)
    assert result.values[0, env.start_state] == -7
    assert torch.count_nonzero(result.values[:, env.is_goal]) == 0
    trajectories = enumerate_trajectories(env, env.start_state, 0)
    assert len(trajectories) == 1
    assert trajectories[0].arrival_step is None
    logp = trajectory_log_prob(env, phi, weights, trajectories, terminal_penalty=7)
    assert logp.tolist() == [0]
    assert expected_features(env, phi, result, [env.start_state]).tolist() == [0, 0, 0]


def test_terminal_penalty_is_created_in_float64_without_float32_rounding():
    env, phi, weights = tiny()
    penalty = 0.123456789012345
    result = plan(env, phi, weights, 0, terminal_penalty=penalty)
    assert result.values[0, env.start_state].item() == -penalty


def test_all_state_values_and_trajectory_gradients_are_finite_including_unreachable():
    occupied = np.zeros((2, 4), dtype=bool)
    occupied[:, 2] = True  # Free right column is disconnected from the start.
    env = GridMDP(occupied, (0, 0, 0), (0, 1), headings=4)
    phi = feature_table(env)
    assert torch.equal(phi[~env.legal], torch.zeros_like(phi[~env.legal]))
    weights = torch.tensor([1., .7, 1.2], dtype=torch.float64, requires_grad=True)
    result = plan(env, phi, weights, 4)
    assert result.values.requires_grad and result.log_policy.requires_grad
    assert result.values.shape[1] == len(env.states)
    assert env.state_index[(0, 3, 0)] not in {env.start_state}
    # Backpropagate every V_0 separately, including unreachable and goal states.
    for s in range(len(env.states)):
        gradient = torch.autograd.grad(result.values[0, s], weights, retain_graph=True)[0]
        assert torch.isfinite(gradient).all()
    nong_goal = enumerate_trajectories(env, env.start_state, 4)
    goal = enumerate_trajectories(env, env.state_index[(0, 1, 0)], 4)
    unreachable = enumerate_trajectories(env, env.state_index[(0, 3, 0)], 4)
    # No record in this batch starts in the right component, which must still
    # be processed by the full DP without contaminating the batch gradients.
    records = [nong_goal[0], nong_goal[-1], goal[0]]
    logp = trajectory_log_prob(env, phi, weights, records)
    assert logp.requires_grad and torch.isfinite(logp).all()
    for entry in logp:
        gradient = torch.autograd.grad(entry, weights, retain_graph=True)[0]
        assert torch.isfinite(gradient).all()
    disconnected_logp = trajectory_log_prob(env, phi, weights, [unreachable[0]])
    assert torch.isfinite(torch.autograd.grad(disconnected_logp.sum(), weights)[0]).all()
    expected = expected_features(env, phi, result, [env.start_state])
    assert torch.isfinite(torch.autograd.grad(expected.sum(), weights)[0]).all()


def test_expected_features_averages_start_distribution_including_duplicate_starts():
    env, phi, weights = tiny()
    result = plan(env, phi, weights, 4)
    start = env.start_state
    goal = env.state_index[(0, 1, 0)]
    single = expected_features(env, phi, result, [start])
    mixture = expected_features(env, phi, result, [start, goal, start])
    torch.testing.assert_close(mixture, single * (2 / 3), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("bad_horizon", [-1, 1.5, True])
def test_invalid_horizon_raises(bad_horizon):
    env, phi, weights = tiny()
    with pytest.raises(ValueError, match="horizon"):
        plan(env, phi, weights, bad_horizon)


@pytest.mark.parametrize("penalty", [-1, float("inf"), float("nan")])
def test_invalid_terminal_penalty_raises(penalty):
    env, phi, weights = tiny()
    with pytest.raises(ValueError, match="terminal_penalty"):
        plan(env, phi, weights, 4, penalty)


@pytest.mark.parametrize("corruption", ["float32", "shape", "nonfinite", "negative",
                                       "masked_nonzero", "masked_inf", "goal_nonzero"])
def test_invalid_feature_tables_are_rejected_not_silently_masked(corruption):
    env, phi, weights = tiny()
    if corruption == "float32":
        phi = phi.float()
    elif corruption == "shape":
        phi = phi[:, :, :2]
    elif corruption == "nonfinite":
        phi[env.start_state, 0, 0] = float("nan")
    elif corruption == "negative":
        phi[env.start_state, 0, 0] = -1
    elif corruption == "masked_nonzero":
        phi[~env.legal] = 1e-300
    elif corruption == "masked_inf":
        phi[~env.legal] = float("inf")
    else:
        phi[env.is_goal, 0, 0] = 1
    with pytest.raises(ValueError, match="features"):
        plan(env, phi, weights, 4)


@pytest.mark.parametrize("weights", [torch.ones(3), torch.ones(2, dtype=torch.float64),
    torch.tensor([1., float("inf"), 1.], dtype=torch.float64)])
def test_invalid_weights_raise(weights):
    env, phi, _ = tiny()
    with pytest.raises(ValueError, match="weights"):
        plan(env, phi, weights, 4)


def test_all_masked_state_is_rejected_before_logsumexp():
    env, phi, weights = tiny()
    env.legal[env.start_state] = False
    phi[env.start_state] = 0
    with pytest.raises(ValueError, match="legal"):
        plan(env, phi, weights, 4)


@pytest.mark.parametrize("starts", [[], [-1], [True], [10000], [0.5]])
def test_expected_features_rejects_invalid_starts(starts):
    env, phi, weights = tiny()
    result = plan(env, phi, weights, 2)
    with pytest.raises(ValueError, match="start"):
        expected_features(env, phi, result, starts)


def test_terminal_margin_uses_strict_inequality_at_reference_weights():
    env = GridMDP(np.zeros((1, 2), dtype=bool), (0, 0, 0), (0, 1), headings=4)
    phi = feature_table(env)
    weights = torch.ones(3, dtype=torch.float64)
    failed = terminal_margin_diagnostic(env, phi, weights, 1, terminal_penalty=2)
    passed = terminal_margin_diagnostic(env, phi, weights, 1, terminal_penalty=3)
    assert failed["max_optimal_finishing_cost"] == 1
    assert failed["margin"] == 1
    assert not failed["passed"] and passed["passed"]
    assert passed["unfinishable_reachable_pairs"] == 2


@pytest.mark.parametrize("entrypoint", ["plan", "likelihood", "expected_features"])
def test_cumulative_feature_overflow_is_rejected_before_invalid_results(entrypoint):
    env = GridMDP(np.array([[False, True, False]]), (0, 0, 0), (0, 2), headings=4)
    phi = feature_table(env, d_safe=1e154)
    weights = torch.tensor([1., .7, 1e-308], dtype=torch.float64, requires_grad=True)
    assert torch.isfinite(phi).all()
    assert torch.isfinite(phi @ weights).all()
    with pytest.raises(ValueError, match="cumulative.*float64"):
        if entrypoint == "plan":
            plan(env, phi, weights, 2)
        elif entrypoint == "likelihood":
            record = enumerate_trajectories(env, env.start_state, 2)[0]
            trajectory_log_prob(env, phi, weights, [record])
        else:
            ordinary = plan(env, feature_table(env), weights, 2)
            expected_features(env, phi, ordinary, [env.start_state])


def test_extreme_but_representable_single_step_has_finite_results_and_gradients():
    env = GridMDP(np.array([[False, True, False]]), (0, 0, 0), (0, 2), headings=4)
    phi = feature_table(env, d_safe=1e154)
    weights = torch.tensor([1., .7, 1e-308], dtype=torch.float64, requires_grad=True)
    result = plan(env, phi, weights, 1)
    assert torch.isfinite(result.values).all()
    value_gradient = torch.autograd.grad(result.values[0, env.start_state], weights)[0]
    assert torch.isfinite(value_gradient).all()
    records = enumerate_trajectories(env, env.start_state, 1)
    logp = trajectory_log_prob(env, phi, weights, records)
    torch.testing.assert_close(logp, torch.full((2,), -np.log(2), dtype=torch.float64),
                               atol=1e-13, rtol=0)
    assert torch.isfinite(torch.autograd.grad(logp.mean(), weights)[0]).all()
