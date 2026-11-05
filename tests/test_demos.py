"""Fixed-length action-labelled trajectories, validation, and unfiltered sampling."""

import numpy as np
import pytest
import torch

from wnp.demos import Trajectory, enumerate_trajectories, sample
from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.soft_dp import plan, trajectory_log_prob


def fixture():
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (0, 1), headings=4)
    phi = feature_table(env)
    weights = torch.tensor([1., .7, 1.2], dtype=torch.float64)
    return env, phi, weights


def test_enumeration_retains_distinct_sequences_with_same_endpoint():
    occupied = np.array([[False, True, False]])
    env = GridMDP(occupied, (0, 0, 0), (0, 2), headings=4)
    records = enumerate_trajectories(env, env.start_state, 2)
    assert {record.actions for record in records} == {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert len(records) == 4
    assert len({record.states[-1] for record in records}) == 2
    assert all(record.arrival_step is None for record in records)


def test_sampler_reproducible_legal_and_unfiltered_with_arrival_steps():
    env, phi, weights = fixture()
    result = plan(env, phi, weights, 4, terminal_penalty=0)
    starts = [env.start_state] * 2048
    records = sample(env, result, starts, seed=11)
    assert records == sample(env, result, starts, seed=11)
    assert records != sample(env, result, starts, seed=29)
    assert len(records) == len(starts)
    assert any(record.arrival_step is None for record in records)
    assert any(record.arrival_step is not None for record in records)
    for record in records:
        assert len(record.actions) == 4 and len(record.states) == 5
        arrival = next((t for t, s in enumerate(record.states) if env.is_goal[s]), None)
        assert record.arrival_step == arrival
        for t, action in enumerate(record.actions):
            assert env.legal[record.states[t], action]
            assert int(env.next_state[record.states[t], action]) == record.states[t + 1]
        if arrival is not None:
            assert record.actions[arrival:] == (0,) * (4 - arrival)


def test_sampling_frequencies_match_exact_trajectory_probabilities():
    env, phi, weights = fixture()
    result = plan(env, phi, weights, 2, terminal_penalty=0)
    legal = enumerate_trajectories(env, env.start_state, 2)
    exact = trajectory_log_prob(env, phi, weights, legal, terminal_penalty=0).exp()
    draws = sample(env, result, [env.start_state] * 12000, seed=31)
    counts = {record.actions: 0 for record in legal}
    for draw in draws:
        counts[draw.actions] += 1
    empirical = torch.tensor([counts[r.actions] / len(draws) for r in legal],
                             dtype=torch.float64)
    assert (empirical - exact).abs().max() < .02


@pytest.mark.parametrize("horizon", [0, 3])
def test_goal_start_arrival_zero_and_unique_padding(horizon):
    env, phi, weights = fixture()
    goal = env.state_index[(0, 1, 0)]
    result = plan(env, phi, weights, horizon)
    records = sample(env, result, [goal], seed=5)
    assert records == [Trajectory((goal,) * (horizon + 1), (0,) * horizon, 0)]
    assert trajectory_log_prob(env, phi, weights, records).tolist() == [0]


def test_zero_horizon_non_goal_sample_is_retained():
    env, phi, weights = fixture()
    result = plan(env, phi, weights, 0)
    assert sample(env, result, [env.start_state], seed=0) == [
        Trajectory((env.start_state,), (), None)]


@pytest.mark.parametrize("bad", ["length", "mixed_horizon", "illegal", "transition",
                                "arrival", "state", "action", "noninteger", "empty"])
def test_trajectory_likelihood_rejects_malformed_records(bad):
    env, phi, weights = fixture()
    s = env.start_state
    g = int(env.next_state[s, 0])
    valid = Trajectory((s, g), (0,), 1)
    if bad == "length":
        records = [Trajectory((s,), (0,), None)]
    elif bad == "mixed_horizon":
        records = [valid, Trajectory((s,), (), None)]
    elif bad == "illegal":
        records = [Trajectory((g, g), (1,), 0)]
    elif bad == "transition":
        records = [Trajectory((s, s), (0,), None)]
    elif bad == "arrival":
        records = [Trajectory((s, g), (0,), None)]
    elif bad == "state":
        records = [Trajectory((-1, s), (0,), None)]
    elif bad == "action":
        records = [Trajectory((s, g), (3,), 1)]
    elif bad == "noninteger":
        records = [Trajectory((s, g), (False,), 1)]
    else:
        records = []
    with pytest.raises(ValueError):
        trajectory_log_prob(env, phi, weights, records)


@pytest.mark.parametrize("seed", [None, True, -1, 1.5])
def test_sampler_rejects_nonreproducible_seeds(seed):
    env, phi, weights = fixture()
    with pytest.raises(ValueError, match="seed"):
        sample(env, plan(env, phi, weights, 2), [env.start_state], seed)


def test_solution_cannot_be_used_with_another_environment():
    env, phi, weights = fixture()
    other, _, _ = fixture()
    with pytest.raises(ValueError, match="environment"):
        sample(other, plan(env, phi, weights, 2), [other.start_state], seed=1)


@pytest.mark.parametrize("start, horizon", [(-1, 2), (True, 2), (0, -1), (0, 1.5)])
def test_enumerator_validates_start_and_horizon(start, horizon):
    env, _, _ = fixture()
    with pytest.raises(ValueError):
        enumerate_trajectories(env, start, horizon)
