"""Hand-derived costs and a separate brute-force Euclidean distance oracle."""

import math

import numpy as np
import pytest
import torch

from wnp.env import GridMDP
from wnp.features import clearance_distances, feature_table


@pytest.mark.parametrize("headings, turn", [(4, math.pi / 2), (8, math.pi / 4)])
def test_cardinal_and_turn_features(headings, turn):
    env = GridMDP(np.zeros((3, 3), dtype=bool), (1, 1, 0), (2, 2), headings)
    phi = feature_table(env)
    assert phi.dtype == torch.float64
    assert phi.shape == (len(env.states), 3, 3)
    torch.testing.assert_close(phi[env.start_state], torch.tensor(
        [[1, 0, 0], [0, turn, 0], [0, turn, 0]], dtype=torch.float64))
    assert torch.isfinite(phi).all()
    assert (phi >= 0).all()


def test_diagonal_forward_distance():
    env = GridMDP(np.zeros((3, 3), dtype=bool), (0, 0, 1), (2, 2))
    assert feature_table(env)[env.start_state, 0].tolist() == [math.sqrt(2), 0, 0]


def test_clearance_charged_at_destination_for_forward_and_turns():
    occupied = np.zeros((3, 4), dtype=bool)
    occupied[0, 2] = True
    env = GridMDP(occupied, (1, 1, 0), (2, 3), headings=4)
    phi = feature_table(env)
    # Forward arrives at (1,2), distance 1. Turns stay at (1,1), sqrt(2).
    assert phi[env.start_state, 0].tolist() == [1, 0, 1]
    expected_turn_clearance = (2 - math.sqrt(2)) ** 2
    assert phi[env.start_state, 1, 2].item() == pytest.approx(expected_turn_clearance)
    assert phi[env.start_state, 2, 2].item() == pytest.approx(expected_turn_clearance)
    smaller = feature_table(env, d_safe=0.5)
    assert torch.count_nonzero(smaller[:, :, 2]) == 0


def test_goal_entry_pays_cost_but_all_subsequent_features_are_zero():
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[0, 2] = True
    env = GridMDP(occupied, (1, 1, 0), (1, 2))
    phi = feature_table(env)
    assert phi[env.start_state, 0].tolist() == [1, 0, 1]
    assert torch.count_nonzero(phi[env.is_goal]) == 0
    assert torch.count_nonzero(phi[~env.legal]) == 0


@pytest.mark.parametrize("shape, seed", [((3, 4), 5), ((7, 5), 19), ((1, 3), 29)])
def test_clearance_distances_match_independent_brute_force(shape, seed):
    occupied = np.random.default_rng(seed).random(shape) < 0.4
    occupied[0, 0] = True
    actual = clearance_distances(occupied)
    assert actual.dtype == np.float64
    obstacles = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                 if occupied[r, c]]
    for r in range(shape[0]):
        for c in range(shape[1]):
            expected = min(math.hypot(r - rr, c - cc) for rr, cc in obstacles)
            assert actual[r, c] == pytest.approx(expected, abs=1e-14)


def test_no_obstacles_means_infinite_distance_and_zero_clearance():
    occupied = np.zeros((2, 3), dtype=bool)
    assert np.isposinf(clearance_distances(occupied)).all()
    env = GridMDP(occupied, (0, 0, 0), (1, 2))
    phi = feature_table(env)
    assert torch.count_nonzero(phi[:, :, 2]) == 0
    assert torch.isfinite(phi).all()


def test_enclosed_cell_turn_clearance_is_finite():
    occupied = np.ones((3, 3), dtype=bool)
    occupied[0, 0] = occupied[1, 1] = False
    env = GridMDP(occupied, (1, 1, 0), (0, 0))
    phi = feature_table(env)
    assert phi[env.start_state, 1].tolist() == [0, math.pi / 4, 1]
    assert phi[env.start_state, 0].tolist() == [0, 0, 0]
    assert torch.isfinite(phi).all()


@pytest.mark.parametrize("d_safe", [-1, float("nan"), float("inf")])
def test_invalid_safe_distance_raises(d_safe):
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1))
    with pytest.raises(ValueError):
        feature_table(env, d_safe)


def test_zero_safe_distance_has_no_clearance_penalty():
    occupied = np.array([[False, True], [False, False]])
    env = GridMDP(occupied, (0, 0, 2), (1, 1))
    assert torch.count_nonzero(feature_table(env, d_safe=0)[:, :, 2]) == 0


@pytest.mark.parametrize("occupied", [np.zeros((0, 2), dtype=bool),
                                     np.zeros((2, 2, 2), dtype=bool),
                                     np.zeros((2, 2))])
def test_clearance_rejects_invalid_occupancy(occupied):
    with pytest.raises(ValueError):
        clearance_distances(occupied)


def test_unrepresentable_clearance_cost_raises_without_overflow_warning():
    occupied = np.array([[False, True], [False, False]])
    env = GridMDP(occupied, (0, 0, 2), (1, 1))
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="float64"):
            feature_table(env, d_safe=1e155)


def test_large_representable_clearance_cost_is_not_clipped():
    occupied = np.array([[False, True], [False, False]])
    env = GridMDP(occupied, (0, 0, 2), (1, 1))
    phi = feature_table(env, d_safe=1e150)
    assert torch.isfinite(phi).all()
    assert phi[env.start_state, 0, 2].item() == pytest.approx(1e300)
