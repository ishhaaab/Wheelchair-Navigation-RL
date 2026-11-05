"""Pin movement rules independently of the transition-table implementation."""

import numpy as np
import pytest
import torch

from wnp.env import GridMDP, generate_layout


@pytest.mark.parametrize("headings", [4, 8])
def test_forward_and_heading_wraparound(headings):
    env = GridMDP(np.zeros((3, 3), dtype=bool), (1, 1, 0), (2, 2), headings)
    s = env.start_state
    assert env.states[env.next_state[s, 0]] == (1, 2, 0)
    assert env.states[env.next_state[s, 1]] == (1, 1, headings - 1)
    assert env.states[env.next_state[s, 2]] == (1, 1, 1)
    last = env.state_index[(1, 1, headings - 1)]
    assert env.states[env.next_state[last, 2]] == (1, 1, 0)


@pytest.mark.parametrize("heading, destination", [
    (0, (1, 2)), (1, (2, 2)), (2, (2, 1)), (3, (2, 0)),
    (4, (1, 0)), (5, (0, 0)), (6, (0, 1)), (7, (0, 2)),
])
def test_all_eight_heading_directions(heading, destination):
    env = GridMDP(np.zeros((3, 3), dtype=bool), (1, 1, heading), (0, 0))
    assert env.legal[env.start_state, 0]
    assert env.states[env.next_state[env.start_state, 0]] == (*destination, heading)


def test_boundaries_obstacles_and_safe_masked_indices():
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[1, 2] = True
    env = GridMDP(occupied, (1, 1, 0), (2, 2))
    assert not env.legal[env.start_state, 0]
    assert not env.legal[env.state_index[(0, 0, 6)], 0]
    assert not env.legal[env.state_index[(2, 0, 3)], 0]
    assert (1, 2, 0) not in env.state_index
    assert torch.all(env.next_state >= 0)
    assert torch.all(env.next_state < len(env.states))
    assert env.next_state.dtype == torch.long
    assert env.legal.dtype == torch.bool
    assert env.is_goal.dtype == torch.bool
    assert torch.all(env.legal.any(dim=1))


@pytest.mark.parametrize("obstacle", [(0, 1), (1, 0), (1, 1)])
def test_diagonal_blocks_either_corner_or_occupied_destination(obstacle):
    occupied = np.zeros((3, 3), dtype=bool)
    occupied[obstacle] = True
    env = GridMDP(occupied, (0, 0, 1), (2, 2))
    assert not env.legal[env.start_state, 0]


@pytest.mark.parametrize("headings", [4, 8])
def test_every_goal_heading_has_one_zero_entropy_continuation(headings):
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1), headings)
    assert int(env.is_goal.sum()) == headings
    for heading in range(headings):
        s = env.state_index[(1, 1, heading)]
        assert env.legal[s].tolist() == [True, False, False]
        assert env.next_state[s, 0] == s


def test_enclosed_cell_still_allows_turns():
    occupied = np.ones((3, 3), dtype=bool)
    occupied[0, 0] = occupied[1, 1] = False
    env = GridMDP(occupied, (1, 1, 0), (0, 0))
    assert env.legal[env.start_state].tolist() == [False, True, True]
    assert not env.goal_reachable()


def test_start_at_goal_is_connected():
    env = GridMDP(np.zeros((1, 1), dtype=bool), (0, 0, 0), (0, 0))
    assert env.goal_reachable()
    assert env.legal[env.start_state].tolist() == [True, False, False]


def test_occupancy_is_copied_from_caller():
    occupied = np.zeros((2, 2), dtype=bool)
    env = GridMDP(occupied, (0, 0, 0), (1, 1))
    occupied[0, 1] = True
    assert not env.occupied[0, 1]
    assert env.legal[env.start_state, 0]
    with pytest.raises(ValueError):
        env.occupied[0, 1] = True


def test_occupied_goal_is_rejected():
    occupied = np.zeros((2, 2), dtype=bool)
    occupied[1, 1] = True
    with pytest.raises(ValueError):
        GridMDP(occupied, (0, 0, 0), (1, 1))


@pytest.mark.parametrize("occupied, start, goal, headings", [
    (np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1), 6),
    (np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1), True),
    (np.zeros((0, 2), dtype=bool), (0, 0, 0), (0, 0), 8),
    (np.zeros((2, 2, 2), dtype=bool), (0, 0, 0), (1, 1), 8),
    (np.zeros((2, 2)), (0, 0, 0), (1, 1), 8),
    (np.ones((2, 2), dtype=bool), (0, 0, 0), (1, 1), 8),
    (np.zeros((2, 2), dtype=bool), (-1, 0, 0), (1, 1), 8),
    (np.zeros((2, 2), dtype=bool), (0, 0, 8), (1, 1), 8),
    (np.zeros((2, 2), dtype=bool), (0, 0, 0), (2, 1), 8),
    (np.zeros((2, 2), dtype=bool), (0.5, 0, 0), (1, 1), 8),
    (np.zeros((2, 2), dtype=bool), (0, 0), (1, 1), 8),
    (np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1, 0), 8),
])
def test_invalid_environment_inputs_raise(occupied, start, goal, headings):
    with pytest.raises(ValueError):
        GridMDP(occupied, start, goal, headings)


def _independent_route_exists(env):
    """Search the published legal graph, independently of the generator's check."""
    seen = {env.start_state}
    pending = [env.start_state]
    while pending:
        s = pending.pop()
        if env.states[s][:2] == env.goal:
            return True
        for a in range(3):
            if env.legal[s, a]:
                successor = int(env.next_state[s, a])
                if successor not in seen:
                    seen.add(successor)
                    pending.append(successor)
    return False


@pytest.mark.parametrize("headings", [4, 8])
@pytest.mark.parametrize("seed", [0, 11, 29])
def test_generator_returns_reproducible_connected_layouts(seed, headings):
    env = generate_layout(seed, headings=headings)
    repeat = generate_layout(seed, headings=headings)
    assert env.occupied.shape == (20, 20)
    np.testing.assert_array_equal(env.occupied, repeat.occupied)
    assert torch.equal(env.next_state, repeat.next_state)
    assert _independent_route_exists(env)
    assert env.goal_reachable()


def test_generator_honors_custom_endpoints_and_exhaustion():
    env = generate_layout(3, shape=(3, 4), obstacle_probability=0,
                          headings=4, start=(2, 0, 3), goal=(0, 3))
    assert env.states[env.start_state] == (2, 0, 3)
    assert env.goal == (0, 3)
    assert _independent_route_exists(env)
    with pytest.raises(RuntimeError, match="connect"):
        generate_layout(3, shape=(3, 3), obstacle_probability=1, max_attempts=2)


def test_generator_seeds_change_occupancy():
    assert not np.array_equal(generate_layout(11).occupied, generate_layout(29).occupied)


@pytest.mark.parametrize("kwargs", [
    {"shape": (0, 2)}, {"shape": (2,)}, {"shape": (2.5, 3)},
    {"obstacle_probability": -0.1}, {"obstacle_probability": 1.1},
    {"obstacle_probability": float("nan")}, {"max_attempts": 0},
    {"max_attempts": 1.5}, {"headings": 3}, {"start": (20, 0, 0)},
    {"goal": (-1, 0)},
])
def test_generator_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        generate_layout(1, **kwargs)


@pytest.mark.parametrize("seed", [None, True, False, np.bool_(True), 1.5, -1,
                                 np.random.default_rng(1), np.random.SeedSequence(1)])
def test_generator_requires_explicit_nonnegative_integer_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        generate_layout(seed, shape=(3, 3), obstacle_probability=0)


@pytest.mark.parametrize("seed", [0, 11, np.int32(11), np.int64(11), np.uint64(11)])
def test_generator_accepts_reproducible_python_and_numpy_integer_seeds(seed):
    env = generate_layout(seed, shape=(5, 5))
    repeat = generate_layout(int(seed), shape=(5, 5))
    np.testing.assert_array_equal(env.occupied, repeat.occupied)
    assert torch.equal(env.next_state, repeat.next_state)
