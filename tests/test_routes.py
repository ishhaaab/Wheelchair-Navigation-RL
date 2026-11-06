"""Preference-free candidate construction against exhaustive legal sequences."""

import math

import numpy as np
import pytest
import torch

from wnp.demos import enumerate_trajectories, validate_trajectories
from wnp.env import GridMDP
from wnp.routes import candidate_pair, shortest_route


def _key(env, route):
    arrival = route.arrival_step
    actions = route.actions[:arrival]
    distance = sum(math.dist(env.states[route.states[t]][:2],
                             env.states[route.states[t+1]][:2]) for t in range(arrival))
    return distance, arrival, actions


@pytest.mark.parametrize('headings,horizon', [(4, 7), (8, 8)])
def test_shortest_matches_exhaustive_distance_count_action_order(headings, horizon):
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1), headings)
    completed = [r for r in enumerate_trajectories(env, env.start_state, horizon)
                 if r.arrival_step is not None]
    route = shortest_route(env)
    assert _key(env, route) == min(_key(env, r) for r in completed)
    assert route.arrival_step == len(route.actions)
    assert route.actions == ((0, 2, 0) if headings == 4 else (2, 0))


def test_pair_uses_best_edge_block_alternative_and_does_not_mutate_graph():
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1), 4)
    before = (env.next_state.clone(), env.legal.clone(), env.occupied.copy())
    pair = candidate_pair(env, 8)
    assert pair.shortest_index == 0
    assert pair.routes[0].actions == (0, 2, 0, 0, 0, 0, 0, 0)
    assert pair.routes[1].actions == (2, 0, 1, 0, 0, 0, 0, 0)
    assert [r.arrival_step for r in pair.routes] == [3, 4]
    validate_trajectories(env, pair.routes)
    assert torch.equal(before[0], env.next_state) and torch.equal(before[1], env.legal)
    np.testing.assert_array_equal(before[2], env.occupied)
    assert candidate_pair(env, 8) == pair


def test_blocked_edges_are_undirected_and_turns_are_not_blocked():
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1), 4)
    blocked = frozenset({frozenset({(0, 1), (0, 0)})})
    assert shortest_route(env, blocked).actions == (2, 0, 1, 0)


def test_corner_blocking_applies_to_search():
    occupied = np.array([[False, True], [False, False]])
    env = GridMDP(occupied, (0, 0, 0), (1, 1), 8)
    route = shortest_route(env)
    assert route.actions == (2, 2, 0, 1, 1, 0)


def test_unreachable_start_goal_and_missing_alternative_terminate():
    env = GridMDP(np.array([[False, True, False]]), (0, 0, 0), (0, 2), 4)
    assert shortest_route(env) is None
    with pytest.raises(ValueError, match='route'):
        candidate_pair(env, 20)
    corridor = GridMDP(np.zeros((1, 3), dtype=bool), (0, 0, 0), (0, 2), 4)
    with pytest.raises(ValueError, match='alternative'):
        candidate_pair(corridor, 20)
    goal = GridMDP(np.zeros((1, 1), dtype=bool), (0, 0, 0), (0, 0), 4)
    assert shortest_route(goal).actions == ()
    with pytest.raises(ValueError, match='alternative'):
        candidate_pair(goal, 20)


def test_symmetric_alternatives_are_retained_when_features_identical():
    from wnp.features import feature_table
    from wnp.route_choice import route_features
    occupied=np.zeros((3,3),dtype=bool); occupied[1,1]=True
    env=GridMDP(occupied,(1,0,0),(1,2),4)
    pair=candidate_pair(env,12)
    totals=route_features(env,feature_table(env),pair)
    assert pair.routes[0].actions!=pair.routes[1].actions
    torch.testing.assert_close(totals[0],totals[1],rtol=0,atol=0)


@pytest.mark.parametrize('horizon', [0, 3, -1, True, 3.5])
def test_pair_requires_two_completed_routes_within_valid_horizon(horizon):
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (1, 1), 4)
    with pytest.raises(ValueError):
        candidate_pair(env, horizon)
