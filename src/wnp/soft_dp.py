"""Exact differentiable float64 finite-horizon maximum-entropy planning."""

from dataclasses import dataclass
from typing import Sequence

import math
import torch

from wnp.demos import (
    Trajectory, _check_environment, _validate_horizon, _validate_starts,
    validate_trajectories,
)
from wnp.env import GridMDP


@dataclass(frozen=True)
class PlanResult:
    """All-state values [T+1,S] and log policy [T,S,3], with autograd intact."""

    values: torch.Tensor
    log_policy: torch.Tensor
    terminal_penalty: float
    env: GridMDP

    @property
    def horizon(self) -> int:
        return self.log_policy.shape[0]


def _validate_features(env: GridMDP, features: torch.Tensor) -> None:
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float64
            or features.device.type != "cpu" or features.shape != (len(env.states), 3, 3)):
        raise ValueError("features must be a CPU float64 tensor of shape [S,3,3]")
    if not bool(torch.isfinite(features).all()) or bool((features < 0).any()):
        raise ValueError("features must be finite nonnegative costs")
    if bool((features[~env.legal] != 0).any()):
        raise ValueError("invalid-action features must be exactly zero")
    if bool((features[env.is_goal] != 0).any()):
        raise ValueError("goal-source features must be exactly zero")


def _validate_feature_horizon(features: torch.Tensor, horizon: int) -> None:
    """Conservative all-state bound for total features and weight derivatives.

    A small weight can hide a huge feature in the reward while its cumulative
    feature count, and hence a required weight derivative, still overflows.
    Divide the float64 limit by T rather than multiplying features by T, so the
    validation itself cannot overflow. The bound may reject a large feature
    that cannot actually recur T times; no clipping or graph disconnection is
    used to pretend such numerical scales are supported.
    """
    if horizon and bool((features > torch.finfo(torch.float64).max / horizon).any()):
        raise ValueError("cumulative features and weight derivatives may exceed float64 range")


def _validate_model(
    env: GridMDP, features: torch.Tensor, weights: torch.Tensor,
    horizon: int, terminal_penalty: float,
) -> int:
    horizon = _validate_horizon(horizon)
    if not math.isfinite(terminal_penalty) or terminal_penalty < 0:
        raise ValueError("terminal_penalty must be finite and nonnegative")
    if not bool(env.legal.any(dim=1).all()):
        raise ValueError("every state must have at least one legal action")
    _validate_features(env, features)
    _validate_feature_horizon(features, horizon)
    if (not isinstance(weights, torch.Tensor) or weights.dtype != torch.float64
            or weights.device.type != "cpu" or weights.shape != (3,)
            or not bool(torch.isfinite(weights).all())):
        raise ValueError("weights must be a finite CPU float64 tensor of shape [3]")
    return horizon


def plan(
    env: GridMDP,
    features: torch.Tensor,
    weights: torch.Tensor,
    horizon: int,
    terminal_penalty: float = 30.0,
) -> PlanResult:
    """Backward soft DP over ALL states, never pruning unreachable components.

    M is a fixed finite penalty, not a completion constraint. Its fixture-specific
    reference-weight margin is diagnosed separately, not changed during fitting.
    Exact-zero invalid features keep reward arithmetic finite. Mask only after
    arithmetic with masked_fill, never multiplication by an infinite sentinel.
    """
    horizon = _validate_model(env, features, weights, horizon, terminal_penalty)
    reward = -(features @ weights)
    if not bool(torch.isfinite(reward).all()):
        raise ValueError("features and weights produce unrepresentable rewards")
    masked_reward = reward.masked_fill(~env.legal, -torch.inf)
    terminal = weights.new_full((len(env.states),), -float(terminal_penalty)).masked_fill(
        env.is_goal, 0.0)
    # Zero dependency keeps even T=0 and all-goal fixtures differentiable.
    terminal = terminal + weights[0] * 0.0
    values = [terminal]
    policies = []
    for _ in range(horizon):
        q = masked_reward + values[-1][env.next_state]
        if not bool(torch.isfinite(q[env.legal]).all()):
            raise ValueError("weights and horizon produce unrepresentable legal scores")
        value = torch.logsumexp(q, dim=1)
        policies.append(q - value[:, None])
        values.append(value)
    stacked_values = torch.stack(values[::-1])
    stacked_policy = (torch.stack(policies[::-1]) if policies else
                      weights.new_empty((0, len(env.states), 3)) + weights[0] * 0.0)
    return PlanResult(stacked_values, stacked_policy, float(terminal_penalty), env)


def trajectory_log_prob(
    env: GridMDP,
    features: torch.Tensor,
    weights: torch.Tensor,
    trajectories: Sequence[Trajectory],
    terminal_penalty: float = 30.0,
) -> torch.Tensor:
    """One conditional log likelihood per legal fixed-length trajectory.

    Includes the fixed terminal score for unfinished records. No time or batch
    reduction occurs here. Different initial states may share a horizon.
    """
    records, horizon = validate_trajectories(env, trajectories)
    result = plan(env, features, weights, horizon, terminal_penalty)
    states = torch.tensor([r.states for r in records], dtype=torch.long)
    actions = torch.tensor([r.actions for r in records], dtype=torch.long).reshape(
        len(records), horizon)
    counts = features[states[:, :-1], actions].sum(dim=1)
    scores = -(counts @ weights) + result.values[-1, states[:, -1]]
    return scores - result.values[0, states[:, 0]]


def expected_features(
    env: GridMDP,
    features: torch.Tensor,
    solution: PlanResult,
    starts: Sequence[int],
) -> torch.Tensor:
    """Exact mean total cost features by forward state-action occupancy.

    Starts define an empirical initial-state distribution, including duplicates.
    Goal padding contributes zero; unfinished mass is retained. The returned
    expectation remains differentiable through the time-dependent policy.
    """
    _check_environment(env, solution)
    _validate_features(env, features)
    _validate_feature_horizon(features, solution.horizon)
    starts = _validate_starts(env, starts)
    indices = torch.tensor(starts, dtype=torch.long)
    occupancy = features.new_zeros(len(env.states)).scatter_add(
        0, indices, features.new_full((len(starts),), 1 / len(starts)))
    total = features.new_zeros(3) + solution.values[0, 0] * 0.0
    for t in range(solution.horizon):
        mass = occupancy[:, None] * solution.log_policy[t].exp()
        total = total + (mass[:, :, None] * features).sum(dim=(0, 1))
        occupancy = occupancy.new_zeros(len(env.states)).scatter_add(
            0, env.next_state.reshape(-1), mass.reshape(-1))
    return total
