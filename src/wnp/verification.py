"""Shared exhaustive distribution checks used by tests and artifact generation.

Run `python -m wnp.verification` to write results/verification.json. This module
does not run Task 4 finite differences, gradcheck, optimization, or recovery.
"""

import argparse
from itertools import product
import json
import math
from pathlib import Path
import platform

import numpy as np
import torch

from wnp.demos import enumerate_trajectories, _validate_starts
from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.soft_dp import _validate_model, expected_features, plan, trajectory_log_prob


def exhaustive_fixture(name: str) -> tuple[GridMDP, torch.Tensor, torch.Tensor, int]:
    """The two predeclared short-horizon Test A fixtures in SPEC.md."""
    if name == "four_heading":
        env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (0, 1), headings=4)
        weights, horizon = [1., .7, 1.2], 4
    elif name == "eight_heading":
        occupied = np.zeros((3, 3), dtype=bool)
        occupied[0, 1] = True
        env = GridMDP(occupied, (1, 0, 0), (2, 2), headings=8)
        weights, horizon = [1.3, .7, 1.2], 5
    else:
        raise ValueError(f"unknown exhaustive fixture: {name}")
    return env, feature_table(env), torch.tensor(weights, dtype=torch.float64), horizon


def terminal_margin_diagnostic(
    env: GridMDP,
    features: torch.Tensor,
    reference_weights: torch.Tensor,
    horizon: int,
    terminal_penalty: float = 30.0,
    margin: float = 1.0,
    starts: tuple[int, ...] | None = None,
) -> dict:
    """Check M against optimal finishing costs at declared reference weights.

    Compute finite-horizon hard minimum costs with +inf for unfinished terminal
    states. Compare only reachable state/time pairs that can still finish, and
    separately count reachable pairs that cannot. This is a diagnostic, not a
    parameter adjustment and not a probability-of-completion guarantee.
    """
    horizon = _validate_model(env, features, reference_weights, horizon, terminal_penalty)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    starts = _validate_starts(env, [env.start_state] if starts is None else starts)
    with torch.no_grad():
        costs = (features @ reference_weights).masked_fill(~env.legal, torch.inf)
        if not bool(torch.isfinite(costs[env.legal]).all()):
            raise ValueError("reference weights produce unrepresentable finishing costs")
        terminal = torch.full((len(env.states),), torch.inf, dtype=torch.float64)
        terminal[env.is_goal] = 0
        optimal = [terminal]
        for _ in range(horizon):
            optimal.append((costs + optimal[-1][env.next_state]).min(dim=1).values)
        optimal = optimal[::-1]
        reachable = torch.zeros(len(env.states), dtype=torch.bool)
        reachable[list(starts)] = True
        starts_can_finish = bool(torch.isfinite(optimal[0][list(starts)]).all())
        maximum = 0.0
        viable_pairs = 0
        unfinishable_pairs = 0
        for t in range(horizon + 1):
            viable = reachable & torch.isfinite(optimal[t])
            viable_pairs += int(viable.sum())
            unfinishable_pairs += int((reachable & ~torch.isfinite(optimal[t])).sum())
            if bool(viable.any()):
                maximum = max(maximum, float(optimal[t][viable].max()))
            if t < horizon:
                successors = env.next_state[reachable][env.legal[reachable]]
                reachable = torch.zeros_like(reachable)
                reachable[successors] = True
    return {
        "terminal_penalty": float(terminal_penalty),
        "margin": float(margin),
        "reference_weights": reference_weights.tolist(),
        "starts": list(starts),
        "horizon": horizon,
        "max_optimal_finishing_cost": maximum,
        "finishable_reachable_pairs": viable_pairs,
        "unfinishable_reachable_pairs": unfinishable_pairs,
        "starts_can_finish": starts_can_finish,
        "passed": starts_can_finish and terminal_penalty > maximum + margin,
        "limitation": "Reference weights only; finite M does not guarantee completion.",
    }


def _independent_legal_sequences(env: GridMDP, horizon: int) -> set[tuple[int, ...]]:
    """Independent cartesian-product enumeration to catch missing DFS branches."""
    legal = set()
    for actions in product(range(3), repeat=horizon):
        state = env.start_state
        for action in actions:
            if not bool(env.legal[state, action]):
                break
            state = int(env.next_state[state, action])
        else:
            legal.add(actions)
    return legal


def verify_distribution(name: str, tolerance: float = 1e-10) -> dict:
    """Compare every action-labelled trajectory and exact feature expectations.

    Reference probabilities use NumPy logaddexp normalization of full trajectory
    costs plus terminal score. They do not use soft-DP values or policies.
    """
    env, features, weights, horizon = exhaustive_fixture(name)
    penalty = 30.0
    diagnostic = terminal_margin_diagnostic(env, features, weights, horizon, penalty)
    assert diagnostic["passed"], "fixture terminal penalty margin failed"
    records = enumerate_trajectories(env, env.start_state, horizon)
    independent = _independent_legal_sequences(env, horizon)
    assert len(records) == len(independent)
    assert {record.actions for record in records} == independent
    feature_array = features.numpy()
    count_rows = []
    finished = []
    for record in records:
        count_rows.append(sum((feature_array[s, a] for s, a in
                               zip(record.states[:-1], record.actions)), np.zeros(3)))
        finished.append(bool(env.is_goal[record.states[-1]]))
    counts = np.asarray(count_rows, dtype=np.float64)
    terminal_scores = np.where(finished, 0.0, -penalty)
    scores = -(counts @ weights.numpy()) + terminal_scores
    reference = np.exp(scores - np.logaddexp.reduce(scores))
    solution = plan(env, features, weights, horizon, penalty)
    policy_products = []
    diagonal = False
    for record in records:
        probabilities = []
        for t, (state, action) in enumerate(zip(record.states[:-1], record.actions)):
            assert bool(env.legal[state, action]), "enumeration used a blocked move"
            probabilities.append(solution.log_policy[t, state, action].exp())
            r, c, _ = env.states[state]
            rr, cc, _ = env.states[record.states[t + 1]]
            diagonal = diagonal or (abs(rr - r) == 1 and abs(cc - c) == 1)
        policy_products.append(float(torch.stack(probabilities).prod()))
    policy_products = np.asarray(policy_products)
    likelihood_probabilities = trajectory_log_prob(
        env, features, weights, records, penalty).exp().numpy()
    exact_features = reference @ counts
    occupancy_features = expected_features(env, features, solution, [env.start_state]).numpy()

    blocked_checks = []
    if name == "eight_heading":
        for time, triple, reason in (
            (1, (1, 0, 7), "occupied destination"),
            (2, (1, 1, 7), "corner blocking"),
            (4, (1, 1, 5), "corner blocking"),
        ):
            state = env.state_index[triple]
            assert not bool(env.legal[state, 0])
            visits = [record for record in records if record.states[time] == state]
            assert visits, "blocked check state not reached at declared time"
            assert all(record.actions[time] != 0 for record in visits)
            blocked_checks.append({"time": time, "state": list(triple), "reason": reason,
                                   "visiting_sequences": len(visits)})
        assert diagonal, "eight-heading fixture did not enumerate a legal diagonal"

    report = {
        "shape": list(env.shape), "headings": env.headings, "horizon": horizon,
        "start": list(env.start), "goal": list(env.goal),
        "occupied_cells": np.argwhere(env.occupied).tolist(),
        "weights": weights.tolist(), "terminal_penalty": penalty,
        "terminal_margin": diagnostic, "tolerance": tolerance,
        "trajectory_count": len(records), "independent_sequence_count": len(independent),
        "completed_count": sum(finished), "unfinished_count": len(records) - sum(finished),
        "max_probability_error": float(np.max(np.abs(policy_products - reference))),
        "max_likelihood_probability_error": float(np.max(np.abs(
            likelihood_probabilities - reference))),
        "reference_normalization_error": float(abs(reference.sum() - 1)),
        "policy_normalization_error": float(abs(policy_products.sum() - 1)),
        "exhaustive_expected_features": exact_features.tolist(),
        "occupancy_expected_features": occupancy_features.tolist(),
        "max_expected_features_error": float(np.max(np.abs(occupancy_features - exact_features))),
        "legal_diagonal_used": diagonal, "blocked_move_checks": len(blocked_checks),
        "blocked_moves": blocked_checks,
    }
    assert report["completed_count"] > 0 and report["unfinished_count"] > 0
    for metric in ("max_probability_error", "max_likelihood_probability_error",
                   "reference_normalization_error", "policy_normalization_error",
                   "max_expected_features_error"):
        assert report[metric] < tolerance, f"{name}: {metric} exceeds tolerance"
    return report


def write_verification(path: str | Path) -> dict:
    """Generate numerical evidence only after both shared fixture checks pass."""
    report = {
        "schema_version": 1, "scope": "Task 3 exhaustive distribution verification",
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__},
        "fixtures": {name: verify_distribution(name)
                     for name in ("four_heading", "eight_heading")},
        "task4_checks": "NOT RUN",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results/verification.json")
    args = parser.parse_args()
    report = write_verification(args.output)
    for name, fixture in report["fixtures"].items():
        print(f"{name}: trajectories={fixture['trajectory_count']}, "
              f"probability_error={fixture['max_probability_error']:.16g}, "
              f"expected_features_error={fixture['max_expected_features_error']:.16g}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
