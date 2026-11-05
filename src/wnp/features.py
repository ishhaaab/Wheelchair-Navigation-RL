"""Float64 action costs in distance, turning, clearance order."""

import math

import numpy as np
import torch

from wnp.env import GridMDP, _occupancy_array


def clearance_distances(occupied: np.ndarray) -> np.ndarray:
    """Exact Euclidean distances to occupied cell centers, in grid units.

    Empty occupancy has distance +inf everywhere. Boundaries are not obstacles.
    Pairwise calculation is intentional for the small grids used here.
    """
    occupied = _occupancy_array(occupied)
    obstacles = np.argwhere(occupied).astype(np.float64)
    if len(obstacles) == 0:
        return np.full(occupied.shape, np.inf, dtype=np.float64)
    cells = np.indices(occupied.shape, dtype=np.float64).reshape(2, -1).T
    offsets = cells[:, None, :] - obstacles[None, :, :]
    squared = np.sum(offsets**2, axis=2)
    return np.sqrt(squared.min(axis=1)).reshape(occupied.shape)


def feature_table(env: GridMDP, d_safe: float = 2.0) -> torch.Tensor:
    """Return CPU float64 [state, action, feature] nonnegative costs.

    Clearance is charged at the destination on both movement and turns.
    Goal-source and illegal actions have zero features; the latter remain
    illegal and must be masked by consumers. Entering a goal pays full cost.
    Raises ValueError if clearance costs cannot be represented in float64.
    """
    if not np.isfinite(d_safe) or d_safe < 0:
        raise ValueError("d_safe must be finite and nonnegative")
    distances = clearance_distances(env.occupied)
    try:
        with np.errstate(over="raise", invalid="raise"):
            clearance = np.maximum(0.0, d_safe - distances) ** 2
    except FloatingPointError as exc:
        raise ValueError("clearance costs must be representable in float64") from exc
    phi = np.zeros((len(env.states), 3, 3), dtype=np.float64)
    states = np.asarray(env.states, dtype=np.int64)
    successors = env.next_state.numpy()
    legal = env.legal.numpy()
    goals = env.is_goal.numpy()
    for s, (r, c, _) in enumerate(env.states):
        if goals[s]:
            continue
        for a in range(3):
            if not legal[s, a]:
                continue
            rr, cc, _ = states[successors[s, a]]
            if a == 0:
                phi[s, a, 0] = math.hypot(int(rr) - r, int(cc) - c)
            else:
                phi[s, a, 1] = 2 * math.pi / env.headings
            phi[s, a, 2] = clearance[rr, cc]
    return torch.from_numpy(phi)
