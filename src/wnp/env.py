"""Deterministic free-cell/heading MDP with masked navigation controls."""

from collections import deque
from numbers import Integral

import numpy as np
import torch


def _occupancy_array(occupied: np.ndarray) -> np.ndarray:
    """Validate occupancy without silently interpreting numeric costs as walls."""
    array = np.asarray(occupied)
    if array.ndim != 2 or 0 in array.shape or array.dtype != np.bool_:
        raise ValueError("occupied must be a nonempty 2D boolean array")
    return array


def _integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _coordinate(value: tuple, bounds: tuple[int, ...], name: str) -> tuple[int, ...]:
    try:
        entries = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer coordinate") from exc
    if len(entries) != len(bounds) or any(
        not _integer(x) or not 0 <= x < limit for x, limit in zip(entries, bounds)
    ):
        raise ValueError(f"{name} must have integer entries within {bounds}")
    return tuple(int(x) for x in entries)


class GridMDP:
    """CPU transition tables indexed by row-major free cell, then heading.

    Start is (row, column, heading); goal is (row, column). The occupancy is
    copied and read-only. Disconnected hand-built fixtures are allowed.
    next_state uses source-state placeholders for illegal actions, whose masks
    are false. Consumers must honor legal rather than use those placeholders.
    """

    def __init__(
        self,
        occupied: np.ndarray,
        start: tuple[int, int, int],
        goal: tuple[int, int],
        headings: int = 8,
    ) -> None:
        if not _integer(headings) or headings not in (4, 8):
            raise ValueError("headings must be 4 or 8")
        self.occupied = _occupancy_array(occupied).copy()
        self.occupied.flags.writeable = False
        self.headings = int(headings)
        self.shape = self.occupied.shape
        self.start = _coordinate(start, (*self.shape, self.headings), "start")
        self.goal = _coordinate(goal, self.shape, "goal")
        if self.occupied[self.start[:2]] or self.occupied[self.goal]:
            raise ValueError("start and goal must be free cells")

        all_directions = ((0, 1), (1, 1), (1, 0), (1, -1),
                          (0, -1), (-1, -1), (-1, 0), (-1, 1))
        self.directions = all_directions[:: 8 // self.headings]
        self.states = tuple(
            (r, c, h)
            for r in range(self.shape[0])
            for c in range(self.shape[1])
            if not self.occupied[r, c]
            for h in range(self.headings)
        )
        self.state_index = {state: index for index, state in enumerate(self.states)}
        self.start_state = self.state_index[self.start]
        count = len(self.states)
        successors = np.repeat(np.arange(count, dtype=np.int64)[:, None], 3, axis=1)
        legal = np.zeros((count, 3), dtype=bool)
        goals = np.zeros(count, dtype=bool)

        for s, (r, c, h) in enumerate(self.states):
            if (r, c) == self.goal:
                goals[s] = True
                legal[s, 0] = True
                continue
            legal[s, 1:] = True
            successors[s, 1] = self.state_index[(r, c, (h - 1) % self.headings)]
            successors[s, 2] = self.state_index[(r, c, (h + 1) % self.headings)]
            dr, dc = self.directions[h]
            rr, cc = r + dr, c + dc
            if not (0 <= rr < self.shape[0] and 0 <= cc < self.shape[1]):
                continue
            if self.occupied[rr, cc]:
                continue
            if dr and dc and (self.occupied[r + dr, c] or self.occupied[r, c + dc]):
                continue
            legal[s, 0] = True
            successors[s, 0] = self.state_index[(rr, cc, h)]

        self.next_state = torch.from_numpy(successors)
        self.legal = torch.from_numpy(legal)
        self.is_goal = torch.from_numpy(goals)

    def goal_reachable(self) -> bool:
        """Search the actual legal state graph from the configured start."""
        successors = self.next_state.numpy()
        legal = self.legal.numpy()
        goals = self.is_goal.numpy()
        seen = {self.start_state}
        pending = deque([self.start_state])
        while pending:
            state = pending.popleft()
            if goals[state]:
                return True
            for action in range(3):
                if legal[state, action]:
                    successor = int(successors[state, action])
                    if successor not in seen:
                        seen.add(successor)
                        pending.append(successor)
        return False


def generate_layout(
    seed: int,
    shape: tuple[int, int] = (20, 20),
    obstacle_probability: float = 0.2,
    headings: int = 8,
    max_attempts: int = 1000,
    *,
    start: tuple[int, int, int] | None = None,
    goal: tuple[int, int] | None = None,
) -> GridMDP:
    """Seeded rejection sampling conditioned on start-goal connectivity.

    Defaults are top-left facing E and bottom-right goal. Occupancy draws are
    independent before conditioning, with endpoints forced free. Raises on
    exhaustion; no carving or silent fallback changes the distribution.
    Seed must be an explicit nonnegative Python or NumPy integer, not a bool.
    """
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be an explicit nonnegative integer, not a boolean")
    try:
        shape = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must contain two positive integers") from exc
    if len(shape) != 2 or any(not _integer(x) or x <= 0 for x in shape):
        raise ValueError("shape must contain two positive integers")
    if not _integer(headings) or headings not in (4, 8):
        raise ValueError("headings must be 4 or 8")
    if not _integer(max_attempts) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if not np.isfinite(obstacle_probability) or not 0 <= obstacle_probability <= 1:
        raise ValueError("obstacle_probability must be finite and within [0, 1]")
    start = _coordinate((0, 0, 0) if start is None else start,
                        (*shape, headings), "start")
    goal = _coordinate((shape[0] - 1, shape[1] - 1) if goal is None else goal,
                       shape, "goal")
    rng = np.random.default_rng(int(seed))
    for _ in range(max_attempts):
        occupied = rng.random(shape) < obstacle_probability
        occupied[start[:2]] = False
        occupied[goal] = False
        env = GridMDP(occupied, start, goal, headings)
        if env.goal_reachable():
            return env
    raise RuntimeError(f"no connected layout found in {max_attempts} attempts")
