"""Fixed-horizon legal action-labelled trajectories, enumeration, and sampling."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from wnp.env import GridMDP, _integer

if TYPE_CHECKING:
    from wnp.soft_dp import PlanResult


@dataclass(frozen=True)
class Trajectory:
    """T actions and T+1 states, with first goal arrival or None if unfinished."""

    states: tuple[int, ...]
    actions: tuple[int, ...]
    arrival_step: int | None


def _validate_horizon(horizon: int) -> int:
    if not _integer(horizon) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    return int(horizon)


def _validate_starts(env: GridMDP, starts: Sequence[int]) -> tuple[int, ...]:
    if isinstance(starts, torch.Tensor):
        if starts.ndim != 1 or starts.dtype != torch.long or starts.device.type != "cpu":
            raise ValueError("starts tensor must be 1D CPU torch.long")
        starts = starts.tolist()
    try:
        entries = tuple(starts)
    except TypeError as exc:
        raise ValueError("starts must be a nonempty sequence of state indices") from exc
    if not entries or any(not _integer(s) or not 0 <= s < len(env.states) for s in entries):
        raise ValueError("starts must be nonempty and contain valid integer state indices")
    return tuple(int(s) for s in entries)


def _arrival_step(env: GridMDP, states: tuple[int, ...]) -> int | None:
    return next((t for t, s in enumerate(states) if bool(env.is_goal[s])), None)


def validate_trajectories(
    env: GridMDP, trajectories: Sequence[Trajectory]
) -> tuple[tuple[Trajectory, ...], int]:
    """Validate length, labels, deterministic transitions, and arrival metadata."""
    records = tuple(trajectories)
    if not records:
        raise ValueError("trajectories must not be empty")
    horizon = None
    for record in records:
        if not isinstance(record, Trajectory):
            raise ValueError("trajectories must contain Trajectory records")
        if not isinstance(record.states, tuple) or not isinstance(record.actions, tuple):
            raise ValueError("trajectory states and actions must be tuples")
        if len(record.states) != len(record.actions) + 1:
            raise ValueError("trajectory needs T+1 states for T actions")
        if horizon is None:
            horizon = len(record.actions)
        elif len(record.actions) != horizon:
            raise ValueError("all trajectories must have the same horizon")
        _validate_starts(env, record.states)
        for t, action in enumerate(record.actions):
            if not _integer(action) or not 0 <= action < 3:
                raise ValueError("trajectory action must be an integer in [0, 2]")
            state = record.states[t]
            if not bool(env.legal[state, action]):
                raise ValueError("trajectory contains an illegal action")
            if int(env.next_state[state, action]) != record.states[t + 1]:
                raise ValueError("trajectory has an inconsistent transition")
        arrival = record.arrival_step
        if arrival is not None and (not _integer(arrival) or arrival < 0):
            raise ValueError("arrival_step must be a nonnegative integer or None")
        if arrival != _arrival_step(env, record.states):
            raise ValueError("arrival_step does not match the first goal state")
    return records, int(horizon)


def enumerate_trajectories(env: GridMDP, start: int, horizon: int) -> list[Trajectory]:
    """Enumerate every legal length-T action sequence without merging paths.

    This is exponential in horizon and intended only for tiny verification MDPs.
    Goal arrival is padded by its sole legal continuation; unfinished paths stay.
    """
    start = _validate_starts(env, [start])[0]
    horizon = _validate_horizon(horizon)
    pending = [((start,), ())]
    records = []
    while pending:
        states, actions = pending.pop()
        if len(actions) == horizon:
            records.append(Trajectory(states, actions, _arrival_step(env, states)))
            continue
        state = states[-1]
        for action in (2, 1, 0):
            if bool(env.legal[state, action]):
                successor = int(env.next_state[state, action])
                pending.append((states + (successor,), actions + (action,)))
    return records


def _check_environment(env: GridMDP, solution: "PlanResult") -> None:
    if solution.env is not env:
        raise ValueError("solution belongs to a different environment")


def sample(
    env: GridMDP,
    solution: "PlanResult",
    starts: Sequence[int],
    seed: int,
) -> list[Trajectory]:
    """Sample each given start once using a local RNG, without filtering.

    Sampling is not differentiable. It does not modify or disconnect the
    differentiable solution, and never rejects an unfinished trajectory.
    """
    _check_environment(env, solution)
    starts = _validate_starts(env, starts)
    if not _integer(seed) or not 0 <= seed < 2**64:
        raise ValueError("seed must be a non-boolean integer within [0, 2**64)")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    current = torch.tensor(starts, dtype=torch.long)
    state_columns = [current]
    action_columns = []
    with torch.no_grad():
        for t in range(solution.horizon):
            probabilities = solution.log_policy[t, current].exp()
            actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            current = env.next_state[current, actions]
            action_columns.append(actions)
            state_columns.append(current)
    states = torch.stack(state_columns, dim=1).tolist()
    actions = (torch.stack(action_columns, dim=1).tolist()
               if action_columns else [[] for _ in starts])
    return [Trajectory(tuple(ss), tuple(aa), _arrival_step(env, tuple(ss)))
            for ss, aa in zip(states, actions)]
