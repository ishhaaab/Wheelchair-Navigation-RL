"""Constrained maximum-likelihood fitting and whole-trajectory bootstrap.

This module supplies inference, not evidence of successful weight recovery.
All optimization uses float64 and the caller's fixed finite terminal penalty.
"""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np
import torch
from torch.nn.functional import softplus

from wnp.demos import Trajectory, validate_trajectories
from wnp.env import GridMDP, _integer
from wnp.soft_dp import _validate_features, _validate_feature_horizon, plan


def free_weight_start(parameterization: str) -> int:
    """First free physical coordinate, selected explicitly, never by raw shape."""
    if parameterization not in ("fixed_distance", "three_free"):
        raise ValueError("parameterization must be 'fixed_distance' or 'three_free'")
    return 1 if parameterization == "fixed_distance" else 0


def physical_weights(raw: torch.Tensor, *, parameterization="fixed_distance") -> torch.Tensor:
    """Softplus positive weights; the legacy default fixes distance to one."""
    start = free_weight_start(parameterization)
    size = 3 - start
    if (not isinstance(raw, torch.Tensor) or raw.dtype != torch.float64
            or raw.device.type != "cpu" or raw.shape != (size,)
            or not bool(torch.isfinite(raw).all())):
        raise ValueError(f"raw parameters must be a finite CPU float64 tensor of shape [{size}]")
    free = softplus(raw)
    if not bool((free > 0).all()):
        raise ValueError("raw parameters underflow the positive softplus parameterization")
    return torch.cat((raw.new_ones(1), free)) if start else free


def _nonnegative_real(value: float, name: str) -> float:
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite, nonnegative, and non-boolean")
    return float(value)


@dataclass(frozen=True)
class _PreparedDemos:
    horizon: int
    mean_features: torch.Tensor
    initial_distribution: torch.Tensor
    unfinished_fraction: float
    trajectory_count: int
    unfinished_count: int


def _prepare(
    env: GridMDP, features: torch.Tensor, trajectories: Sequence[Trajectory],
) -> _PreparedDemos:
    records, horizon = validate_trajectories(env, trajectories)
    _validate_features(env, features)
    _validate_feature_horizon(features, horizon)
    states = torch.tensor([r.states for r in records], dtype=torch.long)
    actions = torch.tensor([r.actions for r in records], dtype=torch.long).reshape(
        len(records), horizon)
    counts = features[states[:, :-1], actions].sum(dim=1)
    # Divide before summing datasets, avoiding a needless N-times-count overflow.
    mean_features = (counts / len(records)).sum(dim=0)
    distribution = torch.bincount(states[:, 0], minlength=len(env.states)).to(torch.float64)
    distribution = distribution / len(records)
    unfinished = sum(record.arrival_step is None for record in records)
    return _PreparedDemos(horizon, mean_features, distribution,
                          unfinished / len(records), len(records), unfinished)


def _objective(
    env: GridMDP, features: torch.Tensor, prepared: _PreparedDemos,
    weights: torch.Tensor, l2: float, terminal_penalty: float,
    parameterization="fixed_distance",
) -> torch.Tensor:
    start = free_weight_start(parameterization)
    solution = plan(env, features, weights, prepared.horizon, terminal_penalty)
    # Mean NLL = E_demo[w.phi - terminal] + E_initial[log Z].
    nll = (prepared.mean_features @ weights
           + terminal_penalty * prepared.unfinished_fraction
           + (prepared.initial_distribution * solution.values[0]).sum())
    if l2 == 0:
        return nll
    # Scale before squaring: avoid 0*inf and avoid overflowing w**2 when the
    # actual lambda/2*w**2 is representable. sqrt(lambda)/sqrt(2) also avoids
    # prematurely underflowing lambda/2 for positive subnormal lambda.
    scaled = weights[start:] * (math.sqrt(l2) / math.sqrt(2.0))
    return nll + scaled.square().sum()


def loss(
    env: GridMDP, features: torch.Tensor, trajectories: Sequence[Trajectory],
    raw: torch.Tensor, l2: float = 0.0, terminal_penalty: float = 30.0,
    *, parameterization="fixed_distance",
) -> torch.Tensor:
    """Mean trajectory NLL plus lambda/2 times squared free physical weights."""
    l2 = _nonnegative_real(l2, "l2")
    terminal_penalty = _nonnegative_real(terminal_penalty, "terminal_penalty")
    return _objective(env, features, _prepare(env, features, trajectories),
                      physical_weights(raw, parameterization=parameterization), l2,
                      terminal_penalty, parameterization)


def classify_fit(
    objective: float | None, raw_gradient: Sequence[float] | None,
    free_weights: Sequence[float] | None, iterations: int,
    *, parameterization="fixed_distance",
) -> str | None:
    """Return the specified failure reason, or None only when every gate passes."""
    size = 3 - free_weight_start(parameterization)
    if not _integer(iterations) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if objective is None or raw_gradient is None or free_weights is None:
        return "non-finite"
    if len(raw_gradient) != size or len(free_weights) != size:
        raise ValueError(f"classification needs {size} raw gradients and free weights")
    if not all(math.isfinite(x) for x in [objective, *raw_gradient, *free_weights]):
        return "non-finite"
    if any(w < 1e-3 or w > 10 for w in free_weights):
        return "boundary"
    if max(abs(g) for g in raw_gradient) > 1e-6 or iterations >= 150:
        return "unconverged"
    return None


class _NumericalFailure(Exception):
    pass


class NumericalDomainError(ValueError):
    """A study must stop; attached records preserve all attempted refits."""

    def __init__(self, detail, fits):
        super().__init__(detail)
        self.fits = fits


def _numerical_value_error(exc: ValueError) -> bool:
    return _planner_domain_error(exc) or "underflow" in str(exc)


def _planner_domain_error(exc: ValueError) -> bool:
    return any(word in str(exc) for word in
               ("numerical domain", "unrepresentable", "cumulative"))


def _finite_list(tensor: torch.Tensor) -> list[float | None]:
    return [float(x) if math.isfinite(x) else None for x in tensor.tolist()]


def fit(
    env: GridMDP, features: torch.Tensor, trajectories: Sequence[Trajectory],
    l2: float = 0.0, max_iter: int = 150, terminal_penalty: float = 30.0,
    *, initial_raw: torch.Tensor | None = None, parameterization="fixed_distance",
) -> dict:
    """Deterministic float64 LBFGS with strong-Wolfe line search, without clipping.

    Success is based on the final optimized objective, raw gradient <=1e-6,
    iterations <150, and physical free weights in [1e-3,10]. Numerical/domain
    failures during optimization remain failed fits with a recorded detail.
    Non-finite diagnostics are represented by None, never fabricated numbers.
    """
    start = free_weight_start(parameterization)
    l2 = _nonnegative_real(l2, "l2")
    terminal_penalty = _nonnegative_real(terminal_penalty, "terminal_penalty")
    if not _integer(max_iter) or not 1 <= max_iter <= 150:
        raise ValueError("max_iter must be an integer in [1,150]")
    _validate_features(env, features)
    if features.requires_grad:
        raise ValueError("fit requires fixed features, not learnable feature tensors")
    prepared = _prepare(env, features, trajectories)
    if initial_raw is None:
        initial_raw = torch.full((3-start,), math.log(math.expm1(1.0)), dtype=torch.float64)
    initial_weights = physical_weights(initial_raw, parameterization=parameterization)
    # Invalid settings/data and unsupported initial scales raise to the caller.
    _objective(env, features, prepared, initial_weights, l2, terminal_penalty, parameterization)
    raw = initial_raw.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([raw], lr=1.0, max_iter=int(max_iter),
        tolerance_grad=1e-9, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    evaluations = 0
    error_detail = None
    numerical_failure = False
    domain_failure = False

    def closure():
        nonlocal evaluations, domain_failure
        evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        try:
            objective = _objective(env, features, prepared,
                physical_weights(raw, parameterization=parameterization),
                l2, terminal_penalty, parameterization)
        except ValueError as exc:
            if _numerical_value_error(exc):
                # All planner range limits are fatal, not only score-budget
                # messages. Softplus underflow remains a rejected nonfinite fit.
                domain_failure = _planner_domain_error(exc)
                raise _NumericalFailure(str(exc)) from exc
            raise
        if not bool(torch.isfinite(objective)):
            raise _NumericalFailure("non-finite optimized objective")
        objective.backward()
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
            raise _NumericalFailure("non-finite raw gradient")
        return objective

    try:
        optimizer.step(closure)
    except _NumericalFailure as exc:
        numerical_failure, error_detail = True, str(exc)

    objective_value = None
    raw_gradient = physical_gradient = None
    # Retain the actual final parameters, including boundary/diverging values.
    with torch.no_grad():
        final_weights = (torch.cat((raw.new_ones(1), softplus(raw))) if start else softplus(raw))
    if bool(torch.isfinite(raw).all()) and bool((final_weights[start:] > 0).all()):
        try:
            physical = final_weights.clone().requires_grad_(True)
            objective = _objective(env, features, prepared, physical, l2,
                                   terminal_penalty, parameterization)
            full_gradient = torch.autograd.grad(objective, physical)[0]
            free_gradient = full_gradient[start:]
            raw_grad = free_gradient * raw.sigmoid()
            if bool(torch.isfinite(objective)) and bool(torch.isfinite(raw_grad).all()):
                objective_value = objective.item()
                raw_gradient = raw_grad.tolist()
                physical_gradient = _finite_list(free_gradient)
            else:
                numerical_failure, error_detail = True, "non-finite final objective/gradient"
        except ValueError as exc:
            if not _numerical_value_error(exc):
                raise
            numerical_failure, error_detail = True, str(exc)
            domain_failure = domain_failure or _planner_domain_error(exc)
    else:
        numerical_failure, error_detail = True, "non-finite/underflowed final parameters"
    state = optimizer.state.get(raw, {})
    iterations = int(state.get("n_iter", 0))
    failure = ("non-finite" if numerical_failure else
               classify_fit(objective_value, raw_gradient, final_weights[start:].tolist(),
                            iterations, parameterization=parameterization))
    physical_norm = (max(abs(x) for x in physical_gradient)
                     if physical_gradient is not None and all(x is not None for x in physical_gradient)
                     else None)
    budget = terminal_penalty + prepared.horizon * (
        (features @ final_weights).abs().max().item() + math.log(3.))
    ratios = final_weights[1:] / final_weights[0]
    return {
        "parameterization": parameterization, "scale": _finite_list(final_weights)[0],
        "cost_ratios": _finite_list(ratios),
        "score_budget": budget if math.isfinite(budget) else None,
        "numerical_domain_failure": domain_failure,
        "weights": _finite_list(final_weights), "raw_parameters": _finite_list(raw),
        "initial_raw_parameters": initial_raw.tolist(), "objective": objective_value,
        "raw_gradient": raw_gradient,
        "raw_gradient_inf_norm": max(map(abs, raw_gradient)) if raw_gradient is not None else None,
        "physical_free_gradient": physical_gradient, "physical_gradient_inf_norm": physical_norm,
        "iterations": iterations, "closure_evaluations": evaluations,
        "success": failure is None, "failure_reason": failure, "failure_detail": error_detail,
        "terminal_penalty": terminal_penalty, "l2": l2, "max_iter": int(max_iter),
        "trajectory_count": prepared.trajectory_count, "horizon": prepared.horizon,
        "unfinished_count": prepared.unfinished_count,
    }


def summarize_bootstrap(fits: Sequence[dict], *, parameterization="fixed_distance") -> dict:
    """Percentile intervals use successful fits only; every failure is counted.

    Default reliability is at least 80 of 100 successes. For custom budgets use
    max(80, ceil(0.8*B)); small unit-test budgets are always marked unreliable.
    """
    start = free_weight_start(parameterization)
    if not fits:
        raise ValueError("bootstrap fits must not be empty")
    successful = []
    failures = {}
    for result in fits:
        if result["success"]:
            weights = result["weights"]
            if len(weights) != 3 or not all(w is not None and math.isfinite(w) and w > 0
                                             for w in weights):
                raise ValueError("successful bootstrap fit must have finite weights")
            successful.append(weights)
        else:
            reason = result["failure_reason"]
            if reason not in ("non-finite", "boundary", "unconverged"):
                raise ValueError("bootstrap failure must have a specified reason")
            failures[reason] = failures.get(reason, 0) + 1
    required = max(80, math.ceil(.8 * len(fits)))
    data = np.asarray(successful)
    intervals = (np.percentile(data[:, start:], [2.5, 97.5], axis=0).T.tolist()
                 if successful else None)
    return {
        "ratio_percentile_intervals": (np.percentile(data[:, 1:] / data[:, :1],
            [2.5, 97.5], axis=0).T.tolist() if successful else None),
        "scale_percentile_interval": (np.percentile(data[:, 0], [2.5, 97.5]).tolist()
                                      if successful else None),
        "refits": len(fits), "successful_refits": len(successful),
        "failed_refits": len(fits) - len(successful), "failure_counts": failures,
        "percentile_intervals": intervals, "percentiles": [2.5, 97.5],
        "required_successes": required, "reliable": len(successful) >= required,
    }


def bootstrap_fit(
    env: GridMDP, features: torch.Tensor, trajectories: Sequence[Trajectory],
    seed: int, refits: int = 100, l2: float = 0.0, max_iter: int = 150,
    terminal_penalty: float = 30.0, *, parameterization="fixed_distance",
) -> dict:
    """Resample N whole records with replacement and refit the identical model."""
    free_weight_start(parameterization)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be an explicit nonnegative, non-boolean integer")
    if not _integer(refits) or refits < 1:
        raise ValueError("refits must be a positive integer")
    records, _ = validate_trajectories(env, trajectories)
    rng = np.random.default_rng(int(seed))
    fits = []
    for index in range(int(refits)):
        resample_seed = int(rng.integers(2**63 - 1))
        indices = np.random.default_rng(resample_seed).integers(len(records), size=len(records))
        selected = [records[int(i)] for i in indices]
        result = fit(env, features, selected, l2, max_iter, terminal_penalty,
                     parameterization=parameterization)
        result.update(bootstrap_index=index, resample_seed=resample_seed)
        fits.append(result)
        if parameterization == "three_free" and result["numerical_domain_failure"]:
            raise NumericalDomainError(result["failure_detail"], fits)
    return {**summarize_bootstrap(fits, parameterization=parameterization),
            "fits": fits, "seed": int(seed),
            "resampling_unit": "whole trajectory", "terminal_penalty": float(terminal_penalty),
            "l2": float(l2)}
