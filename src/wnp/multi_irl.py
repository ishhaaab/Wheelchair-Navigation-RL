"""Shared reward inference across independent, possibly unequal MDP batches.

Reuses the verified single-environment sufficient statistics and likelihood.
No layout generation, sampling, evaluation, or study orchestration occurs here.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch.nn.functional import softplus

from wnp.demos import Trajectory
from wnp.env import GridMDP, _integer
from wnp.irl import (
    NumericalDomainError, _finite_list, _nonnegative_real, _NumericalFailure,
    _numerical_value_error, _objective, _planner_domain_error, _prepare,
    classify_fit, free_weight_start, physical_weights,
)


@dataclass(frozen=True)
class EnvironmentDemos:
    """One environment's fixed features, labels, and fixed terminal penalty."""

    env: GridMDP
    features: torch.Tensor
    trajectories: Sequence[Trajectory]
    terminal_penalty: float = 30.


def _prepare_batches(batches, *, fitting=False):
    batches = tuple(batches)
    if not batches:
        raise ValueError("batches must not be empty")
    prepared = []
    for index, batch in enumerate(batches):
        if not isinstance(batch, EnvironmentDemos) or not isinstance(batch.env, GridMDP):
            raise ValueError("batches must contain EnvironmentDemos with GridMDP environments")
        penalty = _nonnegative_real(batch.terminal_penalty, "terminal_penalty")
        try:
            demos = _prepare(batch.env, batch.features, batch.trajectories)
        except ValueError as exc:
            if fitting and _planner_domain_error(exc):
                attempt = {"numerical_domain_failure": True, "phase": "preparation",
                           "environment_index": index, "failure_detail": str(exc),
                           "closure_evaluations": 0}
                raise NumericalDomainError(str(exc), [attempt]) from exc
            raise
        if fitting and batch.features.requires_grad:
            raise ValueError("fit requires fixed features, not learnable feature tensors")
        prepared.append((batch.env, batch.features, demos, penalty))
    return prepared


def _pooled_objective(prepared, weights, l2, parameterization):
    count = sum(demos.trajectory_count for _, _, demos, _ in prepared)
    # Weighted per-environment means equal sum of all NLLs / total N.
    # Divide first to avoid an unnecessary multiplication by large counts.
    terms = [_objective(env, phi, demos, weights, 0., penalty, parameterization)
             * (demos.trajectory_count / count) for env, phi, demos, penalty in prepared]
    nll = torch.stack(terms).sum()
    if l2 == 0:
        return nll
    start = free_weight_start(parameterization)
    scaled = weights[start:] * (math.sqrt(l2) / math.sqrt(2.))
    return nll + scaled.square().sum()


def loss_multi(
    batches: Sequence[EnvironmentDemos], raw: torch.Tensor, l2: float = 0.,
    *, parameterization="fixed_distance",
) -> torch.Tensor:
    """Pooled mean trajectory NLL plus one physical-weight L2 penalty.

    Different batches may have different horizons, starts, shapes, goals and M.
    A batch itself must contain a nonempty, single-horizon set of legal labels.
    Domain violations raise the planner's ValueError without altering its limit.
    """
    l2 = _nonnegative_real(l2, "l2")
    weights = physical_weights(raw, parameterization=parameterization)
    return _pooled_objective(_prepare_batches(batches), weights, l2, parameterization)


def fit_multi(
    batches: Sequence[EnvironmentDemos], l2: float = 0., max_iter: int = 150,
    *, initial_raw: torch.Tensor | None = None, parameterization="fixed_distance",
) -> dict:
    """Fit one shared reward by float64 LBFGS, with existing success criteria.

    Numerical-domain violations in initial, trial or final evaluations raise
    NumericalDomainError with the attempted parameter diagnostics attached.
    They must not be treated as skippable failed fits by a study runner.
    Other nonfinite, boundary and unconverged fits remain explicit result rows.
    No clipping or alternate-start retries are performed.
    """
    start = free_weight_start(parameterization)
    l2 = _nonnegative_real(l2, "l2")
    if not _integer(max_iter) or not 1 <= max_iter <= 150:
        raise ValueError("max_iter must be an integer in [1,150]")
    prepared = _prepare_batches(batches, fitting=True)
    if initial_raw is None:
        initial_raw = torch.full((3-start,), math.log(math.expm1(1.)), dtype=torch.float64)
    initial_weights = physical_weights(initial_raw, parameterization=parameterization)
    raw = initial_raw.detach().clone().requires_grad_(True)
    evaluations = 0

    def evaluate(weights, phase):
        try:
            return _pooled_objective(prepared, weights, l2, parameterization)
        except ValueError as exc:
            if _planner_domain_error(exc):
                attempt = {"numerical_domain_failure": True, "phase": phase,
                           "failure_detail": str(exc), "parameterization": parameterization,
                           "weights": _finite_list(weights.detach()),
                           "raw_parameters": _finite_list(raw.detach()),
                           "closure_evaluations": evaluations}
                raise NumericalDomainError(str(exc), [attempt]) from exc
            raise

    evaluate(initial_weights, "initial")
    optimizer = torch.optim.LBFGS([raw], lr=1., max_iter=int(max_iter),
        tolerance_grad=1e-9, tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        nonlocal evaluations
        evaluations += 1
        optimizer.zero_grad(set_to_none=True)
        try:
            objective = evaluate(physical_weights(raw, parameterization=parameterization),
                                 "optimization")
        except NumericalDomainError:
            raise
        except ValueError as exc:
            if _numerical_value_error(exc):
                raise _NumericalFailure(str(exc)) from exc
            raise
        if not bool(torch.isfinite(objective)):
            raise _NumericalFailure("non-finite optimized objective")
        objective.backward()
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
            raise _NumericalFailure("non-finite raw gradient")
        return objective

    numerical_failure = False
    error_detail = None
    try:
        optimizer.step(closure)
    except _NumericalFailure as exc:
        numerical_failure, error_detail = True, str(exc)

    objective_value = None
    raw_gradient = physical_gradient = None
    with torch.no_grad():
        final_weights = (torch.cat((raw.new_ones(1), softplus(raw))) if start else softplus(raw))
    if bool(torch.isfinite(raw).all()) and bool((final_weights[start:] > 0).all()):
        physical = final_weights.clone().requires_grad_(True)
        objective = evaluate(physical, "final")
        full_gradient = torch.autograd.grad(objective, physical)[0]
        free_gradient = full_gradient[start:]
        raw_grad = free_gradient * raw.sigmoid()
        if bool(torch.isfinite(objective)) and bool(torch.isfinite(raw_grad).all()):
            objective_value = objective.item()
            raw_gradient = raw_grad.tolist()
            physical_gradient = _finite_list(free_gradient)
        else:
            numerical_failure, error_detail = True, "non-finite final objective/gradient"
    else:
        numerical_failure, error_detail = True, "non-finite/underflowed final parameters"

    iterations = int(optimizer.state.get(raw, {}).get("n_iter", 0))
    failure = ("non-finite" if numerical_failure else classify_fit(
        objective_value, raw_gradient, final_weights[start:].tolist(), iterations,
        parameterization=parameterization))
    counts = [demos.trajectory_count for _, _, demos, _ in prepared]
    unfinished = [demos.unfinished_count for _, _, demos, _ in prepared]
    budgets = [penalty + demos.horizon * ((phi @ final_weights).abs().max().item()
               + math.log(3)) for _, phi, demos, penalty in prepared]
    with torch.no_grad():
        mean_nll = (_pooled_objective(prepared, final_weights, 0., parameterization).item()
                    if objective_value is not None else None)
    total_nll = mean_nll * sum(counts) if mean_nll is not None else None
    physical_norm = (max(abs(x) for x in physical_gradient)
                     if physical_gradient is not None and all(x is not None for x in physical_gradient)
                     else None)
    return {
        "parameterization": parameterization, "weights": _finite_list(final_weights),
        "scale": _finite_list(final_weights)[0],
        "cost_ratios": _finite_list(final_weights[1:] / final_weights[0]),
        "raw_parameters": _finite_list(raw), "initial_raw_parameters": initial_raw.tolist(),
        "objective": objective_value, "mean_nll": mean_nll, "total_nll": total_nll,
        "raw_gradient": raw_gradient,
        "raw_gradient_inf_norm": max(map(abs, raw_gradient)) if raw_gradient is not None else None,
        "physical_free_gradient": physical_gradient, "physical_gradient_inf_norm": physical_norm,
        "iterations": iterations, "closure_evaluations": evaluations,
        "success": failure is None, "failure_reason": failure, "failure_detail": error_detail,
        "numerical_domain_failure": False, "l2": l2, "max_iter": int(max_iter),
        "environment_count": len(prepared), "trajectory_count": sum(counts),
        "trajectory_counts": counts, "unfinished_count": sum(unfinished),
        "unfinished_counts": unfinished,
        "horizons": [demos.horizon for _, _, demos, _ in prepared],
        "terminal_penalties": [penalty for _, _, _, penalty in prepared],
        "score_budgets": [x if math.isfinite(x) else None for x in budgets],
    }
