"""Float64 probabilities conditional on two offered completed trajectories."""

import numpy as np
import torch

from wnp.demos import validate_trajectories
from wnp.env import _integer
from wnp.soft_dp import _validate_features, _validate_feature_horizon


def _tensor(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
            or value.device.type != 'cpu' or value.shape != shape
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f'{name} must be a finite CPU float64 tensor of shape {shape}')


def _log_probs(value):
    _tensor(value, (2,), 'log probabilities')
    if abs(torch.logsumexp(value, 0).item()) > 1e-10 or bool((value > 1e-10).any()):
        raise ValueError('invalid normalized log probabilities')


def route_features(env, features: torch.Tensor, pair) -> torch.Tensor:
    records, horizon = validate_trajectories(env, pair.routes)
    if (len(records) != 2 or pair.shortest_index not in (0,1)
            or any(r.arrival_step is None or r.states[0] != env.start_state for r in records)):
        raise ValueError('pair needs two completed routes from the configured start')
    _validate_features(env, features)
    _validate_feature_horizon(features, horizon)
    totals = torch.stack([features[list(r.states[:-1]), list(r.actions)].sum(0) for r in records])
    _tensor(totals, (2,3), 'route feature totals')
    return totals


def choice_log_probs(totals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    _tensor(totals, (2,3), 'route feature totals')
    _tensor(weights, (3,), 'weights')
    if bool((totals < 0).any()) or bool((weights < 0).any()):
        raise ValueError('cost features and weights must be nonnegative')
    scores = -(totals @ weights)
    _tensor(scores, (2,), 'scores')
    result = scores.log_softmax(0)
    _log_probs(result)
    return result


def predict_route(costs: torch.Tensor, shortest_index: int) -> int:
    _tensor(costs, (2,), 'costs')
    if not _integer(shortest_index) or shortest_index not in (0,1):
        raise ValueError('shortest_index must be 0 or 1')
    return int(shortest_index) if abs((costs[0]-costs[1]).item()) <= 1e-12 else int(costs.argmin())


def draw_choices(oracle_log_probs: torch.Tensor, layout_seed: int, count: int = 64) -> np.ndarray:
    _log_probs(oracle_log_probs)
    if not _integer(layout_seed) or layout_seed < 0 or not _integer(count) or count < 1:
        raise ValueError('seed must be nonnegative and count positive integers')
    rng = np.random.default_rng(np.random.SeedSequence([420001, int(layout_seed)]))
    return rng.choice(2, int(count), p=oracle_log_probs.exp().detach().numpy())


def choice_metrics(log_probs: torch.Tensor | None, prediction: int | None,
                   labels: np.ndarray, oracle_log_probs: torch.Tensor,
                   oracle_costs: torch.Tensor) -> dict:
    _log_probs(oracle_log_probs)
    _tensor(oracle_costs, (2,), 'oracle costs')
    if (not isinstance(labels,np.ndarray) or labels.ndim != 1 or not labels.size
            or labels.dtype.kind not in 'iu' or not np.isin(labels, [0,1]).all()):
        raise ValueError('labels must be a nonempty one-dimensional binary integer array')
    valid = _integer(prediction) and prediction in (0,1)
    metrics = {'sampled_accuracy': float(np.mean(labels == prediction)) if valid else 0.,
               'expected_accuracy': oracle_log_probs.exp()[prediction].item() if valid else 0.,
               'expected_regret': (oracle_costs[prediction]-oracle_costs.min()).item() if valid else None,
               'nll': None, 'brier': None, 'valid_prediction': bool(valid)}
    if log_probs is not None:
        _log_probs(log_probs)
        metrics.update(nll=-log_probs[torch.from_numpy(labels.astype(np.int64))].mean().item(),
                       brier=float(np.mean((log_probs[1].exp().item()-labels)**2)))
    return metrics
