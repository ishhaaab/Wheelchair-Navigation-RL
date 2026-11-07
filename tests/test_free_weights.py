"""Three-free inference must learn scale, preserve derivatives, and count failures."""

import math

import numpy as np
import pytest
import torch

from wnp.demos import enumerate_trajectories
from wnp.env import GridMDP
from wnp.features import feature_table
from wnp.irl import bootstrap_fit, classify_fit, fit, loss, physical_weights, summarize_bootstrap
from wnp.soft_dp import trajectory_log_prob


def tiny_data():
    env = GridMDP(np.zeros((2, 2), dtype=bool), (0, 0, 0), (0, 1), headings=4)
    return env, feature_table(env), enumerate_trajectories(env, env.start_state, 4)


def test_three_softpluses_preserve_scale_and_all_raw_derivatives():
    raw = torch.tensor([-.2, .3, .7], dtype=torch.float64, requires_grad=True)
    weights = physical_weights(raw, parameterization="three_free")
    torch.testing.assert_close(weights, torch.log1p(raw.exp()))
    torch.testing.assert_close(torch.autograd.grad(weights.sum(), raw)[0], raw.sigmoid())
    assert weights[0] != 1


@pytest.mark.parametrize("raw, mode", [
    (torch.zeros(2, dtype=torch.float64), "three_free"),
    (torch.zeros(3), "three_free"),
    (torch.tensor([-1000., 0., 0.], dtype=torch.float64), "three_free"),
    (torch.tensor([0., float('nan'), 0.], dtype=torch.float64), "three_free"),
    (torch.zeros(3, dtype=torch.float64), "implicit"),
])
def test_explicit_mode_rejects_invalid_raw_inputs(raw, mode):
    with pytest.raises(ValueError):
        physical_weights(raw, parameterization=mode)


def test_three_free_regularization_penalizes_distance_and_gradchecks():
    env, features, records = tiny_data()
    raw = torch.tensor([.1, -.2, .3], dtype=torch.float64, requires_grad=True)
    weights = torch.nn.functional.softplus(raw)
    expected = -trajectory_log_prob(env, features, weights, records).mean()
    actual = loss(env, features, records, raw, l2=.2, parameterization="three_free")
    torch.testing.assert_close(actual, expected + .1 * weights.square().sum())
    assert torch.autograd.gradcheck(lambda x: loss(env, features, records, x,
        l2=.2, parameterization="three_free"), (raw,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_all_three_fit_gates_apply_including_scale():
    assert classify_fit(1., [0., 0., 0.], [1., 1., 1.], 1,
                        parameterization="three_free") is None
    assert classify_fit(1., [0., 0., 0.], [10.01, 1., 1.], 1,
                        parameterization="three_free") == "boundary"
    assert classify_fit(1., [1.01e-6, 0., 0.], [1., 1., 1.], 1,
                        parameterization="three_free") == "unconverged"


def test_three_free_fit_reports_actual_gradients_ratios_and_budget():
    env, features, records = tiny_data()
    result = fit(env, features, records, l2=.01, parameterization="three_free")
    raw = torch.tensor(result["raw_parameters"], dtype=torch.float64, requires_grad=True)
    actual = loss(env, features, records, raw, l2=.01, parameterization="three_free")
    gradient = torch.autograd.grad(actual, raw)[0]
    assert len(raw) == 3
    assert result["objective"] == pytest.approx(actual.item(), abs=1e-12)
    assert result["raw_gradient"] == pytest.approx(gradient.tolist(), abs=1e-12)
    assert result["scale"] == result["weights"][0]
    assert result["cost_ratios"] == pytest.approx(
        np.asarray(result["weights"])[1:] / result["weights"][0])
    expected_budget = 30 + 4 * ((features @ torch.tensor(result["weights"],
        dtype=torch.float64)).abs().max().item() + math.log(3))
    assert result["score_budget"] == pytest.approx(expected_budget)


def test_bootstrap_transforms_each_success_before_percentiles_and_excludes_failures():
    fits = [{"weights": [2., 1., 6.], "success": True},
            {"weights": [4., 8., 4.], "success": True},
            {"weights": [1., 50., 50.], "success": False, "failure_reason": "boundary"}]
    summary = summarize_bootstrap(fits, parameterization="three_free")
    assert summary["percentile_intervals"] == pytest.approx(
        np.percentile([[2., 1., 6.], [4., 8., 4.]], [2.5, 97.5], axis=0).T)
    assert summary["ratio_percentile_intervals"] == pytest.approx(
        np.percentile([[.5, 3.], [2., 1.]], [2.5, 97.5], axis=0).T)
    assert summary["scale_percentile_interval"] == pytest.approx([2.05, 3.95])
    assert summary["failed_refits"] == 1
    assert not summary["reliable"]


def test_three_free_bootstrap_preserves_mode_and_repeats_resampling():
    env, features, records = tiny_data()
    first = bootstrap_fit(env, features, records, seed=31, refits=2, l2=.01,
                          parameterization="three_free")
    assert first == bootstrap_fit(env, features, records, seed=31, refits=2, l2=.01,
                                  parameterization="three_free")
    assert all(len(r["raw_parameters"]) == 3 for r in first["fits"])


def test_domain_failure_stops_bootstrap_with_failed_fit_attached(monkeypatch):
    import wnp.irl as irl
    env, features, records = tiny_data()
    # Inject the rare optimizer line-search failure at the fit boundary.
    failed = {"success": False, "failure_reason": "non-finite",
              "failure_detail": "outside supported numerical domain",
              "numerical_domain_failure": True, "weights": [1000., 1000., 1000.]}
    monkeypatch.setattr(irl, "fit", lambda *a, **kw: failed.copy())
    with pytest.raises(irl.NumericalDomainError) as error:
        bootstrap_fit(env, features, records, seed=31, refits=2,
                      parameterization="three_free")
    assert len(error.value.fits) == 1
    assert error.value.fits[0]["numerical_domain_failure"]


@pytest.mark.parametrize("detail", [
    "features and weights produce unrepresentable rewards",
    "weights and horizon produce unrepresentable legal scores",
    "cumulative features and weight derivatives may exceed float64 range",
])
def test_closure_range_violation_aborts_bootstrap_even_when_endpoint_restored(monkeypatch, detail):
    import wnp.irl as irl
    env, features, records = tiny_data()
    real_plan = irl.plan
    trial = False

    def planner(*args, **kwargs):
        if trial:
            raise ValueError(detail)
        return real_plan(*args, **kwargs)

    def step(optimizer, closure):
        nonlocal trial
        trial = True
        try:
            closure()
        finally:
            trial = False  # Final endpoint diagnostics are supported again.

    monkeypatch.setattr(irl, "plan", planner)
    monkeypatch.setattr(torch.optim.LBFGS, "step", step)
    with pytest.raises(irl.NumericalDomainError) as error:
        bootstrap_fit(env, features, records, seed=31, refits=2,
                      parameterization="three_free")
    result = error.value.fits[0]
    assert result["failure_detail"] == detail
    assert result["objective"] is not None
    assert result["numerical_domain_failure"]
