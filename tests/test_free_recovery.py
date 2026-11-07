"""Free-parameter preflight, immutable outputs, and paired ratio accounting."""

from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from wnp.recovery import RecoveryConfig, _fixture, preflight, run_recovery, summarize_recovery
from wnp.soft_dp import plan


def free_config(**changes):
    return replace(RecoveryConfig(), parameterization="three_free", **changes)


def test_preflight_uses_all_three_physical_directions_and_ratio_delta_method():
    config = free_config()
    env, features = _fixture(config)
    report = preflight(config)
    hessian = torch.autograd.functional.hessian(lambda w: plan(env, features, w,
        24, 30).values[0, env.start_state], torch.tensor(config.truth, dtype=torch.float64))
    assert report["free_feature_order"] == ["distance", "turning", "clearance"]
    assert report["covariance"] == pytest.approx(hessian.numpy())
    assert report["passed"]
    assert report["eigenvalues"][0] == pytest.approx(.5589457257)
    assert report["condition_number"] == pytest.approx(53.61531154)
    assert len(report["predicted_asymptotic_std_n2048"]) == 3
    assert len(report["ratio_scale_predicted_std_n2048"]) == 3
    assert report["numerical_budgets"]["upper_fit_corner"] < 1000


def test_doubled_truth_is_allowed_but_fails_unchanged_covariance_gate_before_labels(tmp_path):
    config = free_config(truth=(2., 1.4, 2.4))
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="preflight"):
        run_recovery(config, out)
    record = json.loads((out / "recovery_preflight.json").read_text())
    assert not record["diagnostic"]["gates"]["min_eigenvalue_at_least_0.1"]
    assert not (out / "recovery_demos.jsonl").exists()


def test_existing_three_free_output_refused_even_preflight_only(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        run_recovery(free_config(), tmp_path, preflight_only=True)
    assert sentinel.read_text() == "previous evidence"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_invalid_explicit_recovery_mode_and_legacy_truth_mismatch():
    with pytest.raises(ValueError):
        replace(RecoveryConfig(), parameterization="infer")
    with pytest.raises(ValueError):
        replace(RecoveryConfig(), truth=(2., 1.4, 2.4))


def test_small_free_run_reports_scale_ratios_and_preserves_all_three_intervals(tmp_path):
    config = free_config(seeds=(7,), sample_sizes=(4,), sensitivity_sizes=(4,),
                         bootstrap_refits=2)
    report = run_recovery(config, tmp_path / "new")
    cell = report["cells"][0]
    assert len(cell["point_fit"]["raw_parameters"]) == 3
    assert cell["bootstrap"]["refits"] == 2
    assert len(report["summary"]["by_sample_size"][0]["coverage"]) == 3
    assert "ratio_scale" in report["summary"]
    assert (tmp_path / "new" / "ratio_scale_curve.png").is_file()


@pytest.mark.parametrize("truth", [(100., 100., 100.), (1.7e308, 1.7e308, 1.7e308)])
def test_numerical_domain_conflict_records_failure_before_sampling(tmp_path, truth):
    config = free_config(truth=truth)
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="numerical domain|unrepresentable"):
        run_recovery(config, out)
    assert (out / "numerical_failure.json").is_file()
    assert not (out / "recovery_demos.jsonl").exists()


def test_failed_refit_halts_recovery_and_saves_partial_fit_evidence(tmp_path, monkeypatch):
    import wnp.recovery as recovery
    from wnp.irl import NumericalDomainError
    failed = {"numerical_domain_failure": True,
              "failure_detail": "outside supported numerical domain"}

    def failed_bootstrap(*args, **kwargs):
        raise NumericalDomainError(failed["failure_detail"], [failed])

    monkeypatch.setattr(recovery, "bootstrap_fit", failed_bootstrap)
    config = free_config(seeds=(7,), sample_sizes=(8,), sensitivity_sizes=(8,),
                         bootstrap_refits=2)
    out = tmp_path / "new"
    with pytest.raises(NumericalDomainError):
        run_recovery(config, out)
    record = json.loads((out / "numerical_failure.json").read_text())
    assert record["attempted_bootstrap_fits"] == [failed]
    assert record["point_fit"]["trajectory_count"] == 8
    assert not (out / "recovery.json").exists()


def test_mismatched_baseline_prefix_stops_before_fit(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(json.dumps({"seed": 7, "index": 0, "states": [],
                                     "actions": [], "arrival_step": None}) + "\n")
    config = free_config(seeds=(7,), sample_sizes=(4,), sensitivity_sizes=(4,),
                         bootstrap_refits=1)
    with pytest.raises(ValueError, match="baseline.*prefix"):
        run_recovery(config, tmp_path / "new", baseline_demos=baseline)
    assert not list((tmp_path / "new").glob("recovery_cells/*.json"))


def test_failed_fit_with_subnormal_scale_keeps_finite_serializable_metrics():
    from wnp.recovery import _fit_row, _json_text

    result = {"weights": [1e-320, 1., 1.], "success": False, "failure_reason": "boundary"}
    with np.errstate(over="raise", invalid="raise"):
        row = _fit_row(free_config(), 11, 32, result)
    assert row["weights"] == [1e-320, 1., 1.]
    assert row["failure_reason"] == "boundary"
    assert row["scale_ratio_coordinates"] == [1e-320, None, None]
    assert row["scale_ratio_absolute_error"][0] == 1.
    assert row["scale_ratio_absolute_error"][1:] == [None, None]
    assert row["ratio_euclidean_error"] is None
    json.loads(_json_text(row))


def test_failed_fit_with_huge_weights_retains_record_without_derived_overflow():
    from wnp.recovery import _fit_row, _json_text

    result = {"weights": [1.7e308, 1.7e308, 1.7e308], "success": False, "failure_reason": "boundary"}
    row = _fit_row(free_config(), 11, 32, result)
    assert row["weights"] == [1.7e308, 1.7e308, 1.7e308]
    assert row["euclidean_error"] is None
    json.loads(_json_text(row))
