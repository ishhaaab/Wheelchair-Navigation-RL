"""Scientific gates, transparent aggregation, and small end-to-end recovery runs."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from wnp.recovery import (
    RecoveryConfig, bootstrap_seed, preflight, run_recovery, summarize_recovery,
)


def test_default_recovery_configuration_matches_predeclared_protocol():
    config = RecoveryConfig()
    assert config.seeds == (11, 29, 47, 83, 101)
    assert config.sample_sizes == (32, 128, 512, 2048)
    assert config.truth == (1., .7, 1.2)
    assert config.bootstrap_refits == 100
    assert config.sensitivity_sizes == (32, 2048)
    assert config.l2_values == (0., .001, .01)
    assert len({bootstrap_seed(seed, n) for seed in config.seeds
                for n in config.sample_sizes}) == 20


def test_original_fixture_passes_exact_presampling_gates():
    report = preflight(RecoveryConfig())
    assert report["passed"]
    assert report["completion_probability"] >= .90
    assert report["eigenvalues"][0] >= .1
    assert report["condition_number"] <= 100
    assert report["terminal_margin"]["passed"]
    assert report["covariance_symmetry_error"] < 1e-10
    assert report["state_mass_error"] < 1e-10
    assert np.asarray(report["predicted_asymptotic_std_n2048"]).shape == (2,)


def test_degenerate_fixture_is_recorded_and_stops_before_any_demos(tmp_path):
    config = replace(RecoveryConfig(), goal=(2, 0))
    with pytest.raises(ValueError, match="preflight"):
        run_recovery(config, tmp_path)
    report = json.loads((tmp_path / "recovery_preflight.json").read_text())
    assert not report["diagnostic"]["passed"]
    assert not (tmp_path / "recovery_demos.jsonl").exists()
    assert not list(tmp_path.glob("recovery_cells/*.json"))


@pytest.mark.parametrize("changes", [
    {"sample_sizes": (32, 32)}, {"sample_sizes": (128, 32)},
    {"sample_sizes": (0, 32)}, {"seeds": (True,)}, {"seeds": (11, 11)},
    {"bootstrap_refits": 0}, {"truth": (2., .7, 1.2)},
    {"truth": (1., -.7, 1.2)}, {"sensitivity_sizes": (7,)},
    {"l2_values": (0., float("nan"))},
])
def test_invalid_recovery_configuration_raises(changes):
    with pytest.raises(ValueError):
        replace(RecoveryConfig(), **changes)


def _summary_cells():
    cells = []
    for n, scale in ((32, 1.), (128, .5)):
        for seed, sign in ((11, 1.), (29, -1.)):
            weights = [1., .7 + sign * .1 * scale, 1.2 + sign * .2 * scale]
            cells.append({"seed": seed, "n": n,
                "point_fit": {"weights": weights, "success": True, "failure_reason": None},
                "completion_fraction": 1.,
                "bootstrap": {"reliable": True, "successful_refits": 100, "failed_refits": 0,
                    "percentile_intervals": [[.6, .9], [1.1, 1.5]] if seed == 11
                        else [[.8, .9], [.7, 1.1]]}})
    return cells


def test_error_summary_seed_spread_slopes_and_descriptive_coverage():
    config = replace(RecoveryConfig(), seeds=(11, 29), sample_sizes=(32, 128),
                     sensitivity_sizes=(32, 128))
    report = summarize_recovery(config, _summary_cells(), preflight(config))
    first = report["by_sample_size"][0]
    assert first["successful_point_fits"] == 2
    assert first["mean_absolute_error"] == pytest.approx([.1, .2])
    assert first["mean_euclidean_error"] == pytest.approx(math.sqrt(.05))
    assert first["estimate_sample_std"] == pytest.approx([math.sqrt(.02), math.sqrt(.08)])
    assert first["coverage"] == [{"containing_truth": 1, "reliable_intervals": 2}] * 2
    assert report["log_log_slopes"]["euclidean"] == pytest.approx(-.5)
    assert report["log_log_slopes"]["turning_absolute"] == pytest.approx(-.5)
    assert "descriptive" in report["uncertainty_note"].lower()
    json.dumps(report, allow_nan=False)  # Real 100-refit intervals yield nonempty coverage counts.


def test_failed_point_fits_are_counted_not_silently_averaged():
    config = replace(RecoveryConfig(), seeds=(11, 29), sample_sizes=(32, 128),
                     sensitivity_sizes=(32, 128))
    cells = _summary_cells()
    cells[1]["point_fit"].update(success=False, failure_reason="boundary", weights=[1., 50., 50.])
    report = summarize_recovery(config, cells, preflight(config))
    first = report["by_sample_size"][0]
    assert first["failed_point_fits"] == 1
    assert first["failure_counts"] == {"boundary": 1}
    assert first["mean_absolute_error"] == pytest.approx([.1, .2])
    assert first["estimate_sample_std"] is None


def test_zero_mean_error_does_not_get_an_invented_log_offset():
    config = replace(RecoveryConfig(), seeds=(11, 29), sample_sizes=(32, 128),
                     sensitivity_sizes=(32, 128))
    cells = _summary_cells()
    for cell in cells:
        cell["point_fit"]["weights"] = list(config.truth)
    report = summarize_recovery(config, cells, preflight(config))
    assert all(value is None for value in report["log_log_slopes"].values())


def test_small_runner_preserves_nested_pools_failures_and_all_artifacts(tmp_path):
    config = replace(RecoveryConfig(), seeds=(7,), sample_sizes=(4, 8),
                     sensitivity_sizes=(4, 8), bootstrap_refits=2)
    report = run_recovery(config, tmp_path)
    pool = [json.loads(line) for line in (tmp_path / "recovery_demos.jsonl").read_text().splitlines()]
    assert len(pool) == 8
    assert [row["index"] for row in pool] == list(range(8))
    assert all(row["seed"] == 7 and "arrival_step" in row for row in pool)
    assert len(report["cells"]) == 2
    for cell in report["cells"]:
        assert cell["n"] in (4, 8)
        assert cell["point_fit"]["trajectory_count"] == cell["n"]
        assert cell["bootstrap"]["successful_refits"] + cell["bootstrap"]["failed_refits"] == 2
        assert not cell["bootstrap"]["reliable"]
        assert len(cell["bootstrap"]["fits"]) == 2
        assert cell["bootstrap_seed"] == bootstrap_seed(7, cell["n"])
    assert len(report["regularization"]) == 6
    assert (tmp_path / "recovery_curve.png").read_bytes().startswith(b"\x89PNG")
    for name in ("recovery.json", "recovery_fits.csv", "recovery_bootstrap.csv",
                 "recovery_summary.csv", "recovery_regularization.csv"):
        assert (tmp_path / name).is_file()
    again = run_recovery(config, tmp_path)
    assert again == report  # Verified checkpoints, same data, no statistical retries.


def test_resume_rejects_incompatible_experiment_metadata(tmp_path):
    config = replace(RecoveryConfig(), seeds=(7,), sample_sizes=(4,),
                     sensitivity_sizes=(4,), bootstrap_refits=1)
    run_recovery(config, tmp_path)
    incompatible = replace(config, truth=(1., .8, 1.2))
    with pytest.raises(ValueError, match="incompatible"):
        run_recovery(incompatible, tmp_path)
