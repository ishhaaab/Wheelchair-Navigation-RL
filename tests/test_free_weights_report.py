"""Reporting must finish even with unavailable diagnostics and zero/one good pair."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("successful_pairs", [0, 1])
def test_report_persists_failed_counts_and_undefined_statistics(tmp_path, monkeypatch, successful_pairs):
    script_path = Path(__file__).resolve().parents[1] / "scripts/report_free_weights.py"
    spec = importlib.util.spec_from_file_location("free_weights_report", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    experiment, baseline, output = tmp_path / "experiment", tmp_path / "baseline", tmp_path / "report"
    good = {"weights": [1., .7, 1.2], "success": True, "failure_reason": None,
            "score_budget": 111.56, "numerical_domain_failure": False,
            "raw_gradient_inf_norm": 1e-9, "iterations": 5, "l2": 0., "n": 32}
    failed = {**good, "weights": [None, None, None], "success": False,
              "failure_reason": "non-finite", "score_budget": None,
              "raw_gradient_inf_norm": None}
    old_cell = {"seed": 11, "n": 32, "identity": {"prefix_hash": "same"},
                "point_fit": good, "bootstrap": {"fits": []}}
    old = {"metadata": {"configuration": {"truth": [1., .7, 1.2],
            "sample_sizes": [32], "seeds": [11]}}, "cells": [old_cell], "regularization": [good]}
    new = deepcopy(old)
    new["cells"][0]["point_fit"] = good if successful_pairs else failed
    new["cells"][0]["bootstrap"]["fits"] = [failed]
    new["regularization"] = [good if successful_pairs else failed]
    new["summary"] = {"ratio_scale": {"by_sample_size": []}}
    new["preflight"] = {"numerical_budgets": {"upper_fit_corner": 673.36}}
    write(baseline / "recovery.json", old)
    write(experiment / "recovery/recovery.json", new)
    for path in (baseline / "recovery_demos.jsonl", experiment / "recovery/recovery_demos.jsonl"):
        path.write_text('{"seed":11}\n')
    write(experiment / "recon/baseline_hashes.json", {})
    write(experiment / "recon/population.json", {"rows": []})
    write(experiment / "regression_red/fixed_fit.json", good)
    write(experiment / "regression_green/fixed_fit.json", good)
    write(experiment / "regression_green/three_free_fit.json",
          {**good, "scale": 2., "cost_ratios": [.7, 1.2]})
    monkeypatch.setattr(sys, "argv", [str(script_path), "--experiment", str(experiment),
        "--baseline", str(baseline / "recovery.json"), "--output", str(output)])
    module.main()
    report = json.loads((output / "comparison.json").read_text())
    numeric = report["numerical_accounting"]
    assert numeric["unavailable_budgets"] == (1 if successful_pairs else 2)
    assert numeric["failures"]["non-finite"] == (1 if successful_pairs else 2)
    assert report["paired_summary"][0]["paired_count"] == successful_pairs
    assert report["paired_summary"][0]["sample_std_ratio_error_difference"] is None
    assert "undefined" in (output / "report.md").read_text()
    if not successful_pairs:
        assert numeric["max_fitted_budget"] is None
        assert not report["point_fitted_terminal_margins"][0]["evaluated"]
