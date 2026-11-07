"""Paired comparisons must match actual prefixes, retain failures, and use ratios."""

from copy import deepcopy

import pytest

from wnp.recovery_comparison import compare_recovery


def reports():
    config = {"truth": [1., .7, 1.2], "sample_sizes": [32], "seeds": [11, 29]}
    cells = [{"seed": seed, "n": 32, "identity": {"prefix_hash": str(seed)},
              "point_fit": {"success": True, "failure_reason": None,
                            "weights": [1., .8, 1.4]}}
             for seed in (11, 29)]
    old = {"metadata": {"configuration": config}, "cells": cells}
    new = deepcopy(old)
    new["cells"][0]["point_fit"]["weights"] = [2., 1.4, 2.4]
    new["cells"][1]["point_fit"]["weights"] = [1., .7, 1.3]
    return old, new


def test_comparison_uses_paired_ratio_errors_not_physical_weight_differences():
    old, new = reports()
    result = compare_recovery(old, new)
    first = result["paired_cells"][0]
    assert first["new_ratios"] == [.7, 1.2]
    assert first["ratio_absolute_error_difference"] == pytest.approx([-.1, -.2])
    assert first["scale_error_difference"] == 1.
    assert result["paired_summary"][0]["paired_count"] == 2
    assert result["paired_summary"][0]["mean_ratio_error_difference"] < 0
    assert result["paired_summary"][0]["sample_std_ratio_error_difference"] > 0


def test_comparison_refuses_different_prefixes_even_with_same_seed_and_count():
    old, new = reports()
    new["cells"][0]["identity"]["prefix_hash"] = "different"
    with pytest.raises(ValueError, match="prefix"):
        compare_recovery(old, new)


def test_failed_pairs_stay_in_records_and_out_of_error_summaries():
    old, new = reports()
    new["cells"][1]["point_fit"].update(success=False, failure_reason="boundary")
    result = compare_recovery(old, new)
    assert len(result["paired_cells"]) == 2
    assert result["paired_cells"][1]["new_failure_reason"] == "boundary"
    assert result["paired_summary"][0]["paired_count"] == 1
    assert result["paired_summary"][0]["excluded_pairs"] == 1
    assert result["paired_summary"][0]["sample_std_ratio_error_difference"] is None


def test_comparison_rejects_missing_cells_and_changed_truth():
    old, new = reports()
    new["cells"].pop()
    with pytest.raises(ValueError, match="cells"):
        compare_recovery(old, new)
    old, new = reports()
    new["metadata"]["configuration"]["truth"] = [2., 1.4, 2.4]
    with pytest.raises(ValueError, match="configuration"):
        compare_recovery(old, new)
