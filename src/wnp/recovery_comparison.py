"""Paired descriptive comparisons of historical and three-free recovery fits."""

import numpy as np


def compare_recovery(old: dict, new: dict) -> dict:
    """Compare successful pairs only, refusing incompatible data and fixtures."""
    old_config = dict(old["metadata"]["configuration"])
    new_config = dict(new["metadata"]["configuration"])
    old_config.pop("parameterization", None)
    new_config.pop("parameterization", None)
    if old_config != new_config:
        raise ValueError("paired recovery configuration differs")
    index = lambda report: {(c["seed"], c["n"]): c for c in report["cells"]}
    a, b = index(old), index(new)
    if (a.keys() != b.keys() or len(a) != len(old["cells"])
            or len(b) != len(new["cells"])):
        raise ValueError("paired recovery cells must match without duplicates")
    truth = np.asarray(old_config["truth"], dtype=float)
    target = truth[1:] / truth[0]
    rows = []
    for seed, n in sorted(a, key=lambda key: (key[1], key[0])):
        left, right = a[seed, n], b[seed, n]
        if left["identity"]["prefix_hash"] != right["identity"]["prefix_hash"]:
            raise ValueError(f"paired prefix mismatch for seed={seed}, n={n}")
        old_fit, new_fit = left["point_fit"], right["point_fit"]
        row = {"seed": seed, "n": n, "prefix_hash": left["identity"]["prefix_hash"],
            "paired_success": bool(old_fit["success"] and new_fit["success"]),
            "old_failure_reason": old_fit["failure_reason"],
            "new_failure_reason": new_fit["failure_reason"]}
        if row["paired_success"]:
            ow, nw = np.asarray(old_fit["weights"]), np.asarray(new_fit["weights"])
            orat, nrat = ow[1:]/ow[0], nw[1:]/nw[0]
            oe, ne = np.abs(orat-target), np.abs(nrat-target)
            row.update(old_scale=float(ow[0]), new_scale=float(nw[0]),
                old_ratios=orat.tolist(), new_ratios=nrat.tolist(),
                ratio_estimate_difference=(nrat-orat).tolist(),
                old_ratio_absolute_error=oe.tolist(), new_ratio_absolute_error=ne.tolist(),
                ratio_absolute_error_difference=(ne-oe).tolist(),
                old_ratio_euclidean_error=float(np.linalg.norm(oe)),
                new_ratio_euclidean_error=float(np.linalg.norm(ne)),
                ratio_error_difference=float(np.linalg.norm(ne)-np.linalg.norm(oe)),
                scale_error_difference=float(abs(nw[0]-truth[0])-abs(ow[0]-truth[0])))
        rows.append(row)
    summary = []
    for n in old_config["sample_sizes"]:
        selected = [r for r in rows if r["n"] == n]
        good = [r for r in selected if r["paired_success"]]
        row = {"n": n, "paired_count": len(good), "excluded_pairs": len(selected)-len(good)}
        for name in ("ratio_error_difference", "scale_error_difference",
                     "ratio_estimate_difference", "ratio_absolute_error_difference"):
            data = np.asarray([r[name] for r in good])
            row["mean_" + name] = data.mean(axis=0).tolist() if good else None
            row["sample_std_" + name] = data.std(axis=0, ddof=1).tolist() if len(good)>1 else None
        summary.append(row)
    return {"difference_convention": "three_free minus fixed_distance, paired within seed and n",
            "paired_cells": rows, "paired_summary": summary,
            "interpretation": "Scale 1 is well specified in both models. These data assess the "
                "extra-parameter variance cost, not scale-misspecification bias. Five paired seeds "
                "give descriptive comparisons, not a superiority or scaling-law claim.",
            "regularization_disclosure": "Three-free L2 penalizes all three physical weights, "
                "including scale; old L2 penalizes only turning and clearance. Positive-lambda "
                "sensitivities are not like-for-like."}
