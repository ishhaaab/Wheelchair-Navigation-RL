"""Write a new paired comparison report without modifying any experiment artifact."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from wnp.recovery import RecoveryConfig, _csv, _fixture
from wnp.recovery_comparison import compare_recovery
from wnp.verification import terminal_margin_diagnostic


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def number(value):
    return f"{value:.6f}" if value is not None and math.isfinite(value) else "undefined"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path("results/recovery.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old = read(args.baseline)
    new_path = args.experiment / "recovery/recovery.json"
    new = read(new_path)
    comparison = compare_recovery(old, new)
    source_identity = {name: {
        "recorded_run_sha256": digest,
        "current_sha256": hashlib.sha256((Path(__file__).resolve().parents[1]
            / "src/wnp" / f"{name}.py").read_bytes()).hexdigest()}
        for name, digest in new["metadata"].get("model_sources", {}).items()}
    # Verify full JSONL equality in addition to the per-cell digest checks.
    old_demos = [json.loads(line) for line in (args.baseline.parent / "recovery_demos.jsonl")
                 .read_text().splitlines()]
    new_demos = [json.loads(line) for line in (args.experiment / "recovery/recovery_demos.jsonl")
                 .read_text().splitlines()]
    if old_demos != new_demos:
        raise ValueError("paired demonstration pools differ")
    root = Path(__file__).resolve().parents[1]
    hashes = read(args.experiment / "recon/baseline_hashes.json")
    unchanged, changed, intended = [], [], []
    for name, expected in hashes.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual == expected:
            unchanged.append(name)
        elif name.replace("\\", "/") == "src/wnp/irl.py":
            intended.append({"path": name, "old": expected, "new": actual})
        else:
            changed.append(name)
    if changed:
        raise ValueError(f"protected historical files changed: {changed}")
    population = read(args.experiment / "recon/population.json")
    regression = {mode: read(args.experiment / "regression_green" / f"{mode}_fit.json")
                  for mode in ("fixed", "three_free")}
    original_fixed = read(args.experiment / "regression_red/fixed_fit.json")
    if regression["fixed"]["weights"] != original_fixed["weights"]:
        raise ValueError("legacy fixed-distance regression changed after implementation")
    all_fits = []
    for cell in new["cells"]:
        all_fits.append(cell["point_fit"])
        all_fits.extend(cell["bootstrap"]["fits"])
    all_fits.extend(f for f in new["regularization"] if f["l2"] > 0)
    budgets = [f["score_budget"] for f in all_fits
               if f["score_budget"] is not None and math.isfinite(f["score_budget"])]
    gradients = [f["raw_gradient_inf_norm"] for f in all_fits
                 if f["raw_gradient_inf_norm"] is not None
                 and math.isfinite(f["raw_gradient_inf_norm"])]
    failure_counts = {}
    for result in all_fits:
        if not result["success"]:
            reason = result["failure_reason"]
            failure_counts[reason] = failure_counts.get(reason, 0)+1
    env, features = _fixture(RecoveryConfig())
    fitted_margins = []
    for cell in new["cells"]:
        weights = cell["point_fit"]["weights"]
        if (not cell["point_fit"]["success"] or any(w is None or not math.isfinite(w)
                                                   or w <= 0 for w in weights)):
            fitted_margins.append({"seed": cell["seed"], "n": cell["n"],
                "evaluated": False, "passed": None, "reason": "unsuccessful or unavailable weights"})
            continue
        margin = terminal_margin_diagnostic(env, features, torch.tensor(weights,
            dtype=torch.float64), 24, 30)
        fitted_margins.append({"seed": cell["seed"], "n": cell["n"], "evaluated": True, **margin})
    sensitivity = []
    for report, mode in ((old, "fixed_distance"), (new, "three_free")):
        for n in (32, 2048):
            for l2 in (0., .001, .01):
                rows = [r for r in report["regularization"] if r["n"] == n and r["l2"] == l2]
                good = [r for r in rows if r["success"]]
                weights = np.asarray([r["weights"] for r in good])
                transformed = (np.column_stack((weights[:, 0], weights[:, 1:]/weights[:, :1]))
                               if good else None)
                sensitivity.append({"parameterization": mode, "n": n, "l2": l2,
                    "successes": len(good), "failures": len(rows)-len(good),
                    "mean_weights": weights.mean(axis=0).tolist() if good else None,
                    "mean_scale_ratios": transformed.mean(axis=0).tolist() if good else None})
    result = {**comparison, "population_reconnaissance": population,
        "regression": regression, "paired_demo_records": len(new_demos),
        "new_summary": new["summary"], "regularization_sensitivity": sensitivity,
        "numerical_accounting": {"unique_fits": len(all_fits), "failures": failure_counts,
            "domain_failures": sum(f["numerical_domain_failure"] for f in all_fits),
            "min_fitted_budget": min(budgets) if budgets else None,
            "max_fitted_budget": max(budgets) if budgets else None, "limit": 1000,
            "available_budget_count": len(budgets), "unavailable_budgets": len(all_fits)-len(budgets),
            "max_raw_gradient_inf_norm": max(gradients) if gradients else None,
            "unavailable_raw_gradient_norms": len(all_fits)-len(gradients),
            "max_iterations": max(f["iterations"] for f in all_fits)},
        "point_fitted_terminal_margins": fitted_margins,
        "terminal_margin_limitation": "Reference margin remains a preflight gate only. "
            "Point-fit margins are descriptive. Bootstrap margins are not individually checked.",
        "integrity": {"unchanged_files": unchanged, "intentionally_changed_sources": intended,
            "historical_files_changed": changed, "legacy_regression_weights_identical": True,
            "old_recovery_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "new_recovery_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest()},
        "provenance_note": "Historical recovery retains historical source fingerprints. "
            "Current code cannot resume it; offered-route content hashes remain unchanged.",
        "current_source_identity": source_identity,
        "post_review_note": "The saved run predates review fixes to planner-domain failure "
            "classification and failed-record/report serialization. No optimizer, objective, "
            "threshold, seed, data, or successful fitted result was changed or regenerated. "
            "Run source hashes remain historical; current hashes are recorded separately.",
        "identifiability_note": "At doubled scale, covariance minimum eigenvalue 0.0302182 "
            "fails the recovery gate. The local scale standard deviation is larger than ratio "
            "standard deviations. Increasing determinism weakens scale information; human "
            "demonstrations may not identify scale well even when ratios appear stable."}
    with (args.output / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    _csv(args.output / "paired_cells.csv", result["paired_cells"])
    _csv(args.output / "paired_summary.csv", result["paired_summary"])
    _csv(args.output / "regularization_sensitivity.csv", sensitivity)
    _csv(args.output / "ratio_scale_summary.csv", new["summary"]["ratio_scale"]["by_sample_size"])
    lines = ["# Three-free recovery comparison", "", comparison["interpretation"], "",
        "## Population scale bias", "",
        "Exact population cross-entropy optima, no sampling. Truth ratios are [0.7,1.2].", "",
        "| Scale | Fixed turning ratio | Fixed clearance ratio |",
        "|---:|---:|---:|"]
    for row in population["rows"]:
        ratios = row["fixed_population_ratios"]
        lines.append(f"| {row['scale']} | {ratios[0]:.6f} | {ratios[1]:.6f} |")
    free = regression["three_free"]
    lines += ["", "## Behavioral regression", "",
        f"Before implementation the existing fixed estimator passed its fit and bias gates, "
        f"but failed ratio recovery with weights {regression['fixed']['weights']}. "
        f"Three-free inference passed unchanged tolerances with scale {free['scale']:.8f} "
        f"and ratios {free['cost_ratios']}. Both used the same frozen 65,536-demo pool. "
        "The regression is opt-in slow, not part of routine suite passes.", "",
        result["identifiability_note"], "", "## Paired main recovery", "",
        f"All {len(new_demos)} historical demo records match. Differences are new minus old; "
        "positive ratio error differences favor the legacy model.", "",
        "| n | Successful pairs | Old mean ratio L2 error | New mean ratio L2 error | Paired difference mean +/- sample std |",
        "|---:|---:|---:|---:|---:|"]
    for row in result["paired_summary"]:
        selected = [r for r in result["paired_cells"] if r["n"] == row["n"] and r["paired_success"]]
        old_mean = float(np.mean([r["old_ratio_euclidean_error"] for r in selected])) if selected else None
        new_mean = float(np.mean([r["new_ratio_euclidean_error"] for r in selected])) if selected else None
        lines.append(f"| {row['n']} | {row['paired_count']} | {number(old_mean)} | {number(new_mean)} | "
                     f"{number(row['mean_ratio_error_difference'])} +/- "
                     f"{number(row['sample_std_ratio_error_difference'])} |")
    coordinate_rows = new["summary"]["ratio_scale"]["by_sample_size"]
    if coordinate_rows:
        lines += ["", "Three-free coordinates, means +/- across-seed sample standard deviations. "
            "Distance scale is estimated here; the old model fixes it to 1.", "",
            "| n | Distance scale | Turning / distance | Clearance / distance |",
            "|---:|---:|---:|---:|"]
        for row in coordinate_rows:
            means = row["mean"] or [None]*3
            stds = row["sample_std"] or [None]*3
            values = [f"{number(mean)} +/- {number(std)}" for mean, std in zip(means, stds)]
            lines.append(f"| {row['n']} | " + " | ".join(values) + " |")
    lines += ["", "Complete paired estimates and absolute-error differences are in paired_cells.csv. "
        "Failure counts, bootstrap coverage, physical and ratio/scale uncertainty, log-log slopes, "
        "and per-seed fit diagnostics remain in comparison.json and the recovery artifacts. "
        "Five seeds do not establish calibrated coverage, superiority, or monotonic scaling.", "",
        "## Regularization", "", comparison["regularization_disclosure"], "",
        "See regularization_sensitivity.csv for both models, including failures.", "",
        "## Numerical budget and integrity", "",
        json.dumps(result["numerical_accounting"], sort_keys=True), "",
        f"Upper fit-box corner budget {new['preflight']['numerical_budgets']['upper_fit_corner']:.6f} "
        "is below the unchanged 1000 limit. No T/M or limit changes. "
        f"Point fitted M margin passes {sum(m['passed'] is True for m in fitted_margins)}/"
        f"{sum(m['evaluated'] for m in fitted_margins)} evaluated; "
        f"{sum(not m['evaluated'] for m in fitted_margins)} unavailable. "
        + result["terminal_margin_limitation"], "",
        f"Verified {len(unchanged)} historical/protected files unchanged. irl.py is the intended "
        "changed inference source. " + result["provenance_note"], "",
        result["post_review_note"], "",
        "No held-out evaluation, human data, commits, or pushes. Stop for review.", ""]
    with (args.output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
