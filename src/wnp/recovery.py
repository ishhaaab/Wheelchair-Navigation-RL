"""Gated, reproducible tiny synthetic recovery with transparent fit accounting."""

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
import numpy as np
import torch

from wnp.demos import sample
from wnp.env import GridMDP, _integer
from wnp.features import feature_table
from wnp.irl import bootstrap_fit, fit
from wnp.soft_dp import expected_features, plan
from wnp.verification import terminal_margin_diagnostic


@dataclass(frozen=True)
class RecoveryConfig:
    shape: tuple[int, int] = (5, 5)
    occupied: tuple[tuple[int, int], ...] = ((1, 2), (2, 2))
    start: tuple[int, int, int] = (2, 0, 0)
    goal: tuple[int, int] = (2, 4)
    headings: int = 4
    d_safe: float = 2.0
    horizon: int = 24
    terminal_penalty: float = 30.0
    truth: tuple[float, float, float] = (1., .7, 1.2)
    seeds: tuple[int, ...] = (11, 29, 47, 83, 101)
    sample_sizes: tuple[int, ...] = (32, 128, 512, 2048)
    bootstrap_refits: int = 100
    sensitivity_sizes: tuple[int, ...] = (32, 2048)
    l2_values: tuple[float, ...] = (0., .001, .01)
    torch_threads: int = 1

    def __post_init__(self):
        if (len(self.shape) != 2 or any(not _integer(x) or x < 1 for x in self.shape)):
            raise ValueError("shape must contain two positive integers")
        if (not self.seeds or any(not _integer(x) or x < 0 for x in self.seeds)
                or len(set(self.seeds)) != len(self.seeds)):
            raise ValueError("seeds must be distinct nonnegative integers")
        if (not self.sample_sizes or any(not _integer(x) or x < 1 for x in self.sample_sizes)
                or tuple(sorted(set(self.sample_sizes))) != self.sample_sizes):
            raise ValueError("sample_sizes must be strictly increasing positive integers")
        if (not _integer(self.bootstrap_refits) or self.bootstrap_refits < 1
                or not _integer(self.torch_threads) or self.torch_threads < 1):
            raise ValueError("bootstrap_refits and torch_threads must be positive integers")
        if (len(self.truth) != 3 or self.truth[0] != 1
                or any(not math.isfinite(x) or x <= 0 for x in self.truth)):
            raise ValueError("truth must fix distance=1 and have positive finite weights")
        if (not self.l2_values or self.l2_values[0] != 0
                or any(not math.isfinite(x) or x < 0 for x in self.l2_values)
                or len(set(self.l2_values)) != len(self.l2_values)):
            raise ValueError("l2_values must start with 0 and be distinct finite nonnegative values")
        if not set(self.sensitivity_sizes).issubset(self.sample_sizes):
            raise ValueError("sensitivity_sizes must be in sample_sizes")
        if not _integer(self.horizon) or self.horizon < 0:
            raise ValueError("horizon must be a nonnegative integer")
        if not math.isfinite(self.terminal_penalty) or self.terminal_penalty < 0:
            raise ValueError("terminal_penalty must be finite and nonnegative")
        if not math.isfinite(self.d_safe) or self.d_safe < 0:
            raise ValueError("d_safe must be finite and nonnegative")
        _fixture(self)  # Validate occupied coordinates and start/goal conventions.


def _fixture(config):
    occupied = np.zeros(config.shape, dtype=bool)
    for cell in config.occupied:
        if (len(cell) != 2 or any(not _integer(x) or not 0 <= x < size
                                  for x, size in zip(cell, config.shape))):
            raise ValueError("occupied coordinates must be inside shape")
        occupied[cell] = True
    env = GridMDP(occupied, config.start, config.goal, config.headings)
    return env, feature_table(env, config.d_safe)


def bootstrap_seed(seed: int, n: int) -> int:
    return 73001 + 1000003 * int(seed) + int(n)


def preflight(config: RecoveryConfig) -> dict:
    """Analytic diagnostics only: no demonstration sampling or fitting."""
    env, features = _fixture(config)
    weights = torch.tensor(config.truth, dtype=torch.float64)
    free_value = lambda free: plan(env, features, torch.cat((free.new_ones(1), free)),
        config.horizon, config.terminal_penalty).values[0, env.start_state]
    covariance = torch.autograd.functional.hessian(free_value, weights[1:])
    symmetric = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    positive = bool((eigenvalues > 0).all())
    condition = (eigenvalues[-1] / eigenvalues[0]).item() if positive else None
    with torch.no_grad():
        solution = plan(env, features, weights, config.horizon, config.terminal_penalty)
        distribution = torch.zeros(len(env.states), dtype=torch.float64)
        distribution[env.start_state] = 1
        for t in range(config.horizon):
            mass = distribution[:, None] * solution.log_policy[t].exp()
            distribution = torch.zeros_like(distribution).scatter_add(
                0, env.next_state.reshape(-1), mass.reshape(-1))
        completion = distribution[env.is_goal].sum().item()
        expected = expected_features(env, features, solution, [env.start_state]).tolist()
    margin = terminal_margin_diagnostic(env, features, weights, config.horizon,
                                         config.terminal_penalty)
    gates = {"completion_at_least_0.90": completion >= .90,
             "min_eigenvalue_at_least_0.1": eigenvalues[0].item() >= .1,
             "condition_at_most_100": condition is not None and condition <= 100,
             "terminal_penalty_margin": margin["passed"]}
    predicted = (torch.linalg.inv(symmetric).diagonal().div(2048).sqrt().tolist()
                 if positive else None)
    correlation = (symmetric[0, 1] / (symmetric[0, 0] * symmetric[1, 1]).sqrt()).item() \
        if positive else None
    return {"passed": all(gates.values()), "gates": gates,
        "completion_probability": completion, "state_mass_error": abs(distribution.sum().item() - 1),
        "free_feature_order": ["turning", "clearance"], "covariance": covariance.tolist(),
        "covariance_symmetry_error": (covariance - covariance.T).abs().max().item(),
        "eigenvalues": eigenvalues.tolist(), "eigenvectors": eigenvectors.tolist(),
        "eigenvector_convention": "Columns for ascending eigenvalues of (H+H.T)/2",
        "condition_number": condition, "feature_correlation": correlation,
        "predicted_asymptotic_std_n2048": predicted, "expected_features": expected,
        "terminal_margin": margin}


def _json_text(value):
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"cannot JSON encode {type(item)}")
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=encode) + "\n"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json_text(value), encoding="utf-8")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(_json_text(value).encode()).hexdigest()


def _metadata(config):
    root = Path(__file__).resolve().parent
    sources = {name: hashlib.sha256((root / f"{name}.py").read_bytes()).hexdigest()
               for name in ("env", "features", "soft_dp", "demos", "irl")}
    metadata = {"protocol_version": 1, "configuration": asdict(config), "model_sources": sources,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    metadata = json.loads(_json_text(metadata))
    return {**metadata, "fingerprint": _digest(metadata)}


def _mean_std(values):
    if not values:
        return None, None
    data = np.asarray(values, dtype=np.float64)
    return data.mean(axis=0).tolist(), data.std(axis=0, ddof=1).tolist() if len(values) > 1 else None


def summarize_recovery(config: RecoveryConfig, cells: list[dict], diagnostic: dict) -> dict:
    """Summaries condition on success; bootstrap and independent seeds stay distinct."""
    truth = np.asarray(config.truth[1:])
    rows = []
    for n in config.sample_sizes:
        selected = [cell for cell in cells if cell["n"] == n]
        successful = [cell for cell in selected if cell["point_fit"]["success"]]
        estimates = [cell["point_fit"]["weights"][1:] for cell in successful]
        absolute = [np.abs(np.asarray(w) - truth) for w in estimates]
        errors = [float(np.linalg.norm(error)) for error in absolute]
        mean_abs, std_abs = _mean_std(absolute)
        mean_l2, std_l2 = _mean_std(errors)
        mean_weights, std_weights = _mean_std(estimates)
        failures = {}
        for cell in selected:
            if not cell["point_fit"]["success"]:
                reason = cell["point_fit"]["failure_reason"]
                failures[reason] = failures.get(reason, 0) + 1
        coverage = []
        for j in range(2):
            intervals = [cell["bootstrap"]["percentile_intervals"][j] for cell in selected
                         if cell["bootstrap"]["reliable"]]
            coverage.append({"containing_truth": int(sum(lo <= truth[j] <= hi for lo, hi in intervals)),
                             "reliable_intervals": len(intervals)})
        rows.append({"n": n, "dataset_count": len(selected),
            "successful_point_fits": len(successful), "failed_point_fits": len(selected) - len(successful),
            "failure_counts": failures, "mean_weights": mean_weights, "estimate_sample_std": std_weights,
            "mean_absolute_error": mean_abs, "absolute_error_sample_std": std_abs,
            "mean_euclidean_error": mean_l2, "euclidean_error_sample_std": std_l2,
            "mean_completion_fraction": float(np.mean([c["completion_fraction"] for c in selected]))
                if selected else None, "coverage": coverage})
    slopes = {}
    for name, values in (
        ("turning_absolute", [r["mean_absolute_error"][0] if r["mean_absolute_error"] else None for r in rows]),
        ("clearance_absolute", [r["mean_absolute_error"][1] if r["mean_absolute_error"] else None for r in rows]),
        ("euclidean", [r["mean_euclidean_error"] for r in rows]),
    ):
        slopes[name] = (float(np.polyfit(np.log(config.sample_sizes), np.log(values), 1)[0])
            if len(values) > 1 and all(x is not None and x > 0 for x in values) else None)
    at_2048 = next((r for r in rows if r["n"] == 2048), None)
    return {"by_sample_size": rows, "log_log_slopes": slopes,
        "asymptotic_comparison_n2048": {"predicted_std": diagnostic["predicted_asymptotic_std_n2048"],
            "across_seed_sample_std": at_2048["estimate_sample_std"] if at_2048 else None},
        "uncertainty_note": "Bootstrap intervals describe each dataset; across-seed sample std describes "
            "independent datasets. Coverage and asymptotic comparisons are descriptive, not pass/fail.",
        "slope_caveat": f"Only {len(config.seeds)} independent seeds; noisy descriptive scaling evidence. "
            "Slopes are undefined if any mean error is zero or missing.",
        "error_reduction": "Across-seed means and sample std conditional on successful point fits."}


def _csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _fit_row(config, seed, n, fit_result, **extra):
    weights = fit_result["weights"]
    finite = all(w is not None and math.isfinite(w) for w in weights)
    error = (np.abs(np.asarray(weights[1:]) - config.truth[1:]).tolist() if finite else None)
    return {"seed": seed, "n": n, **fit_result, "absolute_error": error,
            "euclidean_error": float(np.linalg.norm(error)) if error else None, **extra}


def _plot(output, config, cells, summary):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for i, seed in enumerate(config.seeds):
        selected = sorted([c for c in cells if c["seed"] == seed], key=lambda c: c["n"])
        good = [c for c in selected if c["point_fit"]["success"]]
        color = colors(i % 10)
        for j in range(2):
            axes[j].plot([c["n"] for c in good], [c["point_fit"]["weights"][j+1] for c in good],
                         "o-", color=color, linewidth=1, markersize=4, label=f"seed {seed}")
            for cell in selected:
                intervals = cell["bootstrap"]["percentile_intervals"]
                if intervals is not None:
                    lo, hi = intervals[j]
                    x = cell["n"] * math.exp((i - (len(config.seeds)-1)/2) * .025)
                    axes[j].vlines(x, lo, hi, color=color, alpha=.55,
                                   linestyles="solid" if cell["bootstrap"]["reliable"] else "dashed")
        axes[2].plot([c["n"] for c in good], [np.linalg.norm(
            np.asarray(c["point_fit"]["weights"][1:]) - config.truth[1:]) for c in good],
            "o-", color=color, linewidth=.8, alpha=.6, markersize=3)
    for j, name in enumerate(("Turning weight", "Clearance weight")):
        axes[j].axhline(config.truth[j+1], color="black", linestyle="--", linewidth=1, label="truth")
        axes[j].set_ylabel(name)
        axes[j].set_title("Per-dataset 95% bootstrap intervals")
    valid = [r for r in summary["by_sample_size"] if r["mean_euclidean_error"] is not None]
    axes[2].plot([r["n"] for r in valid], [r["mean_euclidean_error"] for r in valid],
                 "ko-", linewidth=2, label="across-seed mean")
    axes[2].set_ylabel("Euclidean free-weight error")
    axes[2].set_title("Independent dataset variability")
    if valid and all(r["mean_euclidean_error"] > 0 for r in valid):
        axes[2].set_yscale("log")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xticks(config.sample_sizes, labels=[str(n) for n in config.sample_sizes])
        axis.set_xlabel("Demonstrations, nested prefixes")
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle("Tiny synthetic recovery; unfinished samples retained", fontsize=13)
    fig.savefig(output / "recovery_curve.png", dpi=160)
    plt.close(fig)


def run_recovery(config: RecoveryConfig, output: str | Path, *, preflight_only=False) -> dict:
    """Persist gates before sampling, resume matching raw cells, and report all fits."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.torch_threads)
    metadata = _metadata(config)
    preflight_path = output / "recovery_preflight.json"
    if preflight_path.exists():
        previous = json.loads(preflight_path.read_text(encoding="utf-8"))
        if previous["metadata"]["fingerprint"] != metadata["fingerprint"]:
            raise ValueError("incompatible recovery metadata; use a different output directory")
    diagnostic = preflight(config)
    _write_json(preflight_path, {"metadata": metadata, "diagnostic": diagnostic})
    if not diagnostic["passed"]:
        raise ValueError("recovery preflight gates failed; document a fixture revision before sampling")
    if preflight_only:
        return {"metadata": metadata, "diagnostic": diagnostic}
    env, features = _fixture(config)
    truth = torch.tensor(config.truth, dtype=torch.float64)
    solution = plan(env, features, truth, config.horizon, config.terminal_penalty)
    cells, regularization, pool_rows = [], [], []
    for seed in config.seeds:
        pool = sample(env, solution, [env.start_state] * max(config.sample_sizes), seed)
        records = [{"seed": seed, "index": index, "states": list(record.states),
                    "actions": list(record.actions), "arrival_step": record.arrival_step}
                   for index, record in enumerate(pool)]
        pool_rows.extend(records)
        for n in config.sample_sizes:
            prefix = pool[:n]
            identity = {"experiment": metadata["fingerprint"], "seed": seed, "n": n,
                        "prefix_hash": _digest(records[:n]), "bootstrap_seed": bootstrap_seed(seed, n)}
            path = output / "recovery_cells" / f"seed_{seed}_n_{n}.json"
            if path.exists():
                cell = json.loads(path.read_text(encoding="utf-8"))
                if cell["identity"] != identity:
                    raise ValueError(f"incompatible recovery checkpoint: {path}")
            else:
                point = fit(env, features, prefix, terminal_penalty=config.terminal_penalty)
                boot = bootstrap_fit(env, features, prefix, bootstrap_seed(seed, n),
                                      refits=config.bootstrap_refits,
                                      terminal_penalty=config.terminal_penalty)
                sensitivity = []
                if n in config.sensitivity_sizes:
                    for l2 in config.l2_values:
                        result = point if l2 == 0 else fit(env, features, prefix, l2=l2,
                                                          terminal_penalty=config.terminal_penalty)
                        sensitivity.append(_fit_row(config, seed, n, result))
                cell = {"identity": identity, "seed": seed, "n": n,
                    "prefix_indices": [0, n], "point_fit": point,
                    "bootstrap_seed": bootstrap_seed(seed, n), "bootstrap": boot,
                    "completion_fraction": sum(r.arrival_step is not None for r in prefix) / n,
                    "regularization": sensitivity}
                _write_json(path, cell)
            cells.append(cell)
            regularization.extend(cell["regularization"])
            print(f"seed={seed} n={n}: point={cell['point_fit']['success']}, "
                  f"bootstrap={cell['bootstrap']['successful_refits']}/{config.bootstrap_refits}, "
                  f"completion={cell['completion_fraction']:.6f}", flush=True)
    with (output / "recovery_demos.jsonl").open("w", encoding="utf-8") as stream:
        for record in pool_rows:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    summary = summarize_recovery(config, cells, diagnostic)
    report = {"metadata": metadata, "preflight": diagnostic, "cells": cells,
              "regularization": regularization, "summary": summary,
              "resume_policy": "Matching configuration, model sources, versions, and prefix hashes only."}
    _write_json(output / "recovery.json", report)
    _csv(output / "recovery_fits.csv", [_fit_row(config, c["seed"], c["n"], c["point_fit"],
        completion_fraction=c["completion_fraction"],
        bootstrap_intervals=c["bootstrap"]["percentile_intervals"],
        bootstrap_reliable=c["bootstrap"]["reliable"],
        bootstrap_successes=c["bootstrap"]["successful_refits"]) for c in cells])
    _csv(output / "recovery_bootstrap.csv", [_fit_row(config, c["seed"], c["n"], result)
                                             for c in cells for result in c["bootstrap"]["fits"]])
    _csv(output / "recovery_summary.csv", summary["by_sample_size"])
    _csv(output / "recovery_regularization.csv", regularization)
    _plot(output, config, cells, summary)
    return report
