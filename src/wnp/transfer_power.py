"""Predeclared Monte Carlo on statistical outcomes, not navigation labels."""

import hashlib
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist

import numpy as np


CANDIDATES = (64, 96, 128, 192, 256, 384, 512, 768, 1024)
SDS = (.5, 1., 1.5)
REPLICATES = 10_000
SEED = 2026110601


def wilson(successes: int, total: int) -> list[float]:
    """Two-sided 95% Monte Carlo interval for a binomial probability."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-half), min(1., center+half)]


def simulate_cell(n: int, delta: float, sd: float, replicates: int, seed: int) -> dict:
    """Six equicorrelated layout contrasts, estimated-SD normal-reference test."""
    if (not isinstance(n, int) or isinstance(n, bool) or n < 2
            or not isinstance(replicates, int) or isinstance(replicates, bool)
            or replicates < 1 or not math.isfinite(sd) or sd <= 0
            or not math.isfinite(delta)):
        raise ValueError("finite delta, positive sd, n>=2 and replicates>=1 required")
    rng = np.random.default_rng(seed)
    threshold = NormalDist().inv_cdf(1-.05/6)
    counts = np.zeros(6, dtype=np.int64)
    joint = any_success = 0
    # Bound memory independently of the declared Monte Carlo sample size.
    for offset in range(0, replicates, 100):
        batch = min(100, replicates-offset)
        common = rng.standard_normal((batch, n, 1))
        separate = rng.standard_normal((batch, n, 6))
        outcomes = delta + sd * (common + separate) / math.sqrt(2)
        statistics = outcomes.mean(axis=1) / (outcomes.std(axis=1, ddof=1) / math.sqrt(n))
        reject = statistics > threshold
        counts += reject.sum(axis=0)
        joint += int(reject.all(axis=1).sum())
        any_success += int(reject.any(axis=1).sum())
    return {"n": n, "delta": delta, "sd": sd, "seed": int(seed),
            "replicates": replicates, "correlation": .5,
            "per_contrast_successes": counts.tolist(),
            "per_contrast_power": (counts/replicates).tolist(),
            "per_contrast_ci": [wilson(int(c), replicates) for c in counts],
            "joint_successes": joint, "joint_power": joint/replicates,
            "joint_ci": wilson(joint, replicates),
            "any_successes": any_success, "any_power": any_success/replicates,
            "any_ci": wilson(any_success, replicates)}


def choose_n(cells, candidates=CANDIDATES, sds=SDS):
    """Smallest candidate meeting the frozen joint-power lower-bound gate."""
    for n in sorted(candidates):
        if all(any(c["n"] == n and c["sd"] == sd and c["joint_ci"][0] >= .90
                   for c in cells) for sd in sds):
            return n
    return None


def power_report() -> dict:
    """Execute exactly the configuration declared in the design document."""
    cells, null = [], []
    for scenario, sd in enumerate(SDS):
        for n in CANDIDATES:
            seed = int(np.random.SeedSequence([SEED, scenario, n]).generate_state(
                1, dtype=np.uint64)[0])
            cells.append(simulate_cell(n, .25, sd, REPLICATES, seed))
    for n in CANDIDATES:
        seed = int(np.random.SeedSequence([SEED, len(SDS), n]).generate_state(
            1, dtype=np.uint64)[0])
        null.append(simulate_cell(n, 0., 1.5, REPLICATES, seed))
    return {"kind": "synthetic statistical outcomes, not navigation data",
            "seed": SEED, "rng": "PCG64", "replicates": REPLICATES,
            "candidates": CANDIDATES, "sds": SDS, "effect_nats": .25,
            "contrasts": 6, "correlation": .5, "family_alpha": .05,
            "target_joint_power_lower_bound": .90, "cells": cells, "null": null,
            "selected_n": choose_n(cells),
            "limitations": ["Normal-reference approximation, not bootstrap calibration",
                            "Assumed effects/SD/correlation, not empirical estimates",
                            "Conditions on five trained models, not training-population power"],
            "versions": {"python": platform.python_version(), "numpy": np.__version__}}


def write_report(directory, report: dict) -> None:
    """Refuse even an empty existing directory. Never resume or overwrite."""
    directory = Path(directory)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "power.json").open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "docs/transfer-preregistration-v1/design.md",
             root / "tests/test_transfer_power.py"]
    fingerprints = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths}
    report = power_report()
    report["source_sha256_before_simulation"] = fingerprints
    write_report(args.output, report)
    print(json.dumps({"selected_n": report["selected_n"], "output": str(args.output)}))
    if report["selected_n"] is None:
        raise SystemExit("No candidate meets the predeclared power gate; stop.")


if __name__ == "__main__":
    main()
