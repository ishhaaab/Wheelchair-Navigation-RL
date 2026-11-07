"""Run only the predeclared tiny synthetic recovery experiment, never later phases."""

import argparse
from dataclasses import replace
from pathlib import Path

from wnp.recovery import RecoveryConfig, run_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Save analytic gates without drawing demonstrations or fitting")
    parser.add_argument("--parameterization", choices=("fixed_distance", "three_free"),
                        default="fixed_distance")
    parser.add_argument("--baseline-demos", type=Path,
                        help="Require exact paired prefixes from a historical demo JSONL")
    args = parser.parse_args()
    config = replace(RecoveryConfig(), parameterization=args.parameterization)
    report = run_recovery(config, args.output, preflight_only=args.preflight_only,
                          baseline_demos=args.baseline_demos)
    diagnostic = report["diagnostic"] if args.preflight_only else report["preflight"]
    print(f"Preflight passed: completion={diagnostic['completion_probability']:.15g}, "
          f"minimum covariance eigenvalue={diagnostic['eigenvalues'][0]:.9g}, "
          f"condition={diagnostic['condition_number']:.9g}")
    if not args.preflight_only:
        for row in report["summary"]["by_sample_size"]:
            print(f"n={row['n']}: mean L2 error={row['mean_euclidean_error']}, "
                  f"std={row['euclidean_error_sample_std']}, failures={row['failed_point_fits']}")
    print(f"Saved results under {args.output}")


if __name__ == "__main__":
    main()
