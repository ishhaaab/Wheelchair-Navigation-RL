"""Run only the predeclared tiny synthetic recovery experiment, never later phases."""

import argparse
from pathlib import Path

from wnp.recovery import RecoveryConfig, run_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Save analytic gates without drawing demonstrations or fitting")
    args = parser.parse_args()
    report = run_recovery(RecoveryConfig(), args.output, preflight_only=args.preflight_only)
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
