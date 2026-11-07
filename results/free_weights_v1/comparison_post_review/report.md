# Three-free recovery comparison

Scale 1 is well specified in both models. These data assess the extra-parameter variance cost, not scale-misspecification bias. Five paired seeds give descriptive comparisons, not a superiority or scaling-law claim.

## Population scale bias

Exact population cross-entropy optima, no sampling. Truth ratios are [0.7,1.2].

| Scale | Fixed turning ratio | Fixed clearance ratio |
|---:|---:|---:|
| 0.5 | 0.399090 | 0.656842 |
| 1.0 | 0.700000 | 1.200000 |
| 2.0 | 1.429186 | 1.409700 |

## Behavioral regression

Before implementation the existing fixed estimator passed its fit and bias gates, but failed ratio recovery with weights [1.0, 1.4313240665766012, 1.4067140729853016]. Three-free inference passed unchanged tolerances with scale 1.99640528 and ratios [0.7022962333912257, 1.1983903117518255]. Both used the same frozen 65,536-demo pool. The regression is opt-in slow, not part of routine suite passes.

At doubled scale, covariance minimum eigenvalue 0.0302182 fails the recovery gate. The local scale standard deviation is larger than ratio standard deviations. Increasing determinism weakens scale information; human demonstrations may not identify scale well even when ratios appear stable.

## Paired main recovery

All 10240 historical demo records match. Differences are new minus old; positive ratio error differences favor the legacy model.

| n | Successful pairs | Old mean ratio L2 error | New mean ratio L2 error | Paired difference mean +/- sample std |
|---:|---:|---:|---:|---:|
| 32 | 5 | 0.109860 | 0.139571 | 0.029712 +/- 0.100244 |
| 128 | 5 | 0.105128 | 0.151688 | 0.046560 +/- 0.083111 |
| 512 | 5 | 0.049759 | 0.076083 | 0.026324 +/- 0.048072 |
| 2048 | 5 | 0.019029 | 0.025195 | 0.006166 +/- 0.011414 |

Three-free coordinates, means +/- across-seed sample standard deviations. Distance scale is estimated here; the old model fixes it to 1.

| n | Distance scale | Turning / distance | Clearance / distance |
|---:|---:|---:|---:|
| 32 | 1.119426 +/- 0.106173 | 0.636989 +/- 0.080643 | 1.161678 +/- 0.134084 |
| 128 | 0.973757 +/- 0.078996 | 0.728277 +/- 0.063050 | 1.264446 +/- 0.157057 |
| 512 | 0.967922 +/- 0.080740 | 0.729333 +/- 0.067503 | 1.213511 +/- 0.045852 |
| 2048 | 1.006746 +/- 0.024338 | 0.693162 +/- 0.017902 | 1.190482 +/- 0.022515 |

Complete paired estimates and absolute-error differences are in paired_cells.csv. Failure counts, bootstrap coverage, physical and ratio/scale uncertainty, log-log slopes, and per-seed fit diagnostics remain in comparison.json and the recovery artifacts. Five seeds do not establish calibrated coverage, superiority, or monotonic scaling.

## Regularization

Three-free L2 penalizes all three physical weights, including scale; old L2 penalizes only turning and clearance. Positive-lambda sensitivities are not like-for-like.

See regularization_sensitivity.csv for both models, including failures.

## Numerical budget and integrity

{"available_budget_count": 2040, "domain_failures": 0, "failures": {}, "limit": 1000, "max_fitted_budget": 184.72276884317182, "max_iterations": 35, "max_raw_gradient_inf_norm": 2.6115224454571725e-07, "min_fitted_budget": 101.5797498468086, "unavailable_budgets": 0, "unavailable_raw_gradient_norms": 0, "unique_fits": 2040}

Upper fit-box corner budget 673.357813 is below the unchanged 1000 limit. No T/M or limit changes. Point fitted M margin passes 20/20 evaluated; 0 unavailable. Reference margin remains a preflight gate only. Point-fit margins are descriptive. Bootstrap margins are not individually checked.

Verified 43 historical/protected files unchanged. irl.py is the intended changed inference source. Historical recovery retains historical source fingerprints. Current code cannot resume it; offered-route content hashes remain unchanged.

The saved run predates review fixes to planner-domain failure classification and failed-record/report serialization. No optimizer, objective, threshold, seed, data, or successful fitted result was changed or regenerated. Run source hashes remain historical; current hashes are recorded separately.

No held-out evaluation, human data, commits, or pushes. Stop for review.
