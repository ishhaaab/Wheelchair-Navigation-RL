# Misspecification experiments, declared before sampling or fitting

## Steps
- [ ] Freeze analytic diagnostics and the new incomplete-trajectory fixture.
- [ ] Test the isolated experiment runner without changing existing modules.
- [ ] Generate declared datasets and fit once with frozen settings.
- [ ] Evaluate fresh held-out demos, report every seed, and run the full suite.

## Scope and preservation
This is a new sensitivity study, not a replacement for historical recovery.
All artifacts go into this new directory using exclusive creation. Existing
results and source modules remain unchanged. No commit or push. Stop for review
after reporting, or immediately on a numerical-domain limit. No clipping,
relaxation of numerical limits, retries with altered settings, seed replacement,
filtering, or tuning after held-out evaluation.

Baseline full pytest before changes: 334 passed, 1 skipped in 13.93 seconds.
The skipped slow regression is reported, not treated as executed.

## Frozen data and optimization
Seeds: 11, 29, 47, 83, 101. Each generating condition uses 2048 training
trajectories and 4096 independent held-out trajectories. Training RNG seed is
the listed seed; held-out RNG seed is 1000000 plus that seed. Reusing these
streams across conditions provides pairing, not independence between conditions.
All trajectories, including unfinished ones, are retained and padded normally.
No bootstrap, regularization sweep, or sample-size sweep is part of this task.
Both fixed_distance and three_free fit each dataset, with identical paired
training and evaluation records. These are single-layout experiments, not
unseen-layout generalization tests.

Use existing float64 CPU fit(), one torch thread, lambda=0, all initially free
physical weights=1, LBFGS strong-Wolfe, max_iter=150 and existing tolerances.
Existing success gates apply: finite objective, raw gradient infinity norm
<=1e-6, iterations <150, free weights in [0.001,10]. Report all failed fits,
their estimates and diagnostics; never drop them from tables. Any planner
numerical-domain error, including one flagged by fit(), stops the entire study.
The score limit remains M + T*(max_abs_action_reward + log(3)) <=1000.

## Experiments and baselines
Base truth is [1,0.7,1.2]. Standard features use d_safe=2, quadratic clearance.
Base layout is 5x5 with occupancy (1,2),(2,2), start (2,0,E), goal (2,4),
four headings, T=24, M=30.

A. Scale: multiply the complete truth vector by 0.5, 1, 2. Generate and fit
standard features. Both parameterizations fit each condition. Compare ratio
and weight errors with the paired scale=1 row for the same seed and method.
Also report fixed_distance minus three_free error/NLL on each paired dataset.

B. Features: generate with d_safe=2.5, base truth, T=24, M=30, but fit standard
d_safe=2 features. The matched d_safe=2 scale=1 dataset is the per-seed baseline.
Report numerical truth-ratio error as a sensitivity diagnostic, not as recovery
of equivalent preferences: coefficients across different feature definitions
are not semantically identical. Report NLL relative to the true generating
distribution on the SAME fresh demos, which makes predictive misspecification
interpretable even when entropies differ between conditions.

C. Terminal penalty: define a NEW fixture by exact, label-free reconnaissance
over widths W=5,...,24, in that order. Use a 5xW grid, the same two occupied
cells and start as above, goal=(2,W-1), T=24, standard features and base truth.
Choose the first width with exact generating completion in [0.70,0.90] at
M=30. If none qualify, stop and report that this declared family failed, without
sampling. Freeze its full coordinates and analytic diagnostics before labels.
The historical >=0.90 completion gate does NOT apply because the purpose is
to exercise finite-M unfinished trajectories rather than high-completion
recovery. Generate once at M=30 and fit M=5,10,30,100 using both methods.
The per-seed, per-method baseline is fit M=30 on exactly the same training and
held-out data. Report weight error, ratio error, NLL, and deltas versus M=30.
M is deliberately misspecified for three fit conditions; the historical
reference finishing-cost margin is diagnostic, not an acceptance gate for
those intentional conditions. Report margin diagnostics including M=30.

## Analytic diagnostics and gates
Before labels, save exact completion, full feature covariance Hessian at truth,
eigenvalues/condition number, and reference terminal-margin diagnostics for
every generator. Record historical covariance thresholds min eigenvalue>=0.1
and condition<=100 as diagnostics, not exclusion gates: failure to identify is
one possible degradation under study and must not be selected away. Scale and
feature completion >=0.90 is likewise recorded as a historical diagnostic,
not grounds to replace a requested condition. Only the new M fixture uses
the specified 0.70--0.90 selection gate. Check actual truth/initial-fit numerical
budgets; any numerical-domain exception stops rather than selects another case.

## Metrics and reporting, frozen before data
For each seed, condition and method report fitted three-vector; signed and
absolute coordinate weight errors; full-vector weight L2 error; fitted
turn/distance and clearance/distance ratios; signed and absolute ratio errors;
ratio L2 error relative to the generating coefficient ratios. Weight errors
for fixed_distance include its imposed distance error.

Held-out NLL is mean complete length-T trajectory negative log probability,
not per-action NLL, including M for every unfinished record. Evaluate each
fit under its declared fitted features and M. Also report oracle NLL under
the true generating features, weights and M on the same held-out records,
and fitted minus oracle NLL. This finite-sample excess may be negative.
Report held-out standard error of the per-trajectory fitted-minus-oracle
NLL difference. No hypothesis test or universal recovery pass threshold is
claimed. Degradation is a signed per-seed delta, never hidden by averaging.
Report training and held-out finished counts and exact completion.

Write one table per experiment with every seed and condition, including fit
failure status. Save raw demos, all fit records, analytic diagnostics, source
hashes, package versions, and machine-readable per-seed metrics. Do not tune
after viewing held-out results. Negative and null findings remain in the report.
