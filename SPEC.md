# Wheelchair navigation preference learning

## Scope

Implement phases 0–3 only. This is a discrete, wheelchair-inspired preference
model, not a medical device or clinical controller. The research question is
whether finite-horizon maximum-entropy IRL can learn relative navigation costs.
Unseen-layout prediction, baselines, and human demonstration collection are
deferred. No evidence exists yet for any resume claim.

## Environment

Coordinates are zero-based `(row, column)`, with rows increasing downward.
Each free cell has four or eight heading states. Eight headings, clockwise,
are E, SE, S, SW, W, NW, N, NE. Four headings are E, S, W, N.
Default generated layouts are 20 by 20. Occupied cells include stationary
people and other obstacles without distinguishing their semantics.
The input occupancy array must be nonempty, two-dimensional, and boolean.
Start is `(row, column, heading_index)` and goal is `(row, column)`; both must
be in bounds and free. Explicitly constructed fixtures may be disconnected;
connectivity is required of generated layouts, not imposed by the constructor.

Action indices are forward=0, left=1, right=2. Left decrements the heading
modulo its count; right increments it. Turning remains in the same cell.
Forward outside the grid or into occupancy is illegal, not a self-loop.
Diagonal forward requires both adjacent cardinal cells to be free, preventing
corner-cutting. Invalid actions have a false mask and a safe placeholder
next-state index, and must never be sampled or enumerated.

Every heading at the goal is absorbing. Only action index 0 is legal there,
and it is a zero-feature self-loop. It is bookkeeping, not another control.
Entering the goal pays ordinary incoming features. Subsequent actions cost
zero. This avoids artificial entropy from multiple goal continuations.

Generation samples seeded independent occupancy, clears start and goal,
and accepts only after graph search using actual legal state transitions
finds a route from the specified start heading to some goal heading.
Bounded rejection raises an error rather than returning disconnected data.
Connectivity does not imply reachability within a requested horizon.

## Features and reward

Features are nonnegative costs in order `[distance, turning, clearance]`.
Forward distance is 1 for cardinal movement and sqrt(2) for diagonal movement.
Turns have zero distance and turning effort `2*pi/heading_count` radians.
Forward has zero turning effort.

Clearance distance is the minimum Euclidean distance from the destination
cell center to occupied cell centers, in grid-cell units. The feature is
`max(0, d_safe-d)**2`, with default `d_safe=2`. Charge this on every legal
non-goal-source action, including turns. This penalizes time spent close to
occupancy and makes turning in narrow areas costly. Destination accounting
assesses the location occupied after each action. No obstacles means infinite
clearance distance and zero clearance cost. Boundaries are not occupied cells.
Invalid-action feature entries are zero and excluded by masks.
They must be exactly zero, not infinity placeholders. Compute rewards from
finite feature/weight arithmetic, then apply torch.where or masked_fill to
invalid actions. Never multiply a mask by an infinite reward or logit.

Reward is `-dot(w, phi)`. Planning and likelihood floating tensors are
`torch.float64`. Temperature is fixed to 1. The legacy `fixed_distance` mode
fits `w=[1, softplus(theta_0), softplus(theta_1)]`. The opt-in `three_free`
mode fits `w=softplus(theta)` for three raw parameters. Initial physical free
weights are all 1. Softplus represents strictly positive weights and can
approach, but not attain, zero. At fixed temperature reward scale is identified
when the trajectory features are informative. Fixing distance imposes scale=1;
it does not remove an unidentifiable reward scale and can bias fitted cost
ratios when the true scale differs. Report distance scale and the ratios
`w_turn/w_distance`, `w_clearance/w_distance` separately. Neither model
establishes unique psychological preferences.

Numerical range validation conservatively requires each per-action feature
to be <= float64_max/(16*T) for T>0. The factor 16 reserves accumulation
headroom so tiny policy-normalization roundoff cannot overflow near-limit
feature expectations or gradients. This protects cumulative feature counts and
weight derivatives even when tiny weights make the rewards look harmless.
It can reject extreme features whose worst-case repeated use is not actually
reachable. Reject unsupported scales explicitly; do not clip their features.
Also require `M + T*(max_abs_action_reward + log(3)) <= 1000` over all states.
This conservative score budget bounds absolute trajectory scores and log
partitions, including action-sequence entropy, at a scale with float64 spacing
below 1.2e-13. It leaves headroom against the declared 1e-10 probability
tolerance and avoids losing an O(1) entropy correction when subtracting large
values. It is a supported numerical domain, not a promise of arbitrary-magnitude
float64 accuracy. Reject outside it before returning policies or likelihoods.

## Finite-horizon distribution

For horizon T, terminal value is 0 at goals and -M elsewhere. M is fixed
per fixture and never learned. M=30 is a candidate for the tiny fixtures
here, not a default justified for arbitrary layouts. Before using a fixture,
code must check `M > max optimal finishing cost-to-go + margin` at its
declared reference weights, with margin=1 cost unit. Compute the maximum
over states reachable from the declared starts that can finish within the
remaining horizon, at all reachable times. Report unreachable-to-goal cases
separately rather than treating their infinite cost as finite. Save the
maximum, margin, weights, and check outcome per fixture. This cost comparison
does not account for trajectory multiplicity and does not guarantee completion.
Limitation: the M margin check applies only at the declared reference weights.
Fitted free weights up to 10 may produce finishing costs that exceed the
M scale. M stays fixed during fitting, so the fitted model remains in the same
trajectory distribution family even when that reference-weight margin no
longer holds.
At each t, `Q_t(s,a)=-w·phi(s,a)+V_(t+1)(next(s,a))` and
`V_t(s)=logsumexp(Q_t(s,a))` over legal actions only.
`log pi_t(a|s)=Q_t(s,a)-V_t(s)`. Every state has at least one legal action.
Use negative infinity for masked logits, never an all-masked reduction.
plan() must not detach any tensors. With differentiable weights, both values
and log policy must retain requires_grad. Verify finite weight gradients of
V_0 and trajectory_log_prob with the backward pass over all states, including
unreachable components and absorbing goal states.

For a legal, action-labelled length-T trajectory conditional on initial state,
score is `-w·sum_t(phi_t)+V_T(s_T)`. The partition function is `exp(V_0(s_0))`.
Log likelihood is score minus `V_0(s_0)`, equivalently the sum of policy log
probabilities. Distinct action sequences remain distinct even if their state
sequences meet. A finite penalty does not enforce completion. Sampling retains
every unfinished trajectory. Early goal arrival receives unique continuation
padding to T, not truncated likelihood. Demo records include `arrival_step`,
the first state index at the goal, 0 for a goal start, and None if unfinished.

Loss is the arithmetic mean of complete trajectory NLLs plus
`lambda/2 * sum(w_free**2)`. Legacy free weights are turning and clearance;
three-free regularization also shrinks distance scale. Positive-lambda
comparisons between parameterizations are therefore not like-for-like.
It is not divided by horizon or total
action count. Fit accepts one shared horizon per call and rejects mismatched
lengths. Different initial states may share a fit. Terminal penalties have
zero weight derivative.
Skip penalty arithmetic entirely when lambda=0. For positive lambda, compute
the equivalent sum of `(sqrt(lambda)/sqrt(2) * w_free)**2` to avoid unnecessary
intermediate square overflow. This changes arithmetic order, not the objective.

## Verification

Test A enumerates every legal length-4 sequence on a 2x2 four-heading grid,
start `(0,0,E)`, goal `(0,1)`, no occupancy, M=30. This includes completed
and unfinished trajectories. Compare stable normalized scores to policy
products with absolute tolerance 1e-10 and check normalization independently.

A second exhaustive Test A uses a 3x3 eight-heading grid, occupied cell
`(0,1)`, start `(1,0,E)`, goal `(2,2)`, T=5, and non-unit weights
`[1.3, 0.7, 1.2]`. Enumerate every legal action sequence with the terminal
score, including unfinished sequences. Check legal diagonal moves occur.
Check the forward mask is False at these reachable state/time pairs:
- At t=1, `(1,0,NE)` toward `(0,1)`, reached by left from the start.
  This is an occupied-destination block, not a corner-cutting check.
- At t=2, `(1,1,NE)` toward `(0,2)`, reached by forward then left.
  The destination is free, but the adjacent corner `(0,1)` is occupied.
- At t=4, `(1,1,NW)` toward `(0,0)`, reached by forward then three lefts.
  The destination is free, but the adjacent corner `(0,1)` is occupied.
Assert no enumerated sequence takes a blocked forward move, including at
these state/time pairs. Do not use `(0,0,SE)` as a within-horizon corner
check: it needs six actions from this start and is unreachable for T=5.
Compare probabilities at the same tolerance.

Test B uses a tiny occupied fixture with nonzero clearance and compares mean
unregularized NLL autograd, central differences with step 1e-5, and empirical
minus exact expected feature counts. Gradient tolerance is 1e-7. Run float64
gradcheck with eps=1e-6, atol=1e-5, rtol=1e-3. Separately test the regularized
softplus-parameterized loss. The Test B fixture must include single-legal-action
states, including absorbing goal states. On this same single-legal-action
fixture, compare the autograd Hessian of `V_0` with the negative central
finite difference of `expected_features`
with respect to physical weights, since `grad(V_0)=-E[phi]`. Use step 1e-5
and absolute tolerance 1e-7. Machine-generate discrepancy artifacts.
`scripts/verify_planner.py` must import the same verification functions used
by the tests, rather than implementing a parallel set of numerical checks.

## Predeclared synthetic recovery experiment

Initial candidate is a 5x5 four-heading layout with occupied cells `(1,2)`
and `(2,2)`, start `(2,0,E)`, goal `(2,4)`, d_safe=2, T=24, M=30.
Ground truth is `[1, 0.7, 1.2]`. Dataset seeds are 11, 29, 47, 83, 101;
sample sizes are 32, 128, 512, 2048. Use nested prefixes of each unfiltered
2048-trajectory pool. Use one CPU torch thread. Bootstrap base seeds are
`73001 + 1000003*dataset_seed + sample_size`; the inference module records
each separately derived resampling seed. Before interpreting estimates, compute the exact
free-feature covariance via the Hessian of log partition at ground truth.
Report eigenvalues, eigenvectors, and condition number. Before any recovery
data run, the ground-truth exact completion probability must be >=0.90, and
the free-weight 2x2 covariance must have minimum eigenvalue >=0.1 and condition
number <=100. Check and save these criteria and the terminal-penalty margin
check before sampling. If the fixture fails, document its revision before
generating final data, then repeat all checks. Never silently replace seeds
or filter samples.

Use deterministic float64 LBFGS, strong-Wolfe line search, maximum 150
iterations, and report objective, closure evaluations, final raw-parameter
and physical-weight gradient norms, and convergence/failure status. Success
requires a finite objective, raw-gradient infinity norm <=1e-6, optimizer
iterations <150, and both free physical weights within inclusive [1e-3, 10].
The raw-gradient criterion applies to the objective actually optimized:
mean trajectory NLL when lambda=0, and the regularized loss defined above
when lambda>0. It is not a check of unregularized NLL gradients for a
regularized fit.
Otherwise report failure with named reason `non-finite`, `boundary`, or
`unconverged`, in that priority when multiple checks fail. Never clip diverging
weights. Primary lambda is 0; sensitivity uses 0, 0.001, 0.01 for every seed
at both n=32 and n=2048.

Bootstrap whole trajectories within each dataset and sample size, 100 refits
per cell, using separately recorded fixed resampling seeds. Report percentile
95% intervals per free weight, successful and failed refit counts, and failure
reasons. Failed fits are excluded from percentiles and explicitly counted;
if fewer than 80 of the 100 refits succeed, mark intervals unreliable.
For an explicitly requested custom refit budget B, require at least
max(80, ceil(0.8*B)) successes. Small unit-test refit budgets therefore cannot
be labeled reliable. Bootstrap intervals
describe uncertainty within a dataset; across-seed spread describes independent
dataset variation. These are different quantities.

Save configuration, versions, all point fits, all bootstrap fits, intervals,
completion proportions, expected completion, identifiability diagnostics,
and regularization sensitivity as generated JSON/CSV under `results/`.
Save a recovery-curve PNG with uncertainty and a completion summary.
Report actual trends even if error does not decrease monotonically.

Error metrics are absolute error per free weight and the Euclidean norm of
the two-component free-weight error. Report per-seed values and across-seed
mean and sample standard deviation for each n. Report the least-squares
log-log slope of mean error versus n, separately for each metric, with an
explicit caveat that five seeds provide weak evidence about scaling. If a
mean error is zero, report the corresponding log slope as undefined rather
than silently adding an offset. Retain failed fits in the fit records, report
their counts, and label error summaries as conditional on successful fits.

At n=2048, report predicted asymptotic standard deviations from
`sqrt(diag(inverse(free_feature_covariance))/2048)` at ground truth alongside
the observed across-seed sample standard deviations of estimates. For every
n and free weight, report how many reliable seed bootstrap intervals contain
truth, with the number of reliable intervals as denominator. These coverage
counts are descriptive only, not a calibrated coverage claim with five seeds.
Both the asymptotic-standard-deviation comparison and interval containment
counts are descriptive diagnostics only, with no pass/fail thresholds.

## Claims and stopping rule

### Three-free scale correction protocol

The predeclared alternative protocol is `docs/free-weights-protocol.md`.
The main recovery retains the original fixture, truth [1,0.7,1.2], seeds,
sample sizes, optimizer, and bootstrap budget. For `three_free`, use the full
3x3 physical-feature covariance. Before sampling, require minimum eigenvalue
>=0.1, condition <=100, completion >=0.90, and the unchanged M margin. Apply
fit-success bounds to all three free weights. Numerical limits remain unchanged.

Exact population reconnaissance at truth scales 0.5,1,2 precedes protocol
freeze. A separate slow/run-once regression at [2,1.4,2.4] tests scale-induced
ratio bias; it is not gated recovery evidence because it fails the covariance
minimum-eigenvalue gate. Main scale-1 recovery compares extra-parameter variance,
not misspecification bias. Use paired per-seed differences on identical prefixes.
Never resume into or rewrite historical outputs or source fingerprints. New
three-free output directories must not exist. No held-out tuning is allowed.

Only float64 differentiable DP, exhaustive agreement, gradient verification,
and synthetic recovery are eligible for support this round after execution.
Combined synthetic-and-keyboard recovery remains PARTIAL at best. Prediction
on 100 unseen 20x20 layouts remains UNSUPPORTED. No human demonstrations
have been collected and no generalization experiment has been run.

Run the full pytest suite and recovery command, document evidence, commit,
and push without force only after checking remote history again. Stop after
phase 3 and wait for orchestrator review. Do not create deferred placeholders.

Current review instruction: Task 4 is checkpointed. Proceed with Task 5's gated
tiny synthetic recovery experiment and evidence-backed reporting, then stop for
orchestrator review. Leave this round uncommitted; SPEC.md remains uncommitted.
Do not push, collect human data, run large-layout tests, or start later phases.
