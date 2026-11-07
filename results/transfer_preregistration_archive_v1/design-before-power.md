# Unseen-layout reward-learning pre-registration

Status: proposed protocol, awaiting user review. Written before power simulation,
new demonstration labels, or final fitting on 2026-11-06. This is a new study,
not an amendment to or replacement for historical recovery artifacts.

## Steps and authorization

- [ ] Step 1: write the design and numerical-budget calculation.
- [ ] Step 2: simulate synthetic statistical outcomes, choose N_test, record results.
- [ ] Step 3: implement and test a shared-weight multi-environment fit.

Stop after Step 3. User approval of this document is required before generating
any held-out layouts, start-goal pairs, or demonstrations. This task generates
no training data either. Tiny deterministic test fixtures and their exhaustive
action sequences are software tests, not study data. No commits or pushes.
Existing modules and artifacts stay intact. All new study runs must use a
previously nonexistent output directory, created with exist_ok=False.

The request supplies the study structure. Numerical constants not specified by
the user below are proposed author choices, not previously approved decisions.

## Question and claims

Does a shared feature-based reward learned on 20 layouts predict full
action-labelled trajectories on unseen layouts better than strong baselines?
This is synthetic distribution transfer, not evidence about human preferences
or wheelchair safety. Evaluate a correctly specified generator and a
misspecified generator separately. Never pool their results to hide a failure.
Null and negative differences remain results. There is no promised improvement.

## Frozen environment and numerical domain

Use existing GridMDP semantics, 10x10 grids, four headings, three controls,
temperature 1, float64, absorbing goals with one zero-feature padding action.
Retain unfinished trajectories. Boundaries are not obstacles. Existing features
are distance, turning in radians, and squared clearance deficit at destination,
including turns. Use d_safe=2 and reward -w dot phi.

Every free destination is at least one cell-center unit from an obstacle.
Thus clearance cost <=1, forward distance=1, and turning=pi/2. Distance and
turning cannot both be nonzero on one action. For all three physical weights
in [0,10], max|r| <= max(20, 10*pi/2+10) = 25.707963267948966.
The linear-deficit misspecification below has the same bound.

Fix terminal penalty M=100 and horizon
T=floor((1000-100)/(25.707963267948966+log(3)))=33.
The conservative all-state score budget is 984.6169933683635 <=1000.
T=34 would give 1011.4235689249806 and is unsupported at the declared weight cap.
This is a horizon-limited 10x10 study, not arbitrary-maze transfer.
Never increase the score limit, clip rewards/features/weights, prune states,
reduce the horizon after seeing outcomes, or change M during fitting.
If a planner call exceeds its numerical domain, preserve diagnostics and stop
the study. The optimizer is unconstrained softplus; its trial steps can exceed
10. A numerical-domain violation is fatal even on a rejected line-search trial.
Finite fitted weights outside [0.001,10] are failed boundary fits, not clipped.

At reference truth [1,0.7,1.2], max action cost <=2.2995574287564278.
Any finishing path of at most 33 actions costs <=75.88539514896212.
Therefore M=100 exceeds that bound plus the required margin 1. Still save the
existing reachable-state/time terminal-margin diagnostic per environment.
This reference check does not guarantee completion or a margin at fitted weights.

## Layouts, pairs, demonstrations, and seeds

Use five independent training replicates k=0,...,4. Each has N_train=20
independent occupancy layouts, four distinct start-goal pairs per occupancy,
and 32 trajectories per pair per generator arm. This gives 2,560 demos per
replicate per arm. All methods within an arm/replicate receive identical demos.
Both arms share geometries, but use independent trajectory RNG streams.
Models are trained separately for each arm and training replicate.

Proposed generator: independent Bernoulli occupancy with probability 0.2.
Draw an occupancy first, then uniformly propose distinct ordered free start and
goal cells and a uniform initial heading. Accept pairs only when legal-action
BFS reaches a goal in <=33 actions. This is a geometric eligibility condition,
not filtering by costs, labels, completion probability, or learned predictions.
Draw four distinct ordered cell/heading/goal tuples, with at most 1,000 pair
proposals per occupancy. Reject occupancy if fewer than four qualify. Permit
at most 1,000 occupancy proposals per layout slot; exhaustion stops the study.
Record all proposal counts. State explicitly that results concern this
conditioned generator, not every connected 10x10 grid.

Test uses N_test independent occupancy layouts, four pairs per layout, and 64
unfiltered trajectories per pair per arm. It is shared across training
replicates and methods. N_test is chosen only by Step 2's rule below. The two
arms share test geometries but have independent labels. No validation or tuning
set is used. No method selection uses test labels.

Use NumPy SeedSequence with root 20261106 and integer spawn keys
(split, component, replicate, layout, pair, arm). Splits: train=1, test=2,
unit tests=9. Components: occupancy=1, pair proposals=2, demonstrations=3,
bootstrap=4. Arm is 0 for matched, 1 for misspecified, and 2 for shared geometry.
For geometry use pair=0 and shared arm=2, with separate component IDs.
Test replicate is always 0. Layout IDs start at 0 within each replicate.
Store spawn keys and derived seeds. Disjoint seed keys are necessary but not
sufficient: check full occupancy fingerprints and ordered pairs for overlap
between train replicates and test; a duplicate is rejected geometrically before
labels. Test must also be internally unique in occupancy. Training replicates
must have distinct occupancies. Do not open test outcomes to change any choice.

## Generator arms and pre-label gates

Matched generator: exact finite-horizon max-ent model with truth [1,0.7,1.2],
distance scale 1, turn/distance ratio 0.7, clearance/distance ratio 1.2.
Misspecified generator: identical model except its clearance feature is
max(0,2-d), not max(0,2-d)**2. Its truth weights remain [1,0.7,1.2].
Learned and fixed-weight models keep squared clearance in both arms. This is
a specific misspecification, not a claim of robustness to arbitrary errors.

Before any study label is sampled, save per-pair numerical-domain and
terminal-margin checks and exact truth completion probabilities in both arms.
Require finite diagnostics, valid masks/features, reachability within T,
score budget <=1000, reference terminal margin >1, and truth completion >=0.90
for every pair. Failures stop the entire study; do not replace a pair based on
these truth-dependent gates. The predefined geometric rejection above is the
only permitted layout/pair rejection. No identifiable-covariance gate is
borrowed from the old recovery study. Save covariance diagnostics on training
geometries, but do not use test covariance to tune anything.

## Models and fitting frozen before labels

1. Soft distance-only: weights [1,0,0], same DP, horizon, and M.
2. Fixed-weight sweep: distance scale s in {0.5,1,2}, turn/distance and
   clearance/distance ratios independently in {0.25,1,4}. All 27 tuples are
   evaluated, with weights [s,s*turn_ratio,s*clearance_ratio], maximum 8.
   Report median, worst, and best-in-hindsight test mean NLL across the 27
   global settings. Do not choose a different oracle weight per layout.
   Also report the setting minimizing training mean trajectory NLL, with
   lexicographic tie-break in (s,turn_ratio,clearance_ratio). This training-only
   selection is the fixed-grid comparator in confirmatory tests.
   Best-in-hindsight is explicitly an optimistic diagnostic, not deployable
   performance and not evidence that the fixed-weight method learned transfer.
3. Behavior cloning: time-independent masked three-action softmax linear policy.
   Each action has a coefficient vector and intercept. Input is current heading
   one-hot, clearance at that action's destination clipped to 2 and divided by
   2, goal direction as sin and cos of bearing relative to current heading,
   and Euclidean goal distance divided by sqrt(162). At the goal use zero goal
   direction. Infinite clearance maps to 1; illegal-action features are zero
   and logits are masked to -inf. Goal padding has probability 1. There is no
   hidden layer, time input, test normalization, or test calibration. Optimize
   mean full-trajectory NLL plus 0.001/2 times squared coefficients including
   intercepts, from zero coefficients. Use float64 deterministic LBFGS with
   strong-Wolfe, 150 iterations, tolerance_grad=1e-9, tolerance_change=1e-14.
   Report convergence by finite objective and raw gradient infinity norm
   <=1e-6, iterations <150. Fixed regularization prevents separation divergence.
4. Learned reward: three_free positive softplus weights, initialized at physical
   [1,1,1], lambda=0. Same LBFGS settings and existing fit-success gates:
   finite objective, raw gradient infinity norm <=1e-6, iterations <150,
   all three weights in [0.001,10]. Report scale and both cost ratios.

Use one CPU Torch thread. No hyperparameter search, retries with alternate
initializations, regularization sensitivity chosen after testing, or seed
selection. Keep every fit and failure. Any failed primary learned or BC fit
makes confirmatory study conclusions incomplete; publish the failures and
available descriptive results without conditioning an overall success claim
on the surviving seeds. Numerical-domain failure instead stops immediately.

## Multi-environment likelihood contract

Represent a batch as environment, fixed feature table, trajectories, and its
fixed M. Within a batch all trajectories share one horizon. Across batches,
different shapes, goals, starts, sample counts, horizons, and M are supported.
The study itself fixes T=33 and M=100. Plan separately per batch with shared w.
Total NLL = sum_e sum_i NLL_ei. Optimize total NLL / total trajectory count
plus lambda/2 times squared free physical weights. This is not a sum of
equally weighted environment means when batch sizes differ, and the penalty
is applied exactly once. Preserve autograd and unfinished terminal scores.
Expose loss_multi and fit_multi in a new module; keep the existing single-env
API and all verified modules unchanged. Reject empty/malformed data and
learnable feature tables in fitting. Return per-batch sizes/horizons/M and
optimizer diagnostics. Domain errors must propagate as NumericalDomainError,
not a successful or skippable fit. Fit calls made by tests are not final fits.

## Metrics, inference, and claim gates

Primary outcome is held-out mean per-trajectory NLL in nats, not per action.
For each model/seed, average trajectories within pair, pairs within layout,
then layouts equally. Include every unfinished trajectory and goal padding.
Report each of the 27 fixed settings as well as median/worst/oracle summaries.

Secondary outcomes use exact occupancy propagation, not selected successful
rollouts. Feature matching error per pair is the Euclidean norm of predicted
minus truth expected cumulative feature counts divided componentwise by
[33,33*pi/2,33]. Always measure using the common squared-clearance feature
table, even for the misspecified generator. Completion is probability mass at
any goal heading after 33 actions. Report predicted completion, truth completion,
and their difference. Mean pairs then layouts. These are descriptive metrics,
with no claims rescued by a secondary metric after primary failure.

For each arm and comparator, paired layout difference D is comparator NLL
minus learned NLL, averaged over the five frozen training replicates. Positive
is better for learning. Confirmatory family comprises 6 contrasts: 2 arms
times distance-only, training-selected fixed grid, and BC. Use 10,000 paired
layout bootstrap resamples with replacement, shared indices across methods,
arms, and training replicates. Report ordinary two-sided 95% percentile CIs
and one-sided Bonferroni lower bounds at alpha=0.05/6. Claim improvement on
one contrast only if its adjusted lower bound >0. Claim transfer against all
baselines in an arm only if all three pass; robustness only if both arms pass.
Report effects in nats even if insignificant. No optional stopping or changes
to N_test. Describe differences smaller than 0.25 nats as below the declared
design target, regardless of significance. Median/worst/oracle and secondary
comparisons are descriptive, not extra confirmatory tests.

Bootstrap unit is the occupancy layout, not pair or trajectory. Separately
report all five training-replicate metrics, paired differences, their mean,
sample SD, minimum and maximum. The layout bootstrap conditions on these
five trained models; it does not estimate training-population uncertainty or
turn five seeds into additional test layouts. Do not pool seed variation with
bootstrap error bars or pretend it shrinks as 1/sqrt(N_test).

## Power simulation protocol frozen before running

No environment, policy, navigation label, or final fit is generated in Step 2.
Simulate six paired NLL-difference outcomes per independent layout as a
multivariate normal with each mean delta=0.25 nats, marginal SD in {0.5,1,1.5},
and correlation 0.5 across contrasts. These are assumptions, not estimates
from historical or held-out study data. Outcomes represent averages over the
five training models; do not divide SD by sqrt(5). Also simulate delta=0 at
SD=1.5 as a familywise false-positive diagnostic.

Use 10,000 independent simulated studies per candidate N in
{64,96,128,192,256,384,512,768,1024}, PCG64 root seed 2026110601.
For each contrast use mean(D)/(sample_SD(D)/sqrt(N)) >
NormalDist().inv_cdf(1-0.05/6). Count both per-contrast rejection rates and
joint success of all six. This normal-reference Monte Carlo is an approximate
power calculation for the eventual bootstrap test, not its exact finite-sample
calibration. Report that limitation; do not claim it simulates bootstrap CIs.

Choose the smallest candidate whose joint-success 95% Wilson lower bound is
>=0.90 in every declared SD scenario. Use z=1.959963984540054 for Wilson bounds.
If none qualifies, report inadequate power within the candidate budget and
stop without enlarging the grid. Report simulated null familywise rejection
rates and Wilson CIs without changing the choice rule after viewing them.
Report per-scenario estimates and Monte Carlo uncertainty. Power does not
guarantee an effect exists or cover training-population uncertainty.

## Artifacts and review boundary

This document and a new power module, command, tests, and output directory are
the only Step 1/2 artifacts. Record simulation configuration, environment
versions, generated outcome summaries, selection, and source fingerprints.
Do not store synthetic statistical draws as navigation demonstrations.
Step 3 adds only a multi-environment inference module and unit tests.
Do not implement or run the final layout generator, BC, baselines, evaluation,
or bootstrap until review. Their definitions above are frozen proposals for
that later implementation. No current result establishes unseen-layout transfer.

## Execution record

Baseline full suite before changes: 311 passed, 1 skipped, 15.71 seconds.
The skipped test is an existing opt-in slow regression. Repowise CLI is not
installed, so verification uses the virtual-environment Python directly.
Power selection and final software verification will be appended after the
predeclared operations, without changing the rules above.
