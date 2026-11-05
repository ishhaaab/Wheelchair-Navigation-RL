# Execution ledger

Plan: Wheelchair preference learning implementation plan, OpenChamber plan
053c56c6-4e3a-40f5-a2c3-c12f04abf33c.
Authority: root SPEC.md and subsequent user instructions.

## Current scope

- [x] Apply all eleven specification review edits.
- [x] Minimal package and environment setup needed for Task 2.
- [x] Task 2 environment tests, RED then GREEN.
- [x] Task 2 feature tests, RED then GREEN.
- [x] Full available suite and Task 2 review.

## Rulings

- Latest user instruction authorizes Task 2 with SPEC.md uncommitted. Leave
  all work uncommitted this turn, avoiding accidental inclusion of the spec.
- There is no repository history or existing source tree. Work in the requested
  standalone directory on an unborn feature branch rather than creating a
  worktree, which requires an initial commit. No other work is present.
- Perform only the setup required to run Task 2, and stop after Task 2, matching
  the narrowed review request. Planner, fitting, and recovery remain deferred.
- The saved plan still says 30 bootstrap refits and general M=30. Revised
  SPEC.md overrides these with 100 refits, 80-success reliability, per-fixture
  penalty validation, and the other approved review constraints. Do not execute
  the outdated later-task defaults.

## Interface preflight

- Task 2 env -> features: integer state triples and transition tables select
  destination clearance; goal-source masking zeros subsequent costs.
- Task 2 -> future planner: torch.long next_state and torch.bool legal/is_goal
  on CPU; features are torch.float64 with shape [states, actions, features].
- Task 2 -> future sampler: action indices 0/1/2 match the spec; exactly one
  legal goal continuation and safe indices for illegal actions.

## Verification evidence

- `git ls-remote --symref https://github.com/ishhaaab/Wheelchair-Navigation-RL.git`
  succeeded and advertised no refs before Git initialization.
- `git init -b phase1-environment` and `git remote add origin
  https://github.com/ishhaaab/Wheelchair-Navigation-RL.git` succeeded. No commits,
  staging, or pushes were performed.
- `python -m venv .venv` and `.venv\Scripts\python.exe -m pip install -e ".[test]"`
  succeeded. Python 3.13.13, torch 2.15.0+cpu, numpy 2.5.4, pytest 9.2.0,
  matplotlib 3.11.2. `.venv\Scripts\python.exe -m pip check` reported no broken
  requirements.
- `.venv\Scripts\python.exe -m pytest tests/test_env.py -q` initially failed
  collection because `wnp.env` did not exist. After implementation, 51 passed.
- `.venv\Scripts\python.exe -m pytest tests/test_features.py -q` initially
  failed collection because `wnp.features` did not exist.
- `.venv\Scripts\python.exe -m pytest -q` after feature implementation passed
  all 68 tests in 2.74 seconds. No tests for deferred phases exist yet.
- `git diff --cached --stat` was empty. `git status --short --untracked-files=all`
  showed only the nine intended source/spec/config/test/ledger files. Environment
  and test caches are ignored. SPEC.md remains untracked and uncommitted.

## Specification change summary

1. Predeclared exact completion >=0.90, covariance minimum eigenvalue >=0.1,
   condition number <=100, and mandatory documented revision before sampling
   if checks fail.
2. Fixture-specific M with code-checked optimal finishing cost margin, not a
   universal M=30. Reachability/horizon domain and margin=1 are documented.
3. Explicit finite-objective, gradient, iteration, and physical-weight fit gates;
   named failures and no clipping.
4. Per-weight absolute and two-weight Euclidean errors, per-seed values,
   mean/sample standard deviation, and log-log slopes with small-seed caveat.
5. Hessian versus negative expected-feature finite differences and single-action
   states in Test B.
6. 100 bootstrap refits and reliability threshold of 80 successes.
7. Second exhaustive eight-heading fixture with diagonal and corner checks.
8. Covariance-predicted asymptotic standard deviation versus seed spread at
   n=2048, plus descriptive truth-containing interval counts per n.
9. Regularization sensitivity at both n=32 and n=2048.
10. Demo arrival_step with None for unfinished and 0 for goal starts.
11. A shared verification-function implementation for tests and CLI script.

## Task 2 final review and fix pass

- A fresh read-only reviewer reported no Critical issues and confirmed the
  core environment/features against an independent exhaustive movement oracle.
  Its Important finding was missing seed validation; its Minor finding was
  possible float64 overflow for large finite d_safe.
- Final Ruling: regrade clearance overflow as Important because accepted inputs
  could inject infinite features into future planner computations. Reject
  unrepresentable computed costs, without clipping. Cost if too strict: some
  unusually large d_safe inputs are rejected instead of returning infinities.
- Both findings were reproduced before changes. The targeted regression command
  `.venv\Scripts\python.exe -m pytest tests/test_env.py tests/test_features.py -q
  -k "seed or unrepresentable or representable"` gave 9 failures and 7 passes.
  None/bool/mutable seed acceptance and overflow occurred in the reported
  production paths; invalid float/bool types also leaked NumPy exceptions.
- Fixed seed validation using the existing integer validation pattern before
  constructing a NumPy generator. Accept nonnegative Python/NumPy integers;
  reject missing, boolean, float, negative, and mutable RNG/SeedSequence inputs.
  `.venv\Scripts\python.exe -m pytest tests/test_env.py -q` then passed 64 tests.
- Fixed clearance overflow by converting arithmetic FloatingPointError into a
  clear ValueError. A large representable cost is tested to remain unclipped.
  `.venv\Scripts\python.exe -m pytest -q` then passed 83 tests in 2.59 seconds.
- Review exclusions ruled on: later-phase implementation is outside current
  Task 2 scope, so no planner or statistical claims are warranted. Historical
  RED/GREEN evidence is recorded above in this session rather than Git history,
  because commits are prohibited. Visual verification does not apply because
  Task 2 produces tables rather than a visual interface. Cost of these scope
  decisions: later-phase behavior remains unverified until implemented.
- Task 2: complete, with the explicit user override that no commit is made.
  No deferred review findings remain. SPEC.md stays uncommitted.

## Follow-up specification review

- Applied the seven follow-up edits without committing SPEC.md. Existing
  per-weight/Euclidean error metrics, seed summaries, log-log slopes, and
  80-of-100 bootstrap cutoff were retained and clarified where requested.
- Replaced the unreachable within-horizon corner check with the reachable
  `(1,1,NE)` at t=2, and added `(1,1,NW)` at t=4. The t=1 occupied-destination
  block is explicitly distinguished from corner cutting. Future exhaustive
  verification must assert masks and exclusion of blocked moves.
- Clarified the same single-action fixture for Hessian verification, convergence
  gradients of the actually optimized objective, the reference-weight-only
  limitation of the M check, and descriptive-only uncertainty comparisons.
- Task 2 is already complete; do not reimplement it or interpret this request
  as authorization to start Task 3. Re-run the existing suite for regression
  evidence and check the corrected fixture paths against current transitions.
- Follow-up verification: `.venv\Scripts\python.exe -B -m pytest -q -W error
  -p no:cacheprovider` passed all 83 tests in 2.86 seconds. A direct fixture
  check confirmed the t=1, t=2, and t=4 states are reached by the declared legal
  action prefixes and forward is masked at each. It also confirmed a legal SE
  diagonal from `(1,1)` to the goal remains available. No exhaustive distribution
  or Hessian checks were run; those still belong to the deferred planner task.

## Task 3 start

- [x] Planner and trajectory edge-case tests, observed RED.
- [x] Both exhaustive Test A fixtures and expected counts, observed RED.
- [x] Differentiable all-state backward pass, scoring, sampling, enumeration.
- [x] Shared verification functions and generated verification artifact.
- [x] Full suite and independent Task 3 review; stop before Task 4.

Task 3 Ruling: latest instruction authorizes Task 3 only, overriding earlier
Task 2-only scope. Keep the feature-branch directory and leave all work
uncommitted, particularly SPEC.md. No Task 4 gradient checks, fitting, bootstrap,
or recovery work is authorized this turn.

Task 3 interface preflight: GridMDP supplies CPU torch.long successors and
torch.bool masks; feature_table supplies finite torch.float64 [S,3,3] tensors.
PlanResult will expose values, log_policy, horizon, terminal_penalty, and its
environment identity. Trajectory will be an immutable record of integer state
and action tuples plus arrival_step. All scoring validates the complete record
and shared horizon before building a single DP per batch.

Task 3 Ruling: the shared numerical verification functions live in
src/wnp/verification.py and can run as a module to generate results/verification.json.
This satisfies Task 3's artifact requirement without beginning Task 4's
verify_planner.py/gradient-check work. The later script must import these same
functions. Terminal-margin checks are fixture diagnostics at reference weights,
not constraints imposed during every plan call or a hard completion condition.

Task 3 baseline: `.venv\Scripts\python.exe -B -m pytest -q -W error
-p no:cacheprovider` passed 83 tests in 2.61 seconds before any Task 3 code.

### Task 3 execution evidence

- Wrote both new test files before the production modules. Running
  `.venv\Scripts\python.exe -B -m pytest tests/test_soft_dp.py tests/test_demos.py
  -q -p no:cacheprovider` failed collection twice because wnp.demos did not
  exist. This established the missing implementation before adding it.
- After demos/soft_dp implementation, `.venv\Scripts\python.exe -B -m pytest
  tests/test_demos.py -q -W error -p no:cacheprovider` passed 24 tests.
- Before adding shared exhaustive verification, the soft-DP test file failed
  collection because wnp.verification did not exist. After implementing the
  checks, the complete suite passed 140 tests in 3.15 seconds.
- Added a fractional-penalty regression after noticing that scalar torch.where
  inferred float32 before the float64 cast. The targeted run failed with
  -0.12345679104328156 versus the declared -0.123456789012345. Constructing the
  terminal tensor with weights.new_full fixed the actual rounding source.
- Strengthened the likelihood-gradient test so its primary record batch leaves
  a disconnected nongGoal component unreachable, then separately tested a
  record starting in that component. All-state V0 gradients are checked one
  state at a time, not just in an aggregate that could hide a bad state.
- `.venv\Scripts\python.exe -B -m pytest -q -W error -p no:cacheprovider`
  passed 141 tests in 3.07 seconds after that fix. No warnings were permitted.
- `.venv\Scripts\python.exe -B -m wnp.verification --output
  results/verification.json` succeeded, generating machine-computed errors,
  counts, configurations, margin diagnostics, versions, and corner/occupancy
  coverage. The artifact labels Task 4 checks as NOT RUN.
- Fresh read-only Task 3 review dispatched after the green suite. Its findings,
  any necessary regression fix pass, and final verification will be recorded
  below. No commits or pushes have been made.

### Task 3 review fix pass and completion

- The read-only review confirmed declared fixtures and ordinary-scale behavior,
  including 216 additional start/horizon pairs and 1,748 trajectory gradients.
  It found one Important boundary issue: finite per-action features/rewards
  do not ensure cumulative feature counts or required weight derivatives fit
  in float64. No Critical or Minor findings were reported.
- Reproduced the exact disconnected three-cell fixture with d_safe=1e154,
  weights [1, 0.7, 1e-308], and T=2. Action features and DP values were finite,
  but total clearance was inf, trajectory likelihood was -inf, and the V0
  clearance-weight gradient was -inf. The overflow occurs before weighting.
- Added regression cases requiring explicit ValueError from plan, likelihood,
  and expected_features for that unsafe horizon/feature combination. The
  targeted run failed all three before the fix. A fourth test checks that
  representable single-step extreme features still yield finite values,
  likelihoods, and weight gradients; it passed before and after the fix.
- Task 3 Ruling: use the conservative all-state bound feature <= float64_max/T
  for T>0, checked without multiplying large numbers. This prevents the
  reproduced cumulative-count/derivative overflow before graph execution.
  Apply it to alternate feature tables passed to expected_features too.
  Cost: it may reject extreme features that cannot actually recur T times.
  Keep the domain restriction explicit in SPEC.md and docstrings; never clip.
- `.venv\Scripts\python.exe -B -m pytest -q -W error -p no:cacheprovider`
  passed all 145 tests in 3.04 seconds after the regression fix. This is
  62 Task 3 tests plus the existing 83 environment/feature tests.
- Review exclusions ruled on: Task 4 and later numerical checks remain deferred;
  no claims about them are made. RED/GREEN ordering is evidenced by this ledger
  and session commands rather than Git history. No staging, commit, or push is
  required by this scoped request, and none is performed. Cost of deferral:
  Task 4 numerical-gradient and optimization claims remain unsupported.
- Re-ran `.venv\Scripts\python.exe -B -m wnp.verification --output
  results/verification.json` after the fix. Both shared exhaustive checks passed
  with unchanged probability/expected-feature errors and regenerated evidence.
- Task 3: complete after one review fix pass, no deferred findings. Stopped
  before Task 4. All work, including SPEC.md, remains uncommitted.

## Tasks 2–3 checkpoint authorization

- On 2026-11-06 the user authorized a local checkpoint commit of the verified
  implementation, tests, package setup, ledger, and generated evidence. This
  supersedes the earlier blanket no-commit ruling; SPEC.md must still remain
  uncommitted. No push or Task 4 work is included in this authorization.
- Re-inspected the feature branch and remote. `git ls-remote --symref origin`
  succeeded and advertised no remote refs. No unrelated working files appeared.
- Fresh `.venv\Scripts\python.exe -B -m pytest -q -W error -p no:cacheprovider`
  passed all 145 tests in 3.61 seconds.
- Fresh `.venv\Scripts\python.exe -B -m wnp.verification --output
  results/verification.json` passed both shared exhaustive checks and
  regenerated the artifact with unchanged numerical discrepancies.
- Stage only explicitly named project files; verify SPEC.md is absent from
  the index and environments/caches remain ignored before committing.
