# GOAL MODE: Production-qualify the complete four-Weyl Teukolsky solver

Repository:

`https://github.com/HengruiZhu99/teuk-kokkos.git`

Current reviewed starting commit:

`6f1db976fb96863554ad3f3269479e1e2a64edc7`

Pinned Kokkos submodule at the reviewed starting point:

`3ec81abe1816109f6f62ac48cef41921f91a4d00`

The previous stop-and-transfer instruction is explicitly revoked.

Continue autonomously across milestones. Do not stop merely to request another
remote review, after producing a plan, after a successful build, or after one
validation case. Commit and push coherent milestone checkpoints, but continue
until the production definition below is satisfied or a genuine external
blocker prevents further work.

If current `main` has advanced beyond the commit above, do not reset or discard
newer work. Record the actual HEAD, compare it with the reviewed starting
commit, audit every intervening change, and continue from the newest clean
state.

The user will commission a fresh independent review after this goal finishes.
Leave the repository easy to audit.

---

## 1. Mission

Bring `teuk-kokkos` to a genuinely production-qualified, human-readable,
performance-portable solver for the complete passive four-Weyl perturbative
system

\[
\Psi_4^{(1)},\qquad
\Psi_0^{(1)},\qquad
\Psi_4^{(2)},\qquad
\Psi_0^{(2)}
\]

on Kerr, with particular emphasis on robust long-duration near-extremal and
high-spin evolution.

The intended production dependency graph is

```text
spin -2 first-order Teukolsky state
        |
        v
first-order ORG reconstruction
        |
        +----------------------+
        |                      |
        v                      v
corrected spin -2          local Route-B
quadratic source           curvature data
        |                      |
        v                      v
spin -2 second-order       corrected raw spin +2
Teukolsky state            quadratic source
                               |
                               v
                         passive spin +2
                         Teukolsky companion
```

All coupled quantities must be evaluated at the same classical RK stage and
the same stage time. The spin `+2` companion must never feed back into the
primary spin `-2` evolution.

Production qualification requires correctness, convergence, restart/replay,
high-spin stability, current accelerator execution, and honest scope. It does
not mean merely that the code compiles or that standalone pointwise fixtures
pass.

---

## 2. Authoritative reading order

Before modifying scientific code, read the following current-tree files in
this order:

1. `docs/MAIN_THREAD_TERMINATION_REPORT_2026-08-12.md`
2. `docs/PRODUCTION_BASELINE.md`
3. `docs/PRODUCTION_PLUS2_CONVENTIONS.md`
4. `docs/EXTERNAL_REVIEW_PROMPT_2026-08-12.md`
5. `docs/EXTERNAL_REVIEW_REMEDIATION_REPORT_2026-08-12.md`
6. `docs/EXTERNAL_REVIEW_HANDOFF_2026-08-12.md`
7. `docs/PLUS2_FIELD_AND_TETRAD_CONVENTIONS.md`
8. `docs/PLUS2_FORMALISM_GATE.md`
9. `docs/PLUS2_SOURCE_COMPACTIFICATION.md`
10. `docs/PLUS2_VALIDATION_AND_TSI_AUDIT.md`
11. `docs/PLUS2_ROUTE_B_LOCAL_CURVATURE_BLOCKER.md`
12. `docs/PLUS2_ROUTE_B_DERIVATIVE_TOWER_BLOCKER.md`
13. `docs/PLUS2_ROUTE_B_ANGULAR_JET_COORDINATOR.md`
14. `docs/D10_5_SBP_PROVENANCE_AND_QUALIFICATION.md`
15. `docs/FOURTH_ORDER_QUALIFICATION.md`
16. `docs/BOUNDARY_SAT_BLOCKER.md`
17. `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`
18. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
19. `PLUS2_SOURCE_TERM_LEDGER.csv`
20. `PLUS2_SOURCE_INPUT_MANIFEST.csv`
21. `PLUS2_SOURCE_NORMALIZATION_LEDGER.csv`
22. `SOURCE_TERM_LEDGER.csv`
23. `equation_spec_v2.yaml`
24. `plus2_equation_spec_proposal.yaml`

`plus2_equation_spec_proposal.yaml` remains a proposal. It is not production
authority when it disagrees with later derivations or the current production
contract.

Root historical checksum/manifests and old reports are commit-scoped evidence,
not current-tree integrity evidence.

Do not assume any report is correct merely because it is checked in. Verify
material claims from derivations, source, generators, independent oracles, and
executable tests.

---

## 3. Scientific invariants

Preserve these conventions unless a new independent derivation and regression
test prove a defect:

```text
metric signature             +---
first-order gauge             outgoing radiation gauge
background tetrad             rotated Kinnersley, gamma=0
perturbative convention       A=A0+epsilon A1+epsilon^2 A2+...
                              with no factorial in A2
azimuthal dependence          exp(+i m phi)
```

Signed sharp is always

\[
X_m^\sharp=\overline{X_{-m}},
\]

never same-mode conjugation.

Preserve

\[
\Psi_4^{(n)}=R Z_-^{(n)},
\]

\[
\Psi_0^{(n)}
=
\frac{R^5}{(L^2-i aR\cos\theta)^4} Z_+^{(n)},
\qquad n\in\{1,2\},
\]

and keep distinct:

```text
raw code-tetrad Psi0
evolved Z_plus
Z0_source = Psi0/R^5
Z1 = Psi1/R^4
local metric-curvature T0[h]
TSI reconstruction
effective Einstein-source variables
```

The raw physical `Psi0` vanishes as `O(R^5)` at smooth future null infinity.
The nontrivial endpoint object required by the source is its leading peeling
coefficient `Z0_source`, not the trivially zero raw endpoint value.

Preserve the corrected evolved spin `+2` forcing

\[
D=L^4+a^2R^2\cos^2\theta,
\qquad
d_-=L^2-i aR\cos\theta,
\]

\[
\boxed{
\mathcal F_+
=
2D\,d_-^4\left(\frac{S_0}{R^7}\right).
}
\]

`S0/R^6` is a separate raw diagnostic. Never pass it to the evolved equation
and never obtain `S0/R^7` by numerically dividing a computed `S0/R^6` by `R`.

Preserve the cancellation-safe identity

\[
\frac{5b-4\rho_0-\bar\rho_0}{R}
=
\frac{
ia\cos\theta\,
\left(L^2-2MR+a^2R^2/L^2\right)
\left(3L^2+5iaR\cos\theta\right)
}{
2\left(L^4+a^2R^2\cos^2\theta\right)^2
}.
\]

Preserve the corrected Bianchi-5 term

\[
\Delta_5 Z_0
=
\eth_4 Z_1
-R\mu_0 Z_0
-4R\tau_0 Z_1
+3\Sigma\psi_{20},
\]

not `3 Sigma H`.

Preserve the corrected linear `Psi0` and `Psi1` sign choices, the full complex
phase in the Berens Eq. 2.44 relation, and

\[
\lambda_+
=
A_{\rm qnm}-2am\omega+(a\omega)^2.
\]

Keep every existing negative control that rejects the old formulas.

Do not change the production spin `-2` continuum equations without an
independent proof of a defect. Shared numerical infrastructure may be improved,
but the disabled-`plus2` production trajectory must remain bitwise identical
where mathematically possible, or pass a predeclared roundoff-equivalence and
convergence gate.

---

## 4. Dependency and readability requirements

Kokkos remains the only mandatory third-party production dependency.

Do not add mandatory production dependencies such as:

```text
Boost
Eigen
BLAS/LAPACK
FFTW
HDF5
NetCDF
yaml-cpp
JSON libraries
fmt
spdlog
CLI11
GoogleTest
Catch2
MPI
```

Development-only Python dependencies are acceptable for symbolic and numerical
oracles. Maintain an exact audit lock with hashes for the supported audit
environment.

Keep the C++ readable to a numerical relativist. Prefer named intermediate
terms corresponding to equations and ledgers. Do not hide the source behind
macros, template metaprogramming, or a generic symbolic-expression framework.

Do not build a generic PDE framework.

---

# 5. Long-horizon multi-agent workflow

The coordinator must actively use subagents so its context is not flooded by
implementation details and test logs.

Use at most five direct child agents concurrently. Use isolated worktrees and
focused file ownership. Do not allow uncontrolled recursive subagent spawning.

Keep these roles active or rotate them only at milestone boundaries:

## Agent A — independent mathematical auditor

Responsibilities:

- independently rederive any formula touched by implementation agents;
- audit signs, factors, spin/boost weights, radial powers, sharp operations,
  source normalization, and endpoint limits;
- maintain symbolic and arbitrary-precision negative controls;
- review proposed stable high-spin coefficient rewrites;
- review the distinction between physical and artificial-dissipation
  tangents in Route B;
- do not own broad runtime refactors.

## Agent B — scri/Route-B numerical-method specialist

Responsibilities:

- production-robust extraction of `Z0_source`, `Z1`, their first and second
  tangents, and all eight source derivative slots;
- endpoint conditioning, peeling residuals, dual estimators, and immutable
  endpoint qualification;
- high-resolution tests through at least `N_R=513`;
- source work-band and angular projection order;
- no checkpoint or CLI ownership.

## Agent C — runtime/provenance/integration specialist

Responsibilities:

- checkpoint authority;
- primary and source receipts;
- four-field coordinator;
- common-stage RK integration;
- replay/restart;
- four-Weyl output;
- runtime configuration and fail-closed enablement;
- preserve primary no-feedback semantics.

## Agent D — high-spin/stability/diagnostics specialist

Responsibilities:

- numerically stable near-extremal Kerr parameterization;
- timestep/stability qualification;
- horizon diagnostics;
- radial-resolution strategy;
- near-extremal and exact-extremal tests;
- performance and resource measurements after correctness.

## Agent E — dedicated build/test monitor

This agent is not an implementation agent unless explicitly reassigned.

Responsibilities:

- maintain separate clean build trees for Serial, OpenMP, and available
  accelerator backends;
- run focused tests after each integration commit;
- run a full clean audit-enabled Serial suite at every milestone;
- monitor long-running test jobs;
- capture full logs under `/tmp/teuk-goal-logs/<commit>/<preset>/`;
- return only concise summaries to the coordinator:
  - exact commit,
  - command,
  - result,
  - elapsed time,
  - exact test counts,
  - first failing test,
  - path to the complete log;
- never paste complete compiler or test logs into the coordinator context;
- never weaken tests or edit physics code merely to make a job green.

If only four child slots are available, always retain the test monitor and
rotate Agents A–D as milestones change.

## Context-control protocol

Create and maintain:

```text
docs/PRODUCTION_STATUS.md
docs/SCIENTIFIC_DECISIONS.md
docs/AUTONOMOUS_TEST_STATUS.md
```

Keep each concise.

Every implementation agent must conclude a work unit with a summary of no more
than approximately 50 lines containing:

```text
branch/worktree
commit SHA
files changed
mathematical choice
tests run
known failures
recommended next integration action
```

Full derivations and logs go into focused repository files or `/tmp`, not the
coordinator conversation.

The coordinator alone merges to `main`.

After merging an agent branch, the test-monitor agent must run the relevant
focused tests before the coordinator proceeds.

---

# 6. Failure and evidence policy

Never:

- weaken a tolerance because a result is red;
- delete a failing test;
- replace a failed pointwise/endpoint test with a friendlier aggregate norm;
- silently choose a new refinement window after seeing results;
- remove difficult signed modes;
- prohibit fine grids merely because the current endpoint method fails;
- reinterpret historical failed evidence;
- manufacture missing endpoint values as zero;
- call compilation accelerator qualification;
- call an interior fixture endpoint qualification;
- call a short finite run a near-extremal stability result.

When a method fails:

1. preserve the exact failure;
2. formulate competing explanations;
3. use reduced tests to distinguish them;
4. repair the numerical formulation;
5. rerun the original gate;
6. add a regression;
7. continue.

---

# 7. Milestone 0 — independently re-establish the exact current baseline

Before editing code:

```bash
git status --short --branch
git rev-parse HEAD
git submodule status
git diff --check
```

Record the actual current commit and compare it with
`6f1db976fb96863554ad3f3269479e1e2a64edc7`.

The current reviewed head received only an incremental final build after its
last edit. The older `396/396` result is pre-final-WIP evidence.

Run a clean exact-head baseline:

```bash
python3 -m venv /tmp/teuk-production-goal
/tmp/teuk-production-goal/bin/pip install --require-hashes \
  -r requirements-audit-lock.txt

cmake --preset serial \
  -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-production-goal/bin/python

cmake --build --preset serial --clean-first --parallel
./build/serial/teuk_tests
ctest --preset serial --output-on-failure
git diff --check
```

Record compiler, CMake, Python, dependency, Kokkos, test-count, and timing
information.

Add or update a CPU GitHub Actions workflow if the repository still lacks
continuous exact-commit verification. It must initialize the Kokkos submodule
and run the same audit-enabled Serial suite.

Do not proceed past unexplained baseline failures.

---

# 8. Milestone 1 — close checkpoint and source authority completely

Finish the incomplete version-4 provenance work before changing endpoint
numerics.

## 8.1 Make privileged checkpoint encoding non-bypassable

The current private token is insufficient if public header-level `detail`
functions can mint or consume a pipeline-bound checkpoint.

Enforce the boundary through the type system:

- only a production four-field/pipeline coordinator may issue or consume a
  pipeline-bound companion checkpoint;
- low-level validation codecs must write an explicitly unbound schema;
- no public `detail` function may accept a forgeable boolean, enum, string, or
  aggregate token to obtain production authority;
- use private constructors, private implementation types, friendship, or a
  non-header implementation boundary as appropriate;
- add compile/runtime negative tests proving that raw code cannot mint or
  restore a production-bound artifact.

## 8.2 Replace pointer identity with a complete source-configuration identity

A pointer to `Plus2LiveSourceComposition` is not sufficient scientific
provenance.

Define an immutable, typed source configuration identity derived from the
actual constructed source objects. Bind at least:

```text
source implementation and version
S0/R7 normalization capability/version
Route-B provider implementation/version
endpoint method/version
parent signed-mode registry
target signed-mode registry
ell_max_parent
ell_max_source_work
ell_max_companion
theta grid and representation
radial grid and radial scheme
M, a, L
sharp convention/version
ordered-pair construction/version
source activation semantics/version
```

Prefer a deterministic content digest over pointer identity.

Only the concrete qualified Route-B/live-source object may issue this receipt.
Validation callbacks remain usable but can never issue production authority.

## 8.3 Bind authority to the complete accepted trajectory

A run must not use an authoritative source for one step, then substitute
manual forcing or a validation callback, and later write a production
checkpoint.

Maintain a monotonic trajectory-authority state. It must be permanently
invalidated by:

```text
validation-only source callback
manual forcing injection
stale generation
source configuration change
registry/band change
failed endpoint qualification
missing sharp partner
wrong normalization
external mutation of authoritative state
```

Production checkpoint save requires a still-valid trajectory authority that
matches the accepted step and time.

## 8.4 Extend the primary checkpoint receipt

The current verified primary receipt binds exact bytes, time, step, and state
count but must also bind the primary spectral and physical semantics:

```text
signed mode registry
parent ell band
theta count and angular representation
radial geometry
radial discretization
field/storage schema
M, a, L
reduction mode and damping
perturbative/scaling convention
source-activation state where relevant
```

Derive these from validated primary metadata rather than accepting companion
claims independently.

The companion must cross-check the primary receipt before mutating either
state.

## 8.5 Complete progress and codec regressions

Test fail-before-callback and fail-before-mutation for:

```text
wrong step with correct time
wrong time with correct step
wrong dt
time != step*dt
primary ahead of companion
companion ahead of primary
repeated resumed step
skipped resumed step
invalid activation time
last_eligibility_time outside its sentinel/time range
raw save -> raw load
raw save -> production load
production save -> raw load
production save -> production load
old v1/v2/v3 formats
forged source receipt
forged primary receipt
```

## 8.6 Atomic paired checkpoint transaction

Before production enablement, add a four-field checkpoint manifest that
atomically identifies the primary and companion artifacts and their content
digests. A half-published pair must not be selectable as a valid restart.

### Milestone-1 gate

Run the complete clean Serial suite and all negative tests. Do not proceed
until this milestone is fully green.

---

# 9. Milestone 2 — audit and stabilize all high-spin Kerr algebra

Create a dedicated high-spin mathematical audit before long runs.

## 9.1 Stable horizon quantities

Avoid cancellation in

\[
\sqrt{M^2-a^2}
\]

near extremality. Use and test an equivalent stable form such as

\[
\delta=\sqrt{(M-|a|)(M+|a|)},
\qquad
r_+=M+\delta,
\qquad
r_-=M-\delta,
\]

with appropriate handling of signed `a`.

Use stable identities where appropriate:

\[
r_+^2+a^2=2Mr_+,
\]

\[
\kappa=\frac{\delta}{2Mr_+},
\qquad
\Omega_H=\frac{a}{2Mr_+}.
\]

Audit every use of `M*M-a*a`, `r_+-r_-`, and equivalent subtractions.

## 9.2 Factor horizon-vanishing radial polynomials

Where it improves conditioning, evaluate

\[
1-\frac{2MR}{L^2}+\frac{a^2R^2}{L^4}
=
\left(1-\frac{r_+R}{L^2}\right)
\left(1-\frac{r_-R}{L^2}\right)
\]

rather than relying only on expanded cancellation near the horizon.

Enforce analytically exact endpoint zeros where the continuum coefficient is
known to vanish.

The mathematical-audit agent must compare the stable and original formulas at
high precision over:

```text
a/M = 0
±0.6
±0.99
±0.999
±0.999999
exact |a|=M where supported
```

Add randomized host/device comparisons.

## 9.3 Exact extremality is a separate regime

Do not treat `a/M=1` as merely another floating-point near-extremal input.

Either:

- derive and qualify exact-extremal coefficient limits and explicitly support
  them; or
- keep exact extremality fail-closed with a clear research-only override.

Near-extremal qualification must not be silently presented as exact-extremal
qualification.

## 9.4 Coefficient regularity

Verify over the full compact exterior:

```text
finite background coefficients
nonzero C_T
correct characteristic endpoint limits
correct signs and complex phases
absence of accidental division by kappa
negative-spin symmetry
```

Any algebraic rewrite requires exact symbolic equality and an executable
high-precision regression.

---

# 10. Milestone 3 — production-robust scri curvature coefficients

This is the largest remaining numerical-method task.

The production problem is not the raw value `Psi0(R=0)`, which vanishes for
the smooth peeling branch. The required quantities are the finite leading
coefficients

\[
Z_{0,\mathrm{source}}
=
\lim_{R\to0}\frac{\Psi_0}{R^5},
\qquad
Z_1
=
\lim_{R\to0}\frac{\Psi_1}{R^4},
\]

their first and second time tangents, and the eight source derivative slots.

The current fixed positive-node quotient/extraction machinery is a diagnostic
oracle, not a production method at fine binary64 resolution.

## 10.1 Compare multiple endpoint strategies before selecting one

The endpoint agent and mathematical agent must independently investigate at
least two of these routes:

### Route E1 — asymptotic coefficient evolution/constraint closure

Derive the regular `R=0` expansion of the Route-B curvature and relevant
Bianchi/Ricci identities.

Use boundary identities algebraically at scri without promoting the weakly
hyperbolic Route-A bulk transport.

Evolve or reconstruct the necessary asymptotic coefficients directly.

### Route E2 — fixed-physical-width orthogonal fit

Use a precomputed, scaled orthogonal-polynomial least-squares/QR fit over a
fixed physical `R` width rather than a fixed number of shrinking nodes.

Use more samples under refinement while keeping the physical window and
conditioning controlled.

Use two independent window/degree choices as a dual estimator.

### Route E3 — direct coefficient-space radial series

Propagate normalized radial series through the Route-B point algebra and
derive the required coefficients before lower-power cancellation.

If grid data must initialize the series, use a condition-controlled estimator,
not a fixed stencil with `h^{-k}` noise growth.

A hybrid of E1–E3 is allowed if independently justified.

Do not select a method only because it makes one existing test green.

## 10.2 Physical tangent versus artificial dissipation

Independently decide and document which tangent belongs in the continuum
curvature/source formula.

Do not automatically assume that artificial numerical dissipation must enter
the physical NP time derivatives.

A strong candidate discretization is:

```text
evolution tangent = physical Teukolsky/reconstruction RHS + numerical dissipation
curvature/source tangent = physical continuum RHS evaluated at the same RK state
```

because the quadratic source is a continuum functional while dissipation is a
vanishing numerical modification.

Alternatively, include dissipation only if a derivation proves that this is
the desired consistent modified-equation discretization.

Whichever choice is selected must demonstrate:

```text
common-stage fourth-order time convergence
spatial convergence with nonzero dissipation
convergence to the zero-dissipation continuum result
no hidden endpoint derivative tower of dissipation error
```

This decision must be reviewed by Agent A and recorded in
`docs/SCIENTIFIC_DECISIONS.md`.

## 10.3 Immutable endpoint qualification

Define a typed, immutable endpoint qualification containing at least:

```text
method and version
physical grid identity
M, a, L
signed registry
angular bands
radial scheme
fit/series order and window
condition estimate
forward-error estimate
peeling residuals
dual-estimator disagreement
oracle/qualification-matrix digest
valid radial-resolution range
```

Every stage must carry a fresh generation-bound endpoint result. The source
must fail globally if any required parent, sharp partner, coefficient,
residual, or certificate is invalid.

Remove the current practice of self-asserting
`independently_qualified_scri_coefficients=true`.

## 10.4 Predeclare a two-regime error gate

Before viewing new results, define:

### Truncation-dominated regime

Require the designed convergence order.

### Roundoff/conditioning-dominated regime

Require:

```text
absolute and relative error below a predeclared ceiling
no catastrophic growth under refinement
condition estimate below its bound
agreement of two independent estimators
bounded peeling residuals
```

Do not demand meaningless monotone ratios once the independent oracle shows
that roundoff dominates.

## 10.5 Required endpoint matrix

At minimum test:

```text
spins:       0, ±0.63, ±0.99, ±0.999, ±0.999999
m sectors:   both signs independently
radial N:    17, 33, 65, 129, 257, 513
locations:   scri, interior controls, horizon
fields:      all six Z0/Z1 value/tangent fields
derivatives: all eight source slots
```

Use arbitrary-precision coordinate-Weyl or another genuinely independent
oracle.

Retain the old positive-node extractors and `N=65` failures as diagnostics and
historical negative evidence.

---

# 11. Milestone 4 — complete and qualify the finite-dimensional source graph

## 11.1 Source work bands

Introduce explicit runtime/configuration distinctions:

```text
ell_max_parent
ell_max_source_work
ell_max_companion
Ntheta_source
```

The current source applies variable-coefficient outer Kerr operators after
intermediate projection. Since projection and multiplication by rational Kerr
coefficients do not commute exactly at finite band, either:

- evaluate the complete source in a larger work band and project only the final
  forcing to the companion band; or
- prove the current ordering equivalent to the intended Galerkin method and
  demonstrate convergence.

Do not silently identify all three bands.

The same issue must be audited for the existing spin `-2` quadratic source.
Change its finite-dimensional ordering only after Agent A proves the numerical
defect and new convergence tests preserve the continuum equations.

## 11.2 Ordered-pair completeness

Preserve deterministic ordered pairs

\[
(m_1,m_2)\to m_t=m_1+m_2.
\]

Do not symmetrize pair order unless an exact identity applies to the specific
term.

Require complete daughter registries by default.

## 11.3 Source diagnostics

Retain individually inspectable families and ordered-pair contributions.

Use active-mode-aware, SBP-radial and Gauss-angular weighted norms. Do not
normalize first-order diagnostics by target-only zero mode slots or use a
simple unweighted collocation-point RMS as a physical norm.

## 11.4 Independent source validation

Retain and extend:

```text
ordinary-NP versus compact-GHP oracle
negative sign/factor controls
signed sharp controls
amplitude scaling
source-work-band convergence
quadrature convergence
scri and horizon limits
host/device parity
```

---

# 12. Milestone 5 — build the real four-field common-stage runtime

Create one production coordinator that owns:

```text
primary SpatialPipeline
Route-B physical-curvature provider
qualified live spin +2 source
spin +2 companion pipeline
source and endpoint authority
paired checkpoints
four-Weyl output
```

Do not run two independent RK4 integrators sequentially.

At every RK stage:

1. construct the primary stage state;
2. evaluate the physical primary/reconstruction stage graph;
3. form the qualified Route-B curvature coefficients;
4. form every signed ordered-pair source family;
5. evaluate the complete `S0/R^7`;
6. form the spin `+2` coordinate forcing;
7. evaluate the spin `+2` companion RHS;
8. combine both states with the same RK tableau.

The accepted source activation state must remain immutable during one complete
step.

Expose validation-only callback APIs separately from production APIs.

Prevent mutable external access from silently altering an authoritative
companion trajectory. If mutable views remain necessary for tests, using them
must permanently mark the trajectory validation-only.

Add a bitwise no-feedback regression:

```text
vary spin +2 initial state
hold every primary input fixed
primary spin -2 trajectory must remain identical
```

Keep `plus2.enabled` fail-closed in `teuk_solver` during this milestone.

---

# 13. Milestone 6 — high-spin radial evolution and stability qualification

## 13.1 Combined explicit stability bound

The current radial characteristic CFL and pure-dissipation bound are necessary
but not sufficient.

Derive and validate a conservative combined timestep policy accounting for:

```text
radial Teukolsky operator
angular spectrum and selected bands
reduction damping
compatible dissipation
reconstruction graph
source work graph
spin +2 companion
high-spin nonnormality
```

A matrix-free startup estimator, frozen-coefficient analysis, or empirically
qualified envelope may be used, but spectral radius alone is insufficient for
a strongly nonnormal operator.

The historical D10-5 `T=200M`, 400,000-step setup with dissipation `0.005`
must remain rejected.

## 13.2 Boundary treatment

Do not add a nonzero SAT merely to claim sophistication.

The continuum has no incoming propagating endpoint mode, while the natural
symmetrizer degenerates at both endpoints.

Either:

- establish long-time semi-discrete stability of the zero-SAT closure over the
  production envelope; or
- derive a correct alternative boundary treatment.

Run normal-mode, random-noise, and outgoing-pulse tests at scri and horizon for
all production radial schemes and high spins.

## 13.3 Horizon-layer resolution

First determine whether the uniform compact `R` grid converges adequately for
the target near-extremal times.

If not, implement one of:

```text
a derived smooth map R=R(x) with the PDE transformed analytically and SBP
applied on uniform computational x;

or

a two-block/multiblock SBP-SAT radial method with a proven/verified interface.
```

Never apply the existing uniform-grid D10-5 coefficients directly to arbitrary
nonuniform `R` nodes.

Any mapping must preserve scri and horizon regularity and come with new
coefficient, CFL, dissipation, and convergence tests.

## 13.4 High-spin campaign matrix

At minimum qualify:

```text
a/M = 0.9, 0.99, 0.999, 0.9999
both signs where meaningful
multiple radial resolutions
multiple angular/source bands
multiple timesteps
multiple dissipation strengths including convergence toward zero
```

Measure time in both `T/M` and `kappa T`.

A production high-spin run should cover a meaningful interval in `kappa T`,
not merely a short fixed coordinate time.

Exact extremality remains a separate campaign.

---

# 14. Milestone 7 — physically meaningful horizon and scri diagnostics

The current nine-point horizon utility differentiates the stored field with
respect to compact `R`. That is a code diagnostic, not yet the desired physical
Aretakis observable.

Add and validate:

```text
stored Z_minus and Z_plus
raw physical Psi4 and Psi0
complex per-(ell,m) horizon values
derivatives of the physical fields
derivatives along a documented regular transverse direction
corotating/demodulated fields using Omega_H
source-family and pair contributions
high-ell tail and angular-ceiling occupation
condition/error estimates for high derivatives
```

For example, account analytically for

\[
\Psi_4=RZ_-,
\qquad
\Psi_0=\frac{R^5}{d_-^4}Z_+,
\]

rather than calling derivatives of `Z` derivatives of the physical Weyl
scalar.

Define carefully which coordinate is held fixed in every transverse
derivative. If using a regular ingoing radial coordinate or infalling tetrad,
document and test the transformation.

Do not reduce the primary science output to a maximum over all modes and theta.

Use a derivative estimator whose conditioning is measured and whose order is
verified through derivative four.

At scri, output both raw zero behavior and finite peeling coefficients without
forming `0/0`.

---

# 15. Milestone 8 — production checkpoint/replay/output integration

After Milestones 1–7 are green:

- atomically save primary and companion checkpoints with one manifest;
- destroy all runtime objects;
- load both artifacts;
- regenerate transient curvature/source workspaces;
- continue the real four-field trajectory;
- compare with an uninterrupted run.

Test wrong/missing/mixed checkpoint pairs.

Integrate the four-Weyl output contract into the actual runtime:

```text
Psi4^(1)
Psi0^(1)
Psi4^(2)
Psi0^(2)
```

Every record must identify:

```text
raw versus regularized value
perturbative order
spin weight
signed m and projected ell
tetrad/gauge convention
boundary
peeling power/coefficient
source normalization
endpoint qualification
code commit and configuration
```

Do not label raw second-order Weyl coefficients gauge invariant.

---

# 16. Milestone 9 — full physical and numerical validation

Standalone point fixtures are necessary but insufficient.

## 16.1 Existing regression suite

All existing:

```text
94-check symbolic audit
source ledgers
half-factor regression
angular operator tests
TSI fixtures
QNM fixtures
Route-B fixtures
checkpoint/replay tests
negative controls
```

must remain green without weaker tolerances.

## 16.2 Full sourced four-field manufactured case

Construct a manufactured ORG metric/curvature case with an independent source
oracle and evolve all four fields.

Demonstrate separate radial, temporal, angular, and source-work-band
convergence, including scri and horizon.

## 16.3 Physically consistent linear fixture

Use a complete normalized QNM/TSI or other constraint-consistent first-order
metric fixture to drive a nonzero quadratic source.

Do not claim physical nonlinear Gaussian data from zero reconstruction fields.

This goal does not require building a CTS/CTT solver inside this repository,
but physical source qualification must use either:

```text
a complete trusted first-order checkpoint,
a fully reconstructed mode fixture,
or a converged retarded-history construction.
```

Keep arbitrary Gaussian forcing fail-closed unless its reconstruction
constraints are genuinely satisfied.

## 16.4 Amplitude scaling

Over at least three first-order amplitudes verify:

\[
\Psi_{0,4}^{(1)}\propto A,
\qquad
S^{(2)}\propto A^2,
\qquad
\Psi_{0,4}^{(2)}\propto A^2.
\]

Under `A -> -A`, first-order fields must reverse while the quadratic source and
second-order response remain unchanged to perturbative accuracy.

## 16.5 Joint four-field convergence

At minimum use three resolutions and coupled time refinement.

Report separately:

```text
bulk
scri
horizon
representative interior radii
each signed m
dominant ell modes
all four Weyl fields
reduction constraints
reconstruction constraints
peeling residuals
source families
total forcing
```

Do not infer convergence from two resolutions.

## 16.6 High-spin QNM and zero-damped modes

Add at least one high-spin QNM/zero-damped-mode validation beyond the current
moderate-Kerr interior fixture.

Vary fit window, extraction location, radial resolution, angular band,
timestep, and dissipation.

## 16.7 Long near-extremal run

Run production-candidate trajectories long enough to evaluate secular
behavior over meaningful `kappa T`.

Do not fit an Aretakis or nonlinear growth exponent until:

```text
radial convergence
angular/source-band convergence
time convergence
dissipation convergence
horizon derivative conditioning
restart equivalence
```

are all demonstrated over the fit interval.

---

# 17. Milestone 10 — accelerator and performance qualification

Serial success is not accelerator qualification.

Run the exact promotion candidate on:

```text
Kokkos Serial
Kokkos OpenMP
the intended accelerator backend
```

Obtain actual current-HEAD Intel B580 SYCL runtime qualification when that is
the target hardware. Distinguish:

```text
configuration
compilation
linking
device discovery
short kernel execution
complete test suite
production-size run
long run
```

No hot RK/source stage may allocate, copy host/device data, or fence.

Measure:

```text
primary RHS time
Route-B curvature time
source time
spin +2 companion time
angular transforms
radial operators
checkpoint size/write time
output throughput
host and device memory
```

Optimize only measured bottlenecks and preserve readable equation code.

Do not claim CUDA or HIP without actual corresponding evidence.

---

# 18. Milestone 11 — final production enablement

`plus2.enabled=false` remains the default throughout development.

Enabling `plus2` must be the last small, reviewable commit.

Before that commit, an independent subagent that did not implement the final
integration must classify every mandatory gate as:

```text
PROVEN
PARTIALLY PROVEN
UNPROVEN
CONTRADICTED
```

Do not enable production while any mandatory gate is below `PROVEN`.

Initial production configuration may deliberately require:

```text
D10-5 or another specifically qualified radial scheme
FreeDamped reduction if StageConstrained is not qualified
qualified endpoint method/certificate
complete signed parent and target registries
sufficient source work band and theta count
stable combined timestep
current source-normalization version
supported zero/checkpoint companion initial policy
paired checkpoint manifest for restart
```

Reject incompatible settings clearly; do not silently repair them.

After enablement, rerun:

```text
clean Serial audit suite
OpenMP suite
accelerator suite
full sourced convergence case
physical restart/replay case
high-spin short qualification
high-spin long qualification
disabled-plus2 spin-minus2 regression
```

---

# 19. Production definition of done

The goal is complete only when all are true:

1. the exact current tree has a clean, reproducible audit-enabled baseline;
2. no public API can forge pipeline checkpoint/source authority;
3. primary, source, endpoint, and trajectory provenance are complete;
4. Route B robustly supplies all six curvature fields and eight derivative
   slots through production radial resolutions;
5. the endpoint method is stable in binary64 at least through `N_R=513`;
6. the complete `S0/R^7` source is independently validated and angularly
   converged;
7. the spin `+2` companion is integrated at the exact common RK stages;
8. no spin `+2` state can affect the primary spin `-2` trajectory;
9. all four Weyl fields exhibit sourced radial/time/angular convergence;
10. checkpoint/restart and replay reproduce the real four-field trajectory;
11. physical horizon/scri diagnostics are qualified;
12. the combined timestep/stability envelope is established;
13. `a/M=0.99`, `0.999`, and `0.9999` runs are qualified over meaningful
    `kappa T`;
14. the exact promotion commit runs on the intended accelerator;
15. long-run resource/stability evidence is recorded;
16. disabled-`plus2` production spin `-2` behavior remains unchanged;
17. all historical red evidence remains preserved;
18. the final independent adversarial review marks every mandatory gate
    `PROVEN`.

If external accelerator hardware or another genuinely unavailable resource
prevents one mandatory gate:

- finish every independent software/mathematical task;
- keep `plus2.enabled` fail-closed;
- return `FAIL-CLOSED RESEARCH-ONLY`;
- state exactly which external evidence is missing.

Do not enable production on the basis of inferred or stale hardware evidence.

---

# 20. Commit, push, and test-monitor cadence

Make small milestone commits.

After each focused integration commit:

1. test monitor runs focused tests;
2. coordinator reviews the diff;
3. merge only when focused tests pass.

At each milestone:

1. clean build;
2. full unit suite;
3. full audit-enabled CTest;
4. `git diff --check`;
5. update concise status documents;
6. push `main`.

Do not stop after pushing a milestone. Continue to the next milestone unless
the final definition of done or a genuine external blocker is reached.

Do not accumulate the entire project in one commit.

---

# 21. Final report

At the end create:

`docs/FINAL_PRODUCTION_QUALIFICATION.md`

It must contain:

```text
exact final commit
Kokkos submodule commit
compiler, Python, backend, driver versions
exact unit/CTest counts and elapsed times
mathematical gate status
endpoint method and conditioning matrix
source-work-band convergence
four-field convergence tables
checkpoint/replay evidence
no-feedback evidence
high-spin stability matrix
highest qualified spin
time interval in T/M and kappa T
accelerator evidence
performance/resource measurements
known limitations
exact production configuration restrictions
```

End with exactly one verdict:

```text
NOT READY
FAIL-CLOSED RESEARCH-ONLY
PRODUCTION-QUALIFIED
```

Select `PRODUCTION-QUALIFIED` only when every mandatory gate above has direct,
current, reproducible evidence.

Start now by spawning the five roles, running the exact-current full baseline,
and independently auditing every change since
`6f1db976fb96863554ad3f3269479e1e2a64edc7`.
