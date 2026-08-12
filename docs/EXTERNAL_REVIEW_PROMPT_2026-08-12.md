# Prompt for a repository-only external reviewer

You have access only to this Git repository. You do **not** have access to the
author's ignored `run/` directory, historical checkpoints, waveform CSVs,
plots, build trees, `/tmp` virtual environments, GPU driver logs, or any other
locally generated artifacts. Treat every statement about those artifacts as a
frozen diagnostic handoff, not as evidence you can reproduce from Git.

Your task is to perform a skeptical, independent, full-tree review of the
current repository, with special depth on the spin `+2` companion/source work,
the standalone Route-B curvature-provider candidate, and the associated spin
`-2` late-time diagnostic.
Begin by recording the exact branch, `git rev-parse HEAD`, parent, status,
submodule revisions, compiler/runtime choices, and the complete diff from the
review anchor `43e9300080140e3df4d05affd67ffa41b5ddbe57`. The scientific
remediation is commit `5b5c356208ca2c470e13e2d24e76f8545c090df0`.
The standalone provider candidate began at exact commit
`92465cdddd3867b22f15b4be10350ac292404c37`. Its independent-oracle and
contract hardening is commit `51c0823b`; resolve the full object name locally
and review its complete diff. The current final commit also adds exact-future-
horizon coordinate authority and a concrete same-stage Route-B live-source
seam; identify that commit from `HEAD` and review it as scientific code, not a
documentation-only update. Do not rely on the root `SHA256SUMS`,
`MANIFEST.md`, or `AUDIT_MANIFEST.md`: those describe an older spin-minus2
audit bundle, not the present tree.

Read these files first and completely:

- `docs/EXTERNAL_REVIEW_REMEDIATION_REPORT_2026-08-12.md`
- `docs/EXTERNAL_REVIEW_HANDOFF_2026-08-12.md`
- `docs/CONVENTIONS.md`
- `docs/PLUS2_FORMALISM_GATE.md`
- `docs/PLUS2_SOURCE_COMPACTIFICATION.md`
- `docs/PLUS2_CONSTRAINED_ENDPOINT_EXTRACTION.md`
- `docs/PLUS2_ROUTE_B_LOCAL_CURVATURE_BLOCKER.md`
- `docs/PLUS2_ROUTE_B_RADIAL_JET_TEUKOLSKY.md`
- `docs/PLUS2_ROUTE_B_RADIAL_JET_RECONSTRUCTION.md`
- `docs/PLUS2_ROUTE_B_ANGULAR_JET_COORDINATOR.md`
- `docs/PLUS2_LINEAR_PSI0_PRODUCTION_GATE.md`
- `docs/PLUS2_REPLAY_AND_OUTPUT.md`
- `docs/FOUR_WEYL_FIELD_VALIDATION.md`
- `PLUS2_SOURCE_INPUT_MANIFEST.csv`
- `PLUS2_SOURCE_NORMALIZATION_LEDGER.csv`
- `plus2_equation_spec_proposal.yaml`
- `include/teuk/plus2_routeb_curvature_spatial.hpp`
- `include/teuk/plus2_live_source_composition.hpp`
- `include/teuk/plus2_source_primitive_spatial.hpp`
- `tests/test_plus2_routeb_curvature_spatial.cpp`
- `tests/test_plus2_routeb_live_source.cpp`
- `tests/test_plus2_routeb_physical_replay.cpp`
- `tools/numerical/generate_plus2_routeb_curvature_coordinate_fixture.py`
- `tests/test_plus2_qnm_kerr.cpp`
- `tools/numerical/generate_plus2_qnm_kerr_fixture.py`

The YAML is explicitly a fail-closed proposal, not a production specification.
Historical documents may contain commit-scoped test counts; do not reinterpret
them as current checkout counts.

## Immutable review constraints

- Do not enable `plus2` production or weaken its solver fail-closed gate.
- Do not invent missing endpoint coefficients, curvature fields, QNM fixtures,
  or physical replay evidence.
- Do not weaken tolerances, delete tests, remove negative controls, or replace
  independent oracles with production-algebra restatements.
- Preserve `+---`, outgoing radiation gauge, `gamma=0`, no factorial
  perturbative expansion, and `X_m^sharp=conjugate(X_-m)`.
- Keep raw `Psi0^(2)`, evolved `Z_plus`, source-normalized `Z0_source`, naive
  TSI reconstruction, and effective Einstein-source variables distinct.
- Separate a mathematical derivation, a standalone implementation, integration,
  and scientific/GPU qualification. Passing one is not evidence for another.

## Primary audit questions

### 1. Source normalization

Independently derive from the explicit code tetrad

```text
d_minus = L^2-i a R cos(theta),
D       = L^4+a^2 R^2 cos(theta)^2,
W_plus  = R^5/d_minus^4 = R f,
f       = R^4/d_minus^4,
A_TT(raw)=R^2 C_T/(2D).
```

Check that the old multiplier `2D/R^3` leaves `C_T f`, while the common evolved
equation requires division by the complete field factor and therefore

```text
N/f = 2D d_minus^4/R^7,
forcing_plus = 2D d_minus^4 (S0/R^7).
```

Then independently derive and check

```text
(5b-4rho0-rhobar0)/R
 = i a cos(theta) E (3L^2+5 i a R cos(theta))/(2D^2)
```

and the cancellation-safe direct `S0/R^7` expression. Confirm that no evolved
path numerically divides a computed `S0/R^6` by `R`, that raw and evolved
diagnostics are typed and versioned separately, and that scri forcing is finite
and generally nonzero. Audit every term against
`PLUS2_SOURCE_NORMALIZATION_LEDGER.csv`, including signed-mode sharp ownership.
Run and inspect, rather than merely trust:

- `tools/symbolic/verify_plus2_source_normalization.py`
- `tools/symbolic/verify_plus2_formalism_gate.py`
- `tools/symbolic/verify_plus2_compact_source.py`
- source C++ tests, including host/device/Jet/scri/horizon/near-extremal cases

Require the historical normalization to fail a negative control.

### 2. Route-B constrained endpoint extraction and provider candidate

Re-solve the exact positive-node moment systems without importing the generated
header. Confirm the `q0=[R^2]f0` weights

```text
29/6, -461/24, 31, -307/12, 65/6, -15/8
```

and `q1=[R]f1` weights

```text
-77/12, 107/6, -39/2, 61/6, -25/12.
```

Also independently derive the promoted seven-positive-node `q0` operator used
only by the complete curvature graph:

```text
319/45, -3929/120, 389/6, -2545/36, 134/3, -1849/120, 203/90.
```

Verify that it extracts power two and annihilates powers `0,1,3,4,5,6`.
Confirm the repository kept the original six-node authority unchanged rather
than silently replacing it. Determine whether the promoted operator is a
scientifically justified fixed-stencil repair or an unjustified response to a
single observed convergence window; audit its conditioning and roundoff
sensitivity independently.

Check exact moments, order, norms, generator reproducibility, 100-digit oracle,
device parity, and the fixed `N=9,17,33` promotion matrix for Schwarzschild,
positive/negative moderate spin, near-extremal spin, and both signed modes.
Verify that peeling residuals `f0(0)`, `f0'(0)`, and `f1(0)` are measured
independently and are not silently overwritten. Treat `N=65` as a nonpromotion
conditioning probe.

The historical direct-`D1(h4)` remedy was rejected because fixed-window cells
failed the non-weakened convergence gate. Confirm that it remains absent. Then
audit candidate `92465cd`, which instead consumes the five generation-stamped
Route-B levels, forms the reviewed curvature numerators independently at time
levels 0, 1, and 2, and applies the constrained quotient extractor directly.
Do not accept the implementation merely because it emits six fields.

For every one of `Z0,Z1,Z0_T,Z1_T,Z0_TT,Z1_TT`, require separately:

- both signed modes and every theta node, not an aggregate maximum alone;
- scri, interior, and future-horizon comparison to an independent oracle;
- fixed `N=9,17,33` endpoint ratios without selecting a friendlier window;
- separately convergent `f0(0)`, `f0'(0)`, and `f1(0)` residuals at all three
  time levels;
- a coordinate full-Weyl comparison that includes Ricci trace subtraction and
  the perturbed ORG tetrad, not a restatement of the production NP algebra;
- exact amplitude linearity, signed-sharp ownership, rotating and
  near-extremal fixtures, and a negative control that distinguishes a wrong
  formula;
- stale generation, stale sharp partner, generation zero, NaN/Inf, wrong
  shape, alias, replay, and partial-readiness fail closure;
- deterministic zeroing/stamps and zero post-construction allocation, copy,
  and fence events on the hot stage.

Trace every derivative slot emitted by the candidate into the live source and
prove that all consumers see one immutable common-stage generation. The
checked-in hardening now claims a coordinate full-Weyl interior fixture,
componentwise endpoint and peeling-residual gates, hostile contracts,
linearity/sharp checks, and hot-stage instrumentation. Do not trust those
claims from prose or pass counts: regenerate the coordinate fixture, inspect
its independence, disaggregate every cell, and confirm the old aggregate test
could not hide a remaining failure. The fixture now includes scri, all eight
derivative slots, and the exact spin-dependent future compact horizon. Verify
that the derivative route applies physical GHP operators to coordinate-Weyl
scalars rather than calling production point operators. Its stored negative
controls omit `Delta_n` rescaling or GHP angular connections. At scri, confirm
that the six-node normalized-scalar extrapolator is distinct from the
production numerator extractor and that both fixed-window remainder budgets
are respected. At the horizon, audit the
two-axis qualification rather than accepting a single absolute tolerance:
radial `N=9,17,33` self-convergence at fixed `ell_max=24`, angular
`ell_max=12,18,24` convergence, and a continuum coordinate-Weyl discrepancy
bounded by the measured radial and angular remainders. Try to falsify the
`5e-11` floor classification and the stated roundoff ceiling rather than
silently relaxing either.

A standalone same-generation seam now consumes one immutable five-level
Route-B tower, evaluates curvature once, packs levels zero through two into the
concrete primitive producer, and reaches the live pair/outer source forcing.
Audit `evaluate_routeb_stage` and its test for duplicate authority, ordering,
stale-level resurrection, hidden host copies, and configuration mismatches.
The follow-up standalone replay test now composes that seam with the actual
source-independent Route-B primary/reconstruction tower and spin-plus-two PDE
through a common two-step RK4 trajectory. Audit bitwise concurrent/replay
equality, exact stage pointers/times/generations, nonzero forcing/response,
linear/quadratic amplitude scaling, no feedback, and hot-path instrumentation.
It remains manufactured, zero-dissipation, same-backend validation: it does
not include production `Psi4^(2)`, four-Weyl packing, a campaign checkpoint,
sourced residual convergence, solver/runtime wiring, or current GPU runtime
qualification. Keep `N=65` non-promoting and do not call the candidate
production-qualified.

### 3. Linear path

Audit the ordinary-NP linear `Psi0` spatial implementation and its coordinate
curvature oracles. Confirm that it is labeled validation-only because it
evaluates individually singular terms and requires external scri data. Do not
promote it unless a genuinely regular GHP formulation has independent endpoint
and rotating-Kerr evidence.

### 4. Checkpoint and replay provenance

Audit companion checkpoint format v3 and every fail-before-mutation path.
Confirm it binds exact `M,a,L`, full radial and angular coordinates, `dt`,
reduction mode, damping, dissipation, source normalization version/name,
primary checkpoint identity, step/time, bands, modes, methods, Git identity,
schema, layout, byte order, and checksum. Verify NaN/Inf rejection in both
metadata and state and exact `time=step*dt`. Confirm legacy v1/v2 restoration is
rejected, not silently reinterpreted, and that checkpoint replay continues from
the restored accepted time.

Do not call generic callback replay or the single-stage Route-B live-source
test scientific replay evidence. The new Route-B sourced trajectory does meet
that narrower bar for the standalone source-independent first-order/
reconstruction graph and passive companion: it is common-stage, bitwise
same-backend, nonzero, quadratic, and one-way. Do not broaden this result to a
production or four-field replay. It omits the production spin-minus-two
second-order trajectory, checkpoint-restored campaign data, output packing,
sourced residual convergence, runtime integration, and accelerator parity.

### 5. Live invalidation and diagnostic zeroing

Construct hostile stale-generation, stale-sharp, nonfinite, readiness-false,
and wrong-shape cases. Verify that validity is determined before normalization
and that rejection clears every exposed field at the rejected point: primitive
and derivative inputs, all pair families, target sums, J/K tangents, raw
`S0/R^6`, regular `S0/R^7`, and evolved forcing. Confirm deterministic stamp
initialization and no stale diagnostic survival.

### 6. Documentation and fixtures

Check all source/formalism equations and the proposal YAML against code and
tests. In particular, Bianchi-5 must contain `+3 Sig psi20`, not `+3 Sig H`.
Confirm the honest validation matrix:

- Schwarzschild normalized QNM: qualified fixture;
- moderate-Kerr real-frequency TSI: qualified fixture;
- moderate-Kerr complex-frequency QNM: qualified at interior points;
- moderate-Kerr QNM horizon/scri endpoints and evolved comparison: open.

For the Kerr QNM fixture, independently check the pinned `qnm==0.4.4`
frequency and angular constant, the conversion
`lambda_plus=A-2 a m omega+(a omega)^2`, and the complex angular Wronskian.
Confirm the generator does not import the frequency solver, that direct use of
`A` as `lambda_plus` fails its negative control, and that the independent
horizon-in Heun plus Kerr coordinate/tetrad chain satisfies the stored-field
operator for both signed sectors.  Do not broaden its two interior points into
an endpoint or evolved-waveform claim.

Do not infer missing generated fixture data from prose.

### 7. Spin-minus2 `T=200M` symptom

The report contains the only available summary of the local campaign. Audit
whether its interpretation is logically consistent with the repository's
operators and configuration, but clearly label it nonreproducible without the
ignored artifacts. The key symptom was late second-order growth starting near
`T=110M`, a smooth horizon-localized layer without a Nyquist radial signature,
and strong angular top-band source/forcing occupancy at `ellmax=8`. The run had
radial KO-like dissipation but no angular filter. Assess the stated conclusion:
under-resolution warning rather than demonstrated instability. Confirm that a
disabled spin-plus2 source-normalization defect cannot cause that spin-minus2
campaign. Recommend an unfiltered radial/angular/timestep refinement matrix;
treat filtering only as a later sensitivity test.

## Full-tree coverage beyond the priority findings

Do not limit the review to files named `plus2`. Inventory every tracked source,
header, test, generator, configuration, script, document, and third-party
notice. Build a dependency map from runtime parsing through `solver_driver`,
`SpatialPipeline`, angular/radial operators, RK stages, source activation,
checkpoint/restart, diagnostics, waveform/output packing, and QNM support.
For each public configuration key, verify parsing, validation, resolved output,
checkpoint provenance, defaults, and disabled behavior.

Audit the production spin `-2` path independently for:

- D4-2, D8-4, and D10-5 radial selection and minimum-grid constraints;
- RK4 stage times/state ownership, reduction constraints, damping, source
  activation snapshots, and compatible dissipation signs/scaling;
- the scheme-aware pure-dissipation timestep guard and its exact scope (it is
  not a combined hyperbolic-dissipative stability proof);
- signed-mode angular transforms, exact-product padding, retained-band
  truncation, endpoint regularity, and the explicit absence of angular
  filtering;
- checkpoint representation, legacy compatibility, checksums, Git/config
  identity, coordinate uniformity tolerance, and fail-before-mutation rules;
- output conventions at scri/horizon, raw versus regularized Weyl fields,
  metadata round trips, and the fact that horizon values are not fluxes;
- allocation, deep-copy, and fence claims on actual hot paths rather than only
  synthetic callbacks.

Audit the complete standalone `+2` graph in dependency order: homogeneous
coefficients, linear `Psi0`, Route-B jet towers and angular coordinator,
curvature/primitive/source producers, pair/outer projections, companion RK4,
checkpoint/replay, four-Weyl output, TSI/QNM fixtures, and runtime fail-closed
settings. Identify duplicate authority, stale adapters, incomplete seams,
unreachable code, hidden same-stage assumptions, and documents whose claims no
longer match code. Confirm that no standalone validation type is accidentally
reachable from production enablement.

Review test methodology, not just pass counts. Classify each test as algebraic
identity, production-parity, independent oracle, manufactured convergence,
structural integration, backend compilation, or actual device runtime. Flag
circular oracles, aggregate norms that can hide a failing component, fixed-grid
RK tests presented as joint space-time convergence, roundoff-floor exceptions,
and stale commit-scoped counts. Inspect ignored/generated-file policy and state
exactly which conclusions cannot be reconstructed from Git alone.

## Verification expectations

Create a fresh Python environment from `requirements.txt`; configure a clean
Release Serial build with symbolic audit enabled; run the complete unit, QNM,
config, symbolic, generator-freshness, numerical-oracle, and CTest suite. Record
exact commands, versions, counts, elapsed time, and failures. Also run
`git diff --check` and regenerate every checked-in generated header in check
mode. Do not claim current-head B580 qualification unless you perform a clean
exact-commit SYCL build and a real Level Zero runtime run on the named device;
compile success or stale binaries are insufficient.

## Required review output

Produce a self-contained report with:

1. exact reviewed Git identity and environment;
2. derivations written out independently, not copied conclusions;
3. every issue classified as confirmed, refuted, partially remediated, or open;
4. exact file/line references for defects and evidence;
5. all commands and numerical results;
6. a claim/evidence matrix separating formula, standalone code, integration,
   convergence, replay, and GPU qualification;
7. explicit remaining blockers and the minimum non-weakened promotion gates;
8. a final verdict: safe to merge as disabled validation infrastructure, needs
   changes, or production-qualified (the last verdict requires closing every
   blocker, not merely passing the standalone suite).

If you find a defect, first report the minimal counterexample and affected
claim. Do not silently repair it while reviewing. If later authorized to edit,
make the narrowest fix, preserve negative controls, rerun the exact failing
gate plus the full suite, and report the new exact commit.
