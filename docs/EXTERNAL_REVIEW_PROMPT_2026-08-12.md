# Prompt for a repository-only external reviewer

You have access only to this Git repository. You do **not** have access to the
author's ignored `run/` directory, historical checkpoints, waveform CSVs,
plots, build trees, `/tmp` virtual environments, GPU driver logs, or any other
locally generated artifacts. Treat every statement about those artifacts as a
frozen diagnostic handoff, not as evidence you can reproduce from Git.

Your task is to perform a skeptical, independent, full-tree review of the spin
`+2` companion/source work and the associated spin `-2` late-time diagnostic.
Begin by recording the exact branch, `git rev-parse HEAD`, parent, status,
submodule revisions, compiler/runtime choices, and the complete diff from the
review anchor `43e9300080140e3df4d05affd67ffa41b5ddbe57`. The scientific
remediation is commit `5b5c356208ca2c470e13e2d24e76f8545c090df0`; the following
documentation commit only adds the review report and this prompt. Do not rely on the
root `SHA256SUMS`, `MANIFEST.md`, or `AUDIT_MANIFEST.md`: those describe an
older spin-minus2 audit bundle, not the present tree.

Read these files first and completely:

- `docs/EXTERNAL_REVIEW_REMEDIATION_REPORT_2026-08-12.md`
- `docs/EXTERNAL_REVIEW_HANDOFF_2026-08-12.md`
- `docs/CONVENTIONS.md`
- `docs/PLUS2_FORMALISM_GATE.md`
- `docs/PLUS2_SOURCE_COMPACTIFICATION.md`
- `docs/PLUS2_CONSTRAINED_ENDPOINT_EXTRACTION.md`
- `docs/PLUS2_ROUTE_B_LOCAL_CURVATURE_BLOCKER.md`
- `docs/PLUS2_LINEAR_PSI0_PRODUCTION_GATE.md`
- `docs/PLUS2_REPLAY_AND_OUTPUT.md`
- `docs/FOUR_WEYL_FIELD_VALIDATION.md`
- `PLUS2_SOURCE_INPUT_MANIFEST.csv`
- `PLUS2_SOURCE_NORMALIZATION_LEDGER.csv`
- `plus2_equation_spec_proposal.yaml`

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

### 2. Route-B constrained endpoint extraction

Re-solve the exact positive-node moment systems without importing the generated
header. Confirm the `q0=[R^2]f0` weights

```text
29/6, -461/24, 31, -307/12, 65/6, -15/8
```

and `q1=[R]f1` weights

```text
-77/12, 107/6, -39/2, 61/6, -25/12.
```

Check exact moments, order, norms, generator reproducibility, 100-digit oracle,
device parity, and the fixed `N=9,17,33` promotion matrix for Schwarzschild,
positive/negative moderate spin, near-extremal spin, and both signed modes.
Verify that peeling residuals `f0(0)`, `f0'(0)`, and `f1(0)` are measured
independently and are not silently overwritten. Treat `N=65` as a nonpromotion
conditioning probe.

Most importantly, verify the scope boundary: this extraction component does
not supply the missing same-stage live numerator graph for all six value/T/TT
curvature fields. The rotating scri dependency on `h4[1]` must remain an open
blocker; no partial adapter may be advertised as a complete Route-B provider.

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

Do not call generic callback replay a physical replay. A physics-level replay
requires one shared-stage primary-curvature-source-companion trajectory, which
remains blocked by the incomplete live curvature provider.

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
- moderate-Kerr complex-frequency QNM and horizon/scri endpoints: open.

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
