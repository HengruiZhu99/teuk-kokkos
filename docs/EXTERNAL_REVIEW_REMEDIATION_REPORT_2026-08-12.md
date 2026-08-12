# External-review remediation report: spin `+2` source and infrastructure

Date: 2026-08-12 (America/New_York)

Review anchor: `43e9300080140e3df4d05affd67ffa41b5ddbe57`

Scientific remediation commit:
`5b5c356208ca2c470e13e2d24e76f8545c090df0`

The exact report checkout is obtained with `git rev-parse HEAD`. This document
is committed after the scientific-code remediation commit, so repository-only
reviewers can audit the complete tree without any ignored `run/`, build, or
generated campaign artifacts.

The final edited checkout was not rebuilt or rerun after the last provenance
and configuration edits, at the owner's explicit request to wrap up without
another run. The strongest exact baseline evidence is recorded below. An
intermediate post-normalization direct unit run reached `379/379`, but later
metadata/configuration edits were made after that run; it is therefore not
presented as exact-final-tree qualification. A repository-only reviewer should
run the commands in `docs/EXTERNAL_REVIEW_PROMPT_2026-08-12.md` before making a
promotion claim.

The root `SHA256SUMS`, `MANIFEST.md`, and `AUDIT_MANIFEST.md` belong to an older
spin-minus2 audit bundle and do not describe this current plus2 checkout. They
must not be used as current-tree integrity evidence. The authoritative review
boundary is Git object identity plus fresh generator/audit/test execution.

## 1. Scope and immutable decisions

This work independently re-derived the external-review allegations before
changing code. It adds no new physics, does not change production spin `-2`
equations, does not remove fail-closed behavior, and does not enable spin
`+2`. `solver_driver.hpp` still rejects every `plus2.enabled=true` mode before
output creation or pipeline allocation.

Conventions remain `+---`, outgoing radiation gauge, `gamma=0`, no factorial
perturbative expansion, and
`X_m^sharp=conjugate(X_-m)` (never same-`m` conjugation). The distinct fields
remain

```text
Psi4 = R Z_minus,
Psi0 = R^5 Z_plus/(L^2-i a R cos(theta))^4,
Z0_source = Psi0/R^5,
Z1 = Psi1/R^4.
```

Raw sourced `Psi0^(2)`, evolved `Z_plus`, source-normalized `Z0_source`, naive
TSI reconstruction, and effective Einstein-source variables are not merged.

## 2. Clean baseline before edits

At exact anchor `43e9300`, the tracked tree was clean and synchronized with
`origin/main`; `git diff --check` passed. A fresh venv
`/tmp/teuk-external-review-20260812` used Python 3.12.3, SymPy 1.14.0, NumPy
2.5.2, SciPy 1.18.0, PyYAML 6.0.3, and mpmath 1.3.0. GCC 13.3.0 and Kokkos
5.1.0 Serial configured with symbolic audit enabled. The direct unit run was
`372/372`; audit-enabled CTest was `19/19`. These are baseline results, not
claims about the final checkout.

## 3. Source-normalization derivation

Define

```text
d_minus = L^2-i a R cos(theta),
D       = L^4+a^2 R^2 cos(theta)^2,
W_plus  = R^5/d_minus^4 = R f,
f       = R^4/d_minus^4.
```

Starting from the explicit Ripley code tetrad, the raw principal coefficient
is

```text
A_TT(raw)=l^T n^T-m^T mbar^T=R^2 C_T/(2D).
```

The historical multiplier `N=2D/R^3` gives

```text
N A_TT(raw) (R f)=C_T f,
```

not the common-equation coefficient `C_T`. Dividing the complete field factor
gives

```text
N/f = 2D d_minus^4/R^7,
forcing_plus = 2D d_minus^4 (S0/R^7).
```

The prior evolved forcing therefore omitted the complete complex field factor.
This finding is **confirmed**.

### Cancellation-safe `S0/R^7`

The raw ledger remains valid as

```text
S0/R^6 = thorn_5 J-(4 rho0+rhobar0)J
        +R eth_6 K+R^2(-4 tau0+pibar0)K-3R psi20 Q.
```

It is retained for diagnostics only. With

```text
b=-E/(2D),
E=L^2-2MR+a^2R^2/L^2,
```

direct substitution of the Kerr background proves the extra cancellation

```text
(5b-4rho0-rhobar0)/R
 = i a cos(theta) E (3L^2+5 i a R cos(theta))/(2D^2).
```

The evolved source is evaluated directly, never by dividing a computed
`S0/R^6` by `R`:

```text
S0/R^7 = A J_T+b J_R+C_J J+eth_6 K
        +R(-4tau0+pibar0)K-3psi20 Q,
A   = 2M(2M-a^2R/L^2)/D,
C_J = (5b-4rho0-rhobar0)/R+i a m/D-3epsilon0+epsilonbar0.
```

Consequently `S0=O(R^7)` and the evolved forcing is finite and generally
nonzero at scri. The old implementation produced an artificial `O(R^4)`
suppression for the same regular source. This finding is **confirmed**.

The independent executable authority is
`tools/symbolic/verify_plus2_source_normalization.py`. It derives the tetrad
coefficient, explicitly rejects the historical multiplier, checks the optical
cancellation, verifies fixed rational random analytic fields, and checks raw
versus compact expressions. `PLUS2_SOURCE_NORMALIZATION_LEDGER.csv` separates
the raw and evolved quantities. C++ tests cover Schwarzschild, moderate Kerr,
near-extremal Kerr, scri/interior/horizon, signed modes, Jet1, host, and device
paths.

## 4. Route-B endpoint result

The dependency audit confirms that rotating scri `Z0_TT` and `Z1_TT` need the
radial coefficient `h4[1]`, while the approved tower exposes only `h4[0]`.
The live six-field local-curvature provider is therefore still unavailable.
That finding is **confirmed**.

The constrained extraction candidates were independently solved as exact
moment systems. On positive nodes `R_j=jh`, the six `q0=[R^2]f0` weights are

```text
29/6, -461/24, 31, -307/12, 65/6, -15/8,
```

and the five `q1=[R]f1` weights are

```text
-77/12, 107/6, -39/2, 61/6, -25/12.
```

Their exact `(L1,L2^2,Linf)` condition summaries are
`(280/3,613067/288,31)` and `(56,60995/72,39/2)`. They never read `R=0` and
therefore do not overwrite peeling residuals. `f0(0)`, an independent D10-5
`f0'(0)`, and `f1(0)` remain separately measured and convergence-gated.

The exact generator, generated header, 100-digit independent audit, C++
host/device tests, and dated design note are checked in. `N=9,17,33` passes
fourth-order gates for Schwarzschild, positive/negative moderate spin,
near-extremal spin, and both signed modes. `N=65` remains a non-promotion
conditioning probe because the complete Route-B tower is already red at that
resolution.

This is a **partial remediation**: the endpoint operator is qualified, but the
same-stage live numerator graph for all six value/T/TT fields is not
implemented. No partial curvature adapter is fabricated.

## 5. Other review findings and remediation

### Linear `Psi0` spatial path

The ordinary-NP point formula and coordinate-curvature oracles remain valid,
but the spatial graph evaluates individually singular terms and relies on an
external scri coefficient. It is now explicitly labeled
`validation-only-ordinary-np-v1`; D4-2/D8-4 cases are historical convergence
comparisons, not production authority. A genuinely regular GHP production
provider remains open. Finding: **confirmed; scope corrected**.

### Companion checkpoint and replay provenance

The prior format did not uniquely identify the physical problem. Version 3
now records exact `M,a,L`, every radial and angular coordinate, `dt`, reduction
mode, damping, dissipation, source-normalization version/name, and primary
checkpoint identity. It rejects NaN/Inf state or provenance before device
mutation, checks `time=step*dt`, and rejects versions 1/2 for restoration
rather than silently reinterpreting them. Checkpoint replay is latched to the
restored accepted time. Finding: **confirmed and remediated for the standalone
companion checkpoint**.

Generic replay equality is still structural. Because the live Route-B
curvature provider is incomplete, there is no honest primary-curvature-source-
companion physical replay trajectory. Finding: **confirmed; promotion gate
remains open**.

### Live source invalidation

The former outer normalization could clear only fields after the first stale
stamp and could leave internal pair/source diagnostics populated. It now
determines validity first, includes the registry-wide source readiness bit,
initializes all stamp buffers deterministically, then clears every projected
field, outer derivative, pair family, target sum/tangent, raw `S0/R^6`, regular
`S0/R^7`, and evolved forcing at a rejected point. Tests inspect the entire
exposed diagnostic surface. Finding: **confirmed and remediated**.

### Documentation, Bianchi-5, QNM/TSI

The formalism/source documents and proposal YAML now advertise the corrected
normalization and include dated errata rather than silently rewriting old
claims. Bianchi-5 remains the already corrected
`+3 Sig psi20`, not the erroneous nonlinear `+3 Sig H` substitution.

Validation status is intentionally unchanged: Schwarzschild normalized QNM
is qualified; moderate-Kerr real-frequency TSI is qualified; moderate-Kerr
complex-frequency QNM and endpoint qualification remain open. No fixture was
invented to close them.

## 6. `T=200M` symptom, diagnostics, and relevance

The campaign data are not in Git. The following is a frozen handoff of
read-only local diagnostics, included so a repo-only reviewer knows the exact
symptom but must not treat it as reproducible repository evidence.

This was a spin `-2`, not spin `+2`, near-extremal B580 run at `a/M=.999`,
`nr=513`, `Ntheta=16`, first/second `ellmax=8/8`, D4-2, RK4, `dt=.0005`,
FreeDamped `gamma=.1`, and radial KO-like `epsilon=.005`. It completed normally
at `T=200M`; final `Psi2 RMS=5.347670999e-5`, second reduction constraint
`6.087864068e-5`, and `C2/Psi2=1.13841`.

The growth began near `T=110M` and slowed late. Radial Nyquist indicators at
the final checkpoint were small (`A_alt=.020693`, `D2=.002450`,
`D4=2.443e-5`), although 99.09% of second-field radial energy lay in the last
64 points. This is a smooth unresolved horizon layer, not demonstrated radial
checkerboard instability.

The stronger warning is angular: the quadratic source occupied the retained
ceiling before the field grew. At `T=100M`, `ell>=7` carried 69.9% of source
power and 77.3% of forcing power; `ell=8` alone carried 62.2% of forcing.
`Ntheta=16` satisfied exact-product padding, so aliasing was controlled, but
Galerkin truncation is not filtering. There was no angular filter. Radial
dissipation was present on both orders at every RK stage, but its `T=200M`
second-order P/Q/Psi RHS contribution was only 0.082/0.146/0.067 percent and
cannot remove angular top-band pileup.

Classification remains: **under-resolution warning, not demonstrated radial
grid-scale runaway**. The source-normalization bug remediated here is in the
disabled spin `+2` path and cannot explain this historical spin `-2` run.
Required follow-up is independent radial, angular-band/node, and timestep
refinement; angular filtering is a sensitivity study only after unfiltered
convergence.

## 7. Files changed by function

- source mathematics and kernels: `plus2_source.hpp`, source spatial/value/
  outer workspaces, live composition, normalization ledger and symbolic gates;
- endpoint extraction: exact generator, generated weight header, device-inline
  operator, 100-digit verifier, curvature initializer, tests, and design note;
- provenance: companion checkpoint/replay, four-Weyl metadata, resolved config;
- scope/docs: formalism, compactification, linear spatial gate, replay,
  Route-B blocker, validation status, proposal YAML, and dated handoff erratum;
- tests: source host/device/Jet/endpoints, live fail-close, angular convergence,
  checkpoint mismatch/nonfinite/time binding, endpoint moments/convergence.

No `full_spatial.hpp`, spin `-2` equation/source kernel, production solver
pipeline, or solver enable gate was changed.

## 8. Claim/evidence matrix

| Claim | Status after remediation | Repository evidence |
|---|---|---|
| Complete spin+2 source normalization | Met standalone | exact derivation, negative old-form regression, random analytic, C++ host/device/Jet/endpoints |
| Raw/evolved source distinction | Met | typed `S0/R6` and `S0/R7` workspaces, two ledgers, versioned metadata |
| Cancellation-safe scri forcing | Met standalone | exact optical identity and D10-5 outer convergence |
| Route-B endpoint extraction | Met as component | rational generator, 100-digit audit, N9/17/33, residual gates |
| Complete Route-B six-field curvature provider | Not met | missing live same-stage numerator graph; N65 full-tower conditioning red |
| Linear `Psi0` production spatial graph | Not met | explicitly validation-only ordinary-NP path |
| Companion checkpoint physical identity | Met standalone v3 | exhaustive mismatch/nonfinite/pre-mutation tests |
| Live invalid diagnostics clear globally | Met | provenance/stamp and full workspace-zero regressions |
| Concrete physics replay | Not met | structural replay only; provider unavailable |
| Schwarzschild QNM | Met fixture | checked-in normalized interior fixture |
| Moderate-Kerr real-frequency TSI | Met fixture | checked-in radial/angular/field fixture |
| Moderate-Kerr complex QNM and endpoints | Open | no claim |
| Current exact-head B580 qualification | Not met | Serial only; local device previously wedged/device-lost |
| Spin `-2` production equations unchanged | Met by diff scope | no production equation/source/pipeline files changed; exact-final-tree suite still needs rerun |
| Spin `+2` production enabled | No | solver remains fail-closed by design |

## 9. Remaining promotion blockers

1. Construct and qualify the complete live same-stage Route-B curvature
   numerator graph for all six fields, including independent coordinate-Weyl
   comparison and the fixed `N=9,17,33` matrix; keep `N=65` red/non-gating.
2. Replace or retain as validation-only the singular-term ordinary-NP linear
   spatial path with a genuinely regular GHP provider.
3. Run primary, curvature, source, and companion through one concurrent and
   replay trajectory; then prove bitwise same-backend and physical residual
   agreement.
4. Complete sourced four-field time/radial/angular residual convergence and
   current exact-head accelerator qualification.
5. Add moderate-spin complex-frequency QNM and horizon/scri endpoint fixtures.
6. Perform the independent `T=200M` spin-minus2 resolution matrix before any
   stability or waveform claim.

Until every item is closed, `plus2.enabled` must remain fail-closed.
