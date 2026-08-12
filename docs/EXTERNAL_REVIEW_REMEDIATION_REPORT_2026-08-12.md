# External-review remediation report: spin `+2` source and infrastructure

Date: 2026-08-12 (America/New_York)

Review anchor: `43e9300080140e3df4d05affd67ffa41b5ddbe57`

Scientific remediation commit:
`5b5c356208ca2c470e13e2d24e76f8545c090df0`

Standalone Route-B curvature-provider candidate:
`92465cdddd3867b22f15b4be10350ac292404c37`

Standalone provider hardening and independent-oracle commit:
`51c0823b` (resolve the full object name with `git rev-parse 51c0823b`)

The exact report checkout is obtained with `git rev-parse HEAD`. This document
is committed after the scientific-code remediation commit, so repository-only
reviewers can audit the complete tree without any ignored `run/`, build, or
generated campaign artifacts.

Exact handoff baseline `fee00bb076fc8ddd591461ad295a8ab65592cfc9`
was freshly configured and built, passed the direct Serial unit suite
`381/381`, and passed audit-enabled CTest `22/22`. During the subsequent
provider-hardening work, a direct Serial run passed `384/384`; that run
preceded the final scientific commit and was not repeated after commit. At the
owner's explicit request, no build or test was run while wrapping up this
report. Therefore the `384/384` result is development evidence, not
exact-final-document-HEAD qualification. A repository-only reviewer must run
the commands requested in `docs/EXTERNAL_REVIEW_PROMPT_2026-08-12.md` before
making any promotion claim.

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

## 4. Route-B endpoint and standalone provider candidate

The dependency audit confirmed that rotating scri `Z0_TT` and `Z1_TT` need the
information represented by the radial coefficient `h4[1]`, while the approved
tower exposes only `h4[0]`. Directly differentiating the computed `h4` profile
failed the predeclared convergence windows, so that rejected prototype remains
absent. That original blocker is **confirmed**.

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

The complete curvature graph exposed a pre-asymptotic failure that an
aggregate norm had hidden: the minimum six-positive-node `q0` extractor gave
only a `13.807` ratio for Schwarzschild `Z0` level zero on the fixed
`N=9,17,33` window. The threshold and window were not weakened. The original
six-node operator remains unchanged as the minimum fourth-order authority. A
separate seven-positive-node promoted operator was derived by an exact moment
solve for the complete curvature graph:

```text
319/45, -3929/120, 389/6, -2545/36, 134/3, -1849/120, 203/90.
```

It extracts the `R^2` coefficient and annihilates powers `0,1,3,4,5,6`.
Its exact norms are `L1=10696/45`, `L2^2=271316521/21600`, and
`Linf=2545/36`. The generated header, exact-moment C++ tests, device parity,
and independent arbitrary-precision verifier cover both operators.

Candidate `92465cd` explores a different closure that does not reconstruct
`h4[1]` by differentiating `h4`. It consumes a generation-stamped five-level
Route-B reconstruction tower, evaluates the reviewed connection and curvature
numerators at each time level, and applies the constrained positive-node
quotient extraction directly to each numerator profile. It emits all six
`Z0,Z1,Z0_T,Z1_T,Z0_TT,Z1_TT` fields, the eight Bianchi derivative slots, and
all nine peeling residual audits. It is allocation-free after construction,
generation-stamped, and globally fail-closed on stale or nonfinite tower data.

This is a **standalone candidate, not a closed remediation or production
provider**. Commit `51c0823b` adds the missing standalone qualification slices:

- a reproducible coordinate full-Weyl fixture that includes Ricci trace
  subtraction and the prescribed perturbed ORG tetrad, covers independently
  isolated `V`, `C`, and `B` sectors plus their sum, both signed modes, three
  time levels, and `a/M=.63,.999,-.74`;
- a six-field, signed-mode, every-theta `N=9,17,33` scri gate whose minimum
  resolved ratios are `18.4274,17.9576,17.0742,18.1942` for
  `a/M=0,.63,-.74,.999` respectively;
- separately gated peeling residuals at every time level, mode, and theta,
  with minimum ratios above `209` and maximum fine residual below
  `4.44e-12` in the development run;
- hostile stale-generation, nonfinite, replay, duplicate-offset, wrong-shape,
  alias, deterministic global-zero/stamp, amplitude-linearity, signed-sharp,
  and zero hot-stage allocation/copy/fence checks.

The coordinate fixture uses `ell_max=12,Ntheta=32`; smaller `ell_max=4`
truncated rotating-Kerr coefficient-induced angular content and produced a
false milliscale mismatch. At the fixed qualified band the development maximum
absolute errors were `9.36e-12` for `Z0` and `9.31e-12` for `Z1`.

Important gaps remain. The coordinate-Weyl samples are interior points on
`R in [0,.8]`, not the exact future horizon or scri. Endpoint authority is
manufactured self-convergence plus independent peeling residuals, not a
coordinate-Weyl endpoint fixture. The eight derivative slots still lack an
independent coordinate derivative fixture. There is no same-stage live-source
composition, primary-curvature-source-companion physical replay, exact-commit
GPU runtime evidence, or solver/runtime wiring. The candidate is not included
by `SpatialPipeline` or `solver_driver`, and `plus2.enabled` remains rejected.
An external reviewer must treat it as a substantially hardened standalone
hypothesis and try to falsify it.

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
- endpoint extraction: exact generator, minimum and promoted generated weights,
  device-inline operators, 100-digit verifier, curvature initializer, tests,
  and design note;
- Route-B candidate: five-level tower consumer, six curvature fields, eight
  derivative slots, nine peeling audits, coordinate full-Weyl fixture,
  disaggregated endpoint/residual tests, and hostile contract tests;
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
| Complete Route-B six-field curvature provider | Hardened standalone candidate only | independent interior coordinate-Weyl fixture, disaggregated scri/residual gates, hostile contracts, and hot-path instrumentation exist; endpoint coordinate oracle, derivative-slot oracle, live integration, replay, and GPU qualification remain open |
| Linear `Psi0` production spatial graph | Not met | explicitly validation-only ordinary-NP path |
| Companion checkpoint physical identity | Met standalone v3 | exhaustive mismatch/nonfinite/pre-mutation tests |
| Live invalid diagnostics clear globally | Met | provenance/stamp and full workspace-zero regressions |
| Concrete physics replay | Not met | structural replay only; candidate provider is not qualified or runtime-wired |
| Schwarzschild QNM | Met fixture | checked-in normalized interior fixture |
| Moderate-Kerr real-frequency TSI | Met fixture | checked-in radial/angular/field fixture |
| Moderate-Kerr complex QNM and endpoints | Open | no claim |
| Current exact-head B580 qualification | Not met | Serial only; local device previously wedged/device-lost |
| Spin `-2` production equations unchanged | Met by diff scope | no production equation/source/pipeline files changed; exact-final-tree suite still needs rerun |
| Spin `+2` production enabled | No | solver remains fail-closed by design |

## 9. Remaining promotion blockers

1. Independently reproduce and try to falsify the new coordinate-Weyl,
   disaggregated endpoint/residual, promoted-extractor, hostile contract, and
   hot-path evidence. Add exact scri/future-horizon coordinate authority and an
   independent oracle for all eight derivative slots. Keep `N=65` explicitly
   non-promoting and do not tune the window after seeing results.
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
