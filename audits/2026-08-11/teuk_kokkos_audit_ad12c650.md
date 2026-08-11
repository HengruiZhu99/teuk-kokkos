# Independent audit of `HengruiZhu99/teuk-kokkos`

**Audited commit:** `ad12c650c3f65a957c34b1a8a8f4a178671b02d1`  
**Audit date:** 2026-08-11  
**Scope:** continuum equations, compactification and GHP conventions, first-order reduction, metric reconstruction, quadratic source, RK staging, angular representation, radial finite differences, initial data, diagnostics, and readiness for extremal/near-extremal science.

## Executive verdict

The repository is a strong implementation effort and gets several difficult pieces right, most notably the corrected quadratic source and the fully coupled RK-stage source evaluation. However, **the current production pipeline is not yet solving the intended first- or second-order Teukolsky equations correctly**.

Three findings block scientific use:

1. **The spin-weighted angular operator in the Teukolsky equation has the wrong eigenvalue.** This is a continuum-equation error and affects both first and second order.
2. **The overcollocated nodal state is not projected back into the retained spin-weighted harmonic band.** This leaves uncontrolled off-band degrees of freedom with an incorrect angular evolution.
3. **The default Gaussian run starts all metric-reconstruction fields at zero and immediately activates the quadratic source.** Those data are not generically consistent with the independent reconstruction constraints, so the early second-order source is not guaranteed to come from a linearized vacuum metric.

Until these are corrected, quasinormal frequencies, tails, horizon growth, and quadratic-instability measurements should not be interpreted physically.

The corrected compact second-order source itself appears correct term by term, including the legacy half-factor correction. The common-stage RK4/JVP implementation is also structurally correct.

---

## Finding 1 — BLOCKER: wrong Teukolsky angular eigenvalue

### Required operator

The implementation paper defines the spin-weighted Laplace–Beltrami operator by

\[
{}_s\Delta_\Omega\,{}_sY_{\ell m}
=-(\ell-s)(\ell+s+1){}_sY_{\ell m}.
\]

Equivalently,

\[
\lambda_{\rm Teuk}(\ell,s)
=s(s+1)-\ell(\ell+1).
\]

This is the lowering-after-raising composition

\[
\mathcal L_{s+1}\mathcal R_s.
\]

The legacy Fortran implementation uses this same eigenvalue.

### Current implementation

`include/teuk/angular.hpp` defines the production Laplacian eigenvalue as

\[
\lambda_{\rm code}(\ell,s)=s^2-\ell(\ell+1),
\]

which is the symmetric average of the two eth compositions. `DeviceAngularPlan` copies this eigenvalue to the production device operator, and both first- and second-order fields use it.

The difference is

\[
\lambda_{\rm code}-\lambda_{\rm Teuk}=-s.
\]

For the gravitational field, `s=-2`, so the code adds a spurious

\[
+2\psi
\]

to the angular contribution of the `P` equation. For example,

\[
\ell=2:\qquad \lambda_{\rm Teuk}=-4,
\qquad \lambda_{\rm code}=-2,
\]

and

\[
\ell=3:\qquad \lambda_{\rm Teuk}=-10,
\qquad \lambda_{\rm code}=-8.
\]

There is no compensating shift in the pointwise `psi` coefficient: `teukolsky_coefficients()` agrees with the paper/legacy coordinate coefficient. Therefore the total operator is wrong.

### Impact

This changes the homogeneous operator at both perturbative orders. It affects:

- linear wave propagation;
- quasinormal-mode frequencies and eigenfunctions;
- late-time tails;
- extremal-horizon scaling;
- the Green function acting on the quadratic source;
- the measured amplitude and exponent of the second-order response.

The current unit tests do not catch this because they test the symmetric eigenvalue against the same helper that implements it.

### Required fix

Change the production spin-weighted Laplacian to

```cpp
inline Real spin_weighted_laplacian_eigenvalue(int ell, int spin) {
  validate_mode(ell, spin, 0);
  return -static_cast<Real>((ell - spin) * (ell + spin + 1));
}
```

or return `lower_after_raise_eigenvalue(ell, spin)` directly.

If the symmetric bundle Laplacian is useful elsewhere, keep it under a different explicit name; do not use it in the Teukolsky `P` equation.

### Required regression tests

1. `s=-2, ell=2 -> -4`.
2. `s=-2, ell=3 -> -10`.
3. Compare the matrix action with an explicit lower-after-raise operation.
4. In Schwarzschild at `R=0`, apply the angular part to a pure `{}_{-2}Y_{22}` and verify the coefficient is `-4`.
5. Add an independent QNM-frequency regression after the PDE fixes are complete.

---

## Finding 2 — BLOCKER: padded angular nullspace is evolved without projection

### Current representation

The production state is stored as

\[
U[m,\mathrm{field},R,\theta_j]
\]

on a common padded Gauss–Legendre grid. For fixed `(s,m)`, however, the angular transform retains only

\[
\ell=\max(|s|,|m|),\ldots,\ell_{\max}.
\]

Thus the number of retained modal degrees of freedom is generally smaller than `theta_nodes`.

The angular derivative has the form

\[
D_\Omega=S\Lambda A,
\]

where `A` analyzes only the retained band and `S` synthesizes it. The operator therefore has a nullspace of dimension

\[
N_\theta-N_{\rm retained}(s,m).
\]

### Why this becomes incorrect

The initial state is bandlimited, but Kerr coefficients depend non-polynomially on `cos(theta)`, and pointwise multiplications/divisions in the RHS create components outside the retained band. The RK update writes the entire unprojected nodal RHS into the state.

On the next stage:

- pointwise terms see these off-band components;
- radial derivatives evolve them;
- the angular derivative analyzes only the retained projection and therefore gives them no correct angular action.

Only the quadratic inner sources `D` and `T` are explicitly projected before the outer angular derivative. The first-order fields, reconstruction fields, their tangents, and second-order fields are not projected at each stage.

This is neither a standard collocation method, in which every nodal degree of freedom corresponds to a retained mode, nor a Galerkin/pseudospectral truncation, in which the complete RHS is projected onto the retained band.

The issue is most extreme for large `|m|`. For `ell_max=4`, `theta_nodes=7`, and `s=-2,m=4`, only the single `ell=4` harmonic is retained while six nodal directions remain outside the angular operator's range.

### Impact

- uncontrolled angular nullspace;
- incorrect finite-resolution evolution;
- possible spurious growth or contamination of horizon derivatives;
- invalidation of the stated exact-product/dealiasing interpretation once the input fields themselves leave the retained bands;
- especially dangerous for long, near-extremal runs.

### Required fix

Use one of these mathematically consistent strategies.

#### Preferred minimally invasive fix: project every stage RHS by field spin

After each subsystem computes its RHS, analyze/synthesize the entire RHS field into its proper retained `(s,m,ell<=ell_max)` band before it is consumed downstream or stored as an RK stage derivative.

Projection groups are:

- spin `-2`: first `P,Q,Psi`; `Lambda`; `B`; second `P,Q,Psi`;
- spin `-1`: `G`, `Pi`, `C`;
- spin `0`: `H`, `U`.

The stage graph should be:

1. compute first-order Teukolsky RHS;
2. project first `P,Q,Psi` RHS;
3. compute reconstruction using the projected first-order tangent;
4. project all seven reconstruction RHSs;
5. compute the tangent/JVP from projected first/reconstruction tangents;
6. project tangent fields;
7. build and project `D,T` as already done;
8. compute second-order RHS;
9. project second `P,Q,Psi` RHS.

Because the RK stage begins bandlimited and every `k_i` is bandlimited, all intermediate stages remain bandlimited.

#### More invasive alternative

Evolve modal coefficients directly and use the padded nodal grid only as scratch for products and coefficient multiplication.

### Required regression tests

For randomized bandlimited states at nonzero Kerr spin, verify for every evolved field

\[
\|(I-P_{s,m})\,\mathrm{RHS}\|\lesssim\epsilon_{\rm roundoff}.
\]

Also verify after each complete RK step

\[
\|(I-P_{s,m})U\|\lesssim\epsilon_{\rm roundoff}.
\]

Repeat for the highest retained `|m|`, where the nullspace is largest.

---

## Finding 3 — BLOCKER FOR SECOND-ORDER SCIENCE: inconsistent default reconstruction initial data

### Current default

`initialize_compactified_gaussian_pulse()` constructs a bandlimited first-order `F`, sets

\[
Q=D_RF,
\]

and chooses `P` so that

\[
\partial_TF=0
\]

initially. All seven reconstruction fields and the second-order field are zero by default.

The full pipeline nevertheless evaluates the reconstruction and quadratic source immediately at every RK stage.

### Why zero reconstruction data are not generically valid

The seven implemented reconstruction transport equations do not constitute all of the independent linearized Einstein constraints on the initial Cauchy slice. The original scheme explicitly discusses this issue: self-consistent reconstruction data on the entire `T=0` slice would require solving first-order constraint equations. It instead begins reconstruction from a null surface outside the compact wave support and delays the physically interpreted second-order evolution until a later time `T_s`.

The legacy implementation monitors independent residuals, including Bianchi constraints and the reality/sharp constraint on `h_ll`. For example, with `G=Lambda=0`, the first independent Bianchi residual contains schematically

\[
-\thorn F+\rho_0F,
\]

which is not generically zero for an arbitrary Gaussian with `partial_T F=0`.

Therefore the early reconstructed fields do not necessarily represent a linearized vacuum metric, and the early quadratic source need not be the correct second-order Einstein source.

### Impact

The code may eventually advect the inconsistency out of a causal region, but the second-order field has already been driven by the inconsistent early source. That contamination cannot be removed merely by examining the solution later.

### Required fix

Implement the independent reconstruction constraints from the formalism/legacy code, at minimum:

1. the `Psi3` Bianchi residual;
2. the `Psi2` Bianchi residual;
3. the `h_ll` reality/sharp residual.

Then choose one of:

- solve constraint-consistent reconstruction initial data on `T=0`;
- initialize from a known complete perturbation such as a QNM solution;
- reproduce the causal null-reconstruction setup of the paper;
- as a temporary diagnostic mode, disable the quadratic source until the true independent residuals fall below a documented tolerance and until the causal inconsistency has left the region of interest.

The last option is useful for development but is not equivalent to complete second-order initial data.

---

## Finding 4 — HIGH: reported reconstruction residuals are not independent constraints

`PipelineReconstructionDiagnostics` recomputes the seven transport equations using the same state, stage RHS, radial derivatives, and angular inputs used by the production evaluator. The right-hand-side formulas are duplicated, which can catch a coding mismatch, but these are still residuals of the equations used to define the RHS.

They are not the independent Bianchi/reality residuals used in the original reconstruction analysis. Consequently:

- very small values do not establish that the reconstructed variables solve all linearized Einstein equations;
- the reported convergence can largely measure dissipation or numerical mismatch in an otherwise algebraic identity;
- documentation calling them “seven independent reconstruction residuals” is misleading.

### Fix

Rename them to `transport_equation_consistency_residuals` and add the genuinely independent residuals listed in Finding 3.

---

## Finding 5 — MEDIUM: zero-SAT boundary treatment is plausible but not proven stable

The characteristic calculation correctly indicates no incoming propagating mode at scri or the horizon for the stated sign convention, and zero physical boundary data are plausible. The repository itself correctly documents that no semi-discrete endpoint energy proof exists for the D4-2 reduction with its degenerate endpoint symmetrizer.

This is not a demonstrated continuum-equation error. It remains a long-time numerical-risk item, especially near extremality.

Required evidence before production claims:

- long-time outgoing-pulse convergence;
- frozen-coefficient/normal-mode tests;
- reflection measurements at both endpoints;
- high-spin resolution sequences;
- tests with both reduction strategies.

---

## Finding 6 — MEDIUM: reduction-constraint evolution is not exactly `C_t=-gamma C` when dissipation is active

Without dissipation, differentiating the complete `psi` RHS gives the desired reduction equation. With independent compatible-dissipation operators added to `Q` and `psi`, however,

\[
\partial_TC_Q
=-\gamma_QC_Q+\mathcal DQ-D_R(\mathcal D\psi).
\]

The last two terms do not generally cancel at the D4-2 boundary closures. Documentation should not state the exact continuum damping law without this qualification.

Add a discrete constraint-source test and verify that the extra term converges away at the expected rate.

---

## Finding 7 — MEDIUM: validation is implementation-internal rather than physical

The test suite is extensive and useful, but several key tests compare one implementation path against formulas or helpers that encode the same choice:

- the angular test enshrines the wrong symmetric eigenvalue;
- the spatial first-order test compares the Kokkos stencil path to the same compact point RHS;
- the source “oracle” duplicates the same expanded algebra;
- the full-pipeline RK test compares the trajectory against a finer trajectory of the same semi-discrete system.

These tests establish coding consistency, not the physical correctness of the full Teukolsky PDE.

Add independent physical regressions after the blockers are fixed:

1. Schwarzschild mode-by-mode analytic angular test;
2. known linear QNM frequency and damping rate;
3. comparison with the trusted Julia/Fortran linear solver at moderate spin;
4. angular `ell_max` and `N_theta` convergence;
5. first-order amplitude scaling and second-order quadratic scaling;
6. independent Bianchi/reality residual convergence;
7. a manufactured full PDE solution with nontrivial angular structure.

---

## Additional numerical cautions

### CFL estimate

The executable checks a radial characteristic CFL only. Angular spectral eigenvalues and explicit dissipation impose additional stability restrictions. Add a combined bound or determine a documented empirical stability rule as a function of `ell_max`, `N_theta`, radial spacing, and dissipation.

### “Exact product padding” is limited

The padding can exactly integrate polynomial spin-harmonic products over a retained band. Kerr background coefficients contain rational functions of `cos(theta)`, so multiplication by the background is not made exact by the same quadrature count. Independent `N_theta`/`ell_max` convergence is still mandatory.

### Horizon derivatives

The nine-point endpoint stencils are mathematically reasonable, but third and fourth derivatives strongly amplify grid noise. Every claimed Aretakis exponent must be demonstrated over multiple radial resolutions and dissipation strengths. These are coordinate derivatives of the rescaled code field; they are not automatically tetrad/gauge-invariant physical observables.

### Angular truncation of quadratic daughters

The quadratic source can generate modes up to roughly twice the seed band. A fixed global `ell_max` truncates those daughters. Nonlinear science requires explicit `ell_max` convergence, not only radial convergence.

---

## Components that passed the static audit

The following parts agree with the corrected specification or are structurally sound:

- Kerr background `mu0`, `tau0`, `pi0`, and `psi20`;
- compactified coordinate factors;
- rescaled `Delta_n`, `eth_n`, and `ethprime_n` point formulas;
- first-order `C_T`, `K`, `H_R`, `G_m`, and lower-order coefficients;
- compact `Q_t = D_R(psi_t)-gamma C_Q` formulation;
- all seven metric-reconstruction transport equations;
- triangular reconstruction dependency order within each RK stage;
- deterministic signed-`m` registry;
- `X_m^sharp = conj(X_{-m})` semantics;
- ordered `(m1,m2)->m_target` source enumeration;
- corrected compact `D_12` and `T_12` source, including the half-factor;
- outer source and coordinate forcing normalization;
- analytic Jet/JVP source tangent;
- common-stage classical RK4;
- D4-2 SBP derivative and negative-semidefinite compatible dissipation structure;
- device-resident scratch/allocation strategy;
- pinned Kokkos submodule and minimal dependency design.

These are substantial successes and should be preserved during remediation.

---

## Priority remediation sequence

1. **Fix the spin-weighted Laplacian eigenvalue.**
2. **Project every stage RHS into the correct angular band.**
3. Add band-invariance tests and rerun all temporal/spatial convergence tests.
4. Implement the true independent reconstruction constraints.
5. Add consistent reconstruction/source-start initial-data handling.
6. Add a Schwarzschild pure-harmonic test and a known QNM regression.
7. Re-run first- and second-order convergence at several `ell_max`, `N_theta`, radial resolutions, timesteps, and dissipation strengths.
8. Only then begin long near-extremal/Aretakis measurements.

---

## Minimal acceptance gates after fixes

The code should not be declared scientifically ready until:

- `{}_{-2}Delta_Omega Y_22 = -4 Y_22` passes on host and device;
- every evolved stage RHS is bandlimited to roundoff;
- a complete RK step preserves band limitation;
- linear QNM frequencies converge to a trusted reference;
- the true independent reconstruction constraints converge;
- the second-order source is activated only on consistent reconstruction data;
- the driven solution remains fourth-order in time;
- angular, radial, temporal, and dissipation convergence are shown independently;
- high-spin horizon derivatives converge over a nontrivial time interval.

---

## Audit limitation

This audit inspected the public source at the commit above through the GitHub API and independently compared its formulas with the corrected handoff and the implementation paper. The available execution container could not clone GitHub or initialize the Kokkos submodule, and the repository has no public GitHub Actions workflow at this commit. Therefore I did **not** independently reproduce the repository's reported `121/121` executable test result. The findings above are source-level mathematical and numerical-method findings; the angular-operator error does not depend on building the code.
