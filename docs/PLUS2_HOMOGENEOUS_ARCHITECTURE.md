# Homogeneous spin +2 architecture slice

Status: reviewable foundation only; not connected to the production pipeline

Base: `f82983e`

This slice establishes the exact regular field scaling, independently checks
the existing compact Teukolsky coefficients at spin `+2`, and provides an
allocation-safe storage boundary for a future passive companion.  It does not
implement `T0[h]`, a quadratic spin `+2` source, runtime configuration,
checkpointing, replay, or common-stage integration.

## Field definition

Ripley et al. arXiv:2010.00162 Eq. (21b), in the paper's horizon-regular code
tetrad and on its continuous cube-root branch, is

```text
Psi0_code = W_plus Z_plus,
W_plus = R (r-i a cos(theta))^-4
       = R^5/(L^2-i a R cos(theta))^4.
```

`include/teuk/plus2_field.hpp` implements this expression without a complex
power function.  The forward map is valid at both endpoints.  The inverse is
named `plus2_regularized_from_code_tetrad_interior` deliberately: at scri,
both a peeling `Psi0` and `W_plus` vanish, so the finite `Z_plus` value must be
obtained from the analytic `R^5` coefficient rather than a pointwise `0/0`.
The helper `plus2_scri_scaling_coefficient(L)` returns the exact coefficient
`L^-8`.

## Homogeneous operator evidence

`tests/ripley_eq22_oracle.hpp` is a host-only transcription of Ripley et al.
Eqs. (22)--(23), specialized directly to `s=+2` and the repository's
`exp(i m phi)` convention.  It intentionally does not call
`teukolsky_coefficients` and uses `std::complex` and standard trigonometric
functions rather than the implementation's Kokkos expressions.

`tests/test_plus2.cpp` compares every first-order-reduction coefficient
`C_T`, `K`, `H_R`, `G_m`, `q`, and `psi` against that oracle at deterministic
Schwarzschild, Kerr, negative-spin, and near-extremal points.  Separate tests
cover:

- exact scri limits and finite future-horizon limits;
- `H_R=0` at both characteristic endpoints;
- the explicit Schwarzschild reduction;
- the lower-after-raise angular eigenvalue
  `-(ell-2)(ell+3)`, including `0`, `-6`, and `-14` for
  `ell=2`, `3`, and `4`;
- host/device compilation of the field scaling.

These tests authorize reuse of the existing coefficient point function for a
future *homogeneous* `Z_plus` evolution.  They do not authorize a quadratic
source or a sourced companion result.

## Optional storage boundary

`Plus2CompanionStorage` is standalone and is not a member of
`SpatialPipeline`.  Its default state is disabled and contains only zero
extents plus a disengaged `std::optional`; it owns no Kokkos View and invokes
no Kokkos operation.  Instrumented tests count both Kokkos allocations and
parallel-for launches and require exact zero for disabled construction.
Consequently the existing primary path is source-identical and bitwise
unaffected by this slice.

The explicit `enabled(...)` factory allocates once:

```text
state             (mode, 3, radial, theta)  P_plus,Q_plus,Z_plus
angular_laplacian (mode, radial, theta)
radial_scratch    (mode, 5, radial, theta)  existing reduction layout
RK workspace      flat stage,k1,k2,k3,k4
```

It calls no evolution kernel; Kokkos may initialize newly allocated Views on
the active backend.  The state and RK shapes follow the existing
`evaluate_sbp_teukolsky_full_stage_rhs` and `DeviceRK4Workspace` contracts.
Signed modes, coordinates, and angular plans should be shared or constructed
by the future integration layer; duplicating them here would allocate data
before an execution design is reviewed.

## Common-stage integration gate

The presence of RK buffers is not permission to advance the companion with a
separate complete RK step.  Once a reviewed source exists, one coupled driver
must execute each stage in this order:

1. form the primary and companion stage states at the same RK abscissa;
2. evaluate the primary reconstruction state and analytic tangents there;
3. derive `Psi0^(1)` and the reviewed spin `+2` source from that same stage;
4. evaluate the passive companion RHS;
5. retain strict one-way data flow: companion data never enters the primary
   RHS.

Until that driver and the quadratic source ledger are reviewed, the storage
class remains intentionally unintegrated and the only supported result of
this slice is the homogeneous coefficient/scaling evidence.

## Linear-curvature erratum found during review

The earlier linear derivation note transcribed arXiv:2008.11770 displayed
Eq. (12).  That equation's outer `kappa` connection signs contradict the
paper's own exact Ricci identity (A9b).  Solving (A9b) agrees with the exact
Weyl formula in Campanelli and Lousto, arXiv:gr-qc/9811019 Appendix A Eq. (A5):

```text
Psi0 = (D-rho-rho_bar-3 epsilon+epsilon_bar) sigma
       -(delta-alpha_bar-3 beta+pi_bar-tau) kappa

     = (thorn-rho-rho_bar) sigma
       -(eth+pi_bar-tau) kappa.
```

`docs/PLUS2_LINEAR_DERIVATION_REVIEW.md` and the symbolic audit now record and
test this correction.  No production `T0[h]` implementation is included.
