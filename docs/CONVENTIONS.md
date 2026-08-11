# Solver conventions

This file fixes the conventions shared by every physics and numerical kernel.
The mathematical authority is `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
and `equation_spec_v2.yaml`.

## Coordinates and units

- Use horizon-penetrating hyperboloidal coordinates `(T, R, theta, phi)` with
  `r = L^2 / R`, future null infinity at `R = 0`, and the outer horizon at
  `R_H = L^2 / r_+`.
- Use `M = 1` internally. Any dimensional conversion belongs at input/output.
- The background tetrad is the rotated Kinnersley tetrad used by the handoff
  derivation, with `gamma = 0`; metric reconstruction uses outgoing radiation
  gauge.

## Complex and mode conventions

- Use `Kokkos::complex<double>` in production fields and `std::complex<double>`
  only in transparent host reference calculations.
- Expand fields as `X = sum_m X_m exp(i m phi)`.
- Store signed `m` values explicitly in a sorted `std::vector<int>`. Never rely
  on associative-container iteration order.
- Define `X_m^sharp = conjugate(X_{-m})`; this is not `conjugate(X_m)`.
- Precompute every ordered source pair `(m1, m2) -> mt` satisfying
  `m1 + m2 = mt`. Both orderings contribute, and `m = 0` is stored once.

## Field metadata and rescaling

| Physical field | Stored field | spin `s` | boost `b` | definition |
|---|---:|---:|---:|---|
| `Psi4^(1)` | `F` | -2 | -2 | `Psi4^(1) = R F` |
| `Psi3^(1)` | `G` | -1 | -1 | `Psi3^(1) = R^2 G` |
| `Psi2^(1)` | `H` | 0 | 0 | `Psi2^(1) = R^3 H` |
| `lambda^(1)` | `Lambda` | -2 | -1 | `lambda^(1) = R Lambda` |
| `pi^(1)` | `Pi` | -1 | 0 | `pi^(1) = R^2 Pi` |
| `h_barm_barm` | `B` | -2 | 0 | `h_barm_barm = R B` |
| `h_l_barm` | `C` | -1 | 1 | `h_l_barm = R^2 C` |
| `mu h_ll` | `U` | 0 | 1 | `mu h_ll = R^3 U` |

The evolved last variable is `mu h_ll`, not `h_ll`. Background rescalings are
`mu = R mu0`, `tau = R^2 tau0`, `pi = R^2 pi0`, and
`Psi2^(0) = R^3 psi20`.

## Angular normalization

- Use unit-normalized spin-weighted spherical harmonics and the raising action
  `R_s {}_sY_lm = sqrt((l-s)(l+s+1)) {}_(s+1)Y_lm`.
- Use the lowering action
  `L_s {}_sY_lm = -sqrt((l+s)(l-s+1)) {}_(s-1)Y_lm`.
- Nonlinear products must preserve `m3 = m1 + m2` and `s3 = s1 + s2` and be
  checked against the generalized spin-weighted Gaunt coefficient in the
  implementer reference.

## Grid and storage

- Radial storage runs from scri (`R = 0`) to the outer horizon (`R = R_H`).
- Start with explicit fourth-order finite differences and one-sided boundary
  closures; add verified SBP operators without removing the readable reference
  operator.
- Use the logical field ordering `(mode, field, radial, theta)`, with `theta`
  contiguous in `Kokkos::LayoutRight`. Dominant kernels parallelize over mode,
  radius, and angle through the active Kokkos execution space.
- No allocations or full-field host copies are permitted inside timesteps.

## Evolution and source semantics

- The Teukolsky state is `(P, Q, psi)` with `Q = partial_R psi`. The first- and
  second-order systems use the same operator; only the second-order forcing
  differs.
- Classical RK4 is the correctness integrator. At every common stage, build
  the first-order state and RHS, all seven reconstruction states and RHSs,
  source values and analytic tangents, the outer quadratic source, and finally
  the second-order RHS.
- Historical endpoint interpolation is forbidden. Product tangents use
  `d_t(AB) = (d_t A)B + A(d_t B)` at the same RK stage.
- Compatible dissipation belongs in the method-of-lines RHS. A transformed
  state invalidates all cached derivatives and RHS values.
- The corrected source term is
  `0.5 * (eth + conjugate(pi) + 2*tau) h_l_barm`; the factor `0.5` multiplies
  the entire parenthesis.

## Diagnostics and interpretation

- Record the reduction constraint, reconstruction residuals, total and
  per-family/per-ordered-pair sources, and horizon transverse derivatives.
- Record `kappa` and `kappa*T` for near-extremal runs.
- Raw horizon `Psi4^(2)` in this tetrad is convention-fixed, not generally a
  gauge/tetrad-invariant observable. Physical horizon claims require a regular
  tetrad or infalling-observer diagnostic.
