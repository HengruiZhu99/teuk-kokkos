# Standalone production gate for linear `Psi0`

Status: qualified point operator, deliberately disconnected from
`SpatialPipeline`

Base: `2cdbf6b`

## Implemented operator

`include/teuk/plus2_linear_psi0.hpp` implements the raw fixed-code-tetrad
linear curvature scalar in the repository's outgoing-radiation gauge.  The
ordinary-NP operations are kept visibly aligned with the primary formulas:

```text
sigma1 = 1/2 [D+2(epsilon_bar-epsilon)+rho-rho_bar] h_mm
         -(tau+pi_bar) h_lm,

kappa1 = [D-2 epsilon-rho_bar] h_lm
         -1/2 [delta-2 alpha_bar-2 beta+pi_bar+tau] h_ll,

Psi0 = [D-rho-rho_bar-3 epsilon+epsilon_bar] sigma1
       -[delta-alpha_bar-3 beta+pi_bar-tau] kappa1.
```

The first two lines are Loutrel et al. arXiv:2008.11770 Eqs. (C1c,C1e).
The last line is obtained by solving its exact Ricci identity (A9b) and is
independently printed in Campanelli--Lousto arXiv:gr-qc/9811019 Appendix A
Eq. (A5).  The code does not use the inconsistent connection signs displayed
in arXiv:2008.11770 Eq. (12).

The point kernel forms the exact directional derivatives from Ripley et al.
arXiv:2010.00162 Eq. (5).  Its background coefficients transcribe Eq. (8)
after the compact coordinate definition `r=L^2/R` in Eq. (6).  The existing
`kerr_background_point` values of `rho`, `epsilon`, `tau`, and `pi` are checked
against this transcription.  `alpha` and `beta`, which were not previously in
that background structure, are independently checked against Eq. (8f,g).

## Same-stage data contract

The API accepts three explicit physical-metric stages:

- `h`: `h_ll`, `h_lm`, and `h_mm` at the current RK stage;
- `h_T`: the analytic same-stage tangent;
- `h_TT`: the analytic tangent of `h_T`.

The typed spatial structure supplies `R`, `TR`, `RR`, `theta`, `T-theta`,
`R-theta`, `theta-theta`, `phi`, `T-phi`, `R-phi`, `theta-phi`, and `phi-phi`
slots.  The explicit slots make missing second derivatives a compile-time
construction issue rather than allowing endpoint interpolation.  For a signed
mode, `plus2_fill_modal_azimuthal_derivatives` fills all five azimuthal slots
with exact `i m` products.

The helper `plus2_org_metric_from_reconstruction` maps the stored fields as

```text
h_mm = R B_sharp,
h_lm = R^2 C_sharp,
h_ll = R^2 U/mu0.
```

It accepts `B_sharp` and `C_sharp` explicitly and cannot manufacture them from
the same signed mode.  The caller must first use
`X_m^sharp=conjugate(X_(-m))`.

Internally, a fixed-size coordinate jet applies product and quotient rules to
the stationary tetrad and background.  It has no allocation, dynamic extent,
virtual dispatch, host-only dependency, or persistent state.  The public
result exposes `sigma1`, `kappa1`, their analytic time tangents, and raw
`psi0_code_tetrad`.  No file in the spatial pipeline includes this header yet.

## Regularized output and scri contract

Ripley et al. Eq. (21b) fixes

```text
Psi0_code = W_plus Z_plus,
W_plus = R^5/(L^2-i a R cos(theta))^4.
```

At every `R>0`, including the future horizon, the helper returns the exact
ratio `Z_plus=Psi0_code/W_plus`.  At scri both numerator and `W_plus` vanish.
Direct division would be `0/0`, and evaluating separately large cancelling
NP terms before division is not cancellation safe.  The scri overload
therefore requires an explicitly available analytic coefficient

```text
psi0_over_radius5 = lim_(R->0) Psi0_code/R^5,
Z_plus_scri = L^8 psi0_over_radius5.
```

This follows independently by taking the exact Eq. (21b) limit.  If that
coefficient is absent, or if `R<0` or `L<=0`, the result has `valid=false`.
The code never substitutes zero, divides by `R^5`, or guesses a boundary
coefficient.  A future pipeline integration must construct this coefficient
from the asymptotically factorized reconstruction graph before enabling scri
output.  This standalone gate intentionally does not claim that construction.

## Independent validation

`tests/plus2_linear_psi0_oracle_fixtures.hpp` contains point values exported
from the independent coordinate-Riemann tools.  Those tools reconstruct the
coordinate metric, vary Christoffels and Riemann directly, and contain no NP
`sigma/kappa` algebra in the curvature branch.  Production agrees with:

| fixture | `a/M` | absolute `Psi0` error |
|---|---:|---:|
| Schwarzschild axisymmetric | `0` | below `8e-13` |
| deterministic moderate Kerr | `0.63` | below `8e-13` |
| deterministic rapid Kerr | `0.91` | below `8e-13` |
| deterministic near-extremal Kerr | `0.999` | below `8e-13` |
| deterministic opposite-spin Kerr | `-0.74` | below `8e-13` |

The underlying oracle comparisons close at `2.44e-17` through `1.50e-16`;
the looser C++ threshold allows decimal fixture serialization without making
the production formula its own oracle.  Further tests cover:

- independent `alpha`, `beta`, `epsilon`, `rho`, `tau`, and `pi` values;
- exact common-amplitude linearity of every metric derivative slot;
- central-difference validation of the analytic `sigma1_T` and `kappa1_T`
  against the explicit `h_TT`, `h_TR`, `h_Ttheta`, and `h_Tphi` inputs;
- signed sharp reconstruction and `Jet1` product/quotient tangents;
- exact modal azimuthal derivative construction;
- fail-closed scri behavior and exact scri/interior/horizon scaling;
- host/device equality and zero allocation during the point kernel.

The recorded Serial validation reports `173/173` unit tests in `0.53 s` and
`3/3` CTest targets in `20.28 s`.  The independent Schwarzschild coordinate,
rotating-Kerr AD, and linear-foundation scripts report `23/23`, `113/113`, and
`22/22` named checks respectively.

A focused Intel Arc B580 qualification used IntelLLVM/oneAPI `2025.3.2`, the
pinned Kokkos `5.1.0` commit `3ec81abe1816109f6f62ac48cef41921f91a4d00`,
and the repository's `sycl-intel-b580` preset with `SERIAL;SYCL`, OpenMP/CUDA/
HIP disabled, and `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.  `sycl-ls` selected
`Intel(R) Arc(TM) B580 Graphics` through Unified Runtime Level Zero V2.  The
focused `teuk_tests` binary passed `173/173` in `4.59 s`; a traced rerun showed
successful `urEnqueueKernelLaunch` calls in `ur::level_zero`, including the
named linear-`Psi0` device-parity/no-allocation test.  This is runtime evidence
for the focused unit binary only.  No external VRAM sampling was recorded, so
the observed `284108 KiB` maximum host RSS is not presented as device-memory
evidence.

## Deliberate boundaries and current TSI scope

This commit adds no state, scratch, launch, include, or member to
`SpatialPipeline`.  It does not derive the stage-local asymptotic
`Psi0/R^5` coefficient, wire radial/angular derivative production, or add
linear diagnostic output.

The earlier normalized-radial blocker has since been closed for two explicit
real-frequency separated fixtures.  `PLUS2_VALIDATION_AND_TSI_AUDIT.md`
records a Schwarzschild horizon-in mode which continues through ORG metric
reconstruction and `T0[h]`, and an independently reviewed moderate-Kerr
`a/M=0.6` fixture which pins the hatted angular/radial factors, signed partner,
and complex phase in the separated TSI.  No normalization was inferred from
an unhatted product.

Those fixtures do not yet provide a moderate-Kerr field-level `T0[h]`
comparison, a normalized QNM, a horizon endpoint value, or an evolved-field
residual.  They therefore qualify the named separated cases, not a general
production TSI constructor.
