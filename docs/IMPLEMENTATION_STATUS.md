# Implementation status

Evidence last updated: 2026-08-11, after runtime finalization and the
independent-audit remediation.

## Remediation status

- [x] Teukolsky angular operator changed to the lower-after-raise eigenvalue
      `-(ell-s)(ell+s+1)` on host and device. The symmetric operator remains
      separately named for diagnostics.
- [x] Complete first-order, reconstruction, tangent, and second-order RK stage
      derivatives are projected into the correct retained fixed-`m` spin band
      before downstream use.
- [x] Independent `Psi3` and `Psi2` Bianchi residuals plus the `h_ll`
      reality/sharp residual are evaluated at every stage and reduced into
      production diagnostics.
- [x] The former seven "reconstruction residuals" are explicitly named
      transport-equation consistency residuals.
- [x] The default Gaussian startup is labelled reconstruction-inconsistent.
      Constraint-aware mode suppresses second-order forcing until a causal
      start time and independent-constraint tolerance both pass.
- [x] Source policy and monotonic accepted-state activation are strict
      checkpoint-v4 metadata and are checked against the caller pipeline on
      write and load.
- [x] The compiled solver is controlled by strict versioned runtime files,
      writes a fully resolved provenance file, supports independent parent and
      daughter bands, and rejects incomplete daughter sets by default.
- [x] Linear-only operation freezes all ten reconstruction/second-order RHS
      fields exactly while retaining the production angular, SBP, projection,
      and common-stage device-RK4 path for the first-order triple.
- [x] Corrected ordered-pair source algebra, explicit sharp lookup, analytic
      JVP, common-stage RK4, D4-2, and Kokkos-neutral kernels were preserved.

Earlier runs at or before audited commit `ad12c65` used the incorrect angular
operator and lacked full-stage band projection. They are not scientific
validation of either perturbative order. Historical performance measurements
are retained in `PERFORMANCE.md` only as explicitly unqualified throughput
data.

## Current hard evidence

- C++ tests: 145/145 on Serial, OpenMP, and Intel Arc B580 SYCL.
- Symbolic source audit: 94/94 with SymPy 1.14.0 on all three build trees.
- Angular regressions: `s=-2`, `ell=2,3,4` give `-4,-10,-18`; explicit
  raising/lowering composition, host/device parity, and the Schwarzschild
  scri `Y_22` action pass. The test rejects the former shift by `-s`.
- Band invariance: relative `(I-P)` residual is below `2e-12` for all 13 RHS
  fields and an RK stage, and below `3e-12` after a complete RK4 step. Modes
  include every signed `m` from `-ell_max` through `ell_max`.
- Production-path QNM regression: the `M=1` Schwarzschild fundamental
  gravitational `ell=m=2` complex-frequency error falls from `2.76e-3` to
  `2.18e-4` under radial refinement, with the final recurrence residual
  `7.57e-8`. At Kerr `a/M=0.7`, angular-band refinement converges to
  `(omega_R,-omega_I)=(0.532995,0.0811631)`, within `5.41e-4` of the
  independent Leaver target.
- Independent reconstruction constraints: radial refinements
  `N_R=17,33,65` exceed ratio `3.5` for both Bianchi residuals; angular
  refinements `ell_max=2,4,6` decrease monotonically and reach below
  `6e-7` (`Psi3`) and `8e-7` (`Psi2`). A separate nonpolynomial Kerr
  projection converges as `N_theta=7,10,14` with fine modal error below
  `1e-9`.
- The seven transport-equation consistency residuals retain their exact and
  radial-refinement tests, but are not used as independent constraints.
- The full nonlinear common-stage trajectory retains fourth-order temporal
  convergence with both refinement ratios above 12. Corrected half-factor,
  scalar ordered-pair oracle, signed sharp, Gaunt/product, source amplitude,
  and source-tangent regressions all pass.
- Compatible dissipation satisfies the SBP negative-semidefinite identity.
  Its reduction-constraint commutator is nonzero at D4-2 closures and refines
  at the tested first-order pointwise and order-3/2 SBP-norm rates.
- Fresh post-remediation checkpoint parity over 7,735 complex values: Serial
  and OpenMP are bitwise identical; Serial/B580 RMS difference is
  `6.57e-20`, maximum `2.68e-18`, relative maximum `3.12e-16`.

Exact commands and the bounded spin/dissipation campaign are recorded in
`POST_AUDIT_REMEDIATION.md`; QNM methods and measured sequences are in
`PRODUCTION_QNM_VALIDATION.md`, and the runtime schema is in
`RUNTIME_CONFIGURATION.md`.

## Tested backends

- Kokkos Serial: GCC 13.3.0.
- Kokkos OpenMP: GCC 13.3.0.
- Kokkos SYCL: IntelLLVM 2025.3.2, Unified Runtime Level Zero V2, Intel Arc
  B580 (`intel_gpu_bmg_g21`), driver `1.15.38308+1`.
- CUDA/HIP: not built or run; no parity claim is made.

## Current scientific scope and limitations

- Constraint-aware startup is a causal development approximation, not a
  general solution of reconstruction constraints on `T=0`. The default
  Gaussian keeps the physical second-order forcing disabled because its true
  Bianchi residuals exceed tolerance. `source_mode=unrestricted` is only for
  explicitly labelled algebra/order tests.
- Zero SAT is consistent with the continuum no-incoming-mode count, but a
  nondegenerate endpoint symmetrizer and semi-discrete stability proof remain
  absent. No general long-time or near-extremal stability theorem is claimed.
- D4-2 has fourth-order interior and second-order boundary closures. Full-state
  radial error decreases in the manufactured test, while the reduction and
  independent-constraint convergence claims are stated separately at their
  measured orders.
- Post-audit `T=0.01` runs are finite for `a/M=0,0.7,0.99,0.999`. Separately,
  the production-path `a/M=0.7` fundamental QNM is a bounded regression.
  Neither result qualifies near-extremal QNMs or Aretakis measurements.
  Third/fourth horizon derivatives are noise-sensitive; the fourth derivative
  did not show a resolved high-spin sequence and is not qualified.
- The executable enforces a radial characteristic CFL check. Angular spectral
  and explicit-dissipation stability are bounded only by the documented tested
  envelope (`ell_max=4`, `N_theta=7`, `dissipation<=0.02`, radial CFL 0.1);
  this is empirical evidence, not a general combined stability proof.
- Exact-product padding applies to polynomial spin-harmonic products. Rational
  Kerr coefficients still require independent `ell_max` and `N_theta`
  convergence.
- Raw horizon `Psi4` and its coordinate derivatives are convention-fixed, not
  generally gauge/tetrad-invariant observables.
- Checkpoint publication uses atomic rename without platform-specific `fsync`;
  FNV-1a detects accidental corruption but is not authentication.
