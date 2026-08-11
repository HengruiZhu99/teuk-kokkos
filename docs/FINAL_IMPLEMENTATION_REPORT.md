# Final implementation report

## Implemented

The repository now executes the complete

```text
Psi4 first order -> seven-field ORG reconstruction
                 -> corrected ordered-pair quadratic source
                 -> driven Psi4 second order
```

pipeline on one common classical-RK4 stage state. It uses explicit signed
azimuthal modes and `X_m^sharp = conjugate(X_-m)`, spin-weighted angular
analysis/synthesis and GHP operations, exactness-derived nonlinear padding,
D4-2 SBP radial derivatives, compatible dissipation, and the same readable
physics kernels for Serial, OpenMP, and SYCL.

Production support includes a band-limited compactified Gaussian initializer,
all-field and reduction diagnostics, per-pair source retention, horizon radial
derivatives through fourth order, CSV diagnostics including per-pair `D`/`T`
source RMS and maxima, strict binary checkpoint/restart, three full-grid
examples, and backend snapshot comparison. No mandatory production dependency
besides pinned Kokkos was added.

## Verification

- Mathematical audit: 94/94 with SymPy 1.14.0.
- C++ suite: 121/121 on Serial, OpenMP, and Intel Arc B580 SYCL.
- The explicit legacy `0.5` connection-factor regression passes.
- Reconstruction residuals refine with combined ratios 24.83 and 20.50.
- The composed nonlinear pipeline has RK4 ratios 16.31 and 16.16.
- A checkpoint interrupted/reconstructed trajectory agrees with uninterrupted
  evolution within `2e-14`; the loaded state itself is bit exact.
- Serial and OpenMP parity is bit exact for the tested coupled trajectory.
- Serial and B580 differ by at most `2.69e-18` over 15,015 complex values.

## Science and hardware exercised

The highest spin tested is `a/M=0.999`. Runs at `N_R=33,65,129` complete
through `T=0.01`, and a zero-SAT `N_R=33` run completes through `T=1` with
finite endpoints and constraints. This is a software/numerical qualification,
not evidence for a physical Aretakis-like growth law.

The tested accelerator is an Intel Arc B580 through Level Zero with IntelLLVM
2025.3.2 and driver `1.15.38308+1`. A five-mode `4096 x 48` workload reaches
4.60 million grid-point-steps/s. CUDA and HIP were not available and are not
claimed.

## Remaining scientific limitations

- Zero SAT is selected because both endpoints have no incoming propagating
  characteristic in the implemented system. The endpoint-degenerate
  symmetrizer prevents a stronger nonzero-SAT energy claim from the supplied
  equations alone.
- Long near-extremal campaigns, convergence of extracted growth rates,
  dissipation/filter sensitivity, and a regular-tetrad horizon observable are
  still required before publishing an instability result.
- D4-2 boundary closures limit global spatial order; the report does not
  relabel boundary-limited evidence as fourth-order full-state convergence.
- The source dominates large B580 RHS time and is the next measured
  optimization target.

Exact commands and current status are in `README.md`,
`IMPLEMENTATION_STATUS.md`, and `PERFORMANCE.md`.
