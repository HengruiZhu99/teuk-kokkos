# Implementation status

Evidence last updated: 2026-08-11.

- [x] Scientific handoff integrity (`sha256sum -c SHA256SUMS`).
- [x] Live symbolic audit: 94 passed, 0 failed with SymPy 1.14.0.
- [x] Kokkos 5.1.0 pinned at
      `3ec81abe1816109f6f62ac48cef41921f91a4d00`.
- [x] Shared coordinate, rescaling, signed-mode, angular, layout, and RK-stage
      conventions.
- [x] Kerr background and compactified GHP point and device operators.
- [x] Deterministic signed-mode registry, explicit sharp lookup, spin-weighted
      transforms, Gaunt oracle, and exactness-derived product padding.
- [x] D4-2 diagonal-norm SBP derivative, compatible dissipation, and compact
      contiguous/strided device stencils.
- [x] First-order Teukolsky evolution and reduction constraint monitoring.
- [x] All seven reconstruction equations, staged dependencies, and independent
      residual refinement.
- [x] Corrected deterministic ordered-pair quadratic source, analytic stage
      tangents, projected outer source, and explicit legacy half-factor test.
- [x] One device-resident 13-field common-stage RK4 pipeline.
- [x] Band-limited Gaussian setup, production diagnostics, horizon derivatives
      through fourth order, and strict atomic checkpoint/restart.
- [x] Diagnostic-time `D`/`T` RMS and maxima for every deterministic ordered
      source pair.
- [x] Full-pipeline fourth-order temporal refinement: ratios 16.31 and 16.16.
- [x] Serial/OpenMP/B580 state parity and runtime tests.
- [x] Full-grid linear, daughter-mode, and near-extremal examples.
- [x] Focused performance measurements and stage-group profile.

## Current hard evidence

- C++ tests: 121/121 on Serial, OpenMP, and SYCL.
- Symbolic audit: 94/94 on all configured CTest trees.
- Reconstruction combined residual RMS at `N_R=17,33,65`:
  `2.26466e-7`, `9.12223e-9`, `4.44894e-10` (ratios 24.83, 20.50).
- Common-stage full-pipeline temporal RMS errors:
  `1.32397e-14`, `8.11787e-16`, `5.02436e-17` (ratios 16.31, 16.16).
- Serial/OpenMP checkpoint parity: bitwise identical over 15,015 complex values.
- Serial/B580 checkpoint parity: RMS `6.60e-20`, maximum `2.69e-18`, relative
  maximum `4.10e-16`.
- `a/M=0.999`, `T=0.01`, `N_R=33,65,129` runs all complete with finite
  endpoints. First reduction RMS decreases from `3.08e-9` to `2.68e-10` to
  `4.61e-11`; second-order RMS is nonzero at every resolution.
- `a/M=0.999`, `N_R=33`, `T=1` remains finite under zero SAT. Final first and
  second reduction RMS values are `1.41e-7` and `9.74e-14`.

## Tested backends

- Kokkos Serial: GCC 13.3.0.
- Kokkos OpenMP: GCC 13.3.0 on an AMD Ryzen 9 7950X.
- Kokkos SYCL: IntelLLVM 2025.3.2, Level Zero, Intel Arc B580, driver
  `1.15.38308+1`.
- CUDA/HIP: not tested on this machine.

## Known limitations

- The characteristic calculation finds no incoming propagating endpoint mode,
  so the current production choice is zero SAT. The natural symmetrizer
  degenerates at the endpoints; no general nonzero-SAT energy theorem is
  claimed. See `BOUNDARY_SAT_BLOCKER.md`.
- The `T=1` result is a finite workload qualification, not an Aretakis or
  nonlinear-instability result. Much longer, higher-resolution science runs
  and dissipation sensitivity studies remain research work.
- D4-2 has fourth-order interior and second-order boundary closures. The linear
  manufactured solution proves fourth-order time convergence and strong
  spatial reduction refinement; no unsupported global fourth-order
  full-state claim is made.
- Raw horizon `Psi4` in this tetrad is convention-fixed rather than a generally
  gauge/tetrad-invariant observable.
- Checkpoint publication uses atomic rename without platform-specific `fsync`;
  FNV-1a detects accidental corruption but is not authentication.
