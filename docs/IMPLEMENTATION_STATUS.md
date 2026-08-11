# Implementation status

Evidence last updated: 2026-08-11.

- [x] Scientific handoff integrity (`sha256sum -c SHA256SUMS`).
- [x] Live symbolic audit: 94 passed, 0 failed with SymPy 1.14.0.
- [x] Shared coordinates, rescalings, signed-mode, angular, layout, and RK-stage
      conventions.
- [x] Kokkos 5.1.0 pinned at
      `3ec81abe1816109f6f62ac48cef41921f91a4d00`.
- [x] Minimal CMake and dependency-free CTest harness.
- [ ] Kerr background and compactified GHP operators integrated.
- [ ] Deterministic signed-mode registry and angular operators integrated.
- [ ] Fourth-order radial finite differences and linear Teukolsky evolution
      integrated and converged.
- [ ] Seven reconstruction equations integrated and residual-converged.
- [ ] Corrected ordered-pair quadratic source integrated and oracle-tested.
- [ ] Fully common-stage first/reconstruction/source/second-order RK4 pipeline.
- [ ] Fourth-order manufactured driven convergence.
- [ ] End-to-end daughter-mode and resolution convergence.
- [ ] Near-extremal horizon diagnostics and representative multi-resolution
      high-spin run.
- [ ] Focused performance pass.

Current backend evidence:

- Kokkos Serial: bootstrap compile and device-space kernel smoke test pass.
- Kokkos OpenMP: not yet tested.
- Kokkos SYCL/Intel Arc B580: oneAPI 2025.3.2 and Level Zero enumeration are
  available; solver kernels are not yet runtime-qualified.
- CUDA/HIP: not tested on this machine.

Known limitations are the unchecked items above. In particular, this repository
must not yet be used to make physical instability claims.
