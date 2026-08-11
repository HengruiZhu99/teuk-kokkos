# Historical pre-remediation performance

> **Not a current qualification.** These measurements predate the independent
> audit remediation. At that time the production Teukolsky angular eigenvalue
> was wrong and complete RK tangents were not projected into the retained
> angular bands. The numbers below are retained only as historical throughput
> data; they do not validate the equations and should not be used to predict
> current performance. A new benchmark is required after the corrected stage
> graph is frozen.

Measured 2026-08-11 before the independent-audit corrections. These were short
focused measurements, not vendor-tuned peak benchmarks.

## Environment and workload

- CPU: AMD Ryzen 9 7950X, 16 cores / 32 hardware threads.
- GPU: Intel Arc B580, Level Zero driver `1.15.38308+1`.
- Compiler: GCC 13.3.0 for Serial/OpenMP; IntelLLVM 2025.3.2 for SYCL.
- Kokkos: 5.1.0 at `3ec81abe1816109f6f62ac48cef41921f91a4d00`.
- Physics: five signed modes `{-4,-2,0,2,4}`, `ell_max=8`, all 13 fields,
  common-stage RK4, reconstruction, analytic source tangents, padded source
  projection, and second-order forcing.
- Timing: 10 evolution steps after construction/warmup; initialization,
  diagnostics, checkpoint I/O, and the separate profiled RHS are excluded from
  `evolution_wall_seconds`.

| Backend | `N_R` | `N_theta` | Evolution seconds | Grid-point-steps/s |
|---|---:|---:|---:|---:|
| Serial | 512 | 24 | 6.3659 | 96,514 |
| OpenMP | 512 | 24 | 5.35486 | 114,737 |
| Arc B580 SYCL | 512 | 24 | 0.225676 | 2,722,490 |
| Arc B580 SYCL | 1024 | 32 | 0.434054 | 3,774,640 |
| Arc B580 SYCL | 2048 | 32 | 0.780190 | 4,200,000 |
| Arc B580 SYCL | 4096 | 48 | 2.13839 | 4,597,100 |

At `512 x 24`, the measured B580/Serial throughput ratio is 28.2. The OpenMP
result is only 1.19 times Serial because this implementation launches many
small stage kernels; this is evidence of a remaining CPU launch/parallel-region
overhead, not a reason to maintain separate physics code.

## RHS stage-group profile

The solver can fence and time one diagnostic RHS after a run. Percentages for
the B580 were:

| Stage group | `512 x 24` | `4096 x 48` |
|---|---:|---:|
| First linear Teukolsky | 3.51% | 3.36% |
| Reconstruction | 18.12% | 12.63% |
| Tangent propagation | 25.84% | 15.89% |
| Quadratic source and projection | 49.30% | 64.74% |
| Second linear Teukolsky | 3.24% | 3.37% |

At the historical commit, the ordered-pair source was the measured
optimization target. Any new optimization decision must first repeat this
profile with the remediated pipeline. No equation-specific GPU fork is
justified by these historical measurements.
