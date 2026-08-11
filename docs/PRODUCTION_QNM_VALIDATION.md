# Production-path quasinormal-mode validation

Date: 2026-08-11

The dedicated `teuk.qnm` CTest exercises the same first-order production path
as a configured linear science run:

```text
Galerkin Gaussian initializer
  -> Kokkos signed-mode angular coordinator
  -> nodal analysis and lower-after-raise modal operator
  -> nodal synthesis
  -> D4-2 SBP radial stage kernel
  -> retained-band RHS projection
  -> caller-owned common-stage device RK4
  -> nodal observation and independent host modal analysis
```

The run uses `second_order.enabled=false`, so the reconstruction/nonlinear
graph is deliberately frozen. This is not the older host radial oracle: the
production coordinator, transforms, projections, radial kernel, state layout,
and RK driver all execute. Compatible dissipation is `0.005`; the zero-
dissipation production trace exposes a late grid-scale contaminant and is not
used to claim a clean QNM fit.

## Independent reference values

The Schwarzschild fundamental gravitational mode uses

```text
M omega_220 = 0.37367168 - 0.08896232 i.
```

For Kerr `a/M=0.7`, `s=-2`, `ell=m=2`, `n=0`, the regression uses

```text
M omega_220 = 0.5326002435510186 - 0.08079287315500745 i.
```

The Kerr value was evaluated at exactly `a/M=0.7` with the independent
Leaver/Cook-Zalutskiy implementation in `qnm` 0.4.4. The associated primary
references and data description are:

- E. Berti, V. Cardoso, and C. M. Will,
  [Phys. Rev. D 73, 064030 (2006)](https://arxiv.org/abs/gr-qc/0512160);
- L. C. Stein,
  [`qnm`: Kerr quasinormal modes](https://arxiv.org/abs/1908.10377);
- E. Berti's [Kerr QNM data page](https://pages.jh.edu/eberti2/ringdown/).

The test does not use the approximate spin-fit coefficients as its target.

## Measured sequences

The Schwarzschild real trace is fit over `45M <= T <= 65M` with a damped
conjugate-pole recurrence. Its complex-frequency error is the Euclidean error
in `(omega_R, -omega_I)`:

| `N_R` | `ellmax` | `N_theta` | fitted `omega_R` | fitted damping | complex error | recurrence residual |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 3 | 5 | 0.372401 | 0.0865129 | 2.76e-3 | 2.48e-6 |
| 25 | 3 | 5 | 0.373831 | 0.0885962 | 3.99e-4 | 4.67e-7 |
| 33 | 3 | 5 | 0.373884 | 0.0890142 | 2.18e-4 | 7.57e-8 |
| 33 | 4 | 7 | 0.373884 | 0.0890142 | 2.18e-4 | 7.57e-8 |

The regression requires both radial error reductions explicitly and demands
agreement across the two angular configurations to `1e-8`.

Zero-time-derivative Kerr data contain prograde and counterrotating branches,
so a complex two-pole recurrence is fit over `55M <= T <= 80M`. Spherical
bands are refined at fixed `N_R=65`:

| `ellmax` | `N_theta` | fitted `omega_R` | fitted damping | recurrence residual |
|---:|---:|---:|---:|---:|
| 3 | 5 | 0.532915 | 0.0811914 | 1.73e-6 |
| 4 | 7 | 0.532994 | 0.0811637 | 1.82e-6 |
| 5 | 8 | 0.532995 | 0.0811631 | 1.83e-6 |

The final angular increment is less than 5% of the preceding increment. The
converged complex-frequency error is `5.41e-4`; frequency and damping errors
are each below `5e-4`. These are bounded production-path regressions, not a
claim of publication-grade spectral extraction or near-extremal stability.

Run the evidence directly with:

```bash
ctest --test-dir build/serial -R teuk.qnm --output-on-failure
```
