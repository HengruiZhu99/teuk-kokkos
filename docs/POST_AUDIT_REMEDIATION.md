# Post-audit remediation evidence

Date: 2026-08-11

Audit authority: `audits/2026-08-11/teuk_kokkos_audit_ad12c650.md` and
`audits/2026-08-11/TEUK_KOKKOS_REMEDIATION_PROMPT.md`.

Audited source: `ad12c650c3f65a957c34b1a8a8f4a178671b02d1`.

Core remediation commits:

```text
0151757 fix: use Teukolsky lower-after-raise angular operator
ab61c9b fix: project full RK stages into retained angular bands
1176e23 feat: monitor independent reconstruction constraints
f03ce83 fix: gate second-order source on reconstruction consistency
1aa2d70 test: add QNM and independent-constraint convergence regressions
fbf7bd4 test: qualify angular quadrature and dissipative constraints
920ee2c fix: persist and report reconstruction source policy
5c2ac7a test: enforce band invariance per field and radial line
92f05c9 fix: label unrestricted source startup in solver output
```

All results at or before the audited commit are pre-remediation and are not
scientifically qualified.

## Equation and representation gates

The host and device Teukolsky operator now use

```text
lambda_Teuk(ell,s) = -(ell-s)(ell+s+1)
                   = s(s+1)-ell(ell+1).
```

Regressions require `(-2,2)->-4`, `(-2,3)->-10`, and `(-2,4)->-18`, compare
against explicit raising then lowering, and verify that the former symmetric
operator differs by `-s`. A pure Schwarzschild `{}_(-2)Y_22` at scri has
`P_T=-4 Psi` when the other point terms vanish.

The full pipeline projects complete tangents at each dependency seam. The
relative residual `||(I-P)X||/||X||` is below `2e-12` for every one of the 13
RHS fields and an explicitly formed RK stage, and below `3e-12` after one
complete RK4 step. The registry in this regression contains every signed mode
from `m=-4` through `m=4`; the `|m|=ell_max` one-dimensional bands therefore
exercise the largest padded nullspaces.

## Reconstruction and source gates

The independent residuals reproduce the legacy equations directly:

```text
C_Psi3 = R eth'_2 G + 4 R^2 pi0 G - thorn_1 F + rho0 F
         - 3 R^2 psi20 Lambda

C_Psi2 = psi20(-3 R^3 mu0 C - 3/2 R^3 tau0 B - 3 R^2 Pi)
         - R eth'_3 H - 3 R^2 pi0 H + thorn_2 G - 2 rho0 G

C_hll  = U/mu0 - U_sharp/conjugate(mu0).
```

Manufactured `N_R=17,33,65` residuals refine by more than 3.5 at both steps;
the fine `Psi3` and `Psi2` RMS values are each below `5e-7`. Independent
angular truncations `ell_max=2,4,6` decrease monotonically and reach below
`6e-7` and `8e-7`. A separate fixed-band Kerr-coefficient projection varies
`N_theta=7,10,14` against a 64-node reference and reaches modal error below
`1e-9`.

The default Gaussian test demonstrates that both Bianchi residuals are
nonzero while the exact sharp/reality residual vanishes. The default policy
therefore leaves `source_active=0` and zeroes the forcing while retaining the
raw source for diagnosis. A separate causal-gate regression independently
crosses the configured time and tolerance gates. The unrestricted path is
used only in explicitly named source-algebra and temporal-order tests.

## Validation ladder

The full post-audit build and test commands were:

```bash
cmake --build --preset serial
ctest --preset serial

cmake --build --preset openmp
OMP_NUM_THREADS=8 ctest --preset openmp

source scripts/source_oneapi_level_zero_gpu.sh
teuk_source_oneapi_level_zero_gpu
cmake --build --preset sycl-intel-b580
UR_LOG_LOADER=level:info ctest --preset sycl-intel-b580
```

Results:

| Backend | C++ | symbolic | runtime |
|---|---:|---:|---:|
| Serial | 133/133 | 94/94 | 0.84 s total CTest |
| OpenMP | 133/133 | 94/94 | 0.92 s total CTest, 8 threads |
| Arc B580 SYCL | 133/133 | 94/94 | 1.83 s total CTest |

The C++ ladder includes the corrected half-factor, ordered-pair scalar oracle,
sharp-mode tests, exact Gaunt/padded products, analytic source tangent,
quadratic amplitude scaling, full common-stage RK4 order, transport equations,
independent constraints, checkpoint/restart, and allocation-free device stage
paths.

The Schwarzschild `M=1`, `ell=m=2` Gaussian ringdown is fit on `45M--65M`.
Frequency and damping must each lie within `8e-4` of the standard fundamental
values `0.37367168` and `0.08896232`, and the two-pole recurrence residual must
be below `1e-5`. The values are consistent with the tabulation in the
[Living Reviews QNM review](https://pmc.ncbi.nlm.nih.gov/articles/PMC5253841/).

The full driven development-mode common-stage trajectory has both temporal
step-doubling ratios above 12, consistent with classical RK4. This verifies
the numerical stage graph; because it deliberately uses unrestricted startup,
it is not a claim of constraint-solved second-order Gaussian data.

## Spin, radial, and dissipation envelope

The following short command family was run on Serial with `ell_max=4`,
`N_theta=7`, signed modes `{-4,-2,0,2,4}`, radial CFL 0.1, and `T=0.01`:

```bash
./build/serial/teuk_solver spatial-pipeline \
  spin=<spin> nr=33 ntheta=7 ellmax=4 modes=-4,-2,0,2,4 \
  final_time=0.01 steps=200 diagnostic_every=200 dissipation=0.01
```

All diagnostics remained finite for `a/M=0,0.7,0.99,0.999`. The source gate
remained inactive, as required for the inconsistent zero-reconstruction
startup, and the second-order state remained zero. These are corrected
first-order smoke tests, not physical Kerr QNM comparisons.

At `a/M=0.7`, `N_R=65`, the first-order `Psi4` RMS at `T=0.01` was:

| dissipation | first `Psi4` RMS | difference from zero dissipation |
|---:|---:|---:|
| 0 | `8.00566465984203e-5` | 0 |
| 0.005 | `8.00566151522671e-5` | `3.14462e-11` |
| 0.01 | `8.00565837071865e-5` | `6.28912e-11` |
| 0.02 | `8.00565208202425e-5` | `1.25778e-10` |

The difference halves with the dissipation coefficient, demonstrating
convergence to the nondissipative result for this bounded workload. This is a
numerical-sensitivity result, not a general physical dissipation study.

With `a/M=0.7`, dissipation `0.01`, 400 steps, and otherwise the same setup,
the first reduction-constraint RMS refined as:

| `N_R` | constraint RMS |
|---:|---:|
| 33 | `8.16641e-9` |
| 65 | `6.99694e-10` |
| 129 | `1.17234e-10` |
| 257 | `2.60650e-11` |

The unweighted all-grid `Psi4` RMS changes with sample count and is not used to
assign a radial order. Field radial convergence is instead covered by the
manufactured full-state regression, which requires monotonic error reduction;
reduction and independent-constraint orders are stated separately.

For `a/M=0.999`, `N_R=33,65,129,257` also remain finite at `T=0.01` and the
first reduction constraint decreases from `3.08e-9` to `1.07e-11`. Higher
horizon coordinate derivatives amplify the initially tiny Gaussian tail: the
fourth derivative is not monotone in the fine sequence. It is explicitly not
qualified, and no horizon-growth exponent is reported.

## Backend identity and parity

Source and dependency identity at validation:

```text
branch                 main
Kokkos submodule       3ec81abe1816109f6f62ac48cef41921f91a4d00 (5.1.0)
Serial/OpenMP compiler GCC 13.3.0
SYCL compiler          IntelLLVM 2025.3.2
SYCL selector          ONEAPI_DEVICE_SELECTOR=level_zero:gpu
device                 Intel Arc B580 Graphics, intel_gpu_bmg_g21
runtime                Unified Runtime Level Zero V2
driver                 1.15.38308+1
```

`UR_LOG_LOADER=level:info` records loading
`libur_adapter_level_zero.so.0` and `libur_adapter_level_zero_v2.so.0`; the
executable reports `Kokkos execution space: SYCL`, and the verbose suite runs
the named Kokkos device kernels.

A fresh four-step checkpoint comparison used all 13 fields on a
`5 x 17 x 7` signed-mode grid, totaling 7,735 complex values:

| comparison | RMS | maximum | relative maximum |
|---|---:|---:|---:|
| Serial/OpenMP | 0 | 0 | 0 |
| Serial/B580 | `6.57453e-20` | `2.68217e-18` | `3.12156e-16` |

This is runtime qualification for the named tests and checkpoint workload on
the named device. It is not CUDA/HIP qualification or a broad GPU science-run
claim.

## Boundary, dissipation, and CFL qualification

The continuum characteristic audit finds no incoming propagating mode at
scri or the horizon, so zero SAT is the current fail-closed choice. The natural
symmetrizer degenerates at both endpoints, and no semi-discrete stability
theorem is claimed.

Without dissipation, `C_Q,T=-gamma C_Q`. With independent compatible
dissipation, the exact discrete law includes
`Dcal(Q)-D_R(Dcal(Psi))`. The new commutator test begins from `Q=D_R Psi`,
proves that the source is nonzero in the D4-2 closures, and observes refinement
ratios above 1.9 in the maximum norm and 2.6 in the SBP norm.

The executable's enforced timestep check is radial-characteristic only. The
post-audit finite-run evidence bounds an empirical envelope of `ell_max=4`,
`N_theta=7`, dissipation at most `0.02`, and radial CFL 0.1. No general
combined radial/angular/dissipation stability bound exists yet; runs outside
that envelope require their own timestep and resolution study.

## Completion boundary

The audit's three scientific blockers are repaired and protected by direct
regressions. The corrected pipeline is reviewable and suitable for further
first-order validation and constraint-aware second-order development. It is
not yet qualified for physical second-order Gaussian evolution, long-time
near-extremal stability, high-order horizon growth, or publication-grade
nonlinear measurements.

## Runtime-finalization follow-up

The finalization working tree atop `2165a2b` adds strict versioned runtime
configuration, separate parent/daughter bands, retained-Galerkin Kerr initial
data, monotonic accepted-state source activation, checkpoint-v4 persistence,
and a dedicated production-path QNM executable. `second_order.enabled=false`
now freezes all ten non-first-order RHS fields exactly; the first-order triple
still traverses the production angular, projection, D4-2 SBP, and device-RK4
path.

Fresh Release build trees used the pinned Kokkos submodule
`3ec81abe1816109f6f62ac48cef41921f91a4d00` (5.1.0):

| backend | C++ assertions | symbolic audit | CTest wall time |
|---|---:|---:|---:|
| Serial, GCC 13.3.0 | 145/145 | 94/94 | 20.84 s |
| OpenMP, GCC 13.3.0, 8 threads | 145/145 | 94/94 | 16.91 s |
| SYCL, IntelLLVM 2025.3.2, Arc B580 | 145/145 | 94/94 | 56.56 s |

The assertions are split into `143/143` fast unit assertions and `2/2`
dedicated QNM assertions. The B580 CTest timings were 12.97 s unit, 32.41 s
QNM, 10.46 s configuration integration, and 0.71 s symbolic audit.
`/usr/bin/time -v` reported 334,468 KiB maximum host RSS for that complete CTest invocation;
no peak-device-memory tool was available. Runtime enumeration identified
Intel Arc B580 Graphics, Unified Runtime Level Zero V2, driver
`1.15.38308+1`; both Level Zero adapters loaded and `teuk_solver backend`
reported `Kokkos execution space: SYCL`.

The production QNM values were backend-identical to the printed precision.
Schwarzschild radial refinement `N_R=17,25,33` reduced complex-frequency error
from `2.76e-3` through `3.99e-4` to `2.18e-4`. At Kerr `a/M=0.7`, spherical
band refinement `(ellmax,N_theta)=(3,5),(4,7),(5,8)` produced
`(omega_R,-omega_I)=(0.532915,0.0811914)`, `(0.532994,0.0811637)`, and
`(0.532995,0.0811631)`. The final result is within `5.41e-4` of the independent
Leaver target. Exact fit windows, residuals, references, and limitations are
recorded in `PRODUCTION_QNM_VALIDATION.md`.

The four checked-in configuration templates were each parsed and run through
one real production step with temporary output and tight timestep overrides.
All passed strict parsing, daughter completeness, angular capacity, radial
CFL, initialization, diagnostics, and evolution. Template step counts were
increased so the unmodified files also pass their own radial-CFL gate.

Representative commands were:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_SYCL=OFF -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-audit-final/bin/python
cmake --build build -j4
ctest --test-dir build --output-on-failure

cmake -S . -B build/qualification-openmp -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SYCL=OFF \
  -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-audit-final/bin/python
cmake --build build/qualification-openmp -j4
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores \
  ctest --test-dir build/qualification-openmp --output-on-failure

source scripts/source_oneapi_level_zero_gpu.sh
teuk_source_oneapi_level_zero_gpu
cmake -S . -B build/qualification-sycl-b580 -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_HIP=OFF -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-audit-final/bin/python
cmake --build build/qualification-sycl-b580 -j2
ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  ctest --test-dir build/qualification-sycl-b580 --output-on-failure
```
