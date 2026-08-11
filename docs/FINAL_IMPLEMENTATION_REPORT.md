# Final implementation report

## Scientific correction

The independent audit at commit `ad12c65` invalidated the earlier first- and
second-order qualification. The production angular operator was the symmetric
spin-covariant Laplacian instead of Teukolsky's lower-after-raise operator;
padded nodal directions were evolved outside the retained harmonic band; and
the default zero reconstruction fields drove a second-order source before the
independent reconstruction constraints were satisfied. Results produced
before this remediation must not be interpreted physically.

The repaired pipeline now uses

```text
{}_s Delta_Omega {}_sY_lm = -(ell-s)(ell+s+1) {}_sY_lm,
```

projects every complete tangent for all 13 fields at the appropriate
dependency point, evaluates independent `Psi3`/`Psi2` Bianchi and `h_ll`
reality/sharp residuals, and defaults to a visible causal/constraint-aware
second-order source gate. The seven equation-defining reconstruction checks
are retained under the accurate name "transport-equation consistency
residuals."

The corrected ordered-pair quadratic source, including the legacy half-factor
repair, signed-mode/sharp semantics, analytic source JVP, common-stage RK4,
D4-2 radial operator, and device-resident Kokkos architecture were preserved.

## Post-audit verification

- Mathematical source audit: 94/94 with SymPy 1.14.0.
- C++ suite: 133/133 on Serial, OpenMP, and Intel Arc B580 SYCL.
- Exact Teukolsky angular values, explicit eth composition, host/device
  parity, and the separated Schwarzschild `Y_22` action pass.
- Every full-pipeline RHS field, an intermediate stage, and one complete RK4
  step lie in their retained angular bands to relative error at most `3e-12`.
- The Schwarzschild fundamental gravitational `ell=m=2` ringdown frequency
  and damping agree with their trusted values within `8e-4`.
- Independent Bianchi residuals converge under separate radial and angular
  refinement; Kerr-coefficient modal projection also converges independently
  with `N_theta`.
- Full driven development-mode evolution remains fourth order in time, while
  all source algebra, Gaunt/product, amplitude-scaling, and sharp-mode
  regressions remain green.
- A fresh 7,735-value checkpoint is identical on Serial/OpenMP and agrees
  between Serial/B580 to maximum `2.68e-18` and relative maximum `3.12e-16`.

The B580 run used IntelLLVM 2025.3.2, Kokkos 5.1.0, Unified Runtime Level Zero
V2, Intel Arc B580 architecture `intel_gpu_bmg_g21`, and driver
`1.15.38308+1`. Runtime logs loaded the Level Zero adapters and the executable
reported Kokkos execution space `SYCL`. CUDA and HIP were not available and
are not claimed.

## Startup and interpretation

The Gaussian initializer supplies a consistent first-order reduction
`Q=D_R Psi4` and zero initial `partial_T Psi4`, but it does not solve the
reconstruction constraints. Consequently, the production default does not
drive `Psi4^(2)` from those data. It computes the raw source for diagnostics
while applying zero forcing until `stage_time>=source_start` and the maximum
true independent residual is at most `source_constraint_tol`.

This is Option C from the audit: a causal startup approximation for development,
not general constraint-solved second-order initial data. Source activation and
the controlling residual are present in diagnostics and checkpoint metadata.
The explicit `unrestricted` mode exists to verify source algebra and RK order;
the corresponding example labels its output nonphysical.

## Bounded science evidence

Short `T=0.01` Serial runs completed at `a/M=0,0.7,0.99,0.999` with finite
diagnostics and the default source gate inactive. At `a/M=0.7`, `N_R=65`, the
first-order `Psi4` RMS approaches the zero-dissipation result monotonically as
dissipation is reduced from `0.02` to `0.01` to `0.005`; the differences from
zero dissipation halve to roundoff-consistent accuracy. Radial refinement from
`N_R=33` through 257 reduces the first reduction constraint from `8.17e-9` to
`2.61e-11`.

These are smoke and numerical-sensitivity checks, not publishable Kerr
ringdown or horizon-growth results. The fourth horizon coordinate derivative
is not resolved in the tested high-spin sequence. No Aretakis exponent,
long-time near-extremal stability, nonzero-SAT energy estimate, or physical
second-order Gaussian result is claimed.

## Remaining limitations

- A constraint-solved reconstruction initializer or a full null-surface causal
  reconstruction remains necessary for general second-order science.
- The continuum endpoint count supports zero SAT, but the D4-2 system still
  lacks a semi-discrete endpoint energy proof.
- With dissipation, the reduction law includes
  `Dcal(Q)-D_R(Dcal(Psi))`; the tested boundary commutator converges away but
  is not identically zero.
- The CLI radial CFL check is qualified only within the post-audit empirical
  envelope. A general combined radial/angular/dissipation stability bound is
  still absent.
- Nonlinear daughter truncation and rational Kerr coefficients require
  independent `ell_max`, `N_theta`, radial, timestep, and dissipation studies
  for each science campaign.
- Historical throughput numbers in `PERFORMANCE.md` predate remediation and
  require remeasurement.

Commands and exact evidence are in `POST_AUDIT_REMEDIATION.md`; conventions
are fixed in `CONVENTIONS.md`.
