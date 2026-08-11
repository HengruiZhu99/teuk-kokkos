# Legacy remediation guide

This document records how to use `teuk-fortran-2020` safely as a regression reference. It is not a recommendation to extend that code into the new production solver.

## 1. Confirmed algebraic source correction

In `src/mod_scd_order_source.f90`, the legacy compact source contains

```fortran
0.5_rp*R*hlmb%edth(:,:,m2_ang)
+ (R**2)*(conjg(pi_0) + 2.0_rp*ta_0)*hlmb%level(:,:,m2_ang)
```

The correct expression is

```fortran
0.5_rp*R*hlmb%edth(:,:,m2_ang)
+ 0.5_rp*(R**2)*(conjg(pi_0) + 2.0_rp*ta_0)*hlmb%level(:,:,m2_ang)
```

because Eq. (15a) contains

\[
\frac12(\eth+\bar\pi+2\tau)h_{l\bar m},
\]

with the factor `1/2` outside the entire parenthesis. The independent B31+B19 elimination gives the same result.

The file `legacy_source_factor_correction.patch` contains the minimal one-line patch.

## 2. The driven time-order defect is not a one-line fix

The legacy source stages are

\[
S_1=S_n,\qquad S_2=S_3=\frac{S_n+S_{n+1}}2,\qquad S_4=S_{n+1}.
\]

Their contribution to RK4 is exactly the trapezoidal rule, so the driven second-order field is globally second order for a time-dependent source. Retaining a fourth-order backward stencil for the endpoint source derivative does not change that conclusion.

A scientifically reliable repair requires a coupled stage-local integrator:

1. construct the first-order Teukolsky stage state;
2. construct all seven reconstruction stage states;
3. evaluate all source primitives at that same stage;
4. evaluate their needed time tangents/JVPs at that same stage;
5. construct the outer source;
6. advance the second-order stage.

Use the new Kokkos implementation rather than retrofitting this architecture into the legacy code.

## 3. Do not reuse an endpoint RHS after filtering

The legacy sequence computes `k5` from the unfiltered endpoint, filters the endpoint state, and then copies `k5` into the next step's `k1`. In general,

\[
F(\mathcal F U)\ne F(U).
\]

Either place dissipation in the method-of-lines RHS, recompute the RHS after filtering, or do not reuse `k5`.

## 4. Signed-mode bookkeeping

Use a sorted explicit signed-mode registry. Never rely on set iteration or infer a requested mode from the presence of its negative. Build a deterministic ordered-pair table

\[
(m_1,m_2)\mapsto m_t=m_1+m_2.
\]

Treat `m=0` as a single mode, not a positive/negative pair.

## 5. Minor cleanup

`scd_order_source_compute` calls `compute_DT("pre_edth_prime",...)` twice consecutively. The duplicate call has no mathematical effect but should be removed.

## 6. Regression policy

After applying the algebraic patch, the legacy code may be used for moderate-spin smooth-solution comparisons, but not as the truth oracle for:

- fourth-order convergence of the driven solution;
- multimode source bookkeeping;
- nonlinear angular de-aliasing;
- exact-extremal or very long near-extremal secular-growth measurements.
