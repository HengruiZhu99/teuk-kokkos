# Goal: Correct and revalidate the first- and second-order Teukolsky solver

Work in the existing `teuk-kokkos` repository. Treat this as a focused scientific-remediation task, not a rewrite.

Read first:

- `teuk_kokkos_audit_ad12c650.md` if copied into the repository;
- `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`;
- `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`;
- `SOURCE_TERM_LEDGER.csv`;
- `docs/CONVENTIONS.md`.

Preserve the parts already correct: Kerr background, compact Teukolsky coefficients, reconstruction transport equations, corrected quadratic source, sharp-mode bookkeeping, ordered mode pairs, common-stage RK4, analytic source tangent, D4-2 radial operator, and Kokkos portability.

## Mandatory blocker 1: fix the angular Teukolsky operator

The production operator must satisfy

\[
{}_s\Delta_\Omega {}_sY_{\ell m}
=-(\ell-s)(\ell+s+1){}_sY_{\ell m}.
\]

The existing `s^2-l(l+1)` eigenvalue is wrong for the Teukolsky equation. Replace it with

```cpp
-(ell - spin) * (ell + spin + 1)
```

or the existing lower-after-raise helper. Keep any symmetric Laplacian under a distinct name only if genuinely needed elsewhere.

Add independent tests:

- `s=-2,l=2 -> -4`;
- `s=-2,l=3 -> -10`;
- explicit lower-after-raise equality;
- host/device parity;
- pure `{}_{-2}Y_{22}` Schwarzschild/scri angular RHS test.

## Mandatory blocker 2: eliminate the padded nodal nullspace

Every evolved RK stage derivative must be projected into the retained fixed-`m` spin-weighted harmonic band.

After each subsystem RHS, project by field spin before the result is consumed downstream:

- spin `-2`: first `P,Q,Psi`, `Lambda`, `B`, second `P,Q,Psi`;
- spin `-1`: `G`, `Pi`, `C`;
- spin `0`: `H`, `U`.

Required stage order:

1. first Teuk RHS;
2. project first RHS;
3. reconstruction chain using projected first tangent;
4. project reconstruction RHS;
5. tangent/JVP using projected fields;
6. project tangent RHS;
7. corrected source and existing `D,T` projection;
8. second Teuk RHS;
9. project second RHS.

Use preallocated scratch only. Do not introduce timestep allocations or host transfers.

Add tests for every field and every signed mode:

\[
\|(I-P)\mathrm{RHS}\|<C\epsilon,
\qquad
\|(I-P)U_{n+1}\|<C\epsilon.
\]

Include the largest `|m|` allowed by the registry.

## Mandatory blocker 3: make reconstruction/source initial data physically consistent

Implement the genuinely independent residuals from the formalism/legacy code:

- `Psi3` Bianchi residual;
- `Psi2` Bianchi residual;
- `h_ll` reality/sharp residual.

Rename the current seven diagnostics to transport-equation consistency residuals; do not call them independent constraints.

Provide a scientifically explicit source policy:

- preferably construct constraint-consistent reconstruction initial data; or
- reproduce the causal null-reconstruction setup; or
- provide a documented source-start gate based on causal time and independent residual tolerance as a development mode.

Do not drive `Psi4^(2)` from the default zero reconstruction data at `T=0` without an explicit warning/error.

## Revalidation

After the three blockers are fixed:

1. Run all existing tests.
2. Add angular-band invariance tests.
3. Add independent QNM/ringdown frequency regression.
4. Demonstrate temporal fourth order for the full driven pipeline.
5. Demonstrate radial convergence for the fields and true independent constraints.
6. Demonstrate angular convergence in both `ell_max` and `theta_nodes`.
7. Repeat at `a/M = 0, 0.7, 0.99, 0.999` where stable.
8. Vary dissipation and show the physical result converges.
9. Update README/status documents honestly; remove any production-ready claim until all gates pass.

Keep commits small and reviewable:

1. fix angular eigenvalue + tests;
2. add full-stage projection + invariance tests;
3. add independent reconstruction constraints;
4. add consistent source initialization/gating;
5. add physical regressions and convergence evidence.

Do not stop after compilation or after the old tests pass. The task is complete only when the corrected PDE and new independent tests pass.
