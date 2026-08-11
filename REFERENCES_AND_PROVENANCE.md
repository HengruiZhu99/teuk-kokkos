# References and provenance

**Bundle version:** 2.0.0  
**Date:** 2026-08-11

## Primary mathematical sources

1. N. Loutrel, J. L. Ripley, E. Giorgi, and F. Pretorius, *Second Order Perturbations of Kerr Black Holes: Reconstruction of the Metric*, arXiv:2008.11770. The general second-order source is Eq. (B17), obtained from Eqs. (B14)-(B15).
2. J. L. Ripley, N. Loutrel, E. Giorgi, and F. Pretorius, *Numerical computation of second order vacuum perturbations of Kerr black holes*, arXiv:2010.00162. The specialized source is Eqs. (14)-(15); the independent derivation used Eqs. (B19), (B29)-(B36), and the coordinate forcing in Eq. (35).
3. M. Casals, S. E. Gralla, and P. Zimmerman, *Horizon Instability of Extremal Kerr Black Holes: Nonaxisymmetric Modes and Enhanced Growth Rate*, arXiv:1606.08505.
4. S. E. Gralla and P. Zimmerman, *Critical Exponents of Extremal Kerr Perturbations*, arXiv:1711.00855.
5. S. A. Teukolsky, *Perturbations of a rotating black hole. I. Fundamental equations for gravitational, electromagnetic, and neutrino-field perturbations*, Astrophys. J. 185, 635 (1973).

## Legacy code snapshot audited

Repository: `JLRipley314/teuk-fortran-2020`  
Commit: `0a12faef3461f04ef71c8d42517eaaba512e34aa`

Files used in the audit, with Git blob identifiers returned by GitHub:

- `src/mod_scd_order_source.f90`  
  Blob: `15bc1206422d0d862a6cacdb5479be8dc771656b`  
  Role: compact quadratic source, history derivative, and mode-pair accumulation.
- `src/mod_teuk.f90`  
  Blob: `a1864e987572708c4511a0ad1fbbdbf608c13560`  
  Role: linear/driven first-order reduction and RK staging.
- `src/mod_metric_recon.f90`  
  Blob: `8ef16c656b339f0b56cd20b4720a3a2a51a128cd`  
  Role: seven metric-reconstruction transport equations.
- `src/mod_ghp.f90`  
  Blob: `c56eb7ce458f0590f0fca438efc8d6c243f4c7b6`  
  Role: rescaled GHP operators.
- `src/mod_bkgrd.f90`  
  Blob: `9036c71d9bfa8d53cd5e91f35f24151cfe778f9c`  
  Role: background Kerr NP fields.
- `src/mod_field.f90`  
  Blob: `65f6deca83253af54be2af8c4729fe965e8294d5`  
  Role: RK state/cache semantics.

The legacy repository is a regression reference only. For the new solver, the source-of-truth order is:

1. `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`
2. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
3. `equation_spec_v2.yaml`
4. `SOURCE_TERM_LEDGER.csv`
5. `verify_second_order_teukolsky.py`

## Independent audit routes

- **Route I:** expand the general type-D second-order Bianchi/Ricci system and recover Eq. (B17) of arXiv:2008.11770.
- **Route II:** impose the outgoing-radiation-gauge and tetrad conditions, then collect Eqs. (B29)-(B36) into Eqs. (14)-(15) of arXiv:2010.00162.
- **Route III:** isolate the disputed coefficient by substituting Eq. (B19) directly into Eq. (B31), without using the final printed Eq. (15a).

All three routes give the same source. Route III proves that the factor `1/2` multiplies both the derivative and connection pieces of `(eth + bar(pi) + 2 tau) h_l_barm`.

## Reproducibility note

The executable audit uses exact SymPy algebra wherever possible, plus a manufactured forced-ODE convergence test. A Mathematica kernel was unavailable, so the Mathematica notebook was inspected for provenance but was not independently executed. This limitation is recorded rather than hidden.
