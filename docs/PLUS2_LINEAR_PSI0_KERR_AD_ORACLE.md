# Rotating-Kerr numerical-AD oracle for linear `Psi0`

Status: executable host validation; no production `T0[h]` implementation

Base: `ccb624a`

## Result

`tools/symbolic/verify_linear_psi0_kerr_ad_oracle.py` closes the rotating
`a != 0` extension left open by the Schwarzschild symbolic oracle.  It
constructs the exact code-coordinate Kerr metric and a real outgoing-radiation
gauge perturbation, varies the coordinate Riemann tensor, and compares the
extreme curvature contraction with the corrected NP `T0[h]` formula.

The coordinate-curvature branch agrees with the corrected formula from
Campanelli--Lousto arXiv:gr-qc/9811019 Appendix A Eq. (A5), equivalently the
formula obtained by solving Loutrel et al. arXiv:2008.11770 Eq. (A9b).  It
rejects the opposite connection signs displayed in the latter paper's
Eq. (12):

| fixture | `a/M` | `(T,R,theta,phi)` | seed, `m` | corrected error | old-sign separation |
|---|---:|---|---|---:|---:|
| moderate | `0.63` | `(0.17,0.72,0.91,0.38)` | `2026081101, 1` | `2.44e-17` | `2.20e-2` |
| rapid | `0.91` | `(-0.23,1.08,1.17,-0.44)` | `2026081102, 2` | `6.26e-17` | `1.49e-1` |
| near extremal | `0.999` | `(-0.11,1.35,1.39,0.21)` | `2026081104, -2` | `1.50e-16` | `1.34e-1` |
| opposite | `-0.74` | `(0.31,0.57,0.73,0.62)` | `2026081103, 3` | `4.16e-17` | `3.89e-2` |

At the same points, the largest reconstructed-background Ricci component is
`1.20e-14` and the largest background extreme scalar is `3.56e-18`.

## Independent coordinate construction

The tool contains a small second-order multivariate jet type local to host
validation.  Each scalar carries its value, all four coordinate derivatives,
and the symmetric `4 x 4` coordinate Hessian.  Algebraic operations, complex
conjugation, trigonometric functions, exponentials, and a pivoted `4 x 4`
matrix inverse propagate these jets.  Analytic derivative fixtures directly
check the jet implementation before any geometry is evaluated.

Starting from Ripley et al. arXiv:2010.00162 Eq. (5), with Eq. (6) defining
`r=L^2/R`, the oracle forms

```text
g^(ab) = l^a n^b+n^a l^b-m^a mbar^b-mbar^a m^b
```

and obtains `g_ab` by jet-valued matrix inversion.  It independently verifies
the full null-tetrad normalization and `g^(ac) g_cb=delta^a_b`.  The physical
spin coefficients transcribed from Eq. (8) are checked against three exact
tetrad commutators, rather than against repository background helpers.

For deterministic pseudorandom smooth projections, the real ORG coordinate
metric is reconstructed as

```text
h_ab = h_ll n_a n_b
       -2 h_lm n_(a mbar_b) -2 h_lmbar n_(a m_b)
       +h_mm mbar_a mbar_b +h_mbar_mbar m_a m_b.
```

The tool checks all three input projections, `h_ab n^b=0`, zero trace,
coordinate reality, and symmetry.  Every fixture has nonzero first and second
derivatives in `T`, `R`, `theta`, and `phi`, plus nonzero mixed `T-R` and
`theta-phi` derivatives.  Coefficients come from fixed random seeds and the
angular dependence includes `exp(i m phi)` with `m=1,2,3`; runs are therefore
repeatable while covering generic complex fields.

Only values and coordinate jets are passed to the curvature branch.  It forms

```text
delta g^(ab) = -g^(ac) h_cd g^(db),
delta Gamma^a_bc = d/deta Gamma^a_bc[g+eta h] at eta=0,
delta R^a_bcd = d/deta R^a_bcd[g+eta h] at eta=0,
delta R_abcd = h_ae R^e_bcd + g_ae delta R^e_bcd,
Psi0_coordinate = -delta R_abcd l^a m^b l^c m^d.
```

This branch contains no NP spin-coefficient perturbation, `sigma1`, `kappa1`,
Ricci identity, or `T0` helper.  All Weyl trace-subtraction terms vanish in the
extreme null contraction.  A separate comparison branch constructs
`sigma1`, `kappa1`, and the two candidate outer operators from the primary NP
formulas.  Sharing only the manufactured input fields, exact background
tetrad, and scalar arithmetic keeps the curvature calculation algebraically
independent of the formula under test.

The final contraction uses the fixed background tetrad.  This equals the
varied extreme Weyl scalar on a type-D background by
Aksteiner--Andersson--Backdahl arXiv:1601.06084 Eq. (A1), so no untracked
first-order tetrad variation is omitted.

## Execution and scope

Run the focused validation with:

```sh
python3 tools/symbolic/verify_linear_psi0_kerr_ad_oracle.py
```

The script uses only the Python standard library and does not require
Mathematica, SymPy, NumPy, or production Kokkos code.  It validates the local
continuum identity at deterministic exterior points.  It does not implement
production `T0[h]`, test device parity for such an implementation, test
stage-local second tangents, or resolve the separately blocked normalized
radial TSI fixture.

The recorded run reports `113/113` named checks in `0.07 s`.  The existing
SymPy foundation, formalism-gate, compact-source, and Schwarzschild-coordinate
scripts also pass under `/tmp/teuk-audit-final/bin/python`.  The complete
Serial unit executable passes `157/157`; the complete preset passes `3/3`
CTest targets in `18.68 s`.
