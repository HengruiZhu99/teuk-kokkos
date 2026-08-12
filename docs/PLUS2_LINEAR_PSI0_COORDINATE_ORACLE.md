# Independent coordinate-tensor oracle for linear `Psi0`

Status: executable host validation; no production `T0[h]` implementation

Base: `337dda2`

## Result

`tools/symbolic/verify_linear_psi0_coordinate_oracle.py` independently
confirms the corrected linear curvature identity

```text
Psi0^(1) = (D-rho-rho_bar-3 epsilon+epsilon_bar) sigma1
           -(delta-alpha_bar-3 beta+pi_bar-tau) kappa1
```

against a direct coordinate variation of the Riemann/Weyl tensor.  It rejects
the opposite connection signs printed in arXiv:2008.11770 Eq. (12).

The test uses the Schwarzschild member `a=0`, `M=1`, `L=2` of the exact
hyperboloidal, horizon-penetrating code tetrad in arXiv:2010.00162
Eqs. (5)--(6).  This is not a flat-space limit: `rho`, `epsilon`, `alpha`, and
`beta` are nonzero, the coordinate Christoffels and background Riemann tensor
are nonzero, and the angular connection terms distinguish the two candidate
formulas.

For two smooth manufactured ORG metrics, including one with complex helical
`exp(i phi)` and `exp(2 i phi)` tetrad projections, the results are:

| fixture | coordinate vs corrected NP | coordinate vs old displayed signs |
|---|---:|---:|
| axisymmetric | `4.23e-18` error | `2.52e-3` separation |
| helical | `1.04e-17` error | `1.14e-2` separation |

The independently reconstructed background Ricci tensor is below
`1.2e-15` at both evaluation points.

The executable oracle reports 23 named checks:

```sh
/tmp/teuk-audit-final/bin/python \
  tools/symbolic/verify_linear_psi0_coordinate_oracle.py
```

On the recorded validation environment it completed in `3.08 s`.  The full
Serial preset also passed `3/3` CTest targets in `18.18 s` (`teuk.unit`,
`teuk.qnm`, and `teuk.config_integration`).

## Coordinate route

The oracle starts only from the contravariant code tetrad.  In the
repository's `+---` convention it constructs

```text
g^(ab) = l^a n^b+n^a l^b-m^a mbar^b-mbar^a m^b
```

and inverts this matrix for `g_ab`.  Given the three ORG projections and
their reality partners, it reconstructs the covariant coordinate
perturbation as

```text
h_ab = h_ll n_a n_b
       -2 h_lm n_(a mbar_b) -2 h_lmbar n_(a m_b)
       +h_mm mbar_a mbar_b +h_mbar_mbar m_a m_b.
```

Exact symbolic checks recover `h_ll`, `h_lm`, and `h_mm`, and verify the full
vector condition `h_ab n^b=0`, trace freedom, and coordinate-metric reality.

At each controlled point, the coordinate route evaluates `g`, `h`, and their
first and second coordinate derivatives.  It then forms, without any NP
identities,

```text
delta g^(ab) = -g^(ac) h_cd g^(db),
delta Gamma^a_bc = d/deta Gamma^a_bc[g+eta h] at eta=0,
delta R^a_bcd = d/deta R^a_bcd[g+eta h] at eta=0,
delta R_abcd = h_ae R^e_bcd + g_ae delta R^e_bcd.
```

The final oracle value is

```text
Psi0_coordinate = -delta R_abcd l^a m^b l^c m^d.
```

For this extreme null contraction, every Ricci trace-subtraction term in the
linearized Weyl tensor vanishes because `l.l=l.m=m.m=0`.  Hence this Riemann
contraction is exactly the Weyl contraction even though the manufactured
metric perturbations are not required to satisfy the linearized vacuum
equations.  Using the fixed background tetrad is valid for the extreme
linearized Weyl scalars by Aksteiner--Andersson--Backdahl
arXiv:1601.06084 Eq. (A1).

## Independent NP comparison path

Only the comparison branch uses `sigma1` and `kappa1`.  Its background spin
coefficients are not copied from the paper: the script derives them from the
coordinate tetrad commutators and checks the null normalization and relevant
commutator channels symbolically.  It evaluates both:

- the formula obtained by solving arXiv:2008.11770 Ricci identity (A9b),
  independently printed in Campanelli--Lousto arXiv:gr-qc/9811019 Eq. (A5);
- the inconsistent connection signs displayed in arXiv:2008.11770 Eq. (12).

No coordinate-curvature intermediate reuses `sigma1`, `kappa1`, an NP spin
coefficient formula, a Weyl-scalar helper, or production repository algebra.

## Scope and remaining gates

This closes the Schwarzschild manufactured part of the planned independent
linear-curvature test.  It proves the corrected connection signs in a curved
background and exercises time, compact radial, polar, and azimuthal
derivatives.  It does not yet prove:

- the rotating-Kerr `a`-dependent terms;
- numerical cancellation of individual `R^-3` and `R^-4` contributions at
  scri;
- stage-local construction of the second metric tangent in production;
- a phase-normalized comparison with separated Kerr modes;
- device parity for a future production `T0[h]` implementation.

The coordinate construction itself is therefore no longer blocked.  A direct
full-Kerr SymPy transcription was explored but not committed: exact inversion
and differentiation of the `a != 0` tetrad-derived metric caused expression
swell beyond a bounded validation test.  This is an unclosed rotating-Kerr
extension, not a blocker to the Schwarzschild sign oracle.  A numerical
dual-number or automatic-differentiation implementation is the preferred next
route for at least one `a != 0` fixture, followed by randomized point and
convergence tests.  The separate normalized radial TSI blocker documented in
`docs/PLUS2_VALIDATION_AND_TSI_AUDIT.md` is unchanged.
