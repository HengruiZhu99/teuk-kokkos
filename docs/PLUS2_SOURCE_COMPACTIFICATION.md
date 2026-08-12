# Compactification of the raw ORG spin `+2` source

**Status:** quadratic source compactification gate passed, conditional on the
separate linear `Z0/Z1` peeling gate.  The reviewed expression now has a
standalone device-callable implementation, but it is not connected to the
runtime pipeline.

This document continues the convention/source gate in
`docs/PLUS2_FORMALISM_GATE.md`.  It starts from the raw, ungauge-fixed
Campanelli--Lousto source, primes it, and only then uses the repository's
outgoing-radiation gauge and rotated Kinnersley tetrad.

## 1. Definite-weight physical source

Introduce the background GHP operators

```
d4p = thorn - 4 rho - rhobar,
d3p = eth   - 4 tau + pibar.
```

The explicit epsilon/alpha/beta terms in the equivalent ordinary-NP
operators are included in `thorn` and `eth` with the weights of their
arguments.  Define the first-order operator corrections

```
B0_1[f] = (ethprime_1 + pi_1)[f],
C1_1[f] = (eth_1      - 4 tau_1)[f],
D1_1[f] = (thorn_1    - 4 rho_1)[f],
Er_1[f] = (thorn_1    - rho_1-rhobar_1)[f],
Et_1[f] = (eth_1      - tau_1+pibar_1)[f].
```

Here a subscript `_1` on an operator means its first perturbation, not a
radial-rescaling label.  The raw ORG source is exactly

```
S0 = d4p( C1_1[Psi1] + 3 sigma Psi2 )
   + d3p( B0_1[Psi0] - D1_1[Psi1] - 3 kappa Psi2 )
   - 3 Psi2_background ( Er_1[sigma] - Et_1[kappa] ).
```

This grouping was verified by independently priming the four-line
ungauge-fixed source sign ledger.  It is not obtained by priming the existing
ORG-specialized minus-2 expression.

## 2. Stored primitives

Let

```
V = U/mu0,
h_ll       = R^2 V,
h_lbar     = R^2 C,
h_lm       = R^2 Csharp,
h_barbar   = R B,
h_mm       = R Bsharp.
```

The source uses the following regular primitives:

```
Psi0  = R^5 Z0,       Psi1  = R^4 Z1,       Psi2 = R^3 H,
sigma = R^2 Sig,      kappa = R^3 Kap,
rho_1 = R^3 Rh,       tau_1 = R^2 Ta,
pi_1  = R^2 Pi,       epsilon_1 = R^2 Ep.
```

`Z0` here is the curvature intermediate `Psi0/R^5`, not the evolved master
field called `Z_plus` in the equation proposal.  With Ripley's regular
variable,

```
Psi0 = [R^5/(L^2-i a R cos(theta))^4] Z_plus,
Z0   = Z_plus/(L^2-i a R cos(theta))^4.
```

The conversion factor is finite and nonzero throughout the exterior domain.

Sharp always means `X_m^sharp=conj(X_-m)`.  It is never replaced by
`conj(X_m)`.

The connection/curvature primitives derivable directly from the current ORG
metric are

```
Sig = 1/2 thorn_1(Bsharp)
    + 1/2 (rho0-rhobar0) Bsharp
    - R^2 (pibar0+tau0) Csharp,

Kap = thorn_2(Csharp) - rhobar0 Csharp
    - 1/2 eth_2(V) - 1/2 R (pibar0+tau0) V,

Rh  = 1/2 U + 1/2 ethprime_2(Csharp) - 1/2 eth_2(C)
    - 1/2 R pi0 Csharp - 1/2 R (pibar0+2 tau0) C,

Ta  = 1/2 Delta_2(Csharp) + 1/2 R mu0 Csharp
    - 1/2 R pi0 Bsharp,

Ep  = -1/4 Delta_2(V)
    - 1/4 R eth_2(C) + 1/4 R ethprime_2(Csharp)
    + 1/4 R (mu0-mubar0) V
    - 1/4 R^2 (pibar0+2 tau0) C
    - 1/4 R^2 (3 pi0+2 taubar0) Csharp.
```

For completeness, the two connection fields eliminated from the stable source
ledger have the stored definitions

```
alpha1 = R^2 Al,
Al = -1/4 Delta_2 C + 1/4 R(2 mu0-mubar0) C
   - 1/4 eth_1 B + 1/2 beta0 B
   - 1/4 R(pibar0+tau0) B,

beta1 = R^2 Be,
Be = -1/4 Delta_2 Csharp - 1/4 R(mu0+2 mubar0) Csharp
   + 1/4 ethprime_1 Bsharp + 1/2 alpha0 Bsharp
   - 1/4 R(pi0+taubar0) Bsharp,

Pi = -1/2 Delta_2 C - 1/2 R mubar0 C - 1/2 R tau0 B.
```

Here `alpha=R alpha0` and `beta=R beta0`.  `Al` and `Be` are audit
primitives, not inputs to the final pole-regular ledger: their potentially
singular background angular connections cancel exactly in `B12`, `C12`,
`D12`, and `Et12`.  The `Pi` line agrees with the currently evolved
reconstruction field.

Here `mubar0=conj(mu0)` is a background coefficient, whereas a sharp suffix
always denotes the mode operation defined above.

`Z0` and `Z1` require the cancellation-safe linear-curvature construction.
Their definitions are fixed, but their regularity is a dependency on the
linear `Psi0/Psi1` gate: the direct Ricci formulas contain lower-power pieces
that cancel only when the linearized field equations hold.  This source gate
does not assume those cancellations term by term or attempt to manufacture
them algebraically.

The corrected linear curvature identity is

```
Psi0 = (thorn-rho-rhobar) sigma - (eth+pibar-tau) kappa.
```

It follows from the Ricci identity and the exact Campanelli--Lousto Weyl
formula.  The separately displayed `Psi_0-1` equation in arXiv:2008.11770 has
the wrong connection signs and is not used.

## 3. Ordered-pair compact inner source

For every ordered pair `(m1,m2)->mt=m1+m2`, define five regular operator
corrections.  All operators below act on the field immediately following
them.

```
C12 = -Csharp1 Delta_4 Z1_2
    + 1/2 Bsharp1 ethprime_4 Z1_2
    + [ -3/2 Delta_2 Csharp1 - 1/2 ethprime_1 Bsharp1
        + R(-3/2 mu0+mubar0) Csharp1
        + R(5/2 pi0+1/2 taubar0) Bsharp1 ] Z1_2,

B12 = -C1 Delta_5 Z0_2
    + 1/2 B1 eth_5 Z0_2
    + [ 1/2 Delta_2 C1 + eth_1 B1
        + R(-2 mu0+1/2 mubar0) C1
        + R(pibar0+1/2 tau0) B1 ] Z0_2,

D12 = -1/2 V1 Delta_4 Z1_2
    + [ 1/2 Delta_2 V1
        + 5/2 R eth_2 C1 - 5/2 R ethprime_2 Csharp1
        + R(-5/2 mu0+1/2 mubar0) V1
        + R^2(5/2 pibar0+5 tau0) C1
        + R^2(7/2 pi0+taubar0) Csharp1 ] Z1_2,

Er12 = -1/2 V1 Delta_2 Sig2 - 3 Ep1 Sig2 + Epsharp1 Sig2
     - R(Rh1+Rhsharp1) Sig2,

Et12 = -Csharp1 Delta_3 Kap2
     + 1/2 Bsharp1 ethprime_3 Kap2
     - 1/2 (ethprime_1 Bsharp1) Kap2
     + R [ mubar0 Csharp1
           + (3/2 pi0+1/2 taubar0) Bsharp1 ] Kap2.
```

The antisymmetric angular pair in `Et12` is essential.  Direct substitution
of the Appendix-B coefficients gives

```
-3 beta1-alphabar1-tau1+pibar1
 = -1/2 bardelta(h_mm) + mubar h_lm
   + (-1/2 alpha-3/2 betabar+3/2 pi+1/2 taubar) h_mm.
```

Combining the first two angular pieces with `delta1(kappa)` yields
`1/2[h_mm ethprime(kappa)-ethprime(h_mm) kappa]`; the remaining connection is
`(3/2 pi+1/2 taubar)h_mm kappa`.  This form contains no pole-singular
`alpha` or `beta` coefficient.  An earlier draft incorrectly introduced
`Delta(Csharp)`, `tau1`, and `pibar1` terms a second time; the independent
ordinary-NP oracle rejects that expression.

They satisfy the exact physical definitions

```
C1_1[Psi1]_12 = R^6 C12,
B0_1[Psi0]_12 = R^7 B12,
D1_1[Psi1]_12 = R^6 D12,
Er_1[sigma]_12 = R^4 Er12,
Et_1[kappa]_12 = R^5 Et12.
```

Now define

```
J12 = R C12 + 3 Sig1 H2,
K12 = R B12 - D12 - 3 Kap1 H2,
Q12 = Er12 - R Et12,

J_mt = sum_(m1+m2=mt) J12,
K_mt = sum_(m1+m2=mt) K12,
Q_mt = sum_(m1+m2=mt) Q12.
```

Their physical weights and powers are

| regular field | physical quantity | spin | boost | physical power |
|---|---|---:|---:|---:|
| `J` | `C1_1[Psi1]+3 sigma Psi2` | 2 | 1 | `R^5 J` |
| `K` | `B0_1[Psi0]-D1_1[Psi1]-3 kappa Psi2` | 1 | 2 | `R^6 K` |
| `Q` | `Er_1[sigma]-Et_1[kappa]` | 2 | 2 | `R^4 Q` |

## 4. Regular outer source and endpoint proof

Using `Psi2_background=R^3 psi20`, the exact compact source is

```
S0/R^6 = thorn_5 J - (4 rho0+rhobar0) J
        + R eth_6 K + R^2(-4 tau0+pibar0) K
        - 3 R psi20 Q.
```

Every coefficient on the right is finite at `R=0` and at the future horizon.
Therefore

```
S0 = O(R^6),
```

which is stronger than the required `S0=O(R^3)` condition.  The coordinate
forcing uses the same normalization for both spins,

```
forcing_plus = (2 Sigma_BL/R) S0
             = 2 (L^4+a^2 R^2 cos(theta)^2) R^3 (S0/R^6).
```

It is finite and in fact vanishes as `O(R^3)` at scri.  At `R=R_H>0`, the
normalization is finite.  The denominators of every background coefficient
and weighted derivative are products of

```
L^2 +/- i a R cos(theta),
L^4 + a^2 R^2 cos(theta)^2,
```

whose squared modulus is strictly positive for `L>0`.  The horizon zero of
the radial `thorn` coefficient is a regular characteristic degeneracy, not a
pole.

This endpoint proof is conditional only on regular `Z0` and `Z1`, whose
construction is the separate linear-curvature gate.  There is no additional
cross-family cancellation required inside the quadratic source.

## 5. Verification and promotion rule

`tools/symbolic/verify_plus2_compact_source.py` provides two independent
routes:

1. ordinary-NP ORG operator perturbations, expanded from the primary metric
   and spin-coefficient formulas;
2. the compact GHP/rescaled ledger above.

It verifies exact identities where tractable, randomized analytic-field
evaluations for the full source, every CSV power/weight entry, ordered-mode
selection, and both endpoint limits.

The deterministic oracle ensemble is

| analytic-field seed | `(M,a,L; T,R,cos(theta),phi)` |
|---:|---|
| `230519332` | `(1,2/5,1; 1/7,2/11,1/3,1/5)` and `(6/5,7/10,4/3; -2/9,3/10,-2/5,2/7)` |
| `201000162` | the same two Kerr points |

At each point the ordinary-NP route and compact route agree separately for all
eleven connection/sharp primitives, the five operator-correction families,
and the full outer source at 60-digit working precision with relative
tolerance `1e-36`.  The ordinary route uses the coordinate tetrad, GHP
definitions, and Appendix-B metric formulas directly; it does not call or
copy a production expression.

## 6. Primary provenance and evidence ledger

| claim | primary authority | independent evidence |
|---|---|---|
| raw ungauge-fixed source | Campanelli--Lousto, arXiv:gr-qc/9811019 Eq. (9); Spiers et al. arXiv:2305.19332 supplemental `TeukolskySource0` | exact prime/sign check in `verify_plus2_formalism_gate.py` |
| GHP derivatives and rotated tetrad | Ripley et al. arXiv:2010.00162, Eqs. `edth_def`, `tetrad_IEF_HC`, `NP_IEF_HC` | coordinate ordinary-NP route in the compact oracle |
| ORG derivative and connection perturbations | Ripley et al. arXiv:2010.00162, Eqs. `lin_ops`, `pert_Rici_rot` | exact connection reductions and four Kerr evaluations |
| common coordinate normalization | Ripley et al. arXiv:2010.00162 source lines 1397--1404 (`2 Sigma_BL/R` for the NP equation and its spin `+2` analogue) | symbolic identity `(2 Sigma_BL/R)S=2(L^4+a^2R^2y^2)S/R^3` |
| `Psi0` Ricci construction | Campanelli--Lousto exact Weyl formula plus Appendix-A Ricci identity of arXiv:2008.11770 | explicit inconsistency audit of the latter paper's displayed `Psi_0-1` line |
| source endpoint order | the preceding primary formulas plus repository rescalings | term-ledger power audit and exact endpoint limits |

The compact quadratic-source subgate is passed.  The proposal as a whole may
be promoted to production authority only after the linear `Z0/Z1` gate
independently demonstrates its peeling cancellations; common-stage runtime
integration, GPU execution, and convergence qualification are subsequent
gates, not symbolic authority.

## 7. Standalone implementation boundary

`include/teuk/plus2_source.hpp` transcribes every one of the 51 rows in
`PLUS2_SOURCE_TERM_LEDGER.csv` into named family members.  It accepts explicit
regular primitives and derivative values rather than constructing `Z0` or
`Z1`, supports both `Complex` and `Jet1<Complex>`, and performs no allocation
or mode lookup in its point functions.  `make_plus2_pair_lookup` is a
setup-time helper that records both signed parent indices and the distinct
negative-mode indices required by sharp.

`tests/test_plus2_source.cpp` reflects the CSV IDs and family counts, compares
every named term with an independent `std::complex` host transcription at 16
deterministic Kerr/random-field points, checks family closure, signed ordered
pairs, sharp lookup, common-amplitude quadratic scaling, every Jet directional
derivative, allocation freedom, regular coordinate normalization, and Kokkos
device parity.  This module deliberately remains absent from
`SpatialPipeline`; its existence does not enable a sourced companion run.
