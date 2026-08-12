# Spin `+2` companion: symbolic-authority gate

**Status:** derivation/specification proposal; **not production authority yet**
**Scope:** raw Campanelli--Lousto (CL) `Psi0^(2)` in the same first-order
outgoing-radiation gauge (ORG) and perturbed tetrad as the existing raw
`Psi4^(2)` evolution.

This note records what can be established exactly from the primary sources
without silently changing the second-order variable.  It deliberately does
not authorize a production `spin_weight=+2` path.  The remaining blockers are
listed at the end.

## 1. Primary-source ledger

The source snapshots used for this audit were:

| source | exact item | revision/checksum |
|---|---|---|
| Spiers--Pound--Moxon, arXiv:2305.19332 | arXiv TeX `Paper-2nd-order-schemes.tex` | SHA256 `1b50e50bda24a32d37c670092d952913f57d3df174ab259bf047640d1a7c4326` |
| Spiers supplemental notebook | `DrAndrewSpiers/NP-and-GHP-Formalisms-for-2nd-order-Teukolsky`, `NP-and-GHP-Formalisms-for-2nd-order-Teukolsky.nb` | git `e46a34ffa8fecd260f5a4b63d32f7a34028b91ff`; SHA256 `7e1d10b115ceb5fbc1a2b68c8d28e92826bed1449293f7e0905462d72062d0f2` |
| Campanelli--Lousto, arXiv:gr-qc/9811019 | arXiv TeX `2orden.tex` | SHA256 `77683b0e633a14d8f2becb113681bf092a56dcd261af22c6e2bef2b5449cc0ee` |
| Loutrel--Ripley--Giorgi--Pretorius, arXiv:2008.11770 | arXiv TeX `Formalism.tex` | SHA256 `ed56d43e6e0407f9c4495dc5ffb65f2bb51653fdf33979e0a745bb288efd852d` |
| Ripley--Loutrel--Giorgi--Pretorius, arXiv:2010.00162 | arXiv TeX `numerics_description.tex` | SHA256 `2518ef1168e552db4ca4fd07ee421fca33f0939ecfc2031d272842b61cbf955e` |

Relevant primary equations/sections are:

- arXiv:2305.19332 Eqs. (27)--(31), (36)--(39), (42), (45), and Appendix B;
- arXiv:gr-qc/9811019 Eqs. (2.8)--(2.12), including its explicit statement
  that the raw `Psi0` equation follows by tetrad interchange;
- arXiv:2008.11770 Eqs. (B14)--(B17), the general first-order tetrad/spin
  coefficient formulas, and the linear curvature formula for `Psi0^(1)`;
- arXiv:2010.00162 Eqs. (6)--(15), (21)--(22), (35), and Appendix B, especially
  the explicit rotated tetrad, ORG tetrad perturbation, and the common
  `s=+/-2` compactified master equation.

## 2. Convention map and the variable that must be evolved

### 2.1 Repository convention

The repository follows arXiv:2008.11770 and arXiv:2010.00162:

- signature `+---`;
- expansion `A=A^(0)+epsilon A^(1)+epsilon^2 A^(2)+...` (no factorial);
- rotated Kinnersley tetrad with `gamma=0`;
- ORG `h_ab n^b=0`, `g^(0)ab h_ab=0`;
- nonzero first-order metric components
  `h_ll`, `h_lm`, `h_lbar`, `h_mm`, and `h_barbar`;
- `X_m^sharp=conj(X_-m)`;
- raw second-order CL coefficient `Psi4^(2)` in that fixed perturbed tetrad.

The repository's NP Weyl definitions carry the overall sign used by
Loutrel et al. (Appendix A of arXiv:2008.11770).  Spiers et al. instead use
signature `-+++` and write the Weyl contractions with the opposite displayed
sign.  Consequently formulas cannot be copied termwise between the two papers
without a complete signature/Riemann/tetrad conversion.

### 2.2 Reduced versus raw second-order curvature

Spiers et al. distinguish

```
psi4^(2) = psi4L^(2) + psi4Q^(2),
psi4L^(2) = T[h^(2)].
```

Their supplemental explicit nonlinear source `S[delta^2 G[h^(1),h^(1)]]`
drives the **reduced** field `psi4L^(2)`.  Its primed counterpart drives
`psi0L^(2)=T'[h^(2)]`.  The current repository instead evolves the **raw CL
coefficient** `Psi4^(2)`.  Therefore the Spiers reduced source is not the
source for a same-variable companion to the current output.

The notebook makes this distinction mechanically visible: its early symbol
`TeukolskySource0` is the *linear* spin-`+2` stress-tensor source operator,
while the Results-section symbol `S0d2GGHP` is built by replacing that
stress tensor with `delta^2 G`.  It is therefore the reduced effective-Einstein-
source construction, not an ORG expansion of the raw CL source.

The companion specified here is consequently

```
Psi0_raw^(2) = coefficient of epsilon^2 in the exact NP Weyl scalar
```

in the repository's fixed first-order ORG/tetrad.  This is the exact primed
counterpart of the existing raw `Psi4^(2)`.  A future reduced-field option is
scientifically attractive, but it would require either converting the
existing minus-2 output to `Psi4L^(2)` or clearly outputting unlike variables.

## 3. GHP prime map

At the **ungauge-fixed** level, prime is the tetrad interchange

```
l <-> n,        m <-> mbar,
D <-> Delta,    delta <-> bardelta,
Psi0 <-> Psi4,  Psi1 <-> Psi3,  Psi2 -> Psi2,
kappa'=-nu, sigma'=-lambda, rho'=-mu,
tau'=-pi, beta'=-alpha, epsilon'=-gamma.
```

For a GHP type `{p,q}`, prime maps `{p,q}->{-p,-q}`.  Complex
conjugation maps `{p,q}->{q,p}`.  Prime is not complex conjugation and is not
the repository's sharp operation.  In particular it does not change Fourier
`m`; sharp changes the stored mode by `m -> -m` and conjugation.

Applied to the background outer operators used in the current source,

```
d4  = Delta + 4 mu + mubar
d3  = bardelta + 3 alpha + betabar + 4 pi - taubar
```

The first line is the repository's `gamma=0` specialization.  Before gauge
and tetrad specialization the ordinary-NP operator is
`d4=Delta+4mu+mubar+3gamma-gammabar`.

the primed operators are

```
d4p = D     - 4 rho - rhobar - 3 epsilon + epsilonbar,
d3p = delta - 3 beta - alphabar - 4 tau + pibar.
```

The shorter radial form without explicit epsilon terms is only a weighted GHP
`thorn` representation on a stated type; it is not the ordinary directional
derivative `D`.  When acting on the definite-weight inner sources, `d4p` and
`d3p` may be rewritten as the corresponding GHP operators, but the source
derivation below retains the unambiguous ordinary-NP form.

## 4. Exact ungauge-fixed raw spin `+2` source

Prime is applied here to the general, pre-gauge CL/Loutrel equation (the
repository's derivation calls this Route I), **not** to the ORG-specialized
minus-2 source.  Define

```
A0 = Delta - 4 gamma + mu,
B0 = bardelta - 4 alpha + pi,
C1 = delta - 2 beta - 4 tau,
D1 = D - 2 epsilon - 4 rho.
```

The exact vacuum raw-field equation is

```
O_plus Psi0^(2) = S0^(2),
```

with

```
S0^(2) =
 - [ d4p A0^(1) - d3p B0^(1) ] Psi0^(1)
 + [ d4p C1^(1) - d3p D1^(1) ] Psi1^(1)
 + 3 [ d4p sigma^(1) - d3p kappa^(1) ] Psi2^(1)
 - 3 Psi2^(0) [ (d4p^(1)+3 rho^(1)) sigma^(1)
                -(d3p^(1)+3 tau^(1)) kappa^(1) ].
```

The two sign changes in the last two lines follow from
`lambda'=-sigma`, `nu'=-kappa`, `mu'=-rho`, and `pi'=-tau`; they are not
optional convention choices.  Every term has spin `+2`, boost `+2` (GHP type
`{4,0}`), the same type as `O_plus Psi0`.

This formula has the same no-factorial perturbative normalization as the
current `Psi4^(2)` source.

The corresponding background operator is the prime of the current
minus-2 operator,

```
O_plus = (D-4 rho-rhobar-3 epsilon+epsilonbar)
               (Delta-4 gamma+mu)
       -(eth-4 tau+pibar)(ethprime+pi)-3 Psi2^(0).
```

This covariant equation and the coordinate master equation in section 6 are
two representations of the same homogeneous operator.

## 5. Direct ORG specialization

The repository's ORG/tetrad gives

```
D^(1)        = -(1/2) h_ll Delta,
Delta^(1)    = 0,
delta^(1)    = -h_lm Delta + (1/2) h_mm bardelta,
bardelta^(1) = -h_lbar Delta + (1/2) h_barbar delta,
gamma^(1) = mu^(1) = nu^(1) = 0.
```

Therefore `A0^(1)=0`, but unlike the minus-2 ORG specialization the final
quadratic operator-correction line does **not** vanish.  Direct substitution
into the ungauge-fixed equation yields

```
S0_ORG =
   d4p [ C1^(1) Psi1^(1) + 3 sigma^(1) Psi2^(1) ]
 + d3p [ B0^(1) Psi0^(1) - D1^(1) Psi1^(1)
                              - 3 kappa^(1) Psi2^(1) ]
 - 3 Psi2^(0) [ E_rho^(1) sigma^(1) - E_tau^(1) kappa^(1) ],
```

where

```
E_rho^(1) = d4p^(1)+3 rho^(1)
          = D^(1)-rho^(1)-rhobar^(1)
            -3 epsilon^(1)+epsilonbar^(1),

E_tau^(1) = d3p^(1)+3 tau^(1)
          = delta^(1)-3 beta^(1)-alphabar^(1)
            -tau^(1)+pibar^(1).
```

For an arbitrary scalar `f`, the three first-order operators can be retained
in the following readable ORG form:

```
B0^(1)[f] = -h_lbar Delta f + (1/2) h_barbar delta f
 + { (1/2 Delta - 2 mu + 1/2 mubar) h_lbar
     +(delta - 2 alphabar + pibar + 1/2 tau) h_barbar } f,

C1^(1)[f] = -h_lm Delta f + (1/2) h_mm bardelta f
 + { (-3/2 Delta - 3/2 mu + mubar) h_lm
     +(-1/2 bardelta - betabar + 5/2 pi + 1/2 taubar) h_mm } f,

D1^(1)[f] = -(1/2) h_ll Delta f
 + { (1/2 Delta - 5/2 mu + 1/2 mubar) h_ll
     +(5/2)(delta - 2 alphabar + pibar + 2 tau) h_lbar
     +(-5/2 bardelta + 5 alpha + 7/2 pi + taubar) h_lm } f.
```

These expansions were obtained solely from the ORG spin-coefficient formulas
in Appendix B of arXiv:2010.00162.  They are algebraically checked by
`tools/symbolic/verify_plus2_formalism_gate.py`.

The required first-order curvature/connection primitives are

| primitive | `(spin,boost)` | leading scri falloff | status from current reconstruction |
|---|---:|---:|---|
| `Psi0^(1)` | `(+2,+2)` | `R^5` | derivable, not stored |
| `Psi1^(1)` | `(+1,+1)` | `R^4` | derivable, not stored |
| `Psi2^(1)` | `(0,0)` | `R^3` | stored as `R^3 H` |
| `sigma^(1)` | `(+2,+1)` | `R^2` | derivable, not stored |
| `kappa^(1)` | `(+1,+2)` | `R^3` | derivable, not stored |
| `rho^(1)` | `(0,+1)` | `R^3` in radiative asymptotics | derivable, not stored |
| `tau^(1)` | `(+1,0)` | at least `R^2` | derivable, not stored |
| `alpha^(1), beta^(1), epsilon^(1)` | connection, use only in weighted combinations | expression-dependent | derivable, not stored |
| `pi^(1)` | `(-1,0)` | `R^2` | stored as `R^2 Pi` |

All are functions of the existing five real-metric ORG components.  No new
independent evolution field is mathematically required, but source evaluation
needs their same-stage values and analytic tangents.  The direct curvature
formula needed for the first item is

```
Psi0^(1) = (D-rho-rhobar-3 epsilon+epsilonbar) sigma^(1)
         -(delta-3 beta-alphabar+pibar-tau) kappa^(1)
         =(thorn-rho-rhobar)sigma^(1)
          -(eth+pibar-tau)kappa^(1).
```

The first equality follows by solving the Ricci identity in Appendix A of
arXiv:2008.11770 and agrees with the exact Weyl formula in
arXiv:gr-qc/9811019.  The separately displayed `Psi_0-1` formula in
arXiv:2008.11770 has the opposite signs on all five connection terms in its
`kappa` bracket and is inconsistent with that paper's own Ricci identity; it
must not be used.

The exact ORG formulas for `kappa^(1)` and `sigma^(1)` are

```
kappa^(1) = (D-2 epsilon-rhobar) h_lm
 -(1/2)(delta-2 alphabar-2 beta+pibar+tau) h_ll,

sigma^(1) = (1/2)(D-2 epsilon+2 epsilonbar+rho-rhobar) h_mm
 -(pibar+tau) h_lm.
```

## 6. Boundary-regular field and compact homogeneous operator

For this exact rotated tetrad, arXiv:2010.00162 derives both signs in the
same hyperboloidal coordinates.  Its regular spin `+2` field is `Z_plus`
defined by

```
Psi0 = W_plus Z_plus,
W_plus = R (Psi2^(0)/M)^(4/3)
       = R^5 / (L^2 - i a R cos(theta))^4.
```

The last equality fixes the continuous real cube-root branch of `-1` and
avoids a fractional power in production code.

Properties:

- `Z_plus` has spin `+2`, boost `+2`;
- at scri, peeling `Psi0=O(R^5)` makes `Z_plus=O(1)`;
- at the future horizon,
  `W_plus(R_H)=R_H^5/(L^2-i a R_H cos(theta))^4` is finite and nonzero;
- the displayed tetrad and all non-polar background spin coefficients are
  regular at the future horizon, so this is also a horizon-regular code-tetrad
  field; no additional horizon boost is required for this numerical variable.

After substituting `Psi0=W_plus Z_plus` **inside the NP operator**, the primary
paper multiplies the full NP equation by

```
N = 2 Sigma_BL/R.
```

It performs no separate division by `W_plus`.  The physical source
normalization is therefore identical for both signs:

```
forcing_s = (2 Sigma_BL/R) S_s
          = 2 (L^4+a^2 R^2 cos^2(theta)) S_s/R^3.
```

In the repository's first-order radial reduction the homogeneous equation is

```
C Z_TT - 2 K Z_TR - Hrad Z_RR - Delta_s Z
+ 2 a (1+4MR/L^2) Z_Tphi + 2a R^2/L^2 Z_Rphi
+ A_T(s=+2) Z_T + A_R(s=+2) Z_R
+ 2aR/L^2 Z_phi + A_0(s=+2) Z = forcing_plus,
```

with the same principal coefficients `C`, `K`, and `Hrad` as the minus-2
equation and the spin `+2` angular operator.  Hence the radial principal
symbol and characteristic speeds are identical to the already audited
minus-2 system.  `Hrad` vanishes linearly at the future horizon and
quadratically at scri, while every displayed coefficient remains finite.

This establishes mathematical reuse of the generic homogeneous coefficient
kernel.  It does **not** yet qualify its source normalization, endpoint
discretization, angular bands, or numerical stability for `s=+2`.

## 7. Ordered-mode and sharp semantics

Every quadratic product is accumulated over ordered pairs
`(m1,m2)->mt=m1+m2`.  Prime leaves `m` unchanged.  Complex-conjugate metric
components are obtained with sharp, e.g.

```
h_mm[m] = (h_barbar[m])^sharp = conj(h_barbar[-m]),
h_lm[m] = (h_lbar[m])^sharp = conj(h_lbar[-m]).
```

The plus-2 source must not substitute `conj(X_m)` for either sharp relation.
Its source ledger therefore needs an explicit `prime_action` column separate
from `sharp_action`.

## 8. Remaining blockers before production authority

The raw/raw output pairing is fixed for this proposal.  The full compact
source ledger, independent SymPy oracle, and scri/horizon power gates are now
closed by `docs/PLUS2_SOURCE_COMPACTIFICATION.md` and
`tools/symbolic/verify_plus2_compact_source.py`.  The preserved Spiers
notebook snapshot remains provenance for `TeukolskySource0`; replaying its
unrelated reduced-effective-stress calculations is not a prerequisite for the
independently checked raw source.

1. **Cancellation-safe `Psi0/Psi1`.**  Primary sources warn that direct
   curvature formulas contain cancellations through `R^4` before the
   `Psi0=O(R^5)` remainder.  A high-precision host oracle and convergence
   comparison against the Bianchi/TSI route are mandatory.
2. **Tetrad-output interpretation.**  `Psi0_raw^(2)` is fixed-convention, not
   a claimed gauge- or tetrad-invariant observable.  Zero companion initial
   data fixes a homogeneous convention but does not prove common
   second-order-metric compatibility with independently initialized
   `Psi4^(2)`.
3. **Implementation and qualification.**  No Kokkos source exists for this
   proposal.  CPU/GPU oracle agreement, endpoint convergence, and waveform
   extraction tests remain mandatory after the linear gate closes.

Until these blockers are closed with generated evidence, production code must
keep the `+2` companion disabled and fail closed rather than treating the
existing generic spin parameter as qualification.
