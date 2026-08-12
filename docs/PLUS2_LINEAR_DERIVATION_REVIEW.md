# Linear spin +2 companion: convention map and derivation review

Status: reviewed derivation, not production implementation

Scope: `Psi0^(1) = T0[h^(1)]`, the regular spin `+2` field, the
homogeneous compactified operator, and linear Teukolsky--Starobinsky (TSI)
validation

Repository base: detached public `main` at the worktree HEAD

This note deliberately stops before production code.  It records formulas
that can be traced to primary sources in the exact conventions of this
repository, and it marks the remaining numerical-normalization work as a
hard gate rather than filling it with an inferred factor.

## 1. Authorities and convention compatibility

The directly compatible primary source is Ripley, Loutrel, Giorgi, and
Pretorius, arXiv:2010.00162.  Its coordinates, tetrad, ORG condition,
perturbed-tetrad choice, Fourier convention, and regularized spin `-2` field
are the sources of the checked-in implementation.  Loutrel, Ripley, Giorgi,
and Pretorius, arXiv:2008.11770, gives the metric-to-curvature identities in
the same NP sign convention.  In particular it defines

```text
Psi0 = -C(l,m,l,m),  Psi4 = -C(n,mbar,n,mbar),
l.n = +1,            m.mbar = -1.
```

Berens, Gravely, and Lupsasca, arXiv:2403.20311, keeps an explicit signature
parameter `epsilon_g`; taking `epsilon_g=-1` gives precisely the convention
above.  Therefore its TSI formulas require no additional Weyl-scalar sign
when that signature is selected.  Its displayed tetrad is the Boyer--Lindquist
Kinnersley tetrad, however, so its raw field components must not be compared
directly with the code-tetrad fields.

The authors' supplemental Mathematica repository was inspected at commit
`cf924707593a58ec889c70ea501d764e99d1d4aa`.  Its saved ORG expressions and
example notebook use the paper's hatted radial/angular normalizations and
Starobinsky factors.  It is a suitable source for future normalized mode
fixtures, but its Einstein-equation notebook checks reconstructed metrics in
Boyer--Lindquist Kinnersley conventions; it is not a direct oracle for the
rotated code-tetrad `T0` contraction.

Aksteiner, Andersson, and Backdahl, arXiv:1601.06084 Eq. (A1), proves that
the extreme linearized Weyl components equal the varied extreme components
even when the dyad is varied:

```text
vartheta(Psi0) = dot(Psi0),  vartheta(Psi4) = dot(Psi4).
```

This is why the fixed perturbed-tetrad prescription used by the repository
does not add an extra first-order background-curvature term to either
extreme scalar.  The same paper supplies the covariant GHP TSI and shows that
the three middle relations contain extra integrability information.  The
initial implementation should use the two extreme identities only; the
middle relations are valuable later diagnostics, not substitutes for the
extreme normalization check.

## 2. Exact tetrad conversion

Let

```text
Delta = r^2 - 2 M r + a^2,
Sigma = r^2 + a^2 cos(theta)^2,
A_b   = Delta/(2 Sigma),
q     = -(r+i a cos(theta))/(r-i a cos(theta))
      = exp[-2 i atan(r/(a cos(theta)))]
```

The quotient defines the smooth limiting phase `q=-1` at `a cos(theta)=0`.
Appendix C of arXiv:2010.00162 derives the code tetrad from Kinnersley in
Eqs. (C3), (C7), (C9), and (C10).  Printed Eq. (C9c) contains
`sin(theta)` inside its arctangent, but direct division of explicit
`m_code` in Eq. (C10c) by `m_Kin` in Eq. (C7c) gives the `cos(theta)` phase
above.  This is also the only phase that reproduces the final code tetrad
in Eq. (5c).  Therefore C9c's `sin` is a localized typo.  The consistent
transformation is

```text
l_code = A_b l_Kin,
n_code = A_b^(-1) n_Kin,
m_code = q m_Kin.
```

Consequently,

```text
Psi0_code = A_b^2 q^2 Psi0_Kin,
Psi4_code = A_b^(-2) q^(-2) Psi4_Kin.
```

The factor `A_b` vanishes at the future horizon.  A physical ingoing
perturbation whose Kinnersley `Psi0` has the familiar `Delta^(-2)` behavior
therefore has finite `Psi0_code`.  Any modal TSI oracle written in Kinnersley
variables must perform this conversion explicitly and must avoid evaluating
the separately singular factors exactly at the horizon.

The coordinate transformation used by the code is radial apart from the
time and azimuth shifts,

```text
T = t + H(r),  phi_code = phi_BL + J(r),  R=L^2/r.
```

Thus `partial_t` at fixed Boyer--Lindquist `(r,theta,phi_BL)` is exactly
`partial_T` at fixed code `(R,theta,phi_code)`.  Radial TSI operators still
need the full chain rule, including both `H'(r)` and `J'(r)`.

## 3. Linear curvature operator in the current ORG

The repository stores the complex-conjugate angular metric components.  For
each signed mode,

```text
h_mm       = R B^sharp,
h_lm       = R^2 C^sharp,
h_ll       = R^2 U/mu0,
X_m^sharp  = conjugate(X_(-m)).
```

No `conjugate(X_m)` shortcut is permitted.

Equations (C1c) and (C1e) of arXiv:2008.11770 (with Eq. (11) repeating the
`kappa1` formula), specialized to the ORG and perturbed tetrad used by
arXiv:2010.00162, give the first step.  Form the first-order spin coefficients

```text
sigma1 = 1/2 [D + 2(epsilon_bar-epsilon) + rho-rho_bar] h_mm
         - (tau+pi_bar) h_lm,

kappa1 = [D - 2 epsilon-rho_bar] h_lm
         - 1/2 [delta-2 alpha_bar-2 beta+pi_bar+tau] h_ll.
```

For the second step, do **not** use displayed Eq. (12) of
arXiv:2008.11770.  Its connection signs contradict the paper's own exact
Ricci identity (A9b).  Solving (A9b) for `Psi0` on a type-D background gives

```text
Psi0^(1) = [D-rho-rho_bar-3 epsilon+epsilon_bar] sigma1
           - [delta-alpha_bar-3 beta+pi_bar-tau] kappa1.       (T0)
```

This corrected expression is independently printed as the exact Weyl-scalar
formula in Campanelli and Lousto, arXiv:gr-qc/9811019 Appendix A Eq. (A5).
Thus the disagreement is localized to displayed Eq. (12), not to the spin
coefficients or to the Appendix-A Ricci identity.

All background coefficients and derivatives here are in the rotated code
tetrad.  This is a second-order differential operator on the reconstructed
metric, not the fourth-order TSI operator on `Psi4`.

The inner formulas have a useful GHP rewrite.  With the repository's
`p=s+b`, `q=-s+b` convention,

```text
sigma1 = 1/2 (thorn + rho-rho_bar) h_mm
         - (tau+pi_bar) h_lm,

kappa1 = (thorn-rho_bar) h_lm
         - 1/2 (eth+pi_bar+tau) h_ll.
```

The corrected outer operations have the especially simple GHP form

```text
delta kappa1 = eth(kappa1) + 3 beta kappa1 + alpha_bar kappa1.

Psi0^(1) = (thorn-rho-rho_bar) sigma1
           - (eth+pi_bar-tau) kappa1.
```

The `alpha_bar` and `beta` terms cancel in the second line; retaining the
opposite-sign combination from arXiv:2008.11770 Eq. (12) would change `T0`.

### Exact input and derivative manifest

| Input | Repository construction | `(s,b)` | scri falloff | Derivatives entering `T0` |
|---|---|---:|---:|---|
| `h_mm` | `R B^sharp` | `(+2,0)` | `R` generically | `D h_mm`, then outer `D` and `delta` |
| `h_lm` | `R^2 C^sharp` | `(+1,+1)` | `R^2` | `D h_lm`; outer derivatives of differentiated and algebraic occurrences |
| `h_ll` | `R^2 U/mu0` | `(0,+2)` | `R^2` | `delta h_ll`, then outer `delta` |
| `sigma1` | formula above | `(+2,+1)` | `R^2` | `D sigma1` |
| `kappa1` | formula above | `(+1,+2)` | `R^3` | `delta kappa1` |
| `rho,epsilon` | `R rho0`, `R^2 epsilon0` from `kerr_background_point` | background | finite rescaled | values and product-rule derivatives |
| `alpha,beta` | exact rotated-tetrad formulas in arXiv:2010.00162 Eq. (6f,g); not presently stored in `KerrBackgroundPoint` | background | `R` | values and product-rule derivatives |
| `tau,pi` | `R^2 tau0`, `R^2 pi0` | background | `R^2` | values and product-rule derivatives |

Here `h_ll` has physical boost weight `+2`; stored `U` represents
`mu h_ll`, whose boost weight is `+1`, matching `docs/CONVENTIONS.md`.
The generic `R` falloff of `h_mm` is not a claim that every intermediate
term peels separately.  Loutrel et al. explicitly warn after Eq. (D4) that
`O(r^-3)` and `O(r^-4)` pieces cancel before `Psi0=O(r^-5)` is recovered.

The stage-local computational manifest is

```text
state:          B, C, U and B^sharp, C^sharp
first tangent:  B_T, C_T, U_T
second tangent: B_TT, C_TT, U_TT
radial data:    first and second D_R applications, including mixed derivatives
angular data:   up to two raised/lowered applications at the required spins
background:     rho, epsilon, alpha, beta, tau, pi and conjugates
output:         Psi0_code and Z_plus = Psi0_code/W_plus
```

No independent evolution of `sigma1` or `kappa1` is needed.  They are
stage-local derived scratch.  Every complete derived tangent must be
projected into its retained spin band before a dependent angular operation.

### Stage data required by `T0`

`T0` needs two derivatives of the metric.  At an RK stage it therefore needs
the same-stage reconstruction state `h`, tangent `h_T`, and second tangent
`h_TT`.  Because the first-order/reconstruction graph is stationary and
linear,

```text
h_T  = L[h],
h_TT = L[h_T].
```

Radial and angular derivatives must be applied to these stage-local values.
Endpoint interpolation and finite differences between output times are not
valid replacements.  A practical implementation can extend the current
analytic-tangent path by one linear operator application; it does not require
stored time history.

The direct formula contains asymptotic cancellations noted explicitly in
arXiv:2008.11770: individual terms can be `O(r^-3)` or `O(r^-4)` while
`Psi0=O(r^-5)`.  Production tests must therefore measure cancellation and
convergence at scri.  A transparent coordinate linearized-Riemann contraction
is required as an independent host oracle.  The alternative Bianchi transport
for `Psi1` and `Psi0` is lower differential order but introduces integration
constants/initial data and is not, by itself, the requested local `T0[h]`
diagnostic.

## 4. Boundary-regular spin +2 field

The regular field is already derived in Eq. (21b) of arXiv:2010.00162 for
this exact code tetrad.  Write

```text
zeta = r - i a cos(theta) = (L^2-i a R cos(theta))/R,
W_plus = R zeta^(-4)
       = R^5/(L^2-i a R cos(theta))^4,
Psi0_code = W_plus Z_plus.
```

This is the unambiguous branch of the paper's notation
`R (Psi2/M)^(4/3)`: since `Psi2/M=-zeta^-3`, take the continuous cube root
`(Psi2/M)^(1/3)=-zeta^-1` and then the fourth power.

At scri,

```text
W_plus = R^5/L^8 [1 + 4 i a R cos(theta)/L^2 + O(R^2)].
```

Hence an outgoing peeling field `Psi0_code=O(R^5)` has finite `Z_plus`.
At the future horizon `R=R_H`, both `R_H` and
`L^2-i a R_H cos(theta)` are finite and nonzero, so `W_plus` is finite and
nonzero.  Since the code tetrad itself is horizon regular, finite physical
`Psi0_code` is equivalent to finite `Z_plus` there.

This scaling is appropriate for the no-incoming-radiation hyperboloidal
problem.  A generic incoming field at future null infinity has a different
peeling branch and need not give finite `Z_plus`; the continuum characteristic
problem supplies no incoming data at that boundary.

## 5. Compactified homogeneous operator

After the scaling above, arXiv:2010.00162 Eq. (22) derives one compact coordinate PDE
for both signs.  Setting `s=+2` in that *derived equation* gives the equation
for `Z_plus`; this is not an inference from a spin-generic C++ helper.

The coefficients `C_T`, `K`, and `H_R`, and the `G_m`, `Q`, and field
coefficients in `include/teuk/teukolsky.hpp` algebraically transcribe the
paper's equation.  The corrected angular action must be

```text
Delta_Omega^(s) Y_lm^(s) = -(ell-s)(ell+s+1) Y_lm^(s).
```

For `s=+2`, this is `-(ell-2)(ell+3)`.  In particular, its value is zero for
`ell=2`; that zero is correct for the lower-after-raise form used by the
paper and is not the symmetric spin-bundle Laplacian.

The radial principal coefficient is

```text
H_R = R^2 (L^4-2 L^2 M R+a^2 R^2)/L^4.
```

It vanishes at both `R=0` and `R=R_H`.  The principal symbol and radial
characteristic speeds are spin independent, so the existing endpoint count
continues to apply.  The lower-order `s=+2` coefficients are polynomial in
`R` and finite at both endpoints; `C_T` is nonzero on the validated exterior
parameter range.  These facts justify reusing the generic kernels only after
the exact host/device equality and boundary tests in Sec. 7 below pass.

## 6. TSI validation route and normalization

The primary production value of `Psi0^(1)` should be `T0[h]`.  The TSI is an
independent validation route.

In the Kinnersley tetrad, the first-form identities from
arXiv:2403.20311 Eq. (1.25) are

```text
l^4 (zeta^4 Psi4)
  = 1/4 L_-1 L_0 L_1 L_2 Psi0 - 3 M partial_t conjugate(Psi0),

Delta^2 [(Sigma/Delta)n]^4 Delta^2 Psi0
  = 1/4 Lbar_-1 Lbar_0 Lbar_1 Lbar_2 (zeta^4 Psi4)
    + 3 M partial_t conjugate(zeta^4 Psi4).
```

These are coupling identities for the two Weyl scalars of the *same real
metric perturbation*.  Their eighth-order second forms do not retain that
coupling information and are therefore insufficient as the main validation.

For a separated mode, use the radial and angular first-form identities with
the paper's explicitly normalized modes.  The invariant products obey

```text
C_(omega ell m) = D_(omega ell m) + (12 M omega)^2.
```

At `a omega=0`, with
`lambda_(+2)=ell(ell+1)-6`,

```text
D = [(ell-1) ell (ell+1) (ell+2)]^2.
```

For a single complex mode of `zeta^4 Psi4`, the corresponding real metric
generically produces *two* modes in `Psi0`: the original `(omega,m)` mode and
the sharp partner `(-conjugate(omega),-m)`.  Equation (2.44) of
arXiv:2403.20311 gives their exact amplitudes.  A comparison that keeps only
the same-`m` term can match the QNM frequency while still have the wrong
phase and normalization.

Recommended validation sequence:

1. Schwarzschild, real-frequency pure mode: verify both first-form TSI
   residuals and the `24^2` angular product for `ell=2`.
2. Schwarzschild QNM: use the complex-frequency analytic radial solution and
   include its sharp partner; verify complex amplitude and phase.
3. Moderate Kerr mode: compute the spheroidal eigenvalue and hatted
   Starobinsky factors in the same normalization as the oracle; verify the
   two-mode relation, not frequency alone.
4. Convert the oracle fields to the code tetrad with Sec. 2 before comparing
   with `T0[h]`.
5. Demonstrate radial, angular, and temporal convergence away from the
   separately singular Kinnersley horizon factors.

## 7. Executable symbolic and numerical test plan

`tools/symbolic/verify_plus2_linear_foundations.py` executes the convention
and scaling checks that do not need a numerical mode solver.  The next
implementation change should turn the following into CTest targets:

| ID | Test | Independent authority | Gate |
|---|---|---|---|
| L0 | GHP/NP rewrite of `sigma1`, `kappa1` | arXiv:2008.11770 App. A | exact symbolic equality |
| L1 | `T0[h]` versus coordinate linearized-Weyl oracle | arXiv:2403.20311 Eq. (A.8) | random smooth fields, host tolerance |
| L2 | sharp-mode closure through all derivatives | repository signed-mode convention | exact index/conjugation tests |
| L3 | `W_plus` scri/horizon expansions | arXiv:2010.00162 Sec. IV A | exact symbolic limits |
| L4 | derived paper PDE versus host `s=+2` coefficients | arXiv:2010.00162 Eq. (22) | random-point equality |
| L5 | device versus host plus-2 coefficients | independent host transcription | backend tolerance |
| L6 | manufactured `Z_plus` PDE | derived compact equation | radial/time convergence |
| L7 | `T0[h]` versus first-form TSI | arXiv:2403.20311 Eqs. (1.25), (2.44) | complex amplitude and phase |
| L8 | homogeneous plus-2 residual of `T0[h]` | Teukolsky 1973 / Ripley 2020 | convergent residual |
| L9 | code-tetrad horizon regularity | exact boost/spin map | finite code fields, predicted Kin scaling |

The current named symbolic outputs are

```text
PASS selected cube root cubes to Psi2/M
PASS plus field scaling is R zeta^-4
PASS plus field scaling compact form
PASS scri leading W_plus/R^5
PASS finite horizon scaling expression
PASS extreme Weyl boost-spin product invariant
PASS explicit tetrad phase rationalization
PASS tetrad phase has unit norm
PASS sigma NP-to-GHP rewrite
PASS kappa NP-to-GHP rewrite
PASS exact Psi0 formula solves Ricci identity A9b
PASS displayed Eq12 signs differ from corrected A5 at nontrivial point
PASS corrected outer Psi0 angular operator NP-to-eth rewrite
PASS T0 spin/boost weights
PASS Schwarzschild angular Starobinsky product
PASS ell=2 angular Starobinsky product
PASS radial-angular Starobinsky product offset
PASS H_R vanishes at scri
PASS H_R vanishes at outer horizon
PASS spin +2 Q coefficient scri limit
PASS spin +2 field coefficient scri limit
PASS compact spin +2 lower-order endpoint finiteness
```

These checks encode arXiv:2010.00162 Eqs. (21b) and (22),
arXiv:2008.11770 Eqs. (C1c), (C1e), and (A9b),
arXiv:gr-qc/9811019 Eq. (A5), and
arXiv:2403.20311 Eqs. (2.22)--(2.24).  They intentionally do not claim the
phase-sensitive hatted-factor validation of Eqs. (2.37), (2.38), and (2.44).

For L1, construct the full coordinate metric from the three ORG tetrad
components and their conjugates, differentiate it symbolically or with
automatic differentiation, evaluate the linearized Weyl tensor including
the background-curvature metric terms, and contract with the *background
code tetrad*.  Do not reuse the NP `sigma/kappa` path in the oracle.

For L7, pin the angular/radial normalization in the fixture metadata.  The
unnormalized product constants `D` and `C` cannot determine the hatted factors
needed for a phase-sensitive first-form comparison.  A missing hatted
normalization is a hard blocker, not a tolerance to loosen.

## 8. Hard blockers and non-blocking conclusions

There is no formal blocker to the regular field or homogeneous plus-2 PDE:
both were derived for the exact code tetrad in arXiv:2010.00162, and the
checked-in coefficient function appears to be their spin-generic
transcription.  This still requires the exact tests L3--L6 before enabling
production `spin_weight=+2`.

There is no formal blocker to a local metric-curvature `Psi0^(1)` diagnostic:
the corrected exact `T0` operator is given above and is supported by two
independent primary identities.  The sign disagreement in
arXiv:2008.11770 Eq. (12) must remain an explicit regression test.  The main
numerical risk is the `R^-3/R^-4` cancellation at scri; convergence against
the coordinate oracle is mandatory.

The following remain hard gates for claiming a TSI-normalized companion:

- implement a mode oracle with explicitly pinned hatted angular and radial
  normalizations;
- include the sharp partner required by a real metric;
- perform the Kinnersley-to-code tetrad conversion;
- verify complex amplitude and phase, not only the QNM frequency.

No production normalization should be inferred from an eighth-order TSI,
from a single sign of `m`, or from the unhatted Starobinsky product alone.

## Primary references

- https://arxiv.org/abs/2008.11770
- https://arxiv.org/abs/2010.00162
- https://arxiv.org/abs/1601.06084
- https://arxiv.org/abs/1908.09095
- https://arxiv.org/abs/2403.20311
- https://arxiv.org/abs/gr-qc/9811019
