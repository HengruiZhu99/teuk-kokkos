# Passive spin `+2` curvature initialization and radial-boundary gate

**Status:** D10-5 quotient operator and continuum characteristic direction
qualified on analytic peeling data; live physical initialization remains
blocked, and rotating Kerr exposes a weakly hyperbolic bulk radial block.

## 1. Two distinct endpoint questions

The passive Route-A state stores

```text
Psi0^(1)=R^5 Z0,       Psi1^(1)=R^4 Z1.
```

Its **initial values** must be obtained from the reconstructed metric. Its
subsequent **radial boundary treatment** is determined by the characteristics
of the triangular Bianchi transport. A regular scri coefficient does not by
itself prove that a boundary closure is stable, and an outflow characteristic
does not determine the initial scri coefficient.

## 2. Continuum characteristics

The `Delta_n` part of both equations is

```text
Delta_n Z = A Z_T+B(R Z_R+n Z),
A=2+4 M R/L^2,       B=R/L^2.
```

The Bianchi-5 right-hand side also contains `eth_4 Z1`. In rotating Kerr its
principal time piece is `c Z1_T`, with
`c=-i a sin(theta)/[sqrt(2)(L^2-i a R cos(theta))]`, while Bianchi-6 is solved
first. The coupled principal matrices therefore have the triangular form

```text
M d_T [Z0,Z1]^T + N d_R [Z0,Z1]^T=lower order,
M=[[A,-c],[0,A]],       N=B R I.
```

Both generalized characteristic speeds are the repeated eigenvalue

```text
v(R)=B R/A=R^2/(2 L^2+4 M R).
```

For `R>0` and `a sin(theta)!=0`, the radial symbol is

```text
M^(-1) N=[[v,c B R/A^2],[0,v]].
```

It has only one eigenvector and is weakly hyperbolic. This does not create a
characteristic with the opposite radial direction, but it can cause a radial
derivative loss and high-frequency polynomial growth. Boundary treatment
cannot repair that bulk defect.

The grid coordinate increases from `R=0` at future null infinity to the
future horizon `R=R_H`. Hence:

| location | speed | continuum direction |
|---|---:|---|
| scri, `R=0` | `0` | characteristic; no incoming field |
| exterior, `0<R<R_H` | positive | toward increasing `R` |
| horizon, `R=R_H` | positive | leaves the computational domain |

No physical incoming radial datum is required by these continuum equations.
Nevertheless, the unreformed rotating two-field system cannot receive a
physical stable-boundary evidence identifier because its bulk radial block is
not strongly hyperbolic. The checked-in typed policy is therefore named
`ContinuumNoIncomingButWeaklyHyperbolicV1`; it is a blocker, not production
authority. A tetrad-adapted reformulation, a derivative-augmented system, or
another explicitly strongly hyperbolic closure must be derived before a
semi-discrete SBP/SAT qualification is meaningful.

## 3. Cancellation-safe initial coefficients

The reviewed primitive formulas split the two curvature fields as

```text
Z0=f0/R^2+z0_regular,
f0=thorn_2 Sig-(rho0+rhobar0)Sig-R eth_3 Kap,
z0_regular=-(pibar0-tau0)Kap,

Z1=f1/R+z1_regular,
f1=thorn_2 Be-rhobar0 Be-eth_2 Ep-beta0 Ep-alpha0 Sig
   -(1/2)V Delta_1 beta0+Epsharp beta0.
```

`z1_regular` is the remaining explicit order-four expression in
`plus2_source_primitives.hpp`. Peeling-compatible data obey

```text
f0(0)=0,       f0'(0)=0,       f1(0)=0.
```

Only under these conditions does repeated l'Hopital give

```text
q0=lim f0/R^2=(1/2) f0''(0),
q1=lim f1/R=f1'(0).
```

`Plus2CurvatureInitializationWorkspace` evaluates the first derivative with
the lower D10-5 SBP row and applies the same D10-5 operator once more for
`f0''`. The nested second derivative has fourth-order endpoint accuracy; the
single `f1'` derivative has fifth-order endpoint accuracy. Interior points,
including the horizon, use the exact quotients because `R>0`. No horizon
value is extrapolated or imposed.

## 4. Fail-closed evidence

Initialization requires a `Plus2CurvatureInitializationContract` containing
typed formula, endpoint-operator, and continuum-boundary identifiers plus
durable metric-curvature and Bianchi-consistency evidence strings. Its
peeling certificate has no public constructor. The only factory accepts
coarse/medium/fine audit samples and requires every peeling residual and both
endpoint coefficient arrays to converge at observed order at least four.

The certificate records the complete qualified fine-grid identity: radial
extent/count/spacing, ordered signed modes, ordered angular nodes, profile ID,
endpoint-residual bound, and every `Z0,Z1` coefficient on the mode/angular
endpoint plane. Initialization computes into temporary storage, audits the
actual `f0(0),f0'(0),f1(0)` values, and compares every coefficient before
copying into the requested state. A different nonleading mode or angular node
is rejected even when mode 0/node 0 agrees. Any rejection leaves the
destination unchanged.

Analytic manufactured profiles cover Schwarzschild (`a=0`) and rotating Kerr
(`a=0.73M`) tetrad-denominator dependence, both signed-mode partners, and all
radial points including scri and the horizon. These tests qualify the
quotient algorithm, not a physical Kerr perturbation or the missing live
metric-curvature graph.

## 5. Exact live-production blocker

The current `Plus2SourcePrimitiveSpatialProducer` intentionally constructs
only the twelve non-curvature primitive rows. It does not yet construct the
same-stage arrays needed above:

- `thorn_2 Sig` and `eth_3 Kap` for `f0`;
- `thorn_2 Be` and `eth_2 Ep` for `f1`;
- `Delta_1 beta0`, `Delta_2 epsilon0`, and
  `bardelta_2 epsilon0` background slots;
- the complete `z1_regular` combination and its endpoint audit;
- a cross-check against the independent raw `T0[h]` production operator and
  the Bianchi residual on Schwarzschild and rotating-Kerr perturbations.

Consequently this commit does not create live physical `Z0,Z1` data, does not
wire initialization into the solver, and does not authorize a physical
Bianchi evolution. Closing that gate requires implementing those arrays from
the same signed-mode `h[0..2]` stage, running the three-resolution certificate
on actual reconstructed data, and separately qualifying the semi-discrete
bulk transport formulation and boundary closure. Until then, the existing
transport must retain its
fail-closed physical initialization/boundary status.
