# Spin `+2` stage-local curvature derivative gate

**Status:** Route A approved and implemented as a standalone mathematical and
production-value gate; no `SpatialPipeline` or runtime wiring.

## 1. Question fixed by this gate

The compact raw spin `+2` source is already expressed in terms of regular
curvature fields

```text
Psi0^(1) = R^5 Z0,   Psi1^(1) = R^4 Z1,   Psi2^(1) = R^3 H.
```

The pair source needs `Delta_5 Z0`, `eth_5 Z0`, `Delta_4 Z1`, and
`ethprime_4 Z1`.  The outer source then needs `thorn_5 J` and `eth_6 K`.
Because every `Delta_n`, `thorn_n`, `eth_n`, and `ethprime_n` in the rotated
tetrad has an explicit `T` component, constructing these slots by repeatedly
differentiating the local metric-curvature formula raises the metric time
order.  This note determines that order and gives an exact vacuum-Bianchi
alternative.  It does not invent endpoint values for `Z0` or `Z1`.

Throughout, `h[k]` means the `k`th `T` derivative of the *complete coupled
linear primary plus reconstruction stage state*, not merely the three metric
components.  Background quantities are stationary.

## 2. Primary equations and convention conversion

The authority is Loutrel--Ripley--Giorgi--Pretorius,
arXiv:2008.11770:

- Appendix A, Bianchi identities (A12e,A12f), labels `bianchi-5` and
  `bianchi-6` in the primary TeX;
- their first-order vacuum expansions (C20a,C20b), labels `psi_1-recon` and
  `psi_0-recon`;
- the perturbed derivative (B8c), label `delta-1`.

The coordinate tetrad, stationary coefficients, and GHP definitions are from
Ripley--Loutrel--Giorgi--Pretorius, arXiv:2010.00162, Eqs. (5), (8), and the
display labelled `edth_def`.  In particular, this repository uses the rotated
Kinnersley tetrad with `gamma=0` but nonzero `epsilon`.

For a field of GHP type `{p,q}`,

```text
eth X      = (delta    - p beta  - q alphabar) X,
ethprime X = (bardelta - p alpha - q betabar) X.
```

Thus `eth Psi1=(delta-2 beta)Psi1`, while `eth Psi2=delta Psi2`.
The repository's exact regular operators are

```text
Delta_n X = R^(-n)   Delta(R^n X),
thorn_n X = R^(-n-1) thorn(R^n X),
eth_n X   = R^(-n-1) eth(R^n X),
ethprime_n X = R^(-n-1) ethprime(R^n X).
```

In ORG, `h_nm=h_mbar_m=0`, so (B8c) becomes

```text
delta^(1) f = -h_lm Delta f + (1/2) h_mm bardelta f.
```

With the repository's signed-mode fields,

```text
h_lm = R^2 Csharp,   h_mm = R Bsharp,
Csharp_m = conj(C_(-m)),   Bsharp_m = conj(B_(-m)).
```

Sharp is never same-mode conjugation.

## 3. Exact vacuum Bianchi closures

Linearizing `bianchi-5` in vacuum, setting the type-D background zeros and
`gamma=0`, and collecting `delta-2 beta` into `eth` gives

```text
Delta Psi0 = eth Psi1 - mu Psi0 - 4 tau Psi1 + 3 sigma1 Psi2_background.
```

After substituting the regular fields, the exact compact closure is

```text
F0 := Delta_5 Z0
   = eth_4 Z1 - R mu0 Z0 - 4 R tau0 Z1 + 3 Sig H.          (B0)
```

Linearizing `bianchi-6` gives

```text
Delta Psi1 = delta Psi2 - 2 mu Psi1 - 3 tau Psi2
             + delta^(1) Psi2_background - 3 tau1 Psi2_background.
```

Using the ORG derivative above and

```text
Psi2_background=R^3 psi20, sigma1=R^2 Sig, tau1=R^2 Ta,
h_lm=R^2 Csharp, h_mm=R Bsharp,
```

the exact compact closure is

```text
F1 := Delta_4 Z1
   = eth_3 H
     + R[-2 mu0 Z1 - 3 tau0 H
         - Csharp Delta_3 psi20
         + (1/2) Bsharp ethprime_3 psi20
         - 3 Ta psi20].                                   (B1)
```

The signs of the two metric terms follow directly from
`delta^(1)=-h_lm Delta+(1/2)h_mm bardelta`; they are not inferred from the
compact source.

### Weight check

Using `(spin,boost)` weights, every term of (B0) has `(2,1)`:

| term | weights |
|---|---|
| `Delta Psi0`, `eth Psi1` | `(2,1)` |
| `mu Psi0` | `(0,-1)+(2,2)=(2,1)` |
| `tau Psi1` | `(1,0)+(1,1)=(2,1)` |
| `sigma1 Psi2` | `(2,1)+(0,0)=(2,1)` |

Every term of (B1) has `(1,0)`:

| term | weights |
|---|---|
| `Delta Psi1`, `eth Psi2` | `(1,0)` |
| `mu Psi1`, `tau Psi2`, `tau1 Psi2` | `(1,0)` |
| `h_lm Delta Psi2_background` | `(1,1)+(0,-1)=(1,0)` |
| `h_mm bardelta Psi2_background` | `(2,0)+(-1,0)=(1,0)` |

The radial rescaling does not change spin or boost weight.

## 4. Turning the closures into exact same-stage tangents

The repository point operator is

```text
Delta_n X = A X_T + B (R X_R+n X),
A = 2+4 M R/L^2,   B=R/L^2.
```

Consequently

```text
X_T = [F-B(R X_R+n X)]/A.                                 (I)
```

For the physical exterior, `A>0`.  At scri, `A=2` and `B=0`; therefore this
inversion introduces no `1/R` quotient.  It is also finite at the future
horizon.

The closure is triangular:

1. evaluate (B1), then use (I) to obtain `Z1_T`;
2. use `Z1_T` in `eth_4 Z1`, evaluate (B0), then obtain `Z0_T`.

Since the background is stationary, exact time differentiation gives

```text
F1_T = eth_3 H_T
     + R[-2 mu0 Z1_T - 3 tau0 H_T
         - Csharp_T Delta_3 psi20
         + (1/2) Bsharp_T ethprime_3 psi20
         - 3 Ta_T psi20],                                  (B1T)

F0_T = eth_4 Z1_T - R mu0 Z0_T - 4 R tau0 Z1_T
       + 3 Sig_T H + 3 Sig H_T.                            (B0T)
```

Apply the time derivative of (I), equivalently apply (I) to
`(F_T,X_T,(X_T)_R)`, first for `Z1_TT` and then for `Z0_TT`.  The ordering is
again triangular because `eth_4 Z1_T` needs the already computed `Z1_TT`.
Only `H_TT`, `Csharp_T`, `Bsharp_T`, `Ta_T`, and `Sig_T` occur.  Thus the
closure requires `h[0..2]`, not `h[3]` or `h[4]`.

The source derivative slots then follow without another metric time level:

```text
Delta_4 Z1       = F1,       (Delta_4 Z1)_T       = F1_T,
Delta_5 Z0       = F0,       (Delta_5 Z0)_T       = F0_T,
ethprime_4 Z1    = ethprime(Z1,Z1_T),
(ethprime_4 Z1)_T= ethprime(Z1_T,Z1_TT),
eth_5 Z0         = eth(Z0,Z0_T),
(eth_5 Z0)_T     = eth(Z0_T,Z0_TT).
```

Radial derivatives of `Z0_T` and `Z1_T` and angular raised/lowered views are
therefore explicit preallocated workspace requirements.

## 5. Exact derivative-order ledger

### 5.1 Fourteen manifest primitives

| primitive rows | value | tangent | second tangent if locally needed |
|---|---:|---:|---:|
| `V,C,B,H,Pi` | `h[0]` | `h[1]` | `h[2]` |
| `Sig,Kap,Rh,Ta,Al,Be,Ep` | `h[1]` | `h[2]` | `h[3]` |
| local metric `Z0,Z1` | `h[2]` | `h[3]` | `h[4]` |

The `Z0` and `Z1` row includes the cancellation-safe curvature quotient.  At
scri its local construction still requires qualified coefficients for
`Psi0/R^5` and `Psi1/R^4`; the Bianchi equations do not authorize guessing
those initial values.

### 5.2 Pair slots: naive local metric route

The notation in the last two columns is `value/tangent`.

| slots | required metric time order |
|---|---:|
| `Delta_4 Z1`, `ethprime_4 Z1`, `Delta_5 Z0`, `eth_5 Z0` | `h[3]/h[4]` |
| `Delta_2 Csharp`, `ethprime_1 Bsharp`, `Delta_2 C`, `eth_1 B`, `Delta_2 V`, `eth_2 C`, `ethprime_2 Csharp` | `h[1]/h[2]` |
| `Delta_2 Sig`, `Delta_3 Kap`, `ethprime_3 Kap` | `h[2]/h[3]` |

Consequently `J` and `K` have value order `h[3]` and tangent order `h[4]`.
`Q` has value order `h[2]` and tangent order `h[3]`.  Production forcing
value uses `thorn_5 J` and `eth_6 K`, hence `J_T` and `K_T`, and therefore
requires `h[4]` by the naive local route.  It does **not** require `h[5]`.
Only the optional final forcing tangent would require `J_TT/K_TT` and
`h[5]`.

### 5.3 Pair slots with Bianchi curvature tangents

With (B0)--(B1T), all curvature slot values and tangents require at most
`h[2]`.  Every metric/connection slot needed by `J`, `K`, and the *value* of
`Q` also requires at most `h[2]`.  There is one API-level exception:
`Delta_2 Sig_T`, `Delta_3 Kap_T`, and `ethprime_3 Kap_T` require `h[3]`, but
they contribute only to `Q_T`.  The production forcing value consumes `Q`,
not `Q_T`.

The existing all-`Jet1` spatial source API nevertheless requires and stores
all eight family tangents.  An `h[0..2]` implementation must therefore split
the contract into:

- `J`/`K` value plus tangent;
- `Q` value;
- optional `Q` and per-family diagnostic tangents only when a higher-order
  input contract is explicitly selected.

Supplying fictitious zero tangents to the current API is not an acceptable
implementation.

## 6. Two exact production contracts

### Route A: Bianchi curvature closure

Two variants are mathematically exact:

1. A producer supplies same-stage regular `Z0,Z1` values and their radial and
   angular workspaces.  Their `T` and `TT` values and all four required source
   derivative slots are computed from (B0)--(B1T).  `Z0,Z1` may be obtained
   from the local metric only where the curvature quotients are qualified.
2. `Z1` and then `Z0` are evolved as passive triangular first-order Bianchi
   companion states.  This avoids rebuilding the delicate quotient at every
   RK stage, but adds initial data, RK storage, boundary conditions, and
   Bianchi constraint monitoring.  Initial data must be consistent with the
   metric curvature, including explicit regular scri coefficients.

Minimum live inputs and workspaces are:

- complete coupled linear `h[0]`, `h[1]`, and `h[2]` stage views;
- signed-mode `Z0,Z1` values, with exact `m -> -m` sharp lookup for metric
  inputs;
- radial derivatives of `Z0,Z1,Z0_T,Z1_T`;
- raised/lowered angular views for `H,H_T,Z0,Z0_T,Z1,Z1_T`;
- `F0,F1,F0_T,F1_T,Z0_T,Z1_T,Z0_TT,Z1_TT` scratch;
- primitive values/tangents, pair `J/K` values/tangents, pair `Q` values,
  projected aggregates, and outer radial/angular scratch.

No closure contains a scri division.  The unresolved scientific work is the
initial/constraint-consistent `Z0,Z1` prescription and a stable boundary
treatment for the two passive transport equations.

### Route B: repeated autonomous stage graph

Let `L` be the exact semi-discrete, projected, coupled linear
Teukolsky-plus-reconstruction RHS graph on the stationary background.  Apply

```text
x[1]=L x[0], x[2]=L x[1], x[3]=L x[2], x[4]=L x[3].
```

The *full coupled graph* must be applied; reconstruction alone is not closed
because it is driven by the primary spin `-2` state.  Then local curvature
uses

```text
Z value: x[0..2],   Z_T: x[1..3],   Z_TT: x[2..4].
```

This gives every pair value/tangent and the production outer forcing value
with `h[0..4]`.  Rolling buffers are permissible, but each application needs
the full graph's radial/angular scratch and no output may alias its input or
operator scratch.  There is no `h[5]` requirement unless the final forcing
tangent is requested.

Route B does not solve the endpoint problem: it needs qualified scri
coefficients for `Z0,Z1`, their first time derivatives, and their second time
derivatives.  Near scri it also repeatedly exposes the `O(R^2)` and `O(R)`
curvature cancellations before division, and four RHS applications amplify
high-frequency radial/angular error.  Those are qualification risks, not
licenses to change the formulas.

## 7. Independent Bianchi checks that are not production closures

The vacuum linearization of `bianchi-2` gives

```text
ethprime_4 Z1 = [thorn_3 H-3 rho0 H]/R
                -2 R pi0 Z1
                -(1/2) V Delta_3 psi20
                -3 R Rh psi20.                             (O1)
```

This is an independent residual for `ethprime_4 Z1`.  Its first bracket must
cancel as `O(R)` at scri, so (O1) is not a production endpoint formula without
a separately qualified limit coefficient.

Similarly, `bianchi-1` gives

```text
ethprime_5 Z0 = [thorn_4 Z1-4 rho0 Z1]/R
                -2 epsilon0 Z1 - R pi0 Z0 + 3 Kap H.       (O0)
```

It is a useful residual but is not one of the source slots and has the same
scri cancellation issue.

## 8. Decision gate

The owner selected Route A.  The standalone implementation consists of:

- `include/teuk/plus2_bianchi.hpp`, which provides (B0), (B1), exact
  `Delta_n` inversion, and fail-closed curvature quotient semantics;
- the split J/K and Q point evaluators in `include/teuk/plus2_source.hpp`;
- `include/teuk/plus2_source_value_spatial.hpp`, whose pair kernel retains
  only J/K tangents and Q value and whose outer kernel emits forcing value;
- `tools/symbolic/verify_plus2_bianchi_closure.py` and focused C++ tests.

The two mathematical alternatives considered were:

- Route A with either supplied same-stage or passively evolved `Z0,Z1`, plus
  the `J/K tangent, Q value` API split; or
- Route B with four exact applications of the full autonomous linear graph
  and explicit three-time-order scri curvature coefficients.

Both routes are exact.  Route A has the lower derivative order and avoids
repeated curvature division, but it introduces curvature state/initial-data
semantics.  Route B preserves a purely local metric definition, but has a
larger workspace and a substantially harder endpoint/cancellation gate.

This standalone commit does not yet construct all radial/angular views from
the live reconstruction state.  In particular, it cannot be wired until the
separate fourth-order one-sided scri quotient/coefficient layer supplies
qualified `Z0,Z1` values.  No endpoint extrapolation is hidden in this gate.
