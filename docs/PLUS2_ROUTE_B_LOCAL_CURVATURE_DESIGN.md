# Route-B local curvature derivative tower

**Decision:** use local metric curvature for rotating production; retain the
two-field Bianchi transport only as a validation path.

## 1. Why Route A is not a production evolution

For a purely radial spatial covector, the rotating Bianchi-5/6 principal
matrices are

```text
M=[[A,-c],[0,A]],       N=B R I,
c=-i a sin(theta)/[sqrt(2)(L^2-i a R cos(theta))].
```

At `R>0` away from the axis, `M^(-1)N` is a nontrivial Jordan block with the
repeated eigenvalue `v=B R/A`. Strong hyperbolicity of the full
multidimensional system would require a uniformly diagonalizable principal
symbol for **every** real spatial covector. Choosing the angular and azimuthal
covector components to be zero recovers this radial Jordan block. Angular
principal terms therefore cannot repair the verdict. A fixed angular
spectral truncation likewise leaves arbitrarily high radial wave number and
the associated derivative-loss mechanism.

Route A remains useful for residual and short resolved validation tests, but
no physical rotating production run should evolve it as an autonomous state.

## 2. Exact derivative tower required by Route B

Let `x` denote only the ten-field linear primary/reconstruction subsystem

```text
(FirstP,FirstQ,FirstPsi,G,Lambda,H,B,Pi,C,U).
```

On the stationary Kerr background its complete projected semi-discrete graph
is an autonomous linear operator `L`. At every primary RK stage construct

```text
h[0]=x,        h[1]=L h[0],        h[2]=L h[1],
h[3]=L h[2],   h[4]=L h[3].
```

Every application must include the selected D10-5 radial operator,
dissipation and reduction convention, all signed-mode sharp reads, and all
angular projections. It must exclude the second-order fields and quadratic
source graph. Because `L` is linear, the last two applications are exactly
the two additional JVPs required by the local curvature route:

```text
h[3]=J_L(h[2]),        h[4]=J_L(h[3]),       J_L=L.
```

The local curvature value operator is stationary and linear. Apply the same
operator three times to consecutive triples:

```text
(h[0],h[1],h[2]) -> (Z0,Z1),
(h[1],h[2],h[3]) -> (Z0_T,Z1_T),
(h[2],h[3],h[4]) -> (Z0_TT,Z1_TT).
```

This produces the six-field adapter without a weakly hyperbolic curvature
evolution and without time differencing saved snapshots.

## 3. What the current `SpatialPipeline` already computes

`SpatialPipeline::evaluate_rhs_at_time(stage,output)` computes `h[1]` in
`output` for the first/reconstruction subsystem. It then applies the same
first/reconstruction graph once to `output` and stores `h[2]` in its private
`tangent_rhs_` view. Thus the arithmetic for `h[0..2]` already exists.

That is not yet a usable derivative-tower API:

- `tangent_rhs_` is private and has no generation/provenance stamp;
- its second-order slots are not a defined state;
- calling the full RHS recursively would also launch source and second-order
  work that is irrelevant to `L`;
- the `Disabled` source policy returns before reconstruction, so the narrowed
  linear graph must not inherit that early return;
- the existing graph scratch is shared, so input/output/tower views need
  explicit non-alias validation;
- a second full-RHS call can incidentally produce `h[3]` and overwrite the
  private view with `h[4]`, but relying on that side effect is not an API or a
  reviewable provenance contract.

The safe change is to factor a source-independent method conceptually named

```text
evaluate_primary_reconstruction_linear_graph(execution,input,output)
```

from the first/reconstruction portions of `evaluate_rhs_at_time`. A
preallocated `Plus2LinearDerivativeTower` should own distinct `h[2]`, `h[3]`,
and `h[4]` views, call that method twice beyond the already available
`h[0..2]` graph, and stamp all five levels with the same RK-stage generation.

## 4. Exact local curvature spatial graph

The reviewed primitive decomposition is

```text
Z0=f0/R^2-(pibar0-tau0)Kap,
f0=thorn_2 Sig-(rho0+rhobar0)Sig-R eth_3 Kap,

Z1=f1/R+z1_regular,
f1=thorn_2 Be-rhobar0 Be-eth_2 Ep-beta0 Ep-alpha0 Sig
   -(1/2)V Delta_1 beta0+Epsharp beta0.
```

`z1_regular` is the explicit expression already implemented by
`plus2_source_primitives`. A standalone value producer for one consecutive
`h` triple needs:

- the existing signed-mode packing of `H,Pi,B,C,U` and sharp partners;
- the existing D10-5 radial and GHP angular slots used to construct
  `Sig,Kap,Be,Ep` and the sharp connection helpers;
- new `D_R Be` and `eth_2 Ep` workspaces;
- `thorn_2 Sig`, `eth_3 Kap`, `thorn_2 Be`, and `eth_2 Ep` point values;
- reviewed analytic or independently generated values of
  `Delta_1 beta0`, `Delta_2 epsilon0`, and
  `bardelta_2 epsilon0` in repository conventions;
- explicit `f0,f1,z0_regular,z1_regular` output views.

Most metric/connection machinery exists in
`Plus2SourcePrimitiveSpatialProducer`, but that class deliberately discards
the sharp helpers and curvature numerators. The three background derivative
slots are present only as caller inputs to the point formula; no current live
spatial producer constructs them. They must be derived from the analytic Kerr
background and checked independently before a local producer is scientific.

At `R=0`, each of the three time levels uses

```text
Z0^[k]=(1/2) d_R^2 f0^[k]+z0_regular^[k],
Z1^[k]=d_R f1^[k]+z1_regular^[k],       k=0,1,2.
```

The first and nested second derivatives use D10-5. Production evidence must
extend the current value-only certificate to all six coefficients and all
peeling residuals `f0^[k](0),d_R f0^[k](0),f1^[k](0)` over three nested
resolutions. One convergent value coefficient cannot authorize tangent or
second-tangent endpoints.

## 5. Independent gates before integration

The local Route-B producer is not qualified until all of the following pass:

1. `Z0` agrees under refinement with the independent coordinate-curvature
   `T0[h]` operator at every interior point and with its separately scaled
   scri coefficient.
2. `Z1` agrees with an independent ordinary-NP oracle and the unused Bianchi
   residuals, without evolving the weakly hyperbolic Route-A system.
3. Schwarzschild and at least one rotating-Kerr manufactured **metric** data
   set, not merely manufactured quotient numerators, converge at fourth order
   including both endpoints for all six fields.
4. Signed `m -> -m` sharp relations, angular band retention, stage generation,
   and amplitude linearity are checked.
5. Hot-stage execution allocates and fences nothing; the one-time
   initialization/audit seam may copy its compact evidence to the host.

## 6. Current implementation boundary

`plus2_curvature_initialization.hpp` implements and tests the D10-5 quotient
and fail-closed certificate for already supplied value-level numerators. The
certificate binds the complete fine endpoint plane, radial geometry, signed
mode ordering, angular nodes, and profile identity. It does not claim that
those analytic numerator profiles are manufactured Kerr metric perturbations.

A production local producer is not added in this candidate because two
scientific prerequisites are absent: the reviewed analytic background
derivative slots and a narrowed, stamped `L` API exposing `h[0..4]` without
recursively invoking nonlinear/second-order work. Implementing around either
absence would hide unreviewed formulas or depend on private scratch side
effects.

Integration note for current main, reviewed without rebasing this candidate:
the concrete non-curvature primitive producer is already bound into the live
source composition. Route B must reuse that binding/provider seam rather than
reimplement it. The local `h[0..4]` curvature provider and the outer
projection/derivative producer remain pending. The concrete next
implementation sequence is therefore:

1. factor and test the ten-field linear graph method;
2. add the preallocated five-level derivative tower and two JVP calls;
3. derive/check the three Kerr-background derivative slots;
4. add the local value producer and call it on the three consecutive triples;
5. extend the three-resolution certificate to all six fields;
6. connect the resulting read-only adapters through the existing concrete
   live-source provider seam;
7. implement and qualify the still-pending outer projection/derivative
   producer separately.
