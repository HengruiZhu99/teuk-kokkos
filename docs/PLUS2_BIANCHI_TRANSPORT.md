# Common-stage spin `+2` Bianchi transport

**Status:** standalone, allocation-free Route-A validation transport and
restart gate with a typed adapter into the standalone live-source composition;
blocked from production use by rotating weak hyperbolicity and not wired to
`solver_driver`. The characteristic and endpoint evidence gate is documented
in `PLUS2_CURVATURE_INITIAL_AND_BOUNDARY_GATE.md`.

For nonzero background rotation, the two-field Route-A radial principal symbol
has a repeated outward characteristic speed but only a Jordan chain: the
`eth_4 Z1` time coupling makes it weakly hyperbolic. Outward speed alone is not
a well-posed production evolution argument. The common-stage implementation
and tests below therefore qualify structural and algebraic validation
machinery only. Physical initialization or boundary evidence cannot promote
this Route-A system past the hyperbolicity blocker.

## Owned state and one-way RK contract

`Plus2BianchiTransport` owns exactly the regular first-order curvature state

```text
Psi0^(1) = R^5 Z0,        Psi1^(1) = R^4 Z1.
```

It also owns the classical-RK4 stage and `k1..k4` storage, radial and angular
scratch, Route-A constraints, source-derivative outputs, readiness, and
generation stamps. `device_one_way_bianchi_transport_rk4_step` advances this
state at the same four stage times as the primary. Its signatures enforce the
dependency direction

```text
primary stage --> h[0..2] producer --> passive Z0/Z1 RHS --> stage observer
```

The primary RHS cannot see the Bianchi state. The `h[0..2]` producer receives
a const-data primary view. The observer receives both the read-only six-field
curvature adapter and the read-only eight-field derivative adapter explicitly,
with common generation stamps; live integration need not capture transport
internals indirectly. There is no path from either passive object back into
the primary update.

At every stage the adapter writes, in fixed order,

```text
(Z0, Z1, Z0_T, Z1_T, Z0_TT, Z1_TT)
```

and stamps every component with that stage's strictly increasing generation.
The companion derivative adapter also exposes

```text
(Delta4 Z1, (Delta4 Z1)_T,
 ethprime4 Z1, (ethprime4 Z1)_T,
 Delta5 Z0, (Delta5 Z0)_T,
 eth5 Z0, (eth5 Z0)_T).
```

These are the four curvature derivative pairs needed by the value-only live
source contract. The live gate copies these pairs into its four corresponding
`J/K` value/tangent rows and retains producer ownership of the other seven
`J/K` rows and all three `Q` rows. Transient adapters are deliberately not
checkpointed; the first post-restart common stage regenerates them.

## Exact triangular closure order

The implementation calls the reviewed point functions in
`plus2_bianchi.hpp`. For each signed mode and collocation point it enqueues:

1. the selected SBP first derivative of `Z0,Z1` and `eth_3 H`,
   `eth_3 H_T`;
2. `F1=Delta_4 Z1` from (B1), followed by exact `Delta_n` inversion for
   `Z1_T`;
3. `eth_4 Z1` and `ethprime_4 Z1`, followed by (B0) and inversion for `Z0_T`;
4. SBP derivatives of `Z0_T,Z1_T`, then the Jet tangent of (B1) and inversion
   for `Z1_TT`;
5. `eth_4 Z1_T` and `ethprime_4 Z1_T`, then the Jet tangent of (B0) and
   inversion for `Z0_TT`;
6. `eth_5 Z0`, `eth_5 Z0_T`, four algebraic `Delta_n-F_n` residuals, and the
   final derivative/curvature generation stamps.

This order is triangular and requires only complete primary/reconstruction
`h[0]`, `h[1]`, and `h[2]`. It never differentiates the metric-curvature
quotient again and never asks for `h[3]` or `h[4]`.

This convenient closure depth does not cure the rotating principal-symbol
defect. The planned production alternative is a local Route-B curvature
provider derived from `h[0..4]`; that provider and its endpoint evidence are
not implemented here.

In particular, the corrected linear Bianchi-5 curvature term is
`+3 Sig psi20`, where `psi20` is stationary background curvature. Its exact
stage tangent is `+3 Sig_T psi20`; neither expression contains an `H` or
`H_T` factor.

All fixed-`m` angular transforms and every Kokkos view are constructed before
the first stage. The repeated stage and full common-RK4 paths perform no
allocation, deep copy, or fence. Kernel ordering is carried by the supplied
execution-space instance.

## Fail-closed scientific contracts

There is no default or zero initializer. Initialization requires a nonempty
evidence identifier and affirmative evidence for metric-curvature consistency
and radial boundary data. A grid containing `R=0` additionally requires
independently qualified `Z0,Z1` peeling coefficients. The transport does not
extrapolate, divide a raw curvature numerator at scri, or manufacture a zero
coefficient.

Construction also fails closed unless `M`, `a`, and `L` are finite, `M>0`,
`L>0`, and `|a|<=M`. The checkpoint metadata validator repeats these physical
background checks before restoration can mutate device state.

Each stage separately requires evidence for common-RK ownership, complete
`h[0..2]`, the retained angular band, the radial boundary treatment, the
selected radial operator, and (at scri) the endpoint coefficients. Before any
stage launches, its boundary evidence identifier must exactly equal the one
accepted at initialization (and preserved by a checkpoint). Before any
closure kernel, all primary fields and stamps and all state values are checked
on device. One missing stamp or nonfinite value invalidates the whole stage:
the RHS, derivative adapter, curvature adapter, constraints, and their stamps
become zero/invalid rather than reusing an earlier stage.

The transport defaults to `D10-5`, which the live source requires because the
Route-A graph differentiates the already differentiated `Z_T` state and needs
the extra boundary order for nested overall fourth order. The shared
`RadialDiscretization` dispatch also permits explicit `D4-2` or `D8-4`
instances for narrower tests, but these are not production live-composition
choices. There is no hidden hard-coded stencil in the transport, and the stage
capability must name the same scheme as the transport instance.

## Restart contract

`plus2_bianchi_transport_checkpoint.hpp` defines a separate versioned binary
format for only the passive `Z0,Z1` state. It records exact scaling and field
order, signed-mode registry, angular/radial geometry, Kerr parameters, radial
scheme, initialization and boundary evidence identifiers, scri readiness,
Git/config provenance, progress, the last generation, and a binary64 FNV-1a
state checksum.

Loading parses, validates, checks provenance/geometry/evidence, verifies the
checksum, and checks the complete state count before mutating device storage.
The next generation continues monotonically. A restored object intentionally
has no `latest_stage()` until it evaluates a fresh common RK stage.

## Qualified scope and remaining production work

The focused tests cover:

- manufactured common-stage RK4 temporal convergence at fixed spatial
  resolution (not a spatial-convergence claim);
- D10-5 nested radial convergence of a Schwarzschild analytic `Z0_TT`
  manufactured solution;
- exact stage times and device-observed generation stamps at all four stages;
- algebraic Route-A value/tangent self-residual closure, an independent
  nonzero `Sig,Sig_T` oracle for `F0,F0_T`, and common-amplitude scaling;
- read-only primary input and one-way driver structure;
- missing initialization, boundary, scri, and per-slot stage evidence;
- checkpoint round trip and validation-before-mutation;
- repeated-stage and full-RK4 allocation/fence absence.

These tests qualify the standalone transport mechanism as validation
machinery, not a well-posed rotating production evolution or physical initial
or boundary data. A standalone three-state coordinator also verifies one
common RK tableau for primary, Route-A validation state, and passive
Teukolsky companion without operator splitting. It does not remove the Route-A
Jordan block. The curvature gate supplies a typed, resolution-bound D10-5
l'Hopital operator for independently produced peeling numerators and proves
the continuum characteristic directions; its analytic profiles are quotient
tests, not physical metric data. Production must pivot to the local Route-B
provider from `h[0..4]`, earn a physical full-plane three-resolution
certificate, and qualify the complete four-field convergence/TSI campaign
before any `solver_driver` wiring.
