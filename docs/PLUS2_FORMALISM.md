# Spin `+2` companion formalism map

This is the entry point for the spin `+2` scientific authority.  It does not
replace the equation-level ledgers and executable audits linked below.

## Selected field

The companion evolves the raw perturbative coefficient `Psi0^(2)` in the same
first-order ORG and fixed rotated-Kinnersley code tetrad as the existing raw
`Psi4^(2)`.  Its regular evolved variable is

```text
Psi0^(2) = R^5 Z_plus^(2)/(L^2-i a R cos(theta))^4.
```

This choice is deliberately different from the reduced linear-in-`h^(2)`
field driven by the Spiers effective Einstein source.  It is also deliberately
not `D_TS[Psi4^(2)]`: a linear Teukolsky-Starobinsky operator omits the
quadratic source/tetrad terms and homogeneous ambiguity at second order.

The exact gauge, tetrad, perturbative, sharp, scaling, and boundary contracts
are in `PLUS2_FIELD_AND_TETRAD_CONVENTIONS.md`.

## Derivation authority

1. `PLUS2_FORMALISM_GATE.md` identifies the raw variable, derives the
   ungauge-fixed prime map, specializes the plus equation directly to ORG,
   and records the source normalization.
2. `PLUS2_SOURCE_COMPACTIFICATION.md` and
   `PLUS2_SOURCE_TERM_LEDGER.csv` define the 51 compact terms and their
   weights, radial powers, ordinary/sharp inputs, and family closure.
3. `tools/symbolic/verify_plus2_formalism_gate.py` and
   `tools/symbolic/verify_plus2_compact_source.py` are the independent exact
   and high-precision gates.
4. `plus2_equation_spec_proposal.yaml` remains fail closed until the runtime
   production provider is qualified.

The linear curvature authority is split between
`PLUS2_LINEAR_DERIVATION_REVIEW.md`, the independent Schwarzschild coordinate
oracle, the rotating-Kerr automatic-differentiation coordinate oracle, and
`PLUS2_LINEAR_PSI0_PRODUCTION_GATE.md`.  The normalized separated-mode TSI
status is maintained in `PLUS2_VALIDATION_AND_TSI_AUDIT.md`.

## Implemented standalone graph

The reviewed standalone path now contains:

- the exact regular homogeneous spin `+2` Teukolsky coefficients and scaling;
- a Kokkos `T0[h]` point/spatial graph;
- all fourteen compact source primitives and the 51-term pair algebra;
- the concrete D10-5 non-curvature primitive producer;
- signed retained-band projection and the concrete `thorn_5 J`/`eth_6 K`
  outer graph;
- a passive spin `+2` Teukolsky triple and provider-independent common-stage
  RK4 seam;
- replay/checkpoint/output contracts with no-feedback and representation
  validation;
- four-field packing with explicit physical/internal scaling and metadata.

These components are not yet reachable from `solver_driver` when
`plus2.enabled=true`.

## Production curvature decision

The initially implemented Route-A `Z0,Z1` Bianchi transport is retained only
as validation machinery.  In rotating Kerr its pure-radial principal symbol
has a nontrivial Jordan block, so the system is weakly rather than strongly
hyperbolic.  Boundary evidence cannot repair that bulk defect.

Rotating production therefore uses Route B: construct local metric curvature
and its two time derivatives from a same-stage primary/reconstruction graph.
The three previously missing stationary Kerr background derivative slots are
implemented and independently audited.  A naïve recursive D10-5
`h[0]..h[4]` tower was rejected: its deepest endpoint ratios are only about
`5--6`, not the fourth-order target near `16`.

`PLUS2_ROUTE_B_DERIVATIVE_TOWER_BLOCKER.md` specifies the controlled remedy:
direct independently generated `D1..D4` radial jets and coefficient-aware
point algebra, initially for zero-dissipation `FreeDamped` validation.  A
nonzero-dissipation production path remains a separate hard gate.

## Promotion boundary

Do not enable the runtime until all of the following are true together:

1. the local Route-B provider produces all six `Z0,Z1` value/time-derivative
   fields and the four required derivative pairs at one RK generation;
2. all six scri coefficients and peeling residuals have a bound convergence
   certificate for physical metric data;
3. the sourced residual converges in radius, angle, and time;
4. concurrent and replay paths agree using the concrete provider;
5. all four fields are written at common accepted times;
6. every available accelerator backend passes the exact production kernels;
7. the disabled path remains allocation/kernel free and preserves the
   qualified minus-two trajectory.

Until then, runtime rejection of `plus2.enabled=true` is intentional
fail-closed behavior, not a missing parser feature.
