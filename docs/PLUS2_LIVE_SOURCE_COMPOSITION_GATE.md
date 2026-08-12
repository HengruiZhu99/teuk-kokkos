# Spin `+2` live-source partial composition gate

**Status:** allocation-free standalone composition boundary and common-stage
validation seam; not wired to `solver_driver` and blocked from scientific
production by the rotating Route-A weak-hyperbolicity finding.

## What is composed

`Plus2LiveSourceComposition` fixes the stage order

1. decoded first-order/reconstruction value, tangent, and second tangent;
2. the D10-5 metric-curvature `Psi0` graph required by the nested fourth-order
   live path;
3. externally supplied same-stage, read-only Bianchi adapters containing
   transported `Z0,Z1` values and first/second tangents plus the four Route-A
   derivative pairs;
4. the concrete D10-5 producer for the twelve non-curvature primitives, seven
   remaining `J/K` pairs, and three value-only `Q` derivatives;
5. value-only ordered-pair source evaluation;
6. projected `J,K,Q`, outer derivatives, coordinate forcing, and target-mode
   gather.

The accepted-step source-activation snapshot is compared with the companion
target. An inactive snapshot writes zero forcing without evaluating a stale
source. The adapter owns all repeated scratch and launches no allocation,
copy, or fence during an active stage.

Every producer-owned primitive, derivative, aggregate, and outer-derivative
slot has a generation stamp. Transported curvature has a separate stamp for
each of `Z0,Z1,Z0_T,Z1_T,Z0_TT,Z1_TT`, and the derivative adapter separately
stamps all eight value/tangent components. All stamps must equal the same live
generation. A missing or stale stamp clears the point readiness bit and the
gathered forcing is zero. This prevents a partial producer from silently
reusing values from a preceding RK stage. Generation IDs must increase at
every adapter invocation. Reconstruction, transported curvature, and Bianchi
derivatives are exposed to producers through read-only views, so this
interface cannot write feedback into the primary or Bianchi state.

Transport owns exactly the value/tangent pairs for `Delta_4 Z1`,
`ethprime_4 Z1`, `Delta_5 Z0`, and `eth_5 Z0`. The normalization kernel copies
those eight adapter components into the corresponding four production `J/K`
rows after the primitive callback and ignores producer stamps for only those
rows. Thus even hostile callback writes cannot replace the common-stage
transport values. The other seven `J/K` rows and all three `Q` rows remain
producer-owned and require current-generation producer stamps.

## Concrete binding and remaining blockers

The scientific `evaluate_stage` overload accepts a generation-stamped
`Plus2PrimitiveReconstructionStage` and invokes
`Plus2SourcePrimitiveSpatialProducer` directly. The caller cannot substitute
a generic primitive callback on this path. The older callback overload is
retained only as a low-level component-test seam. `Plus2BianchiTransport`
supplies the triangular Route-A `Z0,Z1` transport plus the six-field curvature
and eight-field derivative adapters, completing the fourteen primitive rows
and all production inner-derivative slots at a common generation.

The standalone binding is still not `solver_driver` wiring. More decisively,
the rotating two-field Route-A transport has a Jordan radial principal symbol
and is only weakly hyperbolic, despite its repeated speed being outward. The
typed Route-A adapter is therefore validation-only and cannot be promoted by
adding boundary evidence. The expected production direction is a local
Route-B curvature provider from `h[0..4]`; it is not implemented or qualified
here. The composition and companion common-stage RHS keep the provider seam
external so that an algebraic provider need not own an evolved middle state.
The outer projection/derivative producer remains an explicit reviewed seam.
Physical Bianchi initialization, horizon/scri boundary data, and peeling
coefficients remain external evidence requirements. `Z0,Z1` and their four
derivative pairs must come from the common one-way RK state or exact
deterministic replay; they cannot be hidden mutable state in a source callback.
A grid containing
scri is rejected unless the caller asserts independently qualified peeling
coefficients; neither endpoint extrapolation nor a zero coefficient is
invented here.

The binding and its manufactured seam test are not evidence for the physical
Bianchi initial/boundary prescription, a complete runtime integration, or a
physical spin `+2` waveform. See `PLUS2_BIANCHI_TRANSPORT.md` for the exact
transported-state, restart, and fail-closed contracts.

## Current focused evidence

The focused tests establish D10-5 selection, exact callback ordering, typed
common-stage curvature/derivative delivery, exact transport ownership despite
hostile producer writes, wrong-shape and stale-derivative fail closure,
continued producer ownership of the remaining seven `J/K` and all `Q` rows,
immutable activation behavior, target gather, quadratic common-amplitude
scaling, and zero stage allocations/fences. The
ordinary-NP algebra, signed sharp lookup, analytic `J/K` tangents, and compact
outer source remain independently tested by their existing point and spatial
test suites. A one-step manufactured three-state seam additionally exercises
all four common RK stages from primary through passive Route-A validation,
concrete primitive production, live composition, and the passive Teukolsky
companion RHS and advance. It verifies exact stage times and generations
`1..4`, immutable activation, no primary feedback, nonzero forcing and
companion advance, quadratic scaling, global stale-stage fail closure, D10-5,
stable pointers, and no per-stage allocation or fence. A fixed-space
manufactured refinement separately verifies fourth-order RK time convergence.
It deliberately supplies only a test outer adapter and test-only boundary
evidence; it is neither runtime/boundary qualification nor evidence that the
Route-A system is well posed for rotating production.
