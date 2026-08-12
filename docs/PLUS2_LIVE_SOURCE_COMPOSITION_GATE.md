# Spin `+2` live-source partial composition gate

**Status:** allocation-free standalone composition boundary; not wired to
`solver_driver` and not a scientific production qualification.

## What is composed

`Plus2LiveSourceComposition` fixes the stage order

1. decoded first-order/reconstruction value, tangent, and second tangent;
2. the D10-5 metric-curvature `Psi0` graph required by the nested fourth-order
   live path;
3. externally supplied same-stage, read-only Bianchi adapters containing
   transported `Z0,Z1` values and first/second tangents plus the four Route-A
   derivative pairs;
4. the typed fourteen-primitive and remaining producer-owned derivative views;
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

## Exact remaining blocker

The repository does not yet contain the allocation-free spatial graph that
constructs all fourteen primitive rows and their production derivative slots
from the live reconstruction stage. `Plus2BianchiTransport` now supplies a
standalone common-stage triangular Route-A `Z0,Z1` transport plus the six-field
curvature and eight-field derivative adapters. Those adapters now connect
directly to this standalone live gate, but neither component is wired to the
primary and second-order companion runtime.

Consequently the live gate requires explicit capability claims and external
same-stage curvature and derivative views. `Z0,Z1` and their four derivative
pairs must be either part of that common one-way RK state or supplied from an
exact deterministic replay stage. They must not be hidden mutable state in a
source callback. A grid containing scri is rejected unless the caller asserts
independently qualified peeling coefficients; neither endpoint extrapolation
nor a zero coefficient is invented here.

The source and outer callbacks are typed seams for the missing reviewed
spatial graph. Their existence, and the standalone transport, are not evidence
that this graph, the physical Bianchi initial/boundary prescription,
concurrent/replay equivalence, or a physical spin `+2` waveform has been
qualified. See `PLUS2_BIANCHI_TRANSPORT.md` for the exact transported-state,
restart, and fail-closed contracts.

## Current focused evidence

The focused tests establish D10-5 selection, exact callback ordering, typed
common-stage curvature/derivative delivery, exact transport ownership despite
hostile producer writes, wrong-shape and stale-derivative fail closure,
continued producer ownership of the remaining seven `J/K` and all `Q` rows,
immutable activation behavior, target gather, quadratic common-amplitude
scaling, and zero stage allocations/fences. The
ordinary-NP algebra, signed sharp lookup, analytic `J/K` tangents, and compact
outer source remain independently tested by their existing point and spatial
test suites. Those component tests do not replace the missing full live-graph
qualification described above.
