# Spin `+2` live-source partial composition gate

**Status:** allocation-free standalone composition boundary; not wired to
`solver_driver` and not a scientific production qualification.

## What is composed

`Plus2LiveSourceComposition` fixes the stage order

1. decoded first-order/reconstruction value, tangent, and second tangent;
2. the D8-4 metric-curvature `Psi0` graph;
3. externally supplied same-stage transported `Z0,Z1` values and first/second
   tangents;
4. the typed fourteen-primitive and derivative views;
5. value-only ordered-pair source evaluation;
6. projected `J,K,Q`, outer derivatives, coordinate forcing, and target-mode
   gather.

The accepted-step source-activation snapshot is compared with the companion
target. An inactive snapshot writes zero forcing without evaluating a stale
source. The adapter owns all repeated scratch and launches no allocation,
copy, or fence during an active stage.

Every producer-owned primitive, derivative, aggregate, and outer-derivative
slot has a generation stamp. Transported curvature has a separate stamp for
each of `Z0,Z1,Z0_T,Z1_T,Z0_TT,Z1_TT`. A missing or stale stamp clears the
point readiness bit and the gathered forcing is zero. This prevents a partial
producer from silently reusing values from a preceding RK stage. Generation
IDs must increase at every adapter invocation. Reconstruction and transported
curvature are exposed to producers through read-only views, so this interface
cannot write feedback into the primary or Bianchi state.

## Exact remaining blocker

The repository does not yet contain the allocation-free spatial graph that
constructs all fourteen primitive rows and their production derivative slots
from the live reconstruction stage. It also does not yet evolve the triangular
Route A Bianchi transport state in the same coupled RK4 state as the primary
and second-order companion.

Consequently the live gate requires explicit capability claims and external
same-stage curvature views. `Z0,Z1` must be either part of that common one-way
RK state or supplied from an exact deterministic replay stage. They must not
be hidden mutable state in a source callback. A grid containing scri is
rejected unless the caller asserts independently qualified peeling
coefficients; neither endpoint extrapolation nor a zero coefficient is
invented here.

The source and outer callbacks are typed seams for the missing reviewed
spatial graph. Their existence is not evidence that this graph, the Bianchi
initial/boundary prescription, concurrent/replay equivalence, or a physical
spin `+2` waveform has been qualified.

## Current focused evidence

The focused tests establish D8-4 selection, exact callback ordering,
generation-stamp fail closure, immutable activation behavior, target gather,
quadratic common-amplitude scaling, and zero stage allocations/fences. The
ordinary-NP algebra, signed sharp lookup, analytic `J/K` tangents, and compact
outer source remain independently tested by their existing point and spatial
test suites. Those component tests do not replace the missing full live-graph
qualification described above.
