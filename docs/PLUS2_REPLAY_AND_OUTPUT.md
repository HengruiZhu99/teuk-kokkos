# Standalone spin +2 companion and replay orchestration

## Qualified architecture slice

`Plus2ReplayOrchestrator` is a standalone, optional orchestration layer. It is
deliberately not a member of `SpatialPipeline`: the production linear
curvature operator and the reviewed quadratic spin +2 source must close their
own gates before that integration is scientifically valid.

The typed modes are:

- `disabled`: no companion state, scratch, replay buffer, or Kokkos launch;
- `diagnostic_only`: reserves no second-order state and is the intended mode
  for metric-curvature `Psi0^(1)` diagnostics;
- `concurrent`: advances a caller-owned primary state and passive companion at
  the same four RK stages;
- `replay`: owns a replay copy of the primary state and deterministically
  re-executes the same common-stage graph from caller-supplied initial data.

The second-order initial policies are `zero` and `checkpoint`. `zero` is only a
development convention. It does not establish that independently initialized
spin -2 and spin +2 second-order fields are components of a single
second-order metric, nor does it impose a nonlinear Teukolsky-Starobinsky
constraint.

## Common-stage and one-way contract

Both evolving modes call `device_one_way_coupled_rk4_step`. Its primary RHS has
no companion argument, so feedback is excluded by the callable interface. The
companion RHS receives primary and companion states from exactly the same RK4
stage. The wrapper copies the primary pipeline's accepted-state
`SourceActivationState` once at the beginning of an accepted step and supplies
that immutable snapshot to all four companion callbacks. Invalid activation
history is rejected before either RHS is called.

The orchestrator and `Plus2CompanionStorage` allocate all state and RK scratch
at construction. Advancing a step performs no allocation, host/device copy, or
fence, and the companion pointer remains stable. Replay initialization makes
one explicit device copy of the initial primary state; checkpoint operations
are also explicit host/device synchronization points outside timesteps.

## Checkpoint contract

The independent companion checkpoint schema is
`teuk.plus2-companion-checkpoint`, version 4. Versions 1 through 3 can be
parsed only to produce an explicit incompatibility result; they cannot be
restored because version 3 provenance was still authored independently of the
actual companion PDE. Every current
payload records:

- the exact raw fixed-tetrad scaling
  `Psi0_raw_fixed_tetrad=(R^5/(L^2-i*a*R*cos(theta))^4)*Z_plus`, composed
  explicitly with the typed compact-source primitive
  `Z0_source=Psi0_raw_fixed_tetrad/R^5=Z_plus/(L^2-i*a*R*cos(theta))^4`;
- the signed-m registry schema and explicit parent/target registries;
- independent first- and second-order angular bandlimits;
- the typed linear-curvature, sourced-companion, and companion initial-policy
  choices;
- the exact Git commit and runtime configuration schema version;
- native byte order and the required IEEE-754 binary64 representation;
- complex serialization order `real-then-imag`;
- companion radial/angular extents and the exact
  `LayoutRight(mode,field,radial,theta)` storage order with field order
  `(P,Q,Z)`;
- accepted time and step;
- exact `M`, `a`, `L`, every radial and angular coordinate, resolved `dt`,
  reduction mode, reduction damping, and dissipation;
- the typed complete-field source normalization and the canonical provenance
  binding schema `plus2-pipeline-derived-v1`;
- a typed SHA-256 identity of the authoritative primary `metadata.txt` and
  `state.bin` bytes, including a domain separator, framed logical names and
  sizes, and an explicit primary-state schema;
- the shared accepted-state source-activation latch;
- an FNV-1a checksum over binary64 real and imaginary components.

Loading reads and validates the magic, schema, version, scaling, registries,
native byte order, floating-point format, component/storage order, shape, byte
count, angular bands, methods, initial policy, Git/config provenance,
activation history, trailing-data condition, and checksum against host-owned
temporary data. Only after all checks pass is the companion device state
mutated. Checkpoint targets must be new paths, preventing an implicit
overwrite. Same-shaped states from a different scientific configuration are
therefore rejected. Exact Git matching is intentionally conservative; a
future migration across commits requires an explicit reviewed conversion, not
an implicit load.

`Plus2CompanionPipeline` now constructs its checkpoint authority from the
actual `UniformRadialGrid`, `TeukolskyParameters`, theta vector, reduction
enum, dissipation, radial operator, allocated registries, and extents before
any Kokkos storage allocation.  The old configuration fields are accepted
only as either a completely absent legacy bundle or a complete binary64-bit
cross-check; a partial bundle, signed-zero change, or one-ULP coordinate edit
is rejected.  The
actual coordinate vector is then stored canonically.  A zero-initialized run
may bind `dt` once on its first accepted step; a restored run requires a
positive checkpoint `dt`, and every step compares it exactly before either
RHS callback.  Source normalization is a typed capability rather than two
independently mutable strings.

The primary checkpoint writer hashes the exact metadata/state byte buffers
before atomically publishing them and returns a non-aggregate verified receipt.
The primary loader reads each file once, validates those same buffers for
metadata, geometry, byte count, FNV checksum, and off-band content, and issues
the receipt before primary mutation.  Reopening a path can produce only an
explicitly inspection-only digest, never a verified receipt.  Companion save
requires the paired primary state bytes to match the receipt at the exact same
step/time.  Companion restore first checks its supplied primary state against
the receipt, so a wrong primary cannot mutate either restored state.  A later
four-field production coordinator still must own both artifact paths and their
publication transaction; that cross-file lifecycle belongs to the production
integration gate.

The intended pipeline-derived companion writer/loader boundary uses an
authority token owned by `Plus2CompanionPipeline`; the public raw wrapper writes
only `plus2-unbound-codec-v1`, and config-only orchestrator restore throws.  The
frozen Milestone-1 candidate is not yet authoritative: its public header-level
`plus2_checkpoint_detail` implementations still permit bypassing the wrapper
boundary.  They must be made token-inaccessible before version 4 can be treated
as pipeline-issued provenance.

The concrete `Plus2LiveSourceComposition` now owns the privileged Route-B
stage adapter and pipeline evolution latches whether a trajectory used that
identity or a validation-only callback.  Checkpoint-only authority is still
only a composition identity, however; it does not yet bind the complete source
registry/bands/geometry or prove that an untouched/manual state was evolved by
that graph.  Lower-level `Plus2ReplayOrchestrator` callback APIs also remain
public validation machinery even though not every method name contains
`validation_only`.  None of these seams may be used as production checkpoint
authority until the remaining tests and ownership restrictions are closed.

The loader now rejects nonfinite provenance before device mutation and checks
`time == step*dt` within a small binary64 evaluation bound. Replay initialized
from a checkpoint is latched to the restored accepted time: its first and every
subsequent continuation call must use that time, then advances the latch by the
accepted step. This closes the former ambiguity in which a same-shaped
companion could be resumed at an unrelated caller time.

## Determinism evidence and present limitation

`tests/test_plus2_replay.cpp` proves, for a deterministic passive driven
system, bitwise concurrent/replay equality, exact stage times `(t, t+h/2,
t+h/2, t+h)`, unchanged activation input at every stage, primary bitwise
independence under varied companion initial data, stable storage, no Kokkos
allocation during steps, checkpoint round trip, and fail-before-mutate corrupt
checkpoint handling.

`tests/test_plus2_companion_pipeline.cpp` additionally proves that each
physical mismatch (`M,a,L`, radial/theta coordinates, reduction, damping,
dissipation, operator, source authority, primary receipt/content, and `dt`)
rejects before a PDE callback or either state mutation.  Its physical
checkpoint round trip uses a 24-node D10-5 radial grid and three Gauss-Legendre
theta nodes, destroys
the writer, reconstructs the actual PDE objects, restores bitwise, and then
shows that an otherwise same-shaped different-spin PDE cannot consume the
artifact.  `tests/test_sha256.cpp` pins NIST vectors, chunk invariance,
canonical lowercase parsing, an independently frozen 65,537-byte framing
digest, and independent metadata/state byte mutations.

`tests/test_plus2_routeb_physical_replay.cpp` closes the next, narrower
scientific seam using the actual disabled Route-B graph rather than an
arbitrary forcing callback.  Its flat caller-owned primary state contains the
three spin-minus-two first-order fields and seven ORG reconstruction fields.
At each common RK stage it:

1. builds the coefficient-wise angular/radial `h0..h4` Route-B tower from the
   identical primary stage view;
2. packs `h1` as that primary state's source-independent time derivative;
3. evaluates the constrained six-field curvature provider exactly once;
4. applies the concrete primitive, ordered-pair, retained-band outer-source,
   and corrected `S0/R^7` forcing graph;
5. advances the actual spin-plus-two `P,Q,Z_plus` companion PDE.

The frozen rotating-Kerr test uses D10-5, `FreeDamped`, zero dissipation, two
accepted RK steps, active source history, exact future-horizon endpoints, and
nonzero signed parent/target modes.  Concurrent and replay primary states,
final forcing, and companion states are bitwise identical on the same Serial
backend.  Every primary/source stage has the exact time sequence
`(t,t+h/2,t+h/2,t+h)` and the identical primary pointer; generations are
strictly increasing.  The forcing and response are nonzero, the complete
trajectory is linear/quadratic under a common amplitude rescaling, changing
the companion initial state leaves the primary bitwise unchanged, and the
hot two-step trajectory records zero Kokkos allocations and fences.

This is physics-bearing **standalone validation**, but not production replay.
The initial first-order/reconstruction profile is smooth manufactured data,
not a restored campaign checkpoint.  The Route-B graph remains qualified only
for `FreeDamped` with zero dissipation.  The trajectory does not include the
production spin-minus-two second-order state or invoke the four-Weyl output
packer.  It is not wired through `solver_driver`, and it does not establish a
sourced residual convergence campaign or cross-backend reproducibility.

This is an integration foundation, not a four-field production claim. It does
not connect the qualified standalone graph to `SpatialPipeline`, does not
parse the main runtime configuration, and does not write four-field waveform
products. Backend-dependent reduction order in a future production source can
limit bitwise replay across different backends;
same-backend runs should use the exact resolved configuration, build
provenance, initial primary state, and companion checkpoint metadata.

The serialization is version 4. Version 2 introduced radial-scheme identity;
version 3 added physical-problem fields but did not bind them to the concrete
PDE. Version 4 adds pipeline-derived authority, a typed source normalization,
and a typed primary SHA-256 content identity. No old payload is silently
reinterpreted.
