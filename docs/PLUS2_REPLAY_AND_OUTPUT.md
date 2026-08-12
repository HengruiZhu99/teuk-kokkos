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
`teuk.plus2-companion-checkpoint`, version 3. Versions 1 and 2 can be parsed
only to produce an explicit incompatibility result; they cannot be restored
because they do not uniquely identify the physical problem. Every current
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
- the explicitly versioned complete-field source normalization and the
  identity of the primary checkpoint from which this passive state was made;
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

The serialization is version 3. Version 2 introduced radial-scheme identity;
version 3 intentionally breaks compatibility by adding complete
physical-problem and source-normalization provenance. No old payload is
silently reinterpreted.
