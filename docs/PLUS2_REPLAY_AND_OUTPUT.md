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
`teuk.plus2-companion-checkpoint`, version 1. Every payload records:

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

The companion checkpoint does not currently store the resolved timestep.
Consequently, the loader can require a finite nonnegative progress time and
check activation times against it, but it cannot prove a relation such as
`time == step * dt`. The primary resolved configuration/checkpoint remains the
authority for that consistency check; the orchestration layer must not infer a
timestep from the two progress fields.

## Determinism evidence and present limitation

`tests/test_plus2_replay.cpp` proves, for a deterministic passive driven
system, bitwise concurrent/replay equality, exact stage times `(t, t+h/2,
t+h/2, t+h)`, unchanged activation input at every stage, primary bitwise
independence under varied companion initial data, stable storage, no Kokkos
allocation during steps, checkpoint round trip, and fail-before-mutate corrupt
checkpoint handling.

This is an integration foundation, not a four-field production claim. It does
not connect unqualified `T0[h]` or spin +2 source formulas to
`SpatialPipeline`, does not parse the main runtime configuration, and does not
write four-field waveform products. Backend-dependent reduction order in a
future production source can limit bitwise replay across different backends;
same-backend runs should use the exact resolved configuration, build
provenance, initial primary state, and companion checkpoint metadata.

The serialization additions remain checkpoint format version 1 because this
standalone format is still unreleased and has no supported external consumer.
