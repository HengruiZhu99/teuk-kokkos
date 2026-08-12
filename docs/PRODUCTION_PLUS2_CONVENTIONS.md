# Production spin `+2` conventions and promotion contract

Status: authoritative short contract for work after handoff
`c841ef7d4ee5ed2df635e88d37f777e099646095`. This document does not
authorize a production spin `+2` run. Detailed derivations remain in the
formalism, source, Route-B, replay, and validation documents.

## Field and perturbative conventions

- Metric signature: `+---`.
- First-order gauge: outgoing-radiation gauge.
- Tetrad: the repository's rotated Kinnersley code tetrad, with `gamma=0`.
- Expansion: `A=A0+epsilon A1+epsilon^2 A2+...`; no factorial is hidden in
  the second-order coefficient.
- Signed sharp: `X_m^sharp=conjugate(X_-m)`, never same-mode conjugation.
- Compact coordinate: `R=L^2/r`, with scri at `R=0` and the future horizon
  at the upper endpoint.

The four physical code-tetrad fields and their regular variables obey

```text
Psi4^(n) = R Z_minus^(n),
Psi0^(n) = R^5 Z_plus^(n)/(L^2-i a R cos(theta))^4,
n in {1,2}.
```

The source-normalized curvature fields are distinct objects:

```text
Z0_source = Psi0^(1)/R^5
          = Z_plus^(1)/(L^2-i a R cos(theta))^4,
Z1        = Psi1^(1)/R^4.
```

Raw `Psi0`, evolved `Z_plus`, `Z0_source`, local metric curvature, TSI
reconstruction, and effective-Einstein-source variables must remain typed and
named separately. The raw second-order Weyl fields in this convention are
not labelled gauge invariant.

## Source normalization

With

```text
D       = L^4+a^2 R^2 cos(theta)^2,
d_minus = L^2-i a R cos(theta),
```

the only evolved spin `+2` forcing is

```text
forcing_plus = 2 D d_minus^4 (S0/R^7).
```

`S0/R^6` remains a separately named raw diagnostic. It must never be passed
to the evolved equation or divided by `R` numerically. The cancellation-safe
outer source evaluates directly

```text
S0/R^7 = A J_T + b J_R + C_J J
        + eth_6 K + R(-4 tau0+pibar0)K - 3 psi20 Q.
```

The optical cancellation is evaluated analytically:

```text
E = L^2-2MR+a^2R^2/L^2,
b = -E/(2D),
(5b-4rho0-rhobar0)/R
  = i a cos(theta) E (3L^2+5 i a R cos(theta))/(2D^2).
```

Source-normalization version 2 is
`plus2_s0_over_r7_complete_field_factor_v2`. A checkpoint or source adapter
must bind that version to the concrete source implementation, not merely copy
caller-authored text.

## Endpoint and curvature authority

Route A Bianchi transport is validation-only for rotating Kerr. Its radial
principal symbol has a nontrivial Jordan block, so it is weakly hyperbolic and
must not be promoted by a boundary prescription.

Production uses local algebraic Route B and must provide, at one RK stage and
one generation,

```text
Z0_source, Z1, Z0_source_T, Z1_T, Z0_source_TT, Z1_TT,
```

plus all eight derivative slots consumed by the source. No value may be
manufactured at scri by a zero substitution or a raw division by `R`.

The checked-in positive-node `q0/q1` moment extractors are diagnostics and
comparison oracles. They are not production authority at fine binary64
resolution. Production authority requires an immutable endpoint
qualification that binds method/version, physical grid, radial/tower
operators, Kerr parameters, signed registry, angular bands, peeling powers,
condition and forward-error estimates, residual ceilings, independent
estimator agreement, and the complete qualification-matrix digest. A fresh
per-generation stage must fail globally when any required signed parent,
sharp partner, coefficient, stamp, residual, estimator, or certificate fails.

Historical `N=65` roundoff amplification remains red evidence. A replacement
method must use a predeclared two-regime gate: design-order convergence while
truncation dominates, and bounded absolute/relative error plus estimator
agreement when a proven roundoff bound dominates. Fine grids through at
least `N=513` remain mandatory; they may not simply be banned.

The corrected Bianchi-5 identity retained as a negative regression is

```text
Delta_5 Z0 = eth_4 Z1 - R mu0 Z0 - 4R tau0 Z1
             + 3 Sigma psi20,
```

not `+3 Sigma H`.

## Common-RK and no-feedback requirement

At each classical RK4 stage the primary spin `-2` state, reconstruction,
Route-B curvature, ordered-pair source, projected outer source, and passive
spin `+2` companion RHS use the same stage state and time. There is no source
interpolation, endpoint-time differencing, previous-step source, or operator
splitting. The accepted activation latch is immutable across the complete RK
step. No companion value may feed back into the primary update.

Hot stage evaluation allocates nothing, copies nothing between host/device,
and fences nowhere. Signed mode and sharp mappings are precomputed.

## Checkpoint and replay semantics

Checkpoint provenance must be derived from, or exactly cross-validated
against, the actual PDE objects. It binds at least `M,a,L`, every radial and
theta coordinate, accepted time and `dt`, radial scheme, reduction mode and
damping, dissipation, signed registries and bands, source-normalization and
endpoint-method versions, activation state, exact code commit, and a content
identity for the primary checkpoint. Validation and checksum checks precede
all state mutation. Old formats are never silently reinterpreted.

Physical replay means primary, curvature, source, and companion traverse one
restored trajectory. Structural callback equality is not physical replay.

## Promotion gates

`plus2.enabled=false` remains the default. The solver must continue to reject
every enabled mode until all of the following are proven together:

1. robust production-resolution Route-B endpoints and all six/eight fields;
2. complete source algebra and independently converged source work bands;
3. exact common-stage sourced companion with no primary feedback;
4. PDE-bound checkpoint/restart and physical replay;
5. integrated four-Weyl output with unambiguous scaling metadata;
6. sourced four-field radial, temporal, angular, and amplitude convergence;
7. a combined stability/timestep rule and long near-extremal qualification;
8. current exact-candidate accelerator runtime and resource evidence;
9. bitwise or predeclared-equivalent unchanged spin `-2` behavior while
   disabled; and
10. a final independent adversarial promotion audit with every mandatory gate
    classified `PROVEN`.

No tolerance, mode, endpoint, or red refinement result may be weakened,
deleted, aggregated away, or silently rewritten to obtain promotion.
