# Teukolsky radial SAT blocker

## Status

The corrected compact first-order equations determine the continuum radial
characteristic speeds, but they do not yet uniquely determine a nonzero,
energy-stable SAT penalty for the D4-2 semi-discretization. The spatial
evaluator therefore continues to use the verified SBP closures with no SAT
term. No boundary data are imposed.

## Derivable continuum result

Write the radial principal part as `partial_T U = A partial_R U` for
`U=(P,Q,psi)`. Its eigenvalues in this sign convention are

```text
lambda_0 = 0
lambda_+- = (K +- sqrt(K^2 + C_T H_R)) / C_T
```

and coordinate propagation velocities are `v=-lambda`. The coefficient of
`partial_R psi` in the Q equation changes characteristic vectors but not these
eigenvalues.

At scri, `H_R=0`, `K=L^2>0`, and `C_T>0`. There is one nonzero velocity toward
decreasing R, hence out of the lower boundary, plus two stationary modes. At
the outer horizon, with `R_H=L^2/r_+`,

```text
H_R(R_H)=0
K(R_H)=-2 M R_H<0
```

so the one nonzero velocity points toward increasing R, out of the upper
boundary, again with two stationary modes. Thus neither endpoint has a
continuum incoming propagating mode.

The reduction constraint `C_Q=Q-partial_R psi` satisfies exactly

```text
partial_T C_Q = -gamma_Q C_Q
```

and is stationary and damped rather than incoming.

## Why a nonzero SAT is blocked

The natural symmetrizer of the propagating `(P,Q)` block is proportional to
`diag(1,C_T H_R)`. Although it symmetrizes that block and is positive in the
open exterior, it degenerates at both endpoints because `H_R=0`. The supplied
references do not provide:

- a nondegenerate endpoint symmetrizer for the full complex `(P,Q,psi)`
  reduction;
- a normalization and projector for the two zero-speed endpoint modes;
- boundary data for any zero-speed mode;
- a penalty strength tied to a semi-discrete D4-2 energy estimate;
- a decision that reconciles free-damped and stage-constrained reduction at
  the discrete boundary.

Adding a penalty to an outgoing or stationary field without these ingredients
would invent boundary physics and could create reflections or constraint
growth. A zero penalty is consistent with the continuum incoming-mode count,
but this observation alone is not a proof of long-time semi-discrete
stability.

## Evidence required to unblock production SAT qualification

1. Derive a full endpoint energy or characteristic normalization including
   the reduction constraint and complex `G_m` coupling.
2. Prove the frozen-coefficient semi-discrete estimate with the actual D4-2
   norm and closure.
3. Specify penalties only for modes proven incoming under that estimate.
4. Run normal-mode and long-time outgoing-pulse tests at scri and the horizon,
   including near-extremal resolution sequences and both reduction strategies.

The executable characteristic identities and endpoint classifications are
covered by `tests/test_boundary.cpp`.
