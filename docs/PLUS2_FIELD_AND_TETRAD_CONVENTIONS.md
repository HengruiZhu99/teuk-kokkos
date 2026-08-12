# Spin `+2` field and tetrad conventions

This file is the canonical short contract for the four Weyl fields.  Detailed
derivations remain in `PLUS2_FORMALISM_GATE.md`,
`PLUS2_LINEAR_DERIVATION_REVIEW.md`, and
`PLUS2_HOMOGENEOUS_ARCHITECTURE.md`.

## Fixed conventions

- Signature: `+---`.
- Gauge: first-order outgoing-radiation gauge (ORG),
  `h_ab n^b=0` and `g0^ab h_ab=0`.
- Tetrad: the horizon-regular Ripley rotated-Kinnersley code tetrad with
  `gamma=0`.
- Perturbative expansion:
  `A=A0+epsilon A1+epsilon^2 A2+O(epsilon^3)`, with no factorial.
- Signed-mode sharp operation: `X_m^sharp=conjugate(X_-m)`.  Conjugating the
  same stored `m` is not sharp.
- Compact radius: `R=L^2/r`, with future null infinity at `R=0` and the
  future horizon at the upper grid endpoint.

The second-order fields below are raw coefficients in this fixed
gauge/tetrad convention.  They are not labelled gauge invariant.  No linear
spin-reversal operator is applied to `Psi4^(2)` to define `Psi0^(2)`.

## Physical and stored fields

Let

```text
D_plus = L^2-i a R cos(theta),
W_minus = R,
W_plus  = R^5/D_plus^4.
```

The output contract is

| physical fixed-tetrad field | order | spin | boost | stored field | inverse scaling |
|---|---:|---:|---:|---|---|
| `Psi4^(1)` | 1 | -2 | -2 | `F^(1)` | `Psi4^(1)=R F^(1)` |
| `Psi0^(1)` | 1 | +2 | +2 | `Z_plus^(1)` | `Psi0^(1)=W_plus Z_plus^(1)` |
| `Psi4^(2)` | 2 | -2 | -2 | `F^(2)` | `Psi4^(2)=R F^(2)` |
| `Psi0^(2)` | 2 | +2 | +2 | `Z_plus^(2)` | `Psi0^(2)=W_plus Z_plus^(2)` |

The spin and boost columns use `s=(p-q)/2` and `b=(p+q)/2` for a GHP type
`{p,q}`.  Multiplication by the coordinate scaling factors does not change
those weights.

The compact quadratic source uses two additional regular curvature variables
which must not be confused with the evolved field:

| source variable | definition | spin | boost | scri falloff removed |
|---|---|---:|---:|---:|
| `Z0` | `Psi0^(1)/R^5 = Z_plus^(1)/D_plus^4` | +2 | +2 | 5 |
| `Z1` | `Psi1^(1)/R^4` | +1 | +1 | 4 |

At finite `R`, conversion between `Z_plus` and `Z0` is multiplication by
`D_plus^-4`; it is not another evolution equation.  At scri the conversion
is the finite limit `Z0=Z_plus/L^8`.

## Boundary interpretation

At scri the raw code-tetrad values vanish.  Output stores the finite leading
coefficients

```text
lim_(R->0) Psi4/R   = F|scri,
lim_(R->0) Psi0/R^5 = Z_plus|scri/L^8.
```

Code must not recover either coefficient by dividing two zeros.  Interior and
horizon values use multiplication by `W_minus` or `W_plus`.  The raw horizon
value is convention fixed and is not, by itself, a flux observable.

The `Z0,Z1` source coefficients at scri require a qualified peeling
construction.  `plus2_curvature_initialization.hpp` supplies a fail-closed
three-resolution quotient certificate for already supplied numerator data;
it does not turn an analytic test profile into physical metric initial data.

## Method and initial-data semantics

The production-intended linear `Psi0^(1)` method is the local metric-curvature
operator `T0[h^(1)]`.  The normalized Teukolsky-Starobinsky identities are an
independent validation route, not the default field constructor.

The intended `Psi0^(2)` is the raw ORG sourced companion derived from the
ungauge-fixed Campanelli-Lousto/Loutrel equation and specialized directly to
the repository tetrad.  It is not the reduced Spiers effective-Einstein-source
field and is not obtained by priming the already ORG-specialized minus-two
source.

A zero-initialized companion is a documented development convention.  It
does not prove that independently initialized `Psi0^(2)` and `Psi4^(2)` are
components of one common second-order metric.  A checkpoint initialization
must match the complete scientific and representation metadata before device
state mutation.

## Machine-readable/output authority

- `plus2_equation_spec_proposal.yaml` records the fail-closed source proposal.
- `PLUS2_SOURCE_TERM_LEDGER.csv` is the 51-term compact source ledger.
- `four_weyl_output.hpp` owns the exact output scaling strings, field order,
  metadata schema, and boundary semantics.
- `PLUS2_SOURCE_INPUT_MANIFEST.csv` owns the source primitive definitions and
  weights.

The runtime remains disabled for `plus2.enabled=true` until the local
rotating-Kerr curvature provider and the remaining production validation gates
are complete.
