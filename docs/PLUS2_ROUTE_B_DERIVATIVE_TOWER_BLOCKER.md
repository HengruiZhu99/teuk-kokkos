# Route-B `h[0]..h[4]` derivative-tower blocker

Status: production implementation withheld because the D10-5 endpoint gate is
red

Base tested: `aa3d3a2dbd5df578bab4f58c6cbc203ada631d0a`

## Narrow graph experiment

The existing `SpatialPipeline` was experimentally refactored to expose the
source-independent ten-field linear graph

```text
L : (FirstP, FirstQ, FirstPsi, G, Lambda, H, B, Pi, C, U)
    -> partial_T of the same ten fields.
```

The extracted graph used the production signed-mode angular coordinators,
Galerkin projections, selected radial SBP operator, dissipation, and reduction
policy.  It bypassed the quadratic source and second-order Teukolsky kernels,
so it behaved identically for disabled, constraint-aware, and unrestricted
source policies.

A preallocated device tower formed

```text
h[0] = input,
h[k+1] = L(h[k]),  k=0,1,2,3.
```

It used strict ten-field shapes, nonaliasing storage, pointwise generation
stamps, global stale/nonfinite fail-closed behavior, and no allocations or
fences in the hot evaluation path.

The following gates passed in the experimental branch:

- `h[1]` matched the first ten fields of the existing full RHS on rotating
  signed-mode data;
- `h[2]` matched the existing full graph's independently assembled tangent
  RHS;
- disabled and unrestricted source policies produced bitwise-identical `L`;
- every tower level scaled linearly with the input amplitude;
- generation, stale-input, nonfinite-input, shape, alias, pointer-stability,
  no-allocation, and no-fence tests passed.

## Failed endpoint gate

The required deepest `h[4]` endpoint test used:

- Kerr `M=1`, `a=0.63`, `L=1.4`;
- `R` in `[0,0.64]`;
- D10-5 with zero dissipation;
- exact spin-weighted modal angular data through `ell=3` on seven
  Gauss-Legendre nodes;
- smooth exponential/sinusoidal radial profiles in every one of the ten
  fields;
- the reconstructed scalar `U` component of `h[4]` at both radial endpoints.

Endpoint RMS self-differences were

| radial grids | coarse-minus-medium | medium-minus-fine | ratio |
|---|---:|---:|---:|
| `25,49,97` | `5.48566e-7` | `8.98690e-8` | `6.10407` |
| `49,97,193` | `8.98690e-8` | `1.68797e-8` | `5.32409` |

Fourth order requires an asymptotic ratio near `16`; the ratio instead moves
away from that threshold under refinement.  This is not roundoff: the
differences are many orders above binary64 noise, and increasing both radial
frequency and amplitude preserved the sub-fourth-order behavior.

The result is consistent with repeated differentiation of a fifth-order
boundary closure.  One application of `L` is qualified, and the repository's
existing nested `h[2]` use retains fourth-order endpoint behavior.  Four
applications recursively differentiate boundary truncation error enough to
reduce the observed endpoint order.  D10-5 therefore cannot be claimed to
produce a globally fourth-order `h[4]` tower merely because each individual
first-derivative launch uses the D10-5 stencil.

## Required resolution

Do not add the local `Z0/Z1` Route-B producer on this tower.  First qualify one
of the following without weakening the `>15` endpoint-ratio gate:

1. a higher boundary-order diagonal-norm SBP operator (the repeated-derivative
   order count suggests at least a seventh-order boundary closure);
2. reviewed composed/nested boundary closures that approximate the needed
   higher derivatives directly rather than recursively differentiating prior
   boundary error;
3. a lower-time-order curvature/Bianchi formulation that provably avoids
   `h[3]` and `h[4]` while retaining same-stage semantics.

The experimental graph and tower are intentionally not retained in production
because their central scientific qualification gate failed.  The independent
stationary Kerr background derivatives are separate and remain valid.

## Narrowest honest remedy assessment

A direct radial-Taylor-jet graph is feasible, but it is not the same operation
as applying the current D10-5 first-derivative matrix four times.  For the
production `FreeDamped` first-order reduction with zero dissipation, `L` is
first order in `R`.  Direct initial jets through degree four therefore provide

```text
h[0] degree 4 -> h[1] degree 3 -> h[2] degree 2
              -> h[3] degree 1 -> h[4] value.
```

The narrow implementation should generate independent uniform-grid Fornberg
operators `D1,D2,D3,D4`, never powers of `D1`.  A nine-point window gives at
least fourth-order accuracy for every derivative even in a fully one-sided
window.  Offline exact-rational generation must pin every weight and verify
the monomial moments through degree eight; production stores only the reviewed
double coefficients.  Tests must include endpoint polynomial exactness,
exponential/trigonometric convergence, coefficient norms/noise amplification,
and an independent arbitrary-precision oracle.

The current point algebra can be factored over a small product-rule radial jet.
The exact stationary coefficient jets required are:

- Teukolsky `C_T^-1`, `K`, `H_R`, `G_m`, the `Q` coefficient, and the `psi`
  coefficient through radial degree four;
- reconstruction `mu0`, `tau0`, `pi0` and their sharp partners, the
  `Delta_n` inversion factor, and its `R^2/L^2` and `n R/L^2` multipliers
  through the degree consumed at each pass;
- the `eth_n` and `ethprime_n` denominators and connection factors
  `(L^2 +/- i a R cos(theta))^-1` and their squares as coefficient jets.

The angular matrices themselves are independent of `R` and may act on each
radial-jet coefficient.  The exact angular inputs are:

```text
spin Laplacian: Psi,
raise:           F, G, C, Pi,
lower:           Bsharp, Csharp,
projection:      P,Q,Psi,G,Lambda,H,B,Pi,C,U.
```

Only the pure spin-weighted raise/lower or Laplacian matrices commute with
radial differentiation.  The existing `eth`/`ethprime` wrappers as a whole do
not: they also contain the stage tangent and the R-dependent Kerr denominator.
Those point formulas must be evaluated in radial-jet algebra after applying
the pure angular matrix.  The triangular dependency remains exactly

```text
Teukolsky triple -> eth(F) -> G,Lambda -> eth(G) -> H,B,Pi,C
                  -> eth(C),eth(Pi),ethprime(Bsharp),ethprime(Csharp) -> U.
```

This refactor can share templated point formulas with the evolved graph while
leaving `SpatialPipeline`, `solver_driver`, and the evolved D10-5 operator
unchanged.  It must initially reject `StageConstrained`: substituting
`Q=partial_R Psi` makes the Teukolsky graph second order and a degree-four
input jet is insufficient for four applications.

Nonzero compatible dissipation is a separate hard gate.  D10-5 uses

```text
Q_diss = -(epsilon/h) Htilde^-1 A^T A
```

with an undivided sixth difference, a discrete twelfth-derivative KO-like
operator that is `O(h^5)` at the boundary.  A degree-four Taylor jet cannot
represent it, and omitting it would make the tower differ from the actual RK
stage tangent.  The first controlled implementation must therefore require
`dissipation=0` and remain diagnostic-only.  Before source wiring with nonzero
dissipation, generate and test direct composite actions for the dissipation
and each required radial derivative (never recursively differentiate its
output).  If their `h[4]` endpoint ratio is not greater than `15`, the honest
choices are to keep Route B unavailable with dissipation or qualify a
higher-boundary-order SBP plus compatible-dissipation family.

A higher-order evolved operator is the broader alternative.  The empirical
loss suggests a boundary order of at least seven, but any D14-7-like candidate
needs coefficient provenance, exact SBP identity and polynomial checks, a
negative-semidefinite compatible dissipation, spectral-radius/RK4 timestep
bounds, characteristic and full-graph convergence, long-run stability, and
accelerator parity before adoption.  Merely increasing a stencil width is not
a stability argument.

No local `Z0/Z1` producer may consume either remedy until the exact `h[4]`
endpoint ratio, rotating signed-mode oracle, amplitude linearity, common-stage
generation semantics, and hot-path allocation/fence gates are all green.
