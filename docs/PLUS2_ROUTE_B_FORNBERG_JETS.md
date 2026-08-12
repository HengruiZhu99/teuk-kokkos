# Route-B direct radial derivative primitives

Status: foundational diagnostic component only

## Scope

This component provides the radial ingredients proposed in
`PLUS2_ROUTE_B_DERIVATIVE_TOWER_BLOCKER.md` without changing the evolved
spatial graph:

- independently generated direct uniform-grid `D1`, `D2`, `D3`, and `D4`;
- one intact nine-point window, centered where possible and shifted fully to
  either endpoint;
- a device-callable local radial Taylor-jet algebra using normalized
  coefficients `f[k] = partial_R^k f / k!`;
- allocation-free, fence-free evaluation into caller-owned rank-five storage.

The generated table is pinned by the SHA-256 digest of its canonical exact
rational representation.  The offline `Fraction` generator solves each
derivative and each of the nine evaluation rows independently.  The
arbitrary-precision audit checks all monomial moments through degree eight,
and endpoint-inclusive exponential and trigonometric convergence.  The C++
tests check the emitted binary64 table, maximum dimensionless coefficient
norms, active-device `LayoutRight`/padded-`LayoutStride` parity, extent and
overlap rejection, and hot-launch allocation/fence behavior.

The maximum dimensionless row norms expose the expected endpoint noise cost:

| derivative | maximum L1 norm | maximum L2 norm |
|---|---:|---:|
| D1 | 78.0190476190 | 32.7203410300 |
| D2 | 357.587301587 | 154.320425366 |
| D3 | 955.733333333 | 418.116481851 |
| D4 | 1740.8 | 767.217558305 |

These large fully one-sided norms are recorded qualification evidence, not a
stability claim.  Any later Route-B graph must audit roundoff amplification
at its production resolution and amplitude.

## Taylor-jet contract

`RouteBRadialTaylorJet<Degree, Scalar>` is fixed-size value storage suitable
for host and device code.  It supports addition, subtraction, convolution
products, reciprocal/quotient triangular solves, scalar products and
quotients, truncation, and one radial derivative that consumes one degree.
The quotient operation assumes a nonzero zeroth denominator coefficient; a
future graph owns the physical denominator validation before device launch.

The direct derivative view evaluator accepts

```text
input  (mode, field, radial, theta)
output (mode, derivative-1, field, radial, theta)
```

with identical scalar types, nonzero semantic extents, at least nine radial
points, positive finite uniform spacing and inverse spacing, exact output
extents, writable output, and nonoverlapping allocations. Packed or padded
lexicographic Kokkos strides in any dimension order are honored. Stride maps
that do not satisfy the conservative nonoverlap proof are rejected, including
internally aliased `LayoutStride` mappings that would race parallel writes.

## Explicit limitations

This is not an SBP operator and is not used by time evolution.  It supplies
no norm, energy estimate, compatible dissipation, characteristic boundary
treatment, or RK4 timestep bound.  It must not replace D10-5 or D8-4 in
`SpatialPipeline`.

The intended first consumer is only the documented zero-dissipation,
`FreeDamped` Route-B diagnostic.  No `SpatialPipeline` refactor, `L^4` tower,
local `Z0/Z1` producer, runtime option, or solver-driver wiring is included.
`StageConstrained` and nonzero dissipation remain hard gates: the former
changes the radial derivative degree requirement, while the latter cannot be
represented by a degree-four local Taylor jet.  The existing endpoint,
signed-mode, stage-provenance, amplitude-linearity, and no-allocation/fence
gates still apply before any curvature result may consume these primitives.

## Reproduction

```sh
python3 tools/numerical/generate_routeb_fornberg_weights.py \
  --check include/teuk/routeb_fornberg_weights.hpp
python3 tools/numerical/verify_routeb_fornberg.py
```

The Python audit requires `mpmath`.  Both checks are registered with CTest
when `TEUK_ENABLE_SYMBOLIC_AUDIT=ON`.
