# Standalone four-Weyl-field output contract

Status: output and metadata seam qualified in Kokkos Serial; deliberately not
wired to `SpatialPipeline`

## Scope

`include/teuk/four_weyl_output.hpp` provides a standalone output packer for
the four requested fields at one common accepted time:

| name | input numerical variable | physical raw code-tetrad value |
|---|---|---|
| `psi4_order1` | `F^(1)` | `Psi4^(1)=R F^(1)` |
| `psi0_order1` | `Z_plus^(1)` | `Psi0^(1)=R^5 Z_plus^(1)/(L^2-i a R cos(theta))^4` |
| `psi4_order2` | `F^(2)` | `Psi4^(2)=R F^(2)` |
| `psi0_order2` | `Z_plus^(2)` | `Psi0^(2)=R^5 Z_plus^(2)/(L^2-i a R cos(theta))^4` |

The point-input form is azimuthally modal in explicitly sorted signed `m` and
nodal in compact radius and polar angle.  It accepts the union of the parent
and target registries, then emits only parent modes for first-order fields and
only target modes for second-order fields.  The second form accepts complete,
ordered spin-weighted spherical modal coefficients at scri.  A plus-field
spherical modal value is not reconstructed at finite Kerr radius because its
theta-dependent tetrad factor is not diagonal in `ell`; callers must use the
nodal seam there rather than silently applying a scalar modal factor.

This layer has no dependency on `SpatialPipeline`, the runtime parser, the
linear-curvature operator, or the quadratic source.  It therefore cannot make
an unqualified production formula reachable merely by enabling output.

## Boundary semantics

At scri the raw code-tetrad fields vanish.  The output does not divide zero by
zero.  It stores the exact leading physical coefficients

```text
lim_(R->0) Psi4/R   = F|scri,
lim_(R->0) Psi0/R^5 = Z_plus|scri/L^8,
```

together with radial powers one and five.  Interior and horizon rows store the
exact raw multiplication by `W_minus=R` or
`W_plus=R^5/(L^2-i a R cos(theta))^4`.  Horizon metadata states verbatim that
the raw code-tetrad value is convention fixed and is not a flux observable.
No output row labels a raw second-order coefficient gauge invariant.

## Metadata and provenance

The versioned metadata records:

- Git commit and runtime-configuration schema version;
- outgoing-radiation gauge and the horizon-regular rotated-Kinnersley code
  tetrad identifiers;
- the no-factorial perturbative convention;
- both exact radial scaling formulas;
- typed linear-`Psi0`, quadratic-source, and companion-initial-data methods;
- independent first/second `ell` bandlimits and sharp-closed parent/target
  signed-mode registries;
- output cadence and the exact sharp rule
  `X_m^sharp=conjugate(X_-m)`;
- explicit scri and horizon interpretation strings.

Both metadata and the 21-column CSV have strict readers.  They reject unknown
schema versions, missing or reordered metadata keys, malformed values,
nonfinite numbers, inconsistent field spin/order labels, inconsistent scri
power/raw-value semantics, and trailing metadata.  Decimal complex components
use 17 significant digits and round-trip exactly as binary64 values in the
Serial tests.

## Executable validation

`tests/test_four_weyl_output.cpp` covers:

1. disabled construction with zero Kokkos allocation and zero kernel launch;
2. exact metadata serialization, parsing, and canonical reserialization;
3. deterministic field-major and signed-`m` ordering;
4. exact `W_minus` and `W_plus` multiplication at interior points and the
   horizon;
5. explicit scri limiting coefficients, with no inverse scaling;
6. independent host evaluation against the Kokkos packing kernel;
7. common timestamps and output-cadence decisions without input mutation;
8. linear first-order and quadratic second-order amplitude scaling;
9. complete ordered spherical-modal scri input and fail-closed missing rows;
10. fail-closed nonfinite values and mismatched extents;
11. exact CSV round trip and opposite-signed-mode sharp lookup.

The recorded Serial run at the implementation commit passed all `206/206`
unit cases.  This is compile-and-runtime evidence for the Kokkos Serial test
binary only.  No GPU qualification is claimed by this document.

## Integration boundary

Promotion into the solver still requires the coordinator to provide all four
regular fields from one accepted stage/step, verify that the replay checkpoint
and resolved configuration carry compatible band/method provenance, and
exercise concurrent/replay equality with the actual curvature and source
operators.  The packer accepts one time for all four fields, but by design it
cannot prove that independently supplied views came from the same RK stage.
That proof belongs at the future production call site.
