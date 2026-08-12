# D10-5 SBP provenance and qualification

## Why this operator exists

The complete reconstruction/source graph contains paths with two successive
radial first derivatives. D8-4 is fourth order at an endpoint, so the second
application differentiates an existing `O(h^4)` closure error and exposes an
`O(h^3)` endpoint term. The executable full-graph regression measures that
gap. A fifth-order boundary closure leaves an `O(h^4)` result after the same
nested differentiation.

## Coefficient provenance

`include/teuk/sbp105.hpp` implements the diagonal-norm, uniform-grid D10-5
first derivative distributed in the MIT-licensed
[`scicompuu/sbplib`](https://github.com/scicompuu/sbplib) reference library,
file `+sbp/+implementations/d2_10.m`, repository commit
`e64d8c6a740b6d977a2ebcf70f10693c1121a464`. The norm and the complete left
`Q=Htilde D` block are retained as exact rational expressions.
`THIRD_PARTY_NOTICES.md` reproduces the upstream copyright and MIT license
notice that covers these coefficients.

The operator has eleven boundary rows, a maximum 16-point boundary stencil,
and therefore requires at least 22 radial points. Its centered interior
stencil is tenth order and the closure is fifth order. Executable tests verify
the SBP identity, global quintic exactness, interior degree-ten exactness,
fifth-order endpoint convergence, reflected right closure, and device
execution.

This is not claimed to be the rounded minimum-ABTE parameter choice tabulated
by Diener, Dorband, Schnetter, and Tiglio, *Journal of Scientific Computing*
32 (2007), 109-145, DOI 10.1007/s10915-006-9123-7. That paper establishes the
D10-5 family structure and reports, for its penalized periodic-interface
advection test, spectral radius `995.9` and ABTE `20.534` for the
minimum-bandwidth choice versus spectral radius `2.240` and ABTE `0.7661` for
its optimized choice. Its arXiv source archive does not contain the D10-5
coefficient file needed to reproduce the optimized closure at full precision.
The production source therefore names the exact sbplib provenance and makes
no identity claim with that optimized table.

For orientation only, the dimensionless raw first-derivative matrix `D` at
256 points has spectral radius `1.8372113` for this D10-5 closure and
`124.06812` for the repository's classical D8-4 closure. These are raw-`D`
numbers, not the penalized-interface amplification radii reported by Diener et
al.; they must not be compared as if they were the same eigenproblem.

## Compatible dissipation

The D10-5 dispatch uses

`Q_diss = -(epsilon/h) Htilde^-1 A^T A`,

where `A` is the undivided sixth difference
`[1,-6,15,-20,15,-6,1]`. Hence
`u* H Q_diss u = -epsilon ||A u||^2 <= 0` exactly. The operator annihilates
quintics, is `O(h^11)` in the interior, and `O(h^5)` at endpoints, so it does
not reduce the boundary design order. This is the same explicit
negative-semidefinite construction used by the existing D4-2 and D8-4 paths,
extended to the sixth difference.

## Measured complete-graph evidence

The nested spatial regression uses exact in-band angular data, zero
dissipation, and radial grids 49, 97, and 193. These are three-level
self-convergence ratios, not errors against an analytic full-graph solution.
The D8-4 endpoint ratios for the nested source tangent and final forcing remain
between 6 and 10, preserving the known order gap. With D10-5, the measured
ratios are:

| sector | RMS ratio | endpoint ratio |
| --- | ---: | ---: |
| first-order RHS | `37.9407` | `27.2601` |
| reconstruction RHS | `42.1390` | `30.8168` |
| second-order RHS | `37.9341` | `27.2540` |
| final forcing | `38.7559` | `27.2613` |
| inner source | `42.2341` | `31.9291` |
| inner-source tangent | `38.2104` | `27.2613` |
| independent constraints | `42.5644` | `31.0777` |

Every enforced D10-5 endpoint ratio exceeds 15. Values above 16 are reported
only as measurements of these three grid levels; they are not interpreted as
a demonstrated convergence order higher than four.

At fixed D10-5 spatial resolution, common-stage RK4 was independently refined
with 4, 8, and 16 steps against a 256-step trajectory to `T=0.24`. Successive
ratios were `16.4424/16.6569` for the first-order fields,
`16.5912/16.2938` for reconstruction, and `16.2564/15.9335` for the
second-order fields. This is a fixed-space temporal check and is not a
simultaneous spatial/CFL convergence study.

This zero-dissipation full-graph check qualifies derivative composition only.
A trial with `epsilon=0.006` left the forcing difference at a floating-point
floor (`9.15e-17` versus `4.34e-17` in RMS), so it cannot honestly measure
nonzero-dissipation full-graph convergence with this fixture. Dissipation is
instead qualified separately with a resolvable degree-six manufactured field
at `epsilon=0.006`. Its combined derivative-plus-dissipation maximum error has
successive ratios `32.0/32.0`, the endpoint dissipation has `32.0/32.0`, and
one further radial derivative of that endpoint dissipation has `16.0/16.0`.
The latter directly exercises the nested fourth-order boundary budget.

## Explicit dissipation timestep guard

For each scheme the nonzero eigenvalues of `Htilde^-1 A^T A` equal those of
the smaller symmetric positive-semidefinite matrix
`A Htilde^-1 A^T`. Exact finite boundary/interior row patterns give maximum
absolute row sums `64`, `1370.20793`, and `5497.33017` for D4-2, D8-4, and
D10-5. Runtime uses conservative integer spectral-radius bounds `64`, `1371`,
and `5500` and requires

`(dt/h) epsilon bound <= 0.8 * 2.785293563405282`.

Here `2.785...` is RK4's negative-real-axis extent and `0.8` is an explicit
safety factor. Executable tests enumerate every supported extent from 22
through 64; beyond 64 the two closure regions are separated and the same
finite boundary and interior row patterns repeat. This is a sufficient guarded
condition for the pure dissipation subsystem and a necessary production
fail-closed check; it is not a proof of the combined
hyperbolic-plus-dissipative spectrum. At 25 points the measured
dimensionless spectral radius is approximately `5024.36`; for
`epsilon=0.006`, `h=0.0266667`, the unguarded pure-dissipation RK4 limit is
about `0.002464` and the runtime guard limits `dt` to about `0.001801`.

The earlier `T=200`, 400,000-step, near-extremal combination (`dt/h=0.26745`,
`epsilon=0.005`) satisfies the D8-4 guard but fails the D10-5 guard. Runtime
configuration validation now rejects that D10-5 selection before output or
pipeline state is created; a 1,600,000-step resolved configuration satisfies
the guard and is covered by the regression.

## Runtime contract

The strict spelling is `radial_discretization = d10-5`. Existing defaults
remain D4-2 for backward compatibility. Runtime validation rejects fewer than
22 radial points. The typed selection propagates through the primary spatial
pipeline, companion pipeline, initial data, diagnostics, source gate,
checkpoints, replay metadata, and four-Weyl metadata. Existing checkpoint
formats already store the scheme as a strict string, so no format-version bump
is needed; old versioned checkpoints continue to map explicitly to D4-2 and a
scheme mismatch fails before state mutation.

The complete live spin `+2` source composition now requires D10-5. Its nested
source path is not allowed to advertise fourth-order production qualification
under D8-4, even though the lower-level companion evolution remains usable
with any explicitly matching supported radial scheme. The live-composition
constructor itself defaults to D10-5 so its default construction and stage
capability contract are consistent; this does not change the global solver or
configuration default.
