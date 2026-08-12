# Fourth-order qualification contract

This document defines what the repository may call “fourth order overall.” It
does not infer that property from the name of a stencil or integrator. The
executable gates are in `tests/test_fourth_order_qualification.cpp`.

## Radial spatial gate

The production full spin `+2` Teukolsky RHS is sampled on 25, 49, and 97 point
grids, so the radial spacing is halved exactly. The manufactured state has
non-polynomial complex `P`, `Q`, and `Psi` profiles. `Q=partial_R Psi` and `P`
is constructed from an independently prescribed analytic `partial_T Psi`, so
the exact `partial_R Q` and `partial_R partial_T Psi` are available without
copying the production radial operator. The angular action and forcing are
both nonzero. The error is a pointwise maximum over all fields and every radial
point, including scri and the horizon-side endpoint.

The measured errors without dissipation were

| radial points | maximum error |
| ---: | ---: |
| 25 | `3.22569e-6` |
| 49 | `2.01132e-7` |
| 97 | `1.25480e-8` |

The successive ratios were `16.0377` and `16.0290`. The enforced lower bound
is 13, not merely monotone decrease.

## Compatible-dissipation gate

The same endpoint-inclusive manufactured test is repeated with D8-4's
compatible fifth-difference `A^T A` dissipation in the RHS at strength 0.006.
The measured errors were `4.03464e-6`, `2.59559e-7`, and `1.64530e-8`, giving
ratios `15.5442` and `15.7758`. This directly checks that the selected
dissipation does not lower the observed global order. It does not qualify an
arbitrary strength or a post-step filter.

## Common-stage temporal gate

The temporal test suppresses spatial truncation with a radially constant exact
`Psi` and `Q=0`, but it still traverses the D8-4 full companion RHS. The
primary perturbation is numerically evolved, and the source adapter reads that
actual primary RK-stage value. The angular adapter similarly reads the actual
companion stage. Thus the test is a genuinely one-way, stage-coupled passive
evolution rather than a prescribed function of exact time.

For 16, 32, and 64 steps, the primary errors were `2.99222e-9`,
`1.89217e-10`, and `1.18953e-11`, with ratios `15.8137` and `15.9069`. The
companion errors were `9.47522e-7`, `5.99179e-8`, and `3.76693e-9`, with
ratios `15.8137` and `15.9063`. Every enforced ratio must exceed 14.

## Angular and source order budget

The angular implementation is modal rather than a fixed-order finite
difference. For a field inside the retained spin-weighted harmonic band,
analysis, diagonal angular differentiation, and synthesis are exact to
roundoff on a sufficient Gauss-Legendre grid. Quadratic products use the
polynomial-exact padded rule and are independently checked against the exact
Gaunt convolution; the deliberately unpadded regression demonstrates that
the oracle detects aliasing.

Those facts do not make a fixed angular truncation disappear under radial
refinement. Multiplication by rational Kerr coefficients can generate
off-band content, and discarding physical source daughters above
`ell_max_second` is a truncation error independent of radial spacing. Either
effect can flatten a radial convergence plot even when D8-4 is operating at
fourth order. A full run may therefore claim joint fourth-order convergence
only when one of the following is demonstrated:

1. the manufactured/exact solution is wholly inside the retained bands and
   the padded product condition is satisfied; or
2. `ell_max` and `N_theta` are independently refined until angular and source
   truncation errors are below the measured radial `O(h^4)` error at every
   resolution used in the fit.

Filtering is not part of this gate. If a later angular filter changes an
accepted or stage state, every dependent derivative, curvature, source, and
RHS must be recomputed from that changed state. Reusing a pre-filter stage RHS
would invalidate both common-stage RK4 and this qualification.

## Scope limitation

These tests qualify the D8-4 full spin `+2` RHS, its compatible dissipation,
and the one-way common-stage RK4 mechanism. They do not by themselves qualify
the complete four-field production run. That broader claim additionally
requires the reconstruction derivatives, both second-order source paths,
runtime selection/checkpoint metadata, angular/source band studies, and CPU to
GPU parity to traverse the same D8-4 selection.
