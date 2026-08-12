# Complete spin `+2` source angular qualification

**Status:** qualified for the manufactured rotating-Kerr graph described
below. This is an angular discretization gate, not a runtime or physical-data
qualification.

## Graph under test

`tests/test_plus2_full_source_angular_convergence.cpp` executes the complete
concrete stage-local source chain:

```text
h[0], h[1], h[2]
  -> Plus2SourcePrimitiveSpatialProducer (D10-5)
  -> ordered-pair C/B/D/Er/Et/J/K/Q families
  -> target J/K/Q sums
  -> Plus2SourceOuterSpatialProducer projection
  -> thorn_5 J and eth_6 K
  -> compact source and coordinate forcing
```

The case uses `M=1`, `a/M=0.73`, `L=1.2`, and 49 radial points from scri to
the exact compactified horizon. The same D10-5 radial grid is used at every
angular resolution, so this study changes no radial operator or boundary
closure. Its purpose is to isolate angular error after the separately
qualified endpoint-inclusive radial tests.

Stored signed modes are `m=-4..4`. First-order parent modes are `m=-2..2`,
while only the even daughters `{-4,-2,0,2,4}` are requested as source targets.
The positive- and negative-`m` amplitudes are deliberately independent. Every
metric, curvature, and Bianchi input is a decaying modal series through
`ell=12`, with the spin appropriate to that field. This exercises the
primitive angular derivatives, sharp lookup, noncommuting ordered pairs,
quadratic target sums, retained target projection, and the outer `eth_6 K`.

## Independent comparison

The comparison does not read device modal scratch. A host oracle explicitly
computes

```text
2 pi sum_i w_i conjugate({}_sY_lm(theta_i)) f_m(theta_i)
```

at every radial point using all Gauss-Legendre nodes. The theta amplitudes in
this real convention equal their conjugates. The reported RMS contains every
retained comparison coefficient and all 49 radial points. The maximum norm
and a separate maximum over both radial endpoints are also gated.

The high-band/high-node reference is `ell_max=12`, `N_theta=36`. It is an
independent-resolution reference, not an exact physical Kerr solution. The
test therefore reports convergence to this reference and does not claim a
closed-form source value.

## Galerkin truncation

To isolate retained-band error, `N_theta=36` is held fixed while `ell_max` is
raised through `4,6,8,10`. Low output coefficients through `ell=4` are compared
with the `ell_max=12` reference.

| quantity | `ell=4` RMS | `ell=6` RMS | `ell=8` RMS | `ell=10` RMS |
| --- | ---: | ---: | ---: | ---: |
| coordinate forcing | `4.40805e-2` | `4.28692e-3` | `1.86890e-4` | `8.34713e-6` |
| `eth_6 K` | `2.81123e-2` | `3.15659e-3` | `1.55773e-4` | `7.15305e-6` |
| projected `K` | `1.30603e-2` | `1.58463e-3` | `8.80282e-5` | `4.16371e-6` |
| primitive `kappa` | `1.41709e-2` | `6.42336e-5` | `4.02806e-7` | `2.89962e-9` |

The forcing maximum errors are `2.85575e-1, 3.58778e-2, 1.48362e-3,
5.28910e-5`. The endpoint maxima are identical for this manufactured case.
The executable gate requires strict decrease at every refinement, more than a
factor 20 improvement from `ell_max=4` to 10 for the forcing, projected `K`,
`eth_6 K`, and primitive `kappa`, and final RMS/maximum bounds consistent with
the measured margin. This is spectral/modal convergence, not a claim of
fourth-order angular finite differencing.

## Quadrature exactness and filtering

To distinguish quadrature from Galerkin truncation, `ell_max=10` is fixed and
only the Gauss-Legendre node count is raised through `12,18,24`, against the
same-band `N_theta=36` result. Coordinate-forcing RMS errors are

```text
1.64047e-6, 4.20101e-13, 1.09348e-15.
```

The corresponding maxima, including the endpoints, are `1.50468e-5,
5.97221e-12, 7.99438e-15`. The gate requires strict decrease, a factor greater
than 20 from 12 to 24 nodes, and an all-point/endpoint final error below
`2e-14`.

There is no angular filter in this graph. `ell_max` projection is the explicit
Galerkin truncation being refined; it must not be described as dissipation or
filter qualification. A future production filter would change this operator
and require this study to be repeated with that filter enabled.

## Closure and scaling checks

At `ell_max=10`, `N_theta=30`, all eight ordered-pair source families are
nonzero. At every angular node and every radial point, including scri and the
horizon, explicit host sums of pair-family `J`, `K`, and `Q` agree with the
stored target aggregates to `2e-12`. Reversed ordered pairs for target `m=0`
are demonstrably unequal, so the test cannot pass by treating the pair ledger
as unordered.

All non-target stored modes are zero after pair summation, outer projection,
`eth_6 K`, and final forcing to `2e-13`. Rescaling every first-order metric,
curvature, and derivative input by `-1.7` rescales the final modal forcing by
`(-1.7)^2` with relative error below `2e-11`.

## Limits

This evidence does not select a production `ell_max`, because physical
waveforms may have a different angular spectrum and a longer nonlinear tail.
It does not qualify Route-A Bianchi transport, Route-B curvature construction,
GPU execution, source activation, time evolution, checkpoint/replay, or
waveform output. A production campaign still needs an observable-based
`ell_max`/`N_theta` refinement using its actual initial data and any eventual
angular filter.
