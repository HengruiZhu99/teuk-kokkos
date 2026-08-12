# Spin `+2` concrete outer spatial producer

**Status:** qualified standalone D10-5 stage graph; used by the typed live
composition overload, but not wired into `solver_driver`.

## Contract

`Plus2SourceOuterSpatialProducer` consumes one generation-stamped common-stage
pair sum:

- values `J`, `K`, and `Q`;
- analytic stage tangents `J_T` and `K_T`.

For every explicitly configured signed target `m`, it uses the production
`SignedModeAngularCoordinator` to analyze and synthesize the retained bands
with the physical metadata

| field | spin | boost | radial power |
|---|---:|---:|---:|
| `J` | 2 | 1 | 5 |
| `K` | 1 | 2 | 6 |
| `Q` | 2 | 2 | 4 |

The same projections are applied to `J_T` and `K_T`. Non-target stored modes
are deterministically zeroed. The producer then evaluates

```text
thorn_5 J = thorn_n_point(J, J_T, D_R J; n=5,s=2,b=1,m)
eth_6 K   = eth_n_point(K, K_T, R_1 K; s=1,b=2)
```

where `D_R` is the endpoint-inclusive D10-5 SBP derivative and `R_1` is the
exact spin-raising action supplied by the angular plan. The point formulas add
the full rotating-Kerr time, azimuthal, and GHP connection terms. Projected
`Q` is retained algebraically; no unrequested `Q_T` is manufactured.

The output consists of projected `J,K,Q`, `thorn_5 J`, and `eth_6 K`, with a
generation stamp for every stored mode and collocation point. Input/output
shape and alias violations (including internal scratch and overlapping stamp
outputs) and a zero or mismatched generation throw before a launch. Any
nonfinite stage input clears the complete invocation to zero with
zero stamps; any nonfinite result clears its point. Repeated evaluation
performs no allocation, deep copy, or fence.

## Qualification

The rotating-Kerr test uses `a/M=0.73`, the exact compactified horizon as its
upper endpoint, signed modes `m=-2..2`, and the target subset `{-2,0,2}`. It
injects independent (not conjugated) positive and negative signed data,
retained harmonics, and an exactly excluded
`ell=ell_max+1` component and compares projected values and both GHP operators
with independent host harmonic and point-formula oracles. Host/device angular
orthogonality and exact Gaunt product projection are separately qualified by
the angular suites already used by this coordinator.

The outer graph consumes post-pair target sums, so it performs no sharp lookup
or conjugation of its own; sharp ownership remains in the inner pair graph.

The D10-5 test includes scri and the horizon at `N_R=25,49,97`; its measured
maximum errors are `8.93644e-11, 1.51064e-12, 2.45051e-14`, with ratios
`59.1567, 61.6458`. Further
tests cover target filtering, generation mismatch, alias rejection, global
nonfinite clearing, amplitude and independent tangent scaling, and a warmed
allocation/copy/fence audit. The typed one-step Bianchi binding test now uses
this producer rather than its former outer callback.

The separate complete-graph study in
`PLUS2_FULL_SOURCE_ANGULAR_QUALIFICATION.md` carries non-bandlimited rotating
data through the primitive and ordered-pair graph into this producer. It
separately refines retained `ell_max` and Gauss-Legendre node count, checks
`eth_6 K` and final forcing in modal all-radius and endpoint norms, and reaches
the `ell_max=12`, `N_theta=36` reference monotonically. No angular filter is
present in either test.

## Limitations

This is a stage-local spatial operator, not a physical runtime qualification.
It does not supply Bianchi initial data or horizon/scri boundary data, prove
peeling coefficients, select production angular bandlimits, activate a source,
advance a companion state, or emit waveforms. Those remain separate gates.
The producer has one global retained `ell_max` for all configured target modes;
mode-dependent bandlimits would require a new explicit interface and evidence.
