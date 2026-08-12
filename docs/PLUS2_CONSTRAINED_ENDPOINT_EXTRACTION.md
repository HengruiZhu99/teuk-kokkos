# Constrained spin +2 scri coefficient extraction

## Status

The endpoint extraction operator is qualified as a standalone, fail-closed
component. It does **not** complete the rotating Route-B local-curvature
provider: the live graph that constructs all six numerator profiles from the
same `h0..h4` stage remains absent, and the complete `N=65` Route-B tower is a
red binary64-conditioning probe.

## Independent derivation

For uniform strictly positive nodes `R_j=j h`, write

```text
f0(R)=a0+a1 R+q0 R^2+a3 R^3+...
f1(R)=b0+q1 R+b2 R^2+...
```

The six-node `q0` weights solve, in exact rational arithmetic,

```text
sum_j w_j j^k = 1  for k=2,
sum_j w_j j^k = 0  for k=0,1,3,4,5.
```

They are `29/6, -461/24, 31, -307/12, 65/6, -15/8`. The five-node `q1`
weights have unit first moment and zero moments 0,2,3,4 and are
`-77/12, 107/6, -39/2, 61/6, -25/12`.

Thus `q0=h^-2 sum w_j f0(jh)` and `q1=h^-1 sum w_j f1(jh)` have fourth-order
truncation error. The stencils never read `R=0`. Algebraic annihilation of the
constant/linear bases is not evidence of peeling, so `f0(0)`, a separately
differentiated `f0'(0)`, and `f1(0)` remain explicit convergence diagnostics.

Exact condition summaries are:

| extractor | L1 | L2 squared | Linf |
|---|---:|---:|---:|
| q0 | 280/3 | 613067/288 | 31 |
| q1 | 56 | 60995/72 | 39/2 |

The exact generator is
`tools/numerical/generate_plus2_endpoint_extraction.py`; the independent
100-digit audit is `verify_plus2_endpoint_extraction.py`. C++ tests cover
Schwarzschild, positive/negative moderate spin, near-extremal spin, both signed
modes, `N=9,17,33`, independent peeling residual convergence, and device
parity. `N=65` is recorded but cannot promote the full Route-B provider.

## Historical erratum

The earlier `D105NestedLhopitalV1` initializer differentiated `f0` twice. It
was appropriate only as manufactured algorithm evidence and did not resolve
the missing rotating `h4[1]` datum. The active initializer now identifies the
operator as `ConstrainedPositiveNodesV2`; the old enum value remains only to
make the historical distinction explicit and is rejected by the active
contract.
