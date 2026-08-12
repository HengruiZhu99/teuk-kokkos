# Route-B local curvature provider: exact remaining data blocker

**Status:** the linear `Psi1` authority is independently closed, but the
standalone six-field local provider is withheld because the approved closed
`h0..h4` graph does not publish the one radial derivative of `h4` required at
rotating scri. No partial adapter is populated and no runtime, source,
`SpatialPipeline`, or Route-A wiring is added.

**Base audited:** `40f14a3158fbc82c5699b0eaae97256b6ff0c6c4`.

## Independent `Psi1` authority

The production compact expression follows by solving ordinary-NP `riem-5`
in the repository's `+---`, `gamma=gamma1=0` ORG convention:

```text
D beta1-delta epsilon1
 = sigma1(alpha+pi)+beta1(rhobar-epsilonbar)-kappa1 mu
   +epsilon1(alphabar-pibar)+Psi1.
```

Consequently

```text
Psi1 = D beta1 + D1 beta0-rhobar beta1-rhobar1 beta0
       +epsilonbar beta1+epsilonbar1 beta0
       -delta epsilon1-delta1 epsilon0
       +alphabar epsilon1+alphabar1 epsilon0
       -pibar epsilon1-pibar1 epsilon0
       -(alpha+pi)sigma1+mu kappa1.
```

After inserting the repository's compact fields this is exactly the existing
`psi1_leading/R + psi1_order4` algebra in
`plus2_source_primitives.hpp`, including the three reviewed stationary Kerr
background derivatives. The separately printed `Psi1` representation in
arXiv:2008.11770 reverses the `alphabar/pi` epsilon-bracket signs relative to
its own Ricci identity; it is not an authority and must remain an explicit
negative regression.

`verify_linear_psi0_kerr_ad_oracle.py` now supplies the independent route. It
constructs the coordinate background metric from the code tetrad, reconstructs
a real ORG coordinate perturbation, varies Christoffels, Riemann, Ricci, and
the full Weyl trace subtraction, and evaluates

```text
Psi1^(1)=-delta[C_abcd l^a n^b l^c m^d].
```

It includes the repository tetrad perturbations

```text
l1=-1/2 h_ll n,  n1=0,  m1=-h_lm n+1/2 h_mm mbar.
```

The fixed-background-tetrad or Riemann-only contractions are deliberately not
used: background `Psi2` makes the `m1` contribution nonzero, and the `Psi1`
trace terms do not vanish algebraically. The oracle independently verifies
vacuum Ricci, zero background `Psi0/Psi1`, and
`C(l,n,l,n)=-2 Re(Psi2)` before comparing the linear result. Smooth ORG jets
at `a=0.63,0.91,0.999,-0.74` agree with the solved Ricci identity at errors
from `1.4e-17` to `1.7e-16`; the inconsistent printed signs differ by
`4.46e-2` to `1.08e-1`. The pre-existing `T0` comparisons remain at their
prior binary64 errors.

This closes the point algebra, not the radial endpoint provider.

## Exact radial-coefficient ledger

The approved graph stores normalized radial Taylor degree

```text
d(h_k)=4-k.
```

At scri, for a compact field coefficient `[q]`, the stationary operators use

```text
Delta X[q] : h_(k+1)[q], h_k[q-1], and lower coefficients
thorn X[q] : h_k[q], h_(k+1)[q-1]
eth X[q]   : h_k[q], h_(k+1)[q]
```

where rotating `eth` contains the nonzero time coefficient
`-i a sin(theta)/(sqrt(2)L^2)`. Applying this ledger through the triangular
connection construction gives

```text
f0[q] needs h_k[q], h_(k+1)[q-1], h_(k+2)[q-1],
f1[q] needs h_k[q], h_(k+1)[q],   h_(k+2)[q].
```

The removable limits are

```text
Z0_source = [R^2] f0 -(pibar0-tau0) Kap,
Z1        = [R^1] f1 +z1_regular.
```

Thus `(h0,h1,h2)` and `(h1,h2,h3)` contain every coefficient required for
value and first tangent. For `(h2,h3,h4)`, both limits need `h4[1]`, while the
closed graph publishes only the value `h4[0]`. A simple nonvanishing channel
is obtained with only `V=U/mu0`: the rotating time part of `eth` places
`d_R V4|scri` in both `f0[2]` and `f1[1]`. This is a genuine missing datum,
not a removable zero. It vanishes at `a=0` and therefore cannot be qualified
with a Schwarzschild-only test.

Interior and horizon points are not blocked: `R>0` permits direct
`f0/R^2` and `f1/R`. Publishing those four-and-two partial fields while
leaving the two second-tangent scri values invalid would violate the typed
six-field adapter contract, so the provider fails closed globally instead.

## Narrow remedy contract

The least invasive remedy is not a degree-six tower and not a second radial
derivative. After the closed graph projects `h4`, apply the already reviewed
nine-point direct Fornberg `D1` once to every projected level-four field and
then apply the pure angular action coefficient-wise to that derivative. This
supplies exactly `h4[1]`.

The conservative endpoint order budget is fourth order. `h4` inherits the
one-sided direct-`D4` input error at `O(dx^5)`; one `D1` can amplify it to
`O(dx^4)`. The nine-point direct-`D1` stencil's own smooth-function error is
`O(dx^8)`. No `D2(h4)` is needed; it would risk reducing the propagated error
to third order. Differentiating the complete sampled `f0/f1` profiles is a
broader and less well-conditioned alternative and must not substitute for the
coefficient ledger without its own independent gate.

The narrow seam was implemented experimentally against an extended form of
the independent 90-digit closed-graph fixture and then discarded. It compiled
and preserved generation/projection provenance. Before that discard, a fixed,
non-tuned asymptotic-window audit recorded the absolute error of every one of
the 60 disaggregated
`(field,mode,scri/interior/horizon)` cells at
`N=9,13,17,25,33,49,65`.

Those prototype source and generated-fixture changes are deliberately absent
from this candidate, so the following numbers are a discarded-prototype
observation, not reproducible evidence shipped by this commit. They motivate
the fail-closed disposition but are not needed for it: the analytic ledger
above independently proves that the approved graph lacks a generically
nonzero datum.

The frozen `N=9,17,33` window failed two scri cells:

| field, signed mode | N=9 | N=17 | N=33 | ratios |
|---|---:|---:|---:|---:|
| `B, m=+2` | `1.10572e-4` | `8.65881e-6` | `2.11739e-7` | `12.7698, 40.8938` |
| `U, m=+2` | `4.57200e-4` | `3.08309e-5` | `6.87969e-7` | `14.8293, 44.8144` |

Both first ratios are below 15. These are not roundoff-floor failures; their
second ratios show that the expected asymptotic regime begins after `N=9`.
The independently predeclared alternate `N=13,25,49` window is also not a
global remedy. It repairs those two cells, but `Pi,m=+2,scri` has errors
`4.94348e-6,4.19097e-7,8.06306e-9`, whose first ratio is only `11.7956`.
Moreover, many horizon cells encounter binary64 amplification by `N=49`, so
their second alternate ratio is below 15; for example `P,m=-2,horizon` gives
`4.42899`, while `Q,m=+2,horizon` gives `0.710481`.

There is therefore no single globally green adjacent-doubling triple among
the two windows selected before inspection. Selecting a per-cell window,
aggregating errors, or accepting only the formal order argument would weaken
the same gate that qualified the closed tower. The red seam and its generated
fixture were therefore not retained.

Before the local provider can be committed, a revised extension must prove all of
the following on `N=9,17,33` (with `N=65` retained as the documented red
roundoff probe):

1. endpoint-inclusive `D1(h4)` errors and both `Z0TT/Z1TT` scri errors have
   two finite ratios greater than 15 for rotating `a=0.63,0.999` and a
   negative spin, separately for both signed modes; in particular it must
   repair the recorded `B,m=+2,scri` failure without changing the frozen
   `N=9,17,33` promotion window;
2. all background factors entering `f0[2]` and `f1[1]` are analytic radial
   Taylor jets, including `rho0,epsilon0,alpha0,beta0,tau0,pi0` and the three
   merged directional-derivative slots; sampled-background differentiation is
   forbidden;
3. invalid/stale `h4` rows and either sharp partner produce zero output stamps
   and can never be resurrected by angular analysis or projection;
4. the hot provider and the new `D1(h4)` seam allocate nothing and fence
   nowhere on the same ordered execution-space instance;
5. the independent full-coordinate-Weyl oracle supplies expected `Psi1`, and
   the existing independent coordinate `T0` fixtures are reused without
   deriving expected values through the compact primitive path.

The final adapter must distinguish the three convention-fixed quantities:

```text
Psi0_raw = R^5 Zplus/(L^2-i a R cos(theta))^4,
Zplus    = (L^2-i a R cos(theta))^4 Z0_source,
Z1       = Psi1_raw/R^4.
```

`Plus2TransportedCurvatureComponent::Z0` is the source-normalized
`Z0_source`, not `Zplus`. The scri certificate must bind all six source fields
and their cancellation residuals. Until the missing `h4[1]` seam passes the
two-ratio gate, no typed curvature adapter or certificate is scientifically
valid.
