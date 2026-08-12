# Route-B direct reconstruction radial jets

## Scope

`RouteBReconstructionJetTower` is a source-independent diagnostic radial
slice for the seven spin -2 metric-reconstruction fields
`G,Lambda,H,B,Pi,C,U`. It constructs normalized h0 radial Taylor coefficients
once with the direct nine-point D1 through D4 operators, then consumes one
coefficient degree per level through h4. Kerr `mu0`, `tau0`, and `pi0` are
formed directly with radial-jet product and quotient algebra.

This slice is `FreeDamped` and zero-dissipation only. It is not an SBP
evolution operator and has no `SpatialPipeline`, runtime, solver, Z0/Z1, T0,
source, or spin +2 wiring.

## Enforced triangular handshake

Every level is split into the production dependency order:

1. pass 1 consumes the current reconstruction jets, the primary F jet, and a
   prescribed full-GHP `eth1(F,F_T)` jet; it publishes next G and Lambda;
2. pass 2 requires an `eth2(G,G_T)` jet stamped from that exact pass-1 output,
   then publishes next H, B, Pi, and C;
3. pass 3 requires `eth2(C,C_T)`, `eth2(Pi,Pi_T)`,
   `ethprime1(Bsharp,Bsharp_T)`, and `ethprime2(Csharp,Csharp_T)` jets stamped
   for that exact pass. Sharp values are `conj(X[-m])`. It publishes U and
   finalizes the next level.

Generation, level, and pass are bound by distinct nonzero pass tokens for the
reconstruction angular handshake. Wrong-phase or wrong-token reconstruction
angular buffers fail closed. The primary F/Psi4 coefficients carry only the
overall generation stamp, so selecting the matching primary level remains an
external coordinator contract; this slice cannot detect a wrong-level primary
buffer from the same generation. Initialization, external coefficient-wise
angular/projection launches, and all three passes must use the same ordered
execution-space instance. The API is asynchronous, does not fence, and cannot
establish dependencies across backend queues.

The retained next fields are pointwise radial results. Production projects
G/Lambda after pass 1, H/B/Pi/C after pass 2, and U after pass 3. This slice
does not implement that projection write-back handshake. It therefore accepts
prescribed full-GHP angular jets and is not a closed rotating collocation graph;
its internal next level must not be described as production-equivalent data.

## Qualification

An independent 100-digit mpmath generator uses immutable per-mode closure
factories, analytic exponential/trigonometric h0 and angular profiles, an
independently coded Kerr background, nested ordinary differentiation, and the
seven reconstruction equations. It asserts that the two signed modes differ
at h1 before exporting the fixture. At both compactified endpoints, every
non-exact field through h4 has both N=9/17 and N=17/33 absolute-error ratios
above 15. The limiting h4 field is G, with ratios 64.4599 and 49.2187; Pi has
ratios 110.667 and 98.0036. The h4 signal is finite and nonzero.

N=17/33/65 is a non-gating binary64 ceiling probe, and N=65 is already red:

| field | e17 | e33 | e65 | e17/e33 | e33/e65 |
| --- | ---: | ---: | ---: | ---: | ---: |
| G | 7.45403e-10 | 1.51447e-11 | 1.29105e-10 | 49.2187 | 0.117305 |
| Lambda | 1.78955e-9 | 1.95578e-11 | 1.77213e-10 | 91.5004 | 0.110364 |
| H | 3.26002e-9 | 4.01631e-11 | 1.7975e-10 | 81.1694 | 0.223439 |
| B | 5.78963e-9 | 6.29e-11 | 2.22418e-10 | 92.045 | 0.2828 |
| Pi | 7.66389e-9 | 7.82001e-11 | 4.17722e-10 | 98.0036 | 0.187206 |
| C | 1.11055e-8 | 6.46375e-11 | 1.68183e-10 | 171.812 | 0.384328 |
| U | 1.30516e-8 | 3.5478e-11 | 6.69812e-10 | 367.878 | 0.0529671 |

The N=65 errors increase and every second ratio is below one because direct
h^-4 initialization amplifies binary64 roundoff. Qualification is therefore
frozen to the pre-roundoff N=9/17/33 window. Production-sized grids need an
explicit precision and error-budget strategy rather than extrapolation from
this diagnostic qualification.

Additional tests cover exact h1 parity with the production point equations,
a 100-digit background coefficient oracle through radial degree four,
amplitude linearity, signed sharp coupling, pass-token provenance, global
stale/nonfinite poisoning, null/shape/stride/alias rejection, LayoutRight and
padded LayoutStride parity, rejection of StageConstrained/nonzero dissipation,
and allocation-free/fence-free hot pass sequencing. A regression shows why an
angular derivative recovered from sampled values by Fornberg differentiation
cannot replace the coefficient-wise angular jet.

## Follow-on contract

A closed Route-B reconstruction graph must add componentwise full-GHP angular
jet evaluation and projection write-back at every pass, with the same pass
tokens and signed-mode registry. Only after that gate is independently green
may reconstruction h0 through h4 feed a local curvature or quadratic source
producer.
