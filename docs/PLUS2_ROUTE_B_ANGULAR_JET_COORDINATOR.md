# Route-B angular jet coordinator

## Scope

`RouteBAngularJetCoordinator` is a standalone, source-independent diagnostic
graph for the zero-dissipation `FreeDamped` Route-B design. It closes the
approved primary fields `P,Q,Psi` and reconstruction fields
`G,Lambda,H,B,Pi,C,U` from h0 through h4 for a sharp-closed signed-mode
registry. It does not construct T0, Z0, Z1, nonlinear sources, or an RK/live
evolution, and it is not wired into `SpatialPipeline`, the solver, or runtime.

The exact reconstruction metadata used by the graph is:

| Field | spin | boost | radial falloff |
|---|---:|---:|---:|
| G | -1 | -1 | 2 |
| Lambda | -2 | -1 | 1 |
| H | 0 | 0 | 3 |
| B | -2 | 0 | 1 |
| Pi | -1 | 0 | 2 |
| C | -1 | +1 | 2 |
| U | 0 | +1 | 3 |

Primary `P,Q,Psi` all have spin -2. The coordinator owns its towers,
coordinates, angular plans, projection buffers, and pass workspaces. After its
construction-time plan setup, `initialize`, `advance_one_level`, and
`advance_to_h4` allocate nothing and fence nowhere. All construction and graph
calls for a generation must use the same ordered execution-space instance;
cross-queue dependencies are not supported.

## Exact level ordering

At h0, each normalized radial Taylor coefficient through degree four is
Galerkin projected and written back. At level k, with active output degree
`3-k`, the coordinator then performs:

1. apply the Teukolsky angular Laplacian coefficient-wise to Psi, advance the
   primary triple, and project every coefficient of P, Q, and Psi;
2. apply only the pure modal raise coefficient-wise to old/new Psi, assemble
   full Kerr eth with pointwise radial Taylor products and quotients, run
   reconstruction pass 1, and project G and Lambda;
3. construct eth(G) in the same way, run pass 2, and project H, B, Pi, and C;
4. form sharp fields as `conj(field[-m])`, construct eth(C), eth(Pi),
   ethprime(Bsharp), and ethprime(Csharp), run pass 3, and project U.

Full Kerr GHP is deliberately not applied independently to sampled Taylor
coefficients: its `L^2 +/- i a R cos(theta)` denominators and connection terms
mix radial orders. Only the unit-sphere modal raise/lower commutes with radial
differentiation; the remaining factors are evaluated by normalized Taylor-jet
algebra.

Projection and angular outputs carry generation/level/pass tokens reduced from
every source theta node. Invalid, stale, or nonfinite source rows become zero
with zero stamps and cannot be made valid again by projection. Projection
seams are one-shot, nonaliasing, and fail closed. Matching work must remain on
the same ordered queue.

## Independent oracle and convergence range

`generate_routeb_angular_jet_fixture.py` is an independent 90-digit oracle. It
contains its own factorial Wigner-d sum, Gauss-Legendre quadrature, modal
analysis/synthesis and raise/lower factors, normalized radial Taylor algebra,
Kerr GHP quotients, signed sharp lookup, and pass-boundary projections. It does
not import production matrices or fit production output. Precision is set
before any mpmath constant is constructed, and an anti-binary64 assertion
rejects float-rounded `a`, `L`, or damping constants.

The frozen rotating test uses `a=0.63`, `L=1.4`, signed modes `m=-2,+2`, a
three-ell non-polynomial radial/modal h0 profile, `ell_max=5`, and 13 angular
nodes. It compares all ten fields at every angular node, both signs, and scri,
an interior midpoint, and the horizon through h0..h4. For every non-exact h4
field/mode/sample cell, both N9/N17 and N17/N33 radial error ratios exceed 15;
the N33 absolute maxima are below `2e-6`. The signed h4 fields are explicitly
distinct.

At h4 U at scri, every R-weighted radial term vanishes and the remaining
angular/pass algebra is grid-independent. Its observed N9/N17/N33 errors are
`6.76107e-18, 6.54293e-18, 7.65551e-18` for m=-2 and
`8.25720e-18, 1.10057e-17, 5.36584e-18` for m=+2. This single structural
binary64-floor case is gated by absolute error below `2e-15`, not a ratio.

N65 is a non-gating red roundoff-amplification probe. Max errors over sign and
the three radial samples are:

| field | N17 | N33 | N65 | N33/N65 |
|---|---:|---:|---:|---:|
| P | 3.03790e-9 | 2.65417e-11 | 6.53335e-11 | 0.406249 |
| Q | 2.53813e-8 | 7.55541e-10 | 2.74733e-9 | 0.275009 |
| Psi | 6.85807e-10 | 1.11443e-11 | 2.38957e-11 | 0.466372 |
| G | 2.16301e-9 | 4.00509e-11 | 3.19141e-10 | 0.125496 |
| Lambda | 5.88199e-9 | 7.63463e-11 | 2.64775e-10 | 0.288344 |
| H | 3.50788e-9 | 4.02806e-11 | 2.19405e-10 | 0.183590 |
| B | 1.30239e-8 | 1.24255e-10 | 9.11602e-10 | 0.136303 |
| Pi | 8.56648e-9 | 6.50594e-11 | 2.20600e-10 | 0.294920 |
| C | 1.13713e-8 | 8.52921e-11 | 3.11827e-10 | 0.273524 |
| U | 1.38847e-8 | 7.99284e-11 | 2.05404e-10 | 0.389129 |

Thus N9/N17/N33 is the frozen pre-roundoff qualification window. Using deeper
radial jets on finer binary64 production grids requires a separate precision
and error-budget strategy; the N65 probe is not evidence of convergence.

## Limits

This slice changes no evolved SBP operator and adds no dissipation. It provides
the source-independent linear diagnostic graph only. It does not qualify a
second-order source, time integration, stability over 200 M, wave extraction,
or GPU runtime behavior.
