# Route-B direct radial-jet Teukolsky slice

## Scope

`RouteBTeukolskyPrimaryJetTower` is a source-independent, zero-dissipation
diagnostic radial slice for the spin -2 primary Teukolsky triple. It stores
normalized radial Taylor coefficients for h0 through h4 and exposes one
level-at-a-time advancement. It is restricted to `FreeDamped`, nonnegative
reduction damping, and zero compatible dissipation.

This is not an autonomous Teukolsky graph, an SBP evolution operator, or a
runtime solver path. No `SpatialPipeline`, solver, or runtime wiring uses it.
On rotating Kerr, coefficient multiplication mixes ell, so a future external
coordinator must interleave every level as follows:

1. consume the current generation-stamped psi radial coefficients;
2. apply the spin-weighted angular operator independently to every active
   coefficient;
3. return the angular Taylor jet with the same generation;
4. advance exactly one radial-jet level.

The API deliberately does not accept angular values and then differentiate
them radially. Recovering `d_R A_s(psi)` with a Fornberg derivative of sampled
`A_s(psi)` differentiates the previous level's truncation error and recreates
the repeated-D105 failure mode. A regression demonstrates that this forbidden
path differs from the coefficient-wise result.

## Numerical design

h0 is sampled on a uniform compactified-radius grid. The independent
nine-point Fornberg D1 through D4 operators construct its degree-four Taylor
jet once. Kerr coefficients are then formed directly with product and quotient
jet algebra. Each Teukolsky application consumes one coefficient degree:

```text
h0 degree 4 -> h1 degree 3 -> h2 degree 2 -> h3 degree 1 -> h4 degree 0
```

The evolved D10-5 SBP operator is unchanged. This diagnostic uses no
dissipation because a dissipation Taylor jet has not been derived. All storage
is allocated at construction. Initialization and advancement launch without
internal fences, and the hot advance allocates nothing.
Initialization, every coefficient-wise angular launch, and all advances in one
generation must use the same ordered execution-space instance. The asynchronous
API does not establish dependencies between distinct backend queues.

## Qualification

The rotating endpoint gate uses a 100-digit mpmath fixture whose Kerr
coefficients, analytic exponential/trigonometric h0 profiles, prescribed
analytic angular jets, differentiation, and nested recurrence are independent
of the C++ Taylor-jet implementation. The fixture qualifies this radial slice;
it is not an oracle for a closed angular graph. At N=9, 17, and 33, every
non-exact P/Q/psi component through h4 has both absolute-error refinement
ratios above 15. The limiting h4 Q ratios are 15.7427 and 29.8625. The h4
fixture signal is finite and O(1), so the gate is not a zero-signal test.

For the rotating modal integration test, h1 P and psi are endpoint-exact rather
than generic 0/0 convergence cases. At both scri and the horizon the radial
principal coefficient multiplying d_R Q vanishes, and h1 psi is the algebraic
velocity, so their endpoint values contain no numerically differentiated h0
term. The test handles those two components explicitly; every other component
must exceed the fourth-order ratio threshold.

The N=17/33/65 h4 audit is a binary64 resolution-ceiling probe, not a promotion
gate. It remains strongly convergent at N=65. Direct D4 nevertheless has
maximum coefficient L1 norm 1740.8 and scales as h^-4, so refinement will
eventually amplify roundoff. The frozen promotion range remains N=9/17/33;
N=65 is recorded as additional evidence, not excluded because it failed.
The final N=17/33/65 h4 endpoint maximum errors were respectively
`3.90624e-6, 6.99127e-8, 9.95359e-10` for P,
`2.76892e-5, 9.27223e-7, 2.78689e-8` for Q, and
`6.96746e-7, 1.26997e-8, 2.27691e-10` for psi. Their corresponding ratios
were `55.873, 70.239`; `29.863, 33.271`; and `54.863, 55.776`.

Additional gates cover an independently expanded long-double coefficient
oracle through derivative four, host/device point parity, LayoutRight and
padded LayoutStride parity, amplitude linearity, signed-m and spin sensitivity,
generation and coefficient stamps, global poison-on-stale/nonfinite behavior,
strict null/shape/stride/alias checks, immutable copied mode/theta metadata,
and rejection of StageConstrained and nonzero dissipation.

## Limitations and follow-on contract

A production Route-B h0..h4 graph still needs an allocation-free angular
coordinator that acts componentwise on each active radial coefficient and
preserves signed-mode projection at every generation. Only after that graph is
qualified may these radial jets feed a local Z0/Z1 producer. This slice must not
be described as production evolution, SBP stability evidence, or a spin +2
implementation.
