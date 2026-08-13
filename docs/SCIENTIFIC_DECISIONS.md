# Scientific decisions

Updated: 2026-08-12 (America/New_York)

This concise ledger records decisions that constrain production work. The
derivations and historical evidence remain in the documents named by
`PRODUCTION_HIGH_SPIN_GOAL_MODE.md`.

1. Preserve `+---`, first-order ORG, rotated Kinnersley with `gamma=0`, and
   the no-factorial perturbation expansion.
2. Signed sharp is `X_m^sharp=conjugate(X_-m)`; same-mode conjugation is not a
   substitute.
3. Preserve `Psi4=R Z_minus` and
   `Psi0=R^5 Z_plus/(L^2-i a R cos(theta))^4`.
4. Keep raw `Psi0`, evolved `Z_plus`, `Z0_source=Psi0/R^5`, `Z1=Psi1/R^4`,
   local metric curvature, TSI reconstruction, and effective-source variables
   distinct.
5. The evolved spin-plus-two forcing is
   `2 D (L^2-i a R cos(theta))^4 (S0/R^7)`. Raw `S0/R^6` is diagnostic only;
   it is never divided numerically by `R`.
6. Bianchi-5 contains `+3 Sigma psi20`, not `+3 Sigma H`.
7. Route A remains validation-only in rotating Kerr because its pure-radial
   symbol has a nontrivial Jordan block. Production curvature follows local
   Route B.
8. Existing positive-node endpoint extractors remain diagnostic authority,
   not production authority through fine binary64 grids. Milestone 3 must
   compare at least two conditioned endpoint strategies before selection.
9. D10-5 is the qualified nested-derivative baseline. Its pure-dissipation
   timestep guard is necessary but is not a combined stability proof. The
   historical 400,000-step `T=200M` setup is rejected with D10-5 dissipation.
10. All coupled fields use one common classical RK4 stage and stage time. The
    spin-plus-two companion never feeds back into the primary trajectory.
11. Artificial dissipation versus physical curvature tangent remains an open
    Milestone-3 decision; no choice is implied by current standalone graphs.
12. Exact extremality is separate from near extremality and remains
    fail-closed unless independently derived and qualified.
13. Angular filtering is a sensitivity study only after unfiltered
    angular/source-band convergence.
14. `plus2.enabled=false` remains the production default until every mandatory
    gate is independently `PROVEN`.
