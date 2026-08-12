# Route-B stationary Kerr background derivatives

Status: implemented and independently audited

The local `Psi1^(1)` Ricci-identity route requires three derivatives of
stationary rotated-Kinnersley background coefficients.  With
`R=L^2/r`, `y=cos(theta)`, and the repository tetrad,

```text
Delta f    = (R^2/L^2) partial_R f,
bardelta f = R sin(theta)/(sqrt(2)(L^2+i a R y)) partial_y f
```

for a stationary axisymmetric scalar.  Therefore the production slots are

```text
capital_delta1_beta0    = R/L^2 [beta0 + R partial_R beta0],
capital_delta2_epsilon0 = 1/L^2 [2 R epsilon0 + R^2 partial_R epsilon0],
bardelta2_epsilon0      = sin(theta) partial_y epsilon0
                          /[sqrt(2)(L^2+i a R y)].
```

Here physical `beta=R beta0` and `epsilon=R^2 epsilon0`.  In the fixed-tetrad
weight bookkeeping used by the Ricci identity, `Delta beta` has `(1,-1)` and
radial power `R^1`; `Delta epsilon` has `(0,0)` and radial power `R^2`;
`bardelta epsilon` has `(-1,1)` and radial power `R^3`.  The three stored slots
divide out those respective powers.  Because beta and epsilon are connection
coefficients, these individual labels are fixed-tetrad bookkeeping rather
than claims that the slots transform as standalone GHP scalars; the complete
Ricci-identity combinations are covariant.

The implementation differentiates the compact background expressions with a
small device-callable `(R,y)` coordinate jet.  This keeps the formulas visibly
connected to the tetrad operators and does not add an automatic-differentiation
dependency.  Both Delta slots vanish at scri; the bardelta slot is finite and
generally nonzero there for Kerr.  All are finite at the future horizon.

`beta0` and `capital_delta1_beta0` retain the separate ordinary-NP
`cot(theta)` polar-coordinate singularity.  They must be evaluated on the
interior Gauss-Legendre nodes and used only in their weighted angular
combinations.  No finite value is assigned at an exact pole.  The epsilon
slots have finite pole limits; near-axis points are included in the symbolic
and host/device audits.

`tools/symbolic/verify_plus2_background_derivatives.py` independently applies
SymPy derivatives to the printed coordinate expressions and evaluates
Schwarzschild, positive and negative spin, `a=0.999`, scri, horizon, and
near-axis cases at high precision.  The C++ tests additionally compare the
coordinate jet with fourth-order numerical differences and require host/device
parity.
