# Spin `+2` homogeneous validation and normalized-TSI audit

Status: normalized Schwarzschild radial and field fixture passed; the
moderate-Kerr extension remains open

Original homogeneous-validation base: `a31e286`

Normalized radial/field fixture base: `3a7d48f`

## Independent audit of the homogeneous slice

The field helper, coefficient oracle, and optional storage added at
`a31e286` were reviewed independently before adding tests.

- `plus2_code_tetrad_scaling` is an exact multiplication-only implementation
  of Ripley et al. arXiv:2010.00162 Eq. (21b),
  `R^5/(L^2-i a R cos(theta))^4`.
- The inverse helper is explicitly interior-only and does not hide a `0/0`
  evaluation at scri.
- `tests/ripley_eq22_oracle.hpp` directly transcribes Eqs. (22)--(23) at
  `s=+2`; it does not call the production coefficient helper.
- `Plus2CompanionStorage` is not included by or embedded in
  `SpatialPipeline`.  Disabled construction remains a disengaged optional,
  and its instrumented zero-allocation/zero-launch test remains the primary
  path isolation gate.

No discrepancy was found in this slice.  Enabled Kokkos View construction
may use backend initialization kernels, as its documentation already states;
the hard zero-launch requirement applies to the disabled path.

## Added homogeneous validation

`tests/test_plus2_validation.cpp` adds four validation families.

1. A source-free `s=+2`, `ell=3` evolution on the complete compact exterior
   from scri to the horizon self-converges at fourth order under RK4.  The
   smooth polynomial initial field is manufactured, but the evolution
   forcing is identically zero.
2. A separate manufactured test injects an explicitly labeled analytic
   validation forcing and starts with
   `Q-partial_R Z=C0`.  It verifies both fourth-order state convergence and
   the exact continuum law `C(T)=C0 exp(-gamma T)`.  This forcing is a test
   oracle for the linear operator; it is not a physical quadratic source.
3. The principal characteristics for `s=+2` are compared exactly with
   `s=-2` at scri, in the interior, and at the horizon for Schwarzschild,
   rotating, negative-spin, and `a/M=0.999` backgrounds.  Both endpoints
   retain one outgoing and two stationary modes with no incoming mode,
   including at the polar axes.
4. One Kokkos kernel evaluates the `s=+2` coefficient and field-scaling path
   at scri, interior points, and the horizon of an `a/M=0.999` black hole.
   Every result is compared after device-to-host copy with the independent
   `std::complex` Eq. (22) oracle and a direct Eq. (21b) expression.

These are validation-only calls to already generic point/radial kernels.
They do not add a production `s=+2` driver.

## Berens normalized separated fixture audit

Primary supplemental material was inspected at
`cf924707593a58ec889c70ea501d764e99d1d4aa`.  Contrary to a possible reading
of the earlier blocker, the hatted factors are not absent from the primary
source:

- arXiv:2403.20311 Eq. (2.6) fixes the angular modes by a HeunC solution with
  `H(0)=1`;
- Eq. (2.17) fixes the horizon-in/out radial modes, again with explicitly
  normalized HeunC factors;
- Eqs. (2.25) and (2.28)--(2.29) give the individual hatted angular and radial
  Starobinsky factors;
- Eq. (2.44) gives both the same-mode and sharp-partner amplitudes required
  when starting from one mode of `zeta^4 psi4`.

The supplemental `ExampleUsage.nb` independently transcribes these as
`calD`, `calC`, `Dhat`, `DhatPrime`, `Gamma`, `Cin`, and `CinPrime`, then uses
them in its single-Weyl-mode reconstruction rules.  No factor was inferred
from an unhatted product.

### Executable angular subfixture

`tools/symbolic/verify_plus2_tsi_angular_fixture.py` specializes the exact
hatted normalization to Schwarzschild `ell=m=2`.  Eq. (2.6) then gives

```text
S_hat_(+2) = (1-cos(theta))^2,
S_hat_(-2) = (1+cos(theta))^2.
```

The script applies the four first-form angular operators of Eq. (2.37) and
checks both identities with `Dhat=DhatPrime=24`, including their product
`D=24^2`.  It also pins a candidate full fixture to
`M=1`, `a=0`, real `omega=1/5`, `ell=m=2`, horizon-in modes.  For that choice
it verifies the individual radial-factor product from Eqs. (2.27)--(2.29)
and the Eq. (2.44) sharp-to-same amplitude ratio including the phase
`(i/10) C_in_prime/conjugate(C_in_prime)`.

The ten named checks prove exactly the following:

1. both closed-form angular modes have the Eq. (2.6) endpoint
   normalization;
2. the individually normalized factors `Dhat` and `DhatPrime` are each `24`
   and multiply to `D=24^2`;
3. both directions of the first-form angular TSI (2.37) hold as functions of
   `theta`, not only at sample points;
4. for the pinned candidate radial metadata, the separately defined
   `Cin=Gamma` and `CinPrime=C/Gamma` multiply algebraically to `C`;
5. inserting those primary factors into Eq. (2.44) fixes the relative
   sharp-partner amplitude including its complex radial-factor phase.

These angular checks by themselves do **not** prove the radial differential
identities, but the Schwarzschild numerical fixture described below now does
so independently and continues through the field-level `T0[h]` comparison.

### Normalized Schwarzschild radial fixture

`tools/numerical/verify_plus2_tsi_schwarzschild_radial_fixture.py` removes the
Mathematica dependency for the pinned Schwarzschild case

```text
M=1, a=0, omega=1/5, ell=m=2, horizon-in.
```

It implements the canonical confluent-Heun ODE in arXiv:2403.20311 Eq. (A.1),
the `H(0)=1` Frobenius recurrence in Eq. (A.2a), and the full hatted radial
prefactor in Eqs. (2.17)--(2.18).  SciPy `DOP853` integrates both `s=+2` and
`s=-2`; no production Teukolsky coefficient or radial-mode code is called.
Local Taylor jets obtain derivatives through order eight from the independent
second-order radial ODE.  They apply both first forms and both composed second
forms of Eq. (2.38) without finite-differencing a fourth derivative.

At `r={2.5,3,4,7,10}`, the finest run has a maximum relative residual
`1.56e-9`; this maximum is the eighth-order `s=+2` composition at `r=2.5`,
where singular Kinnersley factors amplify binary64 cancellation.  The other
fine residuals are between `3.3e-13` and `9.4e-12`.  Tightening the Frobenius
seed, series order, and ODE tolerances changes the normalized mode values by
`7.10e-10` from coarse to medium and `1.45e-11` from medium to fine.  The
operator residual is therefore bounded by floating cancellation rather than
used as a false monotone-convergence claim.

The script also verifies the real-frequency signed normalization

```text
R_hat(+2,-omega,ell,-m) = conjugate(R_hat(+2,omega,ell,m))
```

and retains the individual hatted factors `C_in=Gamma` and
`C_in_prime=C/Gamma`.

### Field-level normalized `T0[h]` fixture

`tools/numerical/generate_plus2_tsi_schwarzschild_t0_fixture.py` starts from
one normalized mode of `zeta^4 psi4` and constructs its ORG metric using
arXiv:2403.20311 Eqs. (2.50) and (2.52), with `epsilon_g=-1` and

```text
A_ORG=0, B_ORG=64/conjugate(C_in_prime).
```

It explicitly retains both Eq. (2.44) sectors:

```text
same:  (4 Dhat_prime/C_in_prime) exp(-i omega t+i m phi),
sharp: (48 i omega M/conjugate(C_in_prime)) exp(+i omega t-i m phi).
```

The phase regression checks their ratio as
`(i omega/2) C_in_prime/conjugate(C_in_prime)`; replacing either denominator
by its conjugate fails this check.  This corrects the earlier angular-only
script's informal `i/10` statement, which omitted the phase of the complex
radial hatted factor.

Before serialization, the generator independently checks the Ripley
coordinate/tetrad chain.  For Schwarzschild,

```text
T=t+H(r), H=-r+2M log((r-2M)/(2M))-4M log(r), R=1/r,
l_code=A_b l_Kin, n_code=A_b^-1 n_Kin, m_code=-m_Kin,
A_b=Delta/(2r^2).
```

Transforming the Boyer--Lindquist Kinnersley vectors with this Jacobian
reproduces every nonzero component of the repository tetrad.  Metric
projections consequently transform by `(A_b^2,-A_b,1)` for
`(h_ll,h_lm,h_mm)`.  The modal factor becomes
`exp(-i omega T) exp(+i omega H(r))`; the sharp sector gets the opposite
phase.  These checks are separate from the production `T0` implementation.

The committed generated fixture covers `r=4, theta=1.1` and
`r=6, theta=0.8`, both signed sectors, and all three numerical levels.  Its
coarse-to-medium and medium-to-fine changes are `4.21e-10` and `7.93e-12`.
All twelve cases pass through `evaluate_plus2_linear_psi0`; the observed
maximum absolute difference is `1.42e-15`.  The C++ gate is `2e-14`, a factor
fourteen allowance for decimal binary64 serialization and compiler
reassociation, not a weakening to the ODE error.

The generated header records the SHA-256 of both generator scripts and the
following primary provenance:

```text
Berens arXiv source BGL1.tex:
  f5af07f8692e4f8d49f271e41056eadb0bf180e78a6c8574f238930f2298b3cf
Berens supplement commit:
  cf924707593a58ec889c70ea501d764e99d1d4aa
Berens ExampleUsage.nb:
  d2c69c27f34c4da852c355a3240a1f351c05340182ee740bdf8fa56e5c05c974
Ripley arXiv source numerics_description.tex:
  2518ef1168e552db4ca4fd07ee421fca33f0939ecfc2031d272842b61cbf955e
```

Regeneration is fail-closed through the `--check` command and a CTest entry
under `TEUK_ENABLE_SYMBOLIC_AUDIT`.

### Remaining Kerr gate

This result closes the normalized radial and field-level TSI blocker only for
the specified Schwarzschild real-frequency horizon-in mode.  It does not
claim a general or moderate-spin Kerr fixture, a horizon endpoint value, a
QNM normalization, or an evolved-companion comparison.

The moderate-Kerr extension still requires an independent implementation of
the regular spin-weighted spheroidal eigenvalue and the hatted angular Heun
normalization at nonzero `a omega`, followed by the same signed-partner and
radial checks.  Guessing an eigenvalue from production coefficients would
make the oracle circular.  A future extension must:

- solve and converge the `s=+/-2` spheroidal eigenpairs independently;
- reproduce both hatted angular factors and both radial first forms;
- include the `(-omega,-m)` sharp sector with the complex Kerr phases;
- repeat the full Kinnersley-to-code and hyperboloidal chain at `a != 0`.
