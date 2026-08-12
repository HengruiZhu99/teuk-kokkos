# Spin `+2` homogeneous validation and normalized-TSI audit

Status: normalized Schwarzschild radial and field fixture passed; a
moderate-Kerr normalized separated fixture passed; Kerr field-level and QNM
extensions remain open

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

### Normalized moderate-Kerr separated fixture

`tools/numerical/generate_plus2_tsi_kerr_separated_fixture.py` adds a second,
independent fixture at

```text
M=1, a=0.6, omega=1/5, ell=m=2, horizon-in.
```

It uses only the pinned Berens source and supplemental normalization above.
In particular, it does not ask the production coefficient code for an
eigenvalue.  For each spin sign it integrates the canonical class-I Heun
solution from both angular endpoints and solves the Eq. (B.21) midpoint
Wronskian.  The independently obtained fine eigenvalues are

```text
lambda_(+2) = -0.796291369986029
lambda_(-2) =  3.203708630013962,
```

which satisfy the paper's independently predicted difference of four to
`3e-14`.  The modes retain the exact Eq. (2.6) `H(0)=1` normalization rather
than being rescaled after the solve.

Local Taylor jets generated from the independent angular and radial ODEs
apply both first-form ATSIs and RTSIs.  The fine maximum angular residual is
`5.52e-13`.  The four radial residuals (both first forms and both composed
second forms) are respectively

```text
4.95e-13, 4.78e-13, 7.92e-11, 2.07e-12.
```

The individual factors are retained: `Dhat=D/24`, `DhatPrime=24`,
`Cin=Gamma`, and `CinPrime=C/Gamma`, with the Kerr horizon frequency
`k=omega-m Omega_+` in `w=4 M k r_+`.  Their products reproduce `D` and `C`.
The real-frequency signed partner is solved separately at
`(-omega,-m)`; radial conjugation is exact at serialized precision and the
angular spin-flip/conjugation residual is `3.79e-15`.
The Eq. (2.44) sharp-to-same amplitude ratio is
`-0.0550458715596330-0.0834862385321101 i`; it is checked against
`(i omega M/2) CinPrime/conjugate(CinPrime)` and explicitly rejected if its
complex Kerr phase is replaced by the phase-free `i omega M/2` shortcut.

Tightening the endpoint seed, Heun series order, and ODE tolerances reduces
the normalized angular mode change from `5.82e-10` to `1.04e-11` and the
normalized radial mode change from `1.33e-9` to `9.65e-12`.  The generated
header stores all three levels and the SHA-256 of its generator and pinned
primary sources.  CTest regenerates it byte-for-byte under
`TEUK_ENABLE_SYMBOLIC_AUDIT`; the ordinary C++ suite also checks the factor
products, residual bounds, signed symmetry, and convergence metadata.

This closes the moderate-spin, real-frequency, normalized **separated-mode**
TSI gate.  It does not use or modify any production evolution path.

### Remaining Kerr and QNM gates

The two current fixtures still do not claim a QNM normalization, a Kerr
horizon endpoint value, or an evolved-companion comparison.  The
moderate-Kerr fixture stops before metric reconstruction and therefore does
not close the field-level `T0[h]` gate at nonzero spin.  Remaining work is:

- supply an independently pinned Schwarzschild QNM mode normalization (not
  merely the frequency already used by the production ringdown regression)
  and repeat the complex-frequency hatted-mode and sharp-partner checks;
- repeat that normalized QNM calculation at moderate Kerr spin;
- extend the ORG metric construction and the full Kinnersley-to-code and
  Boyer--Lindquist-to-hyperboloidal chain to `a != 0`, then compare against
  `evaluate_plus2_linear_psi0`;
- add horizon-regular endpoint data and an evolved-companion comparison.

These are separate gates.  In particular, the real-frequency fixture must
not be relabeled as a QNM test, and its separated identities must not be
relabeled as a nonzero-spin `T0[h]` comparison.
