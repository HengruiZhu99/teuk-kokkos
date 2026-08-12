# Spin `+2` homogeneous validation and normalized-TSI audit

Status: homogeneous validation only; full separated TSI fixture remains
blocked at normalized radial-mode evaluation

Base: `a31e286`

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
and the Eq. (2.44) sharp-to-same amplitude ratio `i/10`.

The nine named checks prove exactly the following:

1. both closed-form angular modes have the Eq. (2.6) endpoint
   normalization;
2. the individually normalized factors `Dhat` and `DhatPrime` are each `24`
   and multiply to `D=24^2`;
3. both directions of the first-form angular TSI (2.37) hold as functions of
   `theta`, not only at sample points;
4. for the pinned candidate radial metadata, the separately defined
   `Cin=Gamma` and `CinPrime=C/Gamma` multiply algebraically to `C`;
5. inserting those primary factors into Eq. (2.44) fixes the relative
   sharp-partner amplitude to `i/10`.

They do **not** prove the radial differential identities (2.38), evaluate a
radial mode, verify the absolute amplitude or phase of either Weyl scalar,
apply the Kinnersley-to-code-tetrad conversion, exercise the hyperboloidal
radial chain rule, or compare `T0[h]` with an evolved `Z_plus`.  The algebraic
`Cin*CinPrime=C` check ensures that no radial hatted factor was guessed; it is
not a substitute for applying the fourth-order radial operators to the
normalized modes.

### Remaining hard blocker

The full radial/field fixture is not claimed.  It still requires numerical
values and derivatives of the specifically normalized Eq. (2.17) HeunC
radial modes for both spins and the `(-omega,-m)` partner.  The local Wolfram
installation reports that the product is not activated, and the repository
has no independent confluent-Heun/eigenvalue implementation.  The
supplemental repository contains symbolic definitions and notebook examples,
not a portable exported table of normalized radial values.

Closing this blocker requires one of:

- an activated Mathematica run of the supplemental notebook that exports a
  versioned numerical fixture with parameters, normalization, values, and
  derivatives; or
- an independently validated confluent-Heun solver that implements Eq. (2.17)
  and reproduces the hatted first-form radial factors in Eq. (2.38).

After that export, the fixture must still be transformed from the
Boyer--Lindquist Kinnersley tetrad to the repository's rotated regular tetrad
and through the full hyperboloidal radial chain rule.  Until those steps are
present, neither a phase-sensitive full TSI result nor a `T0[h]` production
normalization is authorized.
