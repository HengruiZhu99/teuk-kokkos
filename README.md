# teuk-kokkos

`teuk-kokkos` is a readable, performance-portable C++ solver under active
development for first- and second-order Teukolsky perturbations of Kerr black
holes. The intended pipeline is

```text
Psi4^(1) -> outgoing-radiation-gauge reconstruction
         -> corrected quadratic source -> Psi4^(2)
```

The production dependency is [Kokkos](https://github.com/kokkos/kokkos),
pinned as the `external/kokkos` Git submodule. Python, SymPy, and PyYAML are
development-only dependencies for the supplied mathematical audit.

## Clone, build, and test

```bash
git clone --recursive <repository-url>
cd teuk-kokkos
cmake --preset serial
cmake --build --preset serial
ctest --preset serial
```

An OpenMP build uses the same physics sources:

```bash
cmake --preset openmp
cmake --build --preset openmp
ctest --preset openmp
```

For the local Intel Level Zero path, first activate oneAPI and verify that a
GPU is visible, then use the SYCL preset:

```bash
source scripts/source_oneapi_level_zero_gpu.sh
teuk_source_oneapi_level_zero_gpu
cmake --preset sycl-intel-b580
cmake --build --preset sycl-intel-b580
ctest --preset sycl-intel-b580
```

Configuration or compilation alone does not establish GPU qualification. A
qualification claim additionally requires the device-kernel test to execute on
the named Level Zero device.

## Mathematical audit

The checked-in handoff bundle is the mathematical authority. In an isolated
development environment:

```bash
python3 -m venv /tmp/teuk-audit
/tmp/teuk-audit/bin/python -m pip install -r requirements.txt
/tmp/teuk-audit/bin/python verify_second_order_teukolsky.py
```

To include the same audit in CTest, configure with
`-DTEUK_ENABLE_SYMBOLIC_AUDIT=ON`
and `-DPython3_EXECUTABLE=/tmp/teuk-audit/bin/python`.

The expected final line is:

```text
Completed 94 checks with SymPy 1.14.0: 94 passed, 0 failed.
```

Read the scientific sources in this order:

1. `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`
2. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
3. `equation_spec_v2.yaml`
4. `SOURCE_TERM_LEDGER.csv`
5. `IMPLEMENTATION_AND_VALIDATION_PLAN.md`

The legacy Fortran code is a regression reference, not mathematical authority.
In particular, the corrected term is
`0.5 * (eth + conjugate(pi) + 2*tau) h_l_barm`, and driven forcing must be
evaluated at every common RK stage rather than interpolated between endpoints.

## Repository map

```text
include/teuk/   readable physics and numerical components
src/            executable entry point
tests/          dependency-free CTest suite
examples/       small workstation science runs
docs/           conventions, validation status, and measured performance
external/kokkos pinned mandatory dependency
```

See `docs/CONVENTIONS.md` for coordinates, rescalings, signed-mode handling,
angular normalization, layout, and stage semantics. See
`docs/IMPLEMENTATION_STATUS.md` for the current evidence and limitations.

## Runtime science runs

The primary executable is configured at runtime and no longer requires a new
C++ example or rebuild for each spin, grid, mode set, or pulse:

```bash
./build/serial/teuk_solver --config configs/linear_schwarzschild.cfg
```

Command-line overrides take precedence over the file and are convenient for
sweeps:

```bash
./build/serial/teuk_solver --config configs/near_extremal.cfg \
  spin=0.99 nr=1025 output.directory=run-a099
```

The strict dependency-free parser supports `#` comments, typed booleans,
integers, floating-point/scientific values, signed mode lists, multiple complex
seed modes, separate first/second angular bands, source policy, diagnostics,
and checkpoint cadence. Unknown, duplicate, or malformed keys fail closed.
Configuration is fully validated before the large state allocation.

Every run writes `resolved_config.cfg` beside its output with all defaults and
overrides plus Git, Kokkos, backend, precision, executable, and schema
provenance. Checkpoints persist strict geometry/method/mode compatibility and
the latched source-activation history. See
[`docs/RUNTIME_CONFIGURATION.md`](docs/RUNTIME_CONFIGURATION.md) for the full
schema, restart workflow, mode-completeness rules, and parameter-sweep example.

The checked-in templates are:

```text
configs/linear_schwarzschild.cfg
configs/kerr_ringdown.cfg
configs/second_order_22_self_coupling.cfg
configs/near_extremal.cfg
```

They are runnable development starting points, not blanket scientific
qualification.

## Current validation status

The 2026-08-11 independent audit found that the earlier validation used the
wrong Teukolsky angular eigenvalue, evolved padded off-band nodal directions,
and drove the second-order equation from inconsistent reconstruction data.
**Results produced before the remediation commits beginning at `0151757` are
not scientifically qualified.**

The remediated production operator is `-(ell-s)(ell+s+1)`. Complete RK stage
derivatives are projected into their declared fixed-`m`, fixed-spin bands.
Generated Gaussian data use a retained Galerkin solve for `P`; imported and
restarted data are measured and rejected if meaningfully off-band.

Second-order source activation is a persistent monotonic accepted-state event,
not an RK-stage-local branch. Steps split exactly at a prescribed start time,
all four stages see one flag, and normalized maximum plus weighted constraint
norms use the natural terms of each independent reconstruction equation.
Activation time and consecutive-pass state survive checkpoint/restart.

First-order parent modes and second-order targets may differ. Every quadratic
daughter `m1+m2` is required by default; explicit truncation emits a warning.
`ellmax_first` and `ellmax_second` are independent runtime values.

The dedicated production QNM test now uses the real Kokkos angular
coordinator, transforms, stage projections, SBP radial kernel, state layout,
and common-stage device RK4. Its Schwarzschild complex-frequency error falls
from `2.76e-3` to `3.99e-4` to `2.18e-4` over `N_R=17,25,33` and agrees across
two angular configurations. At `a/M=0.7`, the spherical-band sequence
`ellmax=3,4,5` self-converges and the final complex frequency is within
`5.41e-4` of the independent Leaver value
`0.5326002435510186 - 0.08079287315500745 i`. Exact setup, fit windows,
references, and limitations are in
[`docs/PRODUCTION_QNM_VALIDATION.md`](docs/PRODUCTION_QNM_VALIDATION.md).

The corrected half-factor, ordered-pair oracle, sharp modes, generalized
Gaunt products, analytic source tangent, transport equations, independent
Bianchi/reality checks, unrestricted common-stage RK4 convergence,
activation-event convergence, band invariance, and checkpoint equivalence
remain direct regressions. The symbolic audit still requires 94/94 checks.

The boundary treatment uses the verified D4-2 SBP closure and zero SAT because
the continuum characteristic analysis finds no incoming propagating endpoint
mode. There is no semi-discrete endpoint energy proof. No general
constraint-solved second-order Gaussian initializer, long-time near-extremal
qualification, high-order horizon-growth result, CUDA/HIP claim, or Aretakis
claim is made. See `docs/IMPLEMENTATION_STATUS.md`,
`docs/POST_AUDIT_REMEDIATION.md`, and `docs/BOUNDARY_SAT_BLOCKER.md`.
