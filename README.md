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

## Current validation status

The 2026-08-11 independent audit found that the earlier validation used the
wrong Teukolsky angular eigenvalue, evolved padded off-band nodal directions,
and drove the second-order equation from inconsistent zero reconstruction
data. **Results produced before the remediation commits beginning at
`0151757` are not scientifically qualified.**

The remediated production operator is
`-(ell-s)(ell+s+1)`, every complete RK stage derivative is projected into its
retained spin-weighted band, and the default second-order source is gated by a
causal start time plus genuinely independent reconstruction constraints. The
post-audit suite passes 133/133 C++ tests and 94/94 symbolic checks on Serial,
OpenMP, and Kokkos SYCL on an Intel Arc B580. A fresh 7,735-complex-value
checkpoint is bitwise identical between Serial and OpenMP; Serial and B580
differ by `2.68e-18` at maximum (`3.12e-16` relative maximum). CUDA and HIP
remain untested and are not claimed.

The Schwarzschild `ell=m=2` ringdown regression agrees with the trusted
fundamental frequency and damping to `8e-4`; all 13 RHS fields and a full RK4
step are bandlimited to relative residual below `3e-12`. See
`docs/POST_AUDIT_REMEDIATION.md` for the exact evidence and remaining limits.

Run the workstation-sized coupled example directly:

```bash
./build/serial/teuk_solver spatial-pipeline
```

The command initializes a signed-mode, band-limited Gaussian, evolves all 13
fields, and prints reduction, transport consistency, independent
reconstruction constraints, source-gate state, forcing, and horizon
diagnostics. The Gaussian does not supply constraint-solved reconstruction
data, so the default `source_mode=constraint_aware` keeps second-order forcing
off until both gates pass. Useful overrides include:

```bash
./build/serial/teuk_solver spatial-pipeline \
  spin=0.999 nr=129 ntheta=7 ellmax=4 modes=-4,-2,0,2,4 \
  final_time=0.01 steps=200 diagnostic_every=20 \
  source_start=0.5 source_constraint_tol=1e-10 \
  checkpoint_every=100 output=run-high-spin
```

`source_mode=unrestricted` is intentionally available for source-algebra and
temporal-order experiments. With the default zero reconstruction data it is
not a physical second-order initial-value problem; the executable and example
label that mode explicitly.

An output directory receives `diagnostics.csv` and `source_pairs.csv`. The
latter records RMS and maximum contributions from both inner-source families
`D` and `T` for every deterministic ordered pair `(m1,m2)->target` at each
diagnostic sample.

Checkpoint directories contain strict metadata plus an interleaved complex128
state. Compare states from two backends with:

```bash
scripts/compare_snapshots.py \
  run-serial/checkpoint_00000200/state.bin \
  run-sycl/checkpoint_00000200/state.bin
```

Resume for another interval with the same numerical configuration and time
step:

```bash
./build/serial/teuk_solver spatial-pipeline \
  spin=0.999 nr=129 ntheta=7 ellmax=4 modes=-4,-2,0,2,4 \
  final_time=0.01 steps=200 restart=run-high-spin/checkpoint_00000200 \
  output=run-high-spin-resumed checkpoint_every=200
```

The boundary treatment uses the verified D4-2 SBP closure and zero SAT because
the characteristic analysis finds no incoming propagating mode at either
endpoint. There is still no semi-discrete endpoint energy proof. Compatible
dissipation also adds the documented boundary commutator to the reduction
constraint; the exact law `C_Q,T=-gamma C_Q` applies only without
dissipation. Post-audit short runs cover `a/M=0,0.7,0.99,0.999`, but they do
not qualify long-time stability, fourth horizon derivatives, or an Aretakis
claim. See `docs/IMPLEMENTATION_STATUS.md`,
`docs/POST_AUDIT_REMEDIATION.md`, and `docs/FINAL_IMPLEMENTATION_REPORT.md`.
