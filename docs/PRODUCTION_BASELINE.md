# Production spin `+2` baseline

Date: 2026-08-12 (America/New_York)

This is the machine-specific Milestone-0 baseline for work beginning from the
audited handoff. It is evidence for a clean Serial source/test state, not a
production or accelerator qualification.

## Repository provenance

```text
HEAD and origin/main        c841ef7d4ee5ed2df635e88d37f777e099646095
scientific parent           4bd66b068c74c4439873c20f0e591b9138530eca
prior review anchor         43e9300080140e3df4d05affd67ffa41b5ddbe57
branch / remote             main / origin
remote URL                  git@github.com:HengruiZhu99/teuk-kokkos.git
Kokkos submodule            3ec81abe1816109f6f62ac48cef41921f91a4d00
Kokkos version              5.1.0
```

At baseline start, `HEAD` equalled the audited handoff exactly; there were no
intervening commits. `git status --short --branch` was clean and synchronized
with `origin/main`; `git diff --check` passed. Build files are ignored and do
not alter this classification.

The handoff commit changes only three documentation files relative to the
exact-tested scientific parent. It was not itself tested before this
baseline. The clean run below supplies exact-HEAD evidence.

## Machine and development environment

```text
OS/toolchain target         Linux x86_64
compiler                    g++ 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)
CMake                       3.28.3
C++ standard                C++20
Python                      3.12.3
SymPy                       1.14.0
PyYAML                      6.0.3
NumPy                       2.5.2
SciPy                       1.18.0
mpmath                      1.3.0
Kokkos backend              SERIAL only
build type                  Release
```

The exact CPython 3.12 Linux x86_64 wheels and SHA-256 hashes are pinned in
`requirements-audit-lock.txt`. `requirements.txt` remains the portable
broad-range input and is not sufficient for an exact environment by itself.
`qnm==0.4.4` and its wheel hash are fixture provenance; the checked-in QNM
generators do not import `qnm`.

Environment construction used

```bash
python3 -m venv /tmp/teuk-production-baseline.77bdlc
/tmp/teuk-production-baseline.77bdlc/bin/python -m pip install --upgrade pip
/tmp/teuk-production-baseline.77bdlc/bin/pip install -r requirements.txt
```

## Clean configure, build, and test evidence

Configure:

```bash
cmake --preset serial \
  -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-production-baseline.77bdlc/bin/python
```

Result: PASS. CMake selected GNU 13.3, Kokkos 5.1.0, Release, Serial ON,
and OpenMP/CUDA/HIP/SYCL OFF.

Clean build:

```bash
/usr/bin/time -v cmake --build --preset serial --clean-first --parallel 4
```

Result: PASS; wall `118.22 s`, user `431.85 s`, system `23.75 s`, maximum
resident set `879,948 KiB`.

Direct unit suite:

```bash
/usr/bin/time -v ./build/serial/teuk_tests
```

Result: PASS, `390/390`; wall `84.01 s`, user `83.92 s`, system `0.06 s`,
maximum resident set `104,156 KiB`.

Complete audit-enabled CTest:

```bash
/usr/bin/time -v ctest --preset serial --output-on-failure
```

Result: PASS, `24/24`; wall `185.79 s`, user `205.50 s`, system `0.66 s`,
maximum resident set `104,308 KiB`. Major groups were:

| Test/group | Result | Time |
|---|---:|---:|
| `teuk.unit` | PASS, 390/390 | 84.09 s |
| `teuk.qnm` | PASS, 2 cases | 14.88 s |
| config integration | PASS | 0.02 s |
| source compactification symbolic gate | PASS | 19.01 s |
| source normalization symbolic gate | PASS | 1.68 s |
| Route-B coordinate-curvature fixture freshness | PASS | 29.22 s |
| Route-B angular fixture freshness | PASS | 15.74 s |
| Route-B reconstruction fixture freshness | PASS | 7.46 s |
| all remaining symbolic, arbitrary-precision, TSI, and QNM generators | PASS | see CTest log |

This baseline makes no OpenMP, SYCL compilation, device discovery, B580
runtime, long-run stability, or resource-throughput claim.

## Independently confirmed starting blockers

### Checkpoint metadata is not bound to the actual companion PDE

`Plus2ReplayConfiguration` stores physical provenance and constructs checkpoint
expectations, but `Plus2CompanionPipeline` separately receives the actual
`UniformRadialGrid`, `TeukolskyParameters`, theta vector, reduction enum, and
dissipation. The constructor checks extents and radial scheme, not equality
of `M,a,L`, coordinate values, `dt`, reduction mode/damping, or dissipation.
Evolution uses the separate PDE objects. Therefore internally consistent
metadata A can restore successfully before evolution under physical problem
B. Existing tests prove file-versus-expectation validation, not
expectation-versus-PDE binding.

The checkpoint reader's temporary-host parse, exact metadata comparison,
extent/checksum validation, and fail-before-device-copy ordering are otherwise
sound. The repair must derive canonical expectations from the concrete PDE
objects, reject every retained duplicate before mutation, bind advance `dt`,
bind a content identity for primary state, and bind the concrete source
normalization capability.

### Route-B live endpoint readiness is not conditioned

The constrained endpoint moment formulas and standalone exact/high-precision
tests are sound. The live provider, however, accepts generation stamps and
finiteness, records peeling residuals without magnitude/error gates, and then
stamps every six-field/eight-slot output valid. The live convenience path
constructs a freely forgeable boolean claiming independently qualified scri
coefficients. No immutable method/grid/condition/error certificate is bound
to the stage.

Historical `N=65` evidence is materially red for the complete Route-B
reconstruction/angular tower: every recorded `h4` reconstruction
`N33/N65` error ratio is below one (`0.0529671`--`0.384328`). Current
promotion tests use `N=9,17,33`; `N=65` is recording-only or merely required
finite, and there is no `N=129,257,513` production-resolution evidence.

The production remedy must be intrinsically regular or carry a predeclared
conditioned coefficient construction. It must issue an immutable endpoint
qualification plus per-generation qualified curvature stage, retain the
unchanged `>15` design-order gate where truncation dominates, and use explicit
forward-error/oracle/dual-estimator ceilings where a proven roundoff bound
dominates. Fine grids remain mandatory rather than prohibited.

## Integrity and historical-evidence warnings

- Root `SHA256SUMS`, `MANIFEST.txt`, and `AUDIT_MANIFEST.json` describe the
  older 2026-08-11 spin `-2` audit bundle and are not current-tree integrity
  evidence. The latter two filenames are `.txt` and `.json`; references to
  nonexistent `MANIFEST.md` or `AUDIT_MANIFEST.md` are stale documentation.
- Historical documents contain commit-scoped test counts. They are not
  current totals unless the named commit and environment are reproduced.
- Ignored campaign data, checkpoints, plots, build logs, and local GPU events
  are not Git evidence.
- `plus2_equation_spec_proposal.yaml` remains a proposal.
- Existing B580 observations are commit-stale or local-only. There is no
  exact-HEAD accelerator runtime qualification.

## Baseline verdict

The exact handoff is a clean, green **Serial standalone-validation baseline**.
It is not production-qualified. Production spin `-2` remains the only runtime
path; every `plus2.enabled=true` mode remains deliberately fail-closed.
