# Autonomous test status

Updated: 2026-08-12 (America/New_York)

## Exact-head Milestone-0 baseline

```text
commit       6f1db976fb96863554ad3f3269479e1e2a64edc7
branch       main
Kokkos       3ec81abe1816109f6f62ac48cef41921f91a4d00 / 5.1.0
compiler     GNU C++ 13.3.0
CMake        3.28.3
Python       3.12.3
backend      Kokkos Serial, Release
dependencies mpmath 1.3.0; numpy 2.5.2; PyYAML 6.0.3;
             scipy 1.18.0; sympy 1.14.0
```

Commands and results:

```bash
python3 -m venv /tmp/teuk-production-goal
/tmp/teuk-production-goal/bin/pip install --require-hashes \
  -r requirements-audit-lock.txt
cmake --preset serial \
  -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-production-goal/bin/python
cmake --build --preset serial --clean-first --parallel
./build/serial/teuk_tests
ctest --preset serial --output-on-failure
git diff --check
```

```text
environment install  PASS, 8.01 s
configure            PASS, 0.30 s
clean build          PASS, 231.29 s
direct unit          PASS, 396/396, 94.78 s
full CTest           PASS, 24/24, 205.07 s
diff check           PASS
first failing test   none
```

Complete local logs:

```text
/tmp/teuk-goal-logs/6f1db976fb96863554ad3f3269479e1e2a64edc7/serial/
```

These logs are local evidence and are not committed. The exact commands and
counts above are recorded here; the GitHub Actions Serial workflow runs the
same source-controlled audit sequence on pushes and pull requests.

No OpenMP, SYCL compilation, device discovery, B580 runtime, production-size
run, long run, VRAM, or throughput claim follows from this baseline.
