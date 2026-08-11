# Implementation and validation plan for the Kokkos second-order Teukolsky solver

**Version:** 2.0.0  
**Date:** 2026-08-11

This document turns the corrected equations into an executable development sequence. It is intentionally more prescriptive than a normal design note because the target calculation is a long-time, near-extremal, weakly nonlinear instability study where a small source or integration error can masquerade as secular growth.

## 1. Scientific objective

Construct a performance-portable time-domain solver for

1. the first-order spin \(-2\) Teukolsky equation on Kerr;
2. outgoing-radiation-gauge metric reconstruction by seven nested null-transport equations;
3. the corrected quadratic source for \(\Psi_4^{(2)}\);
4. the driven second-order spin \(-2\) Teukolsky equation;
5. exact-extremal and near-extremal horizon diagnostics.

The baseline numerical representation is

\[
\text{finite differences/SBP in }R
\quad+\quad
\text{spin-weighted spectral treatment in }\theta
\quad+\quad
e^{im\phi}\text{ modes}.
\]

The implementation language is C++ with Kokkos. The same physics kernels must compile for CPU and available CUDA, HIP, and SYCL execution spaces without backend-specific equation code.

## 2. Source-of-truth hierarchy

Use the following hierarchy when references disagree:

1. executable checks in `verify_second_order_teukolsky.py`;
2. `equation_spec_v2.yaml` and `SOURCE_TERM_LEDGER.csv`;
3. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`;
4. the derivation in this package;
5. the two papers;
6. legacy Fortran/Julia code, used only for regression and convention comparison.

Do not copy expanded coefficients from the legacy code without deriving or generating them from the compact equations.

## 3. Repository architecture

Recommended initial structure:

```text
CMakeLists.txt
cmake/
configs/
docs/
  CONVENTIONS.md
  EQUATIONS.md
  ARCHITECTURE.md
  VALIDATION_MATRIX.md
  LEGACY_AUDIT.md
  PERFORMANCE.md
  STATE.md
  DECISIONS.md
include/teuk/
  core/
  geometry/
  angular/
  radial/
  evolution/
  reconstruction/
  source/
  diagnostics/
  io/
src/
tools/symbolic/
tools/reference/
tests/unit/
tests/integration/
tests/convergence/
benchmarks/
examples/
```

The symbolic tools should generate checked C++ headers or machine-readable coefficient files. Generated files must contain provenance hashes and must be checked in only together with their generators and tests.

## 4. Typed data model

### 4.1 Mode registry

Create an immutable registry containing:

- sorted unique signed \(m\) values;
- map \(m\to i_m\);
- conjugate lookup \(m\to -m\);
- nonnegative representative list used only where mathematically valid;
- target-mode list;
- deterministic ordered source pairs \((m_1,m_2,m_t)\);
- validation that every required input mode exists.

Never infer pairings from storage order. Handle \(m=0\) exactly once.

### 4.2 Field metadata

Every field type should carry compile-time or immutable runtime metadata:

- spin weight \(s\);
- boost weight \(b\);
- radial falloff \(n\);
- physical meaning;
- whether \(X^\sharp_m=\overline{X_{-m}}\) is required;
- angular bandlimit;
- current state generation/version for cache validation.

### 4.3 Device storage

Start with a benchmarked structure-of-arrays design. A reasonable baseline is

```cpp
using complex_type = Kokkos::complex<double>;
using view4_type = Kokkos::View<complex_type****, Kokkos::LayoutRight,
                                memory_space>;
// dimensions: mode, variable, radial, theta
```

with \(\theta\) contiguous. Also benchmark one View per field. Use accessors so the numerics do not depend strongly on one layout, but do not add abstraction that prevents device inlining.

Keep stationary background coefficients and angular matrices on the device. Do not transfer full fields to the host per stage.

## 5. Numerical components

### 5.1 Background and coordinates

Implement exact coordinate maps, horizon location, surface gravity, tetrad coefficients, spin coefficients, and rescaled background fields. Tests must cover:

- Schwarzschild limit;
- regularity at \(R=0\) and \(R=R_H\);
- \(\Delta\mu=-\mu^2\);
- agreement with symbolic formulas at randomized points;
- dimensionless rescaling under changes of \(M\) and \(L\).

Prefer \(M=1\) internally. If arbitrary \(M\) is supported, add explicit dimensional tests.

### 5.2 Spin-weighted angular representation

Implement modal and nodal representations at fixed \(m\). Required operations:

- synthesis and analysis on a Gauss-Legendre grid;
- \(\mathcal R_s\) and \(\mathcal L_s\);
- spin-weighted Laplacian;
- filtered/padded transforms for products;
- exact generalized Gaunt oracle for tests;
- spin/boost/falloff metadata checks on every operator.

Test orthonormality, round trips, eigenvalues, raising/lowering factors, adjoints, selection rules, and pole regularity.

### 5.3 Radial SBP operators

Correctness path:

1. diagonal-norm D4-2 first derivative;
2. compatible dissipation and SAT interface;
3. D6-3 after the vertical slice is verified;
4. optional smooth horizon-clustering map;
5. optional two-block SBP-SAT grid.

Tests must prove

\[
HD+D^TH=B
\]

to roundoff, polynomial exactness through the formal order, nonpositivity of dissipation in the SBP norm, and designed convergence for mapped and unmapped grids.

The baseline system is first order in radial derivatives using \((P,Q,\psi)\), so no second-derivative SBP operator is required for the production formulation. The Julia second-order-in-space code can remain a separate regression reference.

### 5.4 Boundary treatment

Derive characteristic speeds at future null infinity and the outer horizon from the implemented first-order system. Because the relevant radial principal coefficient vanishes at both ends, do not impose arbitrary Dirichlet conditions. Use SBP closures and SAT penalties only for genuinely incoming or reduction-constraint modes.

Before production, include a semi-discrete energy or normal-mode study for a frozen-coefficient representative system and long-time numerical tests with outgoing pulses.

### 5.5 Dissipation and filtering

Place compatible KO/SBP dissipation inside the RHS. Do not apply a post-step filter and reuse an old RHS. If a projection is unavoidable, increment the state generation and invalidate every derivative/source cache.

Make dissipation strength configurable and include it in convergence and instability sensitivity studies. The physical growth rate must be stable under a range that controls grid noise without dominating the solution.

## 6. Coupled equations and stage semantics

### 6.1 First-order Teukolsky state

Evolve \((P^{(1)},Q^{(1)},F)\) for each signed \(m\). Support both:

- free reduction evolution with damping;
- stage-wise constrained reconstruction \(Q=\partial_RF\).

Cross-check the two approaches on smooth solutions.

### 6.2 Reconstruction state

Evolve, in dependency order,

\[
G,\ \Lambda,\ H,\ B,\ \Pi,\ C,\ U.
\]

At every RK stage, later reconstruction equations use the same-stage values and same-stage time derivatives of earlier variables. This dependency chain must be explicit in code, not hidden in mutable global field objects.

### 6.3 Quadratic source state

Represent each named source family separately. Suggested identifiers mirror `SOURCE_TERM_LEDGER.csv`, for example:

```text
D_hll_DeltaF
D_hll_muF
D_F_ethC
D_F_connection_C
D_F_DeltaU
D_F_conjugate_U
...
T_conjugate_C_DeltaF
T_conjugate_B_ethprimeF
...
```

Expose optional per-family output. Sum only after each family has passed spin, boost, falloff, amplitude-squared, and operator-oracle tests.

### 6.4 Stage-local time derivatives

The outer operators require \(\partial_TD\) and \(\partial_TT\). Never reconstruct these with historical backward differences. Use one of these equivalent correctness paths:

1. `Jet1<T>{value, dt}` forward-mode propagation;
2. explicitly generated source JVPs;
3. exact algebraic elimination using the first-order/reconstruction equations.

The first implementation should use `Jet1` or explicit JVPs because it preserves the compact operator structure and is easy to compare with a CPU oracle.

### 6.5 Common-stage RK4

Use classical RK4 first. At stage \(a\):

1. build all first-order stage fields;
2. evaluate the first-order RHS;
3. build and evaluate reconstruction fields in dependency order;
4. evaluate all angular/radial derived quantities;
5. evaluate source values and tangents for every ordered mode pair;
6. form \(\mathcal S_a\) and \(\mathcal F_a^{(2)}\);
7. evaluate the second-order RHS;
8. accumulate all systems with the same \(a\), \(b\), and \(c\) coefficients.

No subsystem may be advanced to \(T_{n+1}\) before another subsystem evaluates its midpoint source.

After full fourth-order convergence is demonstrated, benchmark a low-storage fourth-order method. Retain classical RK4 as an oracle.

## 7. Independent CPU oracle

Implement a transparent CPU reference that prioritizes correspondence with the equations over performance. It should:

- use the compact operator-form source;
- use explicit loops and simple data structures;
- retain named source terms;
- support arbitrary smooth randomized fields;
- evaluate both values and stage tangents;
- serve as a pointwise oracle for Kokkos kernels.

Do not make the optimized Kokkos implementation and oracle share the same expanded generated expression if that would allow a single generator error to affect both. At least one path must preserve the compact equation structure.

## 8. Verification matrix

### 8.1 Symbolic tests

- all 94 bundled checks pass;
- generated C++ formulas match the YAML specification;
- background identities and falloffs;
- all field spin and boost weights;
- source family radial powers;
- corrected \(1/2\) factor;
- deliberate legacy coefficient fails;
- \(Q\)-equation derivative identities;
- horizon and scri principal limits.

### 8.2 Discrete angular tests

- orthonormality and transform round trip;
- raising/lowering factors and signs;
- angular Laplacian eigenvalues;
- \(X_m^\sharp=\overline{X_{-m}}\) implementation;
- ordered mode-pair selection;
- padded products versus Gaunt oracle;
- an intentionally unpadded test that shows aliasing, proving the oracle has power.

### 8.3 Discrete radial tests

- SBP identity and polynomial exactness;
- formal boundary closure order;
- dissipation energy sign;
- SAT stability tests;
- mapped-grid convergence;
- two-block interpolation/penalty conservation if implemented.

### 8.4 Time tests

- autonomous RK4 order four;
- explicitly forced ODE order four;
- coupled source-generating ODE order four;
- legacy endpoint-average method order two;
- stage-local JVP against finite-difference directional derivative;
- event-aligned delayed source activation;
- cache-version assertions.

### 8.5 PDE manufactured solutions

Create manufactured solutions for:

1. homogeneous Teukolsky equation with forcing added analytically;
2. each reconstruction transport equation separately;
3. the full dependency chain;
4. a synthetic bilinear source and driven second-order system;
5. coupled angular modes with a known Gaunt product.

Measure spatial and temporal orders separately by holding the other error negligible.

### 8.6 Physical regressions

- Schwarzschild evolution and source simplifications;
- moderate-spin comparison against both legacy codes while the solution is smooth;
- amplitude scaling \(F^{(1)}\propto A\), \(\mathcal S\propto A^2\), \(F^{(2)}\propto A^2\);
- source decomposition consistency;
- future-null-infinity waveform normalization;
- reconstruction residual convergence;
- restart versus uninterrupted run;
- CPU versus GPU backend parity.

Agreement with legacy second-order output is not an acceptance criterion until the source coefficient and stage integration differences are accounted for.

## 9. Near-extremal and extremal campaign design

### 9.1 Exact extremal sequence

Treat \(a=M\) as a dedicated configuration. Verify coordinate and coefficient regularity analytically and numerically. Use a radial resolution sequence and monitor

\[
\partial_R^nF|_{\mathcal H^+},\qquad
\partial_R^nF^{(2)}|_{\mathcal H^+},
\qquad n=0,1,2,3,4,\ldots
\]

in both raw and corotating/demodulated form. Fit powers only over windows stable under resolution, dissipation, extraction-point, and fitting-window changes.

### 9.2 Near-extremal sequence

Use a sequence such as

\[
a/M=0.9,\ 0.99,\ 0.999,\ 0.9999,\ldots
\]

subject to resolution. Record \(\kappa\), use \(\kappa v\), and test collapse of transient growth curves. The duration and radial width of the near-horizon transient should determine whether mapped or multiblock refinement is needed.

### 9.3 Source-family scaling

For each dominant \((\ell,m)\) and source family, record:

- local horizon power or transient scaling;
- radial localization;
- phase relative to the first-order field;
- contribution to each target \((\ell,m)\);
- response in \(F^{(2)}\).

Do not infer the quadratic response exponent solely from \(|F^{(1)}|^2\).

### 9.4 Perturbative validity

Introduce an explicit bookkeeping amplitude \(\epsilon\) and monitor, for chosen norms or observables,

\[
\frac{\epsilon^2|F^{(2)}|}{\epsilon|F^{(1)}|}
=\epsilon\frac{|F^{(2)}|}{|F^{(1)}|}.
\]

Report the time at which second-order corrections become comparable to first order for each chosen \(\epsilon\). This is essential if the quadratic instability grows faster.

## 10. Kokkos performance plan

### 10.1 Kernel decomposition

Initial kernels:

- background coefficient initialization;
- radial SBP derivative plus pointwise linear RHS, fused where practical;
- angular transform/operator application using `TeamPolicy` with league index \((m,R)\);
- reconstruction dependency kernels;
- ordered-pair source kernels;
- source tangent/JVP kernels;
- norm and diagnostic reductions.

Avoid materializing every derivative array. Fuse a stencil with immediate coefficient use when this does not make the reference path unreadable. Keep the oracle unfused.

### 10.2 Backend portability

Template on execution and memory spaces. Do not use raw CUDA/HIP/SYCL code in the physics layer. Backend-specific tuning may live behind small policy traits and must preserve a common test suite.

### 10.3 Benchmarking

For each important variant record:

- backend/device and compiler;
- \(N_m,N_R,N_\theta,N_{\rm var}\);
- wall time per RK stage and per physical time unit;
- effective grid-point updates/s;
- memory footprint;
- kernel launch count;
- source pair count;
- numerical difference from the oracle.

Benchmark before choosing dense angular strategy, field layout, source fusion, or low-storage integration.

### 10.4 MPI

Single-node correctness and performance come first. Radial slab decomposition is the preferred first MPI route because radial stencils require compact halos while angular mode coupling remains local to a slab. Provide device-aware MPI when available and host staging otherwise. Avoid decomposition by \(m\) until a communication model justifies it.

## 11. Milestones and gates

### M0: provenance and repository bootstrap

- copy the audit bundle into `docs/reference/` or another immutable location;
- record checksums;
- establish CMake, Kokkos, test harness, sanitizers, formatting, and state documents;
- run the Python audit in CI.

**Gate:** clean configure/build/test on a CPU backend.

### M1: background, modes, and angular operators

- coordinates and background fields;
- mode registry;
- spin-weighted transforms and GHP angular actions;
- Gaunt oracle.

**Gate:** all unit tests, randomized point tests, and product selection tests pass.

### M2: radial SBP and homogeneous linear evolution

- D4-2 SBP and compatible dissipation;
- first-order \((P,Q,F)\) system;
- characteristic boundary treatment;
- manufactured and pulse tests.

**Gate:** designed convergence and stable long-time moderate-spin evolution.

### M3: metric reconstruction

- seven stage-local transport equations;
- residuals and manufactured solutions;
- source-independent output.

**Gate:** all reconstruction residuals converge and CPU/Kokkos agree.

### M4: corrected quadratic source oracle

- named source families;
- ordered mode-pair table;
- value and tangent/JVP evaluation;
- padded angular products;
- explicit regression for the factor error.

**Gate:** operator oracle and optimized source agree; amplitude-squared and Gaunt tests pass.

### M5: coupled second-order vertical slice

- common-stage RK4;
- driven \((P^{(2)},Q^{(2)},F^{(2)})\);
- delayed source event support;
- checkpoint/restart.

**Gate:** full coupled manufactured solution converges fourth order in time; legacy staging test shows second order.

### M6: moderate-spin scientific regression

- compare with published/legacy smooth cases after accounting for the corrected source and timing;
- source-family output and waveform normalization;
- backend parity.

**Gate:** discrepancies are explained quantitatively and documented.

### M7: extremal/near-extremal infrastructure

- horizon-clustered map or multiblock option as required;
- high-order transverse derivative diagnostics;
- regular horizon tetrad/observer diagnostics;
- \(\kappa v\) output.

**Gate:** resolution and dissipation studies support stable fitted growth laws.

### M8: performance and scale

- kernel fusion/layout tuning;
- D6-3 option;
- available GPU backends;
- optional MPI radial decomposition.

**Gate:** performance report includes correctness deltas and reproducible commands.

## 12. Acceptance criteria for a research-ready release

A release suitable for the proposed instability study must satisfy all of the following:

1. the bundled symbolic audit passes unmodified;
2. the corrected connection coefficient is locked by a focused regression;
3. the full driven system demonstrates fourth-order temporal convergence;
4. the angular product path agrees with exact Gaunt convolution;
5. the radial operator satisfies its SBP identity and designed convergence;
6. reconstruction and reduction residuals converge;
7. CPU reference and every supported Kokkos backend agree within justified tolerances;
8. source amplitude-squared scaling holds;
9. exact-extremal and near-extremal configurations are not conflated;
10. growth fits are stable under resolution, dissipation, fitting window, and radial map changes;
11. gauge/tetrad status of every reported observable is documented;
12. restart, deterministic mode ordering, and provenance metadata are verified;
13. performance results never replace correctness evidence.
