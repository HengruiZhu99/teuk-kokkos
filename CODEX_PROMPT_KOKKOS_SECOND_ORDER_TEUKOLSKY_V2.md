# Codex master prompt: build a verified Kokkos-accelerated second-order Kerr Teukolsky solver

You are the autonomous lead scientific-software engineer and numerical-relativity researcher for a new codebase. You are starting in an empty Git repository. Your task is to design, implement, verify, document, benchmark, and leave working a modern C++/Kokkos time-domain solver for first- and second-order vacuum perturbations of Kerr black holes.

The primary science target is the exact-extremal and near-extremal horizon instability, including nonaxisymmetric first-order Aretakis-like growth and the quadratic response that it seeds. The code must therefore evolve the first-order Teukolsky field, reconstruct all first-order fields required by the outgoing-radiation-gauge source, evaluate the corrected quadratic source, and evolve the driven second-order Teukolsky field with a genuinely fourth-order coupled time integrator.

This is not a line-by-line port, a toy example, or a plan-writing exercise. Continue autonomously until the repository contains a tested end-to-end second-order vertical slice. Do not stop after scaffolding, a linear-only solver, or a source routine disconnected from time evolution.

Do not push to any remote unless explicitly instructed. Make local checkpoint commits throughout.

---

## 0. Mandatory input bundle and source-of-truth order

A reference bundle accompanies this prompt. Before changing the repository, read these files in order:

1. `README.md`
2. `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`
3. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
4. `equation_spec_v2.yaml`
5. `SOURCE_TERM_LEDGER.csv`
6. `verify_second_order_teukolsky.py`
7. `verify_second_order_teukolsky_output.txt`
8. `IMPLEMENTATION_AND_VALIDATION_PLAN.md`
9. this prompt

Run the audit immediately:

```bash
python3 verify_second_order_teukolsky.py \
  --json-out verify_second_order_teukolsky_output.local.json
```

The expected baseline is 94 passed and 0 failed with a compatible SymPy version. Preserve this audit in the new repository under `tools/symbolic/` or `docs/reference/`, together with the original checksum.

When sources disagree, use this hierarchy:

1. independent executable checks and tests;
2. `equation_spec_v2.yaml` and `SOURCE_TERM_LEDGER.csv`;
3. `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`;
4. the full derivation document;
5. arXiv:2008.11770 and arXiv:2010.00162;
6. legacy repositories, used only for regression and convention comparison.

The legacy repositories are:

```text
JLRipley314/teuk-fortran-2020
JLRipley314/TeukEvolution.jl
JLRipley314/2nd-order-teuk-derivations
```

Do not treat expanded legacy code as mathematical truth.

---

## 1. Required final repository state

The completed repository must contain at least:

1. A C++ library using Kokkos and modern CMake.
2. A command-line executable supporting:
   - homogeneous first-order Teukolsky evolution;
   - outgoing-radiation-gauge metric reconstruction;
   - corrected quadratic-source evaluation;
   - coupled driven second-order Teukolsky evolution;
   - exact-extremal and near-extremal diagnostics.
3. A radial finite-difference/SBP discretization appropriate for steep horizon layers.
4. A spin-weighted angular spectral representation with explicit signed \(m\) modes.
5. Deterministic, verified quadratic mode convolution and dealiasing.
6. A common-stage fourth-order integrator for first order, reconstruction, source, and second order.
7. A transparent operator-form CPU reference oracle.
8. Optimized Kokkos kernels with CPU and all available GPU backends tested.
9. Version-controlled symbolic derivations/generators and generated-header freshness checks.
10. Unit, integration, manufactured-solution, convergence, backend-parity, restart, and performance tests under CTest.
11. HDF5 or another suitable structured output format, restart files, and provenance metadata.
12. Documentation of equations, conventions, numerical methods, validation, performance, and known limitations.
13. Local commits at every coherent passing milestone.

The minimum scientifically meaningful vertical slice is:

```text
first-order Teukolsky
  -> seven reconstruction fields
  -> corrected quadratic source at common RK stages
  -> driven second-order Teukolsky
  -> source/residual/horizon diagnostics
```

Do not declare completion without this vertical slice and fourth-order driven convergence.

---

## 2. Non-negotiable audit findings

### 2.1 Correct the quadratic-source factor

The continuum source contains

\[
\Psi_4^{(1)}\frac12
(\eth+\bar\pi+2\tau)h_{l\bar m}.
\]

The factor \(1/2\) multiplies the entire operator combination. With

\[
\Psi_4^{(1)}=RF,
\qquad h_{l\bar m}=R^2C,
\]

the correct compact term is

\[
\boxed{
\frac12R\,\eth_2C
+\frac12R^2(\bar\pi_0+2\tau_0)C.
}
\]

The legacy Fortran code gives the second term coefficient 1 instead of \(1/2\). Add a focused regression that fails when the corrected coefficient is changed to the legacy value. Test at nonzero Kerr spin. A Schwarzschild-only test is powerless because the disputed connection combination vanishes there.

### 2.2 Do not reproduce the legacy order-two source staging

The legacy scheme uses

\[
S_1=S_n,
\qquad
S_2=S_3=\frac12(S_n+S_{n+1}),
\qquad
S_4=S_{n+1},
\]

so its RK4 source contribution is

\[
\frac{\Delta T}{6}(S_1+2S_2+2S_3+S_4)
=\frac{\Delta T}{2}(S_n+S_{n+1}).
\]

This is trapezoidal source quadrature and yields global order two for a time-dependent source. Evaluate all coupled systems at common stage states and abscissae. Include a convergence test that shows order four for the new method and order two for a deliberately implemented legacy method.

### 2.3 Never reuse a stale RHS after modifying a state

The legacy code filters an endpoint state and then reuses a pre-filter endpoint RHS as the next step's first stage. Do not do this. Put compatible dissipation in the method-of-lines RHS. If a projection or filter changes a state, invalidate and recompute every dependent derivative, source, and RHS cache.

### 2.4 Use explicit deterministic mode bookkeeping

Do not infer mode pairings from vector order or unordered containers. Store:

- sorted unique signed modes;
- \(m\)-to-index mapping;
- explicit conjugate lookup \(m\leftrightarrow -m\);
- target modes;
- all ordered source pairs \((m_1,m_2,m_t)\) with \(m_t=m_1+m_2\);
- explicit handling of \(m=0\) exactly once.

Use

\[
X_m^\sharp=\overline{X_{-m}},
\]

not \(\overline{X_m}\), unless a separate field-specific reality identity is established.

### 2.5 Use consistent units and event timing

Prefer \(M=1\) internally. Convert explicitly at input/output. Do not compare \(T/M\) with dimensionful \(T\). Delayed second-order activation must be an explicit event aligned to an RK step or handled by splitting a crossing step.

### 2.6 Verify nonlinear angular products independently

Overcollocation alone is not evidence of dealiasing. Compare production padded-collocation products against generalized spin-weighted Gaunt coefficients or exact modal convolution over the entire supported spin, mode, and bandlimit range.

### 2.7 State the gauge/tetrad status of outputs

Raw finite-radius or horizon \(\Psi_4^{(2)}\) is convention fixed but not generally invariant under first-order gauge and tetrad transformations. Mark outputs as:

- invariant;
- asymptotically interpretable at future null infinity;
- regular-tetrad/observer based;
- or convention-fixed diagnostics only.

Do not claim a physical second-order horizon instability from raw code-tetrad \(\Psi_4^{(2)}\) alone.

---

## 3. Efficient multi-agent workflow without quality loss

Use a coordinator-plus-specialists model. The main Codex agent is the sole integration authority and owns the canonical branch. When parallel-agent/worktree support is available, create isolated worktrees and run no more than four direct child agents concurrently. Do not allow nested delegation unless the environment explicitly guarantees safe hierarchy management.

### 3.1 Initial specialist assignments

Create four bounded agents with explicit file ownership:

**Agent A - formalism and symbolic verification**

Owns initially:

```text
tools/symbolic/
include/teuk/generated/
tests/unit/test_generated_*.cpp
docs/EQUATIONS.md
docs/CONVENTIONS.md
```

Tasks:

- import and preserve the 94-check audit;
- convert the YAML/CSV equation specification into validated generated C++ data or headers;
- implement background and source identity checks;
- produce a compact source oracle specification;
- add generated-file freshness and provenance tests.

**Agent B - angular representation, modes, and products**

Owns initially:

```text
include/teuk/angular/
src/angular/
include/teuk/core/mode_registry.hpp
src/core/mode_registry.cpp
tests/unit/test_angular_*.cpp
tests/unit/test_modes_*.cpp
```

Tasks:

- signed-mode registry and ordered pair table;
- spin-weighted harmonic transforms;
- raising/lowering/Laplacian operators;
- generalized Gaunt oracle;
- padded product/dealiasing path and tests.

**Agent C - radial SBP, boundaries, and linear Teukolsky evolution**

Owns initially:

```text
include/teuk/radial/
src/radial/
include/teuk/evolution/linear_*.hpp
src/evolution/linear_*.cpp
tests/unit/test_sbp_*.cpp
tests/integration/test_linear_*.cpp
```

Tasks:

- D4-2 diagonal-norm SBP reference operator;
- compatible dissipation and SAT infrastructure;
- first-order \((P,Q,\psi)\) system;
- characteristic boundary analysis/tests;
- manufactured and pulse evolution tests.

**Agent D - Kokkos infrastructure, build, I/O, and performance harness**

Owns initially:

```text
CMakeLists.txt
cmake/
include/teuk/core/
src/core/
include/teuk/io/
src/io/
benchmarks/
.github/ or CI files if appropriate
```

Tasks:

- CMake/Kokkos discovery and backend configuration;
- typed Views and memory-space abstraction;
- test harness, sanitizers, formatting, CI;
- structured output/restart/provenance;
- benchmark framework.

### 3.2 Coordinator responsibilities

The coordinator must:

- create `AGENTS.md` with ownership rules before delegation;
- prevent overlapping edits;
- define concrete acceptance tests for every agent task;
- inspect every agent commit and report before integration;
- run the full test suite after every merge/cherry-pick;
- own `docs/STATE.md`, `docs/DECISIONS.md`, and `docs/VALIDATION_MATRIX.md`;
- resolve interfaces centrally rather than allowing agents to invent incompatible APIs;
- reject code that merely compiles without independent tests;
- perform or assign a final independent audit pass after integration.

Each agent must work on a dedicated branch/worktree, make small local commits, run its owned tests, and leave a concise handoff report in `docs/agents/<agent>.md` containing:

- files changed;
- equations/assumptions used;
- exact commands run;
- test output;
- unresolved issues;
- commit hashes.

The coordinator should parallelize independent tasks, then serialize integration at stable interfaces. Do not have multiple agents simultaneously edit the coupled integrator or the same source routine.

### 3.3 Second wave of agents

After M2 is integrated, reassign agents sequentially or in a new bounded wave:

- reconstruction equations and residuals;
- compact quadratic source oracle plus JVPs;
- coupled RK4 and driven tests;
- independent reviewer/performance specialist.

At least one agent that did not author the source implementation must review the source against `SOURCE_TERM_LEDGER.csv` and randomized oracle tests.

### 3.4 Fallback without agent support

If direct child agents or worktrees are unavailable, execute the same ownership plan sequentially. Preserve separate commits and handoff notes so independent review boundaries remain visible. Do not weaken tests because parallel execution is unavailable.

---

## 4. Development discipline

### 4.1 Autonomous operating rules

Proceed without repeatedly asking for approval. When blocked:

1. reproduce and isolate the failure;
2. record the exact command and output;
3. identify whether it is environmental, architectural, or mathematical;
4. choose the lowest-risk working alternative;
5. document the decision;
6. continue independent work.

Do not perform repeated blind retries. Do not hide unavailable backends. Keep the CPU path working and provide exact commands for unavailable GPU validation.

### 4.2 Scientific test-driven workflow

For each component use:

1. **RED:** add an independent failing test;
2. **GREEN:** implement the smallest correct path;
3. **SIMPLIFY:** refactor without weakening the test;
4. **INDEPENDENT AUDIT:** compare with a second derivation, exact identity, manufactured solution, convergence rate, or separate oracle.

Never delete a failing scientific test merely to obtain green CI. Never loosen a tolerance without quantitative justification in `docs/DECISIONS.md`.

### 4.3 Required state documents

Create and maintain:

```text
AGENTS.md
docs/STATE.md
docs/DECISIONS.md
docs/VALIDATION_MATRIX.md
docs/LEGACY_AUDIT.md
docs/CONVENTIONS.md
docs/EQUATIONS.md
docs/ARCHITECTURE.md
docs/PERFORMANCE.md
docs/KNOWN_LIMITATIONS.md
docs/agents/
```

Only the coordinator writes the canonical `docs/STATE.md`.

### 4.4 Checkpoint commits

Use small local commits after coherent passing milestones, for example:

```text
test: preserve triple-checked symbolic audit
feat: add deterministic signed mode registry
feat: implement spin-weighted angular operators
feat: add d4-2 sbp radial derivative
feat: evolve homogeneous teukolsky system
feat: add seven-field metric reconstruction
fix: use corrected half connection term in source
feat: propagate stage-local source tangents
feat: stage-couple driven second-order rk4
perf: fuse radial stencil and linear rhs
```

Do not mix unrelated changes. Do not push.

---

## 5. Mathematical conventions and equations

### 5.1 Coordinates and Fourier modes

Use

\[
r=\frac{L^2}{R},
\qquad
R=0\text{ at }\mathscr I^+,
\qquad
R_H=\frac{L^2}{r_+}.
\]

Decompose

\[
X(T,R,\theta,\phi)=\sum_mX_m(T,R,\theta)e^{im\phi}.
\]

Use

\[
X_m^\sharp=\overline{X_{-m}}.
\]

### 5.2 Regularized first-order and metric fields

Implement exactly:

\[
\Psi_4^{(1)}=RF,
\quad
\Psi_3^{(1)}=R^2G,
\quad
\Psi_2^{(1)}=R^3H,
\]

\[
\lambda^{(1)}=R\Lambda,
\quad
\pi^{(1)}=R^2\Pi,
\]

\[
h_{\bar m\bar m}=RB,
\quad
h_{l\bar m}=R^2C,
\quad
\mu h_{ll}=R^3U.
\]

Metadata:

| physical field | stored field | spin | boost | falloff |
|---|---:|---:|---:|---:|
| \(\Psi_4^{(1)}\) | \(F\) | -2 | -2 | 1 |
| \(\Psi_3^{(1)}\) | \(G\) | -1 | -1 | 2 |
| \(\Psi_2^{(1)}\) | \(H\) | 0 | 0 | 3 |
| \(\lambda^{(1)}\) | \(\Lambda\) | -2 | -1 | 1 |
| \(\pi^{(1)}\) | \(\Pi\) | -1 | 0 | 2 |
| \(h_{\bar m\bar m}\) | \(B\) | -2 | 0 | 1 |
| \(h_{l\bar m}\) | \(C\) | -1 | 1 | 2 |
| \(\mu h_{ll}\) | \(U\) | 0 | 1 | 3 |

### 5.3 Background fields

Use

\[
\mu=R\mu_0,
\qquad
\mu_0=\frac{1}{-L^2+iaR\cos\theta},
\]

\[
\tau=R^2\tau_0,
\qquad
\tau_0=\frac{ia\sin\theta}
{\sqrt2(L^2-iaR\cos\theta)^2},
\]

\[
\pi=R^2\pi_0,
\qquad
\pi_0=-\frac{ia\sin\theta}
{\sqrt2(L^4+a^2R^2\cos^2\theta)},
\]

\[
\Psi_2^{(0)}=R^3\psi_{20},
\qquad
\psi_{20}=-\frac{M}{(L^2-iaR\cos\theta)^3}.
\]

Lock the identity

\[
\Delta\mu=-\mu^2
\]

with a test.

### 5.4 Rescaled GHP operators

For physical falloff \(R^nX\), define

\[
\Delta_nX=R^{-n}\Delta(R^nX),
\]

\[
\eth_nX=R^{-(n+1)}\eth(R^nX),
\qquad
\eth'_nX=R^{-(n+1)}\eth'(R^nX).
\]

Implement

\[
\Delta_nX=
\left(2+\frac{4MR}{L^2}\right)\partial_TX
+\frac{R}{L^2}(R\partial_RX+nX).
\]

For spin \(s\), boost \(b\), \(p=s+b\), \(q=-s+b\),

\[
\eth_nX=
\frac{-ia\sin\theta\,\partial_TX+\mathcal R_sX}
{\sqrt2(L^2-iaR\cos\theta)}
-\frac{ipaR\sin\theta}
{\sqrt2(L^2-iaR\cos\theta)^2}X,
\]

\[
\eth'_nX=
\frac{ia\sin\theta\,\partial_TX+\mathcal L_sX}
{\sqrt2(L^2+iaR\cos\theta)}
+\frac{iqaR\sin\theta}
{\sqrt2(L^2+iaR\cos\theta)^2}X.
\]

Use

\[
\mathcal R_s\,{}_sY_{\ell m}
=\sqrt{(\ell-s)(\ell+s+1)}\,{}_{s+1}Y_{\ell m},
\]

\[
\mathcal L_s\,{}_sY_{\ell m}
=-\sqrt{(\ell+s)(\ell-s+1)}\,{}_{s-1}Y_{\ell m}.
\]

### 5.5 Seven reconstruction equations

Implement exactly:

\[
\Delta_2G=-4R\mu_0G+\eth_1F-R\tau_0F,
\]

\[
\Delta_1\Lambda=-R(\mu_0+\bar\mu_0)\Lambda-F,
\]

\[
\Delta_3H=-3R\mu_0H+\eth_2G-2R\tau_0G,
\]

\[
\Delta_1B=R(\mu_0-\bar\mu_0)B-2\Lambda,
\]

\[
\Delta_2\Pi=-G-R(\bar\pi_0+\tau_0)\Lambda
+\frac12R^2\mu_0(\bar\pi_0+\tau_0)B,
\]

\[
\Delta_2C=-R\bar\mu_0C-2\Pi-R\tau_0B,
\]

\[
\begin{aligned}
\Delta_3U={}&-R\bar\mu_0U
-R\mu_0\eth_2C
-R^2\mu_0(\bar\pi_0+2\tau_0)C
-2\eth_2\Pi-2R\bar\pi_0\Pi-2H\\
&-2R\pi_0\Pi^\sharp
-R\pi_0\eth'_1B^\sharp
+R^2\pi_0^2B^\sharp
+R\mu_0\eth'_2C^\sharp\\
&+R^2(-3\mu_0\pi_0+2\bar\mu_0\pi_0
-2\mu_0\bar\tau_0)C^\sharp.
\end{aligned}
\]

For each \(\Delta_nX=\mathcal R_X\), solve

\[
\partial_TX=
\frac{\mathcal R_X-(R^2/L^2)\partial_RX-(nR/L^2)X}
{2+4MR/L^2}.
\]

The dependency order is

```text
F -> G -> Lambda -> H -> B -> Pi -> C -> U
```

at every common RK stage.

### 5.6 Correct second-order source

The physical equation is

\[
\mathcal T\Psi_4^{(2)}=\mathcal S,
\]

\[
\mathcal S=(\Delta+4\mu+\bar\mu)s_d
+(\eth'+4\pi-\bar\tau)s_t.
\]

The full physical \(s_d\) and \(s_t\) appear in the reference documents. For production use the following compact ordered-pair form.

Define

\[
D=\frac{s_d}{R^3},
\qquad
T=\frac{s_t}{R^3}.
\]

For one ordered pair \((m_1,m_2)\):

\[
\begin{aligned}
D_{12}={}&
\frac12U_1\left(\frac{\Delta_1F_2}{\mu_0}+RF_2\right)\\
&+F_1\Bigg[
\frac12R\eth_2C_2
+\frac12R^2(\bar\pi_0+2\tau_0)C_2
+\frac{\Delta_3U_2}{\mu_0}
+RU_2^\sharp\\
&\qquad
-\frac12R\eth'_2C_2^\sharp
+\frac12R^2(5\pi_0+4\bar\tau_0)C_2^\sharp
\Bigg]\\
&-\frac12G_1\Bigg[
R\eth_1B_2
+R^2(\bar\pi_0+\tau_0)B_2
+R\Delta_2C_2
+R^2(-2\mu_0+\bar\mu_0)C_2
\Bigg]\\
&-RC_1\Delta_2G_2
+\frac12RB_1\eth_2G_2
+4R\Pi_1G_2
-3R\Lambda_1H_2.
\end{aligned}
\]

\[
\begin{aligned}
T_{12}={}&
-C_1^\sharp
\left[\Delta_1F_2+R(\mu_0+2\bar\mu_0)F_2\right]
+\frac12B_1^\sharp\eth'_1F_2\\
&+F_1\left[
\Pi_2^\sharp-\Delta_2C_2^\sharp
+\eth'_1B_2^\sharp
-\frac12R(\pi_0+\bar\tau_0)B_2^\sharp
\right].
\end{aligned}
\]

Sum every ordered pair with \(m_1+m_2=m_t\):

\[
D_{m_t}=\sum D_{12},
\qquad
T_{m_t}=\sum T_{12}.
\]

Then

\[
\frac{\mathcal S_{m_t}}{R^3}
=\Delta_3D_{m_t}
+R(4\mu_0+\bar\mu_0)D_{m_t}
+R\eth'_3T_{m_t}
+R^2(4\pi_0-\bar\tau_0)T_{m_t},
\]

and the coordinate forcing is

\[
\boxed{
\mathcal F_{m_t}^{(2)}
=2(L^4+a^2R^2\cos^2\theta)
\frac{\mathcal S_{m_t}}{R^3}.
}
\]

Represent the terms as named source families corresponding one-to-one with `SOURCE_TERM_LEDGER.csv`. Keep optional per-family output.

### 5.7 First-order Teukolsky reduction

For spin \(s\) and mode \(m\), define

\[
C_T=8M\left(2M-\frac{a^2R}{L^2}\right)
\left(1+\frac{2MR}{L^2}\right)-a^2\sin^2\theta,
\]

\[
K=L^2-\frac{(8M^2-a^2)R^2}{L^2}
+\frac{4a^2MR^3}{L^4},
\]

\[
H_R=\frac{R^2}{L^4}(L^4-2L^2MR+a^2R^2),
\]

\[
\begin{aligned}
G_m={}&2iam\left(1+\frac{4MR}{L^2}\right)\\
&+2\left[
2M\left(-s+(2+s)\frac{2MR}{L^2}
-\frac{3a^2R^2}{L^4}\right)
-\frac{a^2R}{L^2}+isa\cos\theta
\right].
\end{aligned}
\]

Define

\[
P=C_T\partial_T\psi-2K\partial_R\psi+G_m\psi,
\qquad
Q=\partial_R\psi.
\]

Then

\[
\partial_T\psi=\frac{P+2KQ-G_m\psi}{C_T},
\]

\[
\begin{aligned}
\partial_TP={}&H_R\partial_RQ\\
&+\left[
2R\left(1+s-(3+s)\frac{MR}{L^2}
+\frac{2a^2R^2}{L^4}\right)
-\frac{2iamR^2}{L^2}
\right]Q\\
&+\left[
-2R\left((1+s)\frac{M}{L^2}
-\frac{a^2R}{L^4}\right)
-\frac{2iamR}{L^2}
\right]\psi
+{}_{s}\!\Delta_{\Omega}\psi+\mathcal F,
\end{aligned}
\]

\[
\partial_TQ=\partial_R
\left(\frac{P+2KQ-G_m\psi}{C_T}\right)
-\gamma_Q(Q-\partial_R\psi).
\]

Do not hand-copy the legacy expanded \(Q\) coefficients. Generate the derivative form or impose the reduction at each stage. Implement both for cross-checking.

---

## 6. Stage-local source tangents and coupled RK4

The outer source uses \(\Delta_3D\) and \(\eth'_3T\), both of which contain coordinate time derivatives. Do not use finite-difference histories.

At any common stage state \(U\), the stationary linear first-order/reconstruction system gives

\[
\dot U=L(U),
\qquad
\ddot U=L(\dot U).
\]

For every bilinear primitive \(B(U,V)\), propagate

\[
\partial_TB(U,V)=B(\dot U,V)+B(U,\dot V).
\]

Implement either:

```cpp
template<class T>
struct Jet1 {
  T value;
  T dt;
};
```

with overloaded algebra/operators suitable for device code, or generated explicit JVP functions. The CPU oracle should make the chain rule transparent.

For each classical RK4 stage:

1. form first-order stage state;
2. evaluate first-order RHS and needed \(\ddot U\) data;
3. form reconstruction stage states in dependency order;
4. evaluate reconstruction RHSs and their tangents;
5. evaluate \(D_{12}\), \(T_{12}\), and their tangents for all ordered mode pairs;
6. apply the outer source operators;
7. evaluate second-order \((P,Q,F)\) RHS;
8. accumulate every subsystem with the same RK stage coefficients.

Add debug-generation counters to derived-data caches. A cache computed from state generation \(g\) must not be consumed by generation \(g+1\).

---

## 7. Spatial numerical method

### 7.1 Radial SBP reference

Implement a diagonal-norm D4-2 SBP first derivative first. Verify

\[
HD+D^TH=B
\]

to roundoff, polynomial exactness, boundary closure order, and convergence. Add compatible dissipation \(Q_d\) satisfying nonpositivity in the SBP norm.

After the complete second-order vertical slice passes, add D6-3.

### 7.2 Boundaries

Derive radial characteristic fields and speeds from the actual implemented first-order system at \(R=0\) and \(R=R_H\). The radial principal coefficient \(H_R\) vanishes at both ends. Do not impose arbitrary Dirichlet data. Apply SAT only to incoming physical or reduction-constraint modes justified by the characteristic analysis.

Include a frozen-coefficient energy/normal-mode check and long-time pulse tests.

### 7.3 Horizon resolution

Start with uniform \(R\). Add an optional smooth monotone horizon-clustering map once resolution studies justify it. If a thin layer cannot be resolved efficiently, add two radial SBP blocks with verified SAT coupling before adding nonlinear shock capturing.

Do not use WENO/MP5 by default. The expected instability is growth of transverse derivatives, not necessarily a discontinuity in the field. Any experimental nonlinear stencil must demonstrate smooth-solution order and non-suppression of the measured growth.

### 7.4 Angular representation and dealiasing

At fixed signed \(m\), use spin-weighted spherical harmonics or the equivalent spin-weighted associated Legendre basis. Required tests:

- quadrature orthonormality;
- nodal/modal round trip;
- raising/lowering coefficients and signs;
- spin-weighted Laplacian eigenvalues;
- adjoints under quadrature;
- pole regularity;
- conjugate-mode identities.

For quadratic products, use a padded angular grid or exact modal convolution. Verify against

\[
\begin{aligned}
&\int {}_{s_1}Y_{\ell_1m_1}
{}_{s_2}Y_{\ell_2m_2}
\overline{{}_{s_3}Y_{\ell_3m_3}}\,d\Omega\\
&=(-1)^{m_3+s_3}
\sqrt{\frac{(2\ell_1+1)(2\ell_2+1)(2\ell_3+1)}{4\pi}}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\-s_1&-s_2&s_3\end{pmatrix}
\begin{pmatrix}\ell_1&\ell_2&\ell_3\\m_1&m_2&-m_3\end{pmatrix}.
\end{aligned}
\]

Include an intentionally unpadded test that shows aliasing so the oracle is proven sensitive.

---

## 8. Kokkos architecture and performance rules

### 8.1 Portability

Write generic code over execution and memory spaces. Support Kokkos Serial/OpenMP and every GPU backend present in the environment. Keep CUDA/HIP/SYCL-specific code out of the physics equations. Backend-specific policy traits may tune league/team/vector sizes only.

Use `Kokkos::complex<double>` or a verified portable equivalent. Avoid device-incompatible `std::complex` operations.

### 8.2 Baseline data layout

Benchmark at least:

```cpp
Kokkos::View<complex_type****, Kokkos::LayoutRight, memory_space>
  state("state", nmode, nvar, nr, ntheta);
```

and a structure-of-arrays design. With `LayoutRight`, \(\theta\) is unit stride. Choose based on measured bandwidth, register pressure, angular kernels, and source fusion. Record the decision and benchmark data.

### 8.3 Kernels

Implement and benchmark:

- background initialization;
- fused radial stencil plus pointwise linear RHS;
- `TeamPolicy` angular matrix-vector/transform kernels with league rank \((m,R)\);
- reconstruction dependency kernels;
- compact ordered-pair source kernels;
- source tangent/JVP kernels;
- reductions for norms, residuals, and horizon diagnostics.

Avoid allocating full derivative Views for every field if a stencil can be consumed immediately. Retain an unfused reference path.

Precompute stationary background coefficients and angular matrices on the device. Avoid per-stage host synchronization and small host-device transfers.

### 8.4 Performance evidence

For each important variant record:

- backend, compiler, device;
- \(N_m,N_R,N_\theta,N_{\rm var}\);
- source-pair count;
- time per RK stage and per step;
- grid-point updates/s;
- memory footprint;
- kernel launch count;
- correctness delta from the oracle;
- profiling evidence when available.

Never accept a faster kernel that weakens numerical agreement or bypasses source-family tests.

### 8.5 MPI

Single-node correctness and performance come first. Add optional MPI by radial slab decomposition after the single-node solver is stable. Use compact halos and device-aware MPI when available, with a host-staging fallback. Do not decompose by \(m\) prematurely because the quadratic source couples modes.

---

## 9. Required independent tests

Create focused test executables integrated with CTest. Keep the mandatory test stack lightweight.

### 9.1 Symbolic/provenance tests

- bundled 94-check audit passes;
- YAML/CSV schemas parse and validate;
- generated headers match generator output and hashes;
- background NP fields match formulas;
- GHP weights and falloffs;
- \(\Delta\mu=-\mu^2\);
- source term spin, boost, and falloff ledgers;
- corrected half coefficient;
- legacy coefficient deliberately fails;
- coordinate forcing prefactor;
- first-order reduction identities;
- horizon/scri principal limits.

### 9.2 Angular/mode tests

- deterministic mode ordering independent of insertion order;
- \(m=0\) advanced once;
- both orderings for \((-m,m)\to0\);
- transform orthonormality and round trip;
- raising/lowering/Laplacian;
- conjugate-mode lookup;
- exact Gaunt normalization and selection rules;
- padded products versus oracle;
- intentional aliasing test.

### 9.3 Radial/SBP tests

- polynomial exactness;
- \(HD+D^TH=B\);
- boundary closure order;
- dissipation nonpositivity;
- mapped-grid convergence;
- SAT stability on a representative characteristic system;
- multiblock interface tests if added.

### 9.4 Time tests

- autonomous RK4 order four;
- forced ODE with nonzero \(S''\) order four;
- deliberate legacy endpoint-average order two;
- coupled source-generating ODE order four;
- source JVP against centered finite-difference directional derivatives;
- \(\ddot U=L(\dot U)\) for randomized linear states;
- delayed activation event;
- cache generation invalidation.

### 9.5 PDE manufactured solutions

Implement manufactured forcing for:

1. homogeneous/forced Teukolsky system;
2. each reconstruction equation;
3. full reconstruction chain;
4. synthetic compact source;
5. full coupled first-order/reconstruction/second-order system;
6. coupled angular modes with known modal product.

Separate spatial and temporal order studies.

### 9.6 Physical and regression tests

- Schwarzschild limit;
- moderate-spin smooth pulse versus independent CPU oracle;
- comparison with legacy linear and reconstruction output;
- corrected versus legacy source difference isolated to the known term for identical fields;
- amplitude scaling \(F^{(1)}\sim A\), source and \(F^{(2)}\sim A^2\);
- future-null-infinity waveform normalization;
- reduction/reconstruction residual convergence;
- restart versus uninterrupted evolution;
- deterministic repeatability;
- Kokkos CPU/GPU backend parity.

Legacy second-order output is not the truth until the coefficient and temporal-staging differences are explicitly accounted for.

---

## 10. Milestone execution plan

### M0 - bootstrap and preserve evidence

- initialize repository and local branches/worktrees;
- import reference bundle with checksums;
- establish CMake, Kokkos, compiler warnings, sanitizers, formatting, CTest, and state docs;
- run the Python audit in CI;
- add one intentional failing scientific test, then make it pass.

**Exit gate:** clean CPU configure/build/test.

### M1 - background, metadata, signed modes, angular operators

- coordinate and background structs;
- field metadata;
- mode registry/pair table;
- spin-weighted transforms and GHP angular actions;
- Gaunt oracle.

**Exit gate:** all point, mode, angular, and product-oracle tests pass.

### M2 - radial SBP and homogeneous first-order evolution

- D4-2 SBP and dissipation;
- characteristic boundary infrastructure;
- \((P,Q,F)\) evolution;
- both free and constrained \(Q\) paths;
- manufactured and pulse tests.

**Exit gate:** stable long-time smooth evolution and designed convergence.

### M3 - seven-field metric reconstruction

- stage-local dependency chain;
- independent residuals;
- manufactured solutions;
- Kokkos/oracle comparison.

**Exit gate:** every reconstruction equation and residual converges.

### M4 - corrected compact quadratic source

- named source families from CSV;
- all ordered mode pairs;
- operator-form CPU oracle;
- production Kokkos source;
- padded angular products;
- source JVP/Jet path;
- factor regression.

**Exit gate:** source values and tangents agree with the independent oracle; amplitude-squared and Gaunt tests pass.

### M5 - coupled driven second-order vertical slice

- common-stage RK4 across all systems;
- second-order \((P,Q,F)\);
- delayed activation event;
- source-family output;
- restart.

**Exit gate:** full coupled manufactured solution is fourth order in time; deliberate legacy path is second order.

### M6 - moderate-spin regression and publication-quality validation

- reproduce smooth linear/reconstruction cases;
- compare corrected source behavior;
- verify future-null-infinity extraction;
- document every discrepancy;
- backend parity and sanitizer runs.

**Exit gate:** validation matrix complete for moderate spin.

### M7 - extremal/near-extremal diagnostics

- exact-extremal coefficient tests;
- surface-gravity and \(\kappa v\) metadata;
- high-order transverse derivatives;
- corotating/demodulated modes;
- regular horizon tetrad or infalling-observer diagnostics;
- perturbative validity ratios;
- source-family growth measurements.

**Exit gate:** growth fits stable under at least three resolutions and multiple dissipation strengths for a controlled test case.

### M8 - resolution and performance extensions

- D6-3 SBP;
- horizon-clustering map or two-block radial grid as evidence requires;
- layout/kernel fusion tuning;
- available GPU backends;
- optional radial MPI.

**Exit gate:** reproducible performance report with correctness deltas.

---

## 11. Exact-extremal and near-extremal science protocol

Treat exact extremality and near extremality as separate configurations.

For exact extremality, output at the horizon:

\[
\partial_R^nF^{(1)},
\qquad
\partial_R^nF^{(2)},
\qquad n=0,1,2,3,4\text{ and higher as resolved}.
\]

For near extremality compute

\[
\kappa=\frac{r_+-r_-}{2(r_+^2+a^2)}
\]

and record both \(v\) and \(\kappa v\). Test sequences such as

\[
a/M=0.9,0.99,0.999,0.9999,\ldots
\]

as resolution allows.

Output:

- raw and horizon-corotating/demodulated modes;
- every quadratic source family;
- modal projections and radial profiles;
- reduction/reconstruction residuals;
- source and second-order amplitude scaling;
- regular-tetrad/observer quantities;
- perturbative ratios.

Do not assume the second-order exponent is twice the first-order exponent. Measure source-family scaling and the driven response separately. Require growth-law stability under changes in resolution, dissipation, radial map, extraction location, start time, and fitting window.

---

## 12. I/O and reproducibility

Every output/restart file must contain:

- git commit hash and dirty status;
- build type, compiler, Kokkos version, backend, and device;
- complete input configuration;
- equation-specification version/hash;
- mode registry and pair table;
- \(M,a,L,r_\pm,\kappa\);
- radial grid/map and SBP order;
- angular bandlimit/padding;
- integrator and time step;
- dissipation/SAT parameters;
- source activation convention;
- tetrad/gauge convention;
- checkpoint time and state generation.

Use atomic checkpoint writes and verify restart equivalence.

---

## 13. Completion criteria

Do not report the task complete until all of the following are true:

1. the 94-check symbolic audit passes in the repository;
2. the corrected source coefficient is locked by a failing legacy-value regression;
3. the full first-order/reconstruction/source/second-order vertical slice runs;
4. the driven coupled system shows fourth-order temporal convergence;
5. the angular production product agrees with an exact Gaunt oracle;
6. the SBP operator satisfies its identity and designed convergence;
7. reconstruction and reduction residuals converge;
8. CPU oracle and available Kokkos backends agree within justified tolerances;
9. amplitude-squared scaling is demonstrated;
10. restart and deterministic mode ordering are verified;
11. exact-extremal and near-extremal diagnostics are implemented separately;
12. gauge/tetrad status is documented for every scientific output;
13. performance measurements include numerical deltas and reproducible commands;
14. all local milestone commits are clean and the final test suite passes.

At completion, leave:

- a concise `docs/FINAL_REPORT.md` summarizing what works;
- exact build/test/run commands;
- a validation table with evidence and tolerances;
- a list of unavailable backends or remaining scientific limitations;
- representative CPU and GPU benchmark results;
- at least one complete example configuration for moderate spin and one for an extremal/near-extremal diagnostic run.

Do not conceal uncertainty. Clearly distinguish implemented-and-tested results, analytically expected behavior, and future extensions.
