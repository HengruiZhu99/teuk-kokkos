# External review handoff: spin `-2` stability and spin `+2` companion

> **Remediation note (later on 2026-08-12).** This handoff describes the
> pre-remediation anchor and is not current mathematical authority. Subsequent
> work corrects the complete spin+2 source normalization, separates raw
> `S0/R^6` from evolved `S0/R^7`, versions checkpoint physical provenance,
> and adds constrained endpoint extraction. Use the final remediation report
> and the dated errata in the formalism/source documents for current claims.

Date: 2026-08-12 (America/New_York)

Scientific-code head before this report: `8a2de93`

Review range: `b48b20c..8a2de93` (28 commits, 99 files, about 26,000
inserted lines). The report commit itself contains no scientific-code change.
The scientific tree was clean before this report was added.

## 1. Executive status

The existing spin `-2` first/second-order solver remains the only production
runtime. It still passes the complete Serial suite. The spin `+2` work has
closed the equation, normalization, compact source, angular projection,
standalone RK/replay/output, linear-curvature, TSI, QNM, and source-algebra
gates. It has **not** crossed the production-integration boundary because the
same-stage local Route-B curvature provider cannot yet produce all six
`Z0,Z1` value/time-derivative fields at scri with a globally qualified
binary64 endpoint method.

`teuk_solver` deliberately rejects every `plus2.enabled=true` configuration
before writing output or allocating pipeline state:

```text
plus2 mode '<mode>' is not production integrated; use plus2.enabled=false and
plus2.mode=disabled
```

This is fail-closed behavior, not a missing parser. Do not remove it until the
promotion gates in section 9 are met.

Current checked-in Serial evidence at scientific head `8a2de93`:

- `372/372` unit tests pass;
- audit-enabled CTest has `19/19` targets;
- Kokkos submodule is `3ec81abe1816109f6f62ac48cef41921f91a4d00`
  (Kokkos 5.1.0);
- `git diff --check` is clean.

The latest full exact-commit review of the closed Route-B graph was performed
on `40f14a3`: a clean-first Serial build, `372/372` unit tests, and `19/19`
audit CTests passed. Commit `8a2de93` adds only the independently checked
full-coordinate-Weyl `Psi1` oracle extension and a fail-closed blocker
document; its focused build and `3/3` ordinary CTests also passed. The oracle
extension is a material test-code change, not a documentation-only commit.

The Intel Arc B580 is not current qualification evidence for this head.
SYCL compilation/linking succeeded for recent slices, but local-session
runtime probes later failed at the first tiny allocation with Level Zero
out-of-device-memory and device-lost errors after Xe job/VM timeouts. These
device failures are local operational observations, not checked-in artifacts.
No device reset was attempted.
Earlier exact-head B580 evidence is listed in section 8, but it must not be
extended to the current 28-commit series.

## 2. Reproducing the checked-in evidence

The external reviewer does not need any campaign outputs to run the source
and fixture tests. From the repository root:

```bash
python3 -m venv /tmp/teuk-review-venv
/tmp/teuk-review-venv/bin/pip install -r requirements.txt

cmake --preset serial \
  -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-review-venv/bin/python
cmake --build --preset serial --parallel 4
./build/serial/teuk_tests
ctest --preset serial --output-on-failure
git diff --check
```

Important generated evidence is committed as fixture headers together with
deterministic `--check` generators. CTest runs these generators and fails on
byte drift. Examples include:

- `tests/plus2_tsi_schwarzschild_t0_fixtures.hpp`;
- `tests/plus2_tsi_kerr_separated_fixtures.hpp`;
- `tests/plus2_tsi_kerr_t0_fixtures.hpp`;
- `tests/plus2_qnm_schwarzschild_fixtures.hpp`;
- `tests/routeb_teukolsky_primary_fixture.hpp`;
- `tests/routeb_reconstruction_fixture.hpp`;
- `tests/routeb_angular_jet_fixture.hpp`.

The near-extremal `T=200M` campaign under `run/` is intentionally ignored and
will not be present in a clone. Section 7 embeds the configuration, provenance,
diagnostic values, and scientific interpretation needed to review that symptom.
Its checkpoints, waveforms, stability analyzer/report, plots, probe output,
and checksum products are all local-only. Likewise, `build/` binaries and test
logs are not source evidence.

The root `MANIFEST.txt`, `AUDIT_MANIFEST.json`, and `SHA256SUMS` belong to an
older spin `-2` audit bundle. They are not a current-repository integrity
manifest; in particular, the present `SHA256SUMS` check fails for changed
`README.md` and `requirements.txt`. Historical test counts in older reports
such as `IMPLEMENTATION_STATUS.md`, `POST_AUDIT_REMEDIATION.md`, and
`FOUR_WEYL_FIELD_VALIDATION.md` are commit-scoped, not current totals. The
checked-in remediation snapshot under `audits/2026-08-11/` is similarly
historical evidence.

## 3. Convention map that must not drift

The authoritative map is
`docs/PLUS2_FIELD_AND_TETRAD_CONVENTIONS.md`. The essential points are:

- signature `+---`, fixed rotated-Kinnersley code tetrad, first-order ORG,
  and `gamma=0`;
- perturbative coefficients are not silently divided by factorials;
- signed sharp is always `X_m^sharp = conjugate(X_{-m})`, never same-mode
  conjugation;
- the spin `-2` stored regular field satisfies `Psi4 = R Z_minus`;
- the spin `+2` stored regular field satisfies
  `Psi0 = R^5 Z_plus/(L^2-i a R cos(theta))^4`;
- the compact source uses
  `Z0_source = Psi0/R^5 = Z_plus/(L^2-i a R cos(theta))^4` and
  `Z1 = Psi1/R^4`; `Z0_source` and evolved `Z_plus` are different typed
  quantities;
- raw second-order extreme Weyl coefficients are not called gauge invariant;
- the passive `Psi0^(2)` companion is not the naive nonlinear relation
  `D_TS[Psi4^(2)]` and is not the reduced Spiers effective-Einstein-source
  variable.

The compact spin `+2` source was obtained by priming the ungauge-fixed raw
Campanelli-Lousto/Loutrel equation and only then specializing to this ORG and
tetrad. It was not obtained by priming the already ORG-specialized spin `-2`
ledger.

## 4. Present code map

### 4.1 Qualified spin `-2` production runtime

The production path includes:

- signed-mode/angular transforms, Gauss-Legendre collocation, exact Gaunt
  products, and dealiased retained-band projection;
- compactified Teukolsky first-order reduction for `P,Q,Psi`;
- seven-field ORG reconstruction in fixed triangular pass order;
- corrected 51-term second-order spin `-2` source and common-stage tangents;
- common-stage RK4, source-event splitting, checkpoint/restart, diagnostics,
  endpoint modal waveforms, and four-field-format contracts;
- radial D4-2, D8-4, and D10-5 choices, with strict checkpoint/config
  provenance and scheme-dependent dissipation timestep guards.

No existing spin `-2` default is silently changed by the new spin `+2` code.
Disabled `plus2` storage launches and allocates nothing, and the standalone
three-state tests prove one-way/no-feedback structure.

### 4.2 Spin `+2` mathematical and pointwise foundation

The following are implemented and independently tested:

- homogeneous compact spin `+2` equation and endpoint-regular field scaling:
  `include/teuk/plus2_field.hpp`, `include/teuk/plus2_linear_spatial.hpp`;
- exact same-tetrad linear `T0[h]` operator with explicit same-stage
  `h,h_T,h_TT`: `include/teuk/plus2_linear_psi0.hpp`;
- rotating coordinate-Riemann/full-Weyl automatic-differentiation oracles,
  including the ORG tetrad perturbation;
- the 51-row compact raw source ledger and Jet path:
  `PLUS2_SOURCE_TERM_LEDGER.csv`, `include/teuk/plus2_source.hpp`;
- the explicit fourteen-row dependency ledger and fail-closed proposal:
  `PLUS2_SOURCE_INPUT_MANIFEST.csv` and
  `plus2_equation_spec_proposal.yaml`; the YAML remains a proposal, not a
  production specification;
- fourteen primitive rows, signed sharp construction, and background
  derivative slots: `include/teuk/plus2_source_primitives.hpp`;
- symbolic gates:
  `verify_plus2_formalism_gate.py`,
  `verify_plus2_compact_source.py`, and
  `verify_plus2_bianchi_closure.py`.

Two primary-source sign defects are explicitly regression-tested rather than
silently copied:

1. the displayed linear `Psi0` reconstruction signs in Loutrel et al. are
   inconsistent with the paper's own Ricci identity and Campanelli-Lousto;
2. the separately displayed `Psi1` representation reverses the
   `alphabar/pibar` epsilon-bracket signs relative to its own `riem-5`
   identity.

The solved Ricci identities are the authority. The independent full-Weyl
`Psi1` coordinate oracle includes Ricci trace subtraction and

```text
l1 = -1/2 h_ll n,   n1 = 0,
m1 = -h_lm n + 1/2 h_mm mbar.
```

At spins `a={0.63,0.91,0.999,-0.74}`, it agrees with the solved NP expression
at `1.39e-17`--`1.69e-16`; the wrong printed signs differ by
`4.46e-2`--`1.08e-1`.

### 4.3 Concrete standalone nonlinear source graph

The source graph is no longer only point algebra. Standalone components now
provide:

- D10-5, same-generation construction of the twelve non-curvature primitives,
  seven metric J/K derivative pairs, and three value-only Q inputs:
  `plus2_source_primitive_spatial.hpp`;
- exact ownership separation: transported `Z0,Z1` and four transported J/K
  pairs are not overwritten by the non-curvature producer;
- deterministic signed ordered-pair evaluation and per-target family sums;
- retained-band projection followed by concrete `thorn_5 J`, `eth_6 K`, and
  coordinate forcing: `plus2_source_outer_spatial.hpp`;
- typed common-stage live composition and immutable source-activation state:
  `plus2_live_source_composition.hpp`.

The full rotating source angular graph has a checked-in convergence test. For
one `a/M=0.73` fixture, forcing RMS errors at retained
`ell_max={4,6,8,10}` were

```text
4.40805e-2, 4.28692e-3, 1.86890e-4, 8.34713e-6.
```

At fixed `ell_max=10`, quadrature errors against `Ntheta=36` at
`Ntheta={12,18,24}` were

```text
1.64047e-6, 4.20101e-13, 1.09348e-15.
```

This proves the standalone retained graph and quadrature convergence. It does
not prove a live sourced `Psi0^(2)` residual because the local curvature
provider is still missing.

### 4.4 Companion, replay, checkpoints, and output contracts

Standalone code implements:

- passive spin `+2` `P,Q,Z` storage and homogeneous spatial RHS;
- allocation-free common-stage two-state and three-state RK4 seams;
- no-feedback concurrent and deterministic replay orchestration;
- versioned, checksummed companion checkpoints with exact representation,
  scientific-config, registry, and Git provenance;
- four-Weyl packing/output metadata with raw and regularized scalings.

The three-state seam can advance `primary -> curvature provider -> companion`
at identical RK stages, but its Route-A specialization is validation-only.
There is no production `solver_driver` call site for these components.

### 4.5 Route A is retired from rotating production

`plus2_bianchi_transport.hpp` remains useful validation machinery. It is not
a production curvature provider. For rotating Kerr, the pure-radial principal
symbol has a nontrivial Jordan block, so the transport is weakly rather than
strongly hyperbolic. A boundary SAT choice cannot repair that bulk defect.

The corrected Bianchi-5 compact term is

```text
Delta_5 Z0 = eth_4 Z1 - R mu0 Z0 - 4 R tau0 Z1 + 3 Sigma psi20,
```

not `3 Sigma H`. Tests keep the old nonlinear substitution as a negative
regression.

The associated validation-only contracts are
`include/teuk/plus2_transported_curvature.hpp`,
`include/teuk/plus2_curvature_initialization.hpp`, and
`include/teuk/plus2_bianchi_transport_checkpoint.hpp`. They must not be
mistaken for a production call path.

### 4.6 Route-B diagnostic derivative graph

Route B constructs local metric curvature from one same-stage primary and
reconstruction graph. The checked-in, standalone zero-dissipation
`FreeDamped` diagnostic path now contains:

1. exact-rational, independently generated nine-point direct `D1..D4`
   stencils in `include/teuk/routeb_fornberg_weights.hpp` and
   `include/teuk/routeb_fornberg.hpp`, with normalized algebra in
   `include/teuk/routeb_radial_taylor_jet.hpp`;
2. a level-interleavable primary `P,Q,Psi` tower `h0..h4`;
3. an enforced pass1/pass2/pass3 reconstruction tower for
   `G,Lambda,H,B,Pi,C,U`;
4. a signed-mode angular coordinator that applies only pure modal
   raise/lower coefficient-wise, then assembles the full Kerr GHP denominator
   and connection terms through radial-jet products/quotients;
5. projection and generation/level/pass stamps reduced from every source
   theta node, including sharp partners;
6. independent 90-digit fixtures covering all ten fields, both signed modes,
   all levels, and scri/interior/horizon.

All non-exact `h4` cells pass two ratios greater than 15 on the frozen
`N={9,17,33}` pre-roundoff window. `U` at scri is structurally grid-independent
and is instead gated below `2e-15` at its `~1e-17` roundoff floor.

This graph is diagnostic, not production. At `N=65`, all fields show
roundoff amplification; representative `N33/N65` error ratios are below one.
The full table is in `docs/PLUS2_ROUTE_B_ANGULAR_JET_COORDINATOR.md`.

## 5. Exact current blocker: six-field local curvature at scri

The desired typed adapter contains

```text
Z0, Z1, Z0_T, Z1_T, Z0_TT, Z1_TT
```

at one generation. Interior and horizon values are algebraically available.
At rotating scri, the approved tower stores normalized radial degree
`d(h_k)=4-k`. The exact operator ledger shows that the second tangents need
the first radial coefficient `h4[1]`, while the graph publishes only `h4[0]`.
The requirement is real because rotating `eth` contains an unsuppressed time
component. A simple `V=U/mu0` channel contains

```text
Z0_TT numerator coefficient  ~ (1/2) E0^2 d_R V4,
Z1_TT numerator coefficient  ~ (1/4) E0 A0 d_R V4,
E0 = -i a sin(theta)/(sqrt(2) L^2),  A0 = 2.
```

No `D_R^2 h4`, `h5`, or higher angular derivative is required.

### 5.1 Discarded post-differentiation remedy

A direct nine-point `D1` of the projected `h4` profile was implemented in an
isolated worktree and then completely reverted. It was formally fourth order
but had no single globally green binary64 refinement window:

- `N=9,17,33`: `B,m=+2,scri` ratios `12.7698,40.8938` and
  `U,m=+2,scri` ratios `14.8293,44.8144`;
- `N=13,25,49`: `Pi,m=+2,scri` first ratio `11.7956`, while several horizon
  cells were already roundoff-red at `N=49`.

No code from this prototype is in `main`.

### 5.2 Degree-five audit

Carrying degree `5-k` is algebraically sufficient, but the minimal
nine-point endpoint `D5` has dimensionless `L1=2218.67`. At the production
`nr=513` spacing its normalized input-roundoff bound is about
`0.18 ||f||_infinity`, approximately 136 times more sensitive than normalized
`D4`. An eleven-point degree-ten-exact `D5` is about 6.8 times worse again.
Degree five is therefore not the recommended production remedy.

### 5.3 Current promising route: peeling-constrained coefficient extraction

The next route under investigation forms cancellation-safe `Psi0/Psi1`
numerator profiles at nonzero near-scri nodes and extracts the required
`R^2` or `R^1` coefficient with fixed rational moment stencils. A naive
four-node quotient extrapolation `[4,-6,4,-1]` is rejected because it amplifies
peeling residuals as

```text
q0 contamination = (415/144) f0(0)/h^2 + (25/12) f0'(0)/h,
q1 contamination = (25/12) f1(0)/h.
```

The correct candidate needs at least six nonzero nodes for `q0` and five for
`q1`, with exact moment conditions that annihilate the forbidden constant and
linear bases while **separately measuring and convergence-gating**

```text
f0(0)=0,  f0'(0)=0,  f1(0)=0
```

at each time level and signed mode. The fixed extraction weights, norms,
checksums, coordinate-Weyl oracle, production-resolution conditioning, and
two-ratio endpoint convergence are not yet checked in. Until they are, no
six-field adapter or curvature certificate is valid.

## 6. Fourth-order spatial/RK status

The runtime supports `d4-2`, `d8-4`, and `d10-5`. D10-5 was added because the
complete reconstruction/source graph contains depth-two radial derivative
chains. D8-4 loses one endpoint order under that composition. D10-5 has a
fifth-order boundary closure and restores at least fourth order after the
second derivative.

Checked-in evidence for D10-5 includes:

- exact SBP identity and exact-rational coefficient provenance from
  `scicompuu/sbplib` commit `e64d8c6`;
- global degree-five and interior degree-ten exactness;
- negative-semidefinite compatible sixth-difference dissipation;
- zero-dissipation full-graph endpoint ratios above 15;
- a resolvable nonzero-dissipation manufactured nested derivative with ratios
  `32/32` and differentiated endpoint ratios `16/16`;
- fixed-space common-stage RK4 ratios near 16;
- scheme-dependent explicit-dissipation timestep guards.

The old `T=200M`, 400,000-step configuration is stable under its D4-2 guard
but would be rejected for D10-5 with `epsilon=0.005`. The checked runtime test
requires roughly 1.6 million steps for the corresponding guarded D10-5
selection. The Route-B Taylor graph is presently qualified only for
`FreeDamped` and **zero dissipation**. `StageConstrained` and production
nonzero dissipation remain hard gates.

## 7. Near-extremal `T=200M` late-time symptom

### 7.1 Run provenance and configuration

This was a successful spin `-2` SYCL/B580 campaign, not a spin `+2` run.

```text
source head              c7e3904b5e9a2d1e3aacc95d3b5a096b4a51ce23
solver SHA-256           8ebdeb705bb37ce476899587c0da81b9f97cbd2e72961aedb05ac4d8204ceb54
compiler                 Intel oneAPI DPC++/C++ 2025.3.2
device                   Intel Arc B580, Level Zero V2, driver 1.15.38308+1
mass, spin, L            1, 0.999, 1
nr, ntheta               513, 16
ellmax first/second      8 / 8
parents / daughters      m={-2,+2} / {-4,0,+4}
final time / steps       200 / 400000
dt                       0.0005
radial scheme            D4-2
reduction                FreeDamped, gamma=0.1
dissipation              0.005
source activation        T=9.68611053782132
```

The initial field was a normalized `C-infinity` compact bump supported on
`0.30 R_H < R < 0.50 R_H`, amplitude `1e-4`, with zero coordinate-time
derivative and an explicit sharp partner. The source was activated when the
latest inward characteristic from the scri-side support edge reached the
horizon. Zero reconstruction data were not constraint solved, so this is a
causal development experiment, not proven physical second-order initial data.

The solver exited normally after 400,000 steps. Final checkpoint metadata:

```text
step                     400000
time                     199.99999999872657
state bytes              8,536,320
FNV-1a64 metadata        15550603572236078889
FNV-1a64 recomputed      15550603572236078889
evolution wall time      5190.96 s
total wall time          5398.18 s
max host RSS             190384 KiB
```

### 7.2 Observed growth

Final volume diagnostics were

```text
Psi first RMS            8.310767805e-4
Psi second RMS           5.347670999e-5
constraint first RMS     3.051790275e-4
constraint second RMS    6.087864068e-5
C2/Psi2                  1.13841
```

The second-order RMS was nearly flat from `T=50M` to `100M`, crossed twice
that plateau near `T=110.1M`, and continued growing. The fractional growth
slowed late:

```text
Psi2 log slope 150--175M  0.02472/M
Psi2 log slope 175--200M  0.01271/M
horizon slope 150--175M   0.02426/M
horizon slope 175--200M   0.01102/M
```

Source and forcing peaked together at `T=176.45M`, with correlation
`0.999973` over `100--200M`. After the source peak, the source fell to 89% of
peak while the field RMS grew another 34.4%, consistent with propagation lag
or a driven response but not proof of physicality.

Selected final endpoint amplitudes (`scri / horizon`) were

```text
order1 ell2 m=+/-2       1.980313012e-5 / 2.266221905e-2
order2 ell4 m=+/-4       4.555840416e-7 / 1.971619740e-3
order2 ell2 m=0          6.233861837e-12 / 4.701670879e-4
```

### 7.3 Radial diagnosis

The final second-`Psi` horizon-64 high-frequency indicators, normalized to
one for a pure Nyquist sequence, were

```text
alternating projection   2.06930e-2
second difference        2.45019e-3
fourth difference        2.44258e-5
worst line alternating   3.18345e-2
```

This is not an obvious radial checkerboard. It is a smooth, sharply
horizon-localized layer:

```text
energy in final 8/16/32/64 points
                         55.7616% / 79.3420% / 94.3464% / 99.0926%
50% / 90% inward widths  7 / 25 points
```

Without radial refinement, resolution is unproved.

### 7.4 Angular diagnosis

This is the leading warning. The endpoint second-order top-band fraction
peaked near `T=111.05M` with 65.1% of horizon power in `ell>=7` while the
retained ceiling was `ell_max=8`. More importantly, the quadratic source hit
the ceiling before the response field grew:

```text
                         source ell>=7   forcing ell>=7
T=50M                    48.5%           53.2%
T=100M                   69.9%           77.3%
T=100M forcing ell=8                     62.2%
```

The inner `D` family, rather than `T`, carried most of the top-band load.
`Ntheta=16` satisfied the exact-product padding rule (`31 >= 8+8+8=24`), so
the quadratic products were dealiased before retained-band projection. That
Galerkin truncation is **not an angular filter**. The run used no exponential,
top-mode, or other dissipative angular filter.

### 7.5 KO-like dissipation and SAT

The run did include radial dissipation:

```text
Q_diss = -(epsilon/h) Htilde^-1 A^T A,
A = [-1,3,-3,1], epsilon=0.005.
```

It is D4-2 SBP-norm-compatible, negative semidefinite, sixth-derivative
KO-like damping applied to `P,Q,Psi` in both perturbative orders at every RK
stage. It is radial only. The physical SAT penalty was zero because the
compactified characteristic audit found no incoming propagating endpoint
mode and the natural continuum symmetrizer degenerates there.

A fixed-state checkpoint probe at `T=200M` found the production dissipation
contribution was only

```text
P / Q / Psi second-order RHS  0.082% / 0.146% / 0.067%
```

of the zero-dissipation RHS norm, and its SBP energy contribution was negative
at `T=150M` and `200M`. This weak radial damping cannot remove angular
`ell=8` pileup.

### 7.6 Scientific classification

The correct classification is:

> under-resolution warning; no demonstrated radial grid-scale runaway.

The run stayed finite, source/forcing remained coherent, fractional growth
slowed, signed pairs retained symmetry, and radial Nyquist measures stayed
small. It is not qualified as a physical late-time second-order waveform
because the angular spectrum occupied the truncation ceiling and the horizon
layer has not been radially converged.

Required independent matrix:

1. `nr={513,769,1025}`;
2. at fixed `ellmax2=8`, `Ntheta={16,20,24}`;
3. then `ellmax2={8,10,12}` with exactly padded angular nodes, and a separate
   increase of the first-order band;
4. `dt,dt/2,dt/4`;
5. angular-filter sensitivity only after unfiltered band convergence;
6. radial `epsilon={0,0.0025,0.005,0.01}` only after spatial/time baselines.

Filtering is a sensitivity diagnostic, not a substitute for convergence.

## 8. Backend status

Evidence must be associated with exact commits:

- the `T=200M` spin `-2` run and its pre-run `145/145` suite executed on the
  B580 at source head `c7e3904`;
- an earlier standalone spin `+2`/live-composition head `b48b20c` passed
  `262/262` on B580 Level Zero with strict no-allocation/no-fence tests;
- standalone linear `T0[h]` at a later slice passed a focused B580 suite and
  generated successful Level Zero kernel-launch traces;
- recent Route-B direct-jet sources compile and link under IntelLLVM/Kokkos
  SYCL, but current runtime qualification is absent because the device/driver
  entered an OODM/device-lost state on the first tiny allocation.

Do not claim CPU/GPU parity for current `main`. Recovery requires an external
driver/device reset and then a clean exact-head SYCL build, full unit/CTest,
bounded Level Zero trace, and production-kernel run. Host RSS is not VRAM;
no current VRAM peak measurement is available.

## 9. Completion-gate audit against `GOAL_MODE_PLUS2_COMPANION.md`

| Gate | Status | Evidence / missing item |
|---|---|---|
| Exact linear `Psi0^(1)` | Met standalone | Production point/spatial operator plus Schwarzschild and rotating full-coordinate oracles |
| Independent linear TSI | Partial | Schwarzschild and moderate-Kerr real-frequency normalized radial/angular/field fixtures pass, as does a Schwarzschild QNM interior fixture; moderate-spin Kerr QNM and endpoint extensions remain open |
| Regular `Z_plus` scaling | Met | Exact scri/horizon scaling and inverse contracts |
| Homogeneous equation symbolic/characteristic/QNM/convergence | Met standalone | Eq. 21b/22 audits, full-exterior convergence, Schwarzschild normalized QNM |
| Ungauge-fixed then ORG compact source | Met | Formalism gate, 51-row ledger, independent high-precision oracle |
| Every source term checked | Met | Machine-readable ledger, family closure, Jet/device parity |
| Passive sourced companion at same RK stages | Partial | Three-state seam and concrete source graph exist, but local six-field curvature provider is missing and runtime is unwired |
| Replay equals concurrent | Partial | Generic/standalone bitwise evidence exists; no concrete Route-B sourced trajectory |
| Four Weyl fields output | Partial | Strict standalone packer/metadata exists; solver does not emit all four |
| Linear/quadratic amplitude scaling | Met for standalone graphs | Linear curvature and quadratic source/companion tests |
| Sourced `+2` residual converges in time/radius/angle | Not met | Separate angular/radial/RK pieces pass; no complete concrete curvature-to-source-to-companion residual |
| Every available accelerator agrees | Not met for current head | Current B580 runtime unavailable/device-lost |
| Existing spin `-2` suite | Met | Current Serial `372/372` and ordinary production runtime preserved |
| Documentation distinguishes sourced companion and naive TSI | Met | Convention/formalism/source/replay/Route-B documents |

The goal is therefore **not complete** and `plus2.enabled=true` must continue
to fail closed.

## 10. Recommended external-review order

1. Read `GOAL_MODE_PLUS2_COMPANION.md` and
   `docs/PLUS2_FIELD_AND_TETRAD_CONVENTIONS.md`.
2. Audit the source authority in
   `PLUS2_SOURCE_TERM_LEDGER.csv`,
   `docs/PLUS2_FORMALISM_GATE.md`, and
   `docs/PLUS2_SOURCE_COMPACTIFICATION.md`.
3. Run every generator/audit command in section 2.
4. Inspect `plus2_linear_psi0.hpp` against the independent coordinate-Weyl
   oracle; pay special attention to the two paper-sign regressions.
5. Review primitive ownership and sharp-mode provenance through
   `plus2_source_primitive_spatial.hpp`, `plus2_source_outer_spatial.hpp`, and
   `plus2_live_source_composition.hpp`.
6. Verify the runtime rejection in `solver_driver.hpp`; absence of a call site
   is intentional.
7. Review the Route-B graph in this order:
   `routeb_fornberg.hpp`, `routeb_teukolsky_primary_jet.hpp`,
   `routeb_reconstruction_jet.hpp`, `routeb_angular_jet_coordinator.hpp`.
8. Treat `N=65` Route-B evidence as a red conditioning probe, not convergence.
9. Re-derive the scri coefficient extraction and independently assess the
   proposed six-/five-node constrained moment stencils before any provider is
   merged.
10. Only after the provider closes should review proceed to concrete
    concurrent/replay evolution, sourced residual convergence, four-field
    runtime output, and exact-head GPU qualification.

## 11. Things an external reviewer should try to falsify

- a same-mode conjugation accidentally substituted for signed sharp lookup;
- a full Kerr GHP operator applied coefficient-wise instead of only its pure
  unit-sphere angular factor;
- stale generation/level/pass stamps resurrected by angular projection;
- hidden raw division by `R` at scri or a zero substituted for an unavailable
  limit;
- the wrong displayed `Psi0` or `Psi1` paper signs reintroduced;
- a Riemann-only or fixed-tetrad `Psi1` oracle missing Ricci/tetrad terms;
- D8-4 or repeated first derivatives advertised as overall fourth order on
  the depth-two graph;
- D10-5 used with a timestep that passes only the characteristic CFL and not
  the explicit dissipation guard;
- Galerkin truncation described as dissipative angular filtering;
- old B580 results generalized to current main;
- standalone replay/output tests described as production solver integration;
- constraint-compatible endpoint coefficient extraction described as proof
  that physical peeling constraints hold.

Any one of these should block production promotion.
