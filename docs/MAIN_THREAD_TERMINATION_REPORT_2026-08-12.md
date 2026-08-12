# Main-thread termination report: external-review remediation

Date: 2026-08-12 (America/New_York)

Scientific pre-report HEAD:
`4bd66b068c74c4439873c20f0e591b9138530eca`

Review anchor:
`43e9300080140e3df4d05affd67ffa41b5ddbe57`

Primary development base named by the controlling review:
`b48b20c5d8ceed0fb625eff9ed07a12bbcf1a0e1`

Final handoff commit: the Git commit containing this file. A repository-only
reviewer obtains its exact object name with `git rev-parse HEAD`; the pushed
identity was also reported to the owner after the push. A commit cannot contain
its own future SHA without changing that SHA.

## Post-handoff Milestone-1 termination addendum

After the original handoff, work began on physical checkpoint provenance.  The
owner subsequently stopped further implementation and transferred scientific
assessment to a remote reviewer.  This addendum records that later frozen state
without rewriting the evidence below, which remains the history of commit
`c841ef7d4ee5ed2df635e88d37f777e099646095` and its scientific parent.

The later candidate adds a version-4 companion format; bitwise binding of the
actual companion `M,a,L`, radial/theta coordinates, radial operator, reduction,
damping and dissipation; exact-step `dt` latching; a domain-separated SHA-256
primary checkpoint receipt derived from the validated serialized bytes; and a
physical multi-node checkpoint round trip.  Before the final source-authority
edit, the dirty development tree passed `396/396` Serial unit cases.  That
result is not exact-final-candidate evidence: no new full build or CTest was
launched after the stop instruction.

Milestone 1 remains open.  Independent read-only review identified these
unclosed boundaries:

1. public header-level checkpoint `detail` functions can still be called
   without the pipeline token and therefore can mint or consume a nominally
   pipeline-derived binding;
2. the source trajectory now distinguishes a concrete Route-B adapter from
   validation-only callbacks, but checkpoint-only authority is still a bare
   composition identity and does not bind all source registry/band/geometry
   semantics;
3. the verified primary receipt binds exact bytes, state count, time and step,
   but not the signed registry and angular-band metadata that the companion
   independently claims;
4. explicit progress-mismatch and raw-versus-bound codec boundary regressions
   remain incomplete; and
5. the last mechanical source-adapter completion received only an incremental
   Serial `teuk_tests` build, which passed; the binary and CTest were not run.

Accordingly this later commit is a preserved fail-closed research handoff, not
a completed checkpoint milestone.  `plus2.enabled=true` remains rejected and
the spin-minus-two production path remains the only enabled runtime path.

## Executive termination state

The external-review remediation is a coherent and extensively tested
**standalone validation** body, not a production spin `+2` implementation.
The exact scientific pre-report commit passed a fresh audit-enabled Serial
CTest `24/24` in `302.79 s`; its unit target passed `390/390`. The final handoff
commit changes documentation only and was not followed by another scientific
test campaign, per the owner's stop instruction.

The verdict is:

> **MERGEABLE STANDALONE VALIDATION**

The existing spin `-2` production equations were not changed in the
external-review remediation range. `plus2.enabled=true` remains rejected by
the solver before output creation or pipeline allocation. No tolerance was
weakened, no failing test was deleted, and historical claims were retained as
dated or commit-scoped statements with explicit errata.

## A. Provenance

### A.1 Git and dependency identity

At termination:

```text
branch                    main
remote                    origin
remote URL                git@github.com:HengruiZhu99/teuk-kokkos.git
scientific pre-report     4bd66b068c74c4439873c20f0e591b9138530eca
pre-report parent         f578062e741498fd079a549e3b33a8471f65e401
Kokkos submodule          3ec81abe1816109f6f62ac48cef41921f91a4d00
Kokkos version            5.1.0
review anchor             43e9300080140e3df4d05affd67ffa41b5ddbe57
primary development base  b48b20c5d8ceed0fb625eff9ed07a12bbcf1a0e1
```

The anchor-to-scientific-HEAD remediation is ten commits, 54 changed files,
8,515 insertions, and 278 deletions. The larger primary development range
`b48b20c..4bd66b0` is 39 commits, 135 changed files, 35,509 insertions, and
544 deletions. The ten remediation commits are, oldest first:

```text
5b5c356 Fix spin plus2 source normalization and provenance
15b1b7e Document external review remediation handoff
92465cd Add standalone Route-B curvature provider candidate
fee00bb Update repository-only review handoff
51c0823 Harden standalone Route-B curvature validation
d96e1e1 Update repository-only Route-B review handoff
a0d0e3c Bind Route-B curvature to live source stage
1d19a82 Validate Route-B sourced replay trajectory
f578062 Add coordinate authority for Route-B curvature
4bd66b0 Validate normalized moderate Kerr plus2 QNM
```

### A.2 Exact tested environment

The exact scientific pre-report commit was configured and built with:

```text
compiler       GCC 13.3.0
CMake          3.28.3
C++ standard   C++20
backend        Kokkos SERIAL only
Python         3.12.3
SymPy          1.14.0
PyYAML         6.0.3
NumPy          2.5.2
SciPy          1.18.0
mpmath         1.3.0
qnm            0.4.4
numba          0.67.0
```

The exact commands were:

```bash
cmake --preset serial \
  -DTEUK_ENABLE_SYMBOLIC_AUDIT=ON \
  -DPython3_EXECUTABLE=/tmp/teuk-external-review-20260812/bin/python
cmake --build --preset serial --parallel 2
ctest --preset serial --output-on-failure
```

The `/tmp` environment is not in Git. Its versions are recorded above so a
reviewer can reconstruct it from `requirements.txt`; `qnm==0.4.4` was installed
only to independently inspect the pinned reference, and the checked-in Kerr
QNM generator deliberately does not import it.

### A.3 Dirty-tree classification and final packaging

Before the Kerr-QNM scientific commit, the only dirty files were its generator,
generated header, C++ test, CMake registration, and honest-scope documentation.
They were committed coherently as `4bd66b0`. Before this termination report,
the tree was clean and `main` was one commit ahead of `origin/main`.

The final handoff commit contains only:

- this termination report;
- a pointer/status update in
  `docs/EXTERNAL_REVIEW_PROMPT_2026-08-12.md`;
- a pointer/status update in
  `docs/EXTERNAL_REVIEW_REMEDIATION_REPORT_2026-08-12.md`.

No build output, virtual environment, ignored `run/` data, checkpoint, waveform,
plot, GPU log, or temporary diagnostic is included. There were no unrelated
user changes to exclude.

## B. External-review findings, one by one

### B.1 Spin `+2` source normalization

**CONFIRMED AND REMEDIATED.** Independent tetrad algebra gives the raw
principal coefficient

```text
A_TT(raw) = R^2 C_T/(2D),
D = L^4+a^2 R^2 cos^2(theta).
```

With

```text
d_minus = L^2-i a R cos(theta),
Psi0 = W_plus Z_plus,
W_plus = R^5/d_minus^4 = R f,
f = R^4/d_minus^4,
```

the historical multiplier `N=2D/R^3` produces `C_T f`, not `C_T`. The correct
evolved normalization is

```text
N/f = 2D d_minus^4/R^7,
forcing_plus = 2D d_minus^4 (S0/R^7).
```

`tools/symbolic/verify_plus2_source_normalization.py` derives the coefficient,
rejects the old multiplier, and compares independent analytic fields. The C++
source tests cover host/device/Jet, Schwarzschild, rotating and near-extremal
Kerr, and scri/interior/horizon.

### B.2 `S0/R^6`, cancellation-safe `S0/R^7`, and finite scri forcing

**CONFIRMED AND REMEDIATED.** Raw `S0/R^6` is retained as a diagnostic. It is
not divided numerically by `R`. The evolved source uses

```text
S0/R^7 = A J_T + b J_R + C_J J
        + eth_6 K + R(-4 tau0+pibar0)K - 3 psi20 Q,

A   = 2M(2M-a^2 R/L^2)/D,
b   = -E/(2D),
E   = L^2-2MR+a^2R^2/L^2,
C_J = (5b-4rho0-rhobar0)/R
      + i a m/D - 3epsilon0+epsilonbar0.
```

The separately finite optical combination is

```text
(5b-4rho0-rhobar0)/R
 = i a cos(theta) E (3L^2+5 i a R cos(theta))/(2D^2).
```

Therefore regular inputs give `S0=O(R^7)` and finite, generally nonzero
evolved forcing at scri. Typed workspaces, normalization ledger, metadata, and
checkpoint provenance distinguish raw and evolved forms.

### B.3 Raw/output/source field distinctions

**CONFIRMED AND REMEDIATED.** Code and metadata maintain separate meanings for
raw `Psi0`, evolved `Z_plus`, source-normalized `Z0_source`, naive TSI output,
and effective Einstein-source variables. The governing scalings are in section
C. No variable is intentionally overloaded across these roles.

### B.4 Route-B endpoint extraction

**CONFIRMED AND REMEDIATED AS A STANDALONE COMPONENT.** Exact rational moment
solves produce fixed positive-node operators. Generators, exact-moment tests,
condition norms, arbitrary-precision comparisons, host/device parity, and
fixed `N=9,17,33` tests are checked in. `N=65` is retained as a red conditioning
probe, not selected away.

The original minimum six-node `q0` operator remains intact. A separate
seven-node `q0` operator is used only by the complete curvature graph because
the minimum operator exposed a disaggregated fixed-window ratio `13.807`.
Thresholds and windows were not changed.

### B.5 Six curvature fields and all eight derivative slots

**PARTIALLY CONFIRMED / PROVEN STANDALONE ONLY.** The standalone provider emits

```text
Z0, Z1, Z0_T, Z1_T, Z0_TT, Z1_TT
```

and the eight value/tangent slots

```text
Delta4 Z1, (Delta4 Z1)_T,
ethprime4 Z1, (ethprime4 Z1)_T,
Delta5 Z0, (Delta5 Z0)_T,
eth5 Z0, (eth5 Z0)_T.
```

Independent coordinate full-Weyl fixtures now cover scri, interior, exact
future compact horizon, three rotating spins including `a/M=.999` and negative
spin, both signed modes, and every slot. The derivative oracle applies physical
GHP operators to coordinate-Weyl scalars rather than production point algebra.
Wrong controls omit Delta rescaling or GHP angular connections. Recorded minima
are radial ratio `38.5335`, angular ratio `19.6569`; the largest discrepancy is
`7.77717e-6` on a magnitude-27 near-extremal horizon value.

This is not solver/runtime wiring, sourced residual convergence, or GPU
qualification.

### B.6 Peeling residuals

**CONFIRMED AND REMEDIATED STANDALONE.** Endpoint extraction algebra may
annihilate lower polynomial powers but never substitutes for physical peeling.
`f0(0)`, an independent endpoint derivative `f0'(0)`, and `f1(0)` are measured
and convergence-gated at every time level. Invalid residual/certificate data
fail before state mutation.

### B.7 Linear `Psi0` spatial path

**PARTIALLY CONFIRMED; STILL OPEN FOR PRODUCTION.** The ordinary-NP point and
spatial implementation has independent Schwarzschild and rotating full-coordinate
curvature oracles, signed-mode tests, device parity, and D8-4/D10-5 convergence
evidence. It remains labeled `validation-only-ordinary-np-v1` because it
evaluates individually singular terms and requires external scri authority.
A cancellation-safe regular production GHP endpoint implementation is absent.

### B.8 Checkpoint provenance and replay time

**CONFIRMED AND REMEDIATED STANDALONE.** Companion checkpoint v3 binds exact
`M,a,L`, full radial/theta coordinates, `dt`, reduction mode, damping,
dissipation, normalization version/name, primary checkpoint identity, bands,
modes, methods, Git/schema/layout/representation, step/time, and checksum.
NaN/Inf and every mismatch fail before device mutation. Legacy v1/v2 restore
is rejected, and `time=step*dt` plus restored accepted time are checked.

### B.9 Live-source global invalidation

**CONFIRMED AND REMEDIATED.** Validity is decided before normalization. One
stale required field clears every exposed primitive/pair/family/target/outer
derivative, raw `S0/R^6`, regular `S0/R^7`, forcing, and associated stamp at
that point. Tests inspect the complete exposed surface and readiness bit.

### B.10 Bianchi-5

**CONFIRMED AND REMEDIATED.** The correct linear background-curvature coupling
is

```text
Delta_5 Z0 = eth_4 Z1 - R mu0 Z0 - 4R tau0 Z1
             + 3 Sigma psi20,
```

with tangent `+3 Sigma_T psi20`. The former `+3 Sigma H` was a nonlinear
substitution into a linear identity and is retained only as a rejected
regression.

### B.11 Replay

**PARTIALLY CONFIRMED / PROVEN STANDALONE ONLY.** The Route-B physical replay
test now runs actual source-independent primary/reconstruction, curvature,
quadratic source, and passive companion through common RK stages. Concurrent
and replay trajectories agree bitwise on Serial; stage times, pointers,
generations, quadratic scaling, nonzero response, no feedback, and hot-path
instrumentation are checked. It remains manufactured, zero-dissipation,
same-backend validation without production `Psi4^(2)`, checkpoint-restored
campaign state, four-Weyl packing, residual convergence, or runtime wiring.

### B.12 QNM and TSI validation

**PARTIALLY CONFIRMED, WITH INTERIOR QNM GAP REMEDIATED.** Checked-in fixtures
now qualify:

- normalized Schwarzschild radial/angular/field TSI;
- moderate-Kerr real-frequency separated and field-level TSI;
- normalized Schwarzschild complex-frequency QNM at two interior points;
- normalized moderate-Kerr `a/M=.6` complex-frequency QNM at two interior
  points and both signed sectors.

The Kerr QNM reference is `qnm==0.4.4`,

```text
M omega = 0.4940447817813845 - 0.0837652021610416 i,
A_qnm   = -0.8546134005662634 + 0.15669038538845606 i.
```

Berens' constant is independently recovered as

```text
lambda_plus = A_qnm - 2 a m omega + (a omega)^2
            = -1.9549779674091456 + 0.32793056263873765 i.
```

The complex angular Wronskian closes; the solved eigenvalue differs from the
converted reference by `4.298732e-14`. Directly using `A_qnm` as
`lambda_plus` leaves Wronskian residual `11.28565`. Horizon-in radial and
angular Heun functions use unit local Frobenius leading coefficients. Ripley
height/azimuth, tetrad boost/spin phase, compact radius, raw/stored scaling,
and the lower-after-raise angular action yield compact point-operator residuals
near binary64 roundoff. Generator changes fall from `2.411062e-7` to
`8.617746e-9`.

QNM horizon/scri endpoints and an evolved QNM/companion comparison remain open.

### B.13 Fourth-order space/time treatment

**PARTIALLY CONFIRMED / PROVEN FOR STANDALONE GRAPHS.** RK4 is fourth order at
fixed space. D8-4 is fourth order for a single radial derivative but loses one
boundary order when derivative-containing source values are differentiated
again. D10-5 supplies boundary order five and closes the depth-two endpoint
budget; full zero-dissipation source ratios and fixed-space RK4 ratios pass.
Compatible dissipation is separately qualified on a resolvable manufactured
field. These do not constitute sourced four-field joint space/time production
convergence.

### B.14 `T=200M` symptom

**PARTIALLY CONFIRMED AS A FROZEN DIAGNOSTIC; SCIENTIFIC RESOLUTION STILL
OPEN.** The run completed, remained finite, and shows a smooth horizon-localized
second-order layer with substantial angular-ceiling occupation. It is not a
demonstrated radial Nyquist runaway. Section E records the exact local-only
evidence and required refinement.

## C. Mathematical conventions and results

The immutable conventions are:

```text
metric signature                  +---
gauge                             outgoing radiation gauge (ORG)
background tetrad convention      gamma=0
perturbation expansion             no factorial normalization
signed sharp                       X_m^sharp = conjugate(X_-m)
```

Same-`m` conjugation is never a replacement for signed sharp.

The field definitions are:

```text
Psi4 = R Z_minus,
Psi0 = R^5 Z_plus/(L^2-i a R cos(theta))^4,
Z0_source = Psi0/R^5
          = Z_plus/(L^2-i a R cos(theta))^4,
Z1 = Psi1/R^4.
```

Raw sourced `Psi0^(2)`, evolved `Z_plus`, `Z0_source`, naive TSI reconstruction,
and effective Einstein-source variables remain distinct types/concepts.

The corrected evolved source is

```text
forcing_plus = 2D (L^2-i a R cos(theta))^4 (S0/R^7),
D = L^4+a^2R^2cos^2(theta),
```

with cancellation-safe `S0/R^7` given in B.2.

### C.1 Endpoint extraction

On positive nodes `R_j=jh`, the minimum `q0=[R^2]f0` operator is

```text
29/6, -461/24, 31, -307/12, 65/6, -15/8.
```

It annihilates powers zero and one, extracts power two, and cancels powers
three through five. The `q1=[R]f1` operator is

```text
-77/12, 107/6, -39/2, 61/6, -25/12,
```

annihilating power zero, extracting power one, and cancelling powers two
through four. Their exact condition summaries `(L1,L2^2,Linf)` are
`(280/3,613067/288,31)` and `(56,60995/72,39/2)`.

The complete curvature graph uses a separate promoted seven-node `q0`:

```text
319/45, -3929/120, 389/6, -2545/36,
134/3, -1849/120, 203/90.
```

It extracts power two and annihilates powers `0,1,3,4,5,6`. It is separate so
the original authority is not silently rewritten. Its exact norms are
`L1=10696/45`, `L2^2=271316521/21600`, `Linf=2545/36`.

## D. Roadblocks and failed approaches

This history is promotion-critical; red results were not removed or converted
into looser gates.

### D.1 Source-normalization error

The original path multiplied the raw equation by `2D/R^3` after substituting
`Psi0=R f Z_plus`. That leaves `C_T f`, not `C_T`, and suppresses regular scri
forcing by `O(R^4)`. It was replaced by division by the complete field factor
and a direct `S0/R^7` expression. The old form remains a negative regression.

### D.2 Route-A rotating Jordan block

The intended Route-A plan transported `Z0,Z1` with Bianchi equations. In
rotating Kerr the pure-radial principal symbol has a repeated outward speed
but a nontrivial Jordan block because `eth_4 Z1` contains an unsuppressed time
component. The system is weakly, not strongly, hyperbolic for an admissible
spatial covector. A SAT/boundary choice cannot repair that bulk defect.
Route-A remains validation-only and was replaced as the production direction
by local algebraic Route B.

### D.3 Missing `h4[1]` at rotating scri

The original Route-B tower stores radial degree `d(h_k)=4-k`. Exact operator
counting shows rotating `Z0_TT` and `Z1_TT` require only—but genuinely require—
the first radial coefficient `h4[1]`; the tower exposed only `h4[0]`. A simple
`V=U/mu0` channel contains nonzero terms

```text
Z0_TT numerator ~ (1/2) E0^2 d_R V4,
Z1_TT numerator ~ (1/4) E0 A0 d_R V4,
E0=-i a sin(theta)/(sqrt(2)L^2), A0=2.
```

### D.4 Rejected direct `D1(h4)`

A direct nine-point derivative of projected `h4` was implemented and then
reverted. Frozen `N=9,17,33` failed:

```text
B,m=+2,scri  ratios 12.7698, 40.8938
U,m=+2,scri  ratios 14.8293, 44.8144
```

The predeclared alternate `N=13,25,49` repaired those cells but failed
`Pi,m=+2,scri` at `11.7956`; horizon cells were already roundoff-red (`P,m=-2`
ratio `4.42899`, `Q,m=+2` ratio `0.710481`). No per-cell window, aggregate norm,
or lower threshold was substituted. The prototype is absent from Git.

### D.5 Degree-five/direct high-derivative conditioning

Carrying degree `5-k` is algebraically sufficient, but a minimal nine-point
endpoint `D5` has dimensionless `L1=2218.67`. At `nr=513` its normalized
input-roundoff bound is about `0.18||f||_infinity`, roughly 136 times more
sensitive than normalized D4. An eleven-point degree-ten-exact D5 is another
6.8 times worse. It was not promoted.

### D.6 D8-4 nested order loss and source tangent/outer derivative

The complete D8-4 graph revealed endpoint ratios `7.71681` for the source
tangent and `7.42038` for final forcing, approximately third order, even while
direct fields were fourth order. The cause is differentiating values that
already contain boundary `O(h^4)` derivative error. RMS partially hides the
fixed number of lower-order boundary points. The failing evidence is retained
in `FULL_GRAPH_ORDER_GAP.md`.

### D.7 D10-5 repair, dissipation, and timestep restriction

D10-5 supplies degree-five boundary and degree-ten interior accuracy, exact SBP
identity, and compatible negative-semidefinite sixth-difference dissipation.
The zero-dissipation full graph and fixed-space RK4 gates pass. A manufactured
`epsilon=.006` test gives RHS/dissipation ratios `32/32` and a differentiated
endpoint ratio `16/16`.

The explicit dissipation guard is

```text
(dt/h) epsilon spectral_bound <= 0.8 * 2.785293563405282,
```

with bounds `64`, `1371`, `5500` for D4-2, D8-4, D10-5. This guards the pure
dissipative subsystem, not the complete combined spectrum.

The historical `T=200M`, 400,000-step setup has `dt/h=.26745` and
`epsilon=.005`. It passes D4-2 but fails D10-5 by about 2.4 in the unguarded
pure-dissipation estimate. The runtime rejects it before output/state creation;
the checked safe example uses 1.6 million steps. The old campaign cannot simply
be rerun after switching the stencil.

### D.8 `N=65` conditioning failures

Direct D4 initialization has large `h^-4` noise amplification. In the closed
Route-B reconstruction/angular graphs, `N=65` errors rise and many `N33/N65`
ratios fall below one. `N=9,17,33` is the frozen pre-roundoff promotion window;
`N=65` remains recorded as a red conditioning probe. It was never removed to
make a plot look convergent.

### D.9 Incorrect or non-independent test oracles repaired

Several tests exposed their own defects before they were accepted:

- a draft `Et12` compact-source expansion had the wrong coefficients and was
  rejected by the ordinary-NP coordinate route;
- an independent eighth-order centered derivative test used the wrong
  coefficient set; production matched the analytic field after the test oracle
  was repaired;
- a Python reconstruction fixture late-bound per-mode closures, giving
  identical signed outputs; a per-mode factory and signed distinction assertion
  replaced it;
- early angular Route-B code applied full Kerr GHP operators coefficient-wise,
  which is invalid because radial-dependent denominators mix Taylor orders;
  only pure raise/lower is coefficient-wise, followed by radial-jet GHP algebra;
- value-then-Fornberg recovery differentiated prior truncation error and was
  replaced by coefficient-wise angular action on stored jets;
- a horizon coordinate test incorrectly required a discrete one-sided endpoint
  value to equal a continuum oracle with an interior absolute tolerance. It was
  replaced—not loosened—by two-axis radial and angular convergence plus a
  measured remainder budget;
- a SYCL checkpoint test failed exact uniform-grid comparison because
  IntelLLVM reassociated arithmetic by one ULP. The repair was a narrowly
  derived ULP-stable predicate retaining strict storage/metadata checks and
  rejecting material nonuniformity.

### D.10 Literature sign inconsistencies

The displayed linear `Psi0` reconstruction in Loutrel et al. has signs
inconsistent with its own Ricci identity; the solved identity agrees with
Campanelli--Lousto. Independent Schwarzschild and rotating coordinate-Riemann
oracles distinguish the old signs.

The separately printed `Psi1` representation reverses the
`alphabar/pibar` epsilon bracket relative to its own `riem-5`. A full-coordinate
Weyl oracle, including Ricci trace subtraction and perturbed ORG tetrad, agrees
with the solved identity at `1.39e-17`--`1.69e-16`; wrong signs separate by
`4.46e-2`--`1.08e-1`.

### D.11 Incorrect Bianchi `3 Sigma H`

The linear Bianchi-5 background curvature is `3 Sigma psi20`, not
`3 Sigma H` where `H=Psi2^(1)/R^3`. The original symbolic gate had repeated
the same substitution and a test used zero Sigma, masking it. The API,
symbolic gate, and a nonzero Sigma/Sigma_T oracle were corrected.

### D.12 Erroneous Eq. 2.44 phase-free shortcut

The early `i/10` amplitude statement discarded the phase of complex
`C_in_prime`. The correct ratio is

```text
(i/10) C_in_prime/conjugate(C_in_prime).
```

The normalized TSI fixtures retain this phase and both signed sectors.

### D.13 QNM `A` versus `lambda_plus`

The new Kerr QNM work initially compared `qnm`'s angular `A` directly to
Berens' `lambda_plus`; its Wronskian residual was `11.28565`. Independent
convention conversion gives
`lambda_plus=A-2amomega+(aomega)^2`; the complex Wronskian then closes and the
independent root agrees within `4.30e-14`. The wrong convention remains a
negative control.

### D.14 Full-Kerr expression swell and AD/coordinate remedy

Direct full symbolic metric inversion/differentiation for rotating Kerr caused
unbounded expression growth. It was discarded. A standard-library second-order
dual/jet coordinate oracle instead varies inverse metric, Christoffels, Riemann,
Weyl, and the perturbed tetrad numerically at deterministic points. It provides
independent rotating `Psi0/Psi1` authority without importing production NP
algebra.

### D.15 Missing Mathematica/radial tables

The Berens supplement did not export a normalized radial value/derivative
table, and local Mathematica was unactivated. No factors were guessed. Later
independent SciPy DOP853 confluent-Heun/Frobenius generators closed
Schwarzschild and moderate-Kerr real-frequency fixtures and interior QNM
fixtures. They do not close QNM endpoints, evolved comparisons, or every
production field normalization.

### D.16 Scri extrapolation and rejected `/63`

The coordinate-Weyl scri oracle extrapolates already normalized scalars from
six positive nodes with `(6,-15,20,-15,6,-1)`, formally exact through degree
five. The observed fixed-window ratio was only `16.1168`, not the asymptotic
sixth-order factor 64. Applying a `/63` Richardson factor would therefore be
unjustified. The gate retains the `N=33` estimate with a conservative
fourth-order remainder. The provider ratio is `15.2389`; their largest
budgeted discrepancy `4.38026e-4` is reported, not hidden behind an arbitrary
absolute tolerance.

### D.17 B580 OODM/device-lost/wedged episodes

Earlier exact commits passed real Arc B580 Level Zero tests. Later sessions
encountered one narrow cross-backend absolute-tolerance mismatch, hidden
View-capture fences, transient `UR_RESULT_ERROR_DEVICE_LOST`, then a wedged Xe
driver with `sycl-ls` in uninterruptible `add_exec_queue`, and finally tiny
allocations returning `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`. Narrow harness
defects were repaired where proven, but no unauthorized driver/PCI/module reset
was attempted. The current exact HEAD is Serial-qualified only; old B580
binaries are not current-head evidence.

### D.18 Production/runtime blockers

Standalone headers and tests do not equal `SpatialPipeline` or solver wiring.
The production four-field trajectory, restored campaign checkpoint, output
packing, joint residual convergence, exact-head accelerator runtime, and
long-run resource/stability evidence are absent. `plus2.enabled` therefore
remains fail-closed.

## E. Frozen local `T=200M` symptom

**This section is a local diagnostic handoff. The campaign files are ignored
and are not reproducible from Git.**

The spin `-2` run used:

```text
source head              c7e3904b5e9a2d1e3aacc95d3b5a096b4a51ce23
solver SHA-256           8ebdeb705bb37ce476899587c0da81b9f97cbd2e72961aedb05ac4d8204ceb54
compiler/device          oneAPI 2025.3.2 / Arc B580 Level Zero V2
driver                   1.15.38308+1
M,a,L                    1,0.999,1
nr,Ntheta                513,16
ellmax first/second      8/8
parent/daughter m        {-2,+2}/{-4,0,+4}
time/steps/dt            200 / 400000 / 0.0005
radial scheme            D4-2
reduction                FreeDamped gamma=0.1
radial dissipation       epsilon=0.005
angular filter           none
source activation        T=9.68611053782132
```

It exited normally after 400,000 steps at
`T=199.99999999872657`; final checkpoint checksum metadata and recomputation
both gave `15550603572236078889`. Evolution wall time was `5190.96 s`, total
`5398.18 s`, max host RSS `190384 KiB`.

Final diagnostics:

```text
Psi1 RMS                 8.310767805e-4
Psi2 RMS                 5.347670999e-5
constraint1 RMS          3.051790275e-4
constraint2 RMS          6.087864068e-5
C2/Psi2                  1.13841
```

The second-order field was flat from `T=50M` to `100M`, crossed twice that
plateau near `T=110.1M`, and continued growing with slowing logarithmic slope:
`0.02472/M` over 150--175M and `0.01271/M` over 175--200M. Source/forcing
peaked near `176.45M` with correlation `0.999973` over 100--200M.

The final radial high-frequency indicators, normalized to one for pure
Nyquist, were:

```text
alternating projection  2.06930e-2
second difference       2.45019e-3
fourth difference       2.44258e-5
worst line alternating  3.18345e-2
```

The field is nevertheless strongly horizon localized: final 8/16/32/64
points contain `55.7616/79.3420/94.3464/99.0926%` of second-field energy.
This is a smooth unresolved boundary layer, not a demonstrated radial
checkerboard.

The angular warning is stronger. At `T=100M`, `ell>=7` contains 69.9% of
source and 77.3% of forcing power; `ell=8` alone contains 62.2% of forcing.
The endpoint top-band fraction peaked near 65.1%. `Ntheta=16` satisfied exact
product padding, so this is not unpadded quadratic aliasing, but Galerkin
truncation is not filtering. There was no angular filter.

Radial KO-like dissipation was already applied to P/Q/Psi in both orders at
every RK stage. Its final P/Q/Psi second-order RHS contributions were only
`0.082/0.146/0.067%`, and its SBP energy contribution was negative. It cannot
remove angular top-band pileup.

Classification: **under-resolution warning, not demonstrated radial Nyquist
runaway**. The source-normalization defect was in disabled spin `+2` code and
cannot explain this spin `-2` run.

Minimum follow-up matrix:

1. `nr={513,769,1025}`;
2. at `ellmax2=8`, `Ntheta={16,20,24}`;
3. `ellmax2={8,10,12}` with exact padding and a separate first-order-band
   increase;
4. `dt,dt/2,dt/4`;
5. only after those baselines, radial `epsilon={0,.0025,.005,.01}`;
6. angular filtering only as a sensitivity study after unfiltered band
   convergence.

## F. Test and qualification evidence

### F.1 Exact scientific pre-report commit

At exact `4bd66b068c74c4439873c20f0e591b9138530eca`:

```text
cmake configure          PASS, audit enabled, Python 3.12.3
cmake build              PASS, GCC 13.3.0, Kokkos Serial 5.1.0
teuk.unit                PASS, 390/390, 119.40 s under CTest
teuk.qnm                 PASS, 25.58 s
CTest                    PASS, 24/24, 302.79 s
Kerr QNM generation      PASS, 1.55 s
coordinate fixture       PASS, 54.72 s
compact source gate      PASS, 34.36 s
all generators/symbolic  PASS through the 24 CTest targets
git diff --check         PASS before scientific commit
```

CTest includes source normalization, formalism, compact source, Bianchi,
background derivatives, endpoint extraction exact/arbitrary precision,
Fornberg exact/arbitrary precision, Route-B primary/reconstruction/angular and
coordinate generators, TSI radial/angular/T0 generators, both QNM generators,
unit, production QNM, and config integration.

This is exact-commit Serial evidence. It is not SYCL runtime or B580 evidence.

### F.2 Pre-commit Kerr-QNM development evidence

Before `4bd66b0`, the generator reported:

```text
coarse-medium            2.411062e-7
medium-fine              8.617746e-9
qnm lambda error         4.298732e-14
wrong A Wronskian        1.128565e+1
direct unit              390/390
development CTest        24/24, 310.49 s
```

The exact-commit rerun above supersedes it for qualification.

### F.3 Historical and backend evidence

Older commits have real B580 Level Zero passes for selected standalone paths,
and the historical spin `-2` campaign ran on B580. Other later exact commits
were SYCL compile-only because the driver was wedged/OODM/device-lost. None is
current exact-HEAD accelerator qualification. Historical test counts in other
documents apply only to their named commits.

### F.4 Final handoff commit

The final commit adds only documentation. Lightweight `git diff --check` and
staged-scope inspection were performed before commit. Per the stop instruction,
the full scientific suite was not rerun after documentation packaging. The
scientific evidence boundary remains exact `4bd66b0`.

## G. Claim/evidence matrix

| Feature | Status | Repository evidence | Missing promotion evidence |
|---|---|---|---|
| Source normalization | PROVEN STANDALONE ONLY | symbolic derivation, old-form negative control, host/device/Jet/source tests | runtime production trajectory |
| Cancellation-safe `S0/R^7` | PROVEN STANDALONE ONLY | exact optical identity, random analytic and endpoint tests | production sourced campaign |
| Raw/evolved/source/output typing | PROVEN STANDALONE ONLY | ledgers, typed workspaces, checkpoint/output metadata | integrated output |
| Endpoint extractors | PROVEN | exact generators/moments/norms, high precision, N9/17/33 | production conditioning at campaign resolution |
| Six curvature fields | PROVEN STANDALONE ONLY | constrained provider, coordinate-Weyl scri/interior/horizon | runtime, residual, GPU |
| Eight derivative slots | PROVEN STANDALONE ONLY | physical coordinate-GHP oracle and wrong controls | runtime, residual, GPU |
| Peeling audits | PROVEN STANDALONE ONLY | independent residual and convergence gates | physical production initialization |
| Linear `Psi0` endpoint path | PARTIALLY PROVEN | ordinary-NP spatial + coordinate oracles | regular production GHP endpoint path |
| Checkpoint provenance | PROVEN STANDALONE ONLY | v3 fail-before-mutation tests | production four-field checkpoint |
| Live invalidation | PROVEN STANDALONE ONLY | full surface zero/stamp tests | integrated runtime |
| Bianchi-5 | PROVEN | primary-source derivation, symbolic/C++ regressions | none for formula; Route-A remains nonproduction |
| Same-stage source seam | PROVEN STANDALONE ONLY | one immutable tower generation, nonzero forcing | solver wiring |
| Concurrent/replay | PROVEN STANDALONE ONLY | bitwise source-bearing Route-B trajectory | production state/checkpoint/backend |
| Schwarzschild QNM | PROVEN STANDALONE ONLY | normalized interior fixture/operator | endpoints/evolved waveform |
| Moderate-Kerr real TSI | PROVEN STANDALONE ONLY | separated + field-level fixture | evolved waveform/endpoints |
| Moderate-Kerr complex QNM | PROVEN STANDALONE ONLY | independent complex Heun fixture, wrong convention control | endpoints/evolved waveform |
| Fourth-order RK4 | PROVEN STANDALONE ONLY | fixed-space common-stage ratios | joint production convergence |
| D10-5 nested radial order | PROVEN STANDALONE ONLY | SBP identity, full-graph endpoint gates | long-run combined-spectrum stability |
| Angular source graph | PROVEN STANDALONE ONLY | Galerkin/quadrature refinement and exact products | observable production refinement |
| Four-Weyl output | PARTIALLY PROVEN | standalone strict packer/metadata | solver output integration |
| Exact-head B580 | UNPROVEN | none at current head | clean build, real Level Zero run, trace, full tests |
| Long-run sourced stability | UNPROVEN | historical local spin-minus2 diagnosis only | refined production campaign |
| plus2 production | UNPROVEN and fail-closed | explicit solver rejection | every blocker below |

## H. Remaining blockers and minimum closure evidence

1. **Regular production linear `Psi0` endpoint path.** Implement a
   cancellation-safe GHP spatial operator with independent rotating
   coordinate-Weyl scri/interior/horizon comparison, fixed disaggregated
   convergence, signed partners, device/hot-path gates, and no external guessed
   quotient.

2. **QNM horizon/scri endpoints.** Supply independently normalized,
   horizon-regular and scri-regular Kerr QNM limits at fixed refinement windows,
   both signed sectors, with coordinate/tetrad and raw/stored checks. Interior
   Heun values do not close this.

3. **Evolved QNM/companion comparison.** Evolve normalized initial data through
   the same accepted RK stages and compare phase, damping, and normalization
   against the independent fixture over a resolved interval.

4. **Production spin-minus2 second order plus spin-plus2 common trajectory.**
   One common tableau must advance the actual production first/second spin-minus2
   graph, Route-B curvature/source, and passive spin-plus2 state with structural
   no-feedback and immutable activation.

5. **Four-Weyl integration.** Wire all four raw/regularized fields into solver
   output with one accepted timestamp, strict metadata, scri modal completeness,
   horizon non-flux warning, and restart continuity.

6. **Checkpoint-restored production replay.** Restore a real production
   checkpoint, fail-before-mutation on all provenance mismatches, continue from
   accepted time, and prove same-backend concurrent/replay equality through
   physical fields and outputs.

7. **Sourced four-field residual convergence.** Perform radial, angular-band/
   node, and timestep refinement on the complete sourced graph. Require
   endpoint-inclusive fourth order where claimed, convergent reconstruction/
   Teukolsky/curvature/source constraints, and observables below truncation
   budgets. Do not use fixed-space RK4 as joint convergence.

8. **Current exact-HEAD accelerator qualification.** On a recovered Arc B580,
   cleanly configure IntelLLVM/Kokkos SYCL, build exact HEAD, run full unit/CTest,
   show actual Level Zero selection and kernel trace, qualify noalloc/nofence
   paths, and report host/VRAM resources honestly.

9. **Long-run stability/resource qualification.** Run the selected exact
   numerical method through the target duration, track constraints, top-band
   power, endpoint layers, checkerboard measures, checkpoints, and resource
   bounds. The pure-dissipation guard is not a combined-spectrum proof.

10. **`T=200M` refinement campaign.** Complete the matrix in section E without
    filtering first. Only then test angular filters as sensitivity and radial
    dissipation strength.

11. **Solver/runtime integration.** Add typed config-to-pipeline wiring only
    after the scientific graph is qualified, preserving disabled zero-allocation
    behavior and checkpoint/output schemas.

12. **Final promotion audit.** Independently reproduce every derivation,
    generator, test, exact commit/backend run, and campaign result. Until all
    cells are proven, keep `plus2.enabled` rejected.

## I. Integrity warnings

- Ignored/local campaign artifacts, checkpoints, plots, logs, build trees, and
  `/tmp` environments are not in Git.
- Generated fixture headers are checked in only where explicitly present under
  `tests/`; each has a generator freshness CTest where registered.
- Root `SHA256SUMS`, `MANIFEST.md`, and `AUDIT_MANIFEST.md` describe an older
  spin-minus2 bundle and are not current-tree integrity evidence.
- Historical documents contain commit-scoped test counts; they are not current
  pass counts unless the exact commit is named.
- `plus2_equation_spec_proposal.yaml` remains a fail-closed proposal, not a
  production specification.
- Local B580 observations and old binaries are not current exact-HEAD GPU
  qualification.
- The T=200M summary is frozen local evidence and cannot be regenerated from
  this repository alone.

## J. Final verdict and invariants

**Verdict: MERGEABLE STANDALONE VALIDATION.**

- Production spin-minus2 equations/source/pipeline behavior were preserved in
  the external-review remediation range.
- `plus2.enabled` remains fail-closed in `teuk_solver`.
- No tolerance was weakened.
- No failing test was deleted.
- No historical result was silently rewritten; corrections are dated errata or
  later status blocks.
- The repository is not production-qualified for spin `+2`.
