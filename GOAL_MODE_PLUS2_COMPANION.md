# GOAL: Add a rigorously derived one-way spin +2 companion and four-Weyl-field output to `teuk-kokkos`

You are the coordinating computational-relativity scientist and software engineer for this long autonomous task.

Work from the current public `main` branch of `HengruiZhu99/teuk-kokkos`. Inspect the current head rather than assuming an older commit. Preserve the corrected first-order spin -2 equation, the corrected second-order spin -2 source, the metric-reconstruction stage graph, the runtime configuration system, the source-activation semantics, the signed-mode machinery, and the existing Kokkos portability work unless an executable or mathematical regression demonstrates a defect.

The requested scientific capability is to begin from the existing linear spin -2 perturbation data and produce, in one consistent run or deterministic replay,

1. `Psi4^(1)` / spin -2 at first order;
2. `Psi0^(1)` / spin +2 at first order;
3. `Psi4^(2)` / spin -2 at second order;
4. `Psi0^(2)` / spin +2 at second order.

The intended fast architecture is **not** a full reconstruction of the second-order metric. It is a one-way spin +2 companion:

- compute `Psi0^(1)` from the already reconstructed first-order metric in the current gauge and tetrad;
- validate it independently with the linear Teukolsky-Starobinsky identities;
- derive and solve a passive sourced spin +2 second-order Teukolsky equation whose source is built from the same first-order metric history;
- never feed either spin +2 field back into the existing spin -2 evolution.

The spin +2 second-order field is therefore produced from the same first-order spin -2/reconstruction history, not by naively applying the linear spin-reversal operator to `Psi4^(2)`.

Continue until all completion gates below are satisfied as fully as the available environment permits. Do not stop after writing a plan, deriving only the linear map, creating placeholders, or making a source that merely compiles.

---

## 1. Authoritative material to read first

Read the current implementations and conventions in full, including at least:

- `docs/CONVENTIONS.md`
- `CORRECTED_EQUATIONS_IMPLEMENTER_REFERENCE.md`
- `TRIPLE_CHECKED_SECOND_ORDER_TEUKOLSKY_DERIVATION.md`
- `equation_spec_v2.yaml`
- `SOURCE_TERM_LEDGER.csv`
- `include/teuk/teukolsky.hpp`
- `include/teuk/reconstruction.hpp`
- `include/teuk/second_order.hpp`
- `include/teuk/spatial_pipeline.hpp`
- `include/teuk/pipeline_storage.hpp`
- `include/teuk/angular_coordinator.hpp`
- `include/teuk/source_spatial.hpp`
- `include/teuk/source_tangent_spatial.hpp`
- `include/teuk/run_parameters.hpp`
- `include/teuk/config.hpp`
- `include/teuk/solver_driver.hpp`
- all existing mathematical audit and remediation documents.

Read and use the following primary references:

- Spiers, Pound, and Moxon, *Second-order Teukolsky formalism in Kerr spacetime: formulation and nonlinear source*, arXiv:2305.19332, together with its Mathematica notebook `DrAndrewSpiers/NP-and-GHP-Formalisms-for-2nd-order-Teukolsky`;
- Loutrel, Ripley, Giorgi, and Pretorius, *Second Order Perturbations of Kerr Black Holes: Reconstruction of the Metric*, arXiv:2008.11770;
- Ripley, Loutrel, Giorgi, and Pretorius, *Numerical computation of second order vacuum perturbations of Kerr black holes*, arXiv:2010.00162;
- Aksteiner, Andersson, and Bäckdahl, *New identities for linearized gravity on the Kerr spacetime*, arXiv:1601.06084;
- Campanelli and Lousto, *Second order gauge invariant gravitational perturbations of a Kerr black hole*, arXiv:gr-qc/9811019;
- Green, Hollands, and Zimmerman, *Teukolsky formalism for nonlinear Kerr perturbations*, arXiv:1908.09095;
- Berens, Gravely, and Lupsasca, *Gravitational Waves on Kerr Black Holes I: Reconstruction of Linearized Metric Perturbations*, arXiv:2403.20311.

Treat the checked-in current conventions as authoritative for coordinates, tetrad, outgoing-radiation gauge, perturbative normalization, radial rescalings, signed `m`, sharp conjugation, and angular normalization. If a paper uses different conventions, derive and document the conversion before using its formulas.

---

## 2. Scientific non-negotiables

### 2.1 Do not use the naive nonlinear spin reversal

Do not implement

`Psi0^(2) = D_TS[Psi4^(2)]`

with the linear Teukolsky-Starobinsky operator and call it the physical second-order spin +2 field.

At second order the equation is sourced, the source depends on the full first-order metric, and the raw second-order extreme Weyl coefficients depend on the first-order gauge/tetrad convention. A valid nonlinear or sourced Teukolsky-Starobinsky identity, if derived, must contain explicit first-order/source correction terms and a clearly stated homogeneous ambiguity.

### 2.2 Use the same first-order gauge and tetrad for both signs

The existing first-order metric is reconstructed in the code's outgoing-radiation gauge and rotated-Kinnersley tetrad convention.

The spin +2 second-order source must be derived in that **same** first-order gauge and tetrad convention.

Do not simply prime the already ORG-specialized spin -2 source. Priming exchanges the null legs and naturally maps ORG expressions to the corresponding IRG expressions. Instead:

1. begin with the ungauge-fixed general second-order NP/GHP equation;
2. verify its prime symmetry before gauge specialization;
3. specialize the spin +2 equation directly to the current ORG metric and current tetrad;
4. derive its coordinate and compactified form.

### 2.3 Do not mix second-order variables from different formalisms

Determine exactly which second-order spin +2 variable is paired with the existing code's `Psi4^(2)` convention. Do not combine a Campanelli-Lousto invariant waveform on one side with a raw ORG/tetrad Weyl coefficient on the other without deriving the conversion.

State explicitly whether the production output is:

- the raw perturbative coefficient in the fixed current tetrad;
- a quadratic-corrected invariant variable;
- or both.

### 2.4 Do not assume the current spin-generic coefficient function is already the correct compactified spin +2 equation

Although low-level coefficient code accepts a spin-weight parameter, the current field rescaling, boundary regularity, angular coordinator, source normalization, and pipeline have been qualified for the spin -2 variable.

Independently derive the compactified spin +2 equation and its regular evolved field before reusing any generic code.

---

## 3. Target architecture

Implement two related modes.

### 3.1 Concurrent companion mode

The existing first-order/reconstruction/spin -2 second-order pipeline and the passive spin +2 second-order state advance through the **same RK stages**.

The spin +2 state has no feedback into the primary state.

Use this mode for validation, short calculations, and exact concurrent/replay comparisons.

### 3.2 Deterministic replay postprocessing mode

Implement postprocessing as deterministic replay, not as a fourth-order differential operation on sparse snapshots.

Given the original resolved configuration and an initial checkpoint or reproducible initial state:

1. replay the first-order spin -2 and first-order reconstruction stage graph;
2. reproduce the existing spin -2 second-order trajectory if requested;
3. at every RK stage compute the first-order spin +2 curvature data and the spin +2 quadratic source;
4. advance the passive spin +2 second-order state;
5. write all four Weyl fields at common output times.

A sparse snapshot-only mode may compute `Psi0^(1)` from the reconstructed metric, but it must not claim to produce a fourth-order `Psi0^(2)` trajectory unless sufficient dense stage history was stored.

### 3.3 Output fields

At each requested output time provide, by signed `m` and angular/radial location or modal projection:

- `psi4_order1`;
- `psi0_order1`;
- `psi4_order2`;
- `psi0_order2`.

Also provide the internally evolved regularized variables and the exact inverse scaling used to reconstruct the physical tetrad Weyl scalars.

---

## 4. Parallel work plan

Use no more than four direct specialist agents in isolated worktrees.

### Agent A — spin +2 formalism and symbolic source

Own:

- general NP/GHP spin +2 second-order derivation;
- ORG specialization;
- coordinate conversion;
- radial rescaling;
- source term ledger;
- symbolic verification and generated expressions.

Do not write the production Kokkos source until the source ledger and convention map are reviewed.

### Agent B — linear `Psi0^(1)` and Teukolsky-Starobinsky validation

Own:

- the linear Weyl-curvature operator `T0[h^(1)]`;
- stage-consistent time derivatives;
- the covariant/time-domain linear TSI implementation or a rigorously equivalent modal validation path;
- pure-mode and QNM tests.

### Agent C — Kokkos companion/replay integration

Own:

- optional companion storage;
- common-stage RK integration;
- replay mode;
- runtime configuration;
- checkpoint compatibility;
- device-resident scratch and performance.

### Agent D — validation, outputs, and independent audit

Own:

- manufactured solutions;
- source oracle comparisons;
- boundary regularity tests;
- replay/concurrent equivalence;
- CPU/GPU parity;
- output metadata and documentation.

The coordinating agent is the sole integration authority. Establish shared conventions before delegation and review every merge.

---

## 5. Phase I — derive the linear spin +2 field from the reconstructed metric

### 5.1 Implement the linear curvature operator

Derive and implement

`Psi0^(1) = T0[h^(1)]`

in the current coordinates, current rotated-Kinnersley tetrad, and current ORG convention.

Use the existing stored reconstruction fields and their sharp partners to recover all required metric components. Keep the implementation visibly connected to the derivation.

Prefer a GHP/NP expression with the lowest practical differential order over a brute-force coordinate Riemann calculation in production. A transparent coordinate-tensor implementation may be retained as an independent host oracle.

### 5.2 Stage-consistent evaluation

When `Psi0^(1)` or its derivatives are needed by the second-order source, evaluate them from the same RK stage state and stage tangent as the source.

Use the existing analytic tangent/Jet strategy. Do not use endpoint interpolation or coarse-output finite differences in time.

### 5.3 Linear TSI as an independent check

Implement the linear covariant Teukolsky-Starobinsky relation in the same coordinate/tetrad conventions, or a separated-mode form for rigorous validation cases.

Use it to compare against `T0[h^(1)]`.

Because the TSI contains high derivatives and may amplify discretization noise, the production linear `Psi0^(1)` output should default to the metric-curvature route unless convergence studies demonstrate otherwise.

### 5.4 Optional homogeneous spin +2 validation evolution

Optionally initialize a homogeneous spin +2 companion from the metric-derived `Psi0^(1)` and its time derivative, then evolve it as a validation path.

It must converge to the metric-derived field for consistent first-order data.

Do not require this optional state in production once the equivalence is established.

---

## 6. Phase II — derive a boundary-regular spin +2 evolution variable

Let the physical code-tetrad field be `Psi0` and define a stored field schematically by

`Psi0 = W_plus(R,theta) * Z_plus`.

Do not guess `W_plus`.

Derive it from:

- the actual tetrad used in this repository;
- peeling at future null infinity;
- regularity at the future event horizon;
- any necessary tetrad boost;
- the desired finite nonzero limiting behavior of the numerical field.

Perform asymptotic expansions at both

- `R = 0` (`scri+`);
- `R = R_H` (future horizon).

If no single simple scaling is regular at both boundaries, introduce a documented regular tetrad/boosted field for evolution and provide conversion to the code-tetrad `Psi0` where meaningful.

### Required outputs of this derivation

Create:

- `docs/PLUS2_FIELD_AND_TETRAD_CONVENTIONS.md`;
- symbolic asymptotic checks;
- a table of spin weight, boost weight, radial falloff, and physical/internal definitions;
- unit tests showing all transformed PDE coefficients remain finite at both boundaries.

---

## 7. Phase III — derive and implement the homogeneous spin +2 Teukolsky operator

Starting from the master spin +2 Teukolsky equation, derive the equation obeyed by `Z_plus` in the current compactified hyperboloidal coordinates.

Reduce it to the same readable first-order radial structure where possible:

- `P_plus`;
- `Q_plus = partial_R Z_plus`;
- `Z_plus`.

Do not assume every lower-order coefficient is obtained by setting `spin_weight=+2` in the existing spin -2 implementation until exact symbolic equality has been demonstrated after the new field rescaling.

### Required mathematical checks

Verify:

- the principal symbol;
- characteristic speeds at scri and the future horizon;
- the spin +2 angular eigenvalue;
- Schwarzschild limit;
- Kerr coefficients at random points against a symbolic oracle;
- boundary regularity;
- reduction-constraint propagation;
- consistency with separated spin +2 Kerr modes.

Reuse current generic kernels only after these checks pass.

---

## 8. Phase IV — derive the second-order spin +2 source

Implement a passive equation of the form

`O_plus2[Z_plus^(2)] = S_plus2[h^(1)]`.

The source must be derived from the same second-order formalism and perturbative convention as the current spin -2 equation.

### 8.1 General derivation first

Start from the ungauge-fixed general NP/GHP source in Spiers-Pound-Moxon and its notebook.

At the ungauge-fixed level:

- derive the spin +2 equation;
- verify prime symmetry against the spin -2 equation;
- verify GHP weights;
- verify the perturbative-order and factorial convention.

Only then impose the current ORG conditions.

### 8.2 Exact source-input manifest

Determine, rather than guess, the complete set of first-order inputs required by `S_plus2`.

The current reconstruction stores a subset of first-order Weyl scalars, spin coefficients, and metric components. Derive any additional quantities required, potentially including quantities such as `Psi0^(1)`, `Psi1^(1)`, `kappa^(1)`, `sigma^(1)`, or other perturbed connection coefficients.

Create a machine-readable manifest recording for each input:

- physical definition;
- stored/rescaled definition;
- spin and boost weights;
- radial falloff;
- sharp relation;
- how it is computed from the current first-order reconstruction;
- differential order;
- stage tangent.

Do not add independent evolution variables when an algebraic/differential derived quantity is sufficient and stable.

### 8.3 Compact source and term ledger

Create:

- `PLUS2_SOURCE_TERM_LEDGER.csv`;
- `plus2_equation_spec.yaml` or an equivalently transparent machine-readable spec;
- `tools/symbolic/verify_plus2_source.py` and/or Wolfram-language derivation scripts;
- generated C++ only where expressions are too long to maintain safely by hand.

Every source term must have:

- a stable identifier;
- coefficient;
- source family;
- ordered `(m1,m2)->mt` semantics;
- spin/boost weight;
- radial power;
- conjugation/sharp information;
- a corresponding test.

### 8.4 Human-readable production code

Structure production code around readable pieces such as:

- `compute_plus2_inner_source_pair(...)`;
- `apply_plus2_outer_source_operator(...)`;
- `evaluate_plus2_source_tangent(...)`.

Keep physically meaningful intermediate terms. Do not hide the source in opaque metaprogramming.

### 8.5 Common-stage source tangents

Use the same-stage first-order state and tangent.

Propagate analytic time derivatives with product/quotient rules or the existing lightweight Jet machinery.

Do not interpolate sources between endpoints.

---

## 9. Phase V — optional one-way companion storage and RK integration

The existing pipeline has two spin -2 Teukolsky triples and seven reconstruction fields. Add the spin +2 second-order state without disrupting the qualified primary path.

A preferred design is an optional `Plus2CompanionStorage` containing one spin +2 second-order triple and its reusable RK/scratch buffers. However, if the current RK implementation makes an extended flat state substantially simpler and equally readable, a versioned extension of the pipeline state is acceptable.

Hard requirements:

- no operator splitting between the primary and companion states;
- all states use the same RK stage times;
- the companion sees the primary first-order/reconstruction state at that exact stage;
- the companion never feeds back;
- no per-step allocation;
- no required host/device transfer inside a timestep;
- no runtime dependency beyond Kokkos and the standard library.

Use spin +2 angular coordinators with explicit parent/target `m` registries and independent configurable angular bandlimits.

---

## 10. Replay postprocessing

Add a runtime mode conceptually equivalent to:

`plus2.mode = replay`

The replay mode must accept:

- the resolved configuration of the original run;
- an initial state or checkpoint;
- the same primary numerical settings;
- the desired spin +2 companion settings.

It then deterministically re-executes the stage graph and writes the four-field output.

### Replay validation

For a short calculation, run both:

- `plus2.mode = concurrent`;
- `plus2.mode = replay`.

The spin +2 trajectories and all four output fields must agree to the expected backend tolerance.

Document any reproducibility limitation caused by backend reduction order or restart precision.

---

## 11. Runtime configuration

Extend the existing strict runtime configuration system; do not add a new parser dependency.

Add typed parameters conceptually like:

- `plus2.enabled = true|false`;
- `plus2.mode = diagnostic_only|concurrent|replay`;
- `plus2.linear.method = metric_curvature|tsi|both`;
- `plus2.linear.evolve_validation = true|false`;
- `plus2.second.method = sourced_companion`;
- `plus2.second.initial_policy = zero|checkpoint`;
- `plus2.second.checkpoint = ...`;
- `plus2.ell_max_first = ...`;
- `plus2.ell_max_second = ...`;
- `plus2.output.regularized = true`;
- `plus2.output.physical_tetrad_field = true`;
- `plus2.output.source_families = true`;
- `plus2.output.ordered_pairs = false`.

Use current first-order parent modes for `Psi0^(1)` and current second-order target modes for `Psi0^(2)` unless explicitly overridden with a validated registry.

Reject unknown values and incompatible combinations.

Update the resolved configuration and checkpoint schema version.

---

## 12. Second-order initial/homogeneous-content semantics

A spin +2 sourced equation needs its own initial data or past-boundary prescription.

Support at least:

- `zero`: set the spin +2 second-order companion state to zero at the replay/start slice;
- `checkpoint`: load an explicitly supplied companion state.

The `zero` option is a well-defined development convention, but it must be labelled clearly:

- it does not prove that the spin +2 and spin -2 second-order fields are components of one common second-order metric;
- it does not establish the nonlinear Teukolsky-Starobinsky constraints;
- it does not replace a retarded-history or constraint-consistent second-order prescription.

Do not block the companion implementation on solving the general second-order initial-data problem, but do not overclaim the meaning of independently initialized fields.

Use the same accepted-step source-activation state as the existing spin -2 second-order equation so the two sourced companions share the same first-order source history.

---

## 13. Optional research validation: sourced/nonlinear TSI

After the passive spin +2 companion is working, investigate whether an exact sourced identity of schematic form

`D_plus[Psi0^(2)] - C_TS Psi4^(2) = Q_TS[h^(1), source]`

can be derived in the current conventions.

This is a validation/research path, not the initial production route.

Do not invent the correction term.

If a complete identity is derived:

- store the derivation;
- include source and homogeneous terms;
- implement its residual;
- compare it to the passive companion.

If it cannot be completed, document the precise missing identity rather than weakening the claim.

---

## 14. Output and metadata

Output both physical and numerical variables with unambiguous names and metadata.

At minimum include:

- `psi4_order1_code_tetrad`;
- `psi0_order1_code_tetrad`;
- `psi4_order2_code_tetrad`;
- `psi0_order2_code_tetrad`;
- internal regularized spin -2 and spin +2 fields;
- modal coefficients by `(ell,m)` where requested;
- source totals, source families, and optional ordered-pair components;
- tetrad and gauge identifier;
- radial scaling formulas;
- perturbative expansion convention;
- second-order companion initial policy;
- whether linear `Psi0` came from curvature, TSI, or both;
- current git commit and configuration schema.

Do not call a raw second-order Weyl coefficient gauge invariant unless the required quadratic correction has actually been implemented.

---

## 15. Validation ladder

### 15.1 Symbolic checks

Require exact or high-precision checks of:

- spin/boost weights;
- prime symmetry before gauge specialization;
- ORG specialization;
- radial falloffs;
- tetrad-boost factors;
- boundary expansions;
- Schwarzschild limits;
- random-point equality between generated and independently evaluated source expressions.

### 15.2 Linear `Psi0^(1)` checks

Test:

- metric-curvature `Psi0^(1)` against the linear TSI;
- pure spin-weighted harmonic modes;
- Schwarzschild QNMs;
- at least one moderate-spin Kerr QNM;
- radial, angular, and temporal convergence;
- the homogeneous spin +2 Teukolsky residual;
- amplitude linearity;
- CPU/GPU parity.

For separated modes, verify the Teukolsky-Starobinsky normalization and phase, not only frequency.

### 15.3 Spin +2 PDE checks

Add:

- manufactured solutions for the compactified homogeneous equation;
- reduction-constraint convergence;
- characteristic/boundary tests;
- horizon and scri regularity tests;
- full production-path QNM regressions.

### 15.4 Second-order source checks

Require:

- an independent host/CAS source oracle;
- every term-ledger entry tested;
- ordered `m1+m2=mt` selection;
- generalized spin-weighted Gaunt-product checks;
- sharp-conjugation tests;
- quadratic amplitude scaling;
- source-family decomposition closure;
- manufactured driven solutions;
- fourth-order time convergence for a smooth source;
- radial/angular convergence;
- backend parity.

### 15.5 Four-field integration checks

Verify:

- concurrent/replay equivalence;
- checkpoint/restart equivalence;
- common output times and mode conventions;
- no feedback from spin +2 to the primary trajectory;
- disabling the feature reproduces the existing primary trajectory to bitwise or documented floating-point equivalence;
- output cadence does not change the trajectory.

---

## 16. Performance and dependency policy

Keep Kokkos as the only mandatory production dependency.

Symbolic tools may be offline development dependencies only.

When `plus2.enabled=false`:

- allocate no large spin +2 scratch/state unnecessarily;
- add no per-step kernels;
- reproduce the existing performance path.

When enabled:

- keep all repeated coefficient/source data on device;
- allocate once;
- parallelize over mode, radius, and angle;
- profile before fusing readable source terms;
- preserve a clear reference implementation.

Replay mode is intended to allow expensive spin +2 calculations after a primary spin -2 campaign without burdening every production run.

---

## 17. Human readability and provenance

Keep explicit equation-level names.

Do not produce one enormous generated expression without a source ledger and readable grouping.

Generated files must state:

- generating script;
- source paper/equation;
- convention map;
- checksum/freshness test;
- warning not to edit manually.

Create:

- `docs/PLUS2_FORMALISM.md`;
- `docs/PLUS2_FIELD_AND_TETRAD_CONVENTIONS.md`;
- `docs/PLUS2_REPLAY_AND_OUTPUT.md`;
- `docs/FOUR_WEYL_FIELD_VALIDATION.md`;
- `PLUS2_SOURCE_TERM_LEDGER.csv`;
- a machine-readable plus2 equation specification;
- a focused final validation report.

---

## 18. No-shortcut rules

The following do not count as completion:

- applying the linear TSI directly to `Psi4^(2)`;
- deriving the source by priming the already ORG-specialized spin -2 source;
- reconstructing the first-order metric in IRG for the plus2 source while retaining ORG for the minus2 source, then presenting the two raw second-order fields as one gauge-consistent pair;
- setting `spin_weight=+2` in existing coefficients without deriving the field scaling and boundary behavior;
- computing `Psi0^(2)` only at saved snapshots with no time integration;
- interpolating the source between output times;
- omitting tetrad-perturbation or quadratic terms required by the selected second-order variable;
- claiming zero-initialized `Psi0^(2)` and `Psi4^(2)` necessarily come from one second-order metric;
- hiding unverified formulas behind generated code;
- adding a full second-order metric reconstruction instead of completing the narrower companion goal;
- weakening existing spin -2 tests.

---

## 19. Suggested commit sequence

Use small reviewable commits such as:

1. `docs: fix spin +2 perturbative and tetrad conventions`
2. `symbolic: derive linear Psi0 curvature operator in current ORG`
3. `test: validate linear Psi0 against Teukolsky-Starobinsky identities`
4. `symbolic: derive boundary-regular compactified spin +2 equation`
5. `feat: add spin +2 homogeneous Kokkos operator`
6. `symbolic: derive ORG second-order spin +2 source`
7. `test: add plus2 source term ledger and independent oracle`
8. `feat: add passive common-stage spin +2 companion`
9. `feat: add deterministic plus2 replay mode`
10. `feat: output four Weyl fields with complete metadata`
11. `test: validate replay/concurrent and backend parity`
12. `docs: publish four-field validation and limitations`

Do not squash all scientific derivation and implementation into one commit.

---

## 20. Completion gates

The goal is complete only when all of the following hold:

1. The exact first-order `Psi0^(1)` curvature operator in the current tetrad/gauge has been derived and implemented.
2. `Psi0^(1)` agrees with an independent linear TSI calculation under convergence.
3. A regular spin +2 evolved field and its inverse physical scaling have been derived at both scri and the future horizon.
4. The compactified spin +2 homogeneous equation has passed symbolic, characteristic, QNM, and convergence tests.
5. The second-order spin +2 source has been derived from the ungauge-fixed formalism and then specialized to the current ORG; it is not a guessed prime of the simplified spin -2 source.
6. Every source term has a machine-readable ledger entry and independent check.
7. A passive sourced spin +2 second-order state advances at the same RK stages as the primary pipeline and never feeds back.
8. Deterministic replay reproduces concurrent companion evolution.
9. The code outputs all four requested Weyl fields with explicit physical/internal scaling and gauge/tetrad metadata.
10. Linear quantities scale as first-order amplitude and both second-order extreme fields scale quadratically in a qualified test.
11. The sourced spin +2 residual converges in time, radius, and angle.
12. CPU and every available accelerator backend agree to justified tolerances.
13. Existing qualified spin -2 tests remain passing.
14. The documentation clearly distinguishes a sourced companion from a naive transformation of `Psi4^(2)` and clearly states the homogeneous/initial-data convention.

Begin by inspecting the current head, establishing a convention map, and running the existing complete test suite. Then launch the four specialist workstreams, with symbolic formalism as the first merge gate. Implement, test, review, repair, and continue autonomously until the completion gates are met or a genuine externally imposed blocker is documented with all independent progress completed.
