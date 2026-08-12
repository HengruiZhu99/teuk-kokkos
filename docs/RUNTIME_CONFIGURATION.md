# Runtime solver configuration

`teuk_solver` is compiled once and receives every run-dependent science choice
from a strict, dependency-free `key = value` file:

```bash
./build/serial/teuk_solver --config configs/linear_schwarzschild.cfg
```

Trailing `key=value` arguments override file values without recompilation:

```bash
./build/serial/teuk_solver --config configs/near_extremal.cfg \
  spin=0.99 nr=1025 output.directory=run-a099
```

Blank lines and `#` comments are accepted. Keys and enum values are
case-sensitive. Duplicate file keys, duplicate command-line overrides,
unknown keys, malformed values, and unsupported schema versions are errors.
The current required schema marker is `config_version = 1`.

## Typed keys

| Group | Keys |
|---|---|
| Background | `mass`, `spin`, `compactification_length` |
| Grid and bands | `nr`, `ntheta`, `ellmax_first`, `ellmax_second`, `first_order_modes`, `second_order_modes` |
| Evolution | `final_time`, `steps`, `cfl`, `reduction_damping`, `dissipation`, `reduction_mode`, `radial_discretization` |
| Initial data | `initial_data.type`, `initial_data.center`, `initial_data.width`, `initial_data.time_derivative`, `initial_data.add_sharp_partner`, `initial_data.compact_support`, `initial_data.checkpoint_directory` |
| Base Gaussian mode | `initial_data.seed_ell`, `initial_data.seed_m`, `initial_data.amplitude_real`, `initial_data.amplitude_imag` |
| Additional modes | `initial_data.mode.N.ell`, `.m`, `.amplitude_real`, `.amplitude_imag` |
| Second order | `second_order.enabled`, `.source_mode`, `.source_start_time`, `.constraint_tolerance`, `.required_consecutive_passes`, `.allow_truncated_daughter_modes` |
| Spin +2 companion | `plus2.enabled`, `.mode`, `.linear.method`, `.linear.evolve_validation`, `.second.method`, `.second.initial_policy`, `.second.checkpoint`, `.ell_max_first`, `.ell_max_second` |
| Spin +2 output policy | `plus2.output.regularized`, `.physical_tetrad_field`, `.source_families`, `.ordered_pairs` |
| Output | `output.directory`, `output.diagnostic_every`, `output.checkpoint_every` |

`reduction_mode` is `free_damped` or `stage_constrained`.
`radial_discretization` is `d4-2` (the backward-compatible default) or
`d8-4`; D8-4 has fourth-order boundary closures and requires at least 16
radial points. The selected scheme is recorded in resolved configuration and
checkpoint metadata and is used consistently by evolution, reconstruction,
source, initial-data, and constraint-diagnostic radial operations.
`second_order.source_mode` is `constraint_aware` or `unrestricted` when
second-order evolution is enabled. `initial_data.type` is currently
`gaussian` or `checkpoint`; no unverified profile is advertised.

The parsed spin +2 enum values are:

- `plus2.mode`: `disabled`, `diagnostic_only`, `concurrent`, or `replay`;
- `plus2.linear.method`: `metric_curvature`, `tsi`, or `both`;
- `plus2.second.method`: only `sourced_companion`;
- `plus2.second.initial_policy`: `zero` or `checkpoint`.

The checkpoint path is required exactly when the companion initial policy is
`checkpoint`, and is forbidden for `zero`. Disabled and `diagnostic_only`
modes cannot load a second-order checkpoint. Concurrent and replay modes
require the existing spin -2 `second_order.enabled=true` path so a requested
run retains all four Weyl fields. The current first-order parent and
second-order target signed-m registries must fit within the independent plus2
angular bands, and `ntheta` must resolve those bands and their padded products.
Ordered-pair output requires source-family output. Every plus2 value, including
disabled defaults, is emitted into `resolved_config.cfg`.

### Production integration gate

The runtime surface is intentionally present before production activation so
configurations and replay provenance can stabilize. The standalone curvature,
source, companion, and replay components are not yet connected to
`SpatialPipeline`. Therefore every `plus2.enabled=true` mode currently fails
at solver startup with `not production integrated`, after strict parsing and
resolution but before output-directory creation or large Kokkos allocation.
The only executable setting is the default pair
`plus2.enabled=false`, `plus2.mode=disabled`; it leaves the qualified primary
path unchanged. This gate must be removed only with the production wiring and
its four-field validation evidence.

These keys are backward-compatible optional additions with fully resolved
defaults, so the runtime configuration schema remains version 1.

`initial_data.center` and `initial_data.width` are fractions of the compactified
outer-horizon coordinate `R_H`. With `initial_data.compact_support=false`, the
profile is `exp(-x^2)`. With it enabled, `width` is the exact support
half-width and the profile is the normalized smooth bump
`exp(1-1/(1-x^2))` for `|x|<1`, zero otherwise. Gaussian `Q` uses the production D4-2
derivative. The initializer solves the retained Galerkin system for `P`, so a
requested zero coordinate-time derivative is enforced without admitting
Kerr-generated `ellmax_first+1` content. Every generated field is projected
once before entering evolution.

Additional Gaussian modes use consecutive indices in normal usage:

```text
initial_data.mode.1.ell = 3
initial_data.mode.1.m = 1
initial_data.mode.1.amplitude_real = 2.0e-5
initial_data.mode.1.amplitude_imag = -1.0e-5
```

The four fields are required for every additional mode. Seeds are sorted
deterministically by signed `m` and `ell`. Negative-`m` amplitudes are never
inferred unless `initial_data.add_sharp_partner = true`, which adds the
documented conjugate sharp partner and rejects duplicates.

## Validation and mode completeness

The complete typed configuration is validated before the large pipeline state
is allocated. Among other checks, the solver rejects:

- `abs(spin) > mass`, invalid compactification or nonfinite values;
- fewer radial points than the selected SBP operator requires (8 for D4-2,
  16 for D8-4);
- signed mode sets not closed under `m -> -m`;
- `|m|` or seed `ell` outside its declared band;
- an angular grid too small for both retained bands and padded products;
- an explicit timestep above the configured radial characteristic CFL bound;
- incomplete quadratic daughters when second order is enabled.

First-order parent modes and second-order target modes are distinct lists. For
every ordered parent pair, the required daughter satisfies
`m_target = m1 + m2`. Missing daughters fail closed. Setting
`second_order.allow_truncated_daughter_modes = true` is an explicit opt-out;
the executable emits a conspicuous non-mode-complete warning. If
`ellmax_second` is omitted it inherits `ellmax_first`.

When `second_order.enabled = false`, the runtime policy is truly linear-only:
the production angular transform, stage projection, SBP radial operator, and
common-stage RK4 evolve the first Teukolsky triple, while all other RHS fields
are exactly zero.

## Deterministic source activation

Constraint-aware activation is evaluated only on accepted states. All four
stages of a given RK4 substep see one fixed latched flag. If
`source_start_time` lies inside a requested step, the step is split exactly at
that time; an event at the endpoint contributes nothing to the preceding
step.

The three independent reconstruction constraint families report absolute
maximum, normalized maximum, weighted SBP-Gauss RMS, and normalized weighted
norm. Normalization uses the natural left/right terms of each equation, not
the user amplitude. Eligibility uses the controlling normalized value and may
require `required_consecutive_passes` accepted states. Activation is monotonic
and remains diagnostic after it latches.

## Output, provenance, and restart

Before solver allocation, every run writes
`output.directory/resolved_config.cfg`. It contains all resolved defaults,
file values, and overrides plus the Git commit, Kokkos backend/version,
complex-binary64 precision, executable version, and schema version.

Checkpoint metadata records strict geometry, ordered stored/parent/target mode
sets, both angular bands, method parameters, timestep, source policy, latched
activation time, accepted-pass counter, and state checksum. Loading measures
all 13 fields against their retained spin bands before mutating caller state;
meaningful off-band content or any compatibility mismatch is rejected.

Resume into a fresh output directory with the same timestep and numerical
configuration:

```bash
./build/serial/teuk_solver --config configs/kerr_ringdown.cfg \
  initial_data.type=checkpoint \
  initial_data.checkpoint_directory=run-kerr-ringdown/checkpoint_00064000 \
  output.directory=run-kerr-ringdown-resumed
```

`diagnostics.csv` contains state, RHS, reduction, reconstruction, source-gate,
forcing, and horizon diagnostics. `source_pairs.csv` records deterministic
ordered-pair contributions. Checkpoint directories are published through a
single directory rename; publication does not currently issue
platform-specific `fsync`.

## Templates and sweeps

Checked-in starting points are:

- `configs/linear_schwarzschild.cfg`;
- `configs/kerr_ringdown.cfg`;
- `configs/second_order_22_self_coupling.cfg`;
- `configs/near_extremal.cfg`.

They pass the parser, mode-completeness, angular-padding, and radial-CFL gates
as written. They are development templates, not blanket scientific
qualification. External shell or Python loops should supply overrides rather
than generating new C++ programs:

```bash
for spin in 0.9 0.99 0.999; do
  ./build/serial/teuk_solver --config configs/near_extremal.cfg \
    spin=$spin output.directory=run-$spin
done
```
