#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <filesystem>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "teuk/plus2_companion_pipeline.hpp"
#include "teuk/plus2_live_source_composition.hpp"
#include "teuk/angular.hpp"
#include "teuk/pipeline_checkpoint.hpp"

namespace {

constexpr std::size_t P =
    static_cast<std::size_t>(teuk::TeukolskyField::P);
constexpr std::size_t Q =
    static_cast<std::size_t>(teuk::TeukolskyField::Q);
constexpr std::size_t Psi =
    static_cast<std::size_t>(teuk::TeukolskyField::Psi);

struct ZeroComplexFunctor {
  teuk::Complex* output;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    output[i] = teuk::Complex{};
  }
};

struct CopyTripleToStridedFunctor {
  const teuk::Complex* source;
  teuk::Complex* destination;
  std::size_t destination_field_stride;
  std::size_t destination_radial_stride;
  std::size_t destination_theta_stride;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t field_plane = radial_count * theta_count;
    const std::size_t field = flat / field_plane;
    const std::size_t within = flat - field * field_plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    destination[field * destination_field_stride +
                radial * destination_radial_stride +
                theta * destination_theta_stride] = source[flat];
  }
};

struct SpoofedNormalizedSourceAdapter {
  int* calls;

  [[nodiscard]] teuk::Plus2SourceNormalization source_normalization() const {
    return teuk::plus2_source_normalization;
  }

  template <class... Arguments>
  void operator()(Arguments&&...) {
    ++*calls;
  }
};

static_assert(std::is_trivially_copyable_v<ZeroComplexFunctor>);
static_assert(sizeof(ZeroComplexFunctor) < 1800);
static_assert(std::is_trivially_copyable_v<CopyTripleToStridedFunctor>);
static_assert(sizeof(CopyTripleToStridedFunctor) < 1800);

int pipeline_allocations = 0;
int pipeline_fences = 0;

void count_pipeline_allocation(Kokkos::Tools::SpaceHandle, const char*,
                               const void*, std::uint64_t) {
  ++pipeline_allocations;
}

void count_pipeline_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++pipeline_fences;
}

teuk::Plus2ReplayConfiguration pipeline_configuration(
    const std::size_t radial_count, const std::size_t theta_count) {
  teuk::Plus2ReplayConfiguration config;
  config.mode = teuk::Plus2RunMode::Concurrent;
  config.ell_max_first = 2;
  config.ell_max_second = 2;
  config.parent_modes = {-1, 1};
  config.target_modes = {0};
  config.radial_count = radial_count;
  config.theta_count = theta_count;
  config.git_commit = teuk::plus2_build_git_commit();
  config.runtime_config_schema_version = 1;
  return config;
}

teuk::TeukolskyParameters plus2_parameters() {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.31;
  parameters.compactification_length = 1.7;
  parameters.spin_weight = 2;
  parameters.reduction_damping = 1.4;
  return parameters;
}

teuk::Plus2ReplayConfiguration complete_pipeline_configuration(
    const teuk::UniformRadialGrid& grid,
    const std::vector<double>& theta_coordinates,
    const teuk::ReductionEvolution reduction,
    const double dissipation, const double time_step = 0.0) {
  auto configuration =
      pipeline_configuration(grid.size(), theta_coordinates.size());
  const auto parameters = plus2_parameters();
  configuration.mass = parameters.mass;
  configuration.spin = parameters.spin;
  configuration.compactification_length =
      parameters.compactification_length;
  configuration.radial_coordinates.resize(grid.size());
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    configuration.radial_coordinates[radial] = grid.coordinate(radial);
  }
  configuration.theta_coordinates = theta_coordinates;
  configuration.time_step = time_step;
  configuration.reduction_mode =
      reduction == teuk::ReductionEvolution::FreeDamped
          ? "free_damped"
          : "stage_constrained";
  configuration.reduction_damping = parameters.reduction_damping;
  configuration.dissipation = dissipation;
  return configuration;
}

template <class View>
void zero_angular_action(const teuk::ExecutionSpace& execution, const double,
                         const auto&, const View& angular_laplacian) {
  const std::size_t size = angular_laplacian.size();
  auto* output = angular_laplacian.data();
  Kokkos::parallel_for(
      "plus2_pipeline_zero_angular",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, size),
      ZeroComplexFunctor{output});
}

auto zero_source_adapter() {
  return [](const teuk::ExecutionSpace& execution, const double, const auto&,
            const teuk::Plus2StageSourceTarget target) {
    const std::size_t size = target.coordinate_forcing.size();
    auto* forcing = target.coordinate_forcing.data();
    Kokkos::parallel_for(
        "plus2_pipeline_zero_source",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, size),
        ZeroComplexFunctor{forcing});
  };
}

auto inert_primary_rhs() {
  return [](const teuk::ExecutionSpace& execution, const double,
            const auto& input, const auto& output) {
    Kokkos::parallel_for(
        "plus2_pipeline_inert_primary",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0,
                                                   input.extent(0)),
        ZeroComplexFunctor{output.data()});
  };
}

struct ManufacturedResult {
  double error;
  std::vector<teuk::Complex> state;
};

class TemporaryPlus2Checkpoint {
 public:
  TemporaryPlus2Checkpoint() {
    root_ = std::filesystem::temp_directory_path() /
            ("teuk-plus2-pipeline-checkpoint-" +
             std::to_string(std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count()));
    std::filesystem::create_directory(root_);
  }
  ~TemporaryPlus2Checkpoint() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }
  std::filesystem::path path() const { return root_ / "companion.bin"; }
  std::filesystem::path mismatched_path() const {
    return root_ / "mismatched-companion.bin";
  }
  std::filesystem::path primary_path() const { return root_ / "primary"; }
  std::filesystem::path other_primary_path() const {
    return root_ / "other-primary";
  }

 private:
  std::filesystem::path root_;
};

TEST_CASE("plus2 companion derives and rejects every physical PDE mismatch") {
  const teuk::UniformRadialGrid grid(8, 0.0, 0.38);
  const std::vector<double> theta{0.41, 1.2, 2.47};
  const auto parameters = plus2_parameters();
  constexpr double dissipation = 0.003;
  auto canonical = complete_pipeline_configuration(
      grid, theta, teuk::ReductionEvolution::FreeDamped, dissipation, 0.02);

  std::vector<std::function<void(teuk::Plus2ReplayConfiguration&)>> corrupt{
      [](auto& value) { value.mass = 1.01; },
      [](auto& value) { value.spin = -0.31; },
      [](auto& value) { value.compactification_length = 1.71; },
      [](auto& value) {
        value.radial_coordinates[3] = std::nextafter(
            value.radial_coordinates[3],
            std::numeric_limits<double>::infinity());
      },
      [](auto& value) { value.radial_coordinates[0] = -0.0; },
      [](auto& value) {
        value.theta_coordinates[1] = std::nextafter(
            value.theta_coordinates[1],
            std::numeric_limits<double>::infinity());
      },
      [](auto& value) { value.reduction_mode = "stage_constrained"; },
      [](auto& value) { value.reduction_damping = 1.41; },
      [](auto& value) { value.dissipation = 0.004; },
      [](auto& value) {
        value.source_normalization =
            static_cast<teuk::Plus2SourceNormalization>(99);
      }};
  for (std::size_t index = 0; index < corrupt.size(); ++index) {
    auto candidate = canonical;
    corrupt[index](candidate);
    pipeline_allocations = 0;
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        count_pipeline_allocation);
    bool rejected = false;
    try {
      teuk::Plus2CompanionPipeline pipeline(
          candidate, grid, parameters, theta,
          teuk::ReductionEvolution::FreeDamped, dissipation,
          "plus2_physical_mismatch_" + std::to_string(index));
      static_cast<void>(pipeline);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    CHECK(rejected);
    CHECK(pipeline_allocations == 0);
  }

  auto partial = pipeline_configuration(grid.size(), theta.size());
  partial.mass = parameters.mass;
  bool partial_rejected = false;
  try {
    teuk::Plus2CompanionPipeline pipeline(
        partial, grid, parameters, theta,
        teuk::ReductionEvolution::FreeDamped, dissipation,
        "plus2_partial_physical_provenance");
    static_cast<void>(pipeline);
  } catch (const std::invalid_argument&) {
    partial_rejected = true;
  }
  CHECK(partial_rejected);

  auto derived = pipeline_configuration(grid.size(), theta.size());
  derived.time_step = 0.02;
  teuk::Plus2CompanionPipeline pipeline(
      derived, grid, parameters, theta,
      teuk::ReductionEvolution::FreeDamped, dissipation,
      "plus2_derived_physical_provenance");
  CHECK(pipeline.configuration().mass == parameters.mass);
  CHECK(pipeline.configuration().spin == parameters.spin);
  CHECK(pipeline.configuration().compactification_length ==
        parameters.compactification_length);
  CHECK(pipeline.configuration().radial_coordinates ==
        canonical.radial_coordinates);
  CHECK(pipeline.configuration().theta_coordinates == theta);
  CHECK(pipeline.configuration().reduction_mode == "free_damped");
  CHECK(pipeline.configuration().reduction_damping ==
        parameters.reduction_damping);
  CHECK(pipeline.configuration().dissipation == dissipation);
}

TEST_CASE("plus2 timestep mismatch rejects before callbacks or state mutation") {
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(8, 0.0, 0.3);
  constexpr double configured_step = 0.03;
  auto configuration = pipeline_configuration(8, 1);
  configuration.time_step = configured_step;
  teuk::Plus2CompanionPipeline pipeline(
      configuration, grid, plus2_parameters(), {0.91},
      teuk::ReductionEvolution::FreeDamped);
  pipeline.initialize_zero(execution);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_wrong_step_primary", 2);
  Kokkos::parallel_for(
      "plus2_wrong_step_initialize_primary",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, 2),
      KOKKOS_LAMBDA(const std::size_t index) {
        primary(index) = teuk::Complex(0.3 + 0.2 * index, -0.1 * index);
      });
  Kokkos::deep_copy(execution, pipeline.companion_state(),
                    teuk::Complex(0.4, -0.2));
  execution.fence("initialize wrong-step sentinels");
  const auto primary_before = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, primary);
  const auto companion_before = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.companion_state());
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(2);
  int primary_calls = 0;
  int source_calls = 0;
  int angular_calls = 0;
  const auto primary_rhs = [&](const auto& e, const double, const auto& input,
                               const auto& output) {
    ++primary_calls;
    Kokkos::parallel_for(
        "plus2_wrong_step_primary_rhs",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(e, 0, input.extent(0)),
        ZeroComplexFunctor{output.data()});
  };
  const auto source = [&](const auto& e, const double, const auto&,
                          const teuk::Plus2StageSourceTarget target) {
    ++source_calls;
    Kokkos::parallel_for(
        "plus2_wrong_step_source",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(
            e, 0, target.coordinate_forcing.size()),
        ZeroComplexFunctor{target.coordinate_forcing.data()});
  };
  const auto angular = [&](const auto& e, const double, const auto&,
                           const auto& laplacian) {
    ++angular_calls;
    Kokkos::parallel_for(
        "plus2_wrong_step_angular",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(e, 0, laplacian.size()),
        ZeroComplexFunctor{laplacian.data()});
  };
  bool rejected = false;
  try {
    pipeline.advance_concurrent_validation_only(
        execution, primary, 0.0,
        std::nextafter(configured_step,
                       std::numeric_limits<double>::infinity()),
        {}, primary_rhs, source, angular, workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(primary_calls == 0);
  CHECK(source_calls == 0);
  CHECK(angular_calls == 0);
  const auto primary_after = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, primary);
  const auto companion_after = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.companion_state());
  for (std::size_t index = 0; index < primary_before.extent(0); ++index) {
    CHECK(primary_after(index) == primary_before(index));
  }
  for (std::size_t index = 0; index < companion_before.extent(0); ++index) {
    CHECK(companion_after(index) == companion_before(index));
  }

}

TEST_CASE("plus2 source authority rejects before dt callbacks or mutation") {
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(8, 0.0, 0.38);
  teuk::Plus2CompanionPipeline pipeline(
      pipeline_configuration(8, 1), grid, plus2_parameters(), {0.91},
      teuk::ReductionEvolution::FreeDamped, 0.0,
      "plus2_wrong_source_authority");
  pipeline.initialize_zero(execution);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_wrong_source_primary", 2);
  Kokkos::deep_copy(execution, primary, teuk::Complex(2.0, -1.0));
  Kokkos::deep_copy(execution, pipeline.companion_state(),
                    teuk::Complex(-3.0, 4.0));
  execution.fence("set wrong-source mutation sentinels");
  const auto primary_before = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, primary);
  const auto companion_before = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.companion_state());
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(2);
  int primary_calls = 0;
  int source_calls = 0;
  int angular_calls = 0;
  const auto primary_rhs = [&](const auto&, const double, const auto&,
                               const auto&) { ++primary_calls; };
  auto source = [&](const auto&, const double, const auto&,
                    const teuk::Plus2StageSourceTarget) { ++source_calls; };
  const auto angular = [&](const auto&, const double, const auto&,
                           const auto&) { ++angular_calls; };
  bool rejected = false;
  try {
    pipeline.advance_concurrent(execution, primary, 0.0, 0.02, {},
                                primary_rhs, source, angular, workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(pipeline.configuration().time_step == 0.0);
  CHECK(primary_calls == 0);
  CHECK(source_calls == 0);
  CHECK(angular_calls == 0);
  const auto primary_after = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, primary);
  const auto companion_after = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.companion_state());
  for (std::size_t index = 0; index < primary_before.extent(0); ++index) {
    CHECK(primary_after(index) == primary_before(index));
  }
  for (std::size_t index = 0; index < companion_before.extent(0); ++index) {
    CHECK(companion_after(index) == companion_before(index));
  }

  SpoofedNormalizedSourceAdapter spoofed{&source_calls};
  rejected = false;
  try {
    pipeline.advance_concurrent(execution, primary, 0.0, 0.02, {},
                                primary_rhs, spoofed, angular, workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(pipeline.configuration().time_step == 0.0);
  CHECK(primary_calls == 0);
  CHECK(source_calls == 0);
  CHECK(angular_calls == 0);
}

TEST_CASE("plus2 physical checkpoint derives and binds the actual PDE") {
  constexpr std::size_t radial_count = 24;
  constexpr std::size_t theta_count = 3;
  constexpr double step = 0.02;
  constexpr double dissipation = 0.003;
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.38);
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  std::vector<double> theta(theta_count);
  for (std::size_t index = 0; index < theta_count; ++index) {
    theta[index] = angular_grid.theta(index);
  }
  teuk::Plus2SpatialThetaView cos_theta("checkpoint_source_cos", theta_count);
  teuk::Plus2SpatialThetaView sin_theta("checkpoint_source_sin", theta_count);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (std::size_t index = 0; index < theta_count; ++index) {
    host_cos(index) = std::cos(theta[index]);
    host_sin(index) = std::sin(theta[index]);
  }
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  Kokkos::deep_copy(execution, sin_theta, host_sin);
  const teuk::ModeRegistry source_registry({-1, 0, 1}, {-1, 1}, {0});
  const teuk::KerrParameters source_background{
      plus2_parameters().mass, plus2_parameters().spin,
      plus2_parameters().compactification_length};
  teuk::Plus2LiveSourceComposition<> source_composition(
      execution, source_registry, grid, source_background, 2, theta_count,
      cos_theta, sin_theta, "checkpoint_source_authority",
      teuk::RadialDiscretization::D105);
  TemporaryPlus2Checkpoint checkpoint;
  const teuk::ModeRegistry primary_registry({-1, 0, 1});
  const teuk::UniformRadialGrid primary_grid(radial_count, 0.0, 0.38);
  const teuk::KerrParameters primary_background{
      plus2_parameters().mass, plus2_parameters().spin,
      plus2_parameters().compactification_length};
  constexpr int primary_ell_max = 3;
  constexpr int primary_theta_count = 6;
  teuk::SpatialPipeline primary_pipeline(
      execution, primary_registry, primary_grid, primary_ell_max,
      primary_theta_count, primary_background,
      plus2_parameters().reduction_damping, 0.0,
      teuk::ReductionEvolution::FreeDamped,
      "plus2_physical_checkpoint_primary", {},
      teuk::RadialDiscretization::D105);
  teuk::PipelineCheckpointConfiguration primary_configuration;
  primary_configuration.background = primary_background;
  primary_configuration.ell_max_first = primary_ell_max;
  primary_configuration.ell_max_second = primary_ell_max;
  primary_configuration.theta_nodes = primary_theta_count;
  primary_configuration.reduction_damping =
      plus2_parameters().reduction_damping;
  primary_configuration.dissipation = 0.0;
  primary_configuration.reduction = teuk::ReductionEvolution::FreeDamped;
  primary_configuration.time_step = step;
  primary_configuration.source_policy = primary_pipeline.source_policy();
  primary_configuration.radial_discretization =
      teuk::RadialDiscretization::D105;
  const auto primary_identity = teuk::write_pipeline_checkpoint(
      execution, checkpoint.primary_path(), primary_pipeline,
      primary_registry, primary_configuration, {2.0 * step, 2});
  const auto source_authority =
      source_composition.source_provenance_authority();
  std::vector<teuk::Complex> expected;
  {
    auto configuration = pipeline_configuration(radial_count, theta_count);
    configuration.time_step = step;
    configuration.radial_discretization = teuk::RadialDiscretization::D105;
    teuk::Plus2CompanionPipeline pipeline(
        configuration, grid, plus2_parameters(), theta,
        teuk::ReductionEvolution::FreeDamped, dissipation,
        "plus2_physical_checkpoint_writer", teuk::RadialDiscretization::D105);
    pipeline.initialize_zero(execution);
    const auto state = pipeline.companion_state();
    Kokkos::parallel_for(
        "plus2_physical_checkpoint_state",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, state.size()),
        KOKKOS_LAMBDA(const std::size_t index) {
          state(index) = teuk::Complex(
              0.01 * static_cast<double>(index + 1),
              -0.003 * static_cast<double>(index));
        });
    execution.fence("initialize physical plus2 checkpoint state");
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           state);
    expected.resize(host.extent(0));
    for (std::size_t index = 0; index < expected.size(); ++index) {
      expected[index] = host(index);
    }
    Kokkos::View<teuk::Complex*, teuk::MemorySpace> wrong_primary(
        "wrong_primary_checkpoint_state",
        primary_pipeline.storage().flat_state().extent(0));
    Kokkos::deep_copy(execution, wrong_primary,
                      primary_pipeline.storage().flat_state());
    Kokkos::parallel_for(
        "alter_wrong_primary_checkpoint_state",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          wrong_primary(0) += teuk::Complex(1.0, 0.0);
        });
    bool wrong_primary_rejected = false;
    try {
      static_cast<void>(pipeline.save_checkpoint(
          execution, checkpoint.mismatched_path(), {2.0 * step, 2}, {},
          primary_identity, wrong_primary, source_authority));
    } catch (const std::invalid_argument&) {
      wrong_primary_rejected = true;
    }
    CHECK(wrong_primary_rejected);
    CHECK(!std::filesystem::exists(checkpoint.mismatched_path()));
    const auto saved = pipeline.save_checkpoint(
        execution, checkpoint.path(), {2.0 * step, 2}, {}, primary_identity,
        primary_pipeline.storage().flat_state(), source_authority);
    CHECK(saved.mass == plus2_parameters().mass);
    CHECK(saved.spin == plus2_parameters().spin);
    CHECK(saved.compactification_length ==
          plus2_parameters().compactification_length);
    CHECK(saved.radial_coordinates ==
          pipeline.configuration().radial_coordinates);
    CHECK(saved.theta_coordinates == theta);
    CHECK(saved.time_step == step);
    CHECK(saved.primary_checkpoint_identity ==
          primary_identity.content_identity());
  }

  auto restore_configuration =
      pipeline_configuration(radial_count, theta_count);
  restore_configuration.mode = teuk::Plus2RunMode::Replay;
  restore_configuration.primary_value_count =
      primary_pipeline.storage().flat_state().extent(0);
  restore_configuration.initial_policy = teuk::Plus2InitialPolicy::Checkpoint;
  restore_configuration.checkpoint = checkpoint.path();
  restore_configuration.time_step = step;
  restore_configuration.radial_discretization =
      teuk::RadialDiscretization::D105;
  teuk::Plus2CompanionPipeline restored(
      restore_configuration, grid, plus2_parameters(), theta,
      teuk::ReductionEvolution::FreeDamped, dissipation,
      "plus2_physical_checkpoint_reader", teuk::RadialDiscretization::D105);
  const auto restored_metadata = restored.initialize_checkpoint(
      execution, primary_pipeline.storage().flat_state(), primary_identity,
      source_authority);
  CHECK(restored_metadata.progress.time == 2.0 * step);
  CHECK(restored_metadata.progress.step == 2);
  const auto restored_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, restored.companion_state());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    CHECK(restored_host(index) == expected[index]);
  }

  teuk::SpatialPipeline other_primary_pipeline(
      execution, primary_registry, primary_grid, primary_ell_max,
      primary_theta_count, primary_background,
      plus2_parameters().reduction_damping, 0.0,
      teuk::ReductionEvolution::FreeDamped,
      "plus2_physical_checkpoint_other_primary", {},
      teuk::RadialDiscretization::D105);
  Kokkos::deep_copy(execution, other_primary_pipeline.storage().flat_state(),
                    teuk::Complex(1.0, 0.0));
  const auto other_primary_receipt = teuk::write_pipeline_checkpoint(
      execution, checkpoint.other_primary_path(), other_primary_pipeline,
      primary_registry, primary_configuration, {2.0 * step, 2});
  teuk::Plus2CompanionPipeline wrong_receipt_restore(
      restore_configuration, grid, plus2_parameters(), theta,
      teuk::ReductionEvolution::FreeDamped, dissipation,
      "plus2_physical_checkpoint_wrong_receipt",
      teuk::RadialDiscretization::D105);
  Kokkos::deep_copy(execution, wrong_receipt_restore.companion_state(),
                    teuk::Complex(9.0, -5.0));
  execution.fence("set wrong receipt companion sentinel");
  bool wrong_receipt_rejected = false;
  try {
    static_cast<void>(wrong_receipt_restore.initialize_checkpoint(
        execution, other_primary_pipeline.storage().flat_state(),
        other_primary_receipt, source_authority));
  } catch (const std::runtime_error&) {
    wrong_receipt_rejected = true;
  }
  CHECK(wrong_receipt_rejected);
  const auto unchanged_wrong_receipt = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wrong_receipt_restore.companion_state());
  for (std::size_t index = 0; index < unchanged_wrong_receipt.extent(0);
       ++index) {
    CHECK(unchanged_wrong_receipt(index) == teuk::Complex(9.0, -5.0));
  }
  const auto recovered_metadata = wrong_receipt_restore.initialize_checkpoint(
      execution, primary_pipeline.storage().flat_state(), primary_identity,
      source_authority);
  CHECK(recovered_metadata.progress.time == 2.0 * step);
  const auto recovered_companion = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wrong_receipt_restore.companion_state());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    CHECK(recovered_companion(index) == expected[index]);
  }

  teuk::Plus2CompanionPipeline wrong_primary_restore(
      restore_configuration, grid, plus2_parameters(), theta,
      teuk::ReductionEvolution::FreeDamped, dissipation,
      "plus2_physical_checkpoint_wrong_primary",
      teuk::RadialDiscretization::D105);
  Kokkos::deep_copy(execution, wrong_primary_restore.companion_state(),
                    teuk::Complex(8.0, -3.0));
  Kokkos::deep_copy(
      execution, wrong_primary_restore.orchestrator().replay_primary_state(),
      teuk::Complex(-7.0, 2.0));
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> unrelated_primary(
      "plus2_unrelated_primary",
      primary_pipeline.storage().flat_state().extent(0));
  Kokkos::deep_copy(execution, unrelated_primary, teuk::Complex(1.0, 0.0));
  execution.fence("set wrong primary receipt sentinels");
  bool wrong_view_rejected = false;
  try {
    static_cast<void>(wrong_primary_restore.initialize_checkpoint(
        execution, unrelated_primary, primary_identity, source_authority));
  } catch (const std::invalid_argument&) {
    wrong_view_rejected = true;
  }
  CHECK(wrong_view_rejected);
  const auto unchanged_primary = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{},
      wrong_primary_restore.orchestrator().replay_primary_state());
  const auto unchanged_companion = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, wrong_primary_restore.companion_state());
  for (std::size_t index = 0; index < unchanged_primary.extent(0); ++index) {
    CHECK(unchanged_primary(index) == teuk::Complex(-7.0, 2.0));
  }
  for (std::size_t index = 0; index < unchanged_companion.extent(0); ++index) {
    CHECK(unchanged_companion(index) == teuk::Complex(8.0, -3.0));
  }

  auto mismatched_parameters = plus2_parameters();
  mismatched_parameters.spin = 0.32;
  teuk::Plus2CompanionPipeline mismatched(
      restore_configuration, grid, mismatched_parameters, theta,
      teuk::ReductionEvolution::FreeDamped, dissipation,
      "plus2_physical_checkpoint_wrong_pde",
      teuk::RadialDiscretization::D105);
  Kokkos::deep_copy(execution, mismatched.companion_state(),
                    teuk::Complex(8.0, -3.0));
  execution.fence("set physical checkpoint mismatch sentinel");
  bool rejected = false;
  try {
    static_cast<void>(mismatched.initialize_checkpoint(
        execution, primary_pipeline.storage().flat_state(), primary_identity,
        source_authority));
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  const auto unchanged = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, mismatched.companion_state());
  for (std::size_t index = 0; index < unchanged.extent(0); ++index) {
    CHECK(unchanged(index) == teuk::Complex(8.0, -3.0));
  }
}

TEST_CASE("plus2 companion rejects a replay radial scheme mismatch") {
  auto configuration = pipeline_configuration(16, 1);
  configuration.radial_discretization = teuk::RadialDiscretization::D42;
  const teuk::UniformRadialGrid grid(16, 0.0, 0.38);
  pipeline_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_pipeline_allocation);
  bool rejected = false;
  try {
    teuk::Plus2CompanionPipeline pipeline(
        configuration, grid, plus2_parameters(), {0.91},
        teuk::ReductionEvolution::FreeDamped, 0.0,
        "plus2_radial_scheme_mismatch", teuk::RadialDiscretization::D84);
    static_cast<void>(pipeline);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(rejected);
  CHECK(pipeline_allocations == 0);
}

TEST_CASE("plus2 companion accepts an explicitly matching D10-5 scheme") {
  auto configuration = pipeline_configuration(24, 1);
  configuration.radial_discretization = teuk::RadialDiscretization::D105;
  const teuk::UniformRadialGrid grid(24, 0.0, 0.38);
  teuk::Plus2CompanionPipeline pipeline(
      configuration, grid, plus2_parameters(), {0.91},
      teuk::ReductionEvolution::FreeDamped, 0.0, "plus2_d105_selection",
      teuk::RadialDiscretization::D105);
  CHECK(pipeline.configuration().radial_discretization ==
        teuk::RadialDiscretization::D105);
}

ManufacturedResult evolve_manufactured(const int steps) {
  constexpr std::size_t radial_count = 8;
  constexpr double final_time = 0.8;
  constexpr double lambda = 1.1;
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.38);
  const auto parameters = plus2_parameters();
  teuk::Plus2CompanionPipeline pipeline(
      pipeline_configuration(radial_count, 1), grid, parameters, {0.91},
      teuk::ReductionEvolution::FreeDamped, 0.0,
      "plus2_manufactured_pipeline");
  pipeline.initialize_zero(execution);

  const auto state = pipeline.companion_storage().state();
  Kokkos::parallel_for(
      "plus2_manufactured_initial",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, radial_count),
      KOKKOS_LAMBDA(const std::size_t radial) {
        teuk::TeukolskyParameters point_parameters = parameters;
        point_parameters.azimuthal_mode = 0;
        const auto c = teuk::teukolsky_coefficients(
            point_parameters, grid.coordinate(radial), 0.91);
        state(0, P, radial, 0) =
            (lambda * c.time + c.definition) * teuk::Complex(1.0, 0.0);
        state(0, Q, radial, 0) = teuk::Complex{};
        state(0, Psi, radial, 0) = teuk::Complex(1.0, 0.0);
      });

  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_manufactured_primary", 1);
  Kokkos::deep_copy(execution, primary, teuk::Complex(0.2, -0.1));
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace>
      primary_workspace(1);
  auto primary_rhs = inert_primary_rhs();
  const auto angular_action = [](const teuk::ExecutionSpace& e,
                                 const double t, const auto& s,
                                 const auto& lap) {
    zero_angular_action(e, t, s, lap);
  };
  const auto source_adapter =
      [=](const teuk::ExecutionSpace& stage_execution,
          const double stage_time, const auto&,
          const teuk::Plus2StageSourceTarget target) {
        const bool active = target.accepted_activation.active;
        const auto forcing = target.coordinate_forcing;
        Kokkos::parallel_for(
            "plus2_manufactured_source",
            Kokkos::RangePolicy<teuk::ExecutionSpace>(stage_execution, 0,
                                                       radial_count),
            KOKKOS_LAMBDA(const std::size_t radial) {
              teuk::TeukolskyParameters point_parameters = parameters;
              point_parameters.azimuthal_mode = 0;
              const auto c = teuk::teukolsky_coefficients(
                  point_parameters, grid.coordinate(radial), 0.91);
              const teuk::Complex psi =
                  Kokkos::exp(lambda * stage_time);
              const teuk::Complex p =
                  (lambda * c.time + c.definition) * psi;
              forcing(0, radial, 0) =
                  active ? lambda * p - c.psi * psi : teuk::Complex{};
            });
      };

  const double step = final_time / static_cast<double>(steps);
  const teuk::SourceActivationState activation{true, 0.0, 1, 0.0};
  for (int n = 0; n < steps; ++n) {
    pipeline.advance_concurrent_validation_only(
        execution, primary, step * static_cast<double>(n), step, activation,
        primary_rhs, source_adapter, angular_action, primary_workspace);
  }
  execution.fence("finish plus2 manufactured evolution");
  const auto host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.companion_state());
  const double exact = std::exp(lambda * final_time);
  double error = 0.0;
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    error = std::max(error, Kokkos::abs(host(P * radial_count + radial) -
                                        (lambda * teuk::teukolsky_coefficients(
                                                      parameters,
                                                      grid.coordinate(radial),
                                                      0.91)
                                                      .time +
                                         teuk::teukolsky_coefficients(
                                             parameters, grid.coordinate(radial),
                                             0.91)
                                             .definition) *
                                            exact));
    error = std::max(
        error, Kokkos::abs(host(Psi * radial_count + radial) - exact));
    error = std::max(error,
                     Kokkos::abs(host(Q * radial_count + radial)));
  }
  std::vector<teuk::Complex> result(host.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return {error, std::move(result)};
}

std::vector<teuk::Complex> serial_manufactured_rk4(
    const teuk::UniformRadialGrid& grid,
    const teuk::TeukolskyParameters& parameters, const int steps) {
  constexpr double final_time = 0.8;
  constexpr double lambda = 1.1;
  const std::size_t count = grid.size();
  const double h = final_time / static_cast<double>(steps);
  const double inverse_spacing = 1.0 / grid.spacing();
  std::vector<teuk::Complex> y(3 * count);
  for (std::size_t radial = 0; radial < count; ++radial) {
    const auto c = teuk::teukolsky_coefficients(
        parameters, grid.coordinate(radial), 0.91);
    y[P * count + radial] =
        (lambda * c.time + c.definition) * teuk::Complex(1.0, 0.0);
    y[Q * count + radial] = teuk::Complex{};
    y[Psi * count + radial] = teuk::Complex(1.0, 0.0);
  }
  const auto rhs = [&](const double t, const std::vector<teuk::Complex>& x) {
    std::vector<teuk::Complex> result(3 * count);
    std::vector<teuk::Complex> effective_q(count);
    std::vector<teuk::Complex> psi_velocity(count);
    for (std::size_t radial = 0; radial < count; ++radial) {
      effective_q[radial] = x[Q * count + radial];
      const auto c = teuk::teukolsky_coefficients(
          parameters, grid.coordinate(radial), 0.91);
      psi_velocity[radial] = teuk::teukolsky_psi_rhs(
          c, {x[P * count + radial], effective_q[radial],
              x[Psi * count + radial]});
    }
    const teuk::Complex exact_psi(std::exp(lambda * t), 0.0);
    for (std::size_t radial = 0; radial < count; ++radial) {
      const auto c = teuk::teukolsky_coefficients(
          parameters, grid.coordinate(radial), 0.91);
      const teuk::Complex dr_q = teuk::d42_first_derivative_at(
          effective_q.data(), count, radial, inverse_spacing);
      const teuk::Complex dr_psi = teuk::d42_first_derivative_at(
          &x[Psi * count], count, radial, inverse_spacing);
      const teuk::Complex dr_velocity = teuk::d42_first_derivative_at(
          psi_velocity.data(), count, radial, inverse_spacing);
      const teuk::Complex exact_p =
          (lambda * c.time + c.definition) * exact_psi;
      const teuk::Complex forcing = lambda * exact_p - c.psi * exact_psi;
      result[P * count + radial] = teuk::teukolsky_p_rhs(
          c,
          {x[P * count + radial], effective_q[radial],
           x[Psi * count + radial]},
          dr_q, teuk::Complex{}, forcing);
      result[Q * count + radial] = teuk::teukolsky_q_rhs(
          dr_velocity, x[Q * count + radial] - dr_psi,
          parameters.reduction_damping);
      result[Psi * count + radial] = psi_velocity[radial];
    }
    return result;
  };
  for (int n = 0; n < steps; ++n) {
    const double t = h * static_cast<double>(n);
    const auto k1 = rhs(t, y);
    std::vector<teuk::Complex> stage(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      stage[i] = y[i] + 0.5 * h * k1[i];
    }
    const auto k2 = rhs(t + 0.5 * h, stage);
    for (std::size_t i = 0; i < y.size(); ++i) {
      stage[i] = y[i] + 0.5 * h * k2[i];
    }
    const auto k3 = rhs(t + 0.5 * h, stage);
    for (std::size_t i = 0; i < y.size(); ++i) {
      stage[i] = y[i] + h * k3[i];
    }
    const auto k4 = rhs(t + h, stage);
    for (std::size_t i = 0; i < y.size(); ++i) {
      y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
  }
  return y;
}

}  // namespace

TEST_CASE("plus2 driven companion has fourth order time convergence") {
  const auto coarse = evolve_manufactured(4);
  const auto medium = evolve_manufactured(8);
  const auto fine = evolve_manufactured(16);
  CHECK(coarse.error / medium.error > 13.0);
  CHECK(medium.error / fine.error > 13.0);
  CHECK(fine.error < 2.0e-6);
}

TEST_CASE("plus2 device pipeline agrees with independent serial RK4 oracle") {
  constexpr int steps = 11;
  const auto device = evolve_manufactured(steps);
  const auto parameters = plus2_parameters();
  const teuk::UniformRadialGrid grid(8, 0.0, 0.38);
  const auto serial = serial_manufactured_rk4(grid, parameters, steps);
  for (std::size_t i = 0; i < serial.size(); ++i) {
    CHECK_COMPLEX_NEAR(device.state[i], serial[i], 3.0e-12);
  }
}

TEST_CASE("plus2 pipeline uses exact common stage times and activation snapshot") {
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(8, 0.0, 0.3);
  teuk::Plus2CompanionPipeline pipeline(
      pipeline_configuration(8, 1), grid, plus2_parameters(), {1.2},
      teuk::ReductionEvolution::FreeDamped);
  pipeline.initialize_zero(execution);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_stage_time_primary", 1);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(1);
  std::vector<double> primary_times;
  std::vector<double> source_times;
  std::vector<double> angular_times;
  std::vector<teuk::SourceActivationState> snapshots;
  primary_times.reserve(4);
  source_times.reserve(4);
  angular_times.reserve(4);
  snapshots.reserve(4);
  const auto primary_rhs = [&](const teuk::ExecutionSpace& e, const double t,
                               const auto& in, const auto& out) {
    primary_times.push_back(t);
    Kokkos::parallel_for(
        "plus2_stage_time_primary_rhs",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(e, 0, in.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) { out(i) = teuk::Complex{}; });
  };
  const auto source = [&](const teuk::ExecutionSpace& e, const double t,
                          const auto&,
                          const teuk::Plus2StageSourceTarget target) {
    source_times.push_back(t);
    snapshots.push_back(target.accepted_activation);
    const auto forcing = target.coordinate_forcing;
    Kokkos::parallel_for(
        "plus2_stage_time_source",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(e, 0, forcing.size()),
        KOKKOS_LAMBDA(const std::size_t i) {
          forcing.data()[i] = teuk::Complex{};
        });
  };
  const auto angular = [&](const teuk::ExecutionSpace& e, const double t,
                           const auto& s, const auto& lap) {
    angular_times.push_back(t);
    zero_angular_action(e, t, s, lap);
  };
  const teuk::SourceActivationState activation{true, 0.2, 7, 0.34};
  pipeline.advance_concurrent_validation_only(
      execution, primary, 0.35, 0.08, activation, primary_rhs, source, angular,
      workspace);
  execution.fence("finish plus2 stage-time test");
  const std::array<double, 4> expected{0.35, 0.39, 0.39, 0.43};
  CHECK(primary_times.size() == expected.size());
  CHECK(source_times.size() == expected.size());
  CHECK(angular_times.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK_NEAR(primary_times[i], expected[i], 2.0e-16);
    CHECK_NEAR(source_times[i], expected[i], 2.0e-16);
    CHECK_NEAR(angular_times[i], expected[i], 2.0e-16);
    CHECK(snapshots[i].active == activation.active);
    CHECK(snapshots[i].activation_time == activation.activation_time);
    CHECK(snapshots[i].consecutive_passes == activation.consecutive_passes);
    CHECK(snapshots[i].last_eligibility_time ==
          activation.last_eligibility_time);
  }
}

TEST_CASE("plus2 companion cannot feed back into the primary trajectory") {
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(8, 0.0, 0.3);
  teuk::Plus2CompanionPipeline pipeline(
      pipeline_configuration(8, 1), grid, plus2_parameters(), {0.8},
      teuk::ReductionEvolution::FreeDamped);
  pipeline.initialize_zero(execution);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> coupled("coupled_primary",
                                                          5);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> standalone(
      "standalone_primary", 5);
  Kokkos::parallel_for(
      "plus2_no_feedback_initial",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, 5),
      KOKKOS_LAMBDA(const std::size_t i) {
        coupled(i) = teuk::Complex(0.2 * static_cast<double>(i + 1),
                                   -0.03 * static_cast<double>(i));
        standalone(i) = coupled(i);
      });
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> coupled_ws(5);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> standalone_ws(
      5);
  const auto rhs = [](const teuk::ExecutionSpace& e, const double t,
                      const auto& in, const auto& out) {
    Kokkos::parallel_for(
        "plus2_no_feedback_primary_rhs",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(e, 0, in.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) {
          out(i) = (0.17 + 0.01 * static_cast<double>(i)) * in(i) +
                   teuk::Complex(t, -0.2 * t);
        });
  };
  auto source = zero_source_adapter();
  const auto angular = [](const teuk::ExecutionSpace& e, const double t,
                          const auto& s, const auto& lap) {
    zero_angular_action(e, t, s, lap);
  };
  const teuk::SourceActivationState inactive{};
  for (int n = 0; n < 5; ++n) {
    const double time = 0.03 * static_cast<double>(n);
    pipeline.advance_concurrent_validation_only(
        execution, coupled, time, 0.03, inactive, rhs, source, angular,
        coupled_ws);
    teuk::device_classical_rk4_step(execution, standalone, time, 0.03, rhs,
                                    standalone_ws);
  }
  execution.fence("finish plus2 no-feedback comparison");
  const auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       coupled);
  const auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       standalone);
  for (std::size_t i = 0; i < hc.extent(0); ++i) CHECK(hc(i) == hs(i));
}

TEST_CASE("plus2 reduction constraint is preserved or damped as selected") {
  constexpr std::size_t radial_count = 8;
  constexpr double constraint = 0.27;
  constexpr double step = 0.04;
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.3);
  const auto parameters = plus2_parameters();
  const auto run = [&](const teuk::ReductionEvolution reduction,
                       const char* label) {
    teuk::Plus2CompanionPipeline pipeline(
        pipeline_configuration(radial_count, 1), grid, parameters, {1.0},
        reduction, 0.0, label);
    pipeline.initialize_zero(execution);
    const auto state = pipeline.companion_storage().state();
    Kokkos::parallel_for(
        "plus2_constraint_initial",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, radial_count),
        KOKKOS_LAMBDA(const std::size_t radial) {
          const double r = grid.coordinate(radial);
          state(0, P, radial, 0) = teuk::Complex(0.1 + 0.2 * r, -0.03);
          state(0, Psi, radial, 0) =
              teuk::Complex(0.4 + r + 0.3 * r * r, 0.1 * r);
          state(0, Q, radial, 0) =
              teuk::Complex(1.0 + 0.6 * r + constraint, 0.1);
        });
    Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
        "plus2_constraint_primary", 1);
    teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(1);
    auto primary_rhs = inert_primary_rhs();
    auto source = zero_source_adapter();
    const auto angular = [](const teuk::ExecutionSpace& e, const double t,
                            const auto& s, const auto& lap) {
      zero_angular_action(e, t, s, lap);
    };
    pipeline.advance_concurrent_validation_only(
        execution, primary, 0.0, step, {}, primary_rhs, source, angular,
        workspace);
    execution.fence("finish plus2 reduction evolution");
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, pipeline.companion_storage().state());
    std::vector<teuk::Complex> residual(radial_count);
    const double inverse_spacing = 1.0 / grid.spacing();
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const auto derivative = teuk::d42_first_derivative_strided_at(
          &host(0, Psi, 0, 0), radial_count, radial, inverse_spacing, 1);
      residual[radial] = host(0, Q, radial, 0) - derivative;
    }
    return residual;
  };
  const auto constrained =
      run(teuk::ReductionEvolution::StageConstrained, "plus2_constrained");
  const auto damped =
      run(teuk::ReductionEvolution::FreeDamped, "plus2_damped");
  const double x = parameters.reduction_damping * step;
  const double rk4_decay =
      1.0 - x + 0.5 * x * x - x * x * x / 6.0 + x * x * x * x / 24.0;
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    CHECK_COMPLEX_NEAR(constrained[radial], teuk::Complex(constraint, 0.0),
                       2.0e-11);
    CHECK_COMPLEX_NEAR(damped[radial],
                       teuk::Complex(constraint * rk4_decay, 0.0), 2.0e-11);
  }
}

TEST_CASE("plus2 pipeline timestep keeps allocations fences and pointers stable") {
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(8, 0.0, 0.3);
  teuk::Plus2CompanionPipeline pipeline(
      pipeline_configuration(8, 1), grid, plus2_parameters(), {1.1},
      teuk::ReductionEvolution::FreeDamped);
  pipeline.initialize_zero(execution);
  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "plus2_stability_primary", 2);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace> workspace(2);
  auto primary_rhs = inert_primary_rhs();
  auto source = zero_source_adapter();
  const auto angular = [](const teuk::ExecutionSpace& e, const double t,
                          const auto& s, const auto& lap) {
    zero_angular_action(e, t, s, lap);
  };
  const auto* state_pointer = pipeline.companion_state().data();
  const auto* forcing_pointer = pipeline.forcing().data();
  const auto* angular_pointer =
      pipeline.companion_storage().angular_laplacian().data();
  const auto* scratch_pointer =
      pipeline.companion_storage().radial_scratch().data();

  pipeline_allocations = 0;
  pipeline_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_pipeline_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_pipeline_fence);
  pipeline.advance_concurrent_validation_only(
      execution, primary, 0.0, 0.01, {}, primary_rhs, source, angular,
      workspace);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  execution.fence("finish plus2 allocation-free pipeline step");

  CHECK(pipeline_allocations == 0);
  CHECK(pipeline_fences == 0);
  CHECK(state_pointer == pipeline.companion_state().data());
  CHECK(forcing_pointer == pipeline.forcing().data());
  CHECK(angular_pointer ==
        pipeline.companion_storage().angular_laplacian().data());
  CHECK(scratch_pointer == pipeline.companion_storage().radial_scratch().data());
}

TEST_CASE("full spin plus2 D84 RHS matches host oracle for right and strided views") {
  constexpr std::size_t radial_count = 18;
  constexpr std::size_t theta_count = 2;
  constexpr double dissipation = 0.003;
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.32);
  const auto parameters = plus2_parameters();
  teuk::FullSpatialThetaView theta("plus2_d84_theta", theta_count);
  teuk::SignedModeView modes("plus2_d84_modes", 1);
  teuk::FullSpatialStateView right_state("plus2_d84_right_state", 1, 3,
                                          radial_count, theta_count);
  teuk::FullSpatialStateView right_rhs("plus2_d84_right_rhs", 1, 3,
                                        radial_count, theta_count);
  teuk::FullSpatialStateView right_scratch(
      "plus2_d84_right_scratch", 1,
      static_cast<std::size_t>(teuk::TeukolskyRadialScratch::Count),
      radial_count, theta_count);
  teuk::FullSpatialValueView right_angular("plus2_d84_right_angular", 1,
                                            radial_count, theta_count);
  teuk::FullSpatialValueView right_forcing("plus2_d84_right_forcing", 1,
                                            radial_count, theta_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      strided_state_parent("plus2_d84_strided_state_parent", 1, 7,
                           radial_count, theta_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      strided_rhs_parent("plus2_d84_strided_rhs_parent", 1, 7, radial_count,
                         theta_count);
  auto strided_state = Kokkos::subview(
      strided_state_parent, Kokkos::ALL, Kokkos::pair<std::size_t,
                                                       std::size_t>(2, 5),
      Kokkos::ALL, Kokkos::ALL);
  auto strided_rhs = Kokkos::subview(
      strided_rhs_parent, Kokkos::ALL, Kokkos::pair<std::size_t,
                                                     std::size_t>(2, 5),
      Kokkos::ALL, Kokkos::ALL);

  auto ht = Kokkos::create_mirror_view(theta);
  auto hm = Kokkos::create_mirror_view(modes);
  auto hs = Kokkos::create_mirror_view(right_state);
  auto ha = Kokkos::create_mirror_view(right_angular);
  auto hf = Kokkos::create_mirror_view(right_forcing);
  ht(0) = 0.73;
  ht(1) = 1.34;
  hm(0) = 0;
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    const double r = grid.coordinate(radial);
    for (std::size_t t = 0; t < theta_count; ++t) {
      hs(0, P, radial, t) =
          teuk::Complex(0.3 + 0.2 * r + 0.04 * r * r + 0.01 * t,
                        -0.07 + 0.03 * r * r * r);
      hs(0, Q, radial, t) =
          teuk::Complex(-0.2 + 0.5 * r * r + 0.03 * t,
                        0.08 + 0.04 * r);
      hs(0, Psi, radial, t) =
          teuk::Complex(0.4 + 0.6 * r + 0.2 * r * r * r,
                        -0.1 + 0.05 * r * r + 0.02 * t);
      ha(0, radial, t) = teuk::Complex(-0.03 + 0.02 * r, 0.01 * t);
      hf(0, radial, t) = teuk::Complex(0.04 - 0.01 * r, -0.02 - 0.01 * t);
    }
  }
  Kokkos::deep_copy(theta, ht);
  Kokkos::deep_copy(modes, hm);
  Kokkos::deep_copy(right_state, hs);
  Kokkos::parallel_for(
      "plus2_d84_copy_strided_state",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(
          execution, 0, 3 * radial_count * theta_count),
      CopyTripleToStridedFunctor{
          right_state.data(), strided_state.data(), strided_state.stride(1),
          strided_state.stride(2), strided_state.stride(3), radial_count,
          theta_count});
  Kokkos::deep_copy(right_angular, ha);
  Kokkos::deep_copy(right_forcing, hf);

  teuk::evaluate_sbp_teukolsky_full_stage_rhs(
      execution, grid, parameters, theta, modes, right_state, right_angular,
      right_forcing, teuk::ReductionEvolution::FreeDamped, right_scratch,
      right_rhs, dissipation, {}, {}, teuk::RadialDiscretization::D84);
  teuk::evaluate_sbp_teukolsky_full_stage_rhs(
      execution, grid, parameters, theta, modes, strided_state, right_angular,
      right_forcing, teuk::ReductionEvolution::FreeDamped, right_scratch,
      strided_rhs, dissipation, {}, {}, teuk::RadialDiscretization::D84);
  execution.fence("finish plus2 D84 parity evaluations");
  const auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           right_rhs);
  const auto strided = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, strided_rhs);

  const double inverse_spacing = 1.0 / grid.spacing();
  for (std::size_t t = 0; t < theta_count; ++t) {
    std::array<std::vector<teuk::Complex>, 3> state;
    for (auto& field : state) field.resize(radial_count);
    std::vector<teuk::Complex> effective_q(radial_count);
    std::vector<teuk::Complex> velocity(radial_count);
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t field = 0; field < 3; ++field) {
        state[field][radial] = hs(0, field, radial, t);
      }
      effective_q[radial] = state[Q][radial];
      const auto c = teuk::teukolsky_coefficients(
          parameters, grid.coordinate(radial), ht(t));
      velocity[radial] = teuk::teukolsky_psi_rhs(
          c, {state[P][radial], effective_q[radial], state[Psi][radial]});
    }
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const auto c = teuk::teukolsky_coefficients(
          parameters, grid.coordinate(radial), ht(t));
      const auto dr_psi = teuk::d84_first_derivative_at(
          state[Psi].data(), radial_count, radial, inverse_spacing);
      const auto dr_q = teuk::d84_first_derivative_at(
          effective_q.data(), radial_count, radial, inverse_spacing);
      const auto dr_velocity = teuk::d84_first_derivative_at(
          velocity.data(), radial_count, radial, inverse_spacing);
      const teuk::Complex constraint = state[Q][radial] - dr_psi;
      std::array<teuk::Complex, 3> expected{
          teuk::teukolsky_p_rhs(
              c,
              {state[P][radial], effective_q[radial], state[Psi][radial]},
              dr_q, ha(0, radial, t), hf(0, radial, t)),
          teuk::teukolsky_q_rhs(dr_velocity, constraint,
                                parameters.reduction_damping),
          velocity[radial]};
      for (std::size_t field = 0; field < 3; ++field) {
        expected[field] += teuk::d84_compatible_dissipation_at(
            state[field].data(), radial_count, radial, grid.spacing(),
            dissipation);
        CHECK_COMPLEX_NEAR(actual(0, field, radial, t), expected[field],
                           3.0e-12);
        CHECK_COMPLEX_NEAR(strided(0, field, radial, t), expected[field],
                           3.0e-12);
      }
    }
  }
}

TEST_CASE("spin plus2 D84 full RHS has fourth order endpoint convergence") {
  const auto error = [](const std::size_t radial_count) {
    const teuk::ExecutionSpace execution;
    const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.5);
    const auto parameters = plus2_parameters();
    constexpr double theta_value = 1.07;
    constexpr double alpha = 0.43;
    teuk::FullSpatialThetaView theta("plus2_d84_convergence_theta", 1);
    teuk::SignedModeView modes("plus2_d84_convergence_modes", 1);
    teuk::FullSpatialStateView state("plus2_d84_convergence_state", 1, 3,
                                      radial_count, 1);
    teuk::FullSpatialStateView rhs("plus2_d84_convergence_rhs", 1, 3,
                                    radial_count, 1);
    teuk::FullSpatialStateView scratch(
        "plus2_d84_convergence_scratch", 1,
        static_cast<std::size_t>(teuk::TeukolskyRadialScratch::Count),
        radial_count, 1);
    teuk::FullSpatialValueView angular("plus2_d84_convergence_angular", 1,
                                        radial_count, 1);
    teuk::FullSpatialValueView forcing("plus2_d84_convergence_forcing", 1,
                                        radial_count, 1);
    auto ht = Kokkos::create_mirror_view(theta);
    auto hm = Kokkos::create_mirror_view(modes);
    auto hs = Kokkos::create_mirror_view(state);
    ht(0) = theta_value;
    hm(0) = 0;
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double r = grid.coordinate(radial);
      const teuk::Complex psi(std::exp(r), 0.17 * std::exp(0.7 * r));
      const teuk::Complex q(std::exp(r), 0.119 * std::exp(0.7 * r));
      const teuk::Complex velocity = alpha * psi;
      const auto c = teuk::teukolsky_coefficients(parameters, r, theta_value);
      hs(0, Psi, radial, 0) = psi;
      hs(0, Q, radial, 0) = q;
      hs(0, P, radial, 0) =
          c.time * velocity - 2.0 * c.radial_advection * q +
          c.definition * psi;
    }
    Kokkos::deep_copy(theta, ht);
    Kokkos::deep_copy(modes, hm);
    Kokkos::deep_copy(state, hs);
    Kokkos::deep_copy(angular, teuk::Complex{});
    Kokkos::deep_copy(forcing, teuk::Complex{});
    teuk::evaluate_sbp_teukolsky_full_stage_rhs(
        execution, grid, parameters, theta, modes, state, angular, forcing,
        teuk::ReductionEvolution::FreeDamped, scratch, rhs, 0.0, {}, {},
        teuk::RadialDiscretization::D84);
    execution.fence("finish plus2 D84 spatial convergence sample");
    const auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         rhs);
    double maximum = 0.0;
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double r = grid.coordinate(radial);
      const teuk::Complex psi(std::exp(r), 0.17 * std::exp(0.7 * r));
      const teuk::Complex q(std::exp(r), 0.119 * std::exp(0.7 * r));
      const teuk::Complex velocity = alpha * psi;
      const teuk::Complex dr_velocity(
          alpha * std::exp(r), alpha * 0.119 * std::exp(0.7 * r));
      const teuk::Complex dr_q(std::exp(r),
                               0.0833 * std::exp(0.7 * r));
      const auto c = teuk::teukolsky_coefficients(parameters, r, theta_value);
      const teuk::Complex expected_p = teuk::teukolsky_p_rhs(
          c,
          {c.time * velocity - 2.0 * c.radial_advection * q +
               c.definition * psi,
           q, psi},
          dr_q, teuk::Complex{}, teuk::Complex{});
      maximum = std::max(maximum,
                         Kokkos::abs(hr(0, P, radial, 0) - expected_p));
      maximum = std::max(maximum,
                         Kokkos::abs(hr(0, Q, radial, 0) - dr_velocity));
      maximum = std::max(maximum,
                         Kokkos::abs(hr(0, Psi, radial, 0) - velocity));
    }
    return maximum;
  };
  const double coarse = error(18);
  const double medium = error(34);
  const double fine = error(66);
  CHECK(coarse / medium > 12.0);
  CHECK(medium / fine > 12.0);
  CHECK(fine < 2.0e-6);
}
