#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/angular_coordinator.hpp"
#include "teuk/device_rk4.hpp"
#include "teuk/plus2_companion_pipeline.hpp"
#include "teuk/plus2_live_source_composition.hpp"
#include "teuk/routeb_angular_jet_coordinator.hpp"

namespace {

using C = teuk::Complex;
using Execution = teuk::ExecutionSpace;

constexpr std::size_t replay_radial_count = 25;
constexpr std::size_t replay_theta_count = 12;
constexpr std::size_t replay_primary_field_count = 10;
constexpr int replay_ell_max = 5;
constexpr double replay_step = 2.0e-4;
constexpr int replay_steps = 2;

int physical_replay_allocations = 0;
int physical_replay_fences = 0;

void count_physical_replay_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                      const void*, std::uint64_t) {
  ++physical_replay_allocations;
}

void count_physical_replay_fence(const char*, std::uint32_t,
                                 std::uint64_t*) {
  ++physical_replay_fences;
}

using FlatView = Kokkos::View<C*, teuk::MemorySpace>;
using StampView = Kokkos::View<std::uint64_t***, Kokkos::LayoutRight,
                               teuk::MemorySpace>;
using StageView =
    Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace>;
using UnmanagedStateView =
    Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using UnmanagedSingleFieldView =
    Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

std::uint64_t bits(const double value) {
  return std::bit_cast<std::uint64_t>(value);
}

bool bitwise_equal(const C left, const C right) {
  return bits(left.real()) == bits(right.real()) &&
         bits(left.imag()) == bits(right.imag());
}

struct PackRouteBLevelOneRhs {
  const C* primary;
  const C* reconstruction;
  const std::uint64_t* primary_stamps;
  const std::uint64_t* reconstruction_stamps;
  C* output;
  std::size_t mode_count;
  std::size_t radial_count;
  std::size_t theta_count;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t primary_stamp =
        ((mode_count + mode) * radial_count + radial) * theta_count + theta;
    const bool valid = primary_stamps[primary_stamp] == generation &&
                       reconstruction_stamps[primary_stamp] == generation;
    for (std::size_t field = 0; field < 3; ++field) {
      const std::size_t input =
          (((mode_count + mode) * 3 + field) * radial_count + radial) *
              theta_count +
          theta;
      const std::size_t destination =
          ((mode * replay_primary_field_count + field) * radial_count +
           radial) *
              theta_count +
          theta;
      output[destination] = valid ? primary[input] : C{};
    }
    for (std::size_t field = 0; field < 7; ++field) {
      const std::size_t input =
          (((mode_count + mode) * 7 + field) * radial_count + radial) *
              theta_count +
          theta;
      const std::size_t destination =
          ((mode * replay_primary_field_count + field + 3) * radial_count +
           radial) *
              theta_count +
          theta;
      output[destination] = valid ? reconstruction[input] : C{};
    }
  }
};

struct SplitRouteBPrimaryStage {
  const C* input;
  C* primary;
  C* reconstruction;
  std::uint64_t* primary_stamps;
  std::uint64_t* reconstruction_stamps;
  std::size_t radial_count;
  std::size_t theta_count;
  std::uint64_t generation;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t stamp = (mode * radial_count + radial) * theta_count +
                              theta;
    primary_stamps[stamp] = generation;
    reconstruction_stamps[stamp] = generation;
    for (std::size_t field = 0; field < 3; ++field) {
      const std::size_t source =
          ((mode * replay_primary_field_count + field) * radial_count +
           radial) *
              theta_count +
          theta;
      const std::size_t destination =
          ((mode * 3 + field) * radial_count + radial) * theta_count + theta;
      primary[destination] = input[source];
    }
    for (std::size_t field = 0; field < 7; ++field) {
      const std::size_t source =
          ((mode * replay_primary_field_count + field + 3) * radial_count +
           radial) *
              theta_count +
          theta;
      const std::size_t destination =
          ((mode * 7 + field) * radial_count + radial) * theta_count + theta;
      reconstruction[destination] = input[source];
    }
  }
};

static_assert(std::is_trivially_copyable_v<PackRouteBLevelOneRhs>);
static_assert(std::is_trivially_copyable_v<SplitRouteBPrimaryStage>);

class RouteBPhysicalStageGraph {
 public:
  RouteBPhysicalStageGraph(
      const Execution& execution, const teuk::ModeRegistry& registry,
      const teuk::UniformRadialGrid& grid,
      const teuk::TeukolskyParameters& primary_parameters,
      teuk::Plus2SpatialThetaView cos_theta,
      teuk::Plus2SpatialThetaView sin_theta,
      teuk::Plus2RouteBCurvatureSpatialProvider<Execution>& curvature,
      teuk::Plus2SourcePrimitiveSpatialProducer<Execution>& primitive,
      teuk::Plus2SourceOuterSpatialProducer<Execution>& outer,
      teuk::Plus2LiveSourceComposition<Execution>& composition)
      : registry_(registry),
        grid_(grid),
        cos_theta_(cos_theta),
        sin_theta_(sin_theta),
        coordinator_(execution, registry, grid, replay_ell_max,
                     replay_theta_count, primary_parameters,
                     "routeb_physical_replay_graph"),
        stage_primary_("routeb_physical_replay_stage_primary",
                       registry.size(), 3, replay_radial_count,
                       replay_theta_count),
        stage_reconstruction_("routeb_physical_replay_stage_reconstruction",
                              registry.size(), 7, replay_radial_count,
                              replay_theta_count),
        primary_stamps_("routeb_physical_replay_primary_stamps",
                        registry.size(), replay_radial_count,
                        replay_theta_count),
        reconstruction_stamps_(
            "routeb_physical_replay_reconstruction_stamps", registry.size(),
            replay_radial_count, replay_theta_count),
        curvature_(curvature),
        primitive_(primitive),
        outer_(outer),
        composition_(composition) {
    primary_stage_times_.reserve(4 * replay_steps);
    source_stage_times_.reserve(4 * replay_steps);
    source_generations_.reserve(4 * replay_steps);
  }

  template <class InputView, class OutputView>
  void evaluate_primary_rhs(const Execution& execution,
                            const double stage_time, const InputView& input,
                            const OutputView& output) {
    CHECK(input.extent(0) == value_count());
    CHECK(output.extent(0) == value_count());
    ++generation_;
    last_primary_stage_ = input.data();
    primary_stage_times_.push_back(stage_time);
    Kokkos::parallel_for(
        "split_routeb_physical_primary_stage",
        Kokkos::RangePolicy<Execution>(
            execution, 0,
            registry_.size() * replay_radial_count * replay_theta_count),
        SplitRouteBPrimaryStage{input.data(), stage_primary_.data(),
                                stage_reconstruction_.data(),
                                primary_stamps_.data(),
                                reconstruction_stamps_.data(),
                                replay_radial_count, replay_theta_count,
                                generation_});
    coordinator_.initialize(execution, stage_primary_, primary_stamps_,
                            stage_reconstruction_, reconstruction_stamps_,
                            generation_);
    coordinator_.advance_to_h4(execution, generation_);
    Kokkos::parallel_for(
        "pack_routeb_physical_primary_rhs",
        Kokkos::RangePolicy<Execution>(
            execution, 0,
            registry_.size() * replay_radial_count * replay_theta_count),
        PackRouteBLevelOneRhs{
            coordinator_.primary_values().data(),
            coordinator_.reconstruction_values().data(),
            coordinator_.primary_stamps().data(),
            coordinator_.reconstruction_stamps().data(), output.data(),
            registry_.size(), replay_radial_count, replay_theta_count,
            generation_});
  }

  template <class PrimaryStage>
  void evaluate_source(const Execution& execution, const double stage_time,
                       const PrimaryStage& primary_stage,
                       const teuk::Plus2StageSourceTarget& target) {
    common_stage_identity_ =
        common_stage_identity_ && primary_stage.data() == last_primary_stage_;
    source_stage_times_.push_back(stage_time);
    source_generations_.push_back(generation_);
    composition_.evaluate_routeb_stage(
        execution, stage_time,
        teuk::Plus2RouteBCurvatureTowerStage{
            generation_, coordinator_.reconstruction_values(),
            coordinator_.reconstruction_stamps()},
        target.accepted_activation, target, curvature_, primitive_, outer_);
  }

  [[nodiscard]] std::size_t value_count() const {
    return registry_.size() * replay_primary_field_count *
           replay_radial_count * replay_theta_count;
  }
  [[nodiscard]] const auto& primary_stage_times() const {
    return primary_stage_times_;
  }
  [[nodiscard]] const auto& source_stage_times() const {
    return source_stage_times_;
  }
  [[nodiscard]] const auto& source_generations() const {
    return source_generations_;
  }
  [[nodiscard]] bool common_stage_identity() const {
    return common_stage_identity_;
  }

 private:
  const teuk::ModeRegistry& registry_;
  const teuk::UniformRadialGrid& grid_;
  teuk::Plus2SpatialThetaView cos_theta_;
  teuk::Plus2SpatialThetaView sin_theta_;
  teuk::RouteBAngularJetCoordinator<Execution> coordinator_;
  StageView stage_primary_;
  StageView stage_reconstruction_;
  StampView primary_stamps_;
  StampView reconstruction_stamps_;
  teuk::Plus2RouteBCurvatureSpatialProvider<Execution>& curvature_;
  teuk::Plus2SourcePrimitiveSpatialProducer<Execution>& primitive_;
  teuk::Plus2SourceOuterSpatialProducer<Execution>& outer_;
  teuk::Plus2LiveSourceComposition<Execution>& composition_;
  std::uint64_t generation_ = 0;
  const C* last_primary_stage_ = nullptr;
  bool common_stage_identity_ = true;
  std::vector<double> primary_stage_times_;
  std::vector<double> source_stage_times_;
  std::vector<std::uint64_t> source_generations_;
};

struct PhysicalTrajectory {
  std::vector<C> primary;
  std::vector<C> companion;
  std::vector<C> forcing;
  std::vector<double> primary_stage_times;
  std::vector<double> source_stage_times;
  std::vector<std::uint64_t> source_generations;
  bool common_stage_identity = false;
  int allocations = 0;
  int fences = 0;
};

class RouteBPhysicalReplayFixture {
 public:
  RouteBPhysicalReplayFixture(const teuk::Plus2RunMode mode,
                              const double companion_initial,
                              const double primary_amplitude = 1.0)
      : grid_(replay_radial_count, 0.0, future_horizon()),
        cos_theta_("routeb_physical_replay_cos", replay_theta_count),
        sin_theta_("routeb_physical_replay_sin", replay_theta_count),
        initial_primary_("routeb_physical_replay_initial", primary_count()),
        concurrent_primary_("routeb_physical_replay_concurrent_primary",
                            primary_count()),
        primary_workspace_(primary_count()),
        curvature_(execution_, registry_, grid_, background_, replay_ell_max,
                   cos_theta_, sin_theta_,
                   "routeb_physical_replay_curvature"),
        primitive_(execution_, registry_, grid_, background_, replay_ell_max,
                   cos_theta_, sin_theta_,
                   "routeb_physical_replay_primitive",
                   teuk::RadialDiscretization::D105),
        outer_(execution_, registry_, grid_, background_, replay_ell_max,
               cos_theta_, sin_theta_, "routeb_physical_replay_outer",
               teuk::RadialDiscretization::D105),
        composition_(execution_, registry_, grid_, background_,
                     replay_ell_max, replay_theta_count, cos_theta_,
                     sin_theta_, "routeb_physical_replay_composition",
                     teuk::RadialDiscretization::D105),
        graph_(execution_, registry_, grid_, primary_parameters(), cos_theta_,
               sin_theta_, curvature_, primitive_, outer_, composition_),
        target_angular_(execution_, target_registry_, 2, 2, replay_ell_max,
                        replay_theta_count, replay_radial_count, background_),
        pipeline_(make_configuration(mode), grid_, companion_parameters(),
                  theta_coordinates(), teuk::ReductionEvolution::FreeDamped,
                  0.0, "routeb_physical_replay_pipeline",
                  teuk::RadialDiscretization::D105) {
    initialize_geometry_and_primary(primary_amplitude);
    pipeline_.initialize_zero(execution_);
    if (companion_initial != 0.0) {
      Kokkos::deep_copy(execution_, pipeline_.companion_state(),
                        C(companion_initial, -0.3 * companion_initial));
    }
    if (mode == teuk::Plus2RunMode::Replay) {
      pipeline_.initialize_replay_primary(execution_, initial_primary_);
    } else {
      Kokkos::deep_copy(execution_, concurrent_primary_, initial_primary_);
    }
    execution_.fence("initialize Route-B physical replay fixture");
  }

  PhysicalTrajectory run() {
    auto primary_rhs = [&](const Execution& execution, const double time,
                           const auto& input, const auto& output) {
      graph_.evaluate_primary_rhs(execution, time, input, output);
    };
    auto source = [&](const Execution& execution, const double time,
                      const auto& primary,
                      const teuk::Plus2StageSourceTarget target) {
      graph_.evaluate_source(execution, time, primary, target);
    };
    auto angular = [&](const Execution& execution, const double,
                       const auto& companion,
                       const auto& angular_laplacian) {
      UnmanagedSingleFieldView output(
          angular_laplacian.data(), target_registry_.size(), 1,
          replay_radial_count, replay_theta_count);
      target_angular_.laplacian(
          execution, companion,
          static_cast<std::size_t>(teuk::TeukolskyField::Psi), output, 0);
    };
    const teuk::SourceActivationState activation{true, 0.0, 4, 0.0};
    physical_replay_allocations = 0;
    physical_replay_fences = 0;
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        count_physical_replay_allocation);
    Kokkos::Tools::Experimental::set_begin_fence_callback(
        count_physical_replay_fence);
    for (int step_index = 0; step_index < replay_steps; ++step_index) {
      const double time = replay_step * static_cast<double>(step_index);
      if (pipeline_.configuration().mode == teuk::Plus2RunMode::Replay) {
        pipeline_.advance_replay(execution_, time, replay_step, activation,
                                 primary_rhs, source, angular);
      } else {
        pipeline_.advance_concurrent(
            execution_, concurrent_primary_, time, replay_step, activation,
            primary_rhs, source, angular, primary_workspace_);
      }
    }
    Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    const int allocations = physical_replay_allocations;
    const int fences = physical_replay_fences;
    execution_.fence("finish Route-B physical replay trajectory");
    const auto primary =
        pipeline_.configuration().mode == teuk::Plus2RunMode::Replay
            ? pipeline_.orchestrator().replay_primary_state()
            : concurrent_primary_;
    return {copy(primary), copy(pipeline_.companion_state()),
            copy(pipeline_.forcing()), graph_.primary_stage_times(),
            graph_.source_stage_times(), graph_.source_generations(),
            graph_.common_stage_identity(), allocations, fences};
  }

 private:
  template <class View>
  static std::vector<C> copy(const View& view) {
    const auto host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    return std::vector<C>(host.data(), host.data() + host.size());
  }

  [[nodiscard]] double future_horizon() const {
    return background_.compactification_length *
           background_.compactification_length /
           (background_.mass +
            std::sqrt(background_.mass * background_.mass -
                      background_.spin * background_.spin));
  }

  [[nodiscard]] std::size_t primary_count() const {
    return registry_.size() * replay_primary_field_count *
           replay_radial_count * replay_theta_count;
  }

  teuk::TeukolskyParameters primary_parameters() const {
    teuk::TeukolskyParameters parameters;
    parameters.mass = background_.mass;
    parameters.spin = background_.spin;
    parameters.compactification_length =
        background_.compactification_length;
    parameters.spin_weight = -2;
    parameters.azimuthal_mode = 0;
    parameters.reduction_damping = 0.17;
    return parameters;
  }

  teuk::TeukolskyParameters companion_parameters() const {
    auto parameters = primary_parameters();
    parameters.spin_weight = 2;
    return parameters;
  }

  std::vector<double> theta_coordinates() const {
    const auto angular_grid =
        teuk::angular::gauss_legendre(replay_theta_count);
    std::vector<double> result(replay_theta_count);
    for (std::size_t index = 0; index < replay_theta_count; ++index) {
      result[index] = angular_grid.theta(index);
    }
    return result;
  }

  teuk::Plus2ReplayConfiguration make_configuration(
      const teuk::Plus2RunMode mode) const {
    teuk::Plus2ReplayConfiguration configuration;
    configuration.mode = mode;
    configuration.ell_max_first = replay_ell_max;
    configuration.ell_max_second = replay_ell_max;
    configuration.parent_modes = {-2, 2};
    configuration.target_modes = {-4, 0, 4};
    configuration.primary_value_count =
        mode == teuk::Plus2RunMode::Replay ? primary_count() : 0;
    configuration.radial_count = replay_radial_count;
    configuration.theta_count = replay_theta_count;
    configuration.radial_discretization = teuk::RadialDiscretization::D105;
    configuration.mass = background_.mass;
    configuration.spin = background_.spin;
    configuration.compactification_length =
        background_.compactification_length;
    configuration.radial_coordinates.resize(replay_radial_count);
    for (std::size_t radial = 0; radial < replay_radial_count; ++radial) {
      configuration.radial_coordinates[radial] = grid_.coordinate(radial);
    }
    configuration.theta_coordinates = theta_coordinates();
    configuration.time_step = replay_step;
    configuration.reduction_mode = "free_damped";
    configuration.reduction_damping = 0.17;
    configuration.dissipation = 0.0;
    configuration.primary_checkpoint_identity =
        "routeb-physical-replay-initial-v1";
    configuration.git_commit = teuk::plus2_build_git_commit();
    configuration.runtime_config_schema_version = 1;
    return configuration;
  }

  void initialize_geometry_and_primary(const double amplitude) {
    const auto angular_grid =
        teuk::angular::gauss_legendre(replay_theta_count);
    auto host_cos = Kokkos::create_mirror_view(cos_theta_);
    auto host_sin = Kokkos::create_mirror_view(sin_theta_);
    auto host_state = Kokkos::create_mirror_view(initial_primary_);
    for (std::size_t theta = 0; theta < replay_theta_count; ++theta) {
      host_cos(theta) = angular_grid.x[theta];
      host_sin(theta) =
          std::sqrt(std::max(0.0, 1.0 - angular_grid.x[theta] *
                                              angular_grid.x[theta]));
    }
    for (std::size_t mode_index = 0; mode_index < registry_.size();
         ++mode_index) {
      const int mode = registry_.modes()[mode_index];
      const bool parent = registry_.is_parent(mode);
      for (std::size_t radial = 0; radial < replay_radial_count; ++radial) {
        const double radius = grid_.coordinate(radial);
        for (std::size_t theta = 0; theta < replay_theta_count; ++theta) {
          const double angle = angular_grid.theta(theta);
          const double primary_harmonic =
              teuk::angular::spin_weighted_harmonic_theta(4, mode, -2,
                                                           angle);
          const C psi = parent
                            ? amplitude * C(0.031, -0.019) *
                                  std::exp(0.13 * radius) * primary_harmonic
                            : C{};
          const std::array<C, 3> primary{
              parent ? amplitude * C(0.027, 0.011) *
                           std::exp(0.09 * radius) * primary_harmonic
                     : C{},
              0.13 * psi, psi};
          constexpr int reconstruction_spins[7]{-1, -2, 0, -2,
                                                 -1, -1, 0};
          for (std::size_t field = 0; field < replay_primary_field_count;
               ++field) {
            C value{};
            if (field < 3) {
              value = primary[field];
            } else if (parent) {
              const std::size_t reconstruction_field = field - 3;
              const int spin = reconstruction_spins[reconstruction_field];
              const int ell = std::max({4, std::abs(mode), std::abs(spin)});
              const double harmonic =
                  teuk::angular::spin_weighted_harmonic_theta(
                      ell, mode, spin, angle);
              value = amplitude *
                      C(0.018 * static_cast<double>(field + 1),
                        0.007 * static_cast<double>(field + 2)) *
                      std::exp((0.07 + 0.01 * field) * radius) * harmonic;
            }
            const std::size_t index =
                ((mode_index * replay_primary_field_count + field) *
                     replay_radial_count +
                 radial) *
                    replay_theta_count +
                theta;
            host_state(index) = value;
          }
        }
      }
    }
    Kokkos::deep_copy(execution_, cos_theta_, host_cos);
    Kokkos::deep_copy(execution_, sin_theta_, host_sin);
    Kokkos::deep_copy(execution_, initial_primary_, host_state);
  }

  Execution execution_;
  teuk::ModeRegistry registry_{{-4, -2, 0, 2, 4}, {-2, 2}, {-4, 0, 4}};
  teuk::ModeRegistry target_registry_{{-4, 0, 4}};
  teuk::KerrParameters background_{1.0, 0.63, 1.4};
  teuk::UniformRadialGrid grid_;
  teuk::Plus2SpatialThetaView cos_theta_;
  teuk::Plus2SpatialThetaView sin_theta_;
  FlatView initial_primary_;
  FlatView concurrent_primary_;
  teuk::DeviceRK4Workspace<C, Execution> primary_workspace_;
  teuk::Plus2RouteBCurvatureSpatialProvider<Execution> curvature_;
  teuk::Plus2SourcePrimitiveSpatialProducer<Execution> primitive_;
  teuk::Plus2SourceOuterSpatialProducer<Execution> outer_;
  teuk::Plus2LiveSourceComposition<Execution> composition_;
  RouteBPhysicalStageGraph graph_;
  teuk::SignedModeAngularCoordinator<Execution> target_angular_;
  teuk::Plus2CompanionPipeline pipeline_;
};

void check_bitwise_vectors(const std::vector<C>& left,
                           const std::vector<C>& right) {
  CHECK(left.size() == right.size());
  for (std::size_t index = 0; index < left.size(); ++index) {
    CHECK(bitwise_equal(left[index], right[index]));
  }
}

TEST_CASE("Route-B physical source replay equals concurrent common-stage RK4") {
  RouteBPhysicalReplayFixture concurrent(teuk::Plus2RunMode::Concurrent, 0.0);
  RouteBPhysicalReplayFixture replay(teuk::Plus2RunMode::Replay, 0.0);
  RouteBPhysicalReplayFixture scaled(teuk::Plus2RunMode::Concurrent, 0.0,
                                     -1.7);
  const PhysicalTrajectory concurrent_result = concurrent.run();
  const PhysicalTrajectory replay_result = replay.run();
  const PhysicalTrajectory scaled_result = scaled.run();

  check_bitwise_vectors(concurrent_result.primary, replay_result.primary);
  check_bitwise_vectors(concurrent_result.companion, replay_result.companion);
  check_bitwise_vectors(concurrent_result.forcing, replay_result.forcing);
  CHECK(concurrent_result.primary_stage_times ==
        replay_result.primary_stage_times);
  CHECK(concurrent_result.source_stage_times == replay_result.source_stage_times);
  CHECK(concurrent_result.source_generations ==
        replay_result.source_generations);
  CHECK(concurrent_result.common_stage_identity);
  CHECK(replay_result.common_stage_identity);
  CHECK(concurrent_result.allocations == 0);
  CHECK(replay_result.allocations == 0);
  CHECK(concurrent_result.fences == 0);
  CHECK(replay_result.fences == 0);

  const std::array<double, 8> expected_times{
      0.0, 0.5 * replay_step, 0.5 * replay_step, replay_step,
      replay_step, 1.5 * replay_step, 1.5 * replay_step, 2.0 * replay_step};
  CHECK(concurrent_result.primary_stage_times.size() == expected_times.size());
  CHECK(concurrent_result.source_generations.size() == expected_times.size());
  for (std::size_t stage = 0; stage < expected_times.size(); ++stage) {
    CHECK_NEAR(concurrent_result.primary_stage_times[stage],
               expected_times[stage], 2.0e-18);
    CHECK_NEAR(concurrent_result.source_stage_times[stage],
               expected_times[stage], 2.0e-18);
    CHECK(concurrent_result.source_generations[stage] == stage + 1);
  }

  double forcing_maximum = 0.0;
  for (const C value : concurrent_result.forcing) {
    CHECK(std::isfinite(value.real()));
    CHECK(std::isfinite(value.imag()));
    forcing_maximum = std::max(forcing_maximum, Kokkos::abs(value));
  }
  double companion_maximum = 0.0;
  for (const C value : concurrent_result.companion) {
    CHECK(std::isfinite(value.real()));
    CHECK(std::isfinite(value.imag()));
    companion_maximum = std::max(companion_maximum, Kokkos::abs(value));
  }
  CHECK(forcing_maximum > 1.0e-12);
  CHECK(companion_maximum > 1.0e-16);
  for (std::size_t index = 0; index < concurrent_result.primary.size();
       ++index) {
    CHECK_COMPLEX_NEAR(scaled_result.primary[index],
                       -1.7 * concurrent_result.primary[index], 2.0e-12);
  }
  for (std::size_t index = 0; index < concurrent_result.forcing.size();
       ++index) {
    const C expected = 1.7 * 1.7 * concurrent_result.forcing[index];
    const double tolerance =
        3.0e-10 * std::max(1.0, Kokkos::abs(expected));
    CHECK_COMPLEX_NEAR(scaled_result.forcing[index], expected, tolerance);
  }
  for (std::size_t index = 0; index < concurrent_result.companion.size();
       ++index) {
    const C expected = 1.7 * 1.7 * concurrent_result.companion[index];
    const double tolerance =
        3.0e-10 * std::max(1.0, Kokkos::abs(expected));
    CHECK_COMPLEX_NEAR(scaled_result.companion[index], expected, tolerance);
  }
}

TEST_CASE("Route-B physical companion cannot feed back into primary RK4") {
  RouteBPhysicalReplayFixture zero(teuk::Plus2RunMode::Concurrent, 0.0);
  RouteBPhysicalReplayFixture perturbed(teuk::Plus2RunMode::Concurrent, 0.31);
  const PhysicalTrajectory zero_result = zero.run();
  const PhysicalTrajectory perturbed_result = perturbed.run();
  check_bitwise_vectors(zero_result.primary, perturbed_result.primary);

  bool companion_differs = false;
  for (std::size_t index = 0; index < zero_result.companion.size(); ++index) {
    companion_differs = companion_differs ||
                        !bitwise_equal(zero_result.companion[index],
                                       perturbed_result.companion[index]);
  }
  CHECK(companion_differs);
}

}  // namespace
