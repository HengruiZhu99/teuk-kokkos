#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "teuk/plus2_companion_pipeline.hpp"

namespace {

constexpr std::size_t p_field =
    static_cast<std::size_t>(teuk::TeukolskyField::P);
constexpr std::size_t q_field =
    static_cast<std::size_t>(teuk::TeukolskyField::Q);
constexpr std::size_t psi_field =
    static_cast<std::size_t>(teuk::TeukolskyField::Psi);

teuk::TeukolskyParameters qualification_parameters() {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.43;
  parameters.compactification_length = 1.6;
  parameters.spin_weight = 2;
  parameters.reduction_damping = 0.9;
  return parameters;
}

teuk::Plus2ReplayConfiguration qualification_configuration(
    const std::size_t radial_count) {
  teuk::Plus2ReplayConfiguration configuration;
  configuration.mode = teuk::Plus2RunMode::Concurrent;
  configuration.ell_max_first = 2;
  configuration.ell_max_second = 2;
  configuration.parent_modes = {-1, 1};
  configuration.target_modes = {0};
  configuration.radial_count = radial_count;
  configuration.theta_count = 1;
  configuration.git_commit = teuk::plus2_build_git_commit();
  configuration.runtime_config_schema_version = 1;
  configuration.radial_discretization = teuk::RadialDiscretization::D84;
  return configuration;
}

struct PrimaryGrowthKernel {
  const teuk::Complex* input;
  teuk::Complex* output;
  double rate;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t index) const {
    output[index] = rate * input[index];
  }
};

struct PrimaryGrowthRhs {
  double rate;

  template <class InputView, class OutputView>
  void operator()(const teuk::ExecutionSpace& execution, const double,
                  const InputView& input, const OutputView& output) const {
    Kokkos::parallel_for(
        "qualification_primary_growth",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0,
                                                   input.extent(0)),
        PrimaryGrowthKernel{input.data(), output.data(), rate});
  }
};

struct CompanionAngularKernel {
  const teuk::Complex* state;
  teuk::Complex* angular;
  std::size_t radial_count;
  std::size_t state_field_stride;
  std::size_t state_radial_stride;
  std::size_t angular_radial_stride;
  double eigenvalue;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t radial) const {
    angular[radial * angular_radial_stride] =
        eigenvalue * state[psi_field * state_field_stride +
                           radial * state_radial_stride];
  }
};

struct CompanionAngularAction {
  double eigenvalue;

  template <class StageView, class AngularView>
  void operator()(const teuk::ExecutionSpace& execution, const double,
                  const StageView& stage, const AngularView& angular) const {
    Kokkos::parallel_for(
        "qualification_companion_angular",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0,
                                                   stage.extent(2)),
        CompanionAngularKernel{stage.data(), angular.data(), stage.extent(2),
                               stage.stride(1), stage.stride(2),
                               angular.stride(1), eigenvalue});
  }
};

struct StageCoupledSourceKernel {
  const teuk::Complex* primary;
  teuk::Complex* forcing;
  teuk::UniformRadialGrid grid;
  teuk::TeukolskyParameters parameters;
  double theta;
  double growth_rate;
  double angular_eigenvalue;
  bool active;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t radial) const {
    teuk::TeukolskyParameters point_parameters = parameters;
    point_parameters.azimuthal_mode = 0;
    const auto coefficients = teuk::teukolsky_coefficients(
        point_parameters, grid.coordinate(radial), theta);
    forcing[radial] =
        active
            ? (growth_rate *
                   (growth_rate * coefficients.time + coefficients.definition) -
               coefficients.psi - angular_eigenvalue) *
                  primary[0]
            : teuk::Complex{};
  }
};

struct StageCoupledSource {
  teuk::UniformRadialGrid grid;
  teuk::TeukolskyParameters parameters;
  double theta;
  double growth_rate;
  double angular_eigenvalue;

  template <class PrimaryView>
  void operator()(const teuk::ExecutionSpace& execution, const double,
                  const PrimaryView& primary,
                  const teuk::Plus2StageSourceTarget target) const {
    Kokkos::parallel_for(
        "qualification_stage_coupled_source",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, grid.size()),
        StageCoupledSourceKernel{
            primary.data(), target.coordinate_forcing.data(), grid, parameters,
            theta, growth_rate, angular_eigenvalue,
            target.accepted_activation.active});
  }
};

static_assert(std::is_trivially_copyable_v<PrimaryGrowthKernel>);
static_assert(std::is_trivially_copyable_v<CompanionAngularKernel>);
static_assert(std::is_trivially_copyable_v<StageCoupledSourceKernel>);
static_assert(sizeof(PrimaryGrowthKernel) < 1800);
static_assert(sizeof(CompanionAngularKernel) < 1800);
static_assert(sizeof(StageCoupledSourceKernel) < 1800);

struct CoupledEvolutionError {
  double primary = 0.0;
  double companion = 0.0;
};

CoupledEvolutionError evolve_stage_coupled_manufactured(const int step_count) {
  constexpr std::size_t radial_count = 18;
  constexpr double theta = 0.93;
  constexpr double growth_rate = 15.0;
  constexpr double angular_eigenvalue = -4.0;
  constexpr double final_time = 0.03;
  const teuk::Complex initial(0.8, -0.23);
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.46);
  const auto parameters = qualification_parameters();
  teuk::Plus2CompanionPipeline pipeline(
      qualification_configuration(radial_count), grid, parameters, {theta},
      teuk::ReductionEvolution::FreeDamped, 0.0,
      "fourth_order_coupled", teuk::RadialDiscretization::D84);
  pipeline.initialize_zero(execution);

  auto host_state =
      Kokkos::create_mirror_view(pipeline.companion_storage().state());
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    auto point_parameters = parameters;
    point_parameters.azimuthal_mode = 0;
    const auto coefficients = teuk::teukolsky_coefficients(
        point_parameters, grid.coordinate(radial), theta);
    host_state(0, p_field, radial, 0) =
        (growth_rate * coefficients.time + coefficients.definition) * initial;
    host_state(0, q_field, radial, 0) = teuk::Complex{};
    host_state(0, psi_field, radial, 0) = initial;
  }
  Kokkos::deep_copy(execution, pipeline.companion_storage().state(),
                    host_state);

  Kokkos::View<teuk::Complex*, teuk::MemorySpace> primary(
      "fourth_order_primary", 1);
  Kokkos::deep_copy(execution, primary, initial);
  teuk::DeviceRK4Workspace<teuk::Complex, teuk::ExecutionSpace>
      primary_workspace(1);
  const PrimaryGrowthRhs primary_rhs{growth_rate};
  const CompanionAngularAction angular_action{angular_eigenvalue};
  const StageCoupledSource source{grid, parameters, theta, growth_rate,
                                  angular_eigenvalue};
  const teuk::SourceActivationState active{true, 0.0, 1, 0.0};
  const double step = final_time / static_cast<double>(step_count);
  for (int n = 0; n < step_count; ++n) {
    pipeline.advance_concurrent_validation_only(
        execution, primary, step * static_cast<double>(n), step, active,
        primary_rhs, source, angular_action, primary_workspace);
  }
  execution.fence("finish coupled fourth-order qualification");

  const teuk::Complex exact =
      std::exp(growth_rate * final_time) * initial;
  const auto host_primary = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, primary);
  const auto final_state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.companion_storage().state());
  CoupledEvolutionError result;
  result.primary = Kokkos::abs(host_primary(0) - exact);
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    auto point_parameters = parameters;
    point_parameters.azimuthal_mode = 0;
    const auto coefficients = teuk::teukolsky_coefficients(
        point_parameters, grid.coordinate(radial), theta);
    const teuk::Complex exact_p =
        (growth_rate * coefficients.time + coefficients.definition) * exact;
    result.companion =
        std::max(result.companion,
                 Kokkos::abs(final_state(0, p_field, radial, 0) - exact_p));
    result.companion =
        std::max(result.companion,
                 Kokkos::abs(final_state(0, q_field, radial, 0)));
    result.companion =
        std::max(result.companion,
                 Kokkos::abs(final_state(0, psi_field, radial, 0) - exact));
  }
  return result;
}

double spatial_manufactured_error(const std::size_t radial_count,
                                  const double dissipation_strength) {
  constexpr double theta_value = 1.11;
  const teuk::ExecutionSpace execution;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.52);
  const auto parameters = qualification_parameters();
  teuk::FullSpatialThetaView theta("fourth_order_theta", 1);
  teuk::SignedModeView modes("fourth_order_modes", 1);
  teuk::FullSpatialStateView state("fourth_order_state", 1, 3,
                                    radial_count, 1);
  teuk::FullSpatialStateView rhs("fourth_order_rhs", 1, 3, radial_count, 1);
  teuk::FullSpatialStateView scratch(
      "fourth_order_scratch", 1,
      static_cast<std::size_t>(teuk::TeukolskyRadialScratch::Count),
      radial_count, 1);
  teuk::FullSpatialValueView angular("fourth_order_angular", 1,
                                      radial_count, 1);
  teuk::FullSpatialValueView forcing("fourth_order_forcing", 1,
                                      radial_count, 1);
  auto host_theta = Kokkos::create_mirror_view(theta);
  auto host_modes = Kokkos::create_mirror_view(modes);
  auto host_state = Kokkos::create_mirror_view(state);
  auto host_angular = Kokkos::create_mirror_view(angular);
  auto host_forcing = Kokkos::create_mirror_view(forcing);
  host_theta(0) = theta_value;
  host_modes(0) = 0;
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    const double r = grid.coordinate(radial);
    const teuk::Complex psi(std::exp(0.8 * r), 0.19 * std::sin(1.3 * r));
    const teuk::Complex q(0.8 * std::exp(0.8 * r),
                          0.247 * std::cos(1.3 * r));
    const teuk::Complex velocity(0.31 * std::cos(0.6 * r),
                                  -0.17 * std::exp(0.4 * r));
    auto point_parameters = parameters;
    point_parameters.azimuthal_mode = 0;
    const auto coefficients = teuk::teukolsky_coefficients(
        point_parameters, r, theta_value);
    host_state(0, p_field, radial, 0) =
        coefficients.time * velocity -
        2.0 * coefficients.radial_advection * q +
        coefficients.definition * psi;
    host_state(0, q_field, radial, 0) = q;
    host_state(0, psi_field, radial, 0) = psi;
    host_angular(0, radial, 0) =
        teuk::Complex(0.07 * std::sin(0.9 * r),
                      -0.04 * std::cos(0.5 * r));
    host_forcing(0, radial, 0) =
        teuk::Complex(-0.03 * std::exp(0.2 * r), 0.06 * std::sin(0.7 * r));
  }
  Kokkos::deep_copy(theta, host_theta);
  Kokkos::deep_copy(modes, host_modes);
  Kokkos::deep_copy(state, host_state);
  Kokkos::deep_copy(angular, host_angular);
  Kokkos::deep_copy(forcing, host_forcing);
  teuk::evaluate_sbp_teukolsky_full_stage_rhs(
      execution, grid, parameters, theta, modes, state, angular, forcing,
      teuk::ReductionEvolution::FreeDamped, scratch, rhs,
      dissipation_strength, {}, {}, teuk::RadialDiscretization::D84);
  execution.fence("finish spatial fourth-order qualification");
  const auto actual =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rhs);

  double maximum_error = 0.0;
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    const double r = grid.coordinate(radial);
    const teuk::Complex psi(std::exp(0.8 * r), 0.19 * std::sin(1.3 * r));
    const teuk::Complex q(0.8 * std::exp(0.8 * r),
                          0.247 * std::cos(1.3 * r));
    const teuk::Complex dr_q(0.64 * std::exp(0.8 * r),
                             -0.3211 * std::sin(1.3 * r));
    const teuk::Complex velocity(0.31 * std::cos(0.6 * r),
                                  -0.17 * std::exp(0.4 * r));
    const teuk::Complex dr_velocity(-0.186 * std::sin(0.6 * r),
                                     -0.068 * std::exp(0.4 * r));
    auto point_parameters = parameters;
    point_parameters.azimuthal_mode = 0;
    const auto coefficients = teuk::teukolsky_coefficients(
        point_parameters, r, theta_value);
    const teuk::Complex p =
        coefficients.time * velocity -
        2.0 * coefficients.radial_advection * q +
        coefficients.definition * psi;
    const teuk::Complex expected_p = teuk::teukolsky_p_rhs(
        coefficients, {p, q, psi}, dr_q, host_angular(0, radial, 0),
        host_forcing(0, radial, 0));
    maximum_error =
        std::max(maximum_error,
                 Kokkos::abs(actual(0, p_field, radial, 0) - expected_p));
    maximum_error =
        std::max(maximum_error,
                 Kokkos::abs(actual(0, q_field, radial, 0) - dr_velocity));
    maximum_error =
        std::max(maximum_error,
                 Kokkos::abs(actual(0, psi_field, radial, 0) - velocity));
  }
  return maximum_error;
}

}  // namespace

TEST_CASE("D84 full plus2 RHS is globally fourth order with resolved source and angular data") {
  const double coarse = spatial_manufactured_error(25, 0.0);
  const double medium = spatial_manufactured_error(49, 0.0);
  const double fine = spatial_manufactured_error(97, 0.0);
  CHECK(coarse / medium > 13.0);
  CHECK(medium / fine > 13.0);
  CHECK(fine < 2.0e-7);
}

TEST_CASE("D84 compatible dissipation preserves global fourth order") {
  constexpr double dissipation_strength = 0.006;
  const double coarse = spatial_manufactured_error(25, dissipation_strength);
  const double medium = spatial_manufactured_error(49, dissipation_strength);
  const double fine = spatial_manufactured_error(97, dissipation_strength);
  CHECK(coarse / medium > 12.0);
  CHECK(medium / fine > 12.0);
  CHECK(fine < 3.0e-6);
}

TEST_CASE("stage-coupled passive plus2 evolution converges at RK4 order") {
  const auto coarse = evolve_stage_coupled_manufactured(16);
  const auto medium = evolve_stage_coupled_manufactured(32);
  const auto fine = evolve_stage_coupled_manufactured(64);
  CHECK(coarse.primary / medium.primary > 14.0);
  CHECK(medium.primary / fine.primary > 14.0);
  CHECK(coarse.companion / medium.companion > 14.0);
  CHECK(medium.companion / fine.companion > 14.0);
  CHECK(fine.companion < 1.0e-8);
}
