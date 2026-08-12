#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/plus2_bianchi_transport.hpp"
#include "teuk/plus2_live_source_composition.hpp"

namespace {

using C = teuk::Complex;
using execution_space = teuk::ExecutionSpace;

constexpr std::size_t binding_radial_count = 24;
constexpr std::size_t binding_theta_count = 8;
constexpr std::size_t binding_reconstruction_fields = 5;
constexpr int binding_ell_max = 3;

int binding_allocations = 0;
int binding_fences = 0;

void count_binding_allocation(Kokkos::Tools::SpaceHandle, const char*,
                              const void*, std::uint64_t) {
  ++binding_allocations;
}

void count_binding_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++binding_fences;
}

KOKKOS_INLINE_FUNCTION std::size_t binding_flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

struct InitializeBindingCurvatureFunctor {
  C* state;
  teuk::UniformRadialGrid grid;
  double amplitude;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t state_count = static_cast<std::size_t>(
        teuk::Plus2BianchiStateComponent::Count);
    const std::size_t radial = flat / theta_count;
    const std::size_t theta = flat - radial * theta_count;
    const double radius = grid.coordinate(radial);
    const double angle_shape =
        1.0 + 0.02 * static_cast<double>(theta);
    state[binding_flat4(0, 0, radial, theta, state_count, radial_count,
                        theta_count)] =
        amplitude * C(0.09 + 0.025 * radius,
                      -0.018 * angle_shape * (1.0 + radius));
    state[binding_flat4(0, 1, radial, theta, state_count, radial_count,
                        theta_count)] =
        amplitude * C((0.055 - 0.013 * radius) * angle_shape,
                      0.024 * (1.0 + 0.2 * radius));
  }
};

struct WriteBindingCommonStageFunctor {
  C* bianchi;
  std::uint64_t* bianchi_stamps;
  C* value;
  C* tangent;
  C* second;
  std::uint64_t* value_stamps;
  std::uint64_t* tangent_stamps;
  std::uint64_t* second_stamps;
  const C* primary;
  teuk::UniformRadialGrid grid;
  const double* cos_theta;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;
  bool stale_reconstruction;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t bianchi_count = static_cast<std::size_t>(
        teuk::Plus2BianchiPrimaryComponent::Count);
    const std::size_t radial = flat / theta_count;
    const std::size_t theta = flat - radial * theta_count;
    const double radius = grid.coordinate(radial);
    const double x = cos_theta[theta];
    const C amplitude = primary[0];
    const C fields[binding_reconstruction_fields]{
        amplitude * C(0.055 * (1.0 + 0.12 * radius), -0.009 * x),
        amplitude * C(-0.021 * (1.0 + 0.04 * radius), 0.012 * x),
        amplitude * C(0.038 * (1.0 + 0.08 * radius), 0.016 * (1.0 - x * x)),
        amplitude * C(-0.031 * (1.0 + 0.06 * radius), 0.011 * x),
        amplitude * C(0.047 * (1.0 + 0.1 * radius), -0.014 * x)};
    constexpr double rates[binding_reconstruction_fields]{
        0.13, -0.09, 0.16, -0.11, 0.07};
    for (std::size_t field = 0; field < binding_reconstruction_fields;
         ++field) {
      const std::size_t index = binding_flat4(
          0, field, radial, theta, binding_reconstruction_fields,
          radial_count, theta_count);
      value[index] = fields[field];
      tangent[index] = rates[field] * fields[field];
      second[index] = rates[field] * rates[field] * fields[field];
      value_stamps[index] = generation;
      tangent_stamps[index] = generation;
      second_stamps[index] =
          stale_reconstruction && radial == 0 && theta == 0 && field == 4
              ? generation - 1
              : generation;
    }

    const C h = fields[0];
    const C ht = rates[0] * h;
    const C csharp = Kokkos::conj(fields[3]);
    const C bsharp = Kokkos::conj(fields[2]);
    const C input[bianchi_count]{
        h,
        ht,
        rates[0] * ht,
        csharp,
        rates[3] * csharp,
        bsharp,
        rates[2] * bsharp,
        amplitude * C(0.014 * (1.0 + radius), -0.006 * x),
        amplitude * C(-0.004 * (1.0 + radius), 0.002 * x),
        amplitude * C(0.019 * (1.0 + 0.05 * radius), -0.007 * x),
        amplitude * C(0.003 * (1.0 + 0.05 * radius), 0.001 * x),
        C{},
        C{}};
    for (std::size_t field = 0; field < bianchi_count; ++field) {
      const std::size_t index = binding_flat4(
          0, field, radial, theta, bianchi_count, radial_count, theta_count);
      bianchi[index] = input[field];
      bianchi_stamps[index] = generation;
    }
  }
};

struct ZeroBindingPrimaryRhsFunctor {
  C* output;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    output[i] = C{};
  }
};

struct ZeroBindingAngularFunctor {
  C* output;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    output[i] = C{};
  }
};

struct WriteBindingOuterFunctor {
  const C* sums;
  const C* tangents;
  C* projected;
  C* outer;
  std::uint64_t* projected_stamps;
  std::uint64_t* outer_stamps;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t radial = flat / theta_count;
    const std::size_t theta = flat - radial * theta_count;
    constexpr std::size_t aggregate_count = static_cast<std::size_t>(
        teuk::Plus2SpatialAggregate::Count);
    constexpr std::size_t tangent_count = static_cast<std::size_t>(
        teuk::Plus2ProductionJkAggregate::Count);
    constexpr std::size_t outer_count = static_cast<std::size_t>(
        teuk::Plus2SpatialOuterDerivative::Count);
    for (std::size_t field = 0; field < aggregate_count; ++field) {
      const std::size_t index = binding_flat4(
          0, field, radial, theta, aggregate_count, radial_count, theta_count);
      projected[index] = sums[index];
      projected_stamps[index] = generation;
    }
    for (std::size_t field = 0; field < outer_count; ++field) {
      const std::size_t output_index = binding_flat4(
          0, field, radial, theta, outer_count, radial_count, theta_count);
      const std::size_t input_index = binding_flat4(
          0, field, radial, theta, tangent_count, radial_count, theta_count);
      outer[output_index] = tangents[input_index];
      outer_stamps[output_index] = generation;
    }
  }
};

static_assert(std::is_trivially_copyable_v<InitializeBindingCurvatureFunctor>);
static_assert(std::is_trivially_copyable_v<WriteBindingCommonStageFunctor>);
static_assert(std::is_trivially_copyable_v<ZeroBindingPrimaryRhsFunctor>);
static_assert(std::is_trivially_copyable_v<ZeroBindingAngularFunctor>);
static_assert(std::is_trivially_copyable_v<WriteBindingOuterFunctor>);

struct BindingFixture {
  execution_space execution;
  teuk::ModeRegistry registry{{0}};
  teuk::UniformRadialGrid grid{binding_radial_count, 0.0, 0.62};
  teuk::KerrParameters parameters{1.0, 0.0, 1.2};
  teuk::Plus2SpatialThetaView cos_theta{"binding_cos", binding_theta_count};
  teuk::Plus2SpatialThetaView sin_theta{"binding_sin", binding_theta_count};
  teuk::Plus2PrimitiveConstStageView::non_const_type value{
      "binding_value", registry.size(), binding_reconstruction_fields,
      binding_radial_count, binding_theta_count};
  teuk::Plus2PrimitiveConstStageView::non_const_type tangent{
      "binding_tangent", registry.size(), binding_reconstruction_fields,
      binding_radial_count, binding_theta_count};
  teuk::Plus2PrimitiveConstStageView::non_const_type second{
      "binding_second", registry.size(), binding_reconstruction_fields,
      binding_radial_count, binding_theta_count};
  teuk::Plus2PrimitiveConstStageView::non_const_type wrong_value{
      "binding_wrong_value", registry.size(),
      binding_reconstruction_fields - 1, binding_radial_count,
      binding_theta_count};
  teuk::Plus2LiveStampView value_stamps{
      "binding_value_stamps", registry.size(), binding_reconstruction_fields,
      binding_radial_count, binding_theta_count};
  teuk::Plus2LiveStampView tangent_stamps{
      "binding_tangent_stamps", registry.size(),
      binding_reconstruction_fields, binding_radial_count,
      binding_theta_count};
  teuk::Plus2LiveStampView second_stamps{
      "binding_second_stamps", registry.size(), binding_reconstruction_fields,
      binding_radial_count, binding_theta_count};
  Kokkos::View<C*> primary{"binding_primary", 1};
  teuk::DeviceRK4Workspace<C, execution_space> primary_workspace{1};
  std::unique_ptr<teuk::Plus2BianchiTransport<execution_space>> transport;
  std::unique_ptr<teuk::Plus2SourcePrimitiveSpatialProducer<execution_space>>
      primitive_producer;
  std::unique_ptr<teuk::Plus2LiveSourceComposition<execution_space>>
      composition;
  std::unique_ptr<teuk::Plus2CompanionPipeline> pipeline;

  explicit BindingFixture(const double amplitude) {
    const auto angular_grid = teuk::angular::gauss_legendre(
        static_cast<int>(binding_theta_count));
    std::vector<double> theta_coordinates(binding_theta_count);
    auto host_cos = Kokkos::create_mirror_view(cos_theta);
    auto host_sin = Kokkos::create_mirror_view(sin_theta);
    for (std::size_t theta = 0; theta < binding_theta_count; ++theta) {
      host_cos(theta) = angular_grid.x[theta];
      host_sin(theta) =
          std::sqrt(1.0 - angular_grid.x[theta] * angular_grid.x[theta]);
      theta_coordinates[theta] = std::acos(angular_grid.x[theta]);
    }
    Kokkos::deep_copy(execution, cos_theta, host_cos);
    Kokkos::deep_copy(execution, sin_theta, host_sin);
    transport =
        std::make_unique<teuk::Plus2BianchiTransport<execution_space>>(
            execution, registry, grid, parameters, binding_ell_max, cos_theta,
            sin_theta, teuk::RadialDiscretization::D105,
            "binding_transport");
    primitive_producer = std::make_unique<
        teuk::Plus2SourcePrimitiveSpatialProducer<execution_space>>(
        execution, registry, grid, parameters, binding_ell_max, cos_theta,
        sin_theta, "binding_primitive", teuk::RadialDiscretization::D105);
    composition =
        std::make_unique<teuk::Plus2LiveSourceComposition<execution_space>>(
            execution, registry, grid, parameters, binding_ell_max,
            static_cast<int>(binding_theta_count), cos_theta, sin_theta,
            "binding_composition", teuk::RadialDiscretization::D105);
    teuk::Plus2ReplayConfiguration configuration;
    configuration.mode = teuk::Plus2RunMode::Concurrent;
    configuration.ell_max_first = binding_ell_max;
    configuration.ell_max_second = binding_ell_max;
    configuration.parent_modes = {0};
    configuration.target_modes = {0};
    configuration.radial_count = binding_radial_count;
    configuration.theta_count = binding_theta_count;
    configuration.radial_discretization = teuk::RadialDiscretization::D105;
    configuration.git_commit = teuk::plus2_build_git_commit();
    configuration.runtime_config_schema_version = 1;
    teuk::TeukolskyParameters companion_parameters;
    companion_parameters.mass = parameters.mass;
    companion_parameters.spin = parameters.spin;
    companion_parameters.compactification_length =
        parameters.compactification_length;
    companion_parameters.spin_weight = 2;
    companion_parameters.reduction_damping = 1.0;
    pipeline = std::make_unique<teuk::Plus2CompanionPipeline>(
        configuration, grid, companion_parameters,
        std::move(theta_coordinates), teuk::ReductionEvolution::FreeDamped,
        0.0, "binding_pipeline", teuk::RadialDiscretization::D105);
    pipeline->initialize_zero(execution);
    teuk::Plus2BianchiStateView initial(
        "binding_initial", registry.size(),
        static_cast<std::size_t>(teuk::Plus2BianchiStateComponent::Count),
        binding_radial_count, binding_theta_count);
    Kokkos::parallel_for(
        "initialize_binding_curvature",
        Kokkos::RangePolicy<execution_space>(
            execution, 0, binding_radial_count * binding_theta_count),
        InitializeBindingCurvatureFunctor{initial.data(), grid, amplitude,
                                          binding_radial_count,
                                          binding_theta_count});
    transport->initialize(
        execution, initial,
        {"test-only-common-stage-seam-v1", true, true, true,
         "test-only-manufactured-boundary-v1"});
    Kokkos::deep_copy(execution, primary, C(amplitude, 0.0));
    execution.fence("finish concrete binding fixture setup");
  }

  [[nodiscard]] teuk::Plus2BianchiStageCapability bianchi_capability() const {
    return {"test-only-manufactured-boundary-v1", true, true, true, true,
            true, teuk::RadialDiscretization::D105};
  }
};

struct BindingResult {
  std::vector<C> forcing;
  std::vector<C> companion;
  std::vector<std::uint64_t> generations;
  std::vector<double> stage_times;
  C primary{};
  int allocations = 0;
  int fences = 0;
  bool pointers_stable = false;
  bool common_stage_inputs = false;
  teuk::SourceActivationState activation_before{};
  teuk::SourceActivationState activation_after{};
};

BindingResult run_binding_evolution(const double amplitude, const int steps,
                                    const double final_time,
                                    const bool stale_reconstruction,
                                    const bool audit_hot_path,
                                    const bool wrong_shape = false) {
  BindingFixture fixture(amplitude);
  const teuk::SourceActivationState activation{true, 0.0, 3, 0.0};
  BindingResult result;
  result.generations.reserve(4 * static_cast<std::size_t>(steps));
  result.stage_times.reserve(4 * static_cast<std::size_t>(steps));
  std::vector<const C*> primary_rhs_stages;
  std::vector<const C*> producer_primary_stages;
  std::vector<const C*> companion_primary_stages;
  std::vector<double> primary_rhs_times;
  std::vector<double> producer_times;
  const std::size_t stage_count = 4 * static_cast<std::size_t>(steps);
  primary_rhs_stages.reserve(stage_count);
  producer_primary_stages.reserve(stage_count);
  companion_primary_stages.reserve(stage_count);
  primary_rhs_times.reserve(stage_count);
  producer_times.reserve(stage_count);
  bool bianchi_stage_views_match = true;
  const C* const bianchi_state_pointer =
      fixture.transport->flat_state().data();
  const C* const bianchi_stage_pointer =
      fixture.transport->rk_workspace().stage.data();
  result.activation_before = activation;
  auto primary_producer =
      [&](const execution_space& execution, const double stage_time,
          const auto& primary,
          const teuk::Plus2BianchiPrimaryWriteTarget target) {
        producer_primary_stages.push_back(primary.data());
        producer_times.push_back(stage_time);
        Kokkos::parallel_for(
            "write_binding_common_stage",
            Kokkos::RangePolicy<execution_space>(
                execution,
                0, binding_radial_count * binding_theta_count),
            WriteBindingCommonStageFunctor{
                target.fields.data(), target.stamps.data(),
                fixture.value.data(), fixture.tangent.data(),
                fixture.second.data(), fixture.value_stamps.data(),
                fixture.tangent_stamps.data(), fixture.second_stamps.data(),
                primary.data(), fixture.grid, fixture.cos_theta.data(),
                target.generation, binding_radial_count, binding_theta_count,
                stale_reconstruction});
      };
  auto primary_rhs = [&](const execution_space& execution,
                         const double stage_time, const auto& input,
                         const auto& output) {
    primary_rhs_stages.push_back(input.data());
    primary_rhs_times.push_back(stage_time);
    Kokkos::parallel_for(
        "zero_binding_primary_rhs",
        Kokkos::RangePolicy<execution_space>(execution, 0, output.extent(0)),
        ZeroBindingPrimaryRhsFunctor{output.data()});
  };
  auto outer = [](const execution_space& execution, const double,
                  const teuk::Plus2SpatialAggregateView& sums,
                  const teuk::Plus2ProductionJkAggregateView& tangents,
                  const teuk::Plus2LiveOuterWriteTarget target) {
    Kokkos::parallel_for(
        "write_binding_outer_stage",
        Kokkos::RangePolicy<execution_space>(
            execution,
            0, binding_radial_count * binding_theta_count),
        WriteBindingOuterFunctor{
            sums.data(), tangents.data(), target.projected_sum_value.data(),
            target.outer_derivative_value.data(),
            target.projected_sum_value_stamps.data(),
            target.outer_derivative_value_stamps.data(), target.generation,
            binding_radial_count, binding_theta_count});
  };
  auto companion_rhs = [&]
      (const execution_space& execution, const double stage_time,
       const std::uint64_t generation, const auto& primary_stage,
       const auto& bianchi_stage,
       const teuk::Plus2TransportedCurvatureStage& curvature,
       const teuk::Plus2BianchiDerivativeStage& derivatives,
       const auto& companion_stage, const auto& output) {
    const C* const expected_bianchi_stage =
        result.generations.size() % 4 == 0 ? bianchi_state_pointer
                                           : bianchi_stage_pointer;
    bianchi_stage_views_match =
        bianchi_stage_views_match &&
        bianchi_stage.data() == expected_bianchi_stage;
    companion_primary_stages.push_back(primary_stage.data());
    result.generations.push_back(generation);
    result.stage_times.push_back(stage_time);
    const teuk::Plus2PrimitiveReconstructionStage reconstruction{
        generation, wrong_shape ? fixture.wrong_value : fixture.value,
        fixture.tangent, fixture.second,
        fixture.value_stamps, fixture.tangent_stamps,
        fixture.second_stamps};
    const teuk::Plus2LiveSourceCapability capability{
        true, true, true, true, teuk::RadialDiscretization::D105, generation};
    auto source = [&](const execution_space& source_execution,
                      const double source_time, const auto&,
                      const teuk::Plus2StageSourceTarget target) {
      fixture.composition->evaluate_stage(
          source_execution, source_time, reconstruction, curvature,
          derivatives, capability, activation, target,
          *fixture.primitive_producer, outer);
    };
    auto angular = [](const execution_space& angular_execution, const double,
                      const auto&, const auto& angular_laplacian) {
      Kokkos::parallel_for(
          "zero_binding_angular_laplacian",
          Kokkos::RangePolicy<execution_space>(angular_execution, 0,
                                               angular_laplacian.size()),
          ZeroBindingAngularFunctor{angular_laplacian.data()});
    };
    fixture.pipeline->evaluate_common_stage_rhs(
        execution, stage_time, primary_stage, companion_stage, output,
        activation, source, angular);
  };

  const std::array<const C*, 7> pointers_before{
      fixture.primary.data(),
      fixture.transport->flat_state().data(),
      fixture.pipeline->companion_state().data(),
      fixture.primary_workspace.stage.data(),
      fixture.transport->rk_workspace().stage.data(),
      fixture.pipeline->companion_storage().rk_workspace().stage.data(),
      fixture.pipeline->forcing().data()};
  if (audit_hot_path) {
    binding_allocations = 0;
    binding_fences = 0;
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        count_binding_allocation);
    Kokkos::Tools::Experimental::set_begin_fence_callback(count_binding_fence);
  }
  const double step = final_time / static_cast<double>(steps);
  for (int n = 0; n < steps; ++n) {
    teuk::device_one_way_bianchi_companion_rk4_step(
        fixture.execution, fixture.primary,
        fixture.pipeline->companion_state(), static_cast<double>(n) * step,
        step, primary_rhs, primary_producer, companion_rhs,
        fixture.primary_workspace, *fixture.transport,
        fixture.pipeline->companion_storage().rk_workspace(),
        fixture.bianchi_capability());
  }
  if (audit_hot_path) {
    Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    result.allocations = binding_allocations;
    result.fences = binding_fences;
  }
  fixture.execution.fence("finish concrete binding step");
  const auto host_forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.pipeline->forcing());
  result.forcing.assign(host_forcing.data(),
                        host_forcing.data() + host_forcing.size());
  const auto host_companion = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.pipeline->companion_state());
  result.companion.assign(host_companion.data(),
                          host_companion.data() + host_companion.size());
  const auto host_primary = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.primary);
  result.primary = host_primary(0);
  const std::array<const C*, 7> pointers_after{
      fixture.primary.data(),
      fixture.transport->flat_state().data(),
      fixture.pipeline->companion_state().data(),
      fixture.primary_workspace.stage.data(),
      fixture.transport->rk_workspace().stage.data(),
      fixture.pipeline->companion_storage().rk_workspace().stage.data(),
      fixture.pipeline->forcing().data()};
  result.pointers_stable = pointers_before == pointers_after;
  result.common_stage_inputs = bianchi_stage_views_match &&
      primary_rhs_stages == producer_primary_stages &&
      primary_rhs_stages == companion_primary_stages &&
      primary_rhs_times == producer_times &&
      primary_rhs_times == result.stage_times;
  result.activation_after = activation;
  return result;
}

TEST_CASE("plus2 concrete producer binds one three-state common RK step") {
  const auto base = run_binding_evolution(1.0, 1, 0.01, false, true);
  const auto scaled = run_binding_evolution(-1.7, 1, 0.01, false, false);
  const auto stale = run_binding_evolution(1.0, 1, 0.01, true, false);
  const std::array<std::uint64_t, 4> expected_generations{1, 2, 3, 4};
  const std::array<double, 4> expected_times{0.0, 0.005, 0.005, 0.01};
  CHECK(base.generations.size() == expected_generations.size());
  CHECK(base.stage_times.size() == expected_times.size());
  for (std::size_t stage = 0; stage < expected_generations.size(); ++stage) {
    CHECK(base.generations[stage] == expected_generations[stage]);
    CHECK_NEAR(base.stage_times[stage], expected_times[stage], 2.0e-16);
  }
  CHECK_COMPLEX_NEAR(base.primary, C(1.0, 0.0), 0.0);
  CHECK_COMPLEX_NEAR(scaled.primary, C(-1.7, 0.0), 0.0);
  double maximum = 0.0;
  for (std::size_t i = 0; i < base.forcing.size(); ++i) {
    maximum = std::max(maximum, Kokkos::abs(base.forcing[i]));
    CHECK_COMPLEX_NEAR(scaled.forcing[i], 1.7 * 1.7 * base.forcing[i],
                       2.0e-9);
    CHECK_COMPLEX_NEAR(stale.forcing[i], C{}, 0.0);
  }
  CHECK(maximum > 1.0e-9);
  double companion_maximum = 0.0;
  for (std::size_t i = 0; i < base.companion.size(); ++i) {
    companion_maximum =
        std::max(companion_maximum, Kokkos::abs(base.companion[i]));
    CHECK_COMPLEX_NEAR(scaled.companion[i],
                       1.7 * 1.7 * base.companion[i], 2.0e-9);
    CHECK_COMPLEX_NEAR(stale.companion[i], C{}, 0.0);
  }
  CHECK(companion_maximum > 1.0e-12);
  CHECK(base.allocations == 0);
  CHECK(base.fences == 0);
  CHECK(base.pointers_stable);
  CHECK(base.common_stage_inputs);
  CHECK(base.activation_before.active == base.activation_after.active);
  CHECK(base.activation_before.activation_time ==
        base.activation_after.activation_time);
  CHECK(base.activation_before.consecutive_passes ==
        base.activation_after.consecutive_passes);
  CHECK(base.activation_before.last_eligibility_time ==
        base.activation_after.last_eligibility_time);

  bool wrong_shape_rejected = false;
  try {
    (void)run_binding_evolution(1.0, 1, 0.01, false, false, true);
  } catch (const std::invalid_argument&) {
    wrong_shape_rejected = true;
  }
  CHECK(wrong_shape_rejected);
}

double binding_difference_norm(const std::vector<C>& left,
                               const std::vector<C>& right) {
  double result = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    result = std::max(result, Kokkos::abs(left[i] - right[i]));
  }
  return result;
}

TEST_CASE("plus2 three-state live seam has fourth-order common-stage time RK4") {
  // The graph is homogeneous in the primary amplitude and quadratic in its
  // forcing.  This test-only scale keeps the fine-grid truncation error well
  // above binary64 roundoff without changing the RK tableau or CFL scale.
  constexpr double convergence_amplitude = 2000.0;
  const auto coarse =
      run_binding_evolution(convergence_amplitude, 2, 0.02, false, false)
          .companion;
  const auto medium =
      run_binding_evolution(convergence_amplitude, 4, 0.02, false, false)
          .companion;
  const auto fine =
      run_binding_evolution(convergence_amplitude, 8, 0.02, false, false)
          .companion;
  const auto reference =
      run_binding_evolution(convergence_amplitude, 32, 0.02, false, false)
          .companion;
  const double coarse_error = binding_difference_norm(coarse, reference);
  const double medium_error = binding_difference_norm(medium, reference);
  const double fine_error = binding_difference_norm(fine, reference);
  std::cout << "plus2 three-state fixed-space RK4 errors " << coarse_error
            << ' ' << medium_error << ' ' << fine_error << " ratios "
            << coarse_error / medium_error << ' '
            << medium_error / fine_error << '\n';
  CHECK(std::isfinite(coarse_error));
  CHECK(std::isfinite(medium_error));
  CHECK(std::isfinite(fine_error));
  CHECK(coarse_error > medium_error);
  CHECK(medium_error > fine_error);
  CHECK(fine_error > 1.0e-11);
  CHECK(coarse_error < 1.0e-6);
  CHECK(coarse_error / medium_error > 12.0);
  CHECK(medium_error / fine_error > 12.0);
}

}  // namespace
