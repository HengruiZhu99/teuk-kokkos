#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
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
  teuk::Plus2CompanionForcingView forcing{
      "binding_forcing", registry.targets().size(), binding_radial_count,
      binding_theta_count};
  Kokkos::View<C*> primary{"binding_primary", 1};
  teuk::DeviceRK4Workspace<C, execution_space> primary_workspace{1};
  std::unique_ptr<teuk::Plus2BianchiTransport<execution_space>> transport;
  std::unique_ptr<teuk::Plus2SourcePrimitiveSpatialProducer<execution_space>>
      primitive_producer;
  std::unique_ptr<teuk::Plus2LiveSourceComposition<execution_space>>
      composition;

  explicit BindingFixture(const double amplitude) {
    const auto angular_grid = teuk::angular::gauss_legendre(
        static_cast<int>(binding_theta_count));
    auto host_cos = Kokkos::create_mirror_view(cos_theta);
    auto host_sin = Kokkos::create_mirror_view(sin_theta);
    for (std::size_t theta = 0; theta < binding_theta_count; ++theta) {
      host_cos(theta) = angular_grid.x[theta];
      host_sin(theta) =
          std::sqrt(1.0 - angular_grid.x[theta] * angular_grid.x[theta]);
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
  std::array<std::uint64_t, 4> generations{};
  std::size_t stage_count = 0;
  C primary{};
  int allocations = 0;
  int fences = 0;
  teuk::SourceActivationState activation_before{};
  teuk::SourceActivationState activation_after{};
};

BindingResult run_binding_step(const double amplitude,
                               const bool stale_reconstruction,
                               const bool audit_hot_path,
                               const bool wrong_shape = false) {
  BindingFixture fixture(amplitude);
  const teuk::SourceActivationState activation{true, 0.0, 3, 0.0};
  BindingResult result;
  result.activation_before = activation;
  auto primary_producer =
      [&](const execution_space& execution, const double, const auto& primary,
          const teuk::Plus2BianchiPrimaryWriteTarget target) {
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
  auto primary_rhs = [](const execution_space& execution, const double,
                        const auto&, const auto& output) {
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
  auto observer = [&](const execution_space& execution, const double stage_time,
                      const std::uint64_t generation,
                      const teuk::Plus2TransportedCurvatureStage& curvature,
                      const teuk::Plus2BianchiDerivativeStage& derivatives) {
    result.generations[result.stage_count++] = generation;
    const teuk::Plus2PrimitiveReconstructionStage reconstruction{
        generation, wrong_shape ? fixture.wrong_value : fixture.value,
        fixture.tangent, fixture.second,
        fixture.value_stamps, fixture.tangent_stamps,
        fixture.second_stamps};
    const teuk::Plus2LiveSourceCapability capability{
        true, true, true, true, teuk::RadialDiscretization::D105, generation};
    fixture.composition->evaluate_stage(
        execution, stage_time, reconstruction, curvature, derivatives,
        capability, activation, {activation, fixture.forcing},
        *fixture.primitive_producer, outer);
  };

  if (audit_hot_path) {
    binding_allocations = 0;
    binding_fences = 0;
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        count_binding_allocation);
    Kokkos::Tools::Experimental::set_begin_fence_callback(count_binding_fence);
  }
  teuk::device_one_way_bianchi_transport_rk4_step(
      fixture.execution, fixture.primary, 0.0, 0.01, primary_rhs,
      primary_producer, observer, fixture.primary_workspace,
      *fixture.transport, fixture.bianchi_capability());
  if (audit_hot_path) {
    Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
    result.allocations = binding_allocations;
    result.fences = binding_fences;
  }
  fixture.execution.fence("finish concrete binding step");
  const auto host_forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.forcing);
  result.forcing.assign(host_forcing.data(),
                        host_forcing.data() + host_forcing.size());
  const auto host_primary = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.primary);
  result.primary = host_primary(0);
  result.activation_after = activation;
  return result;
}

TEST_CASE("plus2 concrete producer binds one common Bianchi RK step") {
  const auto base = run_binding_step(1.0, false, true);
  const auto scaled = run_binding_step(-1.7, false, false);
  const auto stale = run_binding_step(1.0, true, false);
  const std::array<std::uint64_t, 4> expected_generations{1, 2, 3, 4};
  CHECK(base.stage_count == 4);
  CHECK(base.generations == expected_generations);
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
  CHECK(base.allocations == 0);
  CHECK(base.fences == 0);
  CHECK(base.activation_before.active == base.activation_after.active);
  CHECK(base.activation_before.activation_time ==
        base.activation_after.activation_time);
  CHECK(base.activation_before.consecutive_passes ==
        base.activation_after.consecutive_passes);
  CHECK(base.activation_before.last_eligibility_time ==
        base.activation_after.last_eligibility_time);

  bool wrong_shape_rejected = false;
  try {
    (void)run_binding_step(1.0, false, false, true);
  } catch (const std::invalid_argument&) {
    wrong_shape_rejected = true;
  }
  CHECK(wrong_shape_rejected);
}

}  // namespace
