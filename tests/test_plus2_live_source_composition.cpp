#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "teuk/plus2_live_source_composition.hpp"

namespace {

using C = teuk::Complex;

int live_allocations = 0;
int live_fences = 0;

void count_live_allocation(Kokkos::Tools::SpaceHandle, const char*,
                           const void*, std::uint64_t) {
  ++live_allocations;
}

void count_live_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++live_fences;
}

KOKKOS_INLINE_FUNCTION std::size_t live_flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

struct WriteLiveSourceSlotsFunctor {
  C* primitive_value;
  C* primitive_tangent;
  C* jk_value;
  C* jk_tangent;
  C* q_value;
  std::uint64_t* primitive_value_stamps;
  std::uint64_t* primitive_tangent_stamps;
  std::uint64_t* jk_value_stamps;
  std::uint64_t* jk_tangent_stamps;
  std::uint64_t* q_value_stamps;
  double amplitude;
  std::uint64_t generation;
  std::size_t radial_count;
  std::size_t theta_count;
  bool omit_last_q_stamp;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t mode = flat / (radial_count * theta_count);
    const std::size_t within = flat % (radial_count * theta_count);
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within % theta_count;
    const double x = 0.04 * mode + 0.01 * radial + 0.02 * theta;
    constexpr std::size_t pc =
        static_cast<std::size_t>(teuk::Plus2SpatialPrimitive::Count);
    for (std::size_t field = 2; field < pc; ++field) {
      const std::size_t index = live_flat4(
          mode, field, radial, theta, pc, radial_count, theta_count);
      primitive_value[index] =
          amplitude * C(0.21 + x + 0.01 * field, -0.08 + 0.005 * field);
      primitive_tangent[index] =
          amplitude * C(-0.04 + 0.003 * field, 0.06 + x);
      primitive_value_stamps[index] = generation;
      primitive_tangent_stamps[index] = generation;
    }
    constexpr std::size_t jc = static_cast<std::size_t>(
        teuk::Plus2ProductionJkDerivative::Count);
    for (std::size_t field = 0; field < jc; ++field) {
      const std::size_t index = live_flat4(
          mode, field, radial, theta, jc, radial_count, theta_count);
      jk_value[index] =
          amplitude * C(0.13 + x + 0.007 * field, 0.02 - 0.004 * field);
      jk_tangent[index] =
          amplitude * C(-0.02 + 0.006 * field, 0.03 + x);
      jk_value_stamps[index] = generation;
      jk_tangent_stamps[index] = generation;
    }
    constexpr std::size_t qc = static_cast<std::size_t>(
        teuk::Plus2ProductionQDerivative::Count);
    for (std::size_t field = 0; field < qc; ++field) {
      const std::size_t index = live_flat4(
          mode, field, radial, theta, qc, radial_count, theta_count);
      q_value[index] =
          amplitude * C(0.17 + x + 0.009 * field, -0.05 + 0.003 * field);
      if (!(omit_last_q_stamp && mode == 0 && radial == 0 && theta == 0 &&
            field + 1 == qc)) {
        q_value_stamps[index] = generation;
      }
    }
  }
};

struct WriteLiveOuterSlotsFunctor {
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
    const std::size_t mode = flat / (radial_count * theta_count);
    const std::size_t within = flat % (radial_count * theta_count);
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within % theta_count;
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(teuk::Plus2SpatialAggregate::Count);
    for (std::size_t field = 0; field < aggregate_count; ++field) {
      const std::size_t index = live_flat4(mode, field, radial, theta,
                                           aggregate_count, radial_count,
                                           theta_count);
      projected[index] = sums[index];
      projected_stamps[index] = generation;
    }
    constexpr std::size_t tangent_count = static_cast<std::size_t>(
        teuk::Plus2ProductionJkAggregate::Count);
    constexpr std::size_t outer_count = static_cast<std::size_t>(
        teuk::Plus2SpatialOuterDerivative::Count);
    for (std::size_t field = 0; field < outer_count; ++field) {
      const std::size_t output_index = live_flat4(
          mode, field, radial, theta, outer_count, radial_count, theta_count);
      const std::size_t input_index = live_flat4(
          mode, field, radial, theta, tangent_count, radial_count,
          theta_count);
      outer[output_index] = tangents[input_index];
      outer_stamps[output_index] = generation;
    }
  }
};

static_assert(std::is_trivially_copyable_v<WriteLiveSourceSlotsFunctor>);
static_assert(std::is_trivially_copyable_v<WriteLiveOuterSlotsFunctor>);
static_assert(sizeof(WriteLiveSourceSlotsFunctor) < 1800);
static_assert(sizeof(WriteLiveOuterSlotsFunctor) < 1800);

struct Geometry {
  teuk::Plus2SpatialThetaView cosine;
  teuk::Plus2SpatialThetaView sine;

  explicit Geometry(const std::size_t count)
      : cosine("live_cosine", count), sine("live_sine", count) {
    auto hc = Kokkos::create_mirror_view(cosine);
    auto hs = Kokkos::create_mirror_view(sine);
    for (std::size_t i = 0; i < count; ++i) {
      hc(i) = -0.78 + 1.56 * static_cast<double>(i + 1) /
                         static_cast<double>(count + 1);
      hs(i) = Kokkos::sqrt(1.0 - hc(i) * hc(i));
    }
    Kokkos::deep_copy(cosine, hc);
    Kokkos::deep_copy(sine, hs);
  }
};

struct LiveFixture {
  static constexpr std::size_t radial_count = 24;
  static constexpr std::size_t theta_count = 8;
  teuk::ExecutionSpace execution;
  teuk::ModeRegistry registry{{-1, 0, 1}, {-1, 1}, {0}};
  teuk::UniformRadialGrid grid{radial_count, 0.0, 0.62};
  teuk::KerrParameters parameters{1.0, 0.71, 1.4};
  Geometry angles{theta_count};
  teuk::Plus2LiveSourceComposition<> composition{
      execution, registry, grid, parameters, 3,
      static_cast<int>(theta_count), angles.cosine, angles.sine,
      "live_composition"};
  teuk::Plus2SpatialRank4View reconstruction{
      "live_reconstruction", registry.size(), 3, radial_count, theta_count};
  teuk::Plus2SpatialRank4View tangent{
      "live_tangent", registry.size(), 3, radial_count, theta_count};
  teuk::Plus2SpatialRank4View second{
      "live_second", registry.size(), 3, radial_count, theta_count};
  teuk::Plus2TransportedCurvatureStorageView curvature{
      "live_curvature", registry.size(),
      static_cast<std::size_t>(
          teuk::Plus2TransportedCurvatureComponent::Count),
      radial_count, theta_count};
  teuk::Plus2LiveStampView curvature_stamps{
      "live_curvature_stamps", registry.size(),
      static_cast<std::size_t>(
          teuk::Plus2TransportedCurvatureComponent::Count),
      radial_count, theta_count};
  teuk::Plus2CompanionForcingView forcing{
      "live_forcing", registry.targets().size(), radial_count, theta_count};

  explicit LiveFixture(const double amplitude = 1.0,
                       const std::uint64_t generation = 7) {
    const auto reconstruction_view = reconstruction;
    const auto tangent_view = tangent;
    const auto second_view = second;
    const auto curvature_view = curvature;
    const auto stamp_view = curvature_stamps;
    Kokkos::parallel_for(
        "fill_live_fixture",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(
            execution, 0, registry.size() * radial_count * theta_count),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t mode = flat / (radial_count * theta_count);
          const std::size_t within = flat % (radial_count * theta_count);
          const std::size_t radial = within / theta_count;
          const std::size_t theta = within % theta_count;
          const double x = 0.02 * static_cast<double>(radial) +
                           0.03 * static_cast<double>(theta) +
                           0.05 * static_cast<double>(mode);
          for (std::size_t field = 0; field < 3; ++field) {
            reconstruction_view(mode, field, radial, theta) =
                amplitude * C(0.12 + x + 0.01 * field,
                              -0.07 + 0.02 * field);
            tangent_view(mode, field, radial, theta) =
                amplitude * C(-0.03 + 0.01 * field, 0.02 + x);
            second_view(mode, field, radial, theta) =
                amplitude * C(0.01 + x, -0.015 + 0.01 * field);
          }
          constexpr std::size_t curvature_count =
              static_cast<std::size_t>(
                  teuk::Plus2TransportedCurvatureComponent::Count);
          for (std::size_t field = 0; field < curvature_count; ++field) {
            curvature_view(mode, field, radial, theta) =
                amplitude * C(0.18 + x + 0.02 * field,
                              -0.09 + 0.015 * field);
            stamp_view(mode, field, radial, theta) = generation;
          }
        });
  }

  teuk::Plus2LiveSourceCapability capability(
      const std::uint64_t generation = 7) const {
    return {true, true, true, true, teuk::RadialDiscretization::D105,
            generation};
  }

  void restamp(const std::uint64_t generation) {
    const auto stamps = curvature_stamps;
    Kokkos::parallel_for(
        "restamp_live_curvature",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, stamps.size()),
        KOKKOS_LAMBDA(const std::size_t i) { stamps.data()[i] = generation; });
  }
};

auto complete_source_producer(const double amplitude,
                              const bool omit_last_q_stamp = false,
                              std::vector<int>* order = nullptr) {
  return [=](const teuk::ExecutionSpace& execution, const double,
             const auto&, const auto&, const auto&, const auto&, const auto&,
             const teuk::Plus2TransportedCurvatureStage&,
             const teuk::Plus2LiveSourceWriteTarget target) {
    if (order) order->push_back(1);
    const std::size_t mode_count = target.primitive_value.extent(0);
    const std::size_t radial_count = target.primitive_value.extent(2);
    const std::size_t theta_count = target.primitive_value.extent(3);
    const std::uint64_t generation = target.generation;
    Kokkos::parallel_for(
        "write_live_source_slots",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(
            execution, 0, mode_count * radial_count * theta_count),
        WriteLiveSourceSlotsFunctor{
            target.primitive_value.data(),
            target.primitive_tangent.data(),
            target.jk_derivative_value.data(),
            target.jk_derivative_tangent.data(),
            target.q_derivative_value.data(),
            target.primitive_value_stamps.data(),
            target.primitive_tangent_stamps.data(),
            target.jk_derivative_value_stamps.data(),
            target.jk_derivative_tangent_stamps.data(),
            target.q_derivative_value_stamps.data(),
            amplitude,
            generation,
            radial_count,
            theta_count,
            omit_last_q_stamp});
  };
}

auto complete_outer_producer(std::vector<int>* order = nullptr) {
  return [=](const teuk::ExecutionSpace& execution, const double,
             const teuk::Plus2SpatialAggregateView& sums,
             const teuk::Plus2ProductionJkAggregateView& tangents,
             const teuk::Plus2LiveOuterWriteTarget target) {
    if (order) order->push_back(2);
    const std::size_t modes = sums.extent(0);
    const std::size_t radial_count = sums.extent(2);
    const std::size_t theta_count = sums.extent(3);
    const std::uint64_t generation = target.generation;
    Kokkos::parallel_for(
        "write_live_outer_slots",
        Kokkos::RangePolicy<teuk::ExecutionSpace>(
            execution, 0, modes * radial_count * theta_count),
        WriteLiveOuterSlotsFunctor{
            sums.data(), tangents.data(), target.projected_sum_value.data(),
            target.outer_derivative_value.data(),
            target.projected_sum_value_stamps.data(),
            target.outer_derivative_value_stamps.data(), generation,
            radial_count, theta_count});
  };
}

std::vector<C> evaluate_fixture(LiveFixture& fixture, const double amplitude,
                                const bool omit_stamp = false,
                                std::vector<int>* order = nullptr,
                                const bool active = true) {
  const teuk::SourceActivationState activation{
      active, active ? 0.2 : -1.0, active ? 3 : 0, 0.3};
  fixture.composition.evaluate_stage(
      fixture.execution, 0.42, fixture.reconstruction, fixture.tangent,
      fixture.second,
      {fixture.curvature, fixture.curvature_stamps}, fixture.capability(),
      activation, {activation, fixture.forcing},
      complete_source_producer(amplitude, omit_stamp, order),
      complete_outer_producer(order));
  fixture.execution.fence("finish live composition fixture");
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         fixture.forcing);
  return std::vector<C>(host.data(), host.data() + host.size());
}

TEST_CASE("plus2 live composition is same-stage ordered and quadratic") {
  std::vector<int> order;
  LiveFixture base(1.0);
  const auto value = evaluate_fixture(base, 1.0, false, &order);
  CHECK(order == std::vector<int>({1, 2}));
  LiveFixture scaled(-1.7);
  const auto scaled_value = evaluate_fixture(scaled, -1.7);
  double maximum = 0.0;
  for (std::size_t i = 0; i < value.size(); ++i) {
    maximum = std::max(maximum, Kokkos::abs(value[i]));
    CHECK_COMPLEX_NEAR(scaled_value[i], 1.7 * 1.7 * value[i], 2.0e-9);
  }
  CHECK(maximum > 1.0e-7);
  CHECK(base.composition.radial_discretization() ==
        teuk::RadialDiscretization::D105);
}

TEST_CASE("plus2 live composition fails closed on missing slots and activation") {
  LiveFixture fixture;
  auto missing = evaluate_fixture(fixture, 1.0, true);
  CHECK_COMPLEX_NEAR(missing[0], C{}, 0.0);
  bool retained_ready_point = false;
  for (std::size_t i = 1; i < missing.size(); ++i) {
    retained_ready_point = retained_ready_point || Kokkos::abs(missing[i]) > 0.0;
  }
  CHECK(retained_ready_point);
  Kokkos::deep_copy(fixture.forcing, C(9.0, -3.0));
  fixture.restamp(8);
  const teuk::SourceActivationState inactive_activation{false, -1.0, 0, 0.3};
  fixture.composition.evaluate_stage(
      fixture.execution, 0.43, fixture.reconstruction, fixture.tangent,
      fixture.second, {fixture.curvature, fixture.curvature_stamps},
      fixture.capability(8), inactive_activation,
      {inactive_activation, fixture.forcing}, complete_source_producer(1.0),
      complete_outer_producer());
  fixture.execution.fence("finish inactive live composition");
  const auto inactive_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.forcing);
  std::vector<C> inactive(inactive_host.data(),
                          inactive_host.data() + inactive_host.size());
  for (const C value : inactive) CHECK_COMPLEX_NEAR(value, C{}, 0.0);
}

TEST_CASE("plus2 live composition rejects absent scri and common-stage authority") {
  LiveFixture fixture;
  const teuk::SourceActivationState activation{true, 0.0, 1, 0.0};
  auto capability = fixture.capability();
  capability.independently_qualified_scri_coefficients = false;
  bool scri_rejected = false;
  try {
    fixture.composition.evaluate_stage(
        fixture.execution, 0.0, fixture.reconstruction, fixture.tangent,
        fixture.second, {fixture.curvature, fixture.curvature_stamps},
        capability, activation, {activation, fixture.forcing},
        complete_source_producer(1.0), complete_outer_producer());
  } catch (const std::invalid_argument&) {
    scri_rejected = true;
  }
  CHECK(scri_rejected);
  capability = fixture.capability();
  capability.transported_curvature_in_common_rk_state = false;
  bool stage_rejected = false;
  try {
    fixture.composition.evaluate_stage(
        fixture.execution, 0.0, fixture.reconstruction, fixture.tangent,
        fixture.second, {fixture.curvature, fixture.curvature_stamps},
        capability, activation, {activation, fixture.forcing},
        complete_source_producer(1.0), complete_outer_producer());
  } catch (const std::invalid_argument&) {
    stage_rejected = true;
  }
  CHECK(stage_rejected);
}

TEST_CASE("plus2 live composition stage allocates and fences nothing") {
  LiveFixture fixture;
  const teuk::SourceActivationState activation{true, 0.0, 1, 0.0};
  auto source = complete_source_producer(1.0);
  auto outer = complete_outer_producer();
  fixture.composition.evaluate_stage(
      fixture.execution, 0.0, fixture.reconstruction, fixture.tangent,
      fixture.second, {fixture.curvature, fixture.curvature_stamps},
      fixture.capability(), activation, {activation, fixture.forcing}, source,
      outer);
  fixture.execution.fence("warm live source composition");
  fixture.restamp(8);
  live_allocations = 0;
  live_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_live_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_live_fence);
  fixture.composition.evaluate_stage(
      fixture.execution, 0.1, fixture.reconstruction, fixture.tangent,
      fixture.second, {fixture.curvature, fixture.curvature_stamps},
      fixture.capability(8), activation, {activation, fixture.forcing}, source,
      outer);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  fixture.execution.fence("finish live source no-allocation check");
  CHECK(live_allocations == 0);
  CHECK(live_fences == 0);
}

}  // namespace
