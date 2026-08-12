#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/plus2_live_source_composition.hpp"
#include "teuk/routeb_angular_jet_coordinator.hpp"

namespace {

using C = teuk::Complex;
using Execution = teuk::ExecutionSpace;

int routeb_live_allocations = 0;
int routeb_live_copies = 0;
int routeb_live_fences = 0;

void count_routeb_live_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                  const void*, std::uint64_t) {
  ++routeb_live_allocations;
}

void count_routeb_live_copy(Kokkos::Tools::SpaceHandle, const char*,
                            const void*, Kokkos::Tools::SpaceHandle,
                            const char*, const void*, std::uint64_t) {
  ++routeb_live_copies;
}

void count_routeb_live_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++routeb_live_fences;
}

struct RouteBLiveFixture {
  static constexpr std::size_t radial_count = 25;
  static constexpr int theta_count = 12;
  static constexpr int ell_max = 5;

  Execution execution;
  teuk::ModeRegistry registry{{-4, -2, 0, 2, 4}, {-2, 2}, {-4, 0, 4}};
  teuk::KerrParameters background{1.0, 0.63, 1.4};
  teuk::TeukolskyParameters primary_parameters;
  teuk::UniformRadialGrid grid;
  teuk::Plus2SpatialThetaView cos_theta{"routeb_live_cos", theta_count};
  teuk::Plus2SpatialThetaView sin_theta{"routeb_live_sin", theta_count};
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> primary{
      "routeb_live_primary", registry.size(), 3, radial_count, theta_count};
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps{"routeb_live_primary_stamps", registry.size(),
                     radial_count, theta_count};
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> reconstruction{
      "routeb_live_reconstruction", registry.size(), 7, radial_count,
      theta_count};
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps{"routeb_live_reconstruction_stamps",
                            registry.size(), radial_count, theta_count};
  teuk::RouteBAngularJetCoordinator<Execution> coordinator;
  teuk::Plus2RouteBCurvatureSpatialProvider<Execution> curvature;
  teuk::Plus2SourcePrimitiveSpatialProducer<Execution> primitive;
  teuk::Plus2SourceOuterSpatialProducer<Execution> outer;
  teuk::Plus2LiveSourceComposition<Execution> composition;
  teuk::Plus2CompanionForcingView forcing{
      "routeb_live_forcing", registry.targets().size(), radial_count,
      theta_count};
  teuk::SourceActivationState activation{true, 0.0, 4, 0.0};

  RouteBLiveFixture()
      : primary_parameters(make_primary_parameters()),
        grid(radial_count, 0.0, future_horizon()),
        coordinator(execution, registry, grid, ell_max, theta_count,
                    primary_parameters, "routeb_live_coordinator"),
        curvature(execution, registry, grid, background, ell_max, cos_theta,
                  sin_theta, "routeb_live_curvature"),
        primitive(execution, registry, grid, background, ell_max, cos_theta,
                  sin_theta, "routeb_live_primitive",
                  teuk::RadialDiscretization::D105),
        outer(execution, registry, grid, background, ell_max, cos_theta,
              sin_theta, "routeb_live_outer",
              teuk::RadialDiscretization::D105),
        composition(execution, registry, grid, background, ell_max,
                    theta_count, cos_theta, sin_theta,
                    "routeb_live_composition",
                    teuk::RadialDiscretization::D105) {
    initialize_geometry_and_h0();
  }

  double future_horizon() const {
    return background.compactification_length *
           background.compactification_length /
           (background.mass +
            std::sqrt(background.mass * background.mass -
                      background.spin * background.spin));
  }

  teuk::TeukolskyParameters make_primary_parameters() const {
    teuk::TeukolskyParameters result;
    result.mass = background.mass;
    result.spin = background.spin;
    result.compactification_length = background.compactification_length;
    result.spin_weight = -2;
    result.azimuthal_mode = 0;
    result.reduction_damping = 0.17;
    return result;
  }

  void initialize_geometry_and_h0() {
    const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
    auto host_cos = Kokkos::create_mirror_view(cos_theta);
    auto host_sin = Kokkos::create_mirror_view(sin_theta);
    auto host_primary = Kokkos::create_mirror_view(primary);
    auto host_reconstruction = Kokkos::create_mirror_view(reconstruction);
    for (int theta = 0; theta < theta_count; ++theta) {
      host_cos(theta) = angular_grid.x[theta];
      host_sin(theta) = std::sqrt(std::max(0.0, 1.0 -
                                                   angular_grid.x[theta] *
                                                       angular_grid.x[theta]));
    }
    for (std::size_t mode_index = 0; mode_index < registry.size();
         ++mode_index) {
      const int mode = registry.modes()[mode_index];
      const bool parent = registry.is_parent(mode);
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        const double radius = grid.coordinate(radial);
        for (int theta = 0; theta < theta_count; ++theta) {
          const double angle = angular_grid.theta(theta);
          for (std::size_t field = 0; field < 3; ++field) {
            const double harmonic =
                teuk::angular::spin_weighted_harmonic_theta(
                    4, mode, -2, angle);
            host_primary(mode_index, field, radial, theta) =
                parent ? C(0.021 * (field + 1), -0.013 * (field + 2)) *
                             std::exp((0.17 + 0.03 * field) * radius) *
                             harmonic
                       : C{};
          }
          constexpr int spins[7]{-1, -2, 0, -2, -1, -1, 0};
          for (std::size_t field = 0; field < 7; ++field) {
            const int ell = std::max(4, std::max(std::abs(mode),
                                                std::abs(spins[field])));
            const double harmonic =
                teuk::angular::spin_weighted_harmonic_theta(
                    ell, mode, spins[field], angle);
            host_reconstruction(mode_index, field, radial, theta) =
                parent ? C(0.017 * (field + 1), 0.009 * (field + 2)) *
                             std::exp((0.11 + 0.02 * field) * radius) *
                             harmonic
                       : C{};
          }
        }
      }
    }
    Kokkos::deep_copy(execution, cos_theta, host_cos);
    Kokkos::deep_copy(execution, sin_theta, host_sin);
    Kokkos::deep_copy(execution, primary, host_primary);
    Kokkos::deep_copy(execution, reconstruction, host_reconstruction);
  }

  void close_tower(const std::uint64_t generation) {
    Kokkos::deep_copy(execution, primary_stamps, generation);
    Kokkos::deep_copy(execution, reconstruction_stamps, generation);
    coordinator.initialize(execution, primary, primary_stamps,
                           reconstruction, reconstruction_stamps, generation);
    coordinator.advance_to_h4(execution, generation);
  }

  void evaluate(const std::uint64_t generation,
                const teuk::Plus2RouteBConstTowerStampView& stamps) {
    composition.evaluate_routeb_stage(
        execution, 0.0,
        teuk::Plus2RouteBCurvatureTowerStage{
            generation, coordinator.reconstruction_values(), stamps},
        activation, teuk::Plus2StageSourceTarget{activation, forcing},
        curvature, primitive, outer);
  }
};

TEST_CASE("Route-B tower binds concrete curvature and live source once") {
  RouteBLiveFixture fixture;
  fixture.close_tower(101);
  fixture.evaluate(101, fixture.coordinator.reconstruction_stamps());
  fixture.execution.fence("finish warm Route-B live source stage");

  const auto warm_forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.forcing);
  const auto warm_ready = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.composition.readiness());
  const auto curvature_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.curvature.curvature_stage().stamps);
  double maximum = 0.0;
  for (std::size_t i = 0; i < warm_forcing.size(); ++i) {
    CHECK(std::isfinite(warm_forcing.data()[i].real()));
    CHECK(std::isfinite(warm_forcing.data()[i].imag()));
    maximum = std::max(maximum, Kokkos::abs(warm_forcing.data()[i]));
  }
  CHECK(maximum > 1.0e-12);
  for (std::size_t i = 0; i < warm_ready.size(); ++i)
    CHECK(warm_ready.data()[i] == 1);
  for (std::size_t i = 0; i < curvature_stamps.size(); ++i)
    CHECK(curvature_stamps.data()[i] == 101);

  fixture.close_tower(102);
  fixture.execution.fence("begin Route-B live hot-stage instrumentation");
  routeb_live_allocations = 0;
  routeb_live_copies = 0;
  routeb_live_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_routeb_live_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_routeb_live_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      count_routeb_live_fence);
  fixture.evaluate(102, fixture.coordinator.reconstruction_stamps());
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  fixture.execution.fence("finish Route-B live hot-stage instrumentation");
  CHECK(routeb_live_allocations == 0);
  CHECK(routeb_live_copies == 0);
  CHECK(routeb_live_fences == 0);

  fixture.close_tower(103);
  teuk::Plus2RouteBTowerStampView stale(
      "routeb_live_stale_stamps", 5, fixture.registry.size(),
      RouteBLiveFixture::radial_count, RouteBLiveFixture::theta_count);
  Kokkos::deep_copy(fixture.execution, stale,
                    fixture.coordinator.reconstruction_stamps());
  auto host_stale = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        stale);
  host_stale(4, fixture.registry.index(-2), 0, 0) = 102;
  Kokkos::deep_copy(fixture.execution, stale, host_stale);
  fixture.evaluate(103, stale);
  fixture.execution.fence("finish stale Route-B live source stage");
  const auto rejected_forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.forcing);
  const auto rejected_ready = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.composition.readiness());
  for (std::size_t i = 0; i < rejected_forcing.size(); ++i)
    CHECK(Kokkos::abs(rejected_forcing.data()[i]) == 0.0);
  for (std::size_t i = 0; i < rejected_ready.size(); ++i)
    CHECK(rejected_ready.data()[i] == 0);
}

}  // namespace
