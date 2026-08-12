#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/ghp.hpp"
#include "teuk/plus2_live_source_composition.hpp"
#include "teuk/plus2_source_outer_spatial.hpp"

namespace {

using C = teuk::Complex;
using execution_space = teuk::ExecutionSpace;

constexpr int outer_ell_max = 4;
constexpr int outer_theta_count = 12;
constexpr std::uint64_t outer_generation = 41;

int outer_allocations = 0;
int outer_copies = 0;
int outer_fences = 0;
int outer_launches = 0;

void count_outer_allocation(Kokkos::Tools::SpaceHandle, const char*,
                            const void*, std::uint64_t) {
  ++outer_allocations;
}

void count_outer_copy(Kokkos::Tools::SpaceHandle, const char*, const void*,
                      Kokkos::Tools::SpaceHandle, const char*, const void*,
                      std::uint64_t) {
  ++outer_copies;
}

void count_outer_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++outer_fences;
}

void count_outer_launch(const char*, std::uint32_t, std::uint64_t*) {
  ++outer_launches;
}

std::size_t a(const teuk::Plus2SpatialAggregate field) {
  return static_cast<std::size_t>(field);
}

std::size_t o(const teuk::Plus2SpatialOuterDerivative field) {
  return static_cast<std::size_t>(field);
}

double radial_profile(const double radius) {
  return std::exp(0.63 * radius) * (1.0 + 0.17 * radius);
}

double radial_profile_derivative(const double radius) {
  return std::exp(0.63 * radius) *
         (0.63 * (1.0 + 0.17 * radius) + 0.17);
}

C modal_amplitude(const int m, const double scale = 1.0) {
  return scale * C(0.19 + 0.031 * m, -0.08 + 0.023 * m);
}

double rotating_horizon_radius() {
  constexpr double mass = 1.0;
  constexpr double spin = 0.73;
  constexpr double length = 1.2;
  return length * length /
         (mass + std::sqrt(mass * mass - spin * spin));
}

struct OuterFixture {
  execution_space execution;
  teuk::ModeRegistry registry{{-2, -1, 0, 1, 2}, {-2, -1, 0, 1, 2},
                              {-2, 0, 2}};
  teuk::UniformRadialGrid grid;
  teuk::KerrParameters parameters{1.0, 0.73, 1.2};
  teuk::Plus2SpatialThetaView cos_theta{"outer_cos", outer_theta_count};
  teuk::Plus2SpatialThetaView sin_theta{"outer_sin", outer_theta_count};
  teuk::Plus2SpatialAggregateView sums{
      "outer_sums", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialAggregate::Count),
      grid.size(), outer_theta_count};
  teuk::Plus2ProductionJkAggregateView tangents{
      "outer_tangents", registry.size(),
      static_cast<std::size_t>(teuk::Plus2ProductionJkAggregate::Count),
      grid.size(), outer_theta_count};
  teuk::Plus2SpatialAggregateView projected{
      "outer_projected", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialAggregate::Count),
      grid.size(), outer_theta_count};
  teuk::Plus2SpatialOuterDerivativeView derivatives{
      "outer_derivatives", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), outer_theta_count};
  teuk::Plus2LiveStampView projected_stamps{
      "outer_projected_stamps", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialAggregate::Count),
      grid.size(), outer_theta_count};
  teuk::Plus2LiveStampView derivative_stamps{
      "outer_derivative_stamps", registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      grid.size(), outer_theta_count};
  teuk::Plus2SourceOuterSpatialProducer<execution_space> producer;

  explicit OuterFixture(const std::size_t radial_count,
                        const double scale = 1.0,
                        const double tangent_shift = 0.0)
      : grid(radial_count, 0.0, rotating_horizon_radius()),
        producer(execution, registry, grid, parameters, outer_ell_max,
                 cos_theta, sin_theta, "outer_producer",
                 teuk::RadialDiscretization::D105) {
    const auto angular_grid = teuk::angular::gauss_legendre(outer_theta_count);
    auto hcos = Kokkos::create_mirror_view(cos_theta);
    auto hsin = Kokkos::create_mirror_view(sin_theta);
    for (int node = 0; node < outer_theta_count; ++node) {
      const auto i = static_cast<std::size_t>(node);
      hcos(i) = angular_grid.x[i];
      hsin(i) = std::sqrt(1.0 - angular_grid.x[i] * angular_grid.x[i]);
    }
    Kokkos::deep_copy(execution, cos_theta, hcos);
    Kokkos::deep_copy(execution, sin_theta, hsin);

    auto hsums = Kokkos::create_mirror_view(sums);
    auto htangents = Kokkos::create_mirror_view(tangents);
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      const int m = registry.modes()[mode];
      const bool target = registry.is_target(m);
      const int j_ell = std::max(2, std::abs(m));
      const int k_ell = std::max(1, std::abs(m));
      const int q_ell = std::max(2, std::abs(m));
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        const double radius = grid.coordinate(radial);
        const double profile = radial_profile(radius);
        for (int node = 0; node < outer_theta_count; ++node) {
          const auto theta = angular_grid.theta(static_cast<std::size_t>(node));
          const auto i = static_cast<std::size_t>(node);
          const double j_low = teuk::angular::spin_weighted_harmonic_theta(
              j_ell, m, 2, theta);
          const double j_high = teuk::angular::spin_weighted_harmonic_theta(
              outer_ell_max + 1, m, 2, theta);
          const double k_low = teuk::angular::spin_weighted_harmonic_theta(
              k_ell, m, 1, theta);
          const double k_high = teuk::angular::spin_weighted_harmonic_theta(
              outer_ell_max + 1, m, 1, theta);
          const double q_low = teuk::angular::spin_weighted_harmonic_theta(
              q_ell, m, 2, theta);
          const double q_high = teuk::angular::spin_weighted_harmonic_theta(
              outer_ell_max + 1, m, 2, theta);
          const C amplitude = modal_amplitude(m, scale);
          const double hostile = target ? 1.0 : 91.0;
          hsums(mode, a(teuk::Plus2SpatialAggregate::J), radial, i) =
              hostile * amplitude * profile * (j_low + 0.37 * j_high);
          hsums(mode, a(teuk::Plus2SpatialAggregate::K), radial, i) =
              hostile * C(0.72, -0.13) * amplitude * profile *
              (k_low - 0.29 * k_high);
          hsums(mode, a(teuk::Plus2SpatialAggregate::Q), radial, i) =
              hostile * C(-0.33, 0.41) * amplitude * profile *
              (q_low + 0.23 * q_high);
          htangents(mode, 0, radial, i) =
              (0.21 + tangent_shift) * hostile * amplitude * profile *
              (j_low + 0.11 * j_high);
          htangents(mode, 1, radial, i) =
              (-0.17 + tangent_shift) * hostile * C(0.72, -0.13) *
              amplitude * profile * (k_low + 0.09 * k_high);
        }
      }
    }
    Kokkos::deep_copy(execution, sums, hsums);
    Kokkos::deep_copy(execution, tangents, htangents);
    execution.fence("initialize outer spatial fixture");
  }

  void evaluate(const std::uint64_t stage_generation = outer_generation,
                const std::uint64_t target_generation = outer_generation) {
    const teuk::Plus2LiveOuterWriteTarget target{
        target_generation, projected, derivatives, projected_stamps,
        derivative_stamps};
    producer.evaluate(
        execution, {stage_generation, sums, tangents},
        target);
  }
};

double endpoint_thorn_error(const std::size_t radial_count) {
  OuterFixture fixture(radial_count);
  fixture.evaluate();
  fixture.execution.fence("finish outer endpoint convergence fixture");
  const auto output = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.derivatives);
  const auto hcos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        fixture.cos_theta);
  const auto hsin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        fixture.sin_theta);
  double error = 0.0;
  for (const int m : fixture.registry.targets()) {
    const std::size_t mode = fixture.registry.index(m);
    const int ell = std::max(2, std::abs(m));
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double radius = fixture.grid.coordinate(radial);
      for (int node = 0; node < outer_theta_count; ++node) {
        const auto i = static_cast<std::size_t>(node);
        const double harmonic = teuk::angular::spin_weighted_harmonic_theta(
            ell, m, 2, std::acos(hcos(i)));
        const C amplitude = modal_amplitude(m);
        const C value = amplitude * radial_profile(radius) * harmonic;
        const C tangent = 0.21 * value;
        const C radial_derivative =
            amplitude * radial_profile_derivative(radius) * harmonic;
        const auto background = teuk::kerr_background_point(
            fixture.parameters, radius, hcos(i), hsin(i));
        const C exact = teuk::thorn_n_point(
            value, tangent, radial_derivative, 5, 2, 1, m, radius, hcos(i),
            fixture.parameters.mass, fixture.parameters.spin,
            fixture.parameters.compactification_length,
            background.epsilon0);
        error = std::max(
            error,
            Kokkos::abs(output(mode,
                               o(teuk::Plus2SpatialOuterDerivative::Thorn5J),
                               radial, i) -
                        exact));
      }
    }
  }
  return error;
}

}  // namespace

TEST_CASE("plus2 outer producer projects signed target bands and applies Kerr GHP operators") {
  OuterFixture fixture(49);
  fixture.evaluate();
  fixture.execution.fence("finish outer signed mode oracle");
  const auto projected = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.projected);
  const auto derivatives = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.derivatives);
  const auto projected_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.projected_stamps);
  const auto derivative_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.derivative_stamps);
  const auto hcos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        fixture.cos_theta);
  const auto hsin = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        fixture.sin_theta);

  for (std::size_t mode = 0; mode < fixture.registry.size(); ++mode) {
    const int m = fixture.registry.modes()[mode];
    for (std::size_t radial = 0; radial < fixture.grid.size(); radial += 12) {
      const double radius = fixture.grid.coordinate(radial);
      for (int node = 0; node < outer_theta_count; ++node) {
        const auto i = static_cast<std::size_t>(node);
        for (std::size_t field = 0; field < 3; ++field) {
          CHECK(projected_stamps(mode, field, radial, i) == outer_generation);
        }
        for (std::size_t field = 0; field < 2; ++field) {
          CHECK(derivative_stamps(mode, field, radial, i) == outer_generation);
        }
        if (!fixture.registry.is_target(m)) {
          for (std::size_t field = 0; field < 3; ++field) {
            CHECK_COMPLEX_NEAR(projected(mode, field, radial, i), C{}, 2e-14);
          }
          for (std::size_t field = 0; field < 2; ++field) {
            CHECK_COMPLEX_NEAR(derivatives(mode, field, radial, i), C{},
                               2e-13);
          }
          continue;
        }
        const double theta = std::acos(hcos(i));
        const C amplitude = modal_amplitude(m);
        const double profile = radial_profile(radius);
        const int j_ell = std::max(2, std::abs(m));
        const int k_ell = std::max(1, std::abs(m));
        const int q_ell = std::max(2, std::abs(m));
        const double j_harmonic =
            teuk::angular::spin_weighted_harmonic_theta(j_ell, m, 2, theta);
        const double k_harmonic =
            teuk::angular::spin_weighted_harmonic_theta(k_ell, m, 1, theta);
        const double q_harmonic =
            teuk::angular::spin_weighted_harmonic_theta(q_ell, m, 2, theta);
        const C j_value = amplitude * profile * j_harmonic;
        const C k_value = C(0.72, -0.13) * amplitude * profile * k_harmonic;
        const C q_value = C(-0.33, 0.41) * amplitude * profile * q_harmonic;
        CHECK_COMPLEX_NEAR(
            projected(mode, a(teuk::Plus2SpatialAggregate::J), radial, i),
            j_value, 4e-13);
        CHECK_COMPLEX_NEAR(
            projected(mode, a(teuk::Plus2SpatialAggregate::K), radial, i),
            k_value, 4e-13);
        CHECK_COMPLEX_NEAR(
            projected(mode, a(teuk::Plus2SpatialAggregate::Q), radial, i),
            q_value, 4e-13);

        const double raised =
            k_ell < 2
                ? 0.0
                : teuk::angular::raising_factor(k_ell, 1) *
                      teuk::angular::spin_weighted_harmonic_theta(
                          k_ell, m, 2, theta);
        const C expected_eth = teuk::eth_n_point(
            k_value, -0.17 * k_value,
            C(0.72, -0.13) * amplitude * profile * raised, 1, 2, radius,
            hsin(i), hcos(i), fixture.parameters.spin,
            fixture.parameters.compactification_length);
        CHECK_COMPLEX_NEAR(
            derivatives(mode, o(teuk::Plus2SpatialOuterDerivative::Eth6K),
                        radial, i),
            expected_eth, 2e-12);
      }
    }
  }
}

TEST_CASE("plus2 outer D10-5 thorn converges at both radial endpoints") {
  const double coarse = endpoint_thorn_error(25);
  const double medium = endpoint_thorn_error(49);
  const double fine = endpoint_thorn_error(97);
  std::cout << "plus2 outer D105 endpoint errors " << coarse << " "
            << medium << " " << fine << " ratios " << coarse / medium
            << " " << medium / fine << '\n';
  CHECK(coarse / medium > 20.0);
  CHECK(medium / fine > 20.0);
  CHECK(fine < 2.0e-9);
}

TEST_CASE("plus2 outer producer is allocation free and fails closed") {
  OuterFixture fixture(25);
  CHECK(fixture.producer.matches_configuration(
      fixture.registry, fixture.grid, fixture.parameters, outer_ell_max,
      fixture.cos_theta, fixture.sin_theta));
  teuk::Plus2SourceOuterSpatialProducer<execution_space> wrong_band(
      fixture.execution, fixture.registry, fixture.grid, fixture.parameters,
      outer_ell_max + 1, fixture.cos_theta, fixture.sin_theta,
      "outer_wrong_band", teuk::RadialDiscretization::D105);
  CHECK(!wrong_band.matches_configuration(
      fixture.registry, fixture.grid, fixture.parameters, outer_ell_max,
      fixture.cos_theta, fixture.sin_theta));
  teuk::Plus2SourcePrimitiveSpatialProducer<execution_space> primitive(
      fixture.execution, fixture.registry, fixture.grid, fixture.parameters,
      outer_ell_max, fixture.cos_theta, fixture.sin_theta,
      "outer_preflight_primitive", teuk::RadialDiscretization::D105);
  teuk::Plus2SourcePrimitiveSpatialProducer<execution_space>
      wrong_primitive_band(
          fixture.execution, fixture.registry, fixture.grid,
          fixture.parameters, outer_ell_max + 1, fixture.cos_theta,
          fixture.sin_theta, "outer_wrong_primitive_band",
          teuk::RadialDiscretization::D105);
  teuk::Plus2LiveSourceComposition<execution_space> composition(
      fixture.execution, fixture.registry, fixture.grid, fixture.parameters,
      outer_ell_max, outer_theta_count, fixture.cos_theta, fixture.sin_theta,
      "outer_preflight_composition", teuk::RadialDiscretization::D105);
  bool band_rejected_first = false;
  outer_launches = 0;
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      count_outer_launch);
  try {
    composition.evaluate_stage(
        fixture.execution, 0.0,
        teuk::Plus2PrimitiveReconstructionStage{1, {}, {}, {}, {}, {}, {}},
        teuk::Plus2TransportedCurvatureStage{},
        teuk::Plus2BianchiDerivativeStage{},
        teuk::Plus2LiveSourceCapability{
            true, true, true, true, teuk::RadialDiscretization::D105, 1},
        teuk::SourceActivationState{}, teuk::Plus2StageSourceTarget{},
        primitive, wrong_band);
  } catch (const std::invalid_argument& error) {
    band_rejected_first =
        std::string(error.what()).find("scheme/band") != std::string::npos;
  }
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  CHECK(band_rejected_first);
  CHECK(outer_launches == 0);
  bool primitive_band_rejected_first = false;
  outer_launches = 0;
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      count_outer_launch);
  try {
    composition.evaluate_stage(
        fixture.execution, 0.0,
        teuk::Plus2PrimitiveReconstructionStage{2, {}, {}, {}, {}, {}, {}},
        teuk::Plus2TransportedCurvatureStage{},
        teuk::Plus2BianchiDerivativeStage{},
        teuk::Plus2LiveSourceCapability{
            true, true, true, true, teuk::RadialDiscretization::D105, 2},
        teuk::SourceActivationState{}, teuk::Plus2StageSourceTarget{},
        wrong_primitive_band, fixture.producer);
  } catch (const std::invalid_argument& error) {
    primitive_band_rejected_first =
        std::string(error.what()).find("scheme/band") != std::string::npos;
  }
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  CHECK(primitive_band_rejected_first);
  CHECK(outer_launches == 0);
  fixture.evaluate();
  fixture.execution.fence("warm outer producer");
  outer_allocations = 0;
  outer_copies = 0;
  outer_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_outer_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(count_outer_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_outer_fence);
  fixture.evaluate();
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  fixture.execution.fence("finish audited outer producer");
  CHECK(outer_allocations == 0);
  CHECK(outer_copies == 0);
  CHECK(outer_fences == 0);

  bool stale_rejected = false;
  try {
    fixture.evaluate(outer_generation - 1, outer_generation);
  } catch (const std::invalid_argument&) {
    stale_rejected = true;
  }
  CHECK(stale_rejected);

  bool alias_rejected = false;
  try {
    fixture.producer.evaluate(
        fixture.execution, {outer_generation, fixture.sums, fixture.tangents},
        teuk::Plus2LiveOuterWriteTarget{
            outer_generation, fixture.sums, fixture.derivatives,
            fixture.projected_stamps, fixture.derivative_stamps});
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  CHECK(alias_rejected);

  bool scratch_alias_rejected = false;
  outer_launches = 0;
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      count_outer_launch);
  try {
    fixture.producer.evaluate(
        fixture.execution, {outer_generation, fixture.sums, fixture.tangents},
        teuk::Plus2LiveOuterWriteTarget{
            outer_generation, fixture.projected,
            fixture.producer.projected_tangent(), fixture.projected_stamps,
            fixture.derivative_stamps});
  } catch (const std::invalid_argument&) {
    scratch_alias_rejected = true;
  }
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  CHECK(scratch_alias_rejected);
  CHECK(outer_launches == 0);

  using unmanaged_stamp_view = Kokkos::View<
      std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  unmanaged_stamp_view overlapping_outer_stamps(
      fixture.projected_stamps.data(), fixture.registry.size(),
      static_cast<std::size_t>(teuk::Plus2SpatialOuterDerivative::Count),
      fixture.grid.size(), outer_theta_count);
  bool stamp_alias_rejected = false;
  outer_launches = 0;
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      count_outer_launch);
  try {
    fixture.producer.evaluate(
        fixture.execution, {outer_generation, fixture.sums, fixture.tangents},
        teuk::Plus2LiveOuterWriteTarget{
            outer_generation, fixture.projected, fixture.derivatives,
            fixture.projected_stamps, overlapping_outer_stamps});
  } catch (const std::invalid_argument&) {
    stamp_alias_rejected = true;
  }
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  CHECK(stamp_alias_rejected);
  CHECK(outer_launches == 0);

  auto hsums = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                    fixture.sums);
  hsums(0, 0, 0, 0) = C(std::numeric_limits<double>::quiet_NaN(), 0.0);
  Kokkos::deep_copy(fixture.execution, fixture.sums, hsums);
  fixture.evaluate();
  fixture.execution.fence("finish nonfinite outer producer");
  const auto projected = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.projected);
  const auto derivatives = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.derivatives);
  const auto projected_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.projected_stamps);
  const auto derivative_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.derivative_stamps);
  for (std::size_t i = 0; i < projected.size(); ++i) {
    CHECK_COMPLEX_NEAR(projected.data()[i], C{}, 0.0);
    CHECK(projected_stamps.data()[i] == 0);
  }
  for (std::size_t i = 0; i < derivatives.size(); ++i) {
    CHECK_COMPLEX_NEAR(derivatives.data()[i], C{}, 0.0);
    CHECK(derivative_stamps.data()[i] == 0);
  }
}

TEST_CASE("plus2 outer producer preserves amplitude and tangent linearity") {
  OuterFixture base(25, 1.0, 0.0);
  OuterFixture scaled(25, -1.7, 0.0);
  OuterFixture tangent_shifted(25, 1.0, 0.08);
  base.evaluate();
  scaled.evaluate();
  tangent_shifted.evaluate();
  base.execution.fence("finish outer scaling fixtures");
  const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      base.derivatives);
  const auto y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      scaled.derivatives);
  const auto z = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, tangent_shifted.derivatives);
  double nonzero = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    CHECK_COMPLEX_NEAR(y.data()[i], -1.7 * x.data()[i], 4e-12);
    nonzero = std::max(nonzero, Kokkos::abs(x.data()[i]));
  }
  CHECK(nonzero > 1e-4);
  bool tangent_changed_thorn = false;
  bool tangent_changed_eth = false;
  for (const int m : base.registry.targets()) {
    const std::size_t mode = base.registry.index(m);
    for (std::size_t radial = 0; radial < base.grid.size(); ++radial) {
      for (int theta = 0; theta < outer_theta_count; ++theta) {
        const auto node = static_cast<std::size_t>(theta);
        tangent_changed_thorn =
            tangent_changed_thorn ||
            Kokkos::abs(z(mode, o(teuk::Plus2SpatialOuterDerivative::Thorn5J),
                          radial, node) -
                        x(mode, o(teuk::Plus2SpatialOuterDerivative::Thorn5J),
                          radial, node)) >
                1e-10;
        tangent_changed_eth =
            tangent_changed_eth ||
            Kokkos::abs(z(mode, o(teuk::Plus2SpatialOuterDerivative::Eth6K),
                          radial, node) -
                        x(mode, o(teuk::Plus2SpatialOuterDerivative::Eth6K),
                          radial, node)) >
                1e-10;
      }
    }
  }
  CHECK(tangent_changed_thorn);
  CHECK(tangent_changed_eth);
}
