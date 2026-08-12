#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/plus2_routeb_curvature_spatial.hpp"

namespace {

using C = teuk::Complex;

struct CurvatureRun {
  std::vector<C> scri;
  std::vector<C> audit;
};

CurvatureRun run_analytic_tower(const std::size_t radial_count,
                                const double spin) {
  teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  constexpr std::size_t theta_count = 10;
  constexpr std::size_t field_count = 7;
  constexpr int ell = 4;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.8);
  const teuk::KerrParameters parameters{1.0, spin, 2.0};
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  teuk::Plus2SpatialThetaView cos_theta("routeb_analytic_cos", theta_count);
  teuk::Plus2SpatialThetaView sin_theta("routeb_analytic_sin", theta_count);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (std::size_t theta = 0; theta < theta_count; ++theta) {
    host_cos(theta) = std::cos(angular_grid.theta(theta));
    host_sin(theta) = std::sin(angular_grid.theta(theta));
  }
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  Kokkos::deep_copy(execution, sin_theta, host_sin);

  teuk::Plus2RouteBTowerView tower("routeb_analytic_tower", 5,
                                   registry.size(), field_count,
                                   radial_count, theta_count);
  teuk::Plus2RouteBTowerStampView stamps("routeb_analytic_stamps", 5,
                                         registry.size(), radial_count,
                                         theta_count);
  auto host_tower = Kokkos::create_mirror_view(tower);
  auto host_stamps = Kokkos::create_mirror_view(stamps);
  constexpr int field_spins[field_count]{0, 0, 0, -2, -1, -1, 0};
  const C lambda(0.31, -0.17);
  for (std::size_t level = 0; level < 5; ++level) {
    const C time_factor = Kokkos::pow(lambda, static_cast<int>(level));
    for (std::size_t mode_index = 0; mode_index < registry.size();
         ++mode_index) {
      const int mode = registry.modes()[mode_index];
      for (std::size_t field = 0; field < field_count; ++field) {
        for (std::size_t radial = 0; radial < radial_count; ++radial) {
          const double radius = grid.coordinate(radial);
          const double alpha = 0.23 + 0.031 * static_cast<double>(field) +
                               0.017 * static_cast<double>(mode);
          const double radial_profile =
              radius * radius * std::exp(alpha * radius) *
              (1.0 + (0.11 + 0.013 * static_cast<double>(field)) * radius +
               (0.07 - 0.004 * static_cast<double>(mode)) * radius * radius);
          const C amplitude(
              0.12 * static_cast<double>(field + 1) *
                  (mode > 0 ? 1.0 : 0.73),
              0.05 * static_cast<double>(field + 2) *
                  (mode > 0 ? -0.61 : 0.44));
          for (std::size_t theta = 0; theta < theta_count; ++theta) {
            const double harmonic =
                teuk::angular::spin_weighted_harmonic_theta(
                    ell, mode, field_spins[field],
                    angular_grid.theta(theta));
            host_tower(level, mode_index, field, radial, theta) =
                (field < 2 ? C{} : time_factor * amplitude * radial_profile *
                                             harmonic);
          }
        }
      }
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          host_stamps(level, mode_index, radial, theta) = 29;
        }
      }
    }
  }
  Kokkos::deep_copy(execution, tower, host_tower);
  Kokkos::deep_copy(execution, stamps, host_stamps);
  teuk::Plus2RouteBCurvatureSpatialProvider provider(
      execution, registry, grid, parameters, ell, cos_theta, sin_theta,
      "routeb_analytic_provider");
  provider.evaluate(execution, {29, tower, stamps});
  execution.fence();
  const auto curvature = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, provider.curvature_stage().fields);
  const auto audit = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, provider.endpoint_audit());
  CurvatureRun result;
  result.scri.reserve(registry.size() * curvature.extent(1) * theta_count);
  result.audit.reserve(registry.size() * audit.extent(1) * theta_count);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < curvature.extent(1); ++field) {
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        result.scri.push_back(curvature(mode, field, 0, theta));
      }
    }
    for (std::size_t field = 0; field < audit.extent(1); ++field) {
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        result.audit.push_back(audit(mode, field, theta));
      }
    }
  }
  return result;
}

double max_difference(const std::vector<C>& left,
                      const std::vector<C>& right) {
  CHECK(left.size() == right.size());
  double result = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) {
    result = std::max(result, Kokkos::abs(left[i] - right[i]));
  }
  return result;
}

double max_magnitude(const std::vector<C>& values) {
  double result = 0.0;
  for (const C value : values) result = std::max(result, Kokkos::abs(value));
  return result;
}

}  // namespace

TEST_CASE("Route-B constrained curvature provider preserves the zero tower") {
  teuk::ExecutionSpace execution;
  teuk::ModeRegistry registry({-2, 2}, {-2, 2}, {-2, 2});
  constexpr std::size_t radial_count = 25;
  constexpr std::size_t theta_count = 8;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.8);
  const teuk::KerrParameters parameters{1.0, 0.63, 2.0};
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  teuk::Plus2SpatialThetaView cos_theta("routeb_curvature_cos", theta_count);
  teuk::Plus2SpatialThetaView sin_theta("routeb_curvature_sin", theta_count);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (std::size_t theta = 0; theta < theta_count; ++theta) {
    host_cos(theta) = std::cos(angular_grid.theta(theta));
    host_sin(theta) = std::sin(angular_grid.theta(theta));
  }
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  Kokkos::deep_copy(execution, sin_theta, host_sin);

  teuk::Plus2RouteBTowerView tower("routeb_curvature_zero_tower", 5,
                                   registry.size(), 7, radial_count,
                                   theta_count);
  teuk::Plus2RouteBTowerStampView stamps("routeb_curvature_zero_stamps", 5,
                                         registry.size(), radial_count,
                                         theta_count);
  Kokkos::deep_copy(execution, tower, C{});
  Kokkos::deep_copy(execution, stamps, std::uint64_t{17});
  teuk::Plus2RouteBCurvatureSpatialProvider provider(
      execution, registry, grid, parameters, 4, cos_theta, sin_theta);
  provider.evaluate(execution, {17, tower, stamps});
  execution.fence();

  const auto curvature = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, provider.curvature_stage().fields);
  const auto curvature_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, provider.curvature_stage().stamps);
  const auto derivatives = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, provider.derivative_stage().fields);
  const auto derivative_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, provider.derivative_stage().stamps);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t theta = 0; theta < theta_count; ++theta) {
        for (std::size_t field = 0; field < curvature.extent(1); ++field) {
          CHECK(Kokkos::abs(curvature(mode, field, radial, theta)) < 1.0e-14);
          CHECK(curvature_stamps(mode, field, radial, theta) == 17);
        }
        for (std::size_t field = 0; field < derivatives.extent(1); ++field) {
          CHECK(Kokkos::abs(derivatives(mode, field, radial, theta)) <
                1.0e-14);
          CHECK(derivative_stamps(mode, field, radial, theta) == 17);
        }
      }
    }
  }
}

TEST_CASE("Route-B constrained curvature provider closes all six scri fields") {
  for (const double spin : {0.0, 0.63, -0.74, 0.999}) {
    const auto coarse = run_analytic_tower(9, spin);
    const auto medium = run_analytic_tower(17, spin);
    const auto fine = run_analytic_tower(33, spin);
    const auto ceiling = run_analytic_tower(65, spin);
    const double error_cm = max_difference(coarse.scri, medium.scri);
    const double error_mf = max_difference(medium.scri, fine.scri);
    const double error_fc = max_difference(fine.scri, ceiling.scri);
    CHECK(std::isfinite(error_cm));
    CHECK(std::isfinite(error_mf));
    CHECK(std::isfinite(error_fc));
    CHECK(error_cm > 15.0 * error_mf);
    CHECK(error_mf > 0.0);
    CHECK(max_magnitude(fine.scri) > 1.0e-8);
    // N=65 is recorded only as a conditioning probe, never used to promote
    // the provider or to select a friendlier window.
    std::cout << "Route-B constrained curvature spin " << spin
              << " scri changes " << error_cm << ' ' << error_mf << ' '
              << error_fc << " ratios " << error_cm / error_mf << ' '
              << error_mf / error_fc << '\n';

    const double residual_coarse = max_magnitude(coarse.audit);
    const double residual_medium = max_magnitude(medium.audit);
    const double residual_fine = max_magnitude(fine.audit);
    std::cout << "Route-B constrained peeling spin " << spin << " residuals "
              << residual_coarse << ' ' << residual_medium << ' '
              << residual_fine << '\n';
    CHECK(std::isfinite(residual_fine));
  }
}
