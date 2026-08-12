#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/plus2_routeb_curvature_spatial.hpp"
#include "plus2_routeb_curvature_coordinate_fixture.hpp"

namespace {

using C = teuk::Complex;

int routeb_allocations = 0;
int routeb_copies = 0;
int routeb_fences = 0;

void count_routeb_allocation(Kokkos::Tools::SpaceHandle, const char*,
                             const void*, std::uint64_t) {
  ++routeb_allocations;
}
void count_routeb_copy(Kokkos::Tools::SpaceHandle, const char*, const void*,
                       Kokkos::Tools::SpaceHandle, const char*, const void*,
                       std::uint64_t) {
  ++routeb_copies;
}
void count_routeb_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++routeb_fences;
}

template <class Callable>
bool throws_invalid_argument(Callable&& callable) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

struct CurvatureRun {
  std::vector<C> scri;
  std::vector<C> audit;
};

CurvatureRun run_analytic_tower(const std::size_t radial_count,
                                const double spin,
                                const double common_amplitude = 1.0,
                                const double negative_mode_amplitude = 1.0) {
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
          const double scale =
              common_amplitude * (mode < 0 ? negative_mode_amplitude : 1.0);
          for (std::size_t theta = 0; theta < theta_count; ++theta) {
            const double harmonic =
                teuk::angular::spin_weighted_harmonic_theta(
                    ell, mode, field_spins[field],
                    angular_grid.theta(theta));
            host_tower(level, mode_index, field, radial, theta) =
                (field < 2 ? C{} : scale * time_factor * amplitude * radial_profile *
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

void gate_disaggregated_scri(const CurvatureRun& coarse,
                             const CurvatureRun& medium,
                             const CurvatureRun& fine,
                             const double spin) {
  CHECK(coarse.scri.size() == medium.scri.size());
  CHECK(medium.scri.size() == fine.scri.size());
  double minimum_resolved_ratio = std::numeric_limits<double>::infinity();
  double maximum_floor_error = 0.0;
  bool resolved_green = true;
  for (std::size_t index = 0; index < fine.scri.size(); ++index) {
    const double coarse_medium =
        Kokkos::abs(coarse.scri[index] - medium.scri[index]);
    const double medium_fine =
        Kokkos::abs(medium.scri[index] - fine.scri[index]);
    CHECK(std::isfinite(coarse_medium));
    CHECK(std::isfinite(medium_fine));
    if (medium_fine > 2.0e-13) {
      if (!(coarse_medium > 15.0 * medium_fine)) {
        const std::size_t theta = index % 10;
        const std::size_t mode_field = index / 10;
        const std::size_t field = mode_field % 6;
        const std::size_t mode = mode_field / 6;
        std::cout << "Route-B red scri spin/mode/field/theta " << spin << ' '
                  << mode << ' ' << field << ' ' << theta << " errors "
                  << coarse_medium << ' ' << medium_fine << " ratio "
                  << coarse_medium / medium_fine << '\n';
      }
      minimum_resolved_ratio =
          std::min(minimum_resolved_ratio, coarse_medium / medium_fine);
      resolved_green =
          resolved_green && coarse_medium > 15.0 * medium_fine;
    } else {
      maximum_floor_error = std::max(maximum_floor_error, medium_fine);
      CHECK(coarse_medium < 2.0e-11);
    }
  }
  std::cout << "Route-B disaggregated scri spin " << spin
            << " minimum resolved ratio " << minimum_resolved_ratio
            << " maximum floor error " << maximum_floor_error << '\n';
  CHECK(resolved_green);
}

void gate_disaggregated_peeling(const CurvatureRun& coarse,
                                const CurvatureRun& medium,
                                const CurvatureRun& fine,
                                const double spin) {
  CHECK(coarse.audit.size() == medium.audit.size());
  CHECK(medium.audit.size() == fine.audit.size());
  double minimum_ratio = std::numeric_limits<double>::infinity();
  double maximum_fine = 0.0;
  bool residual_green = true;
  for (std::size_t index = 0; index < fine.audit.size(); ++index) {
    const double coarse_value = Kokkos::abs(coarse.audit[index]);
    const double medium_value = Kokkos::abs(medium.audit[index]);
    const double fine_value = Kokkos::abs(fine.audit[index]);
    CHECK(std::isfinite(coarse_value));
    CHECK(std::isfinite(medium_value));
    CHECK(std::isfinite(fine_value));
    maximum_fine = std::max(maximum_fine, fine_value);
    if (medium_value > 2.0e-14) {
      if (!(coarse_value > 15.0 * medium_value) ||
          !(medium_value > 15.0 * fine_value)) {
        const std::size_t theta = index % 10;
        const std::size_t mode_field = index / 10;
        const std::size_t field = mode_field % 9;
        const std::size_t mode = mode_field / 9;
        std::cout << "Route-B red peeling spin/mode/field/theta " << spin
                  << ' ' << mode << ' ' << field << ' ' << theta
                  << " residuals " << coarse_value << ' ' << medium_value
                  << ' ' << fine_value << " ratios "
                  << coarse_value / medium_value << ' '
                  << medium_value / fine_value << '\n';
      }
      minimum_ratio = std::min(minimum_ratio, coarse_value / medium_value);
      residual_green = residual_green &&
                       coarse_value > 15.0 * medium_value &&
                       medium_value > 15.0 * fine_value;
    } else {
      CHECK(fine_value < 2.0e-13);
    }
  }
  CHECK(maximum_fine < 1.0e-10);
  CHECK(residual_green);
  std::cout << "Route-B disaggregated peeling spin " << spin
            << " minimum ratio " << minimum_ratio << " maximum fine "
            << maximum_fine << '\n';
}

struct CoordinateRun {
  Kokkos::View<const C****, Kokkos::LayoutRight, Kokkos::HostSpace>
      curvature;
  Kokkos::View<const C****, Kokkos::LayoutRight, Kokkos::HostSpace>
      derivatives;
};

CoordinateRun run_coordinate_tower(const int spin_index,
                                   const int case_index,
                                   const std::size_t radial_count,
                                   const int provider_ell_max,
                                   const int radial_power = 2) {
  namespace fixture = plus2_routeb_curvature_coordinate_fixture;
  teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0,
                                     fixture::radial_maxes[spin_index]);
  const teuk::KerrParameters parameters{1.0, fixture::spins[spin_index], 2.0};
  const auto angular_grid = teuk::angular::gauss_legendre(fixture::theta_count);
  teuk::Plus2SpatialThetaView cos_theta("routeb_coordinate_cos",
                                        fixture::theta_count);
  teuk::Plus2SpatialThetaView sin_theta("routeb_coordinate_sin",
                                        fixture::theta_count);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (int theta = 0; theta < fixture::theta_count; ++theta) {
    host_cos(theta) = std::cos(angular_grid.theta(theta));
    host_sin(theta) = std::sin(angular_grid.theta(theta));
  }
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  Kokkos::deep_copy(execution, sin_theta, host_sin);

  constexpr std::size_t field_count = 7;
  teuk::Plus2RouteBTowerView tower(
      "routeb_coordinate_tower", 5, registry.size(), field_count,
      radial_count, fixture::theta_count);
  teuk::Plus2RouteBTowerStampView stamps(
      "routeb_coordinate_stamps", 5, registry.size(), radial_count,
      fixture::theta_count);
  auto host_tower = Kokkos::create_mirror_view(tower);
  auto host_stamps = Kokkos::create_mirror_view(stamps);
  for (std::size_t level = 0; level < 5; ++level) {
    for (std::size_t mode_index = 0; mode_index < registry.size();
         ++mode_index) {
      const int mode = registry.modes()[mode_index];
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        const double radius = grid.coordinate(radial);
        for (int theta = 0; theta < fixture::theta_count; ++theta) {
          for (std::size_t field = 0; field < field_count; ++field) {
            host_tower(level, mode_index, field, radial, theta) = C{};
          }
          for (std::size_t kind = 0; kind < 3; ++kind) {
            if (case_index != 3 && case_index != static_cast<int>(kind)) {
              continue;
            }
            const std::complex<double> amplitude(
                fixture::amplitudes[kind][mode_index].real(),
                fixture::amplitudes[kind][mode_index].imag());
            const std::complex<double> lambda(
                fixture::lambdas[kind][mode_index].real(),
                fixture::lambdas[kind][mode_index].imag());
            const double radial_profile =
                std::pow(radius, radial_power) *
                std::exp(fixture::alphas[kind] * radius) *
                (1.0 + fixture::linear[kind] * radius +
                 fixture::quadratic[kind] * radius * radius);
            const double harmonic =
                teuk::angular::spin_weighted_harmonic_theta(
                    fixture::ell, mode, fixture::field_spins[kind],
                    angular_grid.theta(theta));
            const std::complex<double> value =
                amplitude * std::exp(lambda * fixture::time) *
                std::pow(lambda, static_cast<int>(level)) * radial_profile *
                harmonic;
            const C coefficient(value.real(), value.imag());
            if (kind == 0) {
              const auto background = teuk::kerr_background_point(
                  parameters, radius, host_cos(theta), host_sin(theta));
              host_tower(level, mode_index, 6, radial, theta) =
                  background.mu0 * coefficient;
            } else if (kind == 1) {
              host_tower(level, mode_index, 5, radial, theta) = coefficient;
            } else {
              host_tower(level, mode_index, 3, radial, theta) = coefficient;
            }
          }
          host_stamps(level, mode_index, radial, theta) = 47;
        }
      }
    }
  }
  Kokkos::deep_copy(execution, tower, host_tower);
  Kokkos::deep_copy(execution, stamps, host_stamps);
  teuk::Plus2RouteBCurvatureSpatialProvider provider(
      execution, registry, grid, parameters, provider_ell_max, cos_theta,
      sin_theta, "routeb_coordinate_provider");
  provider.evaluate(execution, {47, tower, stamps});
  execution.fence();
  return {
      Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, provider.curvature_stage().fields),
      Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, provider.derivative_stage().fields)};
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
    gate_disaggregated_scri(coarse, medium, fine, spin);
    gate_disaggregated_peeling(coarse, medium, fine, spin);
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

TEST_CASE("Route-B curvature provider matches independent coordinate Weyl") {
  namespace fixture = plus2_routeb_curvature_coordinate_fixture;
  double maximum_z0_error = 0.0;
  double maximum_z1_error = 0.0;
  const fixture::Expected* worst_z0 = nullptr;
  const fixture::Expected* worst_z1 = nullptr;
  C worst_z0_actual{};
  C worst_z1_actual{};
  std::array<double, 4> case_z0_error{};
  std::array<double, 4> case_z1_error{};
  double minimum_horizon_ratio = std::numeric_limits<double>::infinity();
  double maximum_horizon_fine = 0.0;
  std::size_t resolved_horizon_cells = 0;
  std::size_t horizon_floor_cells = 0;
  double maximum_derivative_error = 0.0;
  double maximum_derivative_oracle_remainder = 0.0;
  const fixture::DerivativeExpected* worst_derivative = nullptr;
  std::size_t worst_derivative_field = 0;
  C worst_derivative_actual{};
  std::array<double, 8> derivative_field_error{};
  std::array<double, 8> derivative_radial_cm{};
  std::array<double, 8> derivative_radial_mf{};
  std::array<double, 8> derivative_angular_12_18{};
  std::array<double, 8> derivative_angular_18_24{};
  std::array<double, 8> derivative_wrong_separation{};
  double minimum_derivative_radial_ratio =
      std::numeric_limits<double>::infinity();
  double minimum_derivative_angular_ratio =
      std::numeric_limits<double>::infinity();
  std::size_t resolved_derivative_radial_cells = 0;
  std::size_t resolved_derivative_angular_cells = 0;
  double maximum_coordinate_scri_error = 0.0;
  double minimum_coordinate_scri_ratio =
      std::numeric_limits<double>::infinity();
  double minimum_provider_scri_ratio =
      std::numeric_limits<double>::infinity();
  const fixture::ScriExpected* worst_coordinate_scri = nullptr;
  std::size_t worst_coordinate_scri_field = 0;
  C worst_coordinate_scri_actual{};
  std::size_t resolved_provider_scri_cells = 0;
  double maximum_coordinate_scri_signal = 0.0;
  for (int case_index = 0; case_index < 4; ++case_index) {
    for (int spin_index = 0; spin_index < 3; ++spin_index) {
      const std::array<CoordinateRun, 3> runs{
          run_coordinate_tower(spin_index, case_index, 9,
                               fixture::provider_ell_max),
          run_coordinate_tower(spin_index, case_index, 17,
                               fixture::provider_ell_max),
          run_coordinate_tower(spin_index, case_index, 33,
                               fixture::provider_ell_max)};
      const CoordinateRun angular12 =
          run_coordinate_tower(spin_index, case_index, 33, 12);
      const CoordinateRun angular18 =
          run_coordinate_tower(spin_index, case_index, 33, 18);
      for (const auto& expected : fixture::expected) {
        if (expected.case_index != case_index ||
            expected.spin_index != spin_index) {
          continue;
        }
      const std::size_t z0_field = 2 * expected.level;
      const std::size_t z1_field = z0_field + 1;
      const C actual_z0 = runs[2].curvature(
          expected.mode_index, z0_field, expected.radial_index,
          expected.theta_index);
      const C actual_z1 = runs[2].curvature(
          expected.mode_index, z1_field, expected.radial_index,
          expected.theta_index);
      const double z0_error = Kokkos::abs(actual_z0 - expected.z0);
      const double z1_error = Kokkos::abs(actual_z1 - expected.z1);
      if (expected.radial_index == fixture::radial_count - 1) {
        const std::size_t coarse_radial = 8;
        const std::size_t medium_radial = 16;
        const C coarse_values[2]{
            runs[0].curvature(expected.mode_index, z0_field, coarse_radial,
                              expected.theta_index),
            runs[0].curvature(expected.mode_index, z1_field, coarse_radial,
                              expected.theta_index)};
        const C medium_values[2]{
            runs[1].curvature(expected.mode_index, z0_field, medium_radial,
                              expected.theta_index),
            runs[1].curvature(expected.mode_index, z1_field, medium_radial,
                              expected.theta_index)};
        const C fine_values[2]{actual_z0, actual_z1};
        const C angular12_values[2]{
            angular12.curvature(expected.mode_index, z0_field,
                                expected.radial_index,
                                expected.theta_index),
            angular12.curvature(expected.mode_index, z1_field,
                                expected.radial_index,
                                expected.theta_index)};
        const C angular18_values[2]{
            angular18.curvature(expected.mode_index, z0_field,
                                expected.radial_index,
                                expected.theta_index),
            angular18.curvature(expected.mode_index, z1_field,
                                expected.radial_index,
                                expected.theta_index)};
        const C exact_values[2]{expected.z0, expected.z1};
        for (std::size_t field = 0; field < 2; ++field) {
          const double coarse_medium =
              Kokkos::abs(coarse_values[field] - medium_values[field]);
          const double medium_fine =
              Kokkos::abs(medium_values[field] - fine_values[field]);
          const double fine_error =
              Kokkos::abs(fine_values[field] - exact_values[field]);
          const double angular_12_18 = Kokkos::abs(
              angular12_values[field] - angular18_values[field]);
          const double angular_18_24 =
              Kokkos::abs(angular18_values[field] - fine_values[field]);
          CHECK(std::isfinite(coarse_medium));
          CHECK(std::isfinite(medium_fine));
          CHECK(std::isfinite(fine_error));
          CHECK(std::isfinite(angular_12_18));
          CHECK(std::isfinite(angular_18_24));
          maximum_horizon_fine = std::max(maximum_horizon_fine, fine_error);
          if (medium_fine > 5.0e-11) {
            if (!(coarse_medium > 15.0 * medium_fine)) {
              std::cout
                  << "Route-B coordinate horizon red case/spin/mode/level/"
                     "theta/field/errors "
                  << case_index << ' ' << spin_index << ' '
                  << expected.mode_index << ' ' << expected.level << ' '
                  << expected.theta_index << ' ' << field << ' '
                  << coarse_medium << ' ' << medium_fine << ' ' << fine_error
                  << " radial ratio " << coarse_medium / medium_fine
                  << " angular increments " << angular_12_18 << ' '
                  << angular_18_24 << '\n';
            }
            CHECK(coarse_medium > 15.0 * medium_fine);
            minimum_horizon_ratio =
                std::min(minimum_horizon_ratio,
                         coarse_medium / medium_fine);
            ++resolved_horizon_cells;
          } else {
            ++horizon_floor_cells;
          }
          if (angular_18_24 > 5.0e-11) {
            if (!(angular_12_18 > 2.0 * angular_18_24)) {
              std::cout
                  << "Route-B coordinate angular red case/spin/mode/level/"
                     "theta/field/increments "
                  << case_index << ' ' << spin_index << ' '
                  << expected.mode_index << ' ' << expected.level << ' '
                  << expected.theta_index << ' ' << field << ' '
                  << angular_12_18 << ' ' << angular_18_24 << '\n';
            }
            CHECK(angular_12_18 > 2.0 * angular_18_24);
          } else {
            CHECK(angular_18_24 < 5.0e-11);
          }
          CHECK(fine_error < medium_fine / 15.0 +
                                 2.0 * angular_18_24 + 5.0e-11);
        }
        continue;
      }
      case_z0_error[case_index] =
          std::max(case_z0_error[case_index], z0_error);
      case_z1_error[case_index] =
          std::max(case_z1_error[case_index], z1_error);
      if (z0_error > maximum_z0_error) {
        maximum_z0_error = z0_error;
        worst_z0 = &expected;
        worst_z0_actual = actual_z0;
      }
      if (z1_error > maximum_z1_error) {
        maximum_z1_error = z1_error;
        worst_z1 = &expected;
        worst_z1_actual = actual_z1;
      }
      }
      for (const auto& expected : fixture::derivative_expected) {
        if (expected.case_index != case_index ||
            expected.spin_index != spin_index) {
          continue;
        }
        maximum_derivative_oracle_remainder =
            std::max(maximum_derivative_oracle_remainder,
                     expected.finite_difference_remainder);
        for (std::size_t field = 0; field < expected.values.size(); ++field) {
          const C actual = runs[2].derivatives(
              expected.mode_index, field, expected.radial_index,
              expected.theta_index);
          const double error = Kokkos::abs(actual - expected.values[field]);
          const std::size_t coarse_radial = expected.radial_index / 4;
          const std::size_t medium_radial = expected.radial_index / 2;
          const double radial_cm = Kokkos::abs(
              runs[0].derivatives(expected.mode_index, field, coarse_radial,
                                  expected.theta_index) -
              runs[1].derivatives(expected.mode_index, field, medium_radial,
                                  expected.theta_index));
          const double radial_mf = Kokkos::abs(
              runs[1].derivatives(expected.mode_index, field, medium_radial,
                                  expected.theta_index) -
              actual);
          const double angular_12_18 = Kokkos::abs(
              angular12.derivatives(expected.mode_index, field,
                                    expected.radial_index,
                                    expected.theta_index) -
              angular18.derivatives(expected.mode_index, field,
                                    expected.radial_index,
                                    expected.theta_index));
          const double angular_18_24 = Kokkos::abs(
              angular18.derivatives(expected.mode_index, field,
                                    expected.radial_index,
                                    expected.theta_index) -
              actual);
          CHECK(std::isfinite(error));
          CHECK(std::isfinite(radial_cm));
          CHECK(std::isfinite(radial_mf));
          CHECK(std::isfinite(angular_12_18));
          CHECK(std::isfinite(angular_18_24));
          if (radial_mf > 5.0e-10) {
            if (!(radial_cm > 15.0 * radial_mf)) {
              std::cout << "Route-B coordinate derivative radial red "
                           "spin/r/theta/mode/field/cm/mf "
                        << spin_index << ' ' << expected.radial_index << ' '
                        << expected.theta_index << ' '
                        << expected.mode_index << ' ' << field << ' '
                        << radial_cm << ' ' << radial_mf << '\n';
            }
            CHECK(radial_cm > 15.0 * radial_mf);
            minimum_derivative_radial_ratio =
                std::min(minimum_derivative_radial_ratio,
                         radial_cm / radial_mf);
            ++resolved_derivative_radial_cells;
          }
          if (angular_18_24 > 5.0e-10) {
            if (!(angular_12_18 > 2.0 * angular_18_24)) {
              std::cout << "Route-B coordinate derivative angular red "
                           "spin/r/theta/mode/field/12-18/18-24 "
                        << spin_index << ' ' << expected.radial_index << ' '
                        << expected.theta_index << ' '
                        << expected.mode_index << ' ' << field << ' '
                        << angular_12_18 << ' ' << angular_18_24 << '\n';
            }
            CHECK(angular_12_18 > 2.0 * angular_18_24);
            minimum_derivative_angular_ratio =
                std::min(minimum_derivative_angular_ratio,
                         angular_12_18 / angular_18_24);
            ++resolved_derivative_angular_cells;
          }
          const double continuum_budget =
              radial_mf / 15.0 + 2.0 * angular_18_24 +
              10.0 * expected.finite_difference_remainder + 5.0e-10;
          if (!(error < continuum_budget)) {
            std::cout << "Route-B coordinate derivative continuum red "
                         "spin/r/theta/mode/field/error/budget "
                      << spin_index << ' ' << expected.radial_index << ' '
                      << expected.theta_index << ' ' << expected.mode_index
                      << ' ' << field << ' ' << error << ' '
                      << continuum_budget << '\n';
          }
          CHECK(error < continuum_budget);
          derivative_field_error[field] =
              std::max(derivative_field_error[field], error);
          derivative_radial_cm[field] =
              std::max(derivative_radial_cm[field], radial_cm);
          derivative_radial_mf[field] =
              std::max(derivative_radial_mf[field], radial_mf);
          derivative_angular_12_18[field] =
              std::max(derivative_angular_12_18[field], angular_12_18);
          derivative_angular_18_24[field] =
              std::max(derivative_angular_18_24[field], angular_18_24);
          derivative_wrong_separation[field] = std::max(
              derivative_wrong_separation[field],
              Kokkos::abs(expected.values[field] -
                          expected.wrong_values[field]));
          if (error > maximum_derivative_error) {
            maximum_derivative_error = error;
            worst_derivative = &expected;
            worst_derivative_field = field;
            worst_derivative_actual = actual;
          }
        }
      }
    }
  }
  for (const int case_index : {3}) {
    for (int spin_index = 0; spin_index < 3; ++spin_index) {
      const std::array<CoordinateRun, 3> runs{
          run_coordinate_tower(spin_index, case_index, 9,
                               fixture::provider_ell_max),
          run_coordinate_tower(spin_index, case_index, 17,
                               fixture::provider_ell_max),
          run_coordinate_tower(spin_index, case_index, 33,
                               fixture::provider_ell_max)};
      const CoordinateRun angular12 =
          run_coordinate_tower(spin_index, case_index, 33, 12);
      const CoordinateRun angular18 =
          run_coordinate_tower(spin_index, case_index, 33, 18);
      for (const auto& expected : fixture::scri_expected) {
        if (expected.case_index != case_index ||
            expected.spin_index != spin_index) {
          continue;
        }
        for (std::size_t field = 0; field < expected.values.size(); ++field) {
          const std::size_t curvature_field = 2 * expected.level + field;
          const C coarse = runs[0].curvature(
              expected.mode_index, curvature_field, 0, expected.theta_index);
          const C medium = runs[1].curvature(
              expected.mode_index, curvature_field, 0, expected.theta_index);
          const C fine = runs[2].curvature(
              expected.mode_index, curvature_field, 0, expected.theta_index);
          const C angular12_value = angular12.curvature(
              expected.mode_index, curvature_field, 0, expected.theta_index);
          const C angular18_value = angular18.curvature(
              expected.mode_index, curvature_field, 0, expected.theta_index);
          const double provider_cm = Kokkos::abs(coarse - medium);
          const double provider_mf = Kokkos::abs(medium - fine);
          const double angular_12_18 =
              Kokkos::abs(angular12_value - angular18_value);
          const double angular_18_24 = Kokkos::abs(angular18_value - fine);
          const C provider_extrapolated = fine + (fine - medium) / 15.0;
          const double error =
              Kokkos::abs(provider_extrapolated - expected.values[field]);
          maximum_coordinate_scri_signal =
              std::max(maximum_coordinate_scri_signal,
                       Kokkos::abs(expected.values[field]));
          CHECK(std::isfinite(error));
          CHECK(expected.coarse_medium[field] >
                15.0 * expected.medium_fine[field]);
          minimum_coordinate_scri_ratio =
              std::min(minimum_coordinate_scri_ratio,
                       expected.coarse_medium[field] /
                           expected.medium_fine[field]);
          if (provider_mf > 5.0e-10) {
            if (!(provider_cm > 15.0 * provider_mf)) {
              std::cout << "Route-B coordinate scri provider red "
                           "spin/theta/mode/level/field/cm/mf "
                        << spin_index << ' ' << expected.theta_index << ' '
                        << expected.mode_index << ' ' << expected.level << ' '
                        << field << ' ' << provider_cm << ' ' << provider_mf
                        << '\n';
            }
            CHECK(provider_cm > 15.0 * provider_mf);
            minimum_provider_scri_ratio =
                std::min(minimum_provider_scri_ratio,
                         provider_cm / provider_mf);
            ++resolved_provider_scri_cells;
          }
          if (angular_18_24 > 5.0e-10) {
            CHECK(angular_12_18 > 2.0 * angular_18_24);
          }
          const double continuum_budget =
              expected.medium_fine[field] / 15.0 + provider_mf / 15.0 +
              2.0 * angular_18_24 + 5.0e-9;
          if (!(error < continuum_budget)) {
            std::cout << "Route-B coordinate scri continuum red "
                         "spin/theta/mode/level/field/error/budget "
                      << spin_index << ' ' << expected.theta_index << ' '
                      << expected.mode_index << ' ' << expected.level << ' '
                      << field << ' ' << error << ' ' << continuum_budget
                      << '\n';
          }
          CHECK(error < continuum_budget);
          if (error > maximum_coordinate_scri_error) {
            maximum_coordinate_scri_error = error;
            worst_coordinate_scri = &expected;
            worst_coordinate_scri_field = field;
            worst_coordinate_scri_actual = provider_extrapolated;
          }
        }
      }
    }
  }
  std::cout << "Route-B coordinate-Weyl maximum Z0/Z1 errors "
            << maximum_z0_error << ' ' << maximum_z1_error << '\n';
  std::cout << "Route-B coordinate-Weyl horizon minimum ratio/fine "
            << minimum_horizon_ratio << ' ' << maximum_horizon_fine
            << " resolved/floor cells " << resolved_horizon_cells << ' '
            << horizon_floor_cells << '\n';
  std::cout << "Route-B coordinate-Weyl derivative maximum error/oracle "
               "remainder "
            << maximum_derivative_error << ' '
            << maximum_derivative_oracle_remainder << '\n';
  std::cout << "Route-B coordinate-Weyl derivative minimum radial/angular "
               "ratios and resolved cells "
            << minimum_derivative_radial_ratio << ' '
            << minimum_derivative_angular_ratio << ' '
            << resolved_derivative_radial_cells << ' '
            << resolved_derivative_angular_cells << '\n';
  for (std::size_t field = 0; field < derivative_field_error.size(); ++field) {
    std::cout << "Route-B coordinate-Weyl derivative field " << field
              << " error/radial cm/mf/angular 12-18/18-24 "
              << derivative_field_error[field] << ' '
              << derivative_radial_cm[field] << ' '
              << derivative_radial_mf[field] << ' '
              << derivative_angular_12_18[field] << ' '
              << derivative_angular_18_24[field] << " wrong separation "
              << derivative_wrong_separation[field] << '\n';
    CHECK(derivative_wrong_separation[field] > 1.0e-6);
  }
  std::cout << "Route-B coordinate-Weyl scri maximum error and minimum "
               "coordinate/provider ratios "
            << maximum_coordinate_scri_error << ' '
            << minimum_coordinate_scri_ratio << ' '
            << minimum_provider_scri_ratio << '\n';
  for (int case_index = 0; case_index < 4; ++case_index) {
    std::cout << "Route-B coordinate-Weyl case " << case_index
              << " Z0/Z1 errors " << case_z0_error[case_index] << ' '
              << case_z1_error[case_index] << '\n';
  }
  if (worst_z0 != nullptr) {
    std::cout << "Route-B worst Z0 fixture spin/r/theta/mode/level "
              << worst_z0->spin_index << ' ' << worst_z0->radial_index << ' '
              << worst_z0->theta_index << ' ' << worst_z0->mode_index << ' '
              << worst_z0->level << " actual " << worst_z0_actual
              << " expected " << worst_z0->z0 << '\n';
  }
  if (worst_z1 != nullptr) {
    std::cout << "Route-B worst Z1 fixture spin/r/theta/mode/level "
              << worst_z1->spin_index << ' ' << worst_z1->radial_index << ' '
              << worst_z1->theta_index << ' ' << worst_z1->mode_index << ' '
              << worst_z1->level << " actual " << worst_z1_actual
              << " expected " << worst_z1->z1 << '\n';
  }
  if (worst_derivative != nullptr) {
    std::cout << "Route-B worst derivative fixture spin/r/theta/mode/field "
              << worst_derivative->spin_index << ' '
              << worst_derivative->radial_index << ' '
              << worst_derivative->theta_index << ' '
              << worst_derivative->mode_index << ' '
              << worst_derivative_field << " actual "
              << worst_derivative_actual << " expected "
              << worst_derivative->values[worst_derivative_field] << '\n';
  }
  if (worst_coordinate_scri != nullptr) {
    std::cout << "Route-B worst coordinate scri spin/theta/mode/level/field "
              << worst_coordinate_scri->spin_index << ' '
              << worst_coordinate_scri->theta_index << ' '
              << worst_coordinate_scri->mode_index << ' '
              << worst_coordinate_scri->level << ' '
              << worst_coordinate_scri_field << " actual "
              << worst_coordinate_scri_actual << " expected "
              << worst_coordinate_scri->values[worst_coordinate_scri_field]
              << '\n';
  }
  CHECK(maximum_z0_error < 5.0e-11);
  CHECK(maximum_z1_error < 5.0e-11);
  CHECK(resolved_horizon_cells > 0);
  CHECK(minimum_horizon_ratio > 15.0);
  CHECK(resolved_derivative_radial_cells > 0);
  CHECK(resolved_derivative_angular_cells > 0);
  CHECK(maximum_derivative_error < 1.0e-5);
  CHECK(resolved_provider_scri_cells > 0);
  CHECK(maximum_coordinate_scri_signal > 1.0e-4);
}

TEST_CASE("Route-B curvature provider fails closed and stays hot") {
  teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  constexpr std::size_t radial_count = 25;
  constexpr std::size_t theta_count = 8;
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.8);
  const teuk::KerrParameters parameters{1.0, 0.63, 2.0};
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  teuk::Plus2SpatialThetaView cos_theta("routeb_contract_cos", theta_count);
  teuk::Plus2SpatialThetaView sin_theta("routeb_contract_sin", theta_count);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  for (std::size_t theta = 0; theta < theta_count; ++theta) {
    host_cos(theta) = std::cos(angular_grid.theta(theta));
    host_sin(theta) = std::sin(angular_grid.theta(theta));
  }
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  Kokkos::deep_copy(execution, sin_theta, host_sin);
  teuk::Plus2RouteBTowerView tower("routeb_contract_tower", 5,
                                   registry.size(), 7, radial_count,
                                   theta_count);
  teuk::Plus2RouteBTowerStampView stamps("routeb_contract_stamps", 5,
                                         registry.size(), radial_count,
                                         theta_count);
  Kokkos::deep_copy(execution, tower, C{});
  Kokkos::deep_copy(execution, stamps, std::uint64_t{1});
  teuk::Plus2RouteBCurvatureSpatialProvider provider(
      execution, registry, grid, parameters, 4, cos_theta, sin_theta,
      "routeb_contract_provider");
  provider.evaluate(execution, {1, tower, stamps});
  execution.fence("finish Route-B valid contract stage");

  const auto check_zero_invalid = [&]() {
    const auto curvature = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, provider.curvature_stage().fields);
    const auto curvature_stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, provider.curvature_stage().stamps);
    const auto derivatives = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, provider.derivative_stage().fields);
    const auto derivative_stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, provider.derivative_stage().stamps);
    const auto audit = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, provider.endpoint_audit());
    const auto audit_stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, provider.endpoint_audit_stamps());
    const auto ready = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, provider.readiness());
    CHECK(ready(0) == 0);
    for (std::size_t i = 0; i < curvature.size(); ++i) {
      CHECK(Kokkos::abs(curvature.data()[i]) == 0.0);
      CHECK(curvature_stamps.data()[i] == 0);
    }
    for (std::size_t i = 0; i < derivatives.size(); ++i) {
      CHECK(Kokkos::abs(derivatives.data()[i]) == 0.0);
      CHECK(derivative_stamps.data()[i] == 0);
    }
    for (std::size_t i = 0; i < audit.size(); ++i) {
      CHECK(Kokkos::abs(audit.data()[i]) == 0.0);
      CHECK(audit_stamps.data()[i] == 0);
    }
  };

  Kokkos::deep_copy(execution, stamps, std::uint64_t{2});
  auto stale = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                    stamps);
  stale(4, 1, radial_count - 1, theta_count - 1) = 1;
  Kokkos::deep_copy(execution, stamps, stale);
  provider.evaluate(execution, {2, tower, stamps});
  execution.fence("finish Route-B stale contract stage");
  check_zero_invalid();

  Kokkos::deep_copy(execution, stamps, std::uint64_t{3});
  Kokkos::deep_copy(execution, tower, C{});
  auto nonfinite = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       tower);
  nonfinite(2, 0, 3, 7, 4) = C(std::numeric_limits<double>::quiet_NaN(), 0.0);
  Kokkos::deep_copy(execution, tower, nonfinite);
  provider.evaluate(execution, {3, tower, stamps});
  execution.fence("finish Route-B nonfinite contract stage");
  check_zero_invalid();

  CHECK(throws_invalid_argument(
      [&] { provider.evaluate(execution, {3, tower, stamps}); }));
  CHECK(throws_invalid_argument([&] {
    provider.evaluate(execution, {4, tower, stamps}, {2, 3, 3, 5, 6});
  }));
  teuk::Plus2RouteBTowerView wrong_shape(
      "routeb_wrong_shape", 4, registry.size(), 7, radial_count, theta_count);
  CHECK(throws_invalid_argument(
      [&] { provider.evaluate(execution, {4, wrong_shape, stamps}); }));

  Kokkos::deep_copy(execution, tower, C{});
  Kokkos::deep_copy(execution, stamps, std::uint64_t{4});
  execution.fence("begin Route-B hot-stage instrumentation");
  routeb_allocations = 0;
  routeb_copies = 0;
  routeb_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_routeb_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(count_routeb_copy);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_routeb_fence);
  provider.evaluate(execution, {4, tower, stamps});
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  execution.fence("finish Route-B hot-stage instrumentation");
  CHECK(routeb_allocations == 0);
  CHECK(routeb_copies == 0);
  CHECK(routeb_fences == 0);
}

TEST_CASE("Route-B curvature provider is linear and sharp paired") {
  const auto base = run_analytic_tower(33, 0.63);
  // The complete provider is linear.  This is checked pointwise over all six
  // fields, both signed modes, and all theta nodes at scri.
  const double amplitude = -1.7;
  const auto scaled = run_analytic_tower(33, 0.63, amplitude);
  for (std::size_t index = 0; index < base.scri.size(); ++index) {
    CHECK(Kokkos::abs(scaled.scri[index] - amplitude * base.scri[index]) <
          3.0e-12);
  }
  // Alter only the negative-mode tower.  Positive-mode curvature must change
  // because Bsharp/Csharp/Usharp are conjugate values from the negative mode;
  // same-m conjugation would make this hostile perturbation invisible.
  const auto changed_negative = run_analytic_tower(33, 0.63, 1.0, 1.41);
  const std::size_t one_mode = 6 * 10;
  double positive_mode_change = 0.0;
  for (std::size_t index = 0; index < one_mode; ++index) {
    positive_mode_change =
        std::max(positive_mode_change,
                 Kokkos::abs(base.scri[one_mode + index] -
                             changed_negative.scri[one_mode + index]));
  }
  CHECK(positive_mode_change > 1.0e-5);
}
