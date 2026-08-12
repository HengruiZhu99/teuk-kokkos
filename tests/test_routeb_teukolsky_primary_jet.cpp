#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/angular_coordinator.hpp"
#include "teuk/boundary.hpp"
#include "teuk/full_spatial.hpp"
#include "teuk/modes.hpp"
#include "teuk/routeb_teukolsky_primary_jet.hpp"
#include "routeb_teukolsky_primary_fixture.hpp"

namespace {

using C = teuk::Complex;
using LC = std::complex<long double>;

int routeb_primary_allocations = 0;
int routeb_primary_fences = 0;

void count_routeb_primary_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                     const void*, std::uint64_t) {
  ++routeb_primary_allocations;
}

void count_routeb_primary_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++routeb_primary_fences;
}

template <class Function>
bool throws_invalid_argument(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

template <class Function>
bool throws_logic_error(Function&& function) {
  try {
    function();
  } catch (const std::logic_error&) {
    return true;
  }
  return false;
}

template <std::size_t Size>
LC polynomial_taylor_coefficient(const std::array<LC, Size>& coefficients,
                                 const long double radius,
                                 const std::size_t order) {
  LC result{};
  for (std::size_t power = order; power < Size; ++power) {
    long double choose = 1.0L;
    for (std::size_t factor = 1; factor <= order; ++factor) {
      choose *= static_cast<long double>(power + 1 - factor) /
                static_cast<long double>(factor);
    }
    result += coefficients[power] * choose *
              std::pow(radius, static_cast<int>(power - order));
  }
  return result;
}

struct TowerResult {
  std::vector<C> endpoints;
  std::vector<std::uint64_t> stamps;
  double old_angular_recovery_difference = 0.0;
};

C fixture_initial_value(const std::size_t field, const double radius) {
  if (field == 0) {
    return C(0.47, -0.18) * std::exp(1.31 * radius) +
           C(0.09, 0.04) * std::sin(2.17 * radius);
  }
  if (field == 1) {
    return C(-0.36, 0.27) * std::exp(1.73 * radius) +
           C(0.07, -0.05) * std::cos(2.43 * radius);
  }
  return C(0.62, 0.11) * std::exp(1.11 * radius) +
         C(-0.08, 0.06) * std::sin(2.71 * radius);
}

C fixture_angular_coefficient(const std::size_t level,
                              const std::size_t order,
                              const double radius) {
  const C exponential_amplitude(0.13 * (level + 1),
                                0.04 * (level + 2));
  const C sinusoidal_amplitude = C(0.03, -0.02) * double(level + 1);
  const double alpha = 1.19 + 0.23 * level;
  const double beta = 2.29 + 0.17 * level;
  const double pi = std::acos(-1.0);
  return (exponential_amplitude * std::pow(alpha, int(order)) *
              std::exp(alpha * radius) +
          sinusoidal_amplitude * std::pow(beta, int(order)) *
              std::sin(beta * radius + 0.5 * pi * double(order))) /
         teuk::routeb_factorial(order);
}

TowerResult run_independent_fixture(const std::size_t radial_count,
                                    const double amplitude = 1.0) {
  constexpr std::uint64_t generation = 2718;
  const teuk::ExecutionSpace execution;
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.reduction_damping = 0.31;
  const teuk::UniformRadialGrid grid(
      radial_count, 0.0,
      teuk::compactified_outer_horizon_radius(parameters));
  Kokkos::View<int*, teuk::MemorySpace> modes("routeb_fixture_modes", 1);
  Kokkos::View<double*, teuk::MemorySpace> theta("routeb_fixture_theta", 1);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> input(
      "routeb_fixture_input", 1, 3, radial_count, 1);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      input_stamps("routeb_fixture_input_stamps", 1, radial_count, 1);
  auto host_modes = Kokkos::create_mirror_view(modes);
  auto host_theta = Kokkos::create_mirror_view(theta);
  auto host_input = Kokkos::create_mirror_view(input);
  host_modes(0) = -2;
  host_theta(0) = 0.82;
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    const double radius = grid.coordinate(radial);
    for (std::size_t field = 0; field < 3; ++field) {
      host_input(0, field, radial, 0) =
          amplitude * fixture_initial_value(field, radius);
    }
  }
  Kokkos::deep_copy(execution, modes, host_modes);
  Kokkos::deep_copy(execution, theta, host_theta);
  Kokkos::deep_copy(execution, input, host_input);
  Kokkos::deep_copy(execution, input_stamps, generation);

  teuk::RouteBTeukolskyPrimaryJetTower tower(1, grid, 1,
                                              "routeb_fixture_tower");
  tower.initialize(execution, parameters, modes, theta, input, input_stamps,
                   generation, teuk::ReductionEvolution::FreeDamped, 0.0);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> angular(
      "routeb_fixture_angular", 1, 4, radial_count, 1);
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      angular_stamps("routeb_fixture_angular_stamps", 1, 4, radial_count, 1);
  auto host_angular = Kokkos::create_mirror_view(angular);
  auto host_angular_stamps = Kokkos::create_mirror_view(angular_stamps);
  for (std::size_t level = 0; level < 4; ++level) {
    for (std::size_t order = 0; order < 4; ++order) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        const bool active = order < 4 - level;
        host_angular(0, order, radial, 0) =
            active ? amplitude * fixture_angular_coefficient(
                                     level, order, grid.coordinate(radial))
                   : C{};
        host_angular_stamps(0, order, radial, 0) =
            active ? generation : 0;
      }
    }
    Kokkos::deep_copy(execution, angular, host_angular);
    Kokkos::deep_copy(execution, angular_stamps, host_angular_stamps);
    tower.advance(execution, angular, angular_stamps, generation,
                  teuk::ReductionEvolution::FreeDamped, 0.0);
  }
  execution.fence("finish independent Route-B primary fixture");
  const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           tower.values());
  const auto stamps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           tower.stamps());
  TowerResult result;
  for (std::size_t level = 0; level < 5; ++level) {
    for (const std::size_t radial : {std::size_t{0}, radial_count - 1}) {
      for (std::size_t field = 0; field < 3; ++field) {
        result.endpoints.push_back(values(level, 0, field, radial, 0));
      }
      result.stamps.push_back(stamps(level, 0, radial, 0));
    }
  }
  return result;
}

struct ContractFixture {
  static constexpr std::size_t radial_count = 9;
  static constexpr std::size_t theta_count = 2;
  static constexpr std::uint64_t generation = 444;
  teuk::ExecutionSpace execution;
  teuk::TeukolskyParameters parameters;
  teuk::UniformRadialGrid grid;
  Kokkos::View<int*, teuk::MemorySpace> modes;
  Kokkos::View<double*, teuk::MemorySpace> theta;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> input;
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      input_stamps;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> angular;
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      angular_stamps;
  teuk::RouteBTeukolskyPrimaryJetTower<> tower;

  ContractFixture()
      : parameters{},
        grid(radial_count, 0.0, 0.7),
        modes("routeb_contract_modes", 1),
        theta("routeb_contract_theta", theta_count),
        input("routeb_contract_input", 1, 3, radial_count, theta_count),
        input_stamps("routeb_contract_input_stamps", 1, radial_count,
                     theta_count),
        angular("routeb_contract_angular", 1, 4, radial_count, theta_count),
        angular_stamps("routeb_contract_angular_stamps", 1, 4, radial_count,
                       theta_count),
        tower(1, grid, theta_count, "routeb_contract_tower") {
    parameters.mass = 1.0;
    parameters.spin = 0.63;
    parameters.compactification_length = 1.4;
    parameters.spin_weight = -2;
    parameters.reduction_damping = 0.31;
    auto host_modes = Kokkos::create_mirror_view(modes);
    auto host_theta = Kokkos::create_mirror_view(theta);
    auto host_input = Kokkos::create_mirror_view(input);
    host_modes(0) = -2;
    host_theta(0) = 0.82;
    host_theta(1) = 1.17;
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t node = 0; node < theta_count; ++node) {
        for (std::size_t field = 0; field < 3; ++field) {
          host_input(0, field, radial, node) =
              fixture_initial_value(field, grid.coordinate(radial)) *
              (1.0 + 0.03 * node);
        }
      }
    }
    Kokkos::deep_copy(execution, modes, host_modes);
    Kokkos::deep_copy(execution, theta, host_theta);
    Kokkos::deep_copy(execution, input, host_input);
    Kokkos::deep_copy(execution, input_stamps, generation);
    fill_angular(0);
  }

  void fill_angular(const std::size_t level) {
    auto host = Kokkos::create_mirror_view(angular);
    auto host_stamps = Kokkos::create_mirror_view(angular_stamps);
    for (std::size_t order = 0; order < 4; ++order) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (std::size_t node = 0; node < theta_count; ++node) {
          const bool active = order < 4 - level;
          host(0, order, radial, node) =
              active ? fixture_angular_coefficient(
                           level, order, grid.coordinate(radial)) *
                           (1.0 + 0.02 * node)
                     : C{};
          host_stamps(0, order, radial, node) =
              active ? generation : 0;
        }
      }
    }
    Kokkos::deep_copy(execution, angular, host);
    Kokkos::deep_copy(execution, angular_stamps, host_stamps);
  }

  void initialize() {
    tower.initialize(execution, parameters, modes, theta, input, input_stamps,
                     generation, teuk::ReductionEvolution::FreeDamped, 0.0);
  }
};

TowerResult run_rotating_tower(const std::size_t radial_count,
                               const bool strided_input = false,
                               const bool strided_angular = false,
                               const double overall_amplitude = 1.0) {
  constexpr int ell_max = 5;
  constexpr int node_count = 7;
  constexpr std::uint64_t generation = 917;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.reduction_damping = 0.31;
  const teuk::UniformRadialGrid grid(
      radial_count, 0.0,
      teuk::compactified_outer_horizon_radius(parameters));
  const auto angular_grid = teuk::angular::gauss_legendre(node_count);
  Kokkos::View<int*, teuk::MemorySpace> modes("routeb_primary_modes",
                                              registry.size());
  Kokkos::View<double*, teuk::MemorySpace> theta("routeb_primary_theta",
                                                  node_count);
  auto host_modes = Kokkos::create_mirror_view(modes);
  auto host_theta = Kokkos::create_mirror_view(theta);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    host_modes(mode) = registry.modes()[mode];
  }
  for (int node = 0; node < node_count; ++node) {
    host_theta(node) = std::acos(angular_grid.x[static_cast<std::size_t>(node)]);
  }
  Kokkos::deep_copy(execution, modes, host_modes);
  Kokkos::deep_copy(execution, theta, host_theta);

  using RightInput =
      Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace>;
  using StrideInput =
      Kokkos::View<C****, Kokkos::LayoutStride, teuk::MemorySpace>;
  RightInput right_input("routeb_primary_right_input", registry.size(), 3,
                         radial_count, node_count);
  StrideInput stride_input(
      "routeb_primary_stride_input",
      Kokkos::LayoutStride(
          registry.size(), 3 * (radial_count * (node_count + 1) + 3) + 7, 3,
          radial_count * (node_count + 1) + 3, radial_count, node_count + 1,
          node_count, 1));
  auto host_input = Kokkos::create_mirror_view(right_input);
  for (std::size_t mode_index = 0; mode_index < registry.size(); ++mode_index) {
    const int m = registry.modes()[mode_index];
    const teuk::angular::SpinWeightedTransform transform(
        -2, m, ell_max, node_count);
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double R = grid.coordinate(radial);
      for (std::size_t field = 0; field < 3; ++field) {
        std::vector<C> modal(transform.mode_count());
        for (std::size_t ell_index = 0; ell_index < modal.size();
             ++ell_index) {
          const double alpha = 0.37 + 0.11 * field + 0.07 * ell_index;
          const C amplitude(0.4 + 0.09 * mode_index + 0.05 * field +
                                0.03 * ell_index,
                            -0.2 + 0.04 * mode_index - 0.02 * field +
                                0.01 * ell_index);
          modal[ell_index] = overall_amplitude * amplitude *
                             C(std::exp(alpha * R),
                               0.13 * std::sin((alpha + 0.4) * R));
        }
        const auto nodal = transform.synthesize(modal);
        for (int node = 0; node < node_count; ++node) {
          host_input(mode_index, field, radial,
                     static_cast<std::size_t>(node)) =
              nodal[static_cast<std::size_t>(node)];
        }
      }
    }
  }
  Kokkos::deep_copy(execution, right_input, host_input);
  auto host_stride = Kokkos::create_mirror_view(stride_input);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < 3; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (int node = 0; node < node_count; ++node) {
          host_stride(mode, field, radial, node) =
              host_input(mode, field, radial, node);
        }
      }
    }
  }
  Kokkos::deep_copy(execution, stride_input, host_stride);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      input_stamps("routeb_primary_input_stamps", registry.size(),
                   radial_count, node_count);
  Kokkos::deep_copy(execution, input_stamps, generation);

  teuk::RouteBTeukolskyPrimaryJetTower tower(
      registry.size(), grid, node_count,
      strided_input ? "routeb_primary_stride_tower"
                    : "routeb_primary_right_tower");
  if (strided_input) {
    tower.initialize(execution, parameters, modes, theta, stride_input,
                     input_stamps, generation,
                     teuk::ReductionEvolution::FreeDamped, 0.0);
  } else {
    tower.initialize(execution, parameters, modes, theta, right_input,
                     input_stamps, generation,
                     teuk::ReductionEvolution::FreeDamped, 0.0);
  }
  teuk::SignedModeAngularCoordinator angular_coordinator(
      execution, registry, -2, 0, ell_max, node_count, radial_count,
      teuk::KerrParameters{parameters.mass, parameters.spin,
                           parameters.compactification_length});
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> angular(
      "routeb_primary_angular", registry.size(), 4, radial_count, node_count);
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      angular_stamps("routeb_primary_angular_stamps", registry.size(),
                     4, radial_count, node_count);
  using StrideAngular =
      Kokkos::View<C****, Kokkos::LayoutStride, teuk::MemorySpace>;
  using StrideAngularStamp = Kokkos::View<std::uint64_t****,
                                          Kokkos::LayoutStride,
                                          teuk::MemorySpace>;
  StrideAngular stride_angular(
      "routeb_primary_stride_angular",
      Kokkos::LayoutStride(registry.size(),
                           4 * (radial_count * (node_count + 1) + 3) + 7,
                           4, radial_count * (node_count + 1) + 3,
                           radial_count, node_count + 1, node_count, 1));
  StrideAngularStamp stride_angular_stamps(
      "routeb_primary_stride_angular_stamps",
      Kokkos::LayoutStride(registry.size(),
                           4 * (radial_count * (node_count + 1) + 5) + 11,
                           4, radial_count * (node_count + 1) + 5,
                           radial_count, node_count + 1, node_count, 1));
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace>
      wrong_angular_derivatives("routeb_wrong_angular_derivatives",
                                registry.size(), 4, 4, radial_count,
                                node_count);
  double old_angular_recovery_difference = 0.0;
  for (std::size_t level = 0; level < 4; ++level) {
    for (std::size_t order = 0; order < 4 - level; ++order) {
      if (strided_angular) {
        angular_coordinator.laplacian(execution, tower.psi_coefficients(),
                                      order, stride_angular, order);
      } else {
        angular_coordinator.laplacian(execution, tower.psi_coefficients(),
                                      order, angular, order);
      }
    }
    if (level == 1 && !strided_angular) {
      teuk::evaluate_routeb_fornberg_derivatives(
          execution, angular, wrong_angular_derivatives, grid.spacing());
      execution.fence("compare forbidden Route-B angular recovery");
      const auto correct_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, angular);
      const auto wrong_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, wrong_angular_derivatives);
      for (std::size_t mode = 0; mode < registry.size(); ++mode) {
        for (std::size_t radial = 0; radial < radial_count; ++radial) {
          for (int node = 0; node < node_count; ++node) {
            old_angular_recovery_difference = std::max(
                old_angular_recovery_difference,
                Kokkos::abs(wrong_host(mode, 0, 0, radial, node) -
                            correct_host(mode, 1, radial, node)));
          }
        }
      }
    }
    if (strided_angular) {
      Kokkos::deep_copy(execution, stride_angular_stamps, generation);
      tower.advance(execution, stride_angular, stride_angular_stamps,
                    generation, teuk::ReductionEvolution::FreeDamped, 0.0);
    } else {
      Kokkos::deep_copy(execution, angular_stamps, generation);
      tower.advance(execution, angular, angular_stamps, generation,
                    teuk::ReductionEvolution::FreeDamped, 0.0);
    }
  }
  execution.fence("finish Route-B rotating primary tower");
  const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           tower.values());
  const auto stamps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           tower.stamps());
  TowerResult result;
  for (std::size_t level = 0; level < 5; ++level) {
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      for (std::size_t field = 0; field < 3; ++field) {
        for (const std::size_t radial : {std::size_t{0}, radial_count - 1}) {
          for (int node = 0; node < node_count; ++node) {
            result.endpoints.push_back(values(level, mode, field, radial,
                                              static_cast<std::size_t>(node)));
            result.stamps.push_back(stamps(level, mode, radial,
                                           static_cast<std::size_t>(node)));
          }
        }
      }
    }
  }
  result.old_angular_recovery_difference = old_angular_recovery_difference;
  return result;
}

TEST_CASE("Route-B Teukolsky coefficient jets reproduce point coefficients") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = -2;
  const double radius = 0.31;
  const double theta = 0.82;
  const auto point = teuk::teukolsky_coefficients(parameters, radius, theta);
  const auto jets =
      teuk::routeb_teukolsky_coefficient_jets<4>(parameters, radius, theta);
  CHECK_COMPLEX_NEAR(jets.time[0], C(point.time, 0.0), 2.0e-14);
  CHECK_COMPLEX_NEAR(jets.radial_advection[0],
                     C(point.radial_advection, 0.0), 2.0e-14);
  CHECK_COMPLEX_NEAR(jets.radial_principal[0],
                     C(point.radial_principal, 0.0), 2.0e-14);
  CHECK_COMPLEX_NEAR(jets.definition[0], point.definition, 2.0e-14);
  CHECK_COMPLEX_NEAR(jets.q[0], point.q, 2.0e-14);
  CHECK_COMPLEX_NEAR(jets.psi[0], point.psi, 2.0e-14);
}

TEST_CASE("Route-B Kerr coefficient derivatives match expanded polynomial oracle") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = -2;
  const long double mass = parameters.mass;
  const long double spin = parameters.spin;
  const long double length = parameters.compactification_length;
  const long double spin_weight = parameters.spin_weight;
  const long double azimuthal_mode = parameters.azimuthal_mode;
  const long double length2 = length * length;
  const long double length4 = length2 * length2;
  const long double spin2 = spin * spin;
  const long double radius = 0.317L;
  const long double theta = 0.82L;
  const LC imaginary(0.0L, 1.0L);
  std::array<std::array<LC, 5>, 6> polynomials{};
  polynomials[0][0] =
      LC(16.0L * mass * mass - spin2 * std::sin(theta) * std::sin(theta),
         0.0L);
  polynomials[0][1] = LC(
      (32.0L * mass * mass * mass - 8.0L * mass * spin2) / length2, 0.0L);
  polynomials[0][2] = LC(-16.0L * mass * mass * spin2 / length4, 0.0L);
  polynomials[1][0] = LC(length2, 0.0L);
  polynomials[1][2] =
      LC(-(8.0L * mass * mass - spin2) / length2, 0.0L);
  polynomials[1][3] = LC(4.0L * spin2 * mass / length4, 0.0L);
  polynomials[2][2] = LC(1.0L, 0.0L);
  polynomials[2][3] = LC(-2.0L * mass / length2, 0.0L);
  polynomials[2][4] = LC(spin2 / length4, 0.0L);
  polynomials[3][0] =
      2.0L * imaginary * spin * azimuthal_mode -
      LC(4.0L * mass * spin_weight, 0.0L) +
      2.0L * imaginary * spin_weight * spin * std::cos(theta);
  polynomials[3][1] =
      8.0L * imaginary * spin * azimuthal_mode * mass / length2 +
      LC(8.0L * mass * mass * (2.0L + spin_weight) / length2 -
             2.0L * spin2 / length2,
         0.0L);
  polynomials[3][2] = LC(-12.0L * mass * spin2 / length4, 0.0L);
  polynomials[4][1] = LC(2.0L * (1.0L + spin_weight), 0.0L);
  polynomials[4][2] =
      LC(-2.0L * (3.0L + spin_weight) * mass / length2, 0.0L) -
      2.0L * imaginary * spin * azimuthal_mode / length2;
  polynomials[4][3] = LC(4.0L * spin2 / length4, 0.0L);
  polynomials[5][1] =
      LC(-2.0L * (1.0L + spin_weight) * mass / length2, 0.0L) -
      2.0L * imaginary * spin * azimuthal_mode / length2;
  polynomials[5][2] = LC(2.0L * spin2 / length4, 0.0L);
  const auto actual = teuk::routeb_teukolsky_coefficient_jets<4>(
      parameters, static_cast<double>(radius), static_cast<double>(theta));
  const std::array<teuk::RouteBRadialTaylorJet<4, C>, 6> actual_fields{
      actual.time, actual.radial_advection, actual.radial_principal,
      actual.definition, actual.q, actual.psi};
  for (std::size_t field = 0; field < actual_fields.size(); ++field) {
    for (std::size_t order = 0; order <= 4; ++order) {
      const LC expected =
          polynomial_taylor_coefficient(polynomials[field], radius, order);
      CHECK_COMPLEX_NEAR(
          actual_fields[field][order],
          C(static_cast<double>(expected.real()),
            static_cast<double>(expected.imag())),
          3.0e-14);
    }
  }
}

TEST_CASE("Route-B Kerr jet respects signed m and equatorial spin parity") {
  teuk::TeukolskyParameters plus;
  plus.mass = 1.0;
  plus.spin = 0.63;
  plus.compactification_length = 1.4;
  plus.spin_weight = -2;
  plus.azimuthal_mode = 2;
  auto minus_m = plus;
  minus_m.azimuthal_mode = -2;
  auto reflected = plus;
  reflected.spin = -plus.spin;
  reflected.azimuthal_mode = -plus.azimuthal_mode;
  const double radius = 0.31;
  const double equator = 0.5 * std::acos(-1.0);
  const auto plus_jet =
      teuk::routeb_teukolsky_coefficient_jets<4>(plus, radius, equator);
  const auto minus_m_jet =
      teuk::routeb_teukolsky_coefficient_jets<4>(minus_m, radius, equator);
  const auto reflected_jet =
      teuk::routeb_teukolsky_coefficient_jets<4>(reflected, radius, equator);
  CHECK(Kokkos::abs(plus_jet.definition[0] - minus_m_jet.definition[0]) >
        1.0e-2);
  CHECK(Kokkos::abs(plus_jet.q[0] - minus_m_jet.q[0]) > 1.0e-2);
  for (std::size_t order = 0; order <= 4; ++order) {
    CHECK_COMPLEX_NEAR(plus_jet.time[order], reflected_jet.time[order],
                       3.0e-14);
    CHECK_COMPLEX_NEAR(plus_jet.definition[order],
                       reflected_jet.definition[order], 3.0e-14);
    CHECK_COMPLEX_NEAR(plus_jet.q[order], reflected_jet.q[order], 3.0e-14);
    CHECK_COMPLEX_NEAR(plus_jet.psi[order], reflected_jet.psi[order],
                       3.0e-14);
  }
}

TEST_CASE("Route-B Teukolsky point jet step matches compact point algebra") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = 2;
  parameters.reduction_damping = 0.27;
  const double radius = 0.29;
  const double theta = 1.03;
  teuk::RouteBTeukolskyStateJet<1> state;
  state.P[0] = C(0.4, -0.2);
  state.P[1] = C(0.7, 0.1);
  state.Q[0] = C(-0.3, 0.5);
  state.Q[1] = C(0.2, -0.6);
  state.psi[0] = C(0.8, 0.3);
  state.psi[1] = C(-0.4, 0.2);
  teuk::RouteBRadialTaylorJet<0, C> angular;
  angular[0] = C(-0.11, 0.07);
  const auto actual = teuk::routeb_teukolsky_primary_jet_step(
      parameters, radius, theta, state, angular);
  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, radius, theta);
  const teuk::TeukolskyState point{state.P[0], state.Q[0], state.psi[0]};
  const C psi_rhs = teuk::teukolsky_psi_rhs(coefficients, point);
  const double epsilon = 1.0e-6;
  auto velocity = [&](const double r) {
    teuk::TeukolskyState shifted{
        state.P[0] + (r - radius) * state.P[1],
        state.Q[0] + (r - radius) * state.Q[1],
        state.psi[0] + (r - radius) * state.psi[1]};
    return teuk::teukolsky_psi_rhs(
        teuk::teukolsky_coefficients(parameters, r, theta), shifted);
  };
  const C derivative_velocity =
      (velocity(radius + epsilon) - velocity(radius - epsilon)) /
      (2.0 * epsilon);
  CHECK_COMPLEX_NEAR(actual.P[0],
                     teuk::teukolsky_p_rhs(coefficients, point, state.Q[1],
                                           angular[0], C{}),
                     2.0e-14);
  CHECK_COMPLEX_NEAR(actual.Q[0],
                     derivative_velocity -
                         parameters.reduction_damping *
                             (state.Q[0] - state.psi[1]),
                     2.0e-9);
  CHECK_COMPLEX_NEAR(actual.psi[0], psi_rhs, 2.0e-14);
}

TEST_CASE("Route-B Teukolsky jet point algebra has device parity") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = -2;
  parameters.reduction_damping = 0.31;
  teuk::RouteBTeukolskyStateJet<4> state;
  teuk::RouteBRadialTaylorJet<3, C> angular;
  for (std::size_t order = 0; order <= 4; ++order) {
    state.P[order] = C(0.2 + 0.03 * order, -0.1 + 0.02 * order);
    state.Q[order] = C(-0.3 + 0.04 * order, 0.2 - 0.01 * order);
    state.psi[order] = C(0.5 - 0.02 * order, 0.07 + 0.01 * order);
    if (order < 4) angular[order] = C(0.11 * order, -0.03 * order);
  }
  const auto expected = teuk::routeb_teukolsky_primary_jet_step(
      parameters, 0.27, 0.82, state, angular);
  Kokkos::View<teuk::RouteBTeukolskyStateJet<3>*, teuk::MemorySpace> result(
      "routeb_primary_point_device", 1);
  Kokkos::parallel_for(
      "routeb_primary_point_device", 1, KOKKOS_LAMBDA(const int) {
        result(0) = teuk::routeb_teukolsky_primary_jet_step(
            parameters, 0.27, 0.82, state, angular);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         result);
  for (std::size_t order = 0; order <= 3; ++order) {
    CHECK_COMPLEX_NEAR(host(0).P[order], expected.P[order], 2.0e-14);
    CHECK_COMPLEX_NEAR(host(0).Q[order], expected.Q[order], 2.0e-14);
    CHECK_COMPLEX_NEAR(host(0).psi[order], expected.psi[order], 2.0e-14);
  }
}

TEST_CASE("Route-B rotating primary jet tower is fourth order at endpoints") {
  const auto coarse = run_rotating_tower(9);
  const auto medium = run_rotating_tower(17);
  const auto fine = run_rotating_tower(33);
  constexpr std::size_t values_per_level = 2 * 3 * 2 * 7;
  bool all_converged = true;
  for (std::size_t level = 1; level < 5; ++level) {
    for (std::size_t field = 0; field < 3; ++field) {
      double coarse_medium = 0.0;
      double medium_fine = 0.0;
      for (std::size_t within = 0; within < values_per_level; ++within) {
        if ((within / 14) % 3 != field) continue;
        const std::size_t index = level * values_per_level + within;
        coarse_medium = std::max(
            coarse_medium,
            Kokkos::abs(coarse.endpoints[index] - medium.endpoints[index]));
        medium_fine = std::max(
            medium_fine,
            Kokkos::abs(medium.endpoints[index] - fine.endpoints[index]));
      }
      const double ratio = medium_fine == 0.0
                               ? std::numeric_limits<double>::infinity()
                               : coarse_medium / medium_fine;
      std::cout << "Route-B Teuk h" << level << " field " << field
                << " endpoint errors " << coarse_medium << ' '
                << medium_fine << " ratio " << ratio << '\n';
      const bool exact_endpoint_component =
          level == 1 && (field == 0 || field == 2);
      all_converged = all_converged &&
                      (exact_endpoint_component
                           ? coarse_medium < 1.0e-13 && medium_fine < 1.0e-13
                           : ratio > 15.0);
    }
  }
  CHECK(all_converged);
  for (const auto stamp : fine.stamps) CHECK(stamp == 917);
}

TEST_CASE("Route-B primary tower matches independent high precision endpoints") {
  const auto coarse = run_independent_fixture(9);
  const auto medium = run_independent_fixture(17);
  const auto fine = run_independent_fixture(33);
  constexpr double exact_floor = 2.0e-13;
  bool all_converged = true;
  for (std::size_t level = 0; level < 5; ++level) {
    for (std::size_t field = 0; field < 3; ++field) {
      double errors[3]{};
      for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
        const std::size_t index = level * 6 + endpoint * 3 + field;
        const C expected =
            routeb_primary_fixture::endpoint_values[level][endpoint][field];
        errors[0] = std::max(errors[0],
                             Kokkos::abs(coarse.endpoints[index] - expected));
        errors[1] = std::max(errors[1],
                             Kokkos::abs(medium.endpoints[index] - expected));
        errors[2] = std::max(errors[2],
                             Kokkos::abs(fine.endpoints[index] - expected));
      }
      const double ratio_9_17 = errors[1] == 0.0
                                     ? std::numeric_limits<double>::infinity()
                                     : errors[0] / errors[1];
      const double ratio_17_33 = errors[2] == 0.0
                                      ? std::numeric_limits<double>::infinity()
                                      : errors[1] / errors[2];
      std::cout << "Route-B independent h" << level << " field " << field
                << " errors " << errors[0] << ' ' << errors[1] << ' '
                << errors[2] << " ratios " << ratio_9_17 << ' '
                << ratio_17_33 << '\n';
      const bool exact_at_double_precision =
          errors[0] < exact_floor && errors[1] < exact_floor &&
          errors[2] < exact_floor;
      all_converged = all_converged &&
                      (exact_at_double_precision ||
                       (std::isfinite(errors[0]) &&
                        std::isfinite(errors[1]) &&
                        std::isfinite(errors[2]) && ratio_9_17 > 15.0 &&
                        ratio_17_33 > 15.0));
    }
  }
  double h4_signal = 0.0;
  for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
    for (std::size_t field = 0; field < 3; ++field) {
      h4_signal = std::max(
          h4_signal,
          Kokkos::abs(routeb_primary_fixture::endpoint_values[4][endpoint]
                                                               [field]));
    }
  }
  CHECK(std::isfinite(h4_signal));
  CHECK(h4_signal > 1.0e-2);
  CHECK(all_converged);

  const auto ceiling = run_independent_fixture(65);
  for (std::size_t field = 0; field < 3; ++field) {
    double errors[3]{};
    for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
      const std::size_t index = 4 * 6 + endpoint * 3 + field;
      const C expected =
          routeb_primary_fixture::endpoint_values[4][endpoint][field];
      errors[0] = std::max(errors[0],
                           Kokkos::abs(medium.endpoints[index] - expected));
      errors[1] = std::max(errors[1],
                           Kokkos::abs(fine.endpoints[index] - expected));
      errors[2] = std::max(errors[2],
                           Kokkos::abs(ceiling.endpoints[index] - expected));
    }
    std::cout << "Route-B independent h4 resolution ceiling field " << field
              << " N17/N33/N65 errors " << errors[0] << ' ' << errors[1]
              << ' ' << errors[2] << " ratios " << errors[0] / errors[1]
              << ' ' << errors[1] / errors[2] << '\n';
  }
}

TEST_CASE("Route-B primary jet tower preserves strided input parity") {
  const auto right = run_rotating_tower(17, false);
  const auto strided = run_rotating_tower(17, true, true);
  CHECK(right.endpoints.size() == strided.endpoints.size());
  for (std::size_t index = 0; index < right.endpoints.size(); ++index) {
    CHECK_COMPLEX_NEAR(right.endpoints[index], strided.endpoints[index],
                       2.0e-12);
  }
}

TEST_CASE("Route-B rotating primary tower is linear and preserves signed modes") {
  constexpr double scale = -0.37;
  const auto baseline = run_rotating_tower(17);
  const auto scaled = run_rotating_tower(17, false, false, scale);
  for (std::size_t index = 0; index < baseline.endpoints.size(); ++index) {
    CHECK_COMPLEX_NEAR(scaled.endpoints[index],
                       scale * baseline.endpoints[index], 3.0e-11);
  }
  constexpr std::size_t values_per_level = 2 * 3 * 2 * 7;
  const std::size_t h4_begin = 4 * values_per_level;
  double signed_mode_difference = 0.0;
  for (std::size_t within_mode = 0; within_mode < 3 * 2 * 7;
       ++within_mode) {
    signed_mode_difference = std::max(
        signed_mode_difference,
        Kokkos::abs(baseline.endpoints[h4_begin + within_mode] -
                    baseline.endpoints[h4_begin + 3 * 2 * 7 + within_mode]));
  }
  CHECK(signed_mode_difference > 1.0e-3);
}

TEST_CASE("Route-B angular jets cannot be recovered from sampled values") {
  const auto result = run_rotating_tower(17);
  std::cout << "Route-B forbidden value-then-Fornberg angular difference "
            << result.old_angular_recovery_difference << '\n';
  CHECK(std::isfinite(result.old_angular_recovery_difference));
  CHECK(result.old_angular_recovery_difference > 1.0e-10);
}

TEST_CASE("Route-B primary tower rejects unsupported policies and malformed views") {
  CHECK(throws_invalid_argument([] {
    teuk::RouteBTeukolskyPrimaryJetTower<> invalid(
        0, teuk::UniformRadialGrid(9, 0.0, 0.7), 1);
  }));
  ContractFixture fixture;
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.theta,
        fixture.input, fixture.input_stamps, 0,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.theta,
        fixture.input, fixture.input_stamps, ContractFixture::generation,
        teuk::ReductionEvolution::StageConstrained, 0.0);
  }));
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.theta,
        fixture.input, fixture.input_stamps, ContractFixture::generation,
        teuk::ReductionEvolution::FreeDamped, 1.0e-4);
  }));
  auto negative_damping = fixture.parameters;
  negative_damping.reduction_damping = -0.1;
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, negative_damping, fixture.modes, fixture.theta,
        fixture.input, fixture.input_stamps, ContractFixture::generation,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> wrong_input(
      "routeb_contract_wrong_input", 1, 2, ContractFixture::radial_count,
      ContractFixture::theta_count);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.theta,
        wrong_input, fixture.input_stamps, ContractFixture::generation,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  using NullInput = Kokkos::View<C****, Kokkos::LayoutRight,
                                 teuk::MemorySpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  NullInput null_input(static_cast<C*>(nullptr), 1, 3,
                       ContractFixture::radial_count,
                       ContractFixture::theta_count);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.theta,
        null_input, fixture.input_stamps, ContractFixture::generation,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  using AliasInput = Kokkos::View<C****, Kokkos::LayoutRight,
                                  teuk::MemorySpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  AliasInput owned_alias(const_cast<C*>(fixture.tower.values().data()), 1, 3,
                         ContractFixture::radial_count,
                         ContractFixture::theta_count);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.theta,
        owned_alias, fixture.input_stamps, ContractFixture::generation,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  fixture.initialize();
  CHECK(throws_logic_error([&] {
    fixture.tower.advance(fixture.execution, fixture.angular,
                          fixture.angular_stamps,
                          ContractFixture::generation + 1,
                          teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  CHECK(throws_invalid_argument([&] {
    fixture.tower.advance(fixture.execution, fixture.angular,
                          fixture.angular_stamps,
                          ContractFixture::generation,
                          teuk::ReductionEvolution::StageConstrained, 0.0);
  }));
  CHECK(throws_invalid_argument([&] {
    fixture.tower.advance(fixture.execution, fixture.angular,
                          fixture.angular_stamps,
                          ContractFixture::generation,
                          teuk::ReductionEvolution::FreeDamped, 0.01);
  }));
  CHECK(throws_invalid_argument([&] {
    Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> wrong_angular(
        "routeb_contract_wrong_angular", 1, 3,
        ContractFixture::radial_count, ContractFixture::theta_count);
    fixture.tower.advance(fixture.execution, wrong_angular,
                          fixture.angular_stamps,
                          ContractFixture::generation,
                          teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  using StrideAngular =
      Kokkos::View<C****, Kokkos::LayoutStride, teuk::MemorySpace>;
  StrideAngular internally_aliased(
      "routeb_contract_internal_angular_alias",
      Kokkos::LayoutStride(1, 72, 4, 1, ContractFixture::radial_count, 4,
                           ContractFixture::theta_count, 1));
  CHECK(throws_invalid_argument([&] {
    fixture.tower.advance(fixture.execution, internally_aliased,
                          fixture.angular_stamps,
                          ContractFixture::generation,
                          teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
}

TEST_CASE("Route-B primary tower latches mode and theta metadata") {
  ContractFixture baseline;
  ContractFixture mutated;
  baseline.initialize();
  mutated.initialize();
  Kokkos::deep_copy(mutated.execution, mutated.modes, 7);
  Kokkos::deep_copy(mutated.execution, mutated.theta, 2.7);
  baseline.tower.advance(baseline.execution, baseline.angular,
                         baseline.angular_stamps, ContractFixture::generation,
                         teuk::ReductionEvolution::FreeDamped, 0.0);
  mutated.tower.advance(mutated.execution, mutated.angular,
                        mutated.angular_stamps, ContractFixture::generation,
                        teuk::ReductionEvolution::FreeDamped, 0.0);
  baseline.execution.fence("finish Route-B latched metadata test");
  const auto expected = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, baseline.tower.values());
  const auto actual = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, mutated.tower.values());
  for (std::size_t field = 0; field < 3; ++field) {
    for (std::size_t radial = 0; radial < ContractFixture::radial_count;
         ++radial) {
      for (std::size_t node = 0; node < ContractFixture::theta_count; ++node) {
        CHECK_COMPLEX_NEAR(actual(1, 0, field, radial, node),
                           expected(1, 0, field, radial, node), 0.0);
      }
    }
  }
}

TEST_CASE("Route-B primary tower globally poisons stale and nonfinite levels") {
  {
    ContractFixture fixture;
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     fixture.input);
    host(0, 1, 4, 1) = C(std::numeric_limits<double>::infinity(), 0.0);
    Kokkos::deep_copy(fixture.execution, fixture.input, host);
    fixture.initialize();
    fixture.execution.fence("finish nonfinite Route-B h0 contract");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.stamps());
    for (std::size_t radial = 0; radial < ContractFixture::radial_count;
         ++radial) {
      for (std::size_t node = 0; node < ContractFixture::theta_count; ++node) {
        CHECK(stamps(0, 0, radial, node) == 0);
      }
    }
  }
  {
    ContractFixture fixture;
    fixture.initialize();
    auto host_stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.angular_stamps);
    host_stamps(0, 2, 3, 1) = ContractFixture::generation - 1;
    Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, host_stamps);
    fixture.tower.advance(fixture.execution, fixture.angular,
                          fixture.angular_stamps,
                          ContractFixture::generation,
                          teuk::ReductionEvolution::FreeDamped, 0.0);
    fixture.execution.fence("finish stale Route-B angular contract");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.stamps());
    for (std::size_t radial = 0; radial < ContractFixture::radial_count;
         ++radial) {
      for (std::size_t node = 0; node < ContractFixture::theta_count; ++node) {
        CHECK(stamps(1, 0, radial, node) == 0);
      }
    }
  }
  {
    ContractFixture fixture;
    fixture.initialize();
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     fixture.angular);
    host(0, 1, 2, 0) = C(0.0, std::numeric_limits<double>::quiet_NaN());
    Kokkos::deep_copy(fixture.execution, fixture.angular, host);
    fixture.tower.advance(fixture.execution, fixture.angular,
                          fixture.angular_stamps,
                          ContractFixture::generation,
                          teuk::ReductionEvolution::FreeDamped, 0.0);
    fixture.execution.fence("finish nonfinite Route-B angular contract");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.stamps());
    for (std::size_t radial = 0; radial < ContractFixture::radial_count;
         ++radial) {
      for (std::size_t node = 0; node < ContractFixture::theta_count; ++node) {
        CHECK(stamps(1, 0, radial, node) == 0);
      }
    }
  }
}

TEST_CASE("Route-B primary coefficient stamps follow the active jet degree") {
  ContractFixture fixture;
  fixture.initialize();
  for (std::size_t level = 0; level < 4; ++level) {
    fixture.execution.fence("inspect Route-B coefficient stamps");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.psi_coefficient_stamps());
    for (std::size_t order = 0; order < 4; ++order) {
      for (std::size_t radial = 0; radial < ContractFixture::radial_count;
           ++radial) {
        for (std::size_t node = 0; node < ContractFixture::theta_count;
             ++node) {
          CHECK(stamps(0, order, radial, node) ==
                (order < 4 - level ? ContractFixture::generation : 0));
        }
      }
    }
    fixture.fill_angular(level);
    fixture.tower.advance(fixture.execution, fixture.angular,
                          fixture.angular_stamps,
                          ContractFixture::generation,
                          teuk::ReductionEvolution::FreeDamped, 0.0);
  }
  CHECK(fixture.tower.current_level() == 4);
  CHECK(fixture.tower.generation() == ContractFixture::generation);
}

TEST_CASE("Route-B primary hot advance allocates and fences nothing") {
  ContractFixture fixture;
  fixture.initialize();
  fixture.execution.fence("warm Route-B primary tower");
  routeb_primary_allocations = 0;
  routeb_primary_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_routeb_primary_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      count_routeb_primary_fence);
  fixture.tower.advance(fixture.execution, fixture.angular,
                        fixture.angular_stamps, ContractFixture::generation,
                        teuk::ReductionEvolution::FreeDamped, 0.0);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  fixture.execution.fence("finish Route-B primary hot audit");
  CHECK(routeb_primary_allocations == 0);
  CHECK(routeb_primary_fences == 0);
}

}  // namespace
