#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "routeb_reconstruction_fixture.hpp"
#include "teuk/routeb_reconstruction_jet.hpp"

namespace {

using C = teuk::Complex;
constexpr std::array<const char*, 7> reconstruction_names{
    "G", "Lambda", "H", "B", "Pi", "C", "U"};

int reconstruction_allocations = 0;
int reconstruction_fences = 0;

void count_reconstruction_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                     const void*, std::uint64_t) {
  ++reconstruction_allocations;
}
void count_reconstruction_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++reconstruction_fences;
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

C analytic_coefficient(const int kind, const int mode,
                       const std::size_t order, const double radius) {
  const C amplitude(0.17 + 0.031 * (kind + 1) + 0.019 * mode,
                    -0.11 + 0.013 * kind - 0.017 * mode);
  const double alpha = 0.61 + 0.037 * kind + 0.023 * mode;
  const C oscillatory = C(0.021, -0.014) * double(kind + 1);
  const double beta = 1.43 + 0.029 * kind + 0.017 * mode;
  return (amplitude * std::pow(alpha, int(order)) *
              std::exp(alpha * radius) +
          oscillatory * std::pow(beta, int(order)) *
              std::sin(beta * radius +
                       0.5 * std::acos(-1.0) * double(order))) /
         teuk::routeb_factorial(order);
}

struct TowerResult {
  std::vector<C> endpoints;
  std::vector<std::uint64_t> stamps;
};

TowerResult run_independent_tower(const std::size_t radial_count,
                                  const double amplitude_scale = 1.0) {
  constexpr std::uint64_t generation = 731;
  const teuk::ExecutionSpace execution;
  const teuk::KerrParameters parameters{1.0, 0.63, 1.4};
  const teuk::UniformRadialGrid grid(
      radial_count, 0.0,
      parameters.compactification_length * parameters.compactification_length /
          (parameters.mass +
           std::sqrt(parameters.mass * parameters.mass -
                     parameters.spin * parameters.spin)));
  Kokkos::View<int*, teuk::MemorySpace> modes("fixture_modes", 2);
  Kokkos::View<std::size_t*, teuk::MemorySpace> sharp("fixture_sharp", 2);
  Kokkos::View<double*, teuk::MemorySpace> theta("fixture_theta", 1);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> input(
      "fixture_input", 2, 7, radial_count, 1);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      input_stamps("fixture_input_stamps", 2, radial_count, 1);
  auto host_modes = Kokkos::create_mirror_view(modes);
  auto host_sharp = Kokkos::create_mirror_view(sharp);
  auto host_theta = Kokkos::create_mirror_view(theta);
  auto host_input = Kokkos::create_mirror_view(input);
  host_modes(0) = -1;
  host_modes(1) = 1;
  host_sharp(0) = 1;
  host_sharp(1) = 0;
  host_theta(0) = 0.82;
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t field = 0; field < 7; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        host_input(mode, field, radial, 0) =
            amplitude_scale * analytic_coefficient(
                                  int(field), int(mode), 0,
                                  grid.coordinate(radial));
      }
    }
  }
  Kokkos::deep_copy(execution, modes, host_modes);
  Kokkos::deep_copy(execution, sharp, host_sharp);
  Kokkos::deep_copy(execution, theta, host_theta);
  Kokkos::deep_copy(execution, input, host_input);
  Kokkos::deep_copy(execution, input_stamps, generation);
  teuk::RouteBReconstructionJetTower tower(2, grid, 1, "fixture_tower");
  tower.initialize(execution, parameters, modes, sharp, theta, input,
                   input_stamps, generation,
                   teuk::ReductionEvolution::FreeDamped, 0.0);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> psi4(
      "fixture_psi4", 2, 4, radial_count, 1);
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      psi4_stamps("fixture_psi4_stamps", 2, 4, radial_count, 1);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> angular(
      "fixture_angular", 2, 4, radial_count, 1);
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      angular_stamps("fixture_angular_stamps", 2, 4, radial_count, 1);
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> pass3(
      "fixture_pass3", 2, 4, 4, radial_count, 1);
  Kokkos::View<std::uint64_t*****, Kokkos::LayoutRight, teuk::MemorySpace>
      pass3_stamps("fixture_pass3_stamps", 2, 4, 4, radial_count, 1);
  auto host_psi4 = Kokkos::create_mirror_view(psi4);
  auto host_angular = Kokkos::create_mirror_view(angular);
  auto host_pass3 = Kokkos::create_mirror_view(pass3);
  for (std::size_t level = 0; level < 4; ++level) {
    const std::size_t active = 4 - level;
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (std::size_t order = 0; order < 4; ++order) {
        for (std::size_t radial = 0; radial < radial_count; ++radial) {
          const double radius = grid.coordinate(radial);
          host_psi4(mode, order, radial, 0) =
              order < active
                  ? amplitude_scale * analytic_coefficient(
                                          40 + 3 * int(level), int(mode),
                                          order, radius)
                  : C{};
          host_angular(mode, order, radial, 0) =
              order < active
                  ? amplitude_scale * analytic_coefficient(
                                          80 + 17 * int(level), int(mode),
                                          order, radius)
                  : C{};
        }
      }
    }
    Kokkos::deep_copy(execution, psi4, host_psi4);
    Kokkos::deep_copy(execution, angular, host_angular);
    Kokkos::deep_copy(execution, psi4_stamps, generation);
    auto token = tower.expected_pass_token();
    Kokkos::deep_copy(execution, angular_stamps, token);
    tower.pass1(execution, psi4, psi4_stamps, angular, angular_stamps,
                generation, token, teuk::ReductionEvolution::FreeDamped, 0.0);
    token = tower.expected_pass_token();
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (std::size_t order = 0; order < 4; ++order) {
        for (std::size_t radial = 0; radial < radial_count; ++radial) {
          host_angular(mode, order, radial, 0) =
              order < active
                  ? amplitude_scale * analytic_coefficient(
                                          81 + 17 * int(level), int(mode),
                                          order, grid.coordinate(radial))
                  : C{};
        }
      }
    }
    Kokkos::deep_copy(execution, angular, host_angular);
    Kokkos::deep_copy(execution, angular_stamps, token);
    tower.pass2(execution, angular, angular_stamps, generation, token,
                teuk::ReductionEvolution::FreeDamped, 0.0);
    token = tower.expected_pass_token();
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (std::size_t slot = 0; slot < 4; ++slot) {
        for (std::size_t order = 0; order < 4; ++order) {
          for (std::size_t radial = 0; radial < radial_count; ++radial) {
            host_pass3(mode, slot, order, radial, 0) =
                order < active
                    ? amplitude_scale * analytic_coefficient(
                                            82 + 17 * int(level) + int(slot),
                                            int(mode), order,
                                            grid.coordinate(radial))
                    : C{};
          }
        }
      }
    }
    Kokkos::deep_copy(execution, pass3, host_pass3);
    Kokkos::deep_copy(execution, pass3_stamps, token);
    tower.pass3(execution, pass3, pass3_stamps, generation, token,
                teuk::ReductionEvolution::FreeDamped, 0.0);
  }
  execution.fence("finish independent Route-B reconstruction fixture");
  const auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           tower.values());
  const auto stamps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           tower.stamps());
  TowerResult result;
  for (std::size_t level = 0; level < 5; ++level) {
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (const std::size_t radial : {std::size_t{0}, radial_count - 1}) {
        for (std::size_t field = 0; field < 7; ++field) {
          result.endpoints.push_back(values(level, mode, field, radial, 0));
        }
        result.stamps.push_back(stamps(level, mode, radial, 0));
      }
    }
  }
  return result;
}

struct ContractFixture {
  static constexpr std::size_t radial_count = 9;
  static constexpr std::uint64_t generation = 991;
  teuk::ExecutionSpace execution;
  teuk::KerrParameters parameters{1.0, 0.63, 1.4};
  teuk::UniformRadialGrid grid{radial_count, 0.0, 0.7};
  Kokkos::View<int*, teuk::MemorySpace> modes{"contract_modes", 2};
  Kokkos::View<std::size_t*, teuk::MemorySpace> sharp{"contract_sharp", 2};
  Kokkos::View<double*, teuk::MemorySpace> theta{"contract_theta", 1};
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> input{
      "contract_input", 2, 7, radial_count, 1};
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      input_stamps{"contract_input_stamps", 2, radial_count, 1};
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> jet{
      "contract_jet", 2, 4, radial_count, 1};
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      jet_stamps{"contract_jet_stamps", 2, 4, radial_count, 1};
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      angular_stamps{"contract_angular_stamps", 2, 4, radial_count, 1};
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> pass3{
      "contract_pass3", 2, 4, 4, radial_count, 1};
  Kokkos::View<std::uint64_t*****, Kokkos::LayoutRight, teuk::MemorySpace>
      pass3_stamps{"contract_pass3_stamps", 2, 4, 4, radial_count, 1};
  teuk::RouteBReconstructionJetTower<> tower{2, grid, 1, "contract_tower"};

  ContractFixture() {
    auto hm = Kokkos::create_mirror_view(modes);
    auto hs = Kokkos::create_mirror_view(sharp);
    auto ht = Kokkos::create_mirror_view(theta);
    auto hi = Kokkos::create_mirror_view(input);
    hm(0) = -1;
    hm(1) = 1;
    hs(0) = 1;
    hs(1) = 0;
    ht(0) = 0.82;
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (std::size_t field = 0; field < 7; ++field) {
        for (std::size_t radial = 0; radial < radial_count; ++radial) {
          hi(mode, field, radial, 0) =
              analytic_coefficient(int(field), int(mode), 0,
                                   grid.coordinate(radial));
        }
      }
    }
    Kokkos::deep_copy(execution, modes, hm);
    Kokkos::deep_copy(execution, sharp, hs);
    Kokkos::deep_copy(execution, theta, ht);
    Kokkos::deep_copy(execution, input, hi);
    Kokkos::deep_copy(execution, input_stamps, generation);
    Kokkos::deep_copy(execution, jet, C(0.04, -0.03));
    Kokkos::deep_copy(execution, pass3, C(-0.02, 0.05));
  }

  void initialize(const std::uint64_t selected_generation = generation) {
    tower.initialize(execution, parameters, modes, sharp, theta, input,
                     input_stamps, selected_generation,
                     teuk::ReductionEvolution::FreeDamped, 0.0);
  }

  void pass1() {
    const auto token = tower.expected_pass_token();
    Kokkos::deep_copy(execution, jet_stamps, generation);
    Kokkos::deep_copy(execution, angular_stamps, token);
    tower.pass1(execution, jet, jet_stamps, jet, angular_stamps, generation,
                token, teuk::ReductionEvolution::FreeDamped, 0.0);
  }
};

TEST_CASE("Route-B reconstruction background jets reproduce point values") {
  const teuk::KerrParameters parameters{1.0, 0.63, 1.4};
  const double radius = 0.31;
  const double theta = 0.82;
  const auto expected = teuk::kerr_background_point(parameters, radius, theta);
  const auto actual = teuk::routeb_reconstruction_background_jets<4>(
      parameters, radius, theta);
  CHECK_COMPLEX_NEAR(actual.mu0[0], expected.mu0, 2.0e-14);
  CHECK_COMPLEX_NEAR(actual.tau0[0], expected.tau0, 2.0e-14);
  CHECK_COMPLEX_NEAR(actual.pi0[0], expected.pi0, 2.0e-14);
}

TEST_CASE("Route-B reconstruction background derivatives match 100-digit oracle") {
  const teuk::KerrParameters parameters{1.0, 0.63, 1.4};
  const auto actual = teuk::routeb_reconstruction_background_jets<4>(
      parameters, 0.317, 0.82);
  const std::array<teuk::RouteBRadialTaylorJet<4, C>, 3> fields{
      actual.mu0, actual.tau0, actual.pi0};
  for (std::size_t field = 0; field < 3; ++field) {
    for (std::size_t order = 0; order <= 4; ++order) {
      CHECK_COMPLEX_NEAR(
          fields[field][order],
          routeb_reconstruction_fixture::background_coefficients[field][order],
          3.0e-14);
    }
  }
}

TEST_CASE("Route-B reconstruction degree-one step matches production point algebra") {
  const teuk::KerrParameters parameters{1.0, 0.63, 1.4};
  const double radius = 0.31;
  const double theta = 0.82;
  teuk::RouteBReconstructionStateJet<1> state;
  teuk::RouteBReconstructionStateJet<1> sharp;
  std::array<teuk::RouteBRadialTaylorJet<1, C>*, 7> fields{
      &state.G, &state.Lambda, &state.H, &state.B,
      &state.Pi, &state.C, &state.U};
  std::array<teuk::RouteBRadialTaylorJet<1, C>*, 7> sharp_fields{
      &sharp.G, &sharp.Lambda, &sharp.H, &sharp.B,
      &sharp.Pi, &sharp.C, &sharp.U};
  for (std::size_t field = 0; field < 7; ++field) {
    (*fields[field])[0] = C(0.2 + 0.03 * field, -0.1 + 0.02 * field);
    (*fields[field])[1] = C(-0.17 + 0.01 * field, 0.08 - 0.015 * field);
    (*sharp_fields[field])[0] =
        C(-0.12 + 0.025 * field, 0.19 - 0.013 * field);
    (*sharp_fields[field])[1] =
        C(0.09 - 0.02 * field, -0.07 + 0.011 * field);
  }
  teuk::RouteBRadialTaylorJet<0, C> psi4;
  teuk::RouteBRadialTaylorJet<0, C> eth1f;
  teuk::RouteBRadialTaylorJet<0, C> eth2g;
  psi4[0] = C(0.41, -0.23);
  eth1f[0] = C(-0.14, 0.07);
  eth2g[0] = C(0.09, 0.11);
  Kokkos::Array<teuk::RouteBRadialTaylorJet<0, C>, 4> angular;
  angular[0][0] = C(0.13, -0.04);
  angular[1][0] = C(-0.07, 0.08);
  angular[2][0] = C(0.16, 0.05);
  angular[3][0] = C(-0.12, -0.09);
  const auto pass1 = teuk::routeb_reconstruction_pass1_jet(
      parameters, radius, theta, state, psi4, eth1f);
  const auto pass2 = teuk::routeb_reconstruction_pass2_jet(
      parameters, radius, theta, state, eth2g);
  const auto u = teuk::routeb_reconstruction_pass3_jet(
      parameters, radius, theta, state, sharp, angular);
  const auto background = teuk::kerr_background_point(parameters, radius, theta);
  const teuk::ReconstructionFields point{
      psi4[0], state.G[0], state.H[0], state.Lambda[0], state.Pi[0],
      state.B[0], state.C[0], state.U[0], Kokkos::conj(sharp.Pi[0]),
      Kokkos::conj(sharp.B[0]), Kokkos::conj(sharp.C[0])};
  const teuk::ReconstructionAngularDerivatives point_angular{
      eth1f[0], eth2g[0], angular[0][0], angular[1][0], angular[2][0],
      angular[3][0]};
  const auto delta = teuk::reconstruction_delta_rhs(
      radius, background, point, point_angular);
  const std::array<C, 7> actual{pass1.G[0], pass1.Lambda[0], pass2.H[0],
                                pass2.B[0], pass2.Pi[0], pass2.C[0], u[0]};
  const std::array<C, 7> delta_values{delta.G, delta.Lambda, delta.H, delta.B,
                                      delta.Pi, delta.C, delta.U};
  const std::array<int, 7> falloff{2, 1, 3, 1, 2, 2, 3};
  for (std::size_t field = 0; field < 7; ++field) {
    const C expected = teuk::reconstruction_time_derivative(
        (*fields[field])[0], (*fields[field])[1], delta_values[field],
        falloff[field], radius, parameters.mass,
        parameters.compactification_length);
    CHECK_COMPLEX_NEAR(actual[field], expected, 3.0e-14);
  }
}

TEST_CASE("Route-B reconstruction enforced pass handshake completes h1") {
  constexpr std::size_t radial_count = 9;
  constexpr std::uint64_t generation = 81;
  const teuk::ExecutionSpace execution;
  const teuk::KerrParameters parameters{1.0, 0.63, 1.4};
  const teuk::UniformRadialGrid grid(9, 0.0, 0.7);
  Kokkos::View<int*, teuk::MemorySpace> modes("recon_modes", 2);
  Kokkos::View<std::size_t*, teuk::MemorySpace> sharp("recon_sharp", 2);
  Kokkos::View<double*, teuk::MemorySpace> theta("recon_theta", 1);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> input(
      "recon_input", 2, 7, radial_count, 1);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      input_stamps("recon_input_stamps", 2, radial_count, 1);
  auto host_modes = Kokkos::create_mirror_view(modes);
  auto host_sharp = Kokkos::create_mirror_view(sharp);
  auto host_theta = Kokkos::create_mirror_view(theta);
  auto host_input = Kokkos::create_mirror_view(input);
  host_modes(0) = -1;
  host_modes(1) = 1;
  host_sharp(0) = 1;
  host_sharp(1) = 0;
  host_theta(0) = 0.82;
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t field = 0; field < 7; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        const double radius = grid.coordinate(radial);
        host_input(mode, field, radial, 0) =
            C(0.2 + 0.03 * mode + 0.02 * field,
              -0.1 + 0.01 * mode - 0.015 * field) *
            std::exp((0.7 + 0.04 * field) * radius);
      }
    }
  }
  Kokkos::deep_copy(execution, modes, host_modes);
  Kokkos::deep_copy(execution, sharp, host_sharp);
  Kokkos::deep_copy(execution, theta, host_theta);
  Kokkos::deep_copy(execution, input, host_input);
  Kokkos::deep_copy(execution, input_stamps, generation);
  teuk::RouteBReconstructionJetTower tower(2, grid, 1);
  tower.initialize(execution, parameters, modes, sharp, theta, input,
                   input_stamps, generation,
                   teuk::ReductionEvolution::FreeDamped, 0.0);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> jet(
      "recon_jet", 2, 4, radial_count, 1);
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      jet_stamps("recon_jet_stamps", 2, 4, radial_count, 1);
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> pass3(
      "recon_pass3", 2, 4, 4, radial_count, 1);
  Kokkos::View<std::uint64_t*****, Kokkos::LayoutRight, teuk::MemorySpace>
      pass3_stamps("recon_pass3_stamps", 2, 4, 4, radial_count, 1);
  Kokkos::deep_copy(execution, jet, C(0.03, -0.02));
  auto token = tower.expected_pass_token();
  Kokkos::deep_copy(execution, jet_stamps, generation);
  Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, teuk::MemorySpace>
      angular_stamps("recon_angular_stamps", 2, 4, radial_count, 1);
  Kokkos::deep_copy(execution, angular_stamps, token);
  tower.pass1(execution, jet, jet_stamps, jet, angular_stamps, generation,
              token, teuk::ReductionEvolution::FreeDamped, 0.0);
  token = tower.expected_pass_token();
  Kokkos::deep_copy(execution, angular_stamps, token);
  tower.pass2(execution, jet, angular_stamps, generation, token,
              teuk::ReductionEvolution::FreeDamped, 0.0);
  token = tower.expected_pass_token();
  Kokkos::deep_copy(execution, pass3, C(0.01, 0.04));
  Kokkos::deep_copy(execution, pass3_stamps, token);
  tower.pass3(execution, pass3, pass3_stamps, generation, token,
              teuk::ReductionEvolution::FreeDamped, 0.0);
  execution.fence("finish Route-B reconstruction handshake");
  CHECK(tower.current_level() == 1);
  const auto stamps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           tower.stamps());
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      CHECK(stamps(1, mode, radial, 0) == generation);
    }
  }
}

TEST_CASE("Route-B reconstruction matches independent rotating endpoints") {
  const auto coarse = run_independent_tower(9);
  const auto medium = run_independent_tower(17);
  const auto fine = run_independent_tower(33);
  bool green = true;
  for (const auto stamp : coarse.stamps) CHECK(stamp == 731);
  for (const auto stamp : medium.stamps) CHECK(stamp == 731);
  for (const auto stamp : fine.stamps) CHECK(stamp == 731);
  for (std::size_t level = 0; level < 5; ++level) {
    for (std::size_t field = 0; field < 7; ++field) {
      double error[3]{};
      for (std::size_t mode = 0; mode < 2; ++mode) {
        for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
          const std::size_t index = level * 28 + mode * 14 + endpoint * 7 +
                                    field;
          const C expected = routeb_reconstruction_fixture::endpoint_values
              [level][mode][endpoint][field];
          error[0] = std::max(
              error[0], Kokkos::abs(coarse.endpoints[index] - expected));
          error[1] = std::max(
              error[1], Kokkos::abs(medium.endpoints[index] - expected));
          error[2] = std::max(
              error[2], Kokkos::abs(fine.endpoints[index] - expected));
        }
      }
      const double ratio1 = error[1] == 0.0
                                ? std::numeric_limits<double>::infinity()
                                : error[0] / error[1];
      const double ratio2 = error[2] == 0.0
                                ? std::numeric_limits<double>::infinity()
                                : error[1] / error[2];
      std::cout << "Route-B reconstruction h" << level << ' '
                << reconstruction_names[field]
                << " errors " << error[0] << ' ' << error[1] << ' '
                << error[2] << " ratios " << ratio1 << ' ' << ratio2
                << '\n';
      const bool exact = error[0] < 2.0e-13 && error[1] < 2.0e-13 &&
                         error[2] < 2.0e-13;
      green = green &&
              (exact || (std::isfinite(error[0]) &&
                         std::isfinite(error[1]) &&
                         std::isfinite(error[2]) && ratio1 > 15.0 &&
                         ratio2 > 15.0));
    }
  }
  double h4_signal = 0.0;
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
      for (std::size_t field = 0; field < 7; ++field) {
        h4_signal = std::max(
            h4_signal,
            Kokkos::abs(routeb_reconstruction_fixture::endpoint_values
                            [4][mode][endpoint][field]));
      }
    }
  }
  CHECK(h4_signal > 1.0e-3);
  CHECK(green);
}

TEST_CASE("Route-B reconstruction is linear and records the N65 ceiling probe") {
  constexpr double scale = -0.43;
  const auto baseline = run_independent_tower(17);
  const auto scaled = run_independent_tower(17, scale);
  for (std::size_t index = 0; index < baseline.endpoints.size(); ++index) {
    CHECK_COMPLEX_NEAR(scaled.endpoints[index],
                       scale * baseline.endpoints[index], 4.0e-11);
  }
  const auto medium = baseline;
  const auto fine = run_independent_tower(33);
  const auto ceiling = run_independent_tower(65);
  for (std::size_t field = 0; field < 7; ++field) {
    double errors[3]{};
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (std::size_t endpoint = 0; endpoint < 2; ++endpoint) {
        const std::size_t index = 4 * 28 + mode * 14 + endpoint * 7 + field;
        const C expected = routeb_reconstruction_fixture::endpoint_values
            [4][mode][endpoint][field];
        errors[0] = std::max(errors[0],
                             Kokkos::abs(medium.endpoints[index] - expected));
        errors[1] = std::max(errors[1],
                             Kokkos::abs(fine.endpoints[index] - expected));
        errors[2] = std::max(errors[2],
                             Kokkos::abs(ceiling.endpoints[index] - expected));
      }
    }
    std::cout << "Route-B reconstruction h4 ceiling "
              << reconstruction_names[field]
              << " errors " << errors[0] << ' ' << errors[1] << ' '
              << errors[2] << " ratios " << errors[0] / errors[1] << ' '
              << errors[1] / errors[2] << '\n';
  }
}

TEST_CASE("Route-B reconstruction rejects value-then-Fornberg angular recovery") {
  constexpr std::size_t points = 17;
  const teuk::KerrParameters parameters{1.0, 0.63, 1.4};
  const teuk::UniformRadialGrid grid(
      points, 0.0,
      parameters.compactification_length * parameters.compactification_length /
          (parameters.mass +
           std::sqrt(parameters.mass * parameters.mass -
                     parameters.spin * parameters.spin)));
  std::array<C, points> values{};
  for (std::size_t radial = 0; radial < points; ++radial) {
    values[radial] = analytic_coefficient(80, 0, 0, grid.coordinate(radial));
  }
  const C wrong = teuk::routeb_fornberg_direct_derivative_at(
      1, values.data(), points, 0, 1.0 / grid.spacing());
  const C correct = analytic_coefficient(80, 0, 1, grid.coordinate(0));
  const double difference = Kokkos::abs(wrong - correct);
  std::cout << "Route-B reconstruction forbidden angular recovery difference "
            << difference << '\n';
  CHECK(difference > 1.0e-10);
}

TEST_CASE("Route-B reconstruction fails closed on policy metadata and pass provenance") {
  {
    ContractFixture fixture;
    CHECK(throws_invalid_argument([&] {
      fixture.tower.initialize(
          fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
          fixture.theta, fixture.input, fixture.input_stamps, 0,
          teuk::ReductionEvolution::FreeDamped, 0.0);
    }));
    CHECK(throws_invalid_argument([&] {
      fixture.tower.initialize(
          fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
          fixture.theta, fixture.input, fixture.input_stamps,
          ContractFixture::generation, teuk::ReductionEvolution::StageConstrained,
          0.0);
    }));
    CHECK(throws_invalid_argument([&] {
      fixture.tower.initialize(
          fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
          fixture.theta, fixture.input, fixture.input_stamps,
          ContractFixture::generation, teuk::ReductionEvolution::FreeDamped,
          0.01);
    }));
    constexpr std::uint64_t zero_token_generation =
        0x9e3779b97f4a7c15ULL ^ 0xd1b54a32d192ed03ULL;
    CHECK(throws_invalid_argument([&] {
      fixture.tower.initialize(
          fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
          fixture.theta, fixture.input, fixture.input_stamps,
          zero_token_generation, teuk::ReductionEvolution::FreeDamped, 0.0);
    }));
  }
  {
    ContractFixture fixture;
    auto modes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      fixture.modes);
    modes(1) = -1;
    Kokkos::deep_copy(fixture.execution, fixture.modes, modes);
    fixture.initialize();
    fixture.execution.fence("finish duplicate Route-B modes");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.stamps());
    CHECK(stamps(0, 0, 0, 0) == 0);
  }
  {
    ContractFixture fixture;
    auto sharp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      fixture.sharp);
    sharp(0) = 0;
    Kokkos::deep_copy(fixture.execution, fixture.sharp, sharp);
    fixture.initialize();
    fixture.execution.fence("finish invalid Route-B sharp map");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.stamps());
    CHECK(stamps(0, 0, 0, 0) == 0);
  }
  {
    ContractFixture fixture;
    auto theta = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      fixture.theta);
    theta(0) = std::numeric_limits<double>::quiet_NaN();
    Kokkos::deep_copy(fixture.execution, fixture.theta, theta);
    fixture.initialize();
    fixture.execution.fence("finish invalid Route-B theta");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.stamps());
    CHECK(stamps(0, 0, 0, 0) == 0);
  }
  {
    ContractFixture fixture;
    fixture.initialize();
    const auto token = fixture.tower.expected_pass_token();
    Kokkos::deep_copy(fixture.execution, fixture.jet_stamps,
                      ContractFixture::generation);
    Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token);
    CHECK(throws_logic_error([&] {
      fixture.tower.pass1(
          fixture.execution, fixture.jet, fixture.jet_stamps, fixture.jet,
          fixture.angular_stamps, ContractFixture::generation, token + 1,
          teuk::ReductionEvolution::FreeDamped, 0.0);
    }));
    fixture.pass1();
    CHECK(throws_logic_error([&] {
      fixture.tower.pass1(
          fixture.execution, fixture.jet, fixture.jet_stamps, fixture.jet,
          fixture.angular_stamps, ContractFixture::generation, token,
          teuk::ReductionEvolution::FreeDamped, 0.0);
    }));
  }
}

TEST_CASE("Route-B reconstruction globally poisons stale pass data") {
  ContractFixture fixture;
  fixture.initialize();
  const auto token = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.jet_stamps,
                    ContractFixture::generation);
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token);
  auto stamps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                     fixture.angular_stamps);
  stamps(1, 2, 4, 0) = token - 1;
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, stamps);
  fixture.tower.pass1(
      fixture.execution, fixture.jet, fixture.jet_stamps, fixture.jet,
      fixture.angular_stamps, ContractFixture::generation, token,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto token2 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token2);
  fixture.tower.pass2(
      fixture.execution, fixture.jet, fixture.angular_stamps,
      ContractFixture::generation, token2,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto token3 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.pass3_stamps, token3);
  fixture.tower.pass3(
      fixture.execution, fixture.pass3, fixture.pass3_stamps,
      ContractFixture::generation, token3,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  fixture.execution.fence("finish stale Route-B reconstruction level");
  const auto level_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.tower.stamps());
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t radial = 0; radial < ContractFixture::radial_count;
         ++radial) {
      CHECK(level_stamps(1, mode, radial, 0) == 0);
    }
  }
}

TEST_CASE("Route-B reconstruction globally poisons nonfinite pass3 data") {
  ContractFixture fixture;
  fixture.initialize();
  fixture.pass1();
  const auto token2 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token2);
  fixture.tower.pass2(
      fixture.execution, fixture.jet, fixture.angular_stamps,
      ContractFixture::generation, token2,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto token3 = fixture.tower.expected_pass_token();
  auto angular = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                      fixture.pass3);
  angular(1, 3, 2, 4, 0) =
      C(std::numeric_limits<double>::quiet_NaN(), 0.0);
  Kokkos::deep_copy(fixture.execution, fixture.pass3, angular);
  Kokkos::deep_copy(fixture.execution, fixture.pass3_stamps, token3);
  fixture.tower.pass3(
      fixture.execution, fixture.pass3, fixture.pass3_stamps,
      ContractFixture::generation, token3,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  fixture.execution.fence("finish nonfinite Route-B reconstruction level");
  const auto stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.tower.stamps());
  const auto values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.tower.values());
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t radial = 0; radial < ContractFixture::radial_count;
         ++radial) {
      CHECK(stamps(1, mode, radial, 0) == 0);
      for (std::size_t field = 0; field < 7; ++field) {
        CHECK_COMPLEX_NEAR(values(1, mode, field, radial, 0), C{}, 0.0);
      }
    }
  }
}

TEST_CASE("Route-B reconstruction coefficient stamps track every active degree") {
  ContractFixture fixture;
  fixture.initialize();
  for (std::size_t level = 0; level < 5; ++level) {
    fixture.execution.fence("inspect Route-B reconstruction coefficient stamps");
    const auto stamps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, fixture.tower.current_coefficient_stamps());
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (std::size_t field = 0; field < 7; ++field) {
        for (std::size_t order = 0; order < 4; ++order) {
          for (std::size_t radial = 0; radial < ContractFixture::radial_count;
               ++radial) {
            const std::size_t active = level == 0 ? 4 : 5 - level;
            CHECK(stamps(mode, field, order, radial, 0) ==
                  (order < active ? ContractFixture::generation : 0));
          }
        }
      }
    }
    if (level == 4) break;
    const auto token1 = fixture.tower.expected_pass_token();
    Kokkos::deep_copy(fixture.execution, fixture.jet_stamps,
                      ContractFixture::generation);
    Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token1);
    fixture.tower.pass1(
        fixture.execution, fixture.jet, fixture.jet_stamps, fixture.jet,
        fixture.angular_stamps, ContractFixture::generation, token1,
        teuk::ReductionEvolution::FreeDamped, 0.0);
    const auto token2 = fixture.tower.expected_pass_token();
    Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token2);
    fixture.tower.pass2(
        fixture.execution, fixture.jet, fixture.angular_stamps,
        ContractFixture::generation, token2,
        teuk::ReductionEvolution::FreeDamped, 0.0);
    const auto token3 = fixture.tower.expected_pass_token();
    Kokkos::deep_copy(fixture.execution, fixture.pass3_stamps, token3);
    fixture.tower.pass3(
        fixture.execution, fixture.pass3, fixture.pass3_stamps,
        ContractFixture::generation, token3,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }
}

TEST_CASE("Route-B reconstruction hot pass sequence allocates and fences nothing") {
  ContractFixture fixture;
  fixture.initialize();
  const auto token1 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.jet_stamps,
                    ContractFixture::generation);
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token1);
  fixture.execution.fence("warm Route-B reconstruction pass sequence");
  reconstruction_allocations = 0;
  reconstruction_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_reconstruction_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      count_reconstruction_fence);
  fixture.tower.pass1(
      fixture.execution, fixture.jet, fixture.jet_stamps, fixture.jet,
      fixture.angular_stamps, ContractFixture::generation, token1,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto token2 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token2);
  fixture.tower.pass2(
      fixture.execution, fixture.jet, fixture.angular_stamps,
      ContractFixture::generation, token2,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto token3 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.pass3_stamps, token3);
  fixture.tower.pass3(
      fixture.execution, fixture.pass3, fixture.pass3_stamps,
      ContractFixture::generation, token3,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  fixture.execution.fence("finish Route-B reconstruction hot audit");
  CHECK(reconstruction_allocations == 0);
  CHECK(reconstruction_fences == 0);
}

TEST_CASE("Route-B reconstruction validates shape null alias and strided views") {
  ContractFixture fixture;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> wrong(
      "contract_wrong", 2, 6, ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
        fixture.theta, wrong, fixture.input_stamps, ContractFixture::generation,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  using NullInput = Kokkos::View<C****, Kokkos::LayoutRight,
                                 teuk::MemorySpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  NullInput null_input(static_cast<C*>(nullptr), 2, 7,
                       ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
        fixture.theta, null_input, fixture.input_stamps,
        ContractFixture::generation, teuk::ReductionEvolution::FreeDamped,
        0.0);
  }));
  using AliasInput = Kokkos::View<C****, Kokkos::LayoutRight,
                                  teuk::MemorySpace,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  AliasInput alias(const_cast<C*>(fixture.tower.values().data()), 2, 7,
                   ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.initialize(
        fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
        fixture.theta, alias, fixture.input_stamps,
        ContractFixture::generation, teuk::ReductionEvolution::FreeDamped,
        0.0);
  }));
  using StrideInput =
      Kokkos::View<C****, Kokkos::LayoutStride, teuk::MemorySpace>;
  StrideInput strided(
      "contract_strided_input",
      Kokkos::LayoutStride(2, 7 * (ContractFixture::radial_count * 2 + 3) + 5,
                           7, ContractFixture::radial_count * 2 + 3,
                           ContractFixture::radial_count, 2, 1, 1));
  auto input_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         fixture.input);
  auto strided_host = Kokkos::create_mirror_view(strided);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t field = 0; field < 7; ++field) {
      for (std::size_t radial = 0; radial < ContractFixture::radial_count;
           ++radial) {
        strided_host(mode, field, radial, 0) =
            input_host(mode, field, radial, 0);
      }
    }
  }
  Kokkos::deep_copy(fixture.execution, strided, strided_host);
  teuk::RouteBReconstructionJetTower strided_tower(
      2, fixture.grid, 1, "contract_strided_tower");
  strided_tower.initialize(
      fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
      fixture.theta, strided, fixture.input_stamps, ContractFixture::generation,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  fixture.initialize();
  fixture.execution.fence("finish Route-B strided h0 parity");
  const auto right_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.tower.values());
  const auto stride_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, strided_tower.values());
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t field = 0; field < 7; ++field) {
      for (std::size_t radial = 0; radial < ContractFixture::radial_count;
           ++radial) {
        CHECK_COMPLEX_NEAR(stride_values(0, mode, field, radial, 0),
                           right_values(0, mode, field, radial, 0), 0.0);
      }
    }
  }
  fixture.pass1();
  strided_tower.initialize(
      fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
      fixture.theta, strided, fixture.input_stamps, ContractFixture::generation,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  using StrideJet =
      Kokkos::View<C****, Kokkos::LayoutStride, teuk::MemorySpace>;
  using StrideStamp = Kokkos::View<std::uint64_t****,
                                   Kokkos::LayoutStride, teuk::MemorySpace>;
  StrideJet strided_jet(
      "contract_strided_jet",
      Kokkos::LayoutStride(2, 4 * (ContractFixture::radial_count * 2 + 3) + 5,
                           4, ContractFixture::radial_count * 2 + 3,
                           ContractFixture::radial_count, 2, 1, 1));
  StrideStamp strided_stamps(
      "contract_strided_stamps",
      Kokkos::LayoutStride(2, 4 * (ContractFixture::radial_count * 2 + 5) + 7,
                           4, ContractFixture::radial_count * 2 + 5,
                           ContractFixture::radial_count, 2, 1, 1));
  auto jet_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       fixture.jet);
  auto sj_host = Kokkos::create_mirror_view(strided_jet);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t order = 0; order < 4; ++order) {
      for (std::size_t radial = 0; radial < ContractFixture::radial_count;
           ++radial) {
        sj_host(mode, order, radial, 0) = jet_host(mode, order, radial, 0);
      }
    }
  }
  Kokkos::deep_copy(fixture.execution, strided_jet, sj_host);
  Kokkos::deep_copy(fixture.execution, strided_stamps,
                    ContractFixture::generation);
  const auto stride_token = strided_tower.expected_pass_token();
  Kokkos::View<std::uint64_t****, Kokkos::LayoutStride, teuk::MemorySpace>
      strided_angular_stamps(
          "contract_strided_angular_stamps",
          Kokkos::LayoutStride(
              2, 4 * (ContractFixture::radial_count * 2 + 7) + 9, 4,
              ContractFixture::radial_count * 2 + 7,
              ContractFixture::radial_count, 2, 1, 1));
  Kokkos::deep_copy(fixture.execution, strided_angular_stamps, stride_token);
  strided_tower.pass1(
      fixture.execution, strided_jet, strided_stamps, strided_jet,
      strided_angular_stamps, ContractFixture::generation, stride_token,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  fixture.execution.fence("finish Route-B strided pass1 parity");
  const auto right_next = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.tower.next_coefficients());
  const auto stride_next = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, strided_tower.next_coefficients());
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t field = 0; field < 2; ++field) {
      for (std::size_t order = 0; order < 4; ++order) {
        for (std::size_t radial = 0; radial < ContractFixture::radial_count;
             ++radial) {
          CHECK_COMPLEX_NEAR(stride_next(mode, field, order, radial, 0),
                             right_next(mode, field, order, radial, 0), 0.0);
        }
      }
    }
  }
}

TEST_CASE("Route-B reconstruction padded strides preserve pass2 pass3 parity") {
  ContractFixture fixture;
  fixture.initialize();

  teuk::RouteBReconstructionJetTower strided_tower(
      2, fixture.grid, 1, "contract_full_strided_tower");
  strided_tower.initialize(
      fixture.execution, fixture.parameters, fixture.modes, fixture.sharp,
      fixture.theta, fixture.input, fixture.input_stamps,
      ContractFixture::generation, teuk::ReductionEvolution::FreeDamped, 0.0);

  using StrideJet =
      Kokkos::View<C****, Kokkos::LayoutStride, teuk::MemorySpace>;
  using StrideStamp = Kokkos::View<std::uint64_t****,
                                   Kokkos::LayoutStride, teuk::MemorySpace>;
  const Kokkos::LayoutStride jet_layout(
      2, 4 * (ContractFixture::radial_count * 2 + 3) + 5, 4,
      ContractFixture::radial_count * 2 + 3,
      ContractFixture::radial_count, 2, 1, 1);
  StrideJet strided_jet("contract_full_strided_jet", jet_layout);
  StrideStamp strided_jet_stamps("contract_full_strided_jet_stamps",
                                 jet_layout);
  StrideStamp strided_angular_stamps(
      "contract_full_strided_angular_stamps", jet_layout);
  auto right_jet = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        fixture.jet);
  auto strided_jet_host = Kokkos::create_mirror_view(strided_jet);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t order = 0; order < 4; ++order) {
      for (std::size_t radial = 0; radial < ContractFixture::radial_count;
           ++radial) {
        strided_jet_host(mode, order, radial, 0) =
            right_jet(mode, order, radial, 0);
      }
    }
  }
  Kokkos::deep_copy(fixture.execution, strided_jet, strided_jet_host);
  Kokkos::deep_copy(fixture.execution, strided_jet_stamps,
                    ContractFixture::generation);

  fixture.pass1();
  auto token = strided_tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, strided_angular_stamps, token);
  strided_tower.pass1(
      fixture.execution, strided_jet, strided_jet_stamps, strided_jet,
      strided_angular_stamps, ContractFixture::generation, token,
      teuk::ReductionEvolution::FreeDamped, 0.0);

  token = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token);
  fixture.tower.pass2(
      fixture.execution, fixture.jet, fixture.angular_stamps,
      ContractFixture::generation, token,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto strided_token2 = strided_tower.expected_pass_token();
  CHECK(strided_token2 == token);
  Kokkos::deep_copy(fixture.execution, strided_angular_stamps,
                    strided_token2);
  strided_tower.pass2(
      fixture.execution, strided_jet, strided_angular_stamps,
      ContractFixture::generation, strided_token2,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  fixture.execution.fence("finish Route-B strided pass2 parity");
  const auto right_pass2 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.tower.next_coefficients());
  const auto strided_pass2 = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, strided_tower.next_coefficients());
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t field = 0; field < 6; ++field) {
      for (std::size_t order = 0; order < 4; ++order) {
        for (std::size_t radial = 0; radial < ContractFixture::radial_count;
             ++radial) {
          CHECK_COMPLEX_NEAR(strided_pass2(mode, field, order, radial, 0),
                             right_pass2(mode, field, order, radial, 0), 0.0);
        }
      }
    }
  }

  using StridePass3 =
      Kokkos::View<C*****, Kokkos::LayoutStride, teuk::MemorySpace>;
  using StridePass3Stamp = Kokkos::View<std::uint64_t*****,
                                        Kokkos::LayoutStride,
                                        teuk::MemorySpace>;
  const Kokkos::LayoutStride pass3_layout(
      2, 4 * (4 * (ContractFixture::radial_count * 2 + 3) + 5) + 7, 4,
      4 * (ContractFixture::radial_count * 2 + 3) + 5, 4,
      ContractFixture::radial_count * 2 + 3,
      ContractFixture::radial_count, 2, 1, 1);
  StridePass3 strided_pass3("contract_full_strided_pass3", pass3_layout);
  StridePass3Stamp strided_pass3_stamps(
      "contract_full_strided_pass3_stamps", pass3_layout);
  auto right_pass3 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         fixture.pass3);
  auto strided_pass3_host = Kokkos::create_mirror_view(strided_pass3);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t slot = 0; slot < 4; ++slot) {
      for (std::size_t order = 0; order < 4; ++order) {
        for (std::size_t radial = 0; radial < ContractFixture::radial_count;
             ++radial) {
          strided_pass3_host(mode, slot, order, radial, 0) =
              right_pass3(mode, slot, order, radial, 0);
        }
      }
    }
  }
  Kokkos::deep_copy(fixture.execution, strided_pass3, strided_pass3_host);

  token = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.pass3_stamps, token);
  fixture.tower.pass3(
      fixture.execution, fixture.pass3, fixture.pass3_stamps,
      ContractFixture::generation, token,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto strided_token3 = strided_tower.expected_pass_token();
  CHECK(strided_token3 == token);
  Kokkos::deep_copy(fixture.execution, strided_pass3_stamps,
                    strided_token3);
  strided_tower.pass3(
      fixture.execution, strided_pass3, strided_pass3_stamps,
      ContractFixture::generation, strided_token3,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  fixture.execution.fence("finish Route-B strided pass3 parity");
  const auto right_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, fixture.tower.values());
  const auto strided_values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, strided_tower.values());
  for (std::size_t mode = 0; mode < 2; ++mode) {
    for (std::size_t field = 0; field < 7; ++field) {
      for (std::size_t radial = 0; radial < ContractFixture::radial_count;
           ++radial) {
        CHECK_COMPLEX_NEAR(strided_values(1, mode, field, radial, 0),
                           right_values(1, mode, field, radial, 0), 0.0);
      }
    }
  }
}

TEST_CASE("Route-B reconstruction rejects malformed advance views") {
  ContractFixture fixture;
  fixture.initialize();
  const auto token1 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.jet_stamps,
                    ContractFixture::generation);
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token1);
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> wrong4(
      "contract_wrong4", 2, 3, ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.pass1(
        fixture.execution, wrong4, fixture.jet_stamps, fixture.jet,
        fixture.angular_stamps, ContractFixture::generation, token1,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  using Null4 = Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Null4 null4(static_cast<C*>(nullptr), 2, 4, ContractFixture::radial_count,
              1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.pass1(
        fixture.execution, null4, fixture.jet_stamps, fixture.jet,
        fixture.angular_stamps, ContractFixture::generation, token1,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  Null4 alias4(const_cast<C*>(fixture.tower.current_coefficients().data()), 2,
               4, ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.pass1(
        fixture.execution, alias4, fixture.jet_stamps, fixture.jet,
        fixture.angular_stamps, ContractFixture::generation, token1,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  fixture.pass1();
  const auto token2 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.angular_stamps, token2);
  fixture.tower.pass2(
      fixture.execution, fixture.jet, fixture.angular_stamps,
      ContractFixture::generation, token2,
      teuk::ReductionEvolution::FreeDamped, 0.0);
  const auto token3 = fixture.tower.expected_pass_token();
  Kokkos::deep_copy(fixture.execution, fixture.pass3_stamps, token3);
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> wrong5(
      "contract_wrong5", 2, 3, 4, ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.pass3(
        fixture.execution, wrong5, fixture.pass3_stamps,
        ContractFixture::generation, token3,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  using Null5 = Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  Null5 null5(static_cast<C*>(nullptr), 2, 4, 4,
              ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.pass3(
        fixture.execution, null5, fixture.pass3_stamps,
        ContractFixture::generation, token3,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
  Null5 alias5(const_cast<C*>(fixture.tower.values().data()), 2, 4, 4,
               ContractFixture::radial_count, 1);
  CHECK(throws_invalid_argument([&] {
    fixture.tower.pass3(
        fixture.execution, alias5, fixture.pass3_stamps,
        ContractFixture::generation, token3,
        teuk::ReductionEvolution::FreeDamped, 0.0);
  }));
}

}  // namespace
