#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <array>
#include <iostream>
#include <limits>

#include "teuk/routeb_angular_jet_coordinator.hpp"
#include "routeb_angular_jet_fixture.hpp"

namespace {

int routeb_angular_allocations = 0;
int routeb_angular_fences = 0;

void count_routeb_angular_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                     const void*, std::uint64_t) {
  ++routeb_angular_allocations;
}

void count_routeb_angular_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++routeb_angular_fences;
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

constexpr std::array<int, 10> closed_field_spins{
    -2, -2, -2, -1, -2, 0, -2, -1, -1, 0};

teuk::Complex closed_profile(const std::size_t kind,
                             const std::size_t mode_index, const int ell,
                             const double radius) {
  const teuk::Complex amplitude(
      0.11 + 0.013 * static_cast<double>(kind + 1) +
          0.017 * static_cast<double>(mode_index) + 0.009 * ell,
      -0.07 + 0.006 * static_cast<double>(kind) -
          0.011 * static_cast<double>(mode_index));
  const double alpha =
      0.47 + 0.021 * static_cast<double>(kind) + 0.013 * ell;
  const teuk::Complex oscillatory =
      teuk::Complex(0.015, -0.008) * static_cast<double>(kind + 1);
  const double beta =
      1.31 + 0.019 * static_cast<double>(kind) + 0.023 * ell;
  return amplitude * std::exp(alpha * radius) +
         oscillatory * std::sin(beta * radius);
}

struct ClosedOracleErrors {
  std::array<std::array<std::array<double, 3>, 2>, 10> h4{};
  std::array<std::array<std::array<std::array<double, 3>, 2>, 10>, 5>
      levels{};
  std::array<teuk::Complex, 2> signed_probe{};
};

ClosedOracleErrors run_closed_oracle(const std::size_t radial_count) {
  constexpr int node_count = routeb_angular_fixture::node_count;
  constexpr int ell_max = 5;
  constexpr std::uint64_t generation = 0x4ab712;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const double horizon = 1.4 * 1.4 / (1.0 + std::sqrt(1.0 - 0.63 * 0.63));
  const teuk::UniformRadialGrid grid(radial_count, 0.0, horizon);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.reduction_damping = 0.31;
  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, ell_max, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("closed_oracle_primary", 2, 3, radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("closed_oracle_primary_stamps", 2, radial_count,
                     node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction("closed_oracle_reconstruction", 2, 7, radial_count,
                     node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("closed_oracle_reconstruction_stamps", 2,
                            radial_count, node_count);
  const auto angular_grid = teuk::angular::gauss_legendre(node_count);
  auto hp = Kokkos::create_mirror_view(primary);
  auto hr = Kokkos::create_mirror_view(reconstruction);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    const int m = registry.modes()[mode];
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double radius = grid.coordinate(radial);
      for (int node = 0; node < node_count; ++node) {
        const double theta = angular_grid.theta(node);
        for (std::size_t field = 0; field < 10; ++field) {
          const int spin = closed_field_spins[field];
          const int ell_min = std::max(std::abs(spin), std::abs(m));
          teuk::Complex value{};
          for (int ell = ell_min; ell <= std::min(ell_max, ell_min + 2);
               ++ell) {
            value += closed_profile(field, mode, ell, radius) *
                     teuk::angular::spin_weighted_harmonic_theta(
                         ell, m, spin, theta);
          }
          if (field < 3) {
            hp(mode, field, radial, node) = value;
          } else {
            hr(mode, field - 3, radial, node) = value;
          }
        }
      }
    }
  }
  Kokkos::deep_copy(execution, primary, hp);
  Kokkos::deep_copy(execution, reconstruction, hr);
  Kokkos::deep_copy(execution, primary_stamps, generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, generation);
  coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                         reconstruction_stamps, generation);
  coordinator.advance_to_h4(execution, generation);
  execution.fence("finish independent closed Route-B oracle");
  const auto pv = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.primary_values());
  const auto rv = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.reconstruction_values());
  ClosedOracleErrors errors;
  for (std::size_t level = 0; level < 5; ++level) {
    for (std::size_t sample = 0; sample < 3; ++sample) {
      const std::size_t radial =
          sample == 0 ? 0 : (sample == 1 ? (radial_count - 1) / 2
                                         : radial_count - 1);
      for (std::size_t mode = 0; mode < 2; ++mode) {
        for (std::size_t field = 0; field < 10; ++field) {
          for (int node = 0; node < node_count; ++node) {
            const teuk::Complex actual =
                field < 3 ? pv(level, mode, field, radial, node)
                          : rv(level, mode, field - 3, radial, node);
            const double error = Kokkos::abs(
                actual - routeb_angular_fixture::sample_values
                             [level][sample][mode][field][node]);
            errors.levels[level][field][mode][sample] =
                std::max(errors.levels[level][field][mode][sample], error);
            if (level == 4)
              errors.h4[field][mode][sample] =
                  std::max(errors.h4[field][mode][sample], error);
            if (level == 4 && sample == 1 && field == 0 && node == 3)
              errors.signed_probe[mode] = actual;
          }
        }
      }
    }
  }
  return errors;
}

TEST_CASE("Route-B angular jet coordinator closes a zero signed-mode tower") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  constexpr std::uint64_t generation = 1701;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.reduction_damping = 0.31;
  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, 5, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("routeb_closed_primary_h0", registry.size(), 3, radial_count,
              node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("routeb_closed_primary_h0_stamps", registry.size(),
                     radial_count, node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction("routeb_closed_reconstruction_h0", registry.size(), 7,
                     radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("routeb_closed_reconstruction_h0_stamps",
                            registry.size(), radial_count, node_count);
  Kokkos::deep_copy(execution, primary, teuk::Complex{});
  Kokkos::deep_copy(execution, reconstruction, teuk::Complex{});
  Kokkos::deep_copy(execution, primary_stamps, generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, generation);
  coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                         reconstruction_stamps, generation);
  coordinator.advance_to_h4(execution, generation);
  execution.fence("finish zero Route-B closed angular tower");
  CHECK(coordinator.current_level() == 4);
  const auto primary_level_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.primary_stamps());
  const auto reconstruction_level_stamps =
      Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, coordinator.reconstruction_stamps());
  for (std::size_t level = 0; level < 5; ++level) {
    for (std::size_t mode = 0; mode < registry.size(); ++mode) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (int node = 0; node < node_count; ++node) {
          CHECK(primary_level_stamps(level, mode, radial, node) == generation);
          CHECK(reconstruction_level_stamps(level, mode, radial, node) ==
                generation);
        }
      }
    }
  }
}

TEST_CASE("Route-B angular jet coordinator closes rotating signed modal data") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  constexpr int ell_max = 5;
  constexpr std::uint64_t generation = 1702;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  parameters.reduction_damping = 0.31;
  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, ell_max, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("routeb_closed_modal_primary_h0", registry.size(), 3,
              radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("routeb_closed_modal_primary_h0_stamps", registry.size(),
                     radial_count, node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction("routeb_closed_modal_reconstruction_h0", registry.size(),
                     7, radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("routeb_closed_modal_reconstruction_h0_stamps",
                            registry.size(), radial_count, node_count);
  const auto angular_grid = teuk::angular::gauss_legendre(node_count);
  auto hp = Kokkos::create_mirror_view(primary);
  auto hr = Kokkos::create_mirror_view(reconstruction);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    const int m = registry.modes()[mode];
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double radius = grid.coordinate(radial);
      for (int node = 0; node < node_count; ++node) {
        const double theta = angular_grid.theta(node);
        const double primary_harmonic =
            teuk::angular::spin_weighted_harmonic_theta(2, m, -2, theta);
        for (std::size_t field = 0; field < 3; ++field) {
          hp(mode, field, radial, node) =
              teuk::Complex(0.13 + 0.02 * field + 0.01 * mode,
                            -0.07 + 0.03 * field) *
              std::exp((0.4 + 0.09 * field) * radius) * primary_harmonic;
        }
        for (std::size_t field = 0; field < 7; ++field) {
          const int spin = teuk::reconstruction_metadata[field].spin;
          const int ell = std::max({std::abs(spin), std::abs(m), 2});
          const double harmonic = teuk::angular::spin_weighted_harmonic_theta(
              ell, m, spin, theta);
          hr(mode, field, radial, node) =
              teuk::Complex(0.08 + 0.017 * field + 0.013 * mode,
                            -0.05 + 0.011 * field) *
              std::exp((0.31 + 0.037 * field) * radius) * harmonic;
        }
      }
    }
  }
  Kokkos::deep_copy(execution, primary, hp);
  Kokkos::deep_copy(execution, reconstruction, hr);
  Kokkos::deep_copy(execution, primary_stamps, generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, generation);
  coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                         reconstruction_stamps, generation);
  coordinator.advance_to_h4(execution, generation);
  execution.fence("finish rotating Route-B closed angular tower");
  const auto values = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.reconstruction_values());
  double maximum = 0.0;
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < 7; ++field) {
      for (std::size_t radial = 0; radial < radial_count; ++radial) {
        for (int node = 0; node < node_count; ++node) {
          const auto value = values(4, mode, field, radial, node);
          CHECK(std::isfinite(value.real()));
          CHECK(std::isfinite(value.imag()));
          maximum = std::max(maximum, Kokkos::abs(value));
        }
      }
    }
  }
  CHECK(maximum > 1.0e-10);
}

TEST_CASE("Route-B angular coordinator preflight is atomic across towers") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  constexpr std::uint64_t generation = 1711;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, 5, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("routeb_atomic_primary", 2, 3, radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("routeb_atomic_primary_stamps", 2, radial_count,
                     node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("routeb_atomic_reconstruction_stamps", 2,
                            radial_count, node_count);
  using AliasedReconstruction = Kokkos::View<
      const teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  const auto owned_primary = coordinator.primary_values();
  AliasedReconstruction reconstruction_alias(owned_primary.data(), 2, 7,
                                             radial_count, node_count);
  Kokkos::deep_copy(execution, primary, teuk::Complex{});
  Kokkos::deep_copy(execution, primary_stamps, generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, generation);
  CHECK(throws_invalid_argument([&] {
    coordinator.initialize(execution, primary, primary_stamps,
                           reconstruction_alias, reconstruction_stamps,
                           generation);
  }));
  CHECK(coordinator.generation() == 0);
  CHECK(coordinator.current_level() == 0);
  const auto pristine_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.primary_stamps());
  for (std::size_t level = 0; level < 5; ++level)
    for (std::size_t mode = 0; mode < 2; ++mode)
      for (std::size_t radial = 0; radial < radial_count; ++radial)
        for (int node = 0; node < node_count; ++node)
          CHECK(pristine_stamps(level, mode, radial, node) == 0);

  const auto owned_coefficients = coordinator.primary_current_coefficients();
  AliasedReconstruction coefficient_alias(owned_coefficients.data(), 2, 7,
                                           radial_count, node_count);
  CHECK(throws_invalid_argument([&] {
    coordinator.initialize(execution, primary, primary_stamps,
                           coefficient_alias, reconstruction_stamps,
                           generation);
  }));
  CHECK(coordinator.generation() == 0);
}

TEST_CASE("Route-B angular coordinator preserves projected h0 fourth jets") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  constexpr std::uint64_t generation = 1712;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, 5, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("routeb_d4_primary", 2, 3, radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("routeb_d4_primary_stamps", 2, radial_count, node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction("routeb_d4_reconstruction", 2, 7, radial_count,
                     node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("routeb_d4_reconstruction_stamps", 2,
                            radial_count, node_count);
  const auto angular_grid = teuk::angular::gauss_legendre(node_count);
  auto hp = Kokkos::create_mirror_view(primary);
  auto hr = Kokkos::create_mirror_view(reconstruction);
  for (std::size_t mode = 0; mode < 2; ++mode) {
    const int m = registry.modes()[mode];
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double radius = grid.coordinate(radial);
      for (int node = 0; node < node_count; ++node) {
        const double theta = angular_grid.theta(node);
        const double y2 = teuk::angular::spin_weighted_harmonic_theta(
            2, m, -2, theta);
        for (std::size_t field = 0; field < 3; ++field)
          hp(mode, field, radial, node) =
              teuk::Complex(0.2 + 0.03 * field, 0.01 * mode) *
              std::exp((0.33 + 0.04 * field) * radius) * y2;
        for (std::size_t field = 0; field < 7; ++field) {
          const int spin = teuk::reconstruction_metadata[field].spin;
          const int ell = std::max({2, std::abs(spin), std::abs(m)});
          const double y = teuk::angular::spin_weighted_harmonic_theta(
              ell, m, spin, theta);
          hr(mode, field, radial, node) =
              teuk::Complex(0.1 + 0.02 * field, -0.01 * mode) *
              std::exp((0.29 + 0.025 * field) * radius) * y;
        }
      }
    }
  }
  Kokkos::deep_copy(execution, primary, hp);
  Kokkos::deep_copy(execution, reconstruction, hr);
  Kokkos::deep_copy(execution, primary_stamps, generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, generation);
  coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                         reconstruction_stamps, generation);
  execution.fence("inspect projected h0 fourth jets");
  const auto pc = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.primary_current_coefficients());
  const auto ps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.primary_current_coefficient_stamps());
  const auto rc = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.reconstruction_current_coefficients());
  const auto rs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{},
      coordinator.reconstruction_current_coefficient_stamps());
  CHECK(Kokkos::abs(pc(0, 0, 4, 0, 0)) > 1.0e-10);
  CHECK(ps(0, 0, 4, 0, 0) == generation);
  for (std::size_t field = 0; field < 7; ++field) {
    CHECK(Kokkos::abs(rc(0, field, 4, 0, 0)) > 1.0e-12);
    CHECK(rs(0, field, 4, 0, 0) == generation);
  }
}

TEST_CASE("Route-B angular coordinator hot initialize and advance do not allocate or fence") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  constexpr std::uint64_t generation = 1713;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, 5, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("routeb_hot_primary", 2, 3, radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("routeb_hot_primary_stamps", 2, radial_count, node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction("routeb_hot_reconstruction", 2, 7, radial_count,
                     node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("routeb_hot_reconstruction_stamps", 2,
                            radial_count, node_count);
  Kokkos::deep_copy(execution, primary, teuk::Complex{});
  Kokkos::deep_copy(execution, reconstruction, teuk::Complex{});
  Kokkos::deep_copy(execution, primary_stamps, generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, generation);
  execution.fence("warm Route-B angular coordinator audit");
  const auto* primary_pointer = coordinator.primary_values().data();
  const auto* reconstruction_pointer =
      coordinator.reconstruction_values().data();
  routeb_angular_allocations = 0;
  routeb_angular_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_routeb_angular_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(
      count_routeb_angular_fence);
  coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                         reconstruction_stamps, generation);
  coordinator.advance_to_h4(execution, generation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  execution.fence("finish Route-B angular coordinator audit");
  CHECK(routeb_angular_allocations == 0);
  CHECK(routeb_angular_fences == 0);
  CHECK(coordinator.primary_values().data() == primary_pointer);
  CHECK(coordinator.reconstruction_values().data() == reconstruction_pointer);
}

TEST_CASE("Route-B closed angular graph matches independent rotating oracle") {
  const auto coarse = run_closed_oracle(9);
  const auto medium = run_closed_oracle(17);
  const auto fine = run_closed_oracle(33);
  const auto ceiling = run_closed_oracle(65);
  constexpr std::array<const char*, 10> names{
      "P", "Q", "Psi", "G", "Lambda", "H", "B", "Pi", "C", "U"};
  for (std::size_t field = 0; field < names.size(); ++field) {
    double max17 = 0.0;
    double max33 = 0.0;
    double max65 = 0.0;
    for (std::size_t mode = 0; mode < 2; ++mode) {
      for (std::size_t sample = 0; sample < 3; ++sample) {
        const double ec = coarse.h4[field][mode][sample];
        const double em = medium.h4[field][mode][sample];
        const double ef = fine.h4[field][mode][sample];
        const double e65 = ceiling.h4[field][mode][sample];
        const double ratio1 = ec / em;
        const double ratio2 = em / ef;
        max17 = std::max(max17, em);
        max33 = std::max(max33, ef);
        max65 = std::max(max65, e65);
        std::cout << "Route-B closed h4 " << names[field] << " mode "
                  << mode << " sample " << sample << " errors " << ec << ' '
                  << em << ' ' << ef << " ratios " << ratio1 << ' ' << ratio2
                  << " ceiling " << ef / e65 << '\n';
        CHECK(std::isfinite(ec));
        CHECK(std::isfinite(em));
        CHECK(std::isfinite(ef));
        // At R=0 the h4 U recurrence contains no numerically differentiated
        // radial term: every R-weighted term vanishes and the remaining
        // angular/pass algebra agrees at binary64 roundoff on every grid.
        const bool exact_scri_u = field == 9 && sample == 0;
        if (exact_scri_u) {
          CHECK(ec < 2.0e-15);
          CHECK(em < 2.0e-15);
          CHECK(ef < 2.0e-15);
        } else {
          CHECK(ec > em);
          CHECK(em > ef);
          CHECK(ratio1 > 15.0);
          CHECK(ratio2 > 15.0);
        }
        CHECK(ef < 2.0e-6);
      }
    }
    std::cout << "Route-B closed ceiling max " << names[field]
              << " N17/N33/N65 " << max17 << ' ' << max33 << ' ' << max65
              << " ratios " << max17 / max33 << ' ' << max33 / max65
              << '\n';
  }
  for (std::size_t level = 0; level < 5; ++level)
    for (std::size_t field = 0; field < 10; ++field)
      for (std::size_t mode = 0; mode < 2; ++mode)
        for (std::size_t sample = 0; sample < 3; ++sample) {
          CHECK(std::isfinite(fine.levels[level][field][mode][sample]));
          CHECK(fine.levels[level][field][mode][sample] < 2.0e-6);
        }
  CHECK(Kokkos::abs(fine.signed_probe[0] - fine.signed_probe[1]) > 1.0e-5);
}

TEST_CASE("Route-B closed angular graph never resurrects poisoned inputs") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  constexpr std::uint64_t generation = 1721;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  auto run = [&](const bool poison_primary) {
    teuk::RouteBAngularJetCoordinator coordinator(
        execution, registry, grid, 5, node_count, parameters,
        poison_primary ? "routeb_primary_poison" : "routeb_sharp_poison");
    Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
        primary("routeb_poison_primary_input", 2, 3, radial_count,
                node_count);
    Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
        primary_stamps("routeb_poison_primary_stamps", 2, radial_count,
                       node_count);
    Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
        reconstruction("routeb_poison_reconstruction_input", 2, 7,
                       radial_count, node_count);
    Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
        reconstruction_stamps("routeb_poison_reconstruction_stamps", 2,
                              radial_count, node_count);
    Kokkos::deep_copy(execution, primary, teuk::Complex(0.2, -0.1));
    Kokkos::deep_copy(execution, reconstruction, teuk::Complex(0.1, 0.03));
    Kokkos::deep_copy(execution, primary_stamps, generation);
    Kokkos::deep_copy(execution, reconstruction_stamps, generation);
    if (poison_primary) {
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       primary_stamps);
      host(0, 0, 0) = generation - 1;
      Kokkos::deep_copy(execution, primary_stamps, host);
    } else {
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       reconstruction);
      // Poison the negative-m partner used by sharp(B,Pi,C) for +m.
      host(0, 5, 0, 0) = teuk::Complex(
          std::numeric_limits<double>::quiet_NaN(), 0.0);
      Kokkos::deep_copy(execution, reconstruction, host);
    }
    coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                           reconstruction_stamps, generation);
    coordinator.advance_to_h4(execution, generation);
    execution.fence("inspect Route-B poisoned graph provenance");
    const auto ps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, coordinator.primary_stamps());
    const auto rs = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, coordinator.reconstruction_stamps());
    for (std::size_t level = 0; level < 5; ++level)
      for (std::size_t mode = 0; mode < 2; ++mode)
        for (std::size_t radial = 0; radial < radial_count; ++radial)
          for (int node = 0; node < node_count; ++node) {
            if (poison_primary) CHECK(ps(level, mode, radial, node) == 0);
            if (!poison_primary) CHECK(rs(level, mode, radial, node) == 0);
            if (poison_primary && level > 0)
              CHECK(rs(level, mode, radial, node) == 0);
          }
  };
  run(true);
  run(false);
}

TEST_CASE("Route-B Kerr GHP jets retain radial coefficients of constant fields") {
  constexpr int node_count = 7;
  const teuk::ExecutionSpace execution;
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      field("routeb_constant_ghp_field", 1, 4, 1, node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      dt("routeb_constant_ghp_dt", 1, 4, 1, node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      pure("routeb_constant_ghp_pure", 1, 4, 1, node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      output("routeb_constant_ghp_output", 1, 4, 1, node_count);
  Kokkos::View<double*, teuk::MemorySpace> radius("routeb_constant_radius", 1);
  Kokkos::View<double*, teuk::MemorySpace> sine("routeb_constant_sine",
                                                node_count);
  Kokkos::View<double*, teuk::MemorySpace> cosine("routeb_constant_cosine",
                                                  node_count);
  const auto grid = teuk::angular::gauss_legendre(node_count);
  auto hf = Kokkos::create_mirror_view(field);
  auto hp = Kokkos::create_mirror_view(pure);
  auto hs = Kokkos::create_mirror_view(sine);
  auto hc = Kokkos::create_mirror_view(cosine);
  for (int node = 0; node < node_count; ++node) {
    const double theta = grid.theta(node);
    hf(0, 0, 0, node) =
        teuk::angular::spin_weighted_harmonic_theta(2, 2, -2, theta);
    hp(0, 0, 0, node) =
        2.0 * teuk::angular::spin_weighted_harmonic_theta(2, 2, -1, theta);
    hs(node) = std::sin(theta);
    hc(node) = std::cos(theta);
  }
  Kokkos::deep_copy(execution, field, hf);
  Kokkos::deep_copy(execution, dt, teuk::Complex{});
  Kokkos::deep_copy(execution, pure, hp);
  Kokkos::deep_copy(execution, output, teuk::Complex{});
  Kokkos::deep_copy(execution, radius, 0.0);
  Kokkos::deep_copy(execution, sine, hs);
  Kokkos::deep_copy(execution, cosine, hc);
  using Functor = teuk::routeb_angular_jet_detail::ApplyGhpJetFunctor;
  Kokkos::parallel_for(
      "routeb_constant_field_ghp_jet",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, node_count),
      Functor{field.data(), dt.data(), pure.data(), output.data(),
              radius.data(),
              {field.stride(0), field.stride(1), field.stride(2),
               field.stride(3)},
              {dt.stride(0), dt.stride(1), dt.stride(2), dt.stride(3)},
              {pure.stride(0), pure.stride(1), pure.stride(2), pure.stride(3)},
              {output.stride(0), output.stride(1), output.stride(2),
               output.stride(3)},
              1, node_count, 4, -2, -2, 0.63, 1.4, sine.data(),
              cosine.data(), false});
  execution.fence("inspect constant-field Kerr GHP jet");
  const auto ho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       output);
  for (int node = 0; node < node_count; ++node) {
    const teuk::Complex value = hf(0, 0, 0, node);
    const teuk::Complex pure0 = hp(0, 0, 0, node);
    const double l2 = 1.4 * 1.4;
    const teuk::Complex expected =
        (teuk::Complex(0.0, 0.63 * hc(node)) * pure0 /
             (l2 * l2) +
         teuk::Complex(0.0, 4.0 * 0.63 * hs(node)) * value /
             (l2 * l2)) /
        std::sqrt(2.0);
    CHECK_COMPLEX_NEAR(ho(0, 1, 0, node), expected, 2.0e-14);
  }
  CHECK(Kokkos::abs(ho(0, 1, 0, node_count / 2)) > 1.0e-4);
}

TEST_CASE("Route-B angular coordinator rejects configuration before mutation") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  CHECK(throws_invalid_argument([&] {
    teuk::RouteBAngularJetCoordinator bad(execution, registry, grid, 5, 0,
                                          parameters);
  }));
  CHECK(throws_invalid_argument([&] {
    teuk::RouteBAngularJetCoordinator bad(execution, registry, grid, 5, -1,
                                          parameters);
  }));
  CHECK(throws_invalid_argument([&] {
    teuk::RouteBAngularJetCoordinator bad(execution, registry, grid, 1,
                                          node_count, parameters);
  }));
  auto bad_parameters = parameters;
  bad_parameters.spin = 1.01;
  CHECK(throws_invalid_argument([&] {
    teuk::RouteBAngularJetCoordinator bad(execution, registry, grid, 5,
                                          node_count, bad_parameters);
  }));
  const teuk::ModeRegistry unclosed({2});
  CHECK(throws_invalid_argument([&] {
    teuk::RouteBAngularJetCoordinator bad(execution, unclosed, grid, 5,
                                          node_count, parameters);
  }));
  const teuk::ModeRegistry outside_band({-6, 6});
  CHECK(throws_invalid_argument([&] {
    teuk::RouteBAngularJetCoordinator bad(execution, outside_band, grid, 5,
                                          node_count, parameters);
  }));

  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, 5, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("routeb_bad_generation_primary", 2, 3, radial_count,
              node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("routeb_bad_generation_primary_stamps", 2, radial_count,
                     node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction("routeb_bad_generation_reconstruction", 2, 7,
                     radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("routeb_bad_generation_reconstruction_stamps", 2,
                            radial_count, node_count);
  constexpr std::uint64_t invalid_generation = 0x6eed0e9da4d94a4fULL;
  Kokkos::deep_copy(execution, primary_stamps, invalid_generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, invalid_generation);
  CHECK(throws_invalid_argument([&] {
    coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                           reconstruction_stamps, invalid_generation);
  }));
  CHECK(coordinator.generation() == 0);
  const auto stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.primary_stamps());
  CHECK(stamps(0, 0, 0, 0) == 0);
}

TEST_CASE("Route-B sharp partner stamp poison cannot survive pass3") {
  constexpr std::size_t radial_count = 9;
  constexpr int node_count = 7;
  constexpr std::uint64_t generation = 1722;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.7);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.63;
  parameters.compactification_length = 1.4;
  parameters.spin_weight = -2;
  teuk::RouteBAngularJetCoordinator coordinator(
      execution, registry, grid, 5, node_count, parameters);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      primary("routeb_sharp_stamp_primary", 2, 3, radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      primary_stamps("routeb_sharp_stamp_primary_stamps", 2, radial_count,
                     node_count);
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction("routeb_sharp_stamp_reconstruction", 2, 7,
                     radial_count, node_count);
  Kokkos::View<std::uint64_t***, Kokkos::LayoutRight, teuk::MemorySpace>
      reconstruction_stamps("routeb_sharp_stamp_reconstruction_stamps", 2,
                            radial_count, node_count);
  Kokkos::deep_copy(execution, primary, teuk::Complex(0.2, -0.1));
  Kokkos::deep_copy(execution, reconstruction, teuk::Complex(0.1, 0.03));
  Kokkos::deep_copy(execution, primary_stamps, generation);
  Kokkos::deep_copy(execution, reconstruction_stamps, generation);
  coordinator.initialize(execution, primary, primary_stamps, reconstruction,
                         reconstruction_stamps, generation);
  const auto const_stamps = coordinator.reconstruction_current_coefficient_stamps();
  using WritableStamps = Kokkos::View<
      std::uint64_t*****, Kokkos::LayoutRight, teuk::MemorySpace,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  WritableStamps writable(const_cast<std::uint64_t*>(const_stamps.data()), 2,
                          7, 5, radial_count, node_count);
  Kokkos::parallel_for(
      "poison_routeb_sharp_partner_stamp",
      Kokkos::RangePolicy<teuk::ExecutionSpace>(execution, 0, 1),
      KOKKOS_LAMBDA(const std::size_t) {
        // -m B is the sharp partner for the +m pass3 eth-prime input.
        writable(0, 3, 0, 0, 0) = 0;
      });
  coordinator.advance_one_level(execution, generation);
  execution.fence("inspect sharp partner stamp poison");
  const auto level_stamps = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coordinator.reconstruction_stamps());
  for (std::size_t mode = 0; mode < 2; ++mode)
    for (std::size_t radial = 0; radial < radial_count; ++radial)
      for (int node = 0; node < node_count; ++node)
        CHECK(level_stamps(1, mode, radial, node) == 0);
}

}  // namespace
