#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "test_harness.hpp"
#include "teuk/angular.hpp"
#include "teuk/angular_coordinator.hpp"
#include "teuk/ghp.hpp"

namespace {

int coordinator_copies = 0;
int coordinator_allocations = 0;

void count_coordinator_copy(Kokkos::Tools::SpaceHandle, const char*,
                            const void*, Kokkos::Tools::SpaceHandle,
                            const char*, const void*, std::uint64_t) {
  ++coordinator_copies;
}

void count_coordinator_allocation(Kokkos::Tools::SpaceHandle, const char*,
                                  const void*, std::uint64_t) {
  ++coordinator_allocations;
}

std::vector<teuk::Complex> align_modal(
    const teuk::angular::SpinWeightedTransform& source,
    const std::vector<teuk::Complex>& operated,
    const teuk::angular::SpinWeightedTransform& destination) {
  std::vector<teuk::Complex> result(destination.mode_count(),
                                    teuk::Complex(0.0, 0.0));
  for (int ell = source.ell_min(); ell <= source.ell_max(); ++ell) {
    if (ell >= destination.ell_min()) {
      result[static_cast<std::size_t>(ell - destination.ell_min())] =
          operated[static_cast<std::size_t>(ell - source.ell_min())];
    }
  }
  return result;
}

}  // namespace

TEST_CASE("signed-mode angular coordinator matches every host plan") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using coordinator_type = teuk::SignedModeAngularCoordinator<execution_space>;
  using memory_space = execution_space::memory_space;
  using field_view = Kokkos::View<teuk::Complex****, Kokkos::LayoutRight,
                                  memory_space>;
  using real_view = Kokkos::View<teuk::Real*, memory_space>;
  constexpr int spin = -1;
  constexpr int boost = 1;
  constexpr int ell_max = 5;
  constexpr int node_count = 9;
  constexpr std::size_t radial_count = 3;
  constexpr std::size_t field_count = 5;
  const teuk::ModeRegistry registry({2, -2, 0, 1, -1});
  const teuk::KerrParameters parameters{1.0, 0.76, 1.25};
  const execution_space execution;
  coordinator_type coordinator(execution, registry, spin, boost, ell_max,
                               node_count, radial_count, parameters);

  field_view fields("coordinator_fields", registry.size(), field_count,
                    radial_count, node_count);
  auto host_fields = Kokkos::create_mirror_view(fields);
  std::vector<std::vector<std::vector<teuk::Complex>>> modal(registry.size());
  std::vector<std::vector<std::vector<teuk::Complex>>> dt_modal(
      registry.size());
  for (std::size_t mode_index = 0; mode_index < registry.size(); ++mode_index) {
    const int m = registry.modes()[mode_index];
    const teuk::angular::SpinWeightedTransform transform(
        spin, m, ell_max, node_count);
    modal[mode_index].resize(radial_count);
    dt_modal[mode_index].resize(radial_count);
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      modal[mode_index][radial].resize(transform.mode_count());
      dt_modal[mode_index][radial].resize(transform.mode_count());
      for (std::size_t ell = 0; ell < transform.mode_count(); ++ell) {
        modal[mode_index][radial][ell] = teuk::Complex(
            0.1 + 0.2 * mode_index + 0.3 * radial + 0.07 * ell,
            -0.4 + 0.05 * mode_index - 0.08 * radial + 0.03 * ell);
        dt_modal[mode_index][radial][ell] = teuk::Complex(
            -0.2 + 0.1 * mode_index - 0.04 * radial + 0.02 * ell,
            0.3 - 0.06 * mode_index + 0.09 * radial - 0.01 * ell);
      }
      const auto nodal = transform.synthesize(modal[mode_index][radial]);
      const auto dt_nodal =
          transform.synthesize(dt_modal[mode_index][radial]);
      for (int node = 0; node < node_count; ++node) {
        const auto i = static_cast<std::size_t>(node);
        host_fields(mode_index, 0, radial, i) = nodal[i];
        host_fields(mode_index, 1, radial, i) = dt_nodal[i];
      }
    }
  }

  real_view radius("coordinator_radius", radial_count);
  real_view sin_theta("coordinator_sin", node_count);
  real_view cos_theta("coordinator_cos", node_count);
  auto host_radius = Kokkos::create_mirror_view(radius);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    host_radius(radial) = 0.12 + 0.19 * radial;
  }
  const auto grid = teuk::angular::gauss_legendre(node_count);
  for (int node = 0; node < node_count; ++node) {
    const auto i = static_cast<std::size_t>(node);
    host_cos(i) = grid.x[i];
    host_sin(i) = std::sqrt(std::max(0.0, 1.0 - grid.x[i] * grid.x[i]));
  }
  Kokkos::deep_copy(execution, fields, host_fields);
  Kokkos::deep_copy(execution, radius, host_radius);
  Kokkos::deep_copy(execution, sin_theta, host_sin);
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  execution.fence("initialize angular coordinator test");

  // Positive controls make the zero launch counters meaningful.
  coordinator_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_coordinator_allocation);
  {
    real_view allocation_probe("coordinator_allocation_probe", 1);
    execution.fence("observe coordinator allocation");
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(coordinator_allocations > 0);
  coordinator_copies = 0;
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_coordinator_copy);
  Kokkos::deep_copy(execution, radius, host_radius);
  execution.fence("observe coordinator copy");
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  CHECK(coordinator_copies > 0);

  coordinator_allocations = 0;
  coordinator_copies = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_coordinator_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_coordinator_copy);
  coordinator.laplacian(execution, fields, 0, fields, 2);
  coordinator.eth(execution, fields, 0, fields, 1, radius, sin_theta,
                  cos_theta, fields, 3);
  coordinator.ethprime(execution, fields, 0, fields, 1, radius, sin_theta,
                       cos_theta, fields, 4);
  execution.fence("finish signed-mode angular coordinator launches");
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  CHECK(coordinator_allocations == 0);
  CHECK(coordinator_copies == 0);

  const auto result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          fields);
  for (std::size_t mode_index = 0; mode_index < registry.size(); ++mode_index) {
    const int m = registry.modes()[mode_index];
    const teuk::angular::SpinWeightedTransform source_transform(
        spin, m, ell_max, node_count);
    const teuk::angular::SpinWeightedTransform raised_transform(
        spin + 1, m, ell_max, node_count);
    const teuk::angular::SpinWeightedTransform lowered_transform(
        spin - 1, m, ell_max, node_count);
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const auto expected_laplacian = source_transform.synthesize(
          source_transform.laplacian(modal[mode_index][radial]));
      const auto raised_nodal = raised_transform.synthesize(align_modal(
          source_transform, source_transform.raise(modal[mode_index][radial]),
          raised_transform));
      const auto lowered_nodal = lowered_transform.synthesize(align_modal(
          source_transform, source_transform.lower(modal[mode_index][radial]),
          lowered_transform));
      for (int node = 0; node < node_count; ++node) {
        const auto i = static_cast<std::size_t>(node);
        CHECK_COMPLEX_NEAR(result(mode_index, 2, radial, i),
                           expected_laplacian[i], 9e-13);
        const auto expected_eth = teuk::eth_n_point(
            host_fields(mode_index, 0, radial, i),
            host_fields(mode_index, 1, radial, i), raised_nodal[i], spin,
            boost, host_radius(radial), host_sin(i), host_cos(i),
            parameters.spin, parameters.compactification_length);
        const auto expected_ethprime = teuk::ethprime_n_point(
            host_fields(mode_index, 0, radial, i),
            host_fields(mode_index, 1, radial, i), lowered_nodal[i], spin,
            boost, host_radius(radial), host_sin(i), host_cos(i),
            parameters.spin, parameters.compactification_length);
        CHECK_COMPLEX_NEAR(result(mode_index, 3, radial, i), expected_eth,
                           1e-12);
        CHECK_COMPLEX_NEAR(result(mode_index, 4, radial, i),
                           expected_ethprime, 1e-12);
      }
    }
  }
}
