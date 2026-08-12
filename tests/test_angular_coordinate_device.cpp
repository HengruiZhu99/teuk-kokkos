#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/angular_coordinate_device.hpp"
#include "teuk/types.hpp"

namespace {

int coordinate_launch_allocations = 0;

void count_coordinate_allocation(
    const Kokkos::Tools::SpaceHandle, const char*, const void*,
    const std::uint64_t) {
  ++coordinate_launch_allocations;
}

std::vector<teuk::Complex> host_nodal(
    const teuk::angular::SpinWeightedTransform& transform,
    const std::vector<teuk::Complex>& modal) {
  return transform.synthesize(modal);
}

std::vector<teuk::Complex> finite_difference(
    const int spin, const int m, const int ell_min,
    const std::vector<teuk::Complex>& modal, const std::vector<double>& theta,
    const bool second) {
  std::vector<teuk::Complex> result(theta.size());
  constexpr double h = 2.0e-4;
  for (std::size_t node = 0; node < theta.size(); ++node) {
    const auto value = [&](const double x) {
      teuk::Complex total(0.0, 0.0);
      for (std::size_t mode = 0; mode < modal.size(); ++mode) {
        total += modal[mode] * teuk::angular::spin_weighted_harmonic_theta(
                                   ell_min + static_cast<int>(mode), m, spin,
                                   x);
      }
      return total;
    };
    if (second) {
      result[node] =
          (-value(theta[node] + 2.0 * h) +
           16.0 * value(theta[node] + h) - 30.0 * value(theta[node]) +
           16.0 * value(theta[node] - h) -
           value(theta[node] - 2.0 * h)) /
          (12.0 * h * h);
    } else {
      result[node] =
          (value(theta[node] - 2.0 * h) -
           8.0 * value(theta[node] - h) +
           8.0 * value(theta[node] + h) -
           value(theta[node] + 2.0 * h)) /
          (12.0 * h);
    }
  }
  return result;
}

}  // namespace

TEST_CASE("device coordinate theta derivatives match independent differences") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  constexpr int spin = 2;
  constexpr int m = -1;
  constexpr int ell_max = 6;
  constexpr int nodes = 14;
  const execution_space execution;
  const teuk::angular::SpinWeightedTransform host(spin, m, ell_max, nodes);
  teuk::DeviceSpinCoordinateDerivativePlan<execution_space> plan(
      execution, spin, m, ell_max, nodes);
  teuk::DeviceSpinCoordinateDerivativePlan<execution_space>::Workspace
      workspace(plan, 2);
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, memory_space> input(
      "theta_coordinate_input", 2, nodes);
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, memory_space> first(
      "theta_coordinate_first_output", 2, nodes);
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, memory_space> second(
      "theta_coordinate_second_output", 2, nodes);
  std::vector<teuk::Complex> modal(host.mode_count());
  for (std::size_t i = 0; i < modal.size(); ++i) {
    modal[i] = teuk::Complex(0.13 * static_cast<double>(i + 1),
                            -0.07 * static_cast<double>(i + 2));
  }
  const auto nodal = host_nodal(host, modal);
  auto host_input = Kokkos::create_mirror_view(input);
  for (int batch = 0; batch < 2; ++batch) {
    for (int node = 0; node < nodes; ++node) {
      host_input(batch, node) = static_cast<double>(batch + 1) * nodal[node];
    }
  }
  Kokkos::deep_copy(execution, input, host_input);
  plan.first(execution, input, first, workspace);
  plan.second(execution, input, second, workspace);
  execution.fence("finish coordinate theta derivative comparison");
  const auto actual_first =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, first);
  const auto actual_second =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, second);
  std::vector<double> theta(static_cast<std::size_t>(nodes));
  for (int node = 0; node < nodes; ++node) {
    theta[static_cast<std::size_t>(node)] =
        host.grid().theta(static_cast<std::size_t>(node));
  }
  const auto expected_first = finite_difference(
      spin, m, host.ell_min(), modal, theta, false);
  const auto expected_second = finite_difference(
      spin, m, host.ell_min(), modal, theta, true);
  for (int batch = 0; batch < 2; ++batch) {
    for (int node = 0; node < nodes; ++node) {
      const double scale = static_cast<double>(batch + 1);
      CHECK_COMPLEX_NEAR(actual_first(batch, node),
                         scale * expected_first[node], 2.0e-9);
      CHECK_COMPLEX_NEAR(actual_second(batch, node),
                         scale * expected_second[node], 2.0e-7);
    }
  }
}

TEST_CASE("coordinate theta derivative launch allocates nothing") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  const execution_space execution;
  teuk::DeviceSpinCoordinateDerivativePlan<execution_space> plan(
      execution, -2, 2, 5, 12);
  teuk::DeviceSpinCoordinateDerivativePlan<execution_space>::Workspace
      workspace(plan, 3);
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, memory_space> input(
      "theta_coordinate_noalloc_input", 3, 12);
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, memory_space> output(
      "theta_coordinate_noalloc_output", 3, 12);
  Kokkos::deep_copy(input, teuk::Complex(0.1, -0.2));
  coordinate_launch_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_coordinate_allocation);
  plan.first(execution, input, output, workspace);
  plan.second(execution, input, output, workspace);
  execution.fence("finish no-allocation coordinate derivative launches");
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(coordinate_launch_allocations == 0);
}

TEST_CASE("coordinate theta derivative plan rejects alias and shape mismatch") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  const execution_space execution;
  teuk::DeviceSpinCoordinateDerivativePlan<execution_space> plan(
      execution, 0, 0, 3, 8);
  teuk::DeviceSpinCoordinateDerivativePlan<execution_space>::Workspace
      workspace(plan, 1);
  Kokkos::View<teuk::Complex**, Kokkos::LayoutRight, memory_space> values(
      "theta_coordinate_alias", 1, 8);
  bool alias_rejected = false;
  try {
    plan.first(execution, values, values, workspace);
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  CHECK(alias_rejected);
}
