#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>

#include "teuk/angular.hpp"
#include "teuk/grid.hpp"
#include "teuk/modes.hpp"
#include "teuk/plus2_linear_spatial.hpp"
#include "teuk/types.hpp"

TEST_CASE("plus2 linear spatial graph is zero preserving and scri fail closed") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  using state_view =
      Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, memory_space>;
  const execution_space execution;
  const teuk::ModeRegistry registry({-1, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  constexpr int theta_count = 8;
  const auto angular_grid = teuk::angular::gauss_legendre(theta_count);
  Kokkos::View<teuk::Real*, memory_space> sin_theta("plus2_linear_sin",
                                                     theta_count);
  Kokkos::View<teuk::Real*, memory_space> cos_theta("plus2_linear_cos",
                                                     theta_count);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  for (int node = 0; node < theta_count; ++node) {
    host_cos(node) = angular_grid.x[static_cast<std::size_t>(node)];
    host_sin(node) = std::sqrt(1.0 - host_cos(node) * host_cos(node));
  }
  Kokkos::deep_copy(execution, sin_theta, host_sin);
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  state_view stage("plus2_linear_zero_stage", registry.size(), 3, grid.size(),
                   theta_count);
  state_view tangent("plus2_linear_zero_tangent", registry.size(), 3,
                     grid.size(), theta_count);
  state_view second("plus2_linear_zero_second", registry.size(), 3,
                    grid.size(), theta_count);
  Kokkos::deep_copy(execution, stage, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(execution, tangent, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(execution, second, teuk::Complex(0.0, 0.0));
  teuk::Plus2LinearPsi0SpatialWorkspace<execution_space> workspace(
      execution, registry, grid.size(), 3, theta_count);
  const teuk::KerrParameters parameters{1.0, 0.61, 1.4};
  workspace.pack_reconstruction_metric(execution, grid, parameters, sin_theta,
                                       cos_theta, stage, tangent, second);
  workspace.evaluate_packed_metric(execution, grid, parameters, sin_theta,
                                   cos_theta);
  execution.fence("finish zero plus2 linear spatial test");
  const auto raw = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.raw_psi0());
  const auto zplus = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.z_plus());
  const auto valid = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, workspace.z_plus_valid());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      for (int theta = 0; theta < theta_count; ++theta) {
        CHECK(raw(mode, radial, static_cast<std::size_t>(theta)) ==
              teuk::Complex(0.0, 0.0));
        CHECK(zplus(mode, radial, static_cast<std::size_t>(theta)) ==
              teuk::Complex(0.0, 0.0));
        CHECK(valid(mode, radial, static_cast<std::size_t>(theta)) ==
              static_cast<std::uint8_t>(radial == 0 ? 0 : 1));
      }
    }
  }
}
