#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "test_harness.hpp"
#include "teuk/angular.hpp"
#include "teuk/ghp.hpp"
#include "teuk/ghp_device.hpp"

namespace {

int ghp_deep_copies = 0;
int ghp_allocations = 0;

void count_ghp_deep_copy(Kokkos::Tools::SpaceHandle, const char*, const void*,
                         Kokkos::Tools::SpaceHandle, const char*, const void*,
                         std::uint64_t) {
  ++ghp_deep_copies;
}

void count_ghp_allocation(Kokkos::Tools::SpaceHandle, const char*,
                          const void*, std::uint64_t) {
  ++ghp_allocations;
}

template <class ViewType>
auto host_copy(const ViewType& view) {
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
}

}  // namespace

TEST_CASE("stage-local rotating Kerr GHP plan matches host point formulas") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using plan_type = teuk::DeviceGhpAngularPlan<execution_space>;
  using nodal_view = plan_type::nodal_view;
  using real_view = plan_type::real_view;
  constexpr int spin = -1;
  constexpr int m = 1;
  constexpr int boost = 1;
  constexpr int ell_max = 5;
  constexpr int node_count = 9;
  constexpr std::size_t batches = 3;
  const teuk::KerrParameters parameters{1.0, 0.82, 1.3};
  const execution_space execution;
  const plan_type plan(execution, spin, m, boost, ell_max, node_count,
                       parameters);
  plan_type::Workspace workspace(plan, batches);

  nodal_view field("ghp_field", batches, node_count);
  nodal_view dt_field("ghp_dt_field", batches, node_count);
  nodal_view eth_result("ghp_eth_result", batches, node_count);
  nodal_view ethprime_result("ghp_ethprime_result", batches, node_count);
  real_view radius("ghp_radius", batches);
  real_view sin_theta("ghp_sin_theta", node_count);
  real_view cos_theta("ghp_cos_theta", node_count);

  const teuk::angular::SpinWeightedTransform source_transform(
      spin, m, ell_max, node_count);
  std::vector<std::vector<teuk::Complex>> modal(batches);
  std::vector<std::vector<teuk::Complex>> dt_modal(batches);
  auto host_field = Kokkos::create_mirror_view(field);
  auto host_dt = Kokkos::create_mirror_view(dt_field);
  for (std::size_t batch = 0; batch < batches; ++batch) {
    modal[batch].resize(source_transform.mode_count());
    dt_modal[batch].resize(source_transform.mode_count());
    for (std::size_t mode = 0; mode < source_transform.mode_count(); ++mode) {
      modal[batch][mode] = teuk::Complex(
          0.2 + 0.3 * batch + 0.15 * mode, -0.4 + 0.1 * batch - 0.07 * mode);
      dt_modal[batch][mode] = teuk::Complex(
          -0.1 + 0.2 * batch - 0.05 * mode, 0.3 - 0.08 * batch + 0.04 * mode);
    }
    const auto nodal = source_transform.synthesize(modal[batch]);
    const auto dt_nodal = source_transform.synthesize(dt_modal[batch]);
    for (int node = 0; node < node_count; ++node) {
      host_field(batch, static_cast<std::size_t>(node)) =
          nodal[static_cast<std::size_t>(node)];
      host_dt(batch, static_cast<std::size_t>(node)) =
          dt_nodal[static_cast<std::size_t>(node)];
    }
  }
  auto host_radius = Kokkos::create_mirror_view(radius);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  for (std::size_t batch = 0; batch < batches; ++batch) {
    host_radius(batch) = 0.15 + 0.2 * batch;
  }
  const auto grid = teuk::angular::gauss_legendre(node_count);
  for (int node = 0; node < node_count; ++node) {
    const auto i = static_cast<std::size_t>(node);
    host_cos(i) = grid.x[i];
    host_sin(i) = std::sqrt(std::max(0.0, 1.0 - grid.x[i] * grid.x[i]));
  }
  Kokkos::deep_copy(execution, field, host_field);
  Kokkos::deep_copy(execution, dt_field, host_dt);
  Kokkos::deep_copy(execution, radius, host_radius);
  Kokkos::deep_copy(execution, sin_theta, host_sin);
  Kokkos::deep_copy(execution, cos_theta, host_cos);
  execution.fence("initialize GHP device test");

  // Positive controls establish that both callbacks observe real activity.
  ghp_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_ghp_allocation);
  {
    nodal_view allocation_probe("ghp_allocation_probe", 1, node_count);
    execution.fence("observe GHP allocation positive control");
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(ghp_allocations > 0);
  ghp_deep_copies = 0;
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_ghp_deep_copy);
  Kokkos::deep_copy(execution, eth_result, teuk::Complex(0.0, 0.0));
  execution.fence("observe GHP deep-copy positive control");
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  CHECK(ghp_deep_copies > 0);

  ghp_allocations = 0;
  ghp_deep_copies = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_ghp_allocation);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_ghp_deep_copy);
  plan.eth(execution, field, dt_field, radius, sin_theta, cos_theta,
           eth_result, workspace);
  plan.ethprime(execution, field, dt_field, radius, sin_theta, cos_theta,
                ethprime_result, workspace);
  execution.fence("finish stage-local GHP launches");
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  CHECK(ghp_allocations == 0);
  CHECK(ghp_deep_copies == 0);

  const auto host_eth = host_copy(eth_result);
  const auto host_ethprime = host_copy(ethprime_result);
  const teuk::angular::SpinWeightedTransform raised_transform(
      spin + 1, m, ell_max, node_count);
  const teuk::angular::SpinWeightedTransform lowered_transform(
      spin - 1, m, ell_max, node_count);
  for (std::size_t batch = 0; batch < batches; ++batch) {
    const auto raised_source = source_transform.raise(modal[batch]);
    const auto lowered_source = source_transform.lower(modal[batch]);
    std::vector<teuk::Complex> raised_modal(raised_transform.mode_count(),
                                           teuk::Complex(0.0, 0.0));
    std::vector<teuk::Complex> lowered_modal(lowered_transform.mode_count(),
                                            teuk::Complex(0.0, 0.0));
    for (int ell = source_transform.ell_min(); ell <= ell_max; ++ell) {
      const auto source_mode =
          static_cast<std::size_t>(ell - source_transform.ell_min());
      if (ell >= raised_transform.ell_min()) {
        raised_modal[static_cast<std::size_t>(ell -
                                             raised_transform.ell_min())] =
            raised_source[source_mode];
      }
      if (ell >= lowered_transform.ell_min()) {
        lowered_modal[static_cast<std::size_t>(ell -
                                              lowered_transform.ell_min())] =
            lowered_source[source_mode];
      }
    }
    const auto raised_nodal = raised_transform.synthesize(raised_modal);
    const auto lowered_nodal = lowered_transform.synthesize(lowered_modal);
    for (int node = 0; node < node_count; ++node) {
      const auto i = static_cast<std::size_t>(node);
      const auto expected_eth = teuk::eth_n_point(
          host_field(batch, i), host_dt(batch, i), raised_nodal[i], spin,
          boost, host_radius(batch), host_sin(i), host_cos(i), parameters.spin,
          parameters.compactification_length);
      const auto expected_ethprime = teuk::ethprime_n_point(
          host_field(batch, i), host_dt(batch, i), lowered_nodal[i], spin,
          boost, host_radius(batch), host_sin(i), host_cos(i), parameters.spin,
          parameters.compactification_length);
      CHECK_COMPLEX_NEAR(host_eth(batch, i), expected_eth, 8e-13);
      CHECK_COMPLEX_NEAR(host_ethprime(batch, i), expected_ethprime, 8e-13);
    }
  }
}

TEST_CASE("Schwarzschild GHP plan reduces to modal raising and lowering factors") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using plan_type = teuk::DeviceGhpAngularPlan<execution_space>;
  constexpr int spin = 0;
  constexpr int m = 0;
  constexpr int ell_max = 4;
  constexpr int node_count = 7;
  const teuk::KerrParameters parameters{1.0, 0.0, 1.5};
  const execution_space execution;
  const plan_type plan(execution, spin, m, 0, ell_max, node_count,
                       parameters);
  plan_type::Workspace workspace(plan, 1);
  plan_type::nodal_view field("schwarzschild_ghp_field", 1, node_count);
  plan_type::nodal_view dt("schwarzschild_ghp_dt", 1, node_count);
  plan_type::nodal_view eth("schwarzschild_ghp_eth", 1, node_count);
  plan_type::nodal_view ethprime("schwarzschild_ghp_ethprime", 1,
                                 node_count);
  plan_type::real_view radius("schwarzschild_ghp_radius", 1);
  plan_type::real_view sin_theta("schwarzschild_ghp_sin", node_count);
  plan_type::real_view cos_theta("schwarzschild_ghp_cos", node_count);

  const teuk::angular::SpinWeightedTransform source_transform(
      spin, m, ell_max, node_count);
  std::vector<teuk::Complex> modal(source_transform.mode_count(),
                                   teuk::Complex(0.0, 0.0));
  modal[3] = teuk::Complex(0.7, -0.4);
  const auto nodal = source_transform.synthesize(modal);
  auto host_field = Kokkos::create_mirror_view(field);
  auto host_dt = Kokkos::create_mirror_view(dt);
  auto host_radius = Kokkos::create_mirror_view(radius);
  auto host_sin = Kokkos::create_mirror_view(sin_theta);
  auto host_cos = Kokkos::create_mirror_view(cos_theta);
  const auto grid = teuk::angular::gauss_legendre(node_count);
  for (int node = 0; node < node_count; ++node) {
    const auto i = static_cast<std::size_t>(node);
    host_field(0, i) = nodal[i];
    host_dt(0, i) = teuk::Complex(10.0 + node, -3.0);  // drops out at a=0
    host_cos(i) = grid.x[i];
    host_sin(i) = std::sqrt(1.0 - grid.x[i] * grid.x[i]);
  }
  host_radius(0) = 0.4;
  Kokkos::deep_copy(field, host_field);
  Kokkos::deep_copy(dt, host_dt);
  Kokkos::deep_copy(radius, host_radius);
  Kokkos::deep_copy(sin_theta, host_sin);
  Kokkos::deep_copy(cos_theta, host_cos);
  plan.eth(execution, field, dt, radius, sin_theta, cos_theta, eth, workspace);
  plan.ethprime(execution, field, dt, radius, sin_theta, cos_theta, ethprime,
                workspace);
  execution.fence("finish Schwarzschild GHP test");

  const auto host_eth = host_copy(eth);
  const auto host_ethprime = host_copy(ethprime);
  const teuk::angular::SpinWeightedTransform raised_transform(
      1, m, ell_max, node_count);
  const teuk::angular::SpinWeightedTransform lowered_transform(
      -1, m, ell_max, node_count);
  std::vector<teuk::Complex> raised_modal(raised_transform.mode_count(),
                                         teuk::Complex(0.0, 0.0));
  std::vector<teuk::Complex> lowered_modal(lowered_transform.mode_count(),
                                          teuk::Complex(0.0, 0.0));
  raised_modal[2] = teuk::angular::raising_factor(3, spin) * modal[3];
  lowered_modal[2] = teuk::angular::lowering_factor(3, spin) * modal[3];
  const auto raised_nodal = raised_transform.synthesize(raised_modal);
  const auto lowered_nodal = lowered_transform.synthesize(lowered_modal);
  const double scale = 1.0 / (std::sqrt(2.0) * 1.5 * 1.5);
  for (int node = 0; node < node_count; ++node) {
    const auto i = static_cast<std::size_t>(node);
    CHECK_COMPLEX_NEAR(host_eth(0, i), scale * raised_nodal[i], 4e-13);
    CHECK_COMPLEX_NEAR(host_ethprime(0, i), scale * lowered_nodal[i], 4e-13);
  }
}
