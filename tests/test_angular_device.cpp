#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "test_harness.hpp"
#include "teuk/angular.hpp"
#include "teuk/angular_device.hpp"

namespace {

int deep_copy_calls = 0;
int allocation_calls = 0;

void count_deep_copy(Kokkos::Tools::SpaceHandle, const char*, const void*,
                     Kokkos::Tools::SpaceHandle, const char*, const void*,
                     std::uint64_t) {
  ++deep_copy_calls;
}

void count_allocation(Kokkos::Tools::SpaceHandle, const char*, const void*,
                      std::uint64_t) {
  ++allocation_calls;
}

}  // namespace

TEST_CASE("device angular plan copies matrices once and launches without copies") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using plan_type = teuk::DeviceAngularPlan<execution_space>;
  constexpr std::size_t batch_count = 3;
  const execution_space execution;

  deep_copy_calls = 0;
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(count_deep_copy);
  const plan_type plan(execution, -2, 2, 6, 10);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  CHECK(deep_copy_calls == 5);

  plan_type::modal_view modal("angular_test_modal", batch_count,
                              plan.mode_count());
  plan_type::nodal_view nodal("angular_test_nodal", batch_count,
                              plan.node_count());
  plan_type::modal_view recovered("angular_test_recovered", batch_count,
                                  plan.mode_count());
  plan_type::modal_view raised("angular_test_raised", batch_count,
                               plan.mode_count());
  plan_type::modal_view lowered("angular_test_lowered", batch_count,
                                plan.mode_count());
  plan_type::modal_view laplacian("angular_test_laplacian", batch_count,
                                  plan.mode_count());

  auto host_modal = Kokkos::create_mirror_view(modal);
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    for (std::size_t mode = 0; mode < plan.mode_count(); ++mode) {
      host_modal(batch, mode) = teuk::Complex(
          0.25 + batch + 0.5 * mode, -0.75 + 0.125 * batch - 0.2 * mode);
    }
  }
  Kokkos::deep_copy(execution, modal, host_modal);
  execution.fence("initialize angular test modal data");

  // Positive control: prove the active backend reports a deliberate View
  // allocation through the same tools callback used around the launches.
  allocation_calls = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(count_allocation);
  {
    plan_type::modal_view allocation_probe("angular_allocation_probe", 1,
                                           plan.mode_count());
    execution.fence("observe angular allocation positive control");
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(allocation_calls > 0);

  deep_copy_calls = 0;
  allocation_calls = 0;
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(count_deep_copy);
  Kokkos::Tools::Experimental::set_allocate_data_callback(count_allocation);
  plan.synthesize(execution, modal, nodal);
  plan.analyze(execution, nodal, recovered);
  plan.raise(execution, modal, raised);
  plan.lower(execution, modal, lowered);
  plan.laplacian(execution, modal, laplacian);
  // Repeated submissions exercise the same precomputed plan and caller views.
  plan.synthesize(execution, modal, nodal);
  execution.fence("finish allocation-free angular launches");
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(deep_copy_calls == 0);
  CHECK(allocation_calls == 0);

  const auto host_nodal = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, nodal);
  const auto host_recovered = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, recovered);
  const auto host_raised = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, raised);
  const auto host_lowered = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, lowered);
  const auto host_laplacian = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, laplacian);

  const teuk::angular::SpinWeightedTransform reference(-2, 2, 6, 10);
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    std::vector<teuk::Complex> reference_modal(plan.mode_count());
    for (std::size_t mode = 0; mode < plan.mode_count(); ++mode) {
      reference_modal[mode] = host_modal(batch, mode);
    }
    const auto expected_nodal = reference.synthesize(reference_modal);
    const auto expected_recovered = reference.analyze(expected_nodal);
    const auto expected_raised = reference.raise(reference_modal);
    const auto expected_lowered = reference.lower(reference_modal);
    const auto expected_laplacian = reference.laplacian(reference_modal);
    for (std::size_t node = 0; node < plan.node_count(); ++node) {
      CHECK_COMPLEX_NEAR(host_nodal(batch, node), expected_nodal[node], 3e-14);
    }
    for (std::size_t mode = 0; mode < plan.mode_count(); ++mode) {
      CHECK_COMPLEX_NEAR(host_recovered(batch, mode),
                         expected_recovered[mode], 3e-14);
      CHECK_COMPLEX_NEAR(host_raised(batch, mode), expected_raised[mode],
                         3e-14);
      CHECK_COMPLEX_NEAR(host_lowered(batch, mode), expected_lowered[mode],
                         3e-14);
      CHECK_COMPLEX_NEAR(host_laplacian(batch, mode),
                         expected_laplacian[mode], 3e-14);
      const int ell = plan.ell_min() + static_cast<int>(mode);
      const double independently_expected =
          -static_cast<double>((ell + 2) * (ell - 1));
      CHECK_COMPLEX_NEAR(host_laplacian(batch, mode),
                         independently_expected * host_modal(batch, mode),
                         3e-14);
    }
  }
}

TEST_CASE("device angular plan rejects mismatched caller views") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using plan_type = teuk::DeviceAngularPlan<execution_space>;
  const execution_space execution;
  const plan_type plan(execution, 0, 1, 4, 7);
  plan_type::modal_view modal("angular_shape_modal", 2, plan.mode_count());
  plan_type::nodal_view wrong_nodal("angular_shape_wrong", 3,
                                    plan.node_count());
  bool rejected = false;
  try {
    plan.synthesize(execution, modal, wrong_nodal);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}
