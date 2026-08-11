#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>

#include "teuk/device_rk4.hpp"
#include "teuk/types.hpp"

namespace {

double device_exponential_error(const int steps) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  Kokkos::View<teuk::Complex*, memory_space> state("state", 3);
  Kokkos::deep_copy(state, teuk::Complex(1.0, 0.0));
  teuk::DeviceRK4Workspace<teuk::Complex, execution_space> workspace(3);
  const auto rhs = [](const execution_space& execution, const double time,
                      const auto& input, const auto& output) {
    const double rate = 1.0 + 0.1 * time;
    Kokkos::parallel_for(
        "device_rk4_test_rhs",
        Kokkos::RangePolicy<execution_space>(execution, 0, input.extent(0)),
        KOKKOS_LAMBDA(const std::size_t i) { output(i) = rate * input(i); });
  };
  const double step = 1.0 / static_cast<double>(steps);
  double time = 0.0;
  const execution_space execution;
  for (int n = 0; n < steps; ++n) {
    teuk::device_classical_rk4_step(execution, state, time, step, rhs,
                                    workspace);
    time += step;
  }
  execution.fence();
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                        state);
  return std::abs(host(0).real() - std::exp(1.05));
}

}  // namespace

TEST_CASE("Kokkos-resident classical RK4 is fourth order") {
  const double coarse = device_exponential_error(8);
  const double medium = device_exponential_error(16);
  const double fine = device_exponential_error(32);
  CHECK(coarse / medium > 14.0);
  CHECK(medium / fine > 14.0);
  CHECK(fine < 4.0e-8);
}

TEST_CASE("Kokkos-resident RK4 rejects a mismatched workspace") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  Kokkos::View<teuk::Complex*, memory_space> state("state", 2);
  teuk::DeviceRK4Workspace<teuk::Complex, execution_space> workspace(3);
  const auto unused_rhs = [](const execution_space&, const double,
                             const auto&, const auto&) {};
  bool rejected = false;
  try {
    teuk::device_classical_rk4_step(execution_space{}, state, 0.0, 0.1,
                                    unused_rhs, workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}
