#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "teuk/device_rk4.hpp"
#include "teuk/types.hpp"

namespace {

struct CoupledResult {
  double primary;
  double companion;
  std::vector<double> primary_times;
  std::vector<double> companion_times;
};

CoupledResult evolve_one_way(const int steps, const double companion_initial) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  Kokkos::View<teuk::Complex*, memory_space> primary("one_way_primary", 1);
  Kokkos::View<teuk::Complex*, memory_space> companion("one_way_companion", 1);
  Kokkos::deep_copy(primary, teuk::Complex(0.7, 0.0));
  Kokkos::deep_copy(companion, teuk::Complex(companion_initial, 0.0));
  teuk::DeviceRK4Workspace<teuk::Complex, execution_space> primary_workspace(1);
  teuk::DeviceRK4Workspace<teuk::Complex, execution_space>
      companion_workspace(1);
  std::vector<double> primary_times;
  std::vector<double> companion_times;
  const auto primary_rhs = [&](const execution_space& execution,
                               const double stage_time, const auto& input,
                               const auto& output) {
    primary_times.push_back(stage_time);
    Kokkos::parallel_for(
        "one_way_test_primary_rhs",
        Kokkos::RangePolicy<execution_space>(execution, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) { output(0) = 0.3 * input(0); });
  };
  const auto companion_rhs = [&](const execution_space& execution,
                                 const double stage_time,
                                 const auto& primary_input,
                                 const auto& companion_input,
                                 const auto& output) {
    companion_times.push_back(stage_time);
    Kokkos::parallel_for(
        "one_way_test_companion_rhs",
        Kokkos::RangePolicy<execution_space>(execution, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) {
          output(0) = primary_input(0) * primary_input(0) -
                      0.2 * companion_input(0);
        });
  };
  const execution_space execution;
  const double step = 1.0 / static_cast<double>(steps);
  double time = 0.0;
  for (int n = 0; n < steps; ++n) {
    teuk::device_one_way_coupled_rk4_step(
        execution, primary, companion, time, step, primary_rhs, companion_rhs,
        primary_workspace, companion_workspace);
    time += step;
  }
  execution.fence("finish one-way coupled RK4 test");
  const auto primary_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, primary);
  const auto companion_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, companion);
  return {primary_host(0).real(), companion_host(0).real(),
          std::move(primary_times), std::move(companion_times)};
}

double exact_companion(const double initial) {
  constexpr double primary_initial = 0.7;
  constexpr double primary_rate = 0.3;
  constexpr double companion_decay = 0.2;
  constexpr double final_time = 1.0;
  return std::exp(-companion_decay * final_time) *
         (initial + primary_initial * primary_initial *
                        (std::exp((2.0 * primary_rate + companion_decay) *
                                  final_time) -
                         1.0) /
                        (2.0 * primary_rate + companion_decay));
}

}  // namespace

TEST_CASE("one-way coupled device RK4 has common stages and fourth order") {
  const auto coarse = evolve_one_way(4, -0.1);
  const auto medium = evolve_one_way(8, -0.1);
  const auto fine = evolve_one_way(16, -0.1);
  const double exact_primary = 0.7 * std::exp(0.3);
  const double exact_passive = exact_companion(-0.1);
  const double primary_coarse = std::abs(coarse.primary - exact_primary);
  const double primary_medium = std::abs(medium.primary - exact_primary);
  const double primary_fine = std::abs(fine.primary - exact_primary);
  const double passive_coarse = std::abs(coarse.companion - exact_passive);
  const double passive_medium = std::abs(medium.companion - exact_passive);
  const double passive_fine = std::abs(fine.companion - exact_passive);
  CHECK(primary_coarse / primary_medium > 14.0);
  CHECK(primary_medium / primary_fine > 14.0);
  CHECK(passive_coarse / passive_medium > 14.0);
  CHECK(passive_medium / passive_fine > 14.0);
  CHECK(coarse.primary_times == coarse.companion_times);
  const std::array<double, 4> first_step_times{0.0, 0.125, 0.125, 0.25};
  for (std::size_t stage = 0; stage < first_step_times.size(); ++stage) {
    CHECK(std::abs(coarse.primary_times[stage] - first_step_times[stage]) <
          1.0e-15);
  }
}

TEST_CASE("one-way coupled device RK4 structurally prevents feedback") {
  const auto left = evolve_one_way(7, -4.0);
  const auto right = evolve_one_way(7, 9.0);
  CHECK(left.primary == right.primary);
  CHECK(std::abs(left.companion - right.companion) > 1.0);
}

TEST_CASE("one-way coupled device RK4 rejects mismatched storage before RHS") {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  Kokkos::View<teuk::Complex*, memory_space> primary("bad_primary", 2);
  Kokkos::View<teuk::Complex*, memory_space> companion("bad_companion", 3);
  teuk::DeviceRK4Workspace<teuk::Complex, execution_space> primary_workspace(2);
  teuk::DeviceRK4Workspace<teuk::Complex, execution_space>
      companion_workspace(2);
  int calls = 0;
  const auto primary_rhs = [&](const auto&, double, const auto&, const auto&) {
    ++calls;
  };
  const auto companion_rhs =
      [&](const auto&, double, const auto&, const auto&, const auto&) {
        ++calls;
      };
  bool rejected = false;
  try {
    teuk::device_one_way_coupled_rk4_step(
        execution_space{}, primary, companion, 0.0, 0.1, primary_rhs,
        companion_rhs, primary_workspace, companion_workspace);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  CHECK(calls == 0);
}
