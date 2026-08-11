#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "teuk/device_rk4.hpp"
#include "teuk/linear_spatial.hpp"
#include "teuk/modes.hpp"
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

struct RadialEvolutionResult {
  std::vector<teuk::Complex> state;
  int rhs_calls = 0;
  bool storage_stable = false;
};

RadialEvolutionResult evolve_device_radial_lines(const int steps) {
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = execution_space::memory_space;
  using flat_view = Kokkos::View<teuk::Complex*, memory_space>;
  using unmanaged_state_view =
      Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, memory_space,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  constexpr std::size_t field_count =
      static_cast<std::size_t>(teuk::TeukolskyField::Count);
  constexpr std::size_t radial_count = 10;
  const teuk::ModeRegistry registry({-1, 1});
  const teuk::UniformRadialGrid grid(radial_count, 0.0, 0.9);
  const std::size_t mode_count = registry.size();
  const std::size_t component_count = mode_count * field_count * radial_count;

  flat_view state("radial_rk_state", component_count);
  auto host_state = Kokkos::create_mirror_view(state);
  const auto flat_index = [](const std::size_t mode, const std::size_t field,
                             const std::size_t radial) {
    return (mode * field_count + field) * radial_count + radial;
  };
  constexpr std::size_t p =
      static_cast<std::size_t>(teuk::TeukolskyField::P);
  constexpr std::size_t q =
      static_cast<std::size_t>(teuk::TeukolskyField::Q);
  constexpr std::size_t psi =
      static_cast<std::size_t>(teuk::TeukolskyField::Psi);
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    const double scale = static_cast<double>(mode + 1);
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      const double radius = grid.coordinate(radial);
      host_state(flat_index(mode, p, radial)) =
          teuk::Complex(scale * (0.2 + 0.1 * radius), -0.03 * radius);
      host_state(flat_index(mode, psi, radial)) =
          teuk::Complex(scale * (0.7 + 0.1 * radius + 0.05 * radius * radius),
                        -0.04 * radius * radius);
      host_state(flat_index(mode, q, radial)) =
          teuk::Complex(scale * (0.1 + 0.1 * radius), -0.08 * radius);
    }
  }
  Kokkos::deep_copy(state, host_state);

  teuk::SignedModeView modes("radial_rk_modes", mode_count);
  auto host_modes = Kokkos::create_mirror_view(modes);
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    host_modes(mode) = registry.modes()[mode];
  }
  Kokkos::deep_copy(modes, host_modes);
  teuk::TeukolskyRadialScratchView scratch(
      "radial_rk_scratch", mode_count,
      static_cast<std::size_t>(teuk::TeukolskyRadialScratch::Count),
      radial_count);
  teuk::TeukolskyRadialValueView angular("radial_rk_angular", mode_count,
                                          radial_count);
  teuk::TeukolskyRadialValueView forcing("radial_rk_forcing", mode_count,
                                          radial_count);
  Kokkos::deep_copy(angular, teuk::Complex(0.0, 0.0));
  Kokkos::deep_copy(forcing, teuk::Complex(0.0, 0.0));

  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.24;
  parameters.compactification_length = 2.0;
  parameters.spin_weight = -2;
  parameters.reduction_damping = 0.3;
  constexpr double theta = 0.77;
  teuk::DeviceRK4Workspace<teuk::Complex, execution_space> workspace(
      component_count);
  const auto state_pointer = state.data();
  const auto stage_pointer = workspace.stage.data();
  const auto k1_pointer = workspace.k1.data();
  const auto k2_pointer = workspace.k2.data();
  const auto k3_pointer = workspace.k3.data();
  const auto k4_pointer = workspace.k4.data();
  const auto scratch_pointer = scratch.data();

  int rhs_calls = 0;
  const auto rhs = [&](const execution_space& execution, const double,
                       const auto& input, const auto& output) {
    ++rhs_calls;
    // Unmanaged wrappers only reshape caller-owned flat RK Views; they do not
    // allocate and remain on the execution space supplied by the RK driver.
    const unmanaged_state_view stage_state(input.data(), mode_count,
                                           field_count, radial_count);
    const unmanaged_state_view stage_rhs(output.data(), mode_count,
                                         field_count, radial_count);
    teuk::evaluate_sbp_teukolsky_radial_lines_rhs(
        execution, grid, modes, stage_state, stage_rhs, scratch, angular,
        forcing, parameters, theta, teuk::ReductionEvolution::FreeDamped);
  };

  constexpr double final_time = 0.2;
  const double step = final_time / static_cast<double>(steps);
  double time = 0.0;
  const execution_space execution;
  for (int n = 0; n < steps; ++n) {
    teuk::device_classical_rk4_step(execution, state, time, step, rhs,
                                    workspace);
    time += step;
  }
  execution.fence();
  const bool storage_stable =
      state.data() == state_pointer && workspace.stage.data() == stage_pointer &&
      workspace.k1.data() == k1_pointer && workspace.k2.data() == k2_pointer &&
      workspace.k3.data() == k3_pointer && workspace.k4.data() == k4_pointer &&
      scratch.data() == scratch_pointer;
  const auto result_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);
  std::vector<teuk::Complex> result(component_count);
  for (std::size_t i = 0; i < component_count; ++i) result[i] = result_host(i);
  return {std::move(result), rhs_calls, storage_stable};
}

double radial_state_distance(const RadialEvolutionResult& left,
                             const RadialEvolutionResult& right) {
  double distance = 0.0;
  for (std::size_t i = 0; i < left.state.size(); ++i) {
    distance =
        std::max(distance, Kokkos::abs(left.state[i] - right.state[i]));
  }
  return distance;
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

TEST_CASE("device RK4 uses supplied radial stage Views and execution queue") {
  const auto steps_2 = evolve_device_radial_lines(2);
  const auto steps_4 = evolve_device_radial_lines(4);
  const auto steps_8 = evolve_device_radial_lines(8);
  const auto steps_16 = evolve_device_radial_lines(16);
  const double difference_2_4 = radial_state_distance(steps_2, steps_4);
  const double difference_4_8 = radial_state_distance(steps_4, steps_8);
  const double difference_8_16 = radial_state_distance(steps_8, steps_16);
  CHECK(difference_2_4 / difference_4_8 > 13.0);
  CHECK(difference_4_8 / difference_8_16 > 14.0);
  CHECK(steps_2.rhs_calls == 8);
  CHECK(steps_4.rhs_calls == 16);
  CHECK(steps_8.rhs_calls == 32);
  CHECK(steps_16.rhs_calls == 64);
  CHECK(steps_2.storage_stable);
  CHECK(steps_4.storage_stable);
  CHECK(steps_8.storage_stable);
  CHECK(steps_16.storage_stable);
}
