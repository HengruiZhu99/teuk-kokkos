#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace teuk {

// Preallocated flat device storage for classical RK4. A Value is normally a
// small state struct (for example TeukolskyState), so all coupled components at
// a logical grid point advance through the same four stages.
template <class Value, class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct DeviceRK4Workspace {
  using memory_space = typename ExecutionSpace::memory_space;
  using view_type = Kokkos::View<Value*, memory_space>;

  explicit DeviceRK4Workspace(const std::size_t component_count)
      : stage("rk4_stage", component_count),
        k1("rk4_k1", component_count),
        k2("rk4_k2", component_count),
        k3("rk4_k3", component_count),
        k4("rk4_k4", component_count) {}

  [[nodiscard]] std::size_t size() const { return stage.extent(0); }

  view_type stage;
  view_type k1;
  view_type k2;
  view_type k3;
  view_type k4;
};

// The RHS callable has signature
//
//   rhs(execution_space, time, input_view, output_view)
//
// and must enqueue all of its kernels on the supplied execution-space
// instance. This keeps stage ordering explicit and permits a multi-kernel RHS
// (radial derivatives, angular transforms, reconstruction, source, forcing)
// without a host fence between operations. The caller chooses when to fence.
template <class Value, class ExecutionSpace, class RightHandSide>
void device_classical_rk4_step(
    const ExecutionSpace& execution_space,
    const Kokkos::View<Value*, typename ExecutionSpace::memory_space>& state,
    const double time, const double step, RightHandSide&& rhs,
    DeviceRK4Workspace<Value, ExecutionSpace>& workspace) {
  const std::size_t size = state.extent(0);
  if (workspace.size() != size) {
    throw std::invalid_argument("device RK4 workspace does not match state");
  }
  using range_policy = Kokkos::RangePolicy<ExecutionSpace>;
  const auto stage = workspace.stage;
  const auto k1 = workspace.k1;
  const auto k2 = workspace.k2;
  const auto k3 = workspace.k3;
  const auto k4 = workspace.k4;

  rhs(execution_space, time, state, k1);
  Kokkos::parallel_for(
      "teuk_rk4_stage_2", range_policy(execution_space, 0, size),
      KOKKOS_LAMBDA(const std::size_t i) {
        stage(i) = state(i) + (0.5 * step) * k1(i);
      });

  rhs(execution_space, time + 0.5 * step, stage, k2);
  Kokkos::parallel_for(
      "teuk_rk4_stage_3", range_policy(execution_space, 0, size),
      KOKKOS_LAMBDA(const std::size_t i) {
        stage(i) = state(i) + (0.5 * step) * k2(i);
      });

  rhs(execution_space, time + 0.5 * step, stage, k3);
  Kokkos::parallel_for(
      "teuk_rk4_stage_4", range_policy(execution_space, 0, size),
      KOKKOS_LAMBDA(const std::size_t i) {
        stage(i) = state(i) + step * k3(i);
      });

  rhs(execution_space, time + step, stage, k4);
  Kokkos::parallel_for(
      "teuk_rk4_accumulate", range_policy(execution_space, 0, size),
      KOKKOS_LAMBDA(const std::size_t i) {
        state(i) += (step / 6.0) *
                    (k1(i) + 2.0 * k2(i) + 2.0 * k3(i) + k4(i));
      });
}

template <class Value, class RightHandSide>
void device_classical_rk4_step(
    const Kokkos::View<Value*, Kokkos::DefaultExecutionSpace::memory_space>&
        state,
    const double time, const double step, RightHandSide&& rhs,
    DeviceRK4Workspace<Value>& workspace) {
  device_classical_rk4_step(Kokkos::DefaultExecutionSpace{}, state, time, step,
                            std::forward<RightHandSide>(rhs), workspace);
}

}  // namespace teuk
