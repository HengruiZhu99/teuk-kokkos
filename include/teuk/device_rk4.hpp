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
template <class Value, class ExecutionSpace, class StateView,
          class RightHandSide>
void device_classical_rk4_step(
    const ExecutionSpace& execution_space,
    const StateView& state, const double time, const double step,
    RightHandSide&& rhs,
    DeviceRK4Workspace<Value, ExecutionSpace>& workspace) {
  static_assert(StateView::rank == 1,
                "device RK4 state must be a rank-one View");
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

// Advance a primary state and a passive companion through the same classical
// RK4 stages.  The separate callable signatures make the one-way dependency
// structural: primary_rhs cannot read the companion, while companion_rhs sees
// both stage-local states.  Both callables must enqueue their work on the
// supplied execution-space instance.
//
//   primary_rhs(exec, time, primary_in, primary_out)
//   companion_rhs(exec, time, primary_in, companion_in, companion_out)
//
// Workspaces are caller-owned and preallocated.  This routine performs no
// allocation, deep copy, or fence.
template <class PrimaryValue, class CompanionValue, class ExecutionSpace,
          class PrimaryStateView, class CompanionStateView,
          class PrimaryRightHandSide, class CompanionRightHandSide>
void device_one_way_coupled_rk4_step(
    const ExecutionSpace& execution_space, const PrimaryStateView& primary,
    const CompanionStateView& companion, const double time, const double step,
    PrimaryRightHandSide&& primary_rhs,
    CompanionRightHandSide&& companion_rhs,
    DeviceRK4Workspace<PrimaryValue, ExecutionSpace>& primary_workspace,
    DeviceRK4Workspace<CompanionValue, ExecutionSpace>& companion_workspace) {
  static_assert(PrimaryStateView::rank == 1 && CompanionStateView::rank == 1,
                "one-way coupled RK4 states must be rank-one Views");
  const std::size_t primary_size = primary.extent(0);
  const std::size_t companion_size = companion.extent(0);
  if (primary_workspace.size() != primary_size ||
      companion_workspace.size() != companion_size) {
    throw std::invalid_argument(
        "one-way coupled RK4 workspace does not match state");
  }

  using range_policy = Kokkos::RangePolicy<ExecutionSpace>;
  const auto primary_stage = primary_workspace.stage;
  const auto primary_k1 = primary_workspace.k1;
  const auto primary_k2 = primary_workspace.k2;
  const auto primary_k3 = primary_workspace.k3;
  const auto primary_k4 = primary_workspace.k4;
  const auto companion_stage = companion_workspace.stage;
  const auto companion_k1 = companion_workspace.k1;
  const auto companion_k2 = companion_workspace.k2;
  const auto companion_k3 = companion_workspace.k3;
  const auto companion_k4 = companion_workspace.k4;

  primary_rhs(execution_space, time, primary, primary_k1);
  companion_rhs(execution_space, time, primary, companion, companion_k1);
  Kokkos::parallel_for(
      "teuk_one_way_primary_stage_2",
      range_policy(execution_space, 0, primary_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        primary_stage(i) = primary(i) + (0.5 * step) * primary_k1(i);
      });
  Kokkos::parallel_for(
      "teuk_one_way_companion_stage_2",
      range_policy(execution_space, 0, companion_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        companion_stage(i) =
            companion(i) + (0.5 * step) * companion_k1(i);
      });

  primary_rhs(execution_space, time + 0.5 * step, primary_stage, primary_k2);
  companion_rhs(execution_space, time + 0.5 * step, primary_stage,
                companion_stage, companion_k2);
  Kokkos::parallel_for(
      "teuk_one_way_primary_stage_3",
      range_policy(execution_space, 0, primary_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        primary_stage(i) = primary(i) + (0.5 * step) * primary_k2(i);
      });
  Kokkos::parallel_for(
      "teuk_one_way_companion_stage_3",
      range_policy(execution_space, 0, companion_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        companion_stage(i) =
            companion(i) + (0.5 * step) * companion_k2(i);
      });

  primary_rhs(execution_space, time + 0.5 * step, primary_stage, primary_k3);
  companion_rhs(execution_space, time + 0.5 * step, primary_stage,
                companion_stage, companion_k3);
  Kokkos::parallel_for(
      "teuk_one_way_primary_stage_4",
      range_policy(execution_space, 0, primary_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        primary_stage(i) = primary(i) + step * primary_k3(i);
      });
  Kokkos::parallel_for(
      "teuk_one_way_companion_stage_4",
      range_policy(execution_space, 0, companion_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        companion_stage(i) = companion(i) + step * companion_k3(i);
      });

  primary_rhs(execution_space, time + step, primary_stage, primary_k4);
  companion_rhs(execution_space, time + step, primary_stage, companion_stage,
                companion_k4);
  Kokkos::parallel_for(
      "teuk_one_way_primary_accumulate",
      range_policy(execution_space, 0, primary_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        primary(i) += (step / 6.0) *
                      (primary_k1(i) + 2.0 * primary_k2(i) +
                       2.0 * primary_k3(i) + primary_k4(i));
      });
  Kokkos::parallel_for(
      "teuk_one_way_companion_accumulate",
      range_policy(execution_space, 0, companion_size),
      KOKKOS_LAMBDA(const std::size_t i) {
        companion(i) += (step / 6.0) *
                        (companion_k1(i) + 2.0 * companion_k2(i) +
                         2.0 * companion_k3(i) + companion_k4(i));
      });
}

template <class Value, class StateView, class RightHandSide>
void device_classical_rk4_step(
    const StateView& state, const double time, const double step,
    RightHandSide&& rhs,
    DeviceRK4Workspace<Value>& workspace) {
  device_classical_rk4_step(Kokkos::DefaultExecutionSpace{}, state, time, step,
                            std::forward<RightHandSide>(rhs), workspace);
}

}  // namespace teuk
