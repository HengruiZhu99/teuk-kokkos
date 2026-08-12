#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace teuk {

namespace device_rk4_detail {

template <class Value>
struct StageFunctor {
  const Value* state;
  const Value* tangent;
  Value* stage;
  double scale;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const {
    stage[i] = state[i] + scale * tangent[i];
  }
};

template <class Value>
struct AccumulateFunctor {
  Value* state;
  const Value* k1;
  const Value* k2;
  const Value* k3;
  const Value* k4;
  double scale;

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const {
    state[i] += scale *
                (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
};

static_assert(
    std::is_trivially_copyable_v<StageFunctor<Kokkos::complex<double>>>);
static_assert(
    std::is_trivially_copyable_v<AccumulateFunctor<Kokkos::complex<double>>>);
static_assert(sizeof(StageFunctor<Kokkos::complex<double>>) < 1800);
static_assert(sizeof(AccumulateFunctor<Kokkos::complex<double>>) < 1800);

}  // namespace device_rk4_detail

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
      device_rk4_detail::StageFunctor<Value>{state.data(), k1.data(),
                                             stage.data(), 0.5 * step});

  rhs(execution_space, time + 0.5 * step, stage, k2);
  Kokkos::parallel_for(
      "teuk_rk4_stage_3", range_policy(execution_space, 0, size),
      device_rk4_detail::StageFunctor<Value>{state.data(), k2.data(),
                                             stage.data(), 0.5 * step});

  rhs(execution_space, time + 0.5 * step, stage, k3);
  Kokkos::parallel_for(
      "teuk_rk4_stage_4", range_policy(execution_space, 0, size),
      device_rk4_detail::StageFunctor<Value>{state.data(), k3.data(),
                                             stage.data(), step});

  rhs(execution_space, time + step, stage, k4);
  Kokkos::parallel_for(
      "teuk_rk4_accumulate", range_policy(execution_space, 0, size),
      device_rk4_detail::AccumulateFunctor<Value>{
          state.data(), k1.data(), k2.data(), k3.data(), k4.data(),
          step / 6.0});
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
      device_rk4_detail::StageFunctor<PrimaryValue>{
          primary.data(), primary_k1.data(), primary_stage.data(),
          0.5 * step});
  Kokkos::parallel_for(
      "teuk_one_way_companion_stage_2",
      range_policy(execution_space, 0, companion_size),
      device_rk4_detail::StageFunctor<CompanionValue>{
          companion.data(), companion_k1.data(), companion_stage.data(),
          0.5 * step});

  primary_rhs(execution_space, time + 0.5 * step, primary_stage, primary_k2);
  companion_rhs(execution_space, time + 0.5 * step, primary_stage,
                companion_stage, companion_k2);
  Kokkos::parallel_for(
      "teuk_one_way_primary_stage_3",
      range_policy(execution_space, 0, primary_size),
      device_rk4_detail::StageFunctor<PrimaryValue>{
          primary.data(), primary_k2.data(), primary_stage.data(),
          0.5 * step});
  Kokkos::parallel_for(
      "teuk_one_way_companion_stage_3",
      range_policy(execution_space, 0, companion_size),
      device_rk4_detail::StageFunctor<CompanionValue>{
          companion.data(), companion_k2.data(), companion_stage.data(),
          0.5 * step});

  primary_rhs(execution_space, time + 0.5 * step, primary_stage, primary_k3);
  companion_rhs(execution_space, time + 0.5 * step, primary_stage,
                companion_stage, companion_k3);
  Kokkos::parallel_for(
      "teuk_one_way_primary_stage_4",
      range_policy(execution_space, 0, primary_size),
      device_rk4_detail::StageFunctor<PrimaryValue>{
          primary.data(), primary_k3.data(), primary_stage.data(), step});
  Kokkos::parallel_for(
      "teuk_one_way_companion_stage_4",
      range_policy(execution_space, 0, companion_size),
      device_rk4_detail::StageFunctor<CompanionValue>{
          companion.data(), companion_k3.data(), companion_stage.data(),
          step});

  primary_rhs(execution_space, time + step, primary_stage, primary_k4);
  companion_rhs(execution_space, time + step, primary_stage, companion_stage,
                companion_k4);
  Kokkos::parallel_for(
      "teuk_one_way_primary_accumulate",
      range_policy(execution_space, 0, primary_size),
      device_rk4_detail::AccumulateFunctor<PrimaryValue>{
          primary.data(), primary_k1.data(), primary_k2.data(),
          primary_k3.data(), primary_k4.data(), step / 6.0});
  Kokkos::parallel_for(
      "teuk_one_way_companion_accumulate",
      range_policy(execution_space, 0, companion_size),
      device_rk4_detail::AccumulateFunctor<CompanionValue>{
          companion.data(), companion_k1.data(), companion_k2.data(),
          companion_k3.data(), companion_k4.data(), step / 6.0});
}

// Advance a primary state and two successively passive states through one
// classical RK4 tableau.  The callable signatures encode the triangular
// dependency and prevent either passive state from feeding back upstream:
//
//   primary_rhs(exec, time, primary_in, primary_out)
//   middle_rhs(exec, time, primary_in, middle_in, middle_out)
//   passive_rhs(exec, time, primary_in, middle_in, passive_in, passive_out)
//
// All three RHS calls at a stage see the identical stage time and the same
// primary/middle stage views.  Workspaces are caller-owned and preallocated;
// this routine performs no allocation, deep copy, or fence.
template <class PrimaryValue, class MiddleValue, class PassiveValue,
          class ExecutionSpace, class PrimaryStateView, class MiddleStateView,
          class PassiveStateView, class PrimaryRightHandSide,
          class MiddleRightHandSide, class PassiveRightHandSide>
void device_one_way_three_state_rk4_step(
    const ExecutionSpace& execution_space, const PrimaryStateView& primary,
    const MiddleStateView& middle, const PassiveStateView& passive,
    const double time, const double step, PrimaryRightHandSide&& primary_rhs,
    MiddleRightHandSide&& middle_rhs, PassiveRightHandSide&& passive_rhs,
    DeviceRK4Workspace<PrimaryValue, ExecutionSpace>& primary_workspace,
    DeviceRK4Workspace<MiddleValue, ExecutionSpace>& middle_workspace,
    DeviceRK4Workspace<PassiveValue, ExecutionSpace>& passive_workspace) {
  static_assert(PrimaryStateView::rank == 1 && MiddleStateView::rank == 1 &&
                    PassiveStateView::rank == 1,
                "three-state one-way RK4 states must be rank-one Views");
  const std::size_t primary_size = primary.extent(0);
  const std::size_t middle_size = middle.extent(0);
  const std::size_t passive_size = passive.extent(0);
  if (primary_workspace.size() != primary_size ||
      middle_workspace.size() != middle_size ||
      passive_workspace.size() != passive_size) {
    throw std::invalid_argument(
        "three-state one-way RK4 workspace does not match state");
  }

  using range_policy = Kokkos::RangePolicy<ExecutionSpace>;
  const auto primary_stage = primary_workspace.stage;
  const auto primary_k1 = primary_workspace.k1;
  const auto primary_k2 = primary_workspace.k2;
  const auto primary_k3 = primary_workspace.k3;
  const auto primary_k4 = primary_workspace.k4;
  const auto middle_stage = middle_workspace.stage;
  const auto middle_k1 = middle_workspace.k1;
  const auto middle_k2 = middle_workspace.k2;
  const auto middle_k3 = middle_workspace.k3;
  const auto middle_k4 = middle_workspace.k4;
  const auto passive_stage = passive_workspace.stage;
  const auto passive_k1 = passive_workspace.k1;
  const auto passive_k2 = passive_workspace.k2;
  const auto passive_k3 = passive_workspace.k3;
  const auto passive_k4 = passive_workspace.k4;

  primary_rhs(execution_space, time, primary, primary_k1);
  middle_rhs(execution_space, time, primary, middle, middle_k1);
  passive_rhs(execution_space, time, primary, middle, passive, passive_k1);
  Kokkos::parallel_for(
      "teuk_three_state_primary_stage_2",
      range_policy(execution_space, 0, primary_size),
      device_rk4_detail::StageFunctor<PrimaryValue>{
          primary.data(), primary_k1.data(), primary_stage.data(),
          0.5 * step});
  Kokkos::parallel_for(
      "teuk_three_state_middle_stage_2",
      range_policy(execution_space, 0, middle_size),
      device_rk4_detail::StageFunctor<MiddleValue>{
          middle.data(), middle_k1.data(), middle_stage.data(), 0.5 * step});
  Kokkos::parallel_for(
      "teuk_three_state_passive_stage_2",
      range_policy(execution_space, 0, passive_size),
      device_rk4_detail::StageFunctor<PassiveValue>{
          passive.data(), passive_k1.data(), passive_stage.data(),
          0.5 * step});

  primary_rhs(execution_space, time + 0.5 * step, primary_stage, primary_k2);
  middle_rhs(execution_space, time + 0.5 * step, primary_stage, middle_stage,
             middle_k2);
  passive_rhs(execution_space, time + 0.5 * step, primary_stage, middle_stage,
              passive_stage, passive_k2);
  Kokkos::parallel_for(
      "teuk_three_state_primary_stage_3",
      range_policy(execution_space, 0, primary_size),
      device_rk4_detail::StageFunctor<PrimaryValue>{
          primary.data(), primary_k2.data(), primary_stage.data(),
          0.5 * step});
  Kokkos::parallel_for(
      "teuk_three_state_middle_stage_3",
      range_policy(execution_space, 0, middle_size),
      device_rk4_detail::StageFunctor<MiddleValue>{
          middle.data(), middle_k2.data(), middle_stage.data(), 0.5 * step});
  Kokkos::parallel_for(
      "teuk_three_state_passive_stage_3",
      range_policy(execution_space, 0, passive_size),
      device_rk4_detail::StageFunctor<PassiveValue>{
          passive.data(), passive_k2.data(), passive_stage.data(),
          0.5 * step});

  primary_rhs(execution_space, time + 0.5 * step, primary_stage, primary_k3);
  middle_rhs(execution_space, time + 0.5 * step, primary_stage, middle_stage,
             middle_k3);
  passive_rhs(execution_space, time + 0.5 * step, primary_stage, middle_stage,
              passive_stage, passive_k3);
  Kokkos::parallel_for(
      "teuk_three_state_primary_stage_4",
      range_policy(execution_space, 0, primary_size),
      device_rk4_detail::StageFunctor<PrimaryValue>{
          primary.data(), primary_k3.data(), primary_stage.data(), step});
  Kokkos::parallel_for(
      "teuk_three_state_middle_stage_4",
      range_policy(execution_space, 0, middle_size),
      device_rk4_detail::StageFunctor<MiddleValue>{
          middle.data(), middle_k3.data(), middle_stage.data(), step});
  Kokkos::parallel_for(
      "teuk_three_state_passive_stage_4",
      range_policy(execution_space, 0, passive_size),
      device_rk4_detail::StageFunctor<PassiveValue>{
          passive.data(), passive_k3.data(), passive_stage.data(), step});

  primary_rhs(execution_space, time + step, primary_stage, primary_k4);
  middle_rhs(execution_space, time + step, primary_stage, middle_stage,
             middle_k4);
  passive_rhs(execution_space, time + step, primary_stage, middle_stage,
              passive_stage, passive_k4);
  Kokkos::parallel_for(
      "teuk_three_state_primary_accumulate",
      range_policy(execution_space, 0, primary_size),
      device_rk4_detail::AccumulateFunctor<PrimaryValue>{
          primary.data(), primary_k1.data(), primary_k2.data(),
          primary_k3.data(), primary_k4.data(), step / 6.0});
  Kokkos::parallel_for(
      "teuk_three_state_middle_accumulate",
      range_policy(execution_space, 0, middle_size),
      device_rk4_detail::AccumulateFunctor<MiddleValue>{
          middle.data(), middle_k1.data(), middle_k2.data(), middle_k3.data(),
          middle_k4.data(), step / 6.0});
  Kokkos::parallel_for(
      "teuk_three_state_passive_accumulate",
      range_policy(execution_space, 0, passive_size),
      device_rk4_detail::AccumulateFunctor<PassiveValue>{
          passive.data(), passive_k1.data(), passive_k2.data(),
          passive_k3.data(), passive_k4.data(), step / 6.0});
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
