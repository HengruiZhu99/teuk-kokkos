#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

#include "teuk/angular_device.hpp"
#include "teuk/background.hpp"
#include "teuk/ghp.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Stage-local device plan for the angular parts of eth_n and eth'_n at fixed
// (s,m). The three DeviceAngularPlans are initialized once. A caller constructs
// Workspace once per desired batch capacity and reuses it at every RK stage.
// No launch below allocates, mirrors, copies, or fences.
template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
class DeviceGhpAngularPlan {
 public:
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using angular_plan = DeviceAngularPlan<execution_space>;
  using modal_view = typename angular_plan::modal_view;
  using nodal_view = typename angular_plan::nodal_view;
  using real_view = Kokkos::View<Real*, memory_space>;

  struct Workspace {
    Workspace(const DeviceGhpAngularPlan& plan, const std::size_t batch_count)
        : source_modal("ghp_source_modal", batch_count,
                       plan.source_mode_count()),
          operated_modal("ghp_operated_modal", batch_count,
                         plan.source_mode_count()),
          raised_modal("ghp_raised_modal", batch_count,
                       plan.raised_mode_count()),
          lowered_modal("ghp_lowered_modal", batch_count,
                        plan.lowered_mode_count()),
          angular_nodal("ghp_angular_nodal", batch_count,
                        plan.node_count()) {}

    [[nodiscard]] std::size_t batch_count() const noexcept {
      return source_modal.extent(0);
    }

    modal_view source_modal;
    modal_view operated_modal;
    modal_view raised_modal;
    modal_view lowered_modal;
    nodal_view angular_nodal;
  };

  DeviceGhpAngularPlan(const execution_space& execution, const int spin,
                       const int m, const int boost, const int ell_max,
                       const int node_count, const KerrParameters& parameters)
      : spin_(spin),
        m_(m),
        boost_(boost),
        kerr_spin_(parameters.spin),
        compactification_length_(parameters.compactification_length),
        source_(execution, spin, m, ell_max, node_count),
        raised_(execution, spin + 1, m, ell_max, node_count),
        lowered_(execution, spin - 1, m, ell_max, node_count) {
    if (compactification_length_ <= 0.0) {
      throw std::invalid_argument(
          "GHP angular plan requires positive compactification length");
    }
  }

  [[nodiscard]] int spin() const noexcept { return spin_; }
  [[nodiscard]] int m() const noexcept { return m_; }
  [[nodiscard]] int boost() const noexcept { return boost_; }
  [[nodiscard]] std::size_t node_count() const noexcept {
    return source_.node_count();
  }
  [[nodiscard]] std::size_t source_mode_count() const noexcept {
    return source_.mode_count();
  }
  [[nodiscard]] std::size_t raised_mode_count() const noexcept {
    return raised_.mode_count();
  }
  [[nodiscard]] std::size_t lowered_mode_count() const noexcept {
    return lowered_.mode_count();
  }

  template <class NodalView, class DtView, class RadiusView,
            class SinThetaView, class CosThetaView, class OutputView>
  void eth(const execution_space& execution, const NodalView& field,
           const DtView& dt_field, const RadiusView& radius,
           const SinThetaView& sin_theta, const CosThetaView& cos_theta,
           const OutputView& output, Workspace& workspace) const {
    validate_point_shapes(field, dt_field, radius, sin_theta, cos_theta,
                          output, workspace);
    source_.analyze(execution, field, workspace.source_modal);
    source_.raise(execution, workspace.source_modal,
                  workspace.operated_modal);
    align_modal(execution, workspace.operated_modal, source_.ell_min(),
                workspace.raised_modal, raised_.ell_min());
    raised_.synthesize(execution, workspace.raised_modal,
                       workspace.angular_nodal);
    apply_eth_points(execution, field, dt_field, radius, sin_theta, cos_theta,
                     output, workspace.angular_nodal);
  }

  template <class NodalView, class DtView, class RadiusView,
            class SinThetaView, class CosThetaView, class OutputView>
  void ethprime(const execution_space& execution, const NodalView& field,
                const DtView& dt_field, const RadiusView& radius,
                const SinThetaView& sin_theta,
                const CosThetaView& cos_theta, const OutputView& output,
                Workspace& workspace) const {
    validate_point_shapes(field, dt_field, radius, sin_theta, cos_theta,
                          output, workspace);
    source_.analyze(execution, field, workspace.source_modal);
    source_.lower(execution, workspace.source_modal,
                  workspace.operated_modal);
    align_modal(execution, workspace.operated_modal, source_.ell_min(),
                workspace.lowered_modal, lowered_.ell_min());
    lowered_.synthesize(execution, workspace.lowered_modal,
                        workspace.angular_nodal);
    apply_ethprime_points(execution, field, dt_field, radius, sin_theta,
                          cos_theta, output, workspace.angular_nodal);
  }

 private:
  template <class InputView, class OutputView>
  void align_modal(const execution_space& execution, const InputView& input,
                   const int input_ell_min, const OutputView& output,
                   const int output_ell_min) const {
    const std::size_t batch_count = input.extent(0);
    const std::size_t input_modes = input.extent(1);
    const std::size_t output_modes = output.extent(1);
    const std::size_t total = batch_count * output_modes;
    Kokkos::parallel_for(
        "align_spin_shifted_modal_band",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / output_modes;
          const std::size_t output_mode = flat - batch * output_modes;
          const int ell = output_ell_min + static_cast<int>(output_mode);
          const int input_mode = ell - input_ell_min;
          output(batch, output_mode) =
              input_mode >= 0 &&
                      static_cast<std::size_t>(input_mode) < input_modes
                  ? input(batch, static_cast<std::size_t>(input_mode))
                  : Complex(0.0, 0.0);
        });
  }

  template <class NodalView, class DtView, class RadiusView,
            class SinThetaView, class CosThetaView, class OutputView>
  void validate_point_shapes(const NodalView& field, const DtView& dt_field,
                             const RadiusView& radius,
                             const SinThetaView& sin_theta,
                             const CosThetaView& cos_theta,
                             const OutputView& output,
                             const Workspace& workspace) const {
    static_assert(NodalView::rank == 2 && DtView::rank == 2 &&
                      RadiusView::rank == 1 && SinThetaView::rank == 1 &&
                      CosThetaView::rank == 1 && OutputView::rank == 2,
                  "GHP point views have invalid ranks");
    const std::size_t batches = field.extent(0);
    const std::size_t nodes = node_count();
    if (field.extent(1) != nodes || dt_field.extent(0) != batches ||
        dt_field.extent(1) != nodes || output.extent(0) != batches ||
        output.extent(1) != nodes || radius.extent(0) != batches ||
        sin_theta.extent(0) != nodes || cos_theta.extent(0) != nodes ||
        workspace.batch_count() != batches) {
      throw std::invalid_argument("GHP angular view shape mismatch");
    }
    if (field.data() == output.data() || dt_field.data() == output.data()) {
      throw std::invalid_argument("GHP output must not alias stage inputs");
    }
  }

  template <class NodalView, class DtView, class RadiusView,
            class SinThetaView, class CosThetaView, class OutputView,
            class AngularView>
  void apply_eth_points(const execution_space& execution,
                        const NodalView& field, const DtView& dt_field,
                        const RadiusView& radius,
                        const SinThetaView& sin_theta,
                        const CosThetaView& cos_theta,
                        const OutputView& output,
                        const AngularView& raised_nodal) const {
    const std::size_t nodes = node_count();
    const std::size_t total = field.extent(0) * nodes;
    const int spin = spin_;
    const int boost = boost_;
    const Real kerr_spin = kerr_spin_;
    const Real length = compactification_length_;
    Kokkos::parallel_for(
        "stage_local_eth_n",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / nodes;
          const std::size_t node = flat - batch * nodes;
          output(batch, node) = eth_n_point(
              field(batch, node), dt_field(batch, node),
              raised_nodal(batch, node), spin, boost, radius(batch),
              sin_theta(node), cos_theta(node), kerr_spin, length);
        });
  }

  template <class NodalView, class DtView, class RadiusView,
            class SinThetaView, class CosThetaView, class OutputView,
            class AngularView>
  void apply_ethprime_points(const execution_space& execution,
                             const NodalView& field,
                             const DtView& dt_field,
                             const RadiusView& radius,
                             const SinThetaView& sin_theta,
                             const CosThetaView& cos_theta,
                             const OutputView& output,
                             const AngularView& lowered_nodal) const {
    const std::size_t nodes = node_count();
    const std::size_t total = field.extent(0) * nodes;
    const int spin = spin_;
    const int boost = boost_;
    const Real kerr_spin = kerr_spin_;
    const Real length = compactification_length_;
    Kokkos::parallel_for(
        "stage_local_ethprime_n",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / nodes;
          const std::size_t node = flat - batch * nodes;
          output(batch, node) = ethprime_n_point(
              field(batch, node), dt_field(batch, node),
              lowered_nodal(batch, node), spin, boost, radius(batch),
              sin_theta(node), cos_theta(node), kerr_spin, length);
        });
  }

  int spin_;
  int m_;
  int boost_;
  Real kerr_spin_;
  Real compactification_length_;
  angular_plan source_;
  angular_plan raised_;
  angular_plan lowered_;
};

}  // namespace teuk
