#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/angular.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Exact ordinary-coordinate theta derivatives for a band-limited fixed-(s,m)
// field.  The construction uses the repository's unit-sphere conventions
//
//   d_theta f = -1/2 (eth_s f + ethprime_s f)
//
// and applies the same identity once more for d_theta^2.  The resulting
// derivative synthesis matrices are built once from spin-weighted harmonics;
// repeated device launches only analyze the input and apply a dense matrix.
// This is an ordinary coordinate derivative, not a GHP derivative.
template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
class DeviceSpinCoordinateDerivativePlan {
 public:
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using modal_view =
      Kokkos::View<Complex**, Kokkos::LayoutRight, memory_space>;
  using nodal_view =
      Kokkos::View<Complex**, Kokkos::LayoutRight, memory_space>;
  using real_matrix_view =
      Kokkos::View<Real**, Kokkos::LayoutRight, memory_space>;

  struct Workspace {
    Workspace(const DeviceSpinCoordinateDerivativePlan& plan,
              const std::size_t batch_count,
              const char* label = "theta_coordinate_derivative")
        : modal(std::string(label) + "_modal", batch_count,
                plan.mode_count()) {}

    [[nodiscard]] std::size_t batch_count() const noexcept {
      return modal.extent(0);
    }
    modal_view modal;
  };

  DeviceSpinCoordinateDerivativePlan(const execution_space& execution,
                                     const int spin, const int m,
                                     const int ell_max,
                                     const int node_count = 0)
      : spin_(spin),
        m_(m),
        ell_min_(angular::minimum_ell(spin, m)),
        ell_max_(ell_max) {
    const angular::SpinWeightedTransform reference(spin, m, ell_max,
                                                   node_count);
    analysis_ = real_matrix_view("theta_coordinate_analysis",
                                 reference.mode_count(),
                                 reference.grid().size());
    first_ = real_matrix_view("theta_coordinate_first",
                              reference.grid().size(),
                              reference.mode_count());
    second_ = real_matrix_view("theta_coordinate_second",
                               reference.grid().size(),
                               reference.mode_count());
    auto host_analysis = Kokkos::create_mirror_view(analysis_);
    auto host_first = Kokkos::create_mirror_view(first_);
    auto host_second = Kokkos::create_mirror_view(second_);
    for (std::size_t mode = 0; mode < reference.mode_count(); ++mode) {
      const int ell = ell_min_ + static_cast<int>(mode);
      const Real raise = factor_raise(ell, spin_);
      const Real lower = factor_lower(ell, spin_);
      for (std::size_t node = 0; node < reference.grid().size(); ++node) {
        const Real theta = reference.grid().theta(node);
        host_analysis(mode, node) = reference.analysis_matrix()(mode, node);
        host_first(node, mode) =
            -0.5 *
            (raise * harmonic_if_valid(ell, m_, spin_ + 1, theta) +
             lower * harmonic_if_valid(ell, m_, spin_ - 1, theta));
        host_second(node, mode) =
            0.25 *
            (raise * factor_raise(ell, spin_ + 1) *
                 harmonic_if_valid(ell, m_, spin_ + 2, theta) +
             raise * factor_lower(ell, spin_ + 1) *
                 harmonic_if_valid(ell, m_, spin_, theta) +
             lower * factor_raise(ell, spin_ - 1) *
                 harmonic_if_valid(ell, m_, spin_, theta) +
             lower * factor_lower(ell, spin_ - 1) *
                 harmonic_if_valid(ell, m_, spin_ - 2, theta));
      }
    }
    Kokkos::deep_copy(execution, analysis_, host_analysis);
    Kokkos::deep_copy(execution, first_, host_first);
    Kokkos::deep_copy(execution, second_, host_second);
    execution.fence("initialize coordinate theta derivative plan");
  }

  [[nodiscard]] int spin() const noexcept { return spin_; }
  [[nodiscard]] int m() const noexcept { return m_; }
  [[nodiscard]] int ell_min() const noexcept { return ell_min_; }
  [[nodiscard]] int ell_max() const noexcept { return ell_max_; }
  [[nodiscard]] std::size_t mode_count() const noexcept {
    return analysis_.extent(0);
  }
  [[nodiscard]] std::size_t node_count() const noexcept {
    return analysis_.extent(1);
  }

  template <class InputView, class OutputView>
  void first(const execution_space& execution, const InputView& input,
             const OutputView& output, Workspace& workspace) const {
    apply(execution, input, output, workspace, first_,
          "fixed_m_coordinate_theta_first");
  }

  template <class InputView, class OutputView>
  void second(const execution_space& execution, const InputView& input,
              const OutputView& output, Workspace& workspace) const {
    apply(execution, input, output, workspace, second_,
          "fixed_m_coordinate_theta_second");
  }

 private:
  static Real factor_raise(const int ell, const int spin) {
    if (std::abs(spin) > ell) return 0.0;
    return angular::raising_factor(ell, spin);
  }

  static Real factor_lower(const int ell, const int spin) {
    if (std::abs(spin) > ell) return 0.0;
    return angular::lowering_factor(ell, spin);
  }

  static Real harmonic_if_valid(const int ell, const int m, const int spin,
                                const Real theta) {
    if (std::abs(spin) > ell || std::abs(m) > ell) return 0.0;
    return angular::spin_weighted_harmonic_theta(ell, m, spin, theta);
  }

  template <class InputView, class OutputView>
  void apply(const execution_space& execution, const InputView& input,
             const OutputView& output, Workspace& workspace,
             const real_matrix_view& derivative,
             const char* kernel_name) const {
    static_assert(InputView::rank == 2 && OutputView::rank == 2,
                  "coordinate theta derivative views must have rank two");
    const std::size_t batches = input.extent(0);
    const std::size_t nodes = node_count();
    const std::size_t modes = mode_count();
    if (input.extent(1) != nodes || output.extent(0) != batches ||
        output.extent(1) != nodes || workspace.batch_count() != batches ||
        input.data() == output.data()) {
      throw std::invalid_argument(
          "coordinate theta derivative view shape or alias mismatch");
    }
    const auto analysis = analysis_;
    const auto modal = workspace.modal;
    Kokkos::parallel_for(
        "fixed_m_coordinate_theta_analyze",
        Kokkos::RangePolicy<execution_space>(execution, 0, batches * modes),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / modes;
          const std::size_t mode = flat - batch * modes;
          Complex value(0.0, 0.0);
          for (std::size_t node = 0; node < nodes; ++node) {
            value += analysis(mode, node) * input(batch, node);
          }
          modal(batch, mode) = value;
        });
    Kokkos::parallel_for(
        kernel_name,
        Kokkos::RangePolicy<execution_space>(execution, 0, batches * nodes),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / nodes;
          const std::size_t node = flat - batch * nodes;
          Complex value(0.0, 0.0);
          for (std::size_t mode = 0; mode < modes; ++mode) {
            value += derivative(node, mode) * modal(batch, mode);
          }
          output(batch, node) = value;
        });
  }

  int spin_;
  int m_;
  int ell_min_;
  int ell_max_;
  real_matrix_view analysis_;
  real_matrix_view first_;
  real_matrix_view second_;
};

}  // namespace teuk
