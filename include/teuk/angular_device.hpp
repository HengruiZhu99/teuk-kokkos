#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

#include "teuk/angular.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Device-resident fixed-(s,m) angular plan. Construction is the only place
// where host reference matrices are built or copied. Every launch below is an
// allocation-free kernel submission using caller-owned rank-2 batch views:
//
//   modal(batch, ell-ell_min), nodal(batch, theta_node).
//
// A batch may flatten any field/radial collection sharing this fixed (s,m).
// Separate physical m values deliberately use separate plans because their
// basis matrices and modal extents differ.
template <class ExecutionSpace = Kokkos::DefaultExecutionSpace>
class DeviceAngularPlan {
 public:
  using execution_space = ExecutionSpace;
  using memory_space = typename execution_space::memory_space;
  using modal_view =
      Kokkos::View<Complex**, Kokkos::LayoutRight, memory_space>;
  using nodal_view =
      Kokkos::View<Complex**, Kokkos::LayoutRight, memory_space>;
  using real_matrix_view =
      Kokkos::View<Real**, Kokkos::LayoutRight, memory_space>;
  using real_vector_view = Kokkos::View<Real*, memory_space>;

  DeviceAngularPlan(const execution_space& execution, const int spin,
                    const int m, const int ell_max, const int node_count = 0)
      : spin_(spin),
        m_(m),
        ell_min_(angular::minimum_ell(spin, m)),
        ell_max_(ell_max) {
    // The allocation-owning host oracle is temporary. Only its five copied
    // arrays survive construction in the execution space's memory space.
    const angular::SpinWeightedTransform host_reference(spin, m, ell_max,
                                                        node_count);
    synthesis_ = real_matrix_view("angular_synthesis",
                                  host_reference.grid().size(),
                                  host_reference.mode_count());
    analysis_ = real_matrix_view("angular_analysis",
                                 host_reference.mode_count(),
                                 host_reference.grid().size());
    raising_ = real_vector_view("angular_raising",
                                host_reference.mode_count());
    lowering_ = real_vector_view("angular_lowering",
                                 host_reference.mode_count());
    laplacian_ = real_vector_view("angular_laplacian",
                                  host_reference.mode_count());
    auto host_synthesis = Kokkos::create_mirror_view(synthesis_);
    auto host_analysis = Kokkos::create_mirror_view(analysis_);
    auto host_raising = Kokkos::create_mirror_view(raising_);
    auto host_lowering = Kokkos::create_mirror_view(lowering_);
    auto host_laplacian = Kokkos::create_mirror_view(laplacian_);

    for (std::size_t node = 0; node < node_count_value(); ++node) {
      for (std::size_t mode = 0; mode < mode_count(); ++mode) {
        host_synthesis(node, mode) =
            host_reference.synthesis_matrix()(node, mode);
        host_analysis(mode, node) =
            host_reference.analysis_matrix()(mode, node);
      }
    }
    for (std::size_t mode = 0; mode < mode_count(); ++mode) {
      const int ell = ell_min_ + static_cast<int>(mode);
      host_raising(mode) = angular::raising_factor(ell, spin_);
      host_lowering(mode) = angular::lowering_factor(ell, spin_);
      host_laplacian(mode) =
          angular::spin_weighted_laplacian_eigenvalue(ell, spin_);
    }

    // Exactly five initialization transfers. The execution-space fence makes
    // the plan safe to launch later on a different instance of the same space.
    Kokkos::deep_copy(execution, synthesis_, host_synthesis);
    Kokkos::deep_copy(execution, analysis_, host_analysis);
    Kokkos::deep_copy(execution, raising_, host_raising);
    Kokkos::deep_copy(execution, lowering_, host_lowering);
    Kokkos::deep_copy(execution, laplacian_, host_laplacian);
    execution.fence("initialize fixed-m angular plan");
  }

  [[nodiscard]] int spin() const noexcept { return spin_; }
  [[nodiscard]] int m() const noexcept { return m_; }
  [[nodiscard]] int ell_min() const noexcept { return ell_min_; }
  [[nodiscard]] int ell_max() const noexcept { return ell_max_; }
  [[nodiscard]] std::size_t mode_count() const noexcept {
    return synthesis_.extent(1);
  }
  [[nodiscard]] std::size_t node_count() const noexcept {
    return node_count_value();
  }

  template <class ModalView, class NodalView>
  void synthesize(const execution_space& execution, const ModalView& modal,
                  const NodalView& nodal) const {
    static_assert(ModalView::rank == 2 && NodalView::rank == 2,
                  "angular batch views must have rank two");
    validate_synthesis_shapes(modal, nodal);
    if (modal.data() == nodal.data()) {
      throw std::invalid_argument(
          "synthesis input and output must not alias");
    }

    const auto synthesis = synthesis_;
    const std::size_t nodes = node_count_value();
    const std::size_t modes = mode_count();
    const std::size_t total = modal.extent(0) * nodes;
    Kokkos::parallel_for(
        "fixed_m_angular_synthesis",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / nodes;
          const std::size_t node = flat - batch * nodes;
          Complex value(0.0, 0.0);
          for (std::size_t mode = 0; mode < modes; ++mode) {
            value += synthesis(node, mode) * modal(batch, mode);
          }
          nodal(batch, node) = value;
        });
  }

  template <class NodalView, class ModalView>
  void analyze(const execution_space& execution, const NodalView& nodal,
               const ModalView& modal) const {
    static_assert(NodalView::rank == 2 && ModalView::rank == 2,
                  "angular batch views must have rank two");
    validate_analysis_shapes(nodal, modal);
    if (nodal.data() == modal.data()) {
      throw std::invalid_argument("analysis input and output must not alias");
    }

    const auto analysis = analysis_;
    const std::size_t nodes = node_count_value();
    const std::size_t modes = mode_count();
    const std::size_t total = nodal.extent(0) * modes;
    Kokkos::parallel_for(
        "fixed_m_angular_analysis",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / modes;
          const std::size_t mode = flat - batch * modes;
          Complex value(0.0, 0.0);
          for (std::size_t node = 0; node < nodes; ++node) {
            value += analysis(mode, node) * nodal(batch, node);
          }
          modal(batch, mode) = value;
        });
  }

  template <class InputView, class OutputView>
  void raise(const execution_space& execution, const InputView& input,
             const OutputView& output) const {
    apply_diagonal(execution, input, output, raising_,
                   "fixed_m_angular_raise");
  }

  template <class InputView, class OutputView>
  void lower(const execution_space& execution, const InputView& input,
             const OutputView& output) const {
    apply_diagonal(execution, input, output, lowering_,
                   "fixed_m_angular_lower");
  }

  template <class InputView, class OutputView>
  void laplacian(const execution_space& execution, const InputView& input,
                 const OutputView& output) const {
    apply_diagonal(execution, input, output, laplacian_,
                   "fixed_m_angular_laplacian");
  }

 private:
  [[nodiscard]] std::size_t node_count_value() const noexcept {
    return synthesis_.extent(0);
  }

  template <class ModalView, class NodalView>
  void validate_synthesis_shapes(const ModalView& modal,
                                 const NodalView& nodal) const {
    if (modal.extent(1) != mode_count() ||
        nodal.extent(1) != node_count_value() ||
        modal.extent(0) != nodal.extent(0)) {
      throw std::invalid_argument("angular synthesis view shape mismatch");
    }
  }

  template <class NodalView, class ModalView>
  void validate_analysis_shapes(const NodalView& nodal,
                                const ModalView& modal) const {
    if (nodal.extent(1) != node_count_value() ||
        modal.extent(1) != mode_count() ||
        nodal.extent(0) != modal.extent(0)) {
      throw std::invalid_argument("angular analysis view shape mismatch");
    }
  }

  template <class InputView, class OutputView>
  void apply_diagonal(const execution_space& execution,
                      const InputView& input, const OutputView& output,
                      const real_vector_view& diagonal,
                      const char* kernel_name) const {
    static_assert(InputView::rank == 2 && OutputView::rank == 2,
                  "angular batch views must have rank two");
    if (input.extent(0) != output.extent(0) ||
        input.extent(1) != mode_count() ||
        output.extent(1) != mode_count()) {
      throw std::invalid_argument("angular diagonal view shape mismatch");
    }
    const std::size_t modes = mode_count();
    const std::size_t total = input.extent(0) * modes;
    Kokkos::parallel_for(
        kernel_name,
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t batch = flat / modes;
          const std::size_t mode = flat - batch * modes;
          output(batch, mode) = diagonal(mode) * input(batch, mode);
        });
  }

  int spin_;
  int m_;
  int ell_min_;
  int ell_max_;
  real_matrix_view synthesis_;
  real_matrix_view analysis_;
  real_vector_view raising_;
  real_vector_view lowering_;
  real_vector_view laplacian_;
};

}  // namespace teuk
