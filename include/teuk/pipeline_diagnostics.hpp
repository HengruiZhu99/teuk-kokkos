#pragma once

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/pipeline_storage.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/types.hpp"

namespace teuk {

struct PipelineNorm {
  double rms = 0.0;
  double maximum = 0.0;
};

struct PipelineFieldDiagnostics {
  PipelineNorm state;
  PipelineNorm rhs;
};

// Host-sized summary returned only when diagnostics are explicitly sampled.
// A successful sample is finite by construction: sample() throws if any
// evolved, RHS, constraint, source, forcing, or endpoint value is nonfinite.
struct PipelineDiagnosticsReport {
  std::array<PipelineFieldDiagnostics, point_pipeline_field_count> fields{};
  PipelineNorm first_reduction_constraint;
  PipelineNorm second_reduction_constraint;
  PipelineNorm source_over_r3;
  PipelineNorm forcing;
  double independent_reconstruction_constraint_maximum = 0.0;
  SourceConstraintNorms normalized_source_constraints;
  SourceActivationState source_activation;
  bool second_order_source_active = false;
  std::size_t point_count = 0;
  bool scri_finite = false;
  bool horizon_finite = false;
  bool all_finite = false;
};

namespace pipeline_diagnostics_detail {

enum ScalarSlot : std::size_t {
  FirstConstraint = 0,
  SecondConstraint = 1,
  Source = 2,
  Forcing = 3,
  ScalarCount = 4,
};

enum FiniteFlag : std::size_t {
  StateFinite = 0,
  RhsFinite = 1,
  FirstConstraintFinite = 2,
  SecondConstraintFinite = 3,
  SourceFinite = 4,
  ForcingFinite = 5,
  ScriFinite = 6,
  HorizonFinite = 7,
  FiniteFlagCount = 8,
};

KOKKOS_INLINE_FUNCTION bool finite_complex(const Complex value) {
  return Kokkos::isfinite(value.real()) && Kokkos::isfinite(value.imag());
}

KOKKOS_INLINE_FUNCTION double magnitude_squared(const Complex value) {
  return value.real() * value.real() + value.imag() * value.imag();
}

template <class StageView>
KOKKOS_INLINE_FUNCTION Complex radial_derivative(
    const StageView& stage, const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t radial_count, const double inverse_spacing,
    const RadialDiscretization discretization) {
  return radial_first_derivative_strided_at(
      discretization, &stage(mode, field, 0, theta), radial_count, radial,
      inverse_spacing, stage.stride(2));
}

}  // namespace pipeline_diagnostics_detail

// Reusable, allocation-owning diagnostics for one fixed full-grid shape. All
// large reductions run in MemorySpace on the caller's execution space. The
// only device-to-host transfer at a sample is this object's fixed-size
// reduction workspace.
class PipelineDiagnostics {
 public:
  PipelineDiagnostics(const std::size_t mode_count,
                      const UniformRadialGrid& radial_grid,
                      const std::size_t theta_count,
                      const std::string& label = "pipeline_diagnostics",
                      const RadialDiscretization radial_discretization =
                          RadialDiscretization::D42)
      : mode_count_(mode_count),
        radial_count_(radial_grid.size()),
        theta_count_(theta_count),
        inverse_spacing_(1.0 / radial_grid.spacing()),
        radial_discretization_(radial_discretization),
        field_sum_squared_(label + "_field_sum_squared",
                           2 * point_pipeline_field_count),
        field_max_squared_(label + "_field_max_squared",
                           2 * point_pipeline_field_count),
        scalar_sum_squared_(label + "_scalar_sum_squared",
                            pipeline_diagnostics_detail::ScalarCount),
        scalar_max_squared_(label + "_scalar_max_squared",
                            pipeline_diagnostics_detail::ScalarCount),
        finite_flags_(label + "_finite_flags",
                      pipeline_diagnostics_detail::FiniteFlagCount),
        host_field_sum_squared_(label + "_host_field_sum_squared",
                                2 * point_pipeline_field_count),
        host_field_max_squared_(label + "_host_field_max_squared",
                                2 * point_pipeline_field_count),
        host_scalar_sum_squared_(label + "_host_scalar_sum_squared",
                                 pipeline_diagnostics_detail::ScalarCount),
        host_scalar_max_squared_(label + "_host_scalar_max_squared",
                                 pipeline_diagnostics_detail::ScalarCount),
        host_finite_flags_(label + "_host_finite_flags",
                           pipeline_diagnostics_detail::FiniteFlagCount),
        host_source_constraint_max_squared_(
            label + "_host_source_constraint_max_squared", 1),
        host_source_active_(label + "_host_source_active", 1) {
    if (mode_count_ == 0 || theta_count_ == 0) {
      throw std::invalid_argument(
          "pipeline diagnostics require nonempty mode and theta dimensions");
    }
    if (radial_count_ < radial_minimum_points(radial_discretization_)) {
      throw std::invalid_argument(
          "pipeline diagnostics grid is too small for selected radial discretization");
    }
  }

  [[nodiscard]] std::size_t mode_count() const { return mode_count_; }
  [[nodiscard]] std::size_t radial_count() const { return radial_count_; }
  [[nodiscard]] std::size_t theta_count() const { return theta_count_; }

  template <class ExecSpace, class StageView, class RhsView,
            class SourceView, class ForcingView>
  PipelineDiagnosticsReport sample(const ExecSpace& execution,
                                   const StageView& stage,
                                   const RhsView& rhs,
                                   const SourceView& source_over_r3,
                                   const ForcingView& forcing) {
    static_assert(StageView::rank == 4 && RhsView::rank == 4,
                  "pipeline state and RHS diagnostics require rank-four views");
    static_assert(SourceView::rank == 3 && ForcingView::rank == 3,
                  "pipeline source diagnostics require rank-three views");
    validate_extents(stage, rhs, source_over_r3, forcing);

    Kokkos::deep_copy(execution, field_sum_squared_, 0.0);
    Kokkos::deep_copy(execution, field_max_squared_, 0.0);
    Kokkos::deep_copy(execution, scalar_sum_squared_, 0.0);
    Kokkos::deep_copy(execution, scalar_max_squared_, 0.0);
    Kokkos::deep_copy(execution, finite_flags_, 1);

    const auto field_sum_squared = field_sum_squared_;
    const auto field_max_squared = field_max_squared_;
    const auto scalar_sum_squared = scalar_sum_squared_;
    const auto scalar_max_squared = scalar_max_squared_;
    const auto finite_flags = finite_flags_;
    const std::size_t radial_count = radial_count_;
    const std::size_t theta_count = theta_count_;
    const double inverse_spacing = inverse_spacing_;
    const RadialDiscretization radial_discretization = radial_discretization_;
    const std::size_t total_points =
        mode_count_ * radial_count_ * theta_count_;

    Kokkos::parallel_for(
        "pipeline diagnostics reductions",
        Kokkos::RangePolicy<ExecSpace>(execution, 0, total_points),
        KOKKOS_LAMBDA(const std::size_t flat) {
          using namespace pipeline_diagnostics_detail;
          const std::size_t theta = flat % theta_count;
          const std::size_t radial = (flat / theta_count) % radial_count;
          const std::size_t mode = flat / (theta_count * radial_count);
          const bool at_scri = radial == 0;
          const bool at_horizon = radial + 1 == radial_count;

          for (std::size_t field = 0; field < point_pipeline_field_count;
               ++field) {
            const Complex state_value = stage(mode, field, radial, theta);
            const Complex rhs_value = rhs(mode, field, radial, theta);
            if (finite_complex(state_value)) {
              const double norm = magnitude_squared(state_value);
              Kokkos::atomic_add(&field_sum_squared(field), norm);
              Kokkos::atomic_max(&field_max_squared(field), norm);
            } else {
              Kokkos::atomic_exchange(&finite_flags(StateFinite), 0);
              if (at_scri)
                Kokkos::atomic_exchange(&finite_flags(ScriFinite), 0);
              if (at_horizon)
                Kokkos::atomic_exchange(&finite_flags(HorizonFinite), 0);
            }
            if (finite_complex(rhs_value)) {
              const double norm = magnitude_squared(rhs_value);
              const std::size_t slot = point_pipeline_field_count + field;
              Kokkos::atomic_add(&field_sum_squared(slot), norm);
              Kokkos::atomic_max(&field_max_squared(slot), norm);
            } else {
              Kokkos::atomic_exchange(&finite_flags(RhsFinite), 0);
              if (at_scri)
                Kokkos::atomic_exchange(&finite_flags(ScriFinite), 0);
              if (at_horizon)
                Kokkos::atomic_exchange(&finite_flags(HorizonFinite), 0);
            }
          }

          const Complex first_constraint =
              stage(mode, static_cast<std::size_t>(PipelineField::FirstQ),
                    radial, theta) -
              radial_derivative(
                  stage, mode,
                  static_cast<std::size_t>(PipelineField::FirstPsi), radial,
                  theta, radial_count, inverse_spacing,
                  radial_discretization);
          const Complex second_constraint =
              stage(mode, static_cast<std::size_t>(PipelineField::SecondQ),
                    radial, theta) -
              radial_derivative(
                  stage, mode,
                  static_cast<std::size_t>(PipelineField::SecondPsi), radial,
                  theta, radial_count, inverse_spacing,
                  radial_discretization);
          if (finite_complex(first_constraint)) {
            const double norm = magnitude_squared(first_constraint);
            Kokkos::atomic_add(&scalar_sum_squared(FirstConstraint), norm);
            Kokkos::atomic_max(&scalar_max_squared(FirstConstraint), norm);
          } else {
            Kokkos::atomic_exchange(&finite_flags(FirstConstraintFinite), 0);
          }
          if (finite_complex(second_constraint)) {
            const double norm = magnitude_squared(second_constraint);
            Kokkos::atomic_add(&scalar_sum_squared(SecondConstraint), norm);
            Kokkos::atomic_max(&scalar_max_squared(SecondConstraint), norm);
          } else {
            Kokkos::atomic_exchange(&finite_flags(SecondConstraintFinite), 0);
          }

          const Complex source_value = source_over_r3(mode, radial, theta);
          const Complex forcing_value = forcing(mode, radial, theta);
          if (finite_complex(source_value)) {
            const double norm = magnitude_squared(source_value);
            Kokkos::atomic_add(&scalar_sum_squared(Source), norm);
            Kokkos::atomic_max(&scalar_max_squared(Source), norm);
          } else {
            Kokkos::atomic_exchange(&finite_flags(SourceFinite), 0);
            if (at_scri)
              Kokkos::atomic_exchange(&finite_flags(ScriFinite), 0);
            if (at_horizon)
              Kokkos::atomic_exchange(&finite_flags(HorizonFinite), 0);
          }
          if (finite_complex(forcing_value)) {
            const double norm = magnitude_squared(forcing_value);
            Kokkos::atomic_add(&scalar_sum_squared(Forcing), norm);
            Kokkos::atomic_max(&scalar_max_squared(Forcing), norm);
          } else {
            Kokkos::atomic_exchange(&finite_flags(ForcingFinite), 0);
            if (at_scri)
              Kokkos::atomic_exchange(&finite_flags(ScriFinite), 0);
            if (at_horizon)
              Kokkos::atomic_exchange(&finite_flags(HorizonFinite), 0);
          }
        });

    Kokkos::deep_copy(execution, host_field_sum_squared_,
                      field_sum_squared_);
    Kokkos::deep_copy(execution, host_field_max_squared_,
                      field_max_squared_);
    Kokkos::deep_copy(execution, host_scalar_sum_squared_,
                      scalar_sum_squared_);
    Kokkos::deep_copy(execution, host_scalar_max_squared_,
                      scalar_max_squared_);
    Kokkos::deep_copy(execution, host_finite_flags_, finite_flags_);
    execution.fence("copy pipeline diagnostics report");

    PipelineDiagnosticsReport report;
    report.point_count = total_points;
    const double inverse_count = 1.0 / static_cast<double>(total_points);
    for (std::size_t field = 0; field < point_pipeline_field_count; ++field) {
      report.fields[field].state = make_norm(field, inverse_count);
      report.fields[field].rhs =
          make_norm(point_pipeline_field_count + field, inverse_count);
    }
    report.first_reduction_constraint =
        make_scalar_norm(pipeline_diagnostics_detail::FirstConstraint,
                         inverse_count);
    report.second_reduction_constraint =
        make_scalar_norm(pipeline_diagnostics_detail::SecondConstraint,
                         inverse_count);
    report.source_over_r3 =
        make_scalar_norm(pipeline_diagnostics_detail::Source, inverse_count);
    report.forcing =
        make_scalar_norm(pipeline_diagnostics_detail::Forcing, inverse_count);
    report.scri_finite =
        host_finite_flags_(pipeline_diagnostics_detail::ScriFinite) != 0;
    report.horizon_finite =
        host_finite_flags_(pipeline_diagnostics_detail::HorizonFinite) != 0;
    report.all_finite = all_finite();

    if (!report.all_finite) {
      throw std::runtime_error(nonfinite_message());
    }
    return report;
  }

  template <class ExecSpace, class Storage, class SourceView,
            class ForcingView>
  PipelineDiagnosticsReport sample_storage(
      const ExecSpace& execution, const Storage& storage,
      const SourceView& source_over_r3, const ForcingView& forcing) {
    return sample(execution, storage.state(), storage.rhs(), source_over_r3,
                  forcing);
  }

  template <class ExecSpace, class Pipeline>
  PipelineDiagnosticsReport sample_pipeline(const ExecSpace& execution,
                                             const Pipeline& pipeline) {
    if (pipeline.radial_discretization() != radial_discretization_) {
      throw std::invalid_argument(
          "pipeline diagnostics radial discretization does not match pipeline");
    }
    auto report = sample_storage(execution, pipeline.storage(),
                                 pipeline.source_over_r3(),
                                 pipeline.forcing());
    Kokkos::deep_copy(execution, host_source_constraint_max_squared_,
                      pipeline.source_constraint_max_squared());
    Kokkos::deep_copy(execution, host_source_active_,
                      pipeline.source_active());
    execution.fence("copy second-order source activation diagnostics");
    report.independent_reconstruction_constraint_maximum =
        std::sqrt(host_source_constraint_max_squared_(0));
    report.second_order_source_active = host_source_active_(0) != 0;
    report.normalized_source_constraints = pipeline.source_constraint_norms();
    report.source_activation = pipeline.source_activation_state();
    return report;
  }

 private:
  template <class StageView, class RhsView, class SourceView,
            class ForcingView>
  void validate_extents(const StageView& stage, const RhsView& rhs,
                        const SourceView& source,
                        const ForcingView& forcing) const {
    const bool stage_matches =
        stage.extent(0) == mode_count_ &&
        stage.extent(1) == point_pipeline_field_count &&
        stage.extent(2) == radial_count_ && stage.extent(3) == theta_count_;
    const bool rhs_matches =
        rhs.extent(0) == mode_count_ &&
        rhs.extent(1) == point_pipeline_field_count &&
        rhs.extent(2) == radial_count_ && rhs.extent(3) == theta_count_;
    const bool source_matches = source.extent(0) == mode_count_ &&
                                source.extent(1) == radial_count_ &&
                                source.extent(2) == theta_count_;
    const bool forcing_matches = forcing.extent(0) == mode_count_ &&
                                 forcing.extent(1) == radial_count_ &&
                                 forcing.extent(2) == theta_count_;
    if (!stage_matches || !rhs_matches || !source_matches ||
        !forcing_matches) {
      throw std::invalid_argument(
          "pipeline diagnostics view extents do not match configured grid");
    }
  }

  [[nodiscard]] PipelineNorm make_norm(const std::size_t slot,
                                       const double inverse_count) const {
    return {std::sqrt(host_field_sum_squared_(slot) * inverse_count),
            std::sqrt(host_field_max_squared_(slot))};
  }

  [[nodiscard]] PipelineNorm make_scalar_norm(
      const std::size_t slot, const double inverse_count) const {
    return {std::sqrt(host_scalar_sum_squared_(slot) * inverse_count),
            std::sqrt(host_scalar_max_squared_(slot))};
  }

  [[nodiscard]] bool all_finite() const {
    for (std::size_t flag = 0;
         flag < pipeline_diagnostics_detail::FiniteFlagCount; ++flag) {
      if (host_finite_flags_(flag) == 0) return false;
    }
    return true;
  }

  [[nodiscard]] std::string nonfinite_message() const {
    using namespace pipeline_diagnostics_detail;
    std::string message = "pipeline diagnostics found nonfinite values in";
    if (host_finite_flags_(StateFinite) == 0) message += " state";
    if (host_finite_flags_(RhsFinite) == 0) message += " rhs";
    if (host_finite_flags_(FirstConstraintFinite) == 0)
      message += " first-constraint";
    if (host_finite_flags_(SecondConstraintFinite) == 0)
      message += " second-constraint";
    if (host_finite_flags_(SourceFinite) == 0) message += " source";
    if (host_finite_flags_(ForcingFinite) == 0) message += " forcing";
    if (host_finite_flags_(ScriFinite) == 0) message += " scri-endpoint";
    if (host_finite_flags_(HorizonFinite) == 0)
      message += " horizon-endpoint";
    return message;
  }

  std::size_t mode_count_;
  std::size_t radial_count_;
  std::size_t theta_count_;
  double inverse_spacing_;
  RadialDiscretization radial_discretization_;
  Kokkos::View<double*, MemorySpace> field_sum_squared_;
  Kokkos::View<double*, MemorySpace> field_max_squared_;
  Kokkos::View<double*, MemorySpace> scalar_sum_squared_;
  Kokkos::View<double*, MemorySpace> scalar_max_squared_;
  Kokkos::View<int*, MemorySpace> finite_flags_;
  Kokkos::View<double*, Kokkos::HostSpace> host_field_sum_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_field_max_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_scalar_sum_squared_;
  Kokkos::View<double*, Kokkos::HostSpace> host_scalar_max_squared_;
  Kokkos::View<int*, Kokkos::HostSpace> host_finite_flags_;
  Kokkos::View<double*, Kokkos::HostSpace>
      host_source_constraint_max_squared_;
  Kokkos::View<int*, Kokkos::HostSpace> host_source_active_;
};

}  // namespace teuk
