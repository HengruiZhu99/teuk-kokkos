#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#include "teuk/angular_coordinator.hpp"
#include "teuk/device_rk4.hpp"
#include "teuk/full_spatial.hpp"
#include "teuk/pipeline_bands.hpp"
#include "teuk/pipeline_storage.hpp"
#include "teuk/reconstruction_constraints.hpp"
#include "teuk/source_spatial.hpp"
#include "teuk/source_activation.hpp"
#include "teuk/source_tangent_spatial.hpp"

namespace teuk {

struct SpatialPipelineTiming {
  double first_linear_seconds = 0.0;
  double reconstruction_seconds = 0.0;
  double tangent_seconds = 0.0;
  double source_seconds = 0.0;
  double second_linear_seconds = 0.0;

  [[nodiscard]] double total_seconds() const {
    return first_linear_seconds + reconstruction_seconds + tangent_seconds +
           source_seconds + second_linear_seconds;
  }
};

enum class SecondOrderSourceMode {
  // Linear-only evolution: retain the second-order storage contract but never
  // apply or activate the quadratic forcing.
  Disabled,
  // Development-safe default: the raw quadratic source is diagnosed, but the
  // coordinate forcing is applied only after both the configured causal start
  // time and the independent reconstruction-constraint tolerance are met.
  ConstraintAware,
  // Explicit algebra/testing opt-in. This can drive from inconsistent metric
  // data and must not be interpreted as a physical second-order solution.
  Unrestricted,
};

struct SecondOrderSourcePolicy {
  SecondOrderSourceMode mode = SecondOrderSourceMode::ConstraintAware;
  double source_start_time = 0.0;
  double normalized_constraint_tolerance = 1.0e-10;
  int required_consecutive_passes = 1;

  [[nodiscard]] static SecondOrderSourcePolicy unrestricted() {
    return {SecondOrderSourceMode::Unrestricted, 0.0, 0.0, 1};
  }

  [[nodiscard]] static SecondOrderSourcePolicy disabled() {
    return {SecondOrderSourceMode::Disabled, 0.0, 0.0, 1};
  }
};

namespace spatial_pipeline_detail {

inline constexpr TeukolskyFullFieldOffsets first_fields{
    static_cast<std::size_t>(PipelineField::FirstP),
    static_cast<std::size_t>(PipelineField::FirstQ),
    static_cast<std::size_t>(PipelineField::FirstPsi)};
inline constexpr TeukolskyFullFieldOffsets second_fields{
    static_cast<std::size_t>(PipelineField::SecondP),
    static_cast<std::size_t>(PipelineField::SecondQ),
    static_cast<std::size_t>(PipelineField::SecondPsi)};
inline constexpr ReconstructionFullFieldOffsets reconstruction_fields{
    static_cast<std::size_t>(PipelineField::G),
    static_cast<std::size_t>(PipelineField::Lambda),
    static_cast<std::size_t>(PipelineField::H),
    static_cast<std::size_t>(PipelineField::B),
    static_cast<std::size_t>(PipelineField::Pi),
    static_cast<std::size_t>(PipelineField::C),
    static_cast<std::size_t>(PipelineField::U)};

enum class LinearAngularSlot : std::size_t {
  First = 0,
  FirstTangent = 1,
  Second = 2,
  Count = 3,
};

enum class SourceExtraSlot : std::size_t {
  Eth1B = 0,
  EthPrime1F = 1,
  Eth1BTangent = 2,
  EthPrime1FTangent = 3,
  Count = 4,
};

enum class SharpSlot : std::size_t {
  B = 0,
  C = 1,
  BDt = 2,
  CDt = 3,
  BDDt = 4,
  CDDt = 5,
  Count = 6,
};

enum class IndependentAngularSlot : std::size_t {
  EthPrime2G = 0,
  EthPrime3H = 1,
  Count = 2,
};

enum class IndependentConstraintSlot : std::size_t {
  Psi3Bianchi = 0,
  Psi2Bianchi = 1,
  HllReality = 2,
  Count = 3,
};

enum class IndependentConstraintTermSlot : std::size_t {
  Psi3Left = 0,
  Psi3Right = 1,
  Psi2Left = 2,
  Psi2Right = 3,
  HllLeft = 4,
  HllRight = 5,
  Count = 6,
};

inline constexpr ReconstructionFullFieldOffsets compact_reconstruction_fields{
    0, 1, 2, 3, 4, 5, 6};

}  // namespace spatial_pipeline_detail

// Complete device-resident spatial stage graph. The evolved state has the
// documented 13-field ordering. Angular fields live on a padded common
// Gauss-Legendre grid while every retained fixed-m band is truncated at
// ell_max. All scratch is allocated in the constructor and reused by RK4.
class SpatialPipeline {
 public:
  SpatialPipeline(const ExecutionSpace& execution,
                  const ModeRegistry& registry,
                  const UniformRadialGrid& radial_grid, const int ell_max,
                  const int theta_nodes, const KerrParameters& background,
                  const double reduction_damping = 0.1,
                  const double dissipation = 0.0,
                  const ReductionEvolution reduction =
                      ReductionEvolution::FreeDamped,
                  const std::string& label = "full_pipeline",
                  const SecondOrderSourcePolicy source_policy = {})
      : SpatialPipeline(execution, registry, radial_grid,
                        PipelineAngularBands{ell_max, ell_max}, theta_nodes,
                        background, reduction_damping, dissipation, reduction,
                        label, source_policy) {}

  SpatialPipeline(const ExecutionSpace& execution,
                  const ModeRegistry& registry,
                  const UniformRadialGrid& radial_grid,
                  const PipelineAngularBands angular_bands,
                  const int theta_nodes, const KerrParameters& background,
                  const double reduction_damping = 0.1,
                  const double dissipation = 0.0,
                  const ReductionEvolution reduction =
                      ReductionEvolution::FreeDamped,
                  const std::string& label = "full_pipeline",
                  const SecondOrderSourcePolicy source_policy = {})
      : registry_(registry),
        angular_bands_(angular_bands),
        background_(background),
        teukolsky_{background.mass,
                   background.spin,
                   background.compactification_length,
                   -2,
                   0,
                   reduction_damping},
        dissipation_(dissipation),
        reduction_(reduction),
        source_policy_(source_policy),
        storage_(registry, radial_grid, angular::gauss_legendre(theta_nodes),
                 label),
        first_angular_(execution, registry, -2, -2,
                       angular_bands.ell_max_first, theta_nodes,
                       radial_grid.size(), background, registry.parents()),
        second_angular_(execution, registry, -2, -2,
                        angular_bands.ell_max_second, theta_nodes,
                        radial_grid.size(), background, registry.targets()),
        g_angular_(execution, registry, -1, -1,
                   angular_bands.ell_max_first, theta_nodes,
                   radial_grid.size(), background, registry.parents()),
        b_angular_(execution, registry, -2, 0,
                   angular_bands.ell_max_first, theta_nodes,
                   radial_grid.size(), background, registry.parents()),
        pi_angular_(execution, registry, -1, 0,
                    angular_bands.ell_max_first, theta_nodes,
                    radial_grid.size(), background, registry.parents()),
        c_angular_(execution, registry, -1, 1,
                   angular_bands.ell_max_first, theta_nodes,
                   radial_grid.size(), background, registry.parents()),
        scalar_angular_(execution, registry, 0, 0,
                        angular_bands.ell_max_first, theta_nodes,
                        radial_grid.size(), background, registry.parents()),
        b_sharp_angular_(execution, registry, 2, 0,
                         angular_bands.ell_max_first, theta_nodes,
                         radial_grid.size(), background, registry.parents()),
        c_sharp_angular_(execution, registry, 1, 1,
                         angular_bands.ell_max_first, theta_nodes,
                         radial_grid.size(), background, registry.parents()),
        t_angular_(execution, registry, -1, -2,
                   angular_bands.ell_max_second, theta_nodes,
                   radial_grid.size(), background, registry.targets()),
        linear_angular_(label + "_linear_angular", registry.size(),
                        static_cast<std::size_t>(
                            spatial_pipeline_detail::LinearAngularSlot::Count),
                        radial_grid.size(), theta_nodes),
        reconstruction_angular_(
            label + "_reconstruction_angular", registry.size(),
            static_cast<std::size_t>(ReconstructionAngularInput::Count),
            radial_grid.size(), theta_nodes),
        reconstruction_tangent_angular_(
            label + "_reconstruction_tangent_angular", registry.size(),
            static_cast<std::size_t>(ReconstructionAngularInput::Count),
            radial_grid.size(), theta_nodes),
        independent_angular_(
            label + "_independent_angular", registry.size(),
            static_cast<std::size_t>(
                spatial_pipeline_detail::IndependentAngularSlot::Count),
            radial_grid.size(), theta_nodes),
        independent_constraints_(
            label + "_independent_constraints", registry.size(),
            static_cast<std::size_t>(
                spatial_pipeline_detail::IndependentConstraintSlot::Count),
            radial_grid.size(), theta_nodes),
        independent_constraint_terms_(
            label + "_independent_constraint_terms", registry.size(),
            static_cast<std::size_t>(spatial_pipeline_detail::
                                         IndependentConstraintTermSlot::Count),
            radial_grid.size(), theta_nodes),
        source_constraint_evaluator_(
            registry.size(), radial_grid, angular::gauss_legendre(theta_nodes),
            label + "_source_constraint_evaluator"),
        source_extra_(
            label + "_source_extra", registry.size(),
            static_cast<std::size_t>(
                spatial_pipeline_detail::SourceExtraSlot::Count),
            radial_grid.size(), theta_nodes),
        sharp_work_(label + "_sharp_work", registry.size(),
                    static_cast<std::size_t>(
                        spatial_pipeline_detail::SharpSlot::Count),
                    radial_grid.size(), theta_nodes),
        first_scratch_(label + "_first_scratch", registry.size(),
                       static_cast<std::size_t>(
                           TeukolskyRadialScratch::Count),
                       radial_grid.size(), theta_nodes),
        first_tangent_scratch_(
            label + "_first_tangent_scratch", registry.size(),
            static_cast<std::size_t>(TeukolskyRadialScratch::Count),
            radial_grid.size(), theta_nodes),
        second_scratch_(label + "_second_scratch", registry.size(),
                        static_cast<std::size_t>(
                            TeukolskyRadialScratch::Count),
                        radial_grid.size(), theta_nodes),
        reconstruction_radial_(label + "_reconstruction_radial",
                               registry.size(), 7, radial_grid.size(),
                               theta_nodes),
        reconstruction_tangent_radial_(
            label + "_reconstruction_tangent_radial", registry.size(), 7,
            radial_grid.size(), theta_nodes),
        tangent_rhs_(label + "_tangent_rhs", registry.size(),
                     point_pipeline_field_count, radial_grid.size(),
                     theta_nodes),
        zero_forcing_(label + "_zero_forcing", registry.size(),
                      radial_grid.size(), theta_nodes),
        source_fields_(label + "_source_fields", registry.size(),
                       static_cast<std::size_t>(SpatialSourceField::Count),
                       radial_grid.size(), theta_nodes),
        source_field_tangents_(
            label + "_source_field_tangents", registry.size(),
            static_cast<std::size_t>(SpatialSourceField::Count),
            radial_grid.size(), theta_nodes),
        source_derivatives_(
            label + "_source_derivatives", registry.size(),
            static_cast<std::size_t>(SpatialSourceDerivative::Count),
            radial_grid.size(), theta_nodes),
        source_derivative_tangents_(
            label + "_source_derivative_tangents", registry.size(),
            static_cast<std::size_t>(SpatialSourceDerivative::Count),
            radial_grid.size(), theta_nodes),
        inner_source_(registry, radial_grid.size(), theta_nodes,
                      label + "_inner_source"),
        projected_inner_(label + "_projected_inner", registry.size(), 2,
                         radial_grid.size(), theta_nodes),
        projected_inner_tangent_(
            label + "_projected_inner_tangent", registry.size(), 2,
            radial_grid.size(), theta_nodes),
        ethprime_t_(label + "_ethprime_t", registry.size(),
                    radial_grid.size(), theta_nodes),
        source_over_r3_(label + "_source_over_r3", registry.size(),
                        radial_grid.size(), theta_nodes),
        forcing_(label + "_forcing", registry.size(), radial_grid.size(),
                 theta_nodes),
        source_constraint_max_squared_(
            label + "_source_constraint_max_squared", 1),
        source_active_(label + "_source_active", 1),
        rk_workspace_(storage_.value_count()) {
    const int exact_product_nodes =
        (2 * angular_bands.ell_max_first +
         angular_bands.ell_max_second + 2) /
        2;
    if (angular_bands.ell_max_first < 3 ||
        angular_bands.ell_max_second < 3 ||
        theta_nodes <
            std::max({angular_bands.ell_max_first + 1,
                      angular_bands.ell_max_second + 1,
                      exact_product_nodes})) {
      throw std::invalid_argument(
          "full pipeline angular bands require exact-product padding");
    }
    if (dissipation < 0.0) {
      throw std::invalid_argument("pipeline dissipation must be nonnegative");
    }
    if (!std::isfinite(source_policy_.source_start_time) ||
        source_policy_.source_start_time < 0.0 ||
        !std::isfinite(source_policy_.normalized_constraint_tolerance) ||
        source_policy_.normalized_constraint_tolerance < 0.0 ||
        source_policy_.required_consecutive_passes < 1) {
      throw std::invalid_argument("pipeline source policy is invalid");
    }
    Kokkos::deep_copy(execution, storage_.state(), Complex(0.0, 0.0));
    Kokkos::deep_copy(execution, storage_.rhs(), Complex(0.0, 0.0));
    Kokkos::deep_copy(execution, zero_forcing_, Complex(0.0, 0.0));
    Kokkos::deep_copy(execution, source_constraint_max_squared_, 0.0);
    if (source_policy_.mode == SecondOrderSourceMode::Unrestricted &&
        source_policy_.source_start_time == 0.0) {
      source_activation_.active = true;
      source_activation_.activation_time = 0.0;
      source_activation_.consecutive_passes =
          source_policy_.required_consecutive_passes;
    }
    sync_source_activation(execution);
    execution.fence("initialize full spatial pipeline");
    // Pinned Kokkos SYCL lazily reserves a four-entry round-robin indirect
    // kernel-functor pool. Four zero-state graph submissions populate every
    // slot during explicit initialization so production stages are allocation
    // free on every supported backend.
    for (int warmup = 0; warmup < 4; ++warmup) {
      evaluate_rhs(execution, storage_.state(), storage_.rhs());
    }
    execution.fence("warm full spatial pipeline kernels");
  }

  [[nodiscard]] SpatialPipelineStorage& storage() { return storage_; }
  [[nodiscard]] const SpatialPipelineStorage& storage() const {
    return storage_;
  }
  [[nodiscard]] SpatialOuterSourceView source_over_r3() const {
    return source_over_r3_;
  }
  [[nodiscard]] SpatialOuterSourceView forcing() const { return forcing_; }
  [[nodiscard]] Kokkos::View<double*, MemorySpace>
  source_constraint_max_squared() const {
    return source_constraint_max_squared_;
  }
  [[nodiscard]] Kokkos::View<int*, MemorySpace> source_active() const {
    return source_active_;
  }
  [[nodiscard]] SecondOrderSourcePolicy source_policy() const {
    return source_policy_;
  }
  [[nodiscard]] SourceActivationState source_activation_state() const {
    return source_activation_;
  }
  [[nodiscard]] const SourceConstraintNorms& source_constraint_norms() const {
    return source_constraint_norms_;
  }
  [[nodiscard]] std::span<const int> last_step_stage_source_states() const {
    return {last_step_stage_source_states_.data(),
            last_step_stage_source_state_count_};
  }
  [[nodiscard]] FullSpatialStateView independent_reconstruction_constraint_terms()
      const {
    return independent_constraint_terms_;
  }

  void reset_source_activation(const ExecutionSpace& execution) {
    source_activation_ = {};
    source_constraint_norms_ = {};
    if (source_policy_.mode == SecondOrderSourceMode::Unrestricted &&
        source_policy_.source_start_time == 0.0) {
      source_activation_.active = true;
      source_activation_.activation_time = 0.0;
      source_activation_.consecutive_passes =
          source_policy_.required_consecutive_passes;
    }
    Kokkos::deep_copy(execution, source_constraint_max_squared_, 0.0);
    sync_source_activation(execution);
  }

  void restore_source_activation(const ExecutionSpace& execution,
                                 const SourceActivationState& state) {
    validate_source_activation_state(state);
    source_activation_ = state;
    sync_source_activation(execution);
  }

  bool evaluate_source_activation_on_accepted_state(
      const ExecutionSpace& execution, const double accepted_time) {
    return consider_source_activation(execution, accepted_time);
  }
  [[nodiscard]] SpatialInnerSourceView inner_source() const {
    return projected_inner_;
  }
  [[nodiscard]] SpatialInnerSourceView inner_source_tangent() const {
    return projected_inner_tangent_;
  }
  [[nodiscard]] SpatialPairSourceView per_pair_source() const {
    return inner_source_.per_pair_value();
  }
  [[nodiscard]] FullSpatialStateView reconstruction_radial_derivatives()
      const {
    return reconstruction_radial_;
  }
  [[nodiscard]] FullSpatialStateView reconstruction_angular_inputs() const {
    return reconstruction_angular_;
  }
  [[nodiscard]] FullSpatialStateView independent_reconstruction_constraints()
      const {
    return independent_constraints_;
  }
  [[nodiscard]] KerrParameters background() const { return background_; }
  [[nodiscard]] PipelineAngularBands angular_bands() const {
    return angular_bands_;
  }

  template <class StageView, class OutputView>
  void evaluate_rhs(const ExecutionSpace& execution, const StageView& stage,
                    const OutputView& output,
                    SpatialPipelineTiming* timing = nullptr) {
    evaluate_rhs_at_time(execution, stage, output, 0.0, timing);
  }

  template <class StageView, class OutputView>
  void evaluate_rhs_at_time(const ExecutionSpace& execution,
                            const StageView& stage, const OutputView& output,
                            const double stage_time,
                            SpatialPipelineTiming* timing = nullptr) {
    (void)stage_time;
    static_assert(StageView::rank == 4 && OutputView::rank == 4,
                  "pipeline stages must have rank four");
    validate_stage(stage, output);
    using Clock = std::chrono::steady_clock;
    Clock::time_point timing_start;
    if (timing != nullptr) {
      *timing = {};
      execution.fence("begin profiled full spatial RHS");
      timing_start = Clock::now();
    }
    const auto record_timing = [&](double& destination,
                                   const char* fence_label) {
      if (timing == nullptr) return;
      execution.fence(fence_label);
      const auto now = Clock::now();
      destination = std::chrono::duration<double>(now - timing_start).count();
      timing_start = now;
    };

    first_angular_.laplacian(
        execution, stage,
        static_cast<std::size_t>(PipelineField::FirstPsi), linear_angular_,
        static_cast<std::size_t>(
            spatial_pipeline_detail::LinearAngularSlot::First));
    const auto first_laplacian = Kokkos::subview(
        linear_angular_, Kokkos::ALL,
        static_cast<std::size_t>(
            spatial_pipeline_detail::LinearAngularSlot::First),
        Kokkos::ALL, Kokkos::ALL);
    evaluate_sbp_teukolsky_full_stage_rhs(
        execution, storage_.radial_grid(), teukolsky_, storage_.theta(),
        storage_.modes(), stage, first_laplacian, zero_forcing_, reduction_,
        first_scratch_, output, dissipation_,
        spatial_pipeline_detail::first_fields,
        spatial_pipeline_detail::first_fields);
    project_teukolsky_rhs(execution, output,
                          spatial_pipeline_detail::first_fields);
    if (timing != nullptr) {
      record_timing(timing->first_linear_seconds,
                    "profile first linear spatial RHS");
    }

    evaluate_reconstruction_chain(execution, stage, output,
                                  reconstruction_radial_,
                                  reconstruction_angular_);
    evaluate_independent_reconstruction_constraints(execution, stage, output);
    if (timing != nullptr) {
      record_timing(timing->reconstruction_seconds,
                    "profile reconstruction spatial RHS");
    }

    first_angular_.laplacian(
        execution, output,
        static_cast<std::size_t>(PipelineField::FirstPsi), linear_angular_,
        static_cast<std::size_t>(
            spatial_pipeline_detail::LinearAngularSlot::FirstTangent));
    const auto tangent_laplacian = Kokkos::subview(
        linear_angular_, Kokkos::ALL,
        static_cast<std::size_t>(
            spatial_pipeline_detail::LinearAngularSlot::FirstTangent),
        Kokkos::ALL, Kokkos::ALL);
    evaluate_sbp_teukolsky_full_stage_rhs(
        execution, storage_.radial_grid(), teukolsky_, storage_.theta(),
        storage_.modes(), output, tangent_laplacian, zero_forcing_, reduction_,
        first_tangent_scratch_, tangent_rhs_, dissipation_,
        spatial_pipeline_detail::first_fields,
        spatial_pipeline_detail::first_fields);
    project_teukolsky_rhs(execution, tangent_rhs_,
                          spatial_pipeline_detail::first_fields);
    evaluate_reconstruction_chain(execution, output, tangent_rhs_,
                                  reconstruction_tangent_radial_,
                                  reconstruction_tangent_angular_);
    if (timing != nullptr) {
      record_timing(timing->tangent_seconds,
                    "profile tangent spatial RHS");
    }

    evaluate_source_angular_derivatives(execution, stage, output,
                                        tangent_rhs_);
    prepare_source_inputs(execution, stage, output, tangent_rhs_);
    evaluate_spatial_inner_source_tangent(
        execution, storage_.radial_grid(), background_, storage_.cos_theta(),
        storage_.sin_theta(), source_fields_, source_field_tangents_,
        source_derivatives_, source_derivative_tangents_, inner_source_);
    project_inner_source(execution);
    apply_source_activation(execution);
    if (timing != nullptr) {
      record_timing(timing->source_seconds,
                    "profile quadratic spatial source");
    }

    second_angular_.laplacian(
        execution, stage,
        static_cast<std::size_t>(PipelineField::SecondPsi), linear_angular_,
        static_cast<std::size_t>(
            spatial_pipeline_detail::LinearAngularSlot::Second));
    const auto second_laplacian = Kokkos::subview(
        linear_angular_, Kokkos::ALL,
        static_cast<std::size_t>(
            spatial_pipeline_detail::LinearAngularSlot::Second),
        Kokkos::ALL, Kokkos::ALL);
    evaluate_sbp_teukolsky_full_stage_rhs(
        execution, storage_.radial_grid(), teukolsky_, storage_.theta(),
        storage_.modes(), stage, second_laplacian, forcing_, reduction_,
        second_scratch_, output, dissipation_,
        spatial_pipeline_detail::second_fields,
        spatial_pipeline_detail::second_fields);
    project_teukolsky_rhs(execution, output,
                          spatial_pipeline_detail::second_fields);
    if (timing != nullptr) {
      record_timing(timing->second_linear_seconds,
                    "profile second linear spatial RHS");
    }
  }

  void step(const ExecutionSpace& execution, const double time,
            const double time_step) {
    if (!std::isfinite(time) || !std::isfinite(time_step) ||
        time < 0.0 || !(time_step > 0.0)) {
      throw std::invalid_argument("pipeline step requires finite positive time data");
    }
    last_step_stage_source_state_count_ = 0;
    const double end_time = time + time_step;
    if (!std::isfinite(end_time)) {
      throw std::invalid_argument("pipeline step end time is not finite");
    }
    const double start_time = source_policy_.source_start_time;
    if (!source_activation_.active && time >= start_time) {
      consider_source_activation(execution, time);
    }
    if (!source_activation_.active && time < start_time &&
        start_time < end_time) {
      advance_fixed_source_step(execution, time, start_time - time);
      consider_source_activation(execution, start_time);
      advance_fixed_source_step(execution, start_time,
                                end_time - start_time);
    } else {
      advance_fixed_source_step(execution, time, time_step);
    }
    // Accepted endpoint decisions affect only the following step. In
    // particular, an event at end_time contributes nothing to the preceding
    // RK4 update.
    if (!source_activation_.active && end_time >= start_time) {
      consider_source_activation(execution, end_time);
    }
  }

 private:
  template <class StageView, class OutputView>
  void validate_stage(const StageView& stage, const OutputView& output) const {
    const auto valid = [&](const auto& view) {
      return view.extent(0) == storage_.mode_count() &&
             view.extent(1) == point_pipeline_field_count &&
             view.extent(2) == storage_.radial_count() &&
             view.extent(3) == storage_.theta_count();
    };
    if (!valid(stage) || !valid(output)) {
      throw std::invalid_argument("full pipeline stage extents do not match");
    }
  }

  template <class StageView, class OutputView>
  void evaluate_reconstruction_chain(
      const ExecutionSpace& execution, const StageView& stage,
      const OutputView& output, const FullSpatialStateView& radial_derivatives,
      const FullSpatialStateView& angular_values) {
    using namespace spatial_pipeline_detail;
    evaluate_sbp_reconstruction_full_radial_derivatives(
        execution, storage_.radial_grid(), stage, radial_derivatives,
        reconstruction_fields, compact_reconstruction_fields);
    first_angular_.eth(
        execution, stage, static_cast<std::size_t>(PipelineField::FirstPsi),
        output, static_cast<std::size_t>(PipelineField::FirstPsi),
        storage_.radius(), storage_.sin_theta(), storage_.cos_theta(),
        angular_values,
        static_cast<std::size_t>(ReconstructionAngularInput::Eth1F));
    const auto psi4 = Kokkos::subview(
        stage, Kokkos::ALL,
        static_cast<std::size_t>(PipelineField::FirstPsi), Kokkos::ALL,
        Kokkos::ALL);
    const auto eth1_f = Kokkos::subview(
        angular_values, Kokkos::ALL,
        static_cast<std::size_t>(ReconstructionAngularInput::Eth1F),
        Kokkos::ALL, Kokkos::ALL);
    evaluate_sbp_reconstruction_full_pass1(
        execution, storage_.radial_grid(), background_, storage_.theta(),
        storage_.sharp_indices(), stage, psi4, eth1_f, radial_derivatives,
        output, reconstruction_fields, compact_reconstruction_fields,
        reconstruction_fields);
    g_angular_.project(execution, output, reconstruction_fields.G, output,
                       reconstruction_fields.G);
    first_angular_.project(execution, output, reconstruction_fields.Lambda,
                           output, reconstruction_fields.Lambda);

    g_angular_.eth(
        execution, stage, static_cast<std::size_t>(PipelineField::G), output,
        static_cast<std::size_t>(PipelineField::G), storage_.radius(),
        storage_.sin_theta(), storage_.cos_theta(), angular_values,
        static_cast<std::size_t>(ReconstructionAngularInput::Eth2G));
    const auto eth2_g = Kokkos::subview(
        angular_values, Kokkos::ALL,
        static_cast<std::size_t>(ReconstructionAngularInput::Eth2G),
        Kokkos::ALL, Kokkos::ALL);
    evaluate_sbp_reconstruction_full_pass2(
        execution, storage_.radial_grid(), background_, storage_.theta(),
        storage_.sharp_indices(), stage, psi4, eth2_g, radial_derivatives,
        output, reconstruction_fields, compact_reconstruction_fields,
        reconstruction_fields);
    scalar_angular_.project(execution, output, reconstruction_fields.H,
                            output, reconstruction_fields.H);
    b_angular_.project(execution, output, reconstruction_fields.B, output,
                       reconstruction_fields.B);
    pi_angular_.project(execution, output, reconstruction_fields.Pi, output,
                        reconstruction_fields.Pi);
    c_angular_.project(execution, output, reconstruction_fields.C, output,
                       reconstruction_fields.C);

    c_angular_.eth(
        execution, stage, static_cast<std::size_t>(PipelineField::C), output,
        static_cast<std::size_t>(PipelineField::C), storage_.radius(),
        storage_.sin_theta(), storage_.cos_theta(), angular_values,
        static_cast<std::size_t>(ReconstructionAngularInput::Eth2C));
    pi_angular_.eth(
        execution, stage, static_cast<std::size_t>(PipelineField::Pi), output,
        static_cast<std::size_t>(PipelineField::Pi), storage_.radius(),
        storage_.sin_theta(), storage_.cos_theta(), angular_values,
        static_cast<std::size_t>(ReconstructionAngularInput::Eth2Pi));
    pack_sharp_fields(execution, stage, output);
    b_sharp_angular_.ethprime(
        execution, sharp_work_, static_cast<std::size_t>(SharpSlot::B),
        sharp_work_, static_cast<std::size_t>(SharpSlot::BDt),
        storage_.radius(), storage_.sin_theta(), storage_.cos_theta(),
        angular_values,
        static_cast<std::size_t>(
            ReconstructionAngularInput::EthPrime1BSharp));
    c_sharp_angular_.ethprime(
        execution, sharp_work_, static_cast<std::size_t>(SharpSlot::C),
        sharp_work_, static_cast<std::size_t>(SharpSlot::CDt),
        storage_.radius(), storage_.sin_theta(), storage_.cos_theta(),
        angular_values,
        static_cast<std::size_t>(
            ReconstructionAngularInput::EthPrime2CSharp));
    evaluate_sbp_reconstruction_full_pass3(
        execution, storage_.radial_grid(), background_, storage_.theta(),
        storage_.sharp_indices(), stage, psi4, angular_values,
        radial_derivatives, output, reconstruction_fields,
        compact_reconstruction_fields, reconstruction_fields);
    scalar_angular_.project(execution, output, reconstruction_fields.U,
                            output, reconstruction_fields.U);
  }

  template <class OutputView>
  void project_teukolsky_rhs(
      const ExecutionSpace& execution, const OutputView& output,
      const TeukolskyFullFieldOffsets& fields) {
    auto& angular = fields.P == spatial_pipeline_detail::second_fields.P
                        ? second_angular_
                        : first_angular_;
    angular.project(execution, output, fields.P, output, fields.P);
    angular.project(execution, output, fields.Q, output, fields.Q);
    angular.project(execution, output, fields.Psi, output, fields.Psi);
  }

  template <class StageView, class DtView>
  void evaluate_independent_reconstruction_constraints(
      const ExecutionSpace& execution, const StageView& stage,
      const DtView& dt) {
    using namespace spatial_pipeline_detail;
    g_angular_.ethprime(
        execution, stage, reconstruction_fields.G, dt,
        reconstruction_fields.G, storage_.radius(), storage_.sin_theta(),
        storage_.cos_theta(), independent_angular_,
        static_cast<std::size_t>(IndependentAngularSlot::EthPrime2G));
    scalar_angular_.ethprime(
        execution, stage, reconstruction_fields.H, dt,
        reconstruction_fields.H, storage_.radius(), storage_.sin_theta(),
        storage_.cos_theta(), independent_angular_,
        static_cast<std::size_t>(IndependentAngularSlot::EthPrime3H));

    const auto constraints = independent_constraints_;
    const auto constraint_terms = independent_constraint_terms_;
    const auto angular_values = independent_angular_;
    const auto first_scratch = first_scratch_;
    const auto reconstruction_radial = reconstruction_radial_;
    const auto sharp = storage_.sharp_indices();
    const auto modes = storage_.modes();
    const auto radius = storage_.radius();
    const auto cos_theta = storage_.cos_theta();
    const auto sin_theta = storage_.sin_theta();
    const KerrParameters parameters = background_;
    const std::size_t radial_count = storage_.radial_count();
    const std::size_t theta_count = storage_.theta_count();
    const std::size_t total = storage_.mode_count() * radial_count * theta_count;
    Kokkos::parallel_for(
        "teuk_independent_reconstruction_constraints",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t plane = radial_count * theta_count;
          const std::size_t mode = flat / plane;
          const std::size_t within = flat - mode * plane;
          const std::size_t radial = within / theta_count;
          const std::size_t theta = within - radial * theta_count;
          const double coordinate = radius(radial);
          const auto background = kerr_background_point(
              parameters, coordinate, cos_theta(theta), sin_theta(theta));
          const Complex F = stage(mode, first_fields.Psi, radial, theta);
          const Complex G =
              stage(mode, reconstruction_fields.G, radial, theta);
          const Complex H =
              stage(mode, reconstruction_fields.H, radial, theta);
          const Complex thorn1_F = thorn_n_point(
              F, dt(mode, first_fields.Psi, radial, theta),
              first_scratch(
                  mode,
                  static_cast<std::size_t>(
                      TeukolskyRadialScratch::RadialDerivativePsi),
                  radial, theta),
              1, -2, -2, modes(mode), coordinate, cos_theta(theta),
              parameters.mass, parameters.spin,
              parameters.compactification_length, background.epsilon0);
          const Complex thorn2_G = thorn_n_point(
              G, dt(mode, reconstruction_fields.G, radial, theta),
              reconstruction_radial(mode, compact_reconstruction_fields.G,
                                    radial, theta),
              2, -1, -1, modes(mode), coordinate, cos_theta(theta),
              parameters.mass, parameters.spin,
              parameters.compactification_length, background.epsilon0);
          const auto terms = independent_reconstruction_constraint_terms_point(
              coordinate, background, F, G, H,
              stage(mode, reconstruction_fields.Lambda, radial, theta),
              stage(mode, reconstruction_fields.Pi, radial, theta),
              stage(mode, reconstruction_fields.B, radial, theta),
              stage(mode, reconstruction_fields.C, radial, theta),
              stage(mode, reconstruction_fields.U, radial, theta),
              Kokkos::conj(stage(sharp(mode), reconstruction_fields.U, radial,
                                 theta)),
              thorn1_F, thorn2_G,
              angular_values(
                  mode,
                  static_cast<std::size_t>(
                      IndependentAngularSlot::EthPrime2G),
                  radial, theta),
              angular_values(
                  mode,
                  static_cast<std::size_t>(
                      IndependentAngularSlot::EthPrime3H),
                  radial, theta));
          IndependentReconstructionConstraints residuals{
              terms.psi3_bianchi.left - terms.psi3_bianchi.right,
              terms.psi2_bianchi.left - terms.psi2_bianchi.right,
              terms.hll_reality.left - terms.hll_reality.right};
          constraints(mode,
                      static_cast<std::size_t>(
                          IndependentConstraintSlot::Psi3Bianchi),
                      radial, theta) = residuals.psi3_bianchi;
          constraints(mode,
                      static_cast<std::size_t>(
                          IndependentConstraintSlot::Psi2Bianchi),
                      radial, theta) = residuals.psi2_bianchi;
          constraints(mode,
                      static_cast<std::size_t>(
                          IndependentConstraintSlot::HllReality),
                      radial, theta) = residuals.hll_reality;
          constraint_terms(
              mode, static_cast<std::size_t>(
                        IndependentConstraintTermSlot::Psi3Left),
              radial, theta) = terms.psi3_bianchi.left;
          constraint_terms(
              mode, static_cast<std::size_t>(
                        IndependentConstraintTermSlot::Psi3Right),
              radial, theta) = terms.psi3_bianchi.right;
          constraint_terms(
              mode, static_cast<std::size_t>(
                        IndependentConstraintTermSlot::Psi2Left),
              radial, theta) = terms.psi2_bianchi.left;
          constraint_terms(
              mode, static_cast<std::size_t>(
                        IndependentConstraintTermSlot::Psi2Right),
              radial, theta) = terms.psi2_bianchi.right;
          constraint_terms(
              mode, static_cast<std::size_t>(
                        IndependentConstraintTermSlot::HllLeft),
              radial, theta) = terms.hll_reality.left;
          constraint_terms(
              mode, static_cast<std::size_t>(
                        IndependentConstraintTermSlot::HllRight),
              radial, theta) = terms.hll_reality.right;
        });
  }

  template <class StageView, class DtView>
  void pack_sharp_fields(const ExecutionSpace& execution,
                         const StageView& stage, const DtView& dt) {
    using namespace spatial_pipeline_detail;
    const auto sharp = storage_.sharp_indices();
    const auto work = sharp_work_;
    const std::size_t radial_count = storage_.radial_count();
    const std::size_t theta_count = storage_.theta_count();
    const std::size_t total = storage_.mode_count() * radial_count * theta_count;
    Kokkos::parallel_for(
        "teuk_pack_sharp_reconstruction_fields",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t plane = radial_count * theta_count;
          const std::size_t mode = flat / plane;
          const std::size_t within = flat - mode * plane;
          const std::size_t radial = within / theta_count;
          const std::size_t theta = within - radial * theta_count;
          const std::size_t source = sharp(mode);
          work(mode, static_cast<std::size_t>(SharpSlot::B), radial, theta) =
              Kokkos::conj(stage(
                  source, static_cast<std::size_t>(PipelineField::B), radial,
                  theta));
          work(mode, static_cast<std::size_t>(SharpSlot::C), radial, theta) =
              Kokkos::conj(stage(
                  source, static_cast<std::size_t>(PipelineField::C), radial,
                  theta));
          work(mode, static_cast<std::size_t>(SharpSlot::BDt), radial, theta) =
              Kokkos::conj(dt(
                  source, static_cast<std::size_t>(PipelineField::B), radial,
                  theta));
          work(mode, static_cast<std::size_t>(SharpSlot::CDt), radial, theta) =
              Kokkos::conj(dt(
                  source, static_cast<std::size_t>(PipelineField::C), radial,
                  theta));
        });
  }

  template <class StageView, class DtView, class DDtView>
  void evaluate_source_angular_derivatives(
      const ExecutionSpace& execution, const StageView& stage,
      const DtView& dt, const DDtView& ddt) {
    using namespace spatial_pipeline_detail;
    b_angular_.eth(
        execution, stage, static_cast<std::size_t>(PipelineField::B), dt,
        static_cast<std::size_t>(PipelineField::B), storage_.radius(),
        storage_.sin_theta(), storage_.cos_theta(), source_extra_,
        static_cast<std::size_t>(SourceExtraSlot::Eth1B));
    first_angular_.ethprime(
        execution, stage, static_cast<std::size_t>(PipelineField::FirstPsi),
        dt, static_cast<std::size_t>(PipelineField::FirstPsi),
        storage_.radius(), storage_.sin_theta(), storage_.cos_theta(),
        source_extra_,
        static_cast<std::size_t>(SourceExtraSlot::EthPrime1F));
    b_angular_.eth(
        execution, dt, static_cast<std::size_t>(PipelineField::B), ddt,
        static_cast<std::size_t>(PipelineField::B), storage_.radius(),
        storage_.sin_theta(), storage_.cos_theta(), source_extra_,
        static_cast<std::size_t>(SourceExtraSlot::Eth1BTangent));
    first_angular_.ethprime(
        execution, dt, static_cast<std::size_t>(PipelineField::FirstPsi), ddt,
        static_cast<std::size_t>(PipelineField::FirstPsi), storage_.radius(),
        storage_.sin_theta(), storage_.cos_theta(), source_extra_,
        static_cast<std::size_t>(SourceExtraSlot::EthPrime1FTangent));
  }

  template <class StageView, class DtView, class DDtView>
  void prepare_source_inputs(const ExecutionSpace& execution,
                             const StageView& stage, const DtView& dt,
                             const DDtView& ddt) {
    using namespace spatial_pipeline_detail;
    const auto fields = source_fields_;
    const auto field_tangents = source_field_tangents_;
    const auto derivatives = source_derivatives_;
    const auto derivative_tangents = source_derivative_tangents_;
    const auto first_scratch = first_scratch_;
    const auto tangent_scratch = first_tangent_scratch_;
    const auto recon_dr = reconstruction_radial_;
    const auto recon_tangent_dr = reconstruction_tangent_radial_;
    const auto recon_angular = reconstruction_angular_;
    const auto recon_tangent_angular = reconstruction_tangent_angular_;
    const auto extra = source_extra_;
    const auto sharp = storage_.sharp_indices();
    const auto grid = storage_.radial_grid();
    const std::size_t radial_count = storage_.radial_count();
    const std::size_t theta_count = storage_.theta_count();
    const std::size_t total = storage_.mode_count() * radial_count * theta_count;
    const KerrParameters parameters = background_;
    Kokkos::parallel_for(
        "teuk_prepare_spatial_source_inputs",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, total),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t plane = radial_count * theta_count;
          const std::size_t mode = flat / plane;
          const std::size_t within = flat - mode * plane;
          const std::size_t radial = within / theta_count;
          const std::size_t theta = within - radial * theta_count;
          const std::size_t sharp_mode = sharp(mode);
          const double radius = grid.coordinate(radial);
          const std::size_t state_slots[8] = {
              static_cast<std::size_t>(PipelineField::FirstPsi),
              static_cast<std::size_t>(PipelineField::G),
              static_cast<std::size_t>(PipelineField::H),
              static_cast<std::size_t>(PipelineField::Lambda),
              static_cast<std::size_t>(PipelineField::Pi),
              static_cast<std::size_t>(PipelineField::B),
              static_cast<std::size_t>(PipelineField::C),
              static_cast<std::size_t>(PipelineField::U)};
          const std::size_t source_slots[8] = {
              static_cast<std::size_t>(SpatialSourceField::F),
              static_cast<std::size_t>(SpatialSourceField::G),
              static_cast<std::size_t>(SpatialSourceField::H),
              static_cast<std::size_t>(SpatialSourceField::Lambda),
              static_cast<std::size_t>(SpatialSourceField::Pi),
              static_cast<std::size_t>(SpatialSourceField::B),
              static_cast<std::size_t>(SpatialSourceField::C),
              static_cast<std::size_t>(SpatialSourceField::U)};
          for (std::size_t i = 0; i < 8; ++i) {
            fields(mode, source_slots[i], radial, theta) =
                stage(mode, state_slots[i], radial, theta);
            field_tangents(mode, source_slots[i], radial, theta) =
                dt(mode, state_slots[i], radial, theta);
          }
          const Complex delta1_f = delta_n_point(
              stage(mode, first_fields.Psi, radial, theta),
              dt(mode, first_fields.Psi, radial, theta),
              first_scratch(
                  mode,
                  static_cast<std::size_t>(
                      TeukolskyRadialScratch::RadialDerivativePsi),
                  radial, theta),
              1, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta1_f_dt = delta_n_point(
              dt(mode, first_fields.Psi, radial, theta),
              ddt(mode, first_fields.Psi, radial, theta),
              tangent_scratch(
                  mode,
                  static_cast<std::size_t>(
                      TeukolskyRadialScratch::RadialDerivativePsi),
                  radial, theta),
              1, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta3_u = delta_n_point(
              stage(mode, reconstruction_fields.U, radial, theta),
              dt(mode, reconstruction_fields.U, radial, theta),
              recon_dr(mode, compact_reconstruction_fields.U, radial, theta),
              3, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta3_u_dt = delta_n_point(
              dt(mode, reconstruction_fields.U, radial, theta),
              ddt(mode, reconstruction_fields.U, radial, theta),
              recon_tangent_dr(mode, compact_reconstruction_fields.U, radial,
                               theta),
              3, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta2_c = delta_n_point(
              stage(mode, reconstruction_fields.C, radial, theta),
              dt(mode, reconstruction_fields.C, radial, theta),
              recon_dr(mode, compact_reconstruction_fields.C, radial, theta),
              2, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta2_c_dt = delta_n_point(
              dt(mode, reconstruction_fields.C, radial, theta),
              ddt(mode, reconstruction_fields.C, radial, theta),
              recon_tangent_dr(mode, compact_reconstruction_fields.C, radial,
                               theta),
              2, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta2_g = delta_n_point(
              stage(mode, reconstruction_fields.G, radial, theta),
              dt(mode, reconstruction_fields.G, radial, theta),
              recon_dr(mode, compact_reconstruction_fields.G, radial, theta),
              2, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta2_g_dt = delta_n_point(
              dt(mode, reconstruction_fields.G, radial, theta),
              ddt(mode, reconstruction_fields.G, radial, theta),
              recon_tangent_dr(mode, compact_reconstruction_fields.G, radial,
                               theta),
              2, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta2_c_sharp = delta_n_point(
              Kokkos::conj(stage(sharp_mode, reconstruction_fields.C, radial,
                                 theta)),
              Kokkos::conj(dt(sharp_mode, reconstruction_fields.C, radial,
                              theta)),
              Kokkos::conj(recon_dr(sharp_mode,
                                    compact_reconstruction_fields.C, radial,
                                    theta)),
              2, radius, parameters.mass,
              parameters.compactification_length);
          const Complex delta2_c_sharp_dt = delta_n_point(
              Kokkos::conj(dt(sharp_mode, reconstruction_fields.C, radial,
                              theta)),
              Kokkos::conj(ddt(sharp_mode, reconstruction_fields.C, radial,
                               theta)),
              Kokkos::conj(recon_tangent_dr(
                  sharp_mode, compact_reconstruction_fields.C, radial,
                  theta)),
              2, radius, parameters.mass,
              parameters.compactification_length);

          const auto set_derivative = [&](const SpatialSourceDerivative slot,
                                          const Complex value,
                                          const Complex tangent) {
            const std::size_t i = static_cast<std::size_t>(slot);
            derivatives(mode, i, radial, theta) = value;
            derivative_tangents(mode, i, radial, theta) = tangent;
          };
          set_derivative(SpatialSourceDerivative::Delta1F, delta1_f,
                         delta1_f_dt);
          set_derivative(SpatialSourceDerivative::Delta3U, delta3_u,
                         delta3_u_dt);
          set_derivative(
              SpatialSourceDerivative::Eth2C,
              recon_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::Eth2C),
                  radial, theta),
              recon_tangent_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::Eth2C),
                  radial, theta));
          set_derivative(
              SpatialSourceDerivative::EthPrime2CSharp,
              recon_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::EthPrime2CSharp),
                  radial, theta),
              recon_tangent_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::EthPrime2CSharp),
                  radial, theta));
          set_derivative(
              SpatialSourceDerivative::Eth1B,
              extra(mode, static_cast<std::size_t>(SourceExtraSlot::Eth1B),
                    radial, theta),
              extra(mode,
                    static_cast<std::size_t>(
                        SourceExtraSlot::Eth1BTangent),
                    radial, theta));
          set_derivative(SpatialSourceDerivative::Delta2C, delta2_c,
                         delta2_c_dt);
          set_derivative(SpatialSourceDerivative::Delta2G, delta2_g,
                         delta2_g_dt);
          set_derivative(
              SpatialSourceDerivative::Eth2G,
              recon_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::Eth2G),
                  radial, theta),
              recon_tangent_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::Eth2G),
                  radial, theta));
          set_derivative(
              SpatialSourceDerivative::EthPrime1F,
              extra(mode,
                    static_cast<std::size_t>(SourceExtraSlot::EthPrime1F),
                    radial, theta),
              extra(mode,
                    static_cast<std::size_t>(
                        SourceExtraSlot::EthPrime1FTangent),
                    radial, theta));
          set_derivative(SpatialSourceDerivative::Delta2CSharp,
                         delta2_c_sharp, delta2_c_sharp_dt);
          set_derivative(
              SpatialSourceDerivative::EthPrime1BSharp,
              recon_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::EthPrime1BSharp),
                  radial, theta),
              recon_tangent_angular(
                  mode,
                  static_cast<std::size_t>(
                      ReconstructionAngularInput::EthPrime1BSharp),
                  radial, theta));
        });
  }

  void project_inner_source(const ExecutionSpace& execution) {
    constexpr std::size_t D =
        static_cast<std::size_t>(SpatialInnerSourceComponent::D);
    constexpr std::size_t T =
        static_cast<std::size_t>(SpatialInnerSourceComponent::T);
    second_angular_.project(execution, inner_source_.summed_value(), D,
                            projected_inner_, D);
    second_angular_.project(execution, inner_source_.summed_tangent(), D,
                            projected_inner_tangent_, D);
    t_angular_.project(execution, inner_source_.summed_value(), T,
                       projected_inner_, T);
    t_angular_.project(execution, inner_source_.summed_tangent(), T,
                       projected_inner_tangent_, T);
    using EthPrime4D =
        Kokkos::View<Complex****, Kokkos::LayoutRight, MemorySpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    EthPrime4D ethprime_view(ethprime_t_.data(), storage_.mode_count(), 1,
                             storage_.radial_count(), storage_.theta_count());
    t_angular_.ethprime(
        execution, projected_inner_, T, projected_inner_tangent_, T,
        storage_.radius(), storage_.sin_theta(), storage_.cos_theta(),
        ethprime_view, 0);
    evaluate_spatial_outer_source_from_ethprime(
        execution, storage_.radial_grid(), background_, storage_.cos_theta(),
        storage_.sin_theta(), projected_inner_, projected_inner_tangent_,
        ethprime_t_, source_over_r3_, forcing_);
  }

  void apply_source_activation(const ExecutionSpace& execution) {
    const auto forcing = forcing_;
    const std::size_t forcing_points = forcing.size();
    const bool active = source_activation_.active;
    Kokkos::parallel_for(
        "apply_latched_second_order_source_state",
        Kokkos::RangePolicy<ExecutionSpace>(execution, 0, forcing_points),
        KOKKOS_LAMBDA(const std::size_t flat) {
          if (!active) {
            forcing.data()[flat] = Complex(0.0, 0.0);
          }
        });
  }

  static void validate_source_activation_state(
      const SourceActivationState& state) {
    if (state.consecutive_passes < 0 ||
        !std::isfinite(state.activation_time) ||
        !std::isfinite(state.last_eligibility_time) ||
        (state.active && state.activation_time < 0.0) ||
        (!state.active && state.activation_time != -1.0) ||
        state.last_eligibility_time < -1.0) {
      throw std::invalid_argument("invalid second-order source activation state");
    }
  }

  void sync_source_activation(const ExecutionSpace& execution) {
    Kokkos::deep_copy(execution, source_active_,
                      source_activation_.active ? 1 : 0);
  }

  bool consider_source_activation(const ExecutionSpace& execution,
                                  const double accepted_time) {
    if (!std::isfinite(accepted_time) || accepted_time < 0.0) {
      throw std::invalid_argument(
          "source activation requires a finite accepted-state time");
    }
    if (source_activation_.active ||
        accepted_time < source_policy_.source_start_time ||
        accepted_time <= source_activation_.last_eligibility_time) {
      return source_activation_.active;
    }
    if (source_policy_.mode == SecondOrderSourceMode::Disabled) return false;
    bool passed = true;
    if (source_policy_.mode == SecondOrderSourceMode::Unrestricted) {
      source_activation_.last_eligibility_time = accepted_time;
      source_activation_.consecutive_passes =
          source_policy_.required_consecutive_passes;
      source_activation_.active = true;
      source_activation_.activation_time = accepted_time;
      sync_source_activation(execution);
      return true;
    }
    evaluate_rhs_at_time(execution, storage_.state(), storage_.rhs(),
                         accepted_time);
    source_constraint_norms_ = source_constraint_evaluator_.sample(
        execution, independent_constraints_, independent_constraint_terms_);
    passed = source_constraint_norms_.all_finite &&
             source_constraint_norms_.controlling_normalized <=
                 source_policy_.normalized_constraint_tolerance;
    const double maximum = std::max(
        {source_constraint_norms_.psi3_bianchi.absolute_maximum,
         source_constraint_norms_.psi2_bianchi.absolute_maximum,
         source_constraint_norms_.hll_reality.absolute_maximum});
    Kokkos::deep_copy(execution, source_constraint_max_squared_,
                      maximum * maximum);
    source_activation_.last_eligibility_time = accepted_time;
    source_activation_.consecutive_passes = passed
        ? std::min(source_activation_.consecutive_passes + 1,
                   source_policy_.required_consecutive_passes)
        : 0;
    if (source_activation_.consecutive_passes >=
        source_policy_.required_consecutive_passes) {
      source_activation_.active = true;
      source_activation_.activation_time = accepted_time;
      sync_source_activation(execution);
    }
    return source_activation_.active;
  }

  void advance_fixed_source_step(const ExecutionSpace& execution,
                                 const double time,
                                 const double time_step) {
    if (!(time_step > 0.0)) return;
    auto flat_state = storage_.flat_state();
    const auto rhs = [this](const ExecutionSpace& stage_execution,
                            const double stage_time, const auto& flat_stage,
                            const auto& flat_output) {
      if (last_step_stage_source_state_count_ >=
          last_step_stage_source_states_.size()) {
        throw std::runtime_error("source-event split exceeded RK stage capacity");
      }
      last_step_stage_source_states_[last_step_stage_source_state_count_++] =
          source_activation_.active ? 1 : 0;
      evaluate_rhs_at_time(stage_execution, storage_.reshape(flat_stage),
                           storage_.reshape(flat_output), stage_time);
    };
    device_classical_rk4_step(execution, flat_state, time, time_step, rhs,
                              rk_workspace_);
  }

  ModeRegistry registry_;
  PipelineAngularBands angular_bands_;
  KerrParameters background_;
  TeukolskyParameters teukolsky_;
  double dissipation_;
  ReductionEvolution reduction_;
  SecondOrderSourcePolicy source_policy_;
  SpatialPipelineStorage storage_;
  SignedModeAngularCoordinator<> first_angular_;
  SignedModeAngularCoordinator<> second_angular_;
  SignedModeAngularCoordinator<> g_angular_;
  SignedModeAngularCoordinator<> b_angular_;
  SignedModeAngularCoordinator<> pi_angular_;
  SignedModeAngularCoordinator<> c_angular_;
  SignedModeAngularCoordinator<> scalar_angular_;
  SignedModeAngularCoordinator<> b_sharp_angular_;
  SignedModeAngularCoordinator<> c_sharp_angular_;
  SignedModeAngularCoordinator<> t_angular_;
  FullSpatialStateView linear_angular_;
  FullSpatialStateView reconstruction_angular_;
  FullSpatialStateView reconstruction_tangent_angular_;
  FullSpatialStateView independent_angular_;
  FullSpatialStateView independent_constraints_;
  FullSpatialStateView independent_constraint_terms_;
  SourceConstraintEvaluator source_constraint_evaluator_;
  SourceConstraintNorms source_constraint_norms_;
  SourceActivationState source_activation_;
  FullSpatialStateView source_extra_;
  FullSpatialStateView sharp_work_;
  FullSpatialStateView first_scratch_;
  FullSpatialStateView first_tangent_scratch_;
  FullSpatialStateView second_scratch_;
  FullSpatialStateView reconstruction_radial_;
  FullSpatialStateView reconstruction_tangent_radial_;
  FullSpatialStateView tangent_rhs_;
  FullSpatialValueView zero_forcing_;
  SpatialSourceFieldView source_fields_;
  SpatialSourceFieldView source_field_tangents_;
  SpatialSourceDerivativeView source_derivatives_;
  SpatialSourceDerivativeView source_derivative_tangents_;
  SpatialInnerSourceTangentWorkspace inner_source_;
  SpatialInnerSourceView projected_inner_;
  SpatialInnerSourceView projected_inner_tangent_;
  SpatialOuterSourceView ethprime_t_;
  SpatialOuterSourceView source_over_r3_;
  SpatialOuterSourceView forcing_;
  Kokkos::View<double*, MemorySpace> source_constraint_max_squared_;
  Kokkos::View<int*, MemorySpace> source_active_;
  DeviceRK4Workspace<Complex> rk_workspace_;
  std::array<int, 8> last_step_stage_source_states_{};
  std::size_t last_step_stage_source_state_count_ = 0;
};

}  // namespace teuk
