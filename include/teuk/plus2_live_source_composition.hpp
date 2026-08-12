#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "teuk/plus2_companion_pipeline.hpp"
#include "teuk/plus2_linear_spatial.hpp"
#include "teuk/plus2_source_value_spatial.hpp"

namespace teuk {

// This is the narrow live-composition gate between the already qualified
// kernels.  It intentionally does not hide Z0/Z1 as mutable adapter state:
// callers must supply their exact common-RK-stage values and first/second
// tangents.  The missing production primitive/Bianchi spatial graph writes the
// remaining typed slots through Plus2LiveSourceWriteTarget.
enum class Plus2TransportedCurvatureComponent : std::size_t {
  Z0 = 0,
  Z1 = 1,
  Z0T = 2,
  Z1T = 3,
  Z0TT = 4,
  Z1TT = 5,
  Count = 6,
};

using Plus2LiveStampView =
    Kokkos::View<std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;
using Plus2LiveConstStampView =
    Kokkos::View<const std::uint64_t****, Kokkos::LayoutRight, MemorySpace>;
using Plus2LiveReadinessView =
    Kokkos::View<std::uint8_t**, Kokkos::LayoutRight, MemorySpace>;
using Plus2TransportedCurvatureStorageView = Plus2SpatialRank4View;
using Plus2TransportedCurvatureView =
    Kokkos::View<const Complex****, Kokkos::LayoutRight, MemorySpace>;

struct Plus2TransportedCurvatureStage {
  Plus2TransportedCurvatureView fields;
  Plus2LiveConstStampView stamps;
};

// These are scientific capabilities, not optional performance hints.  The
// adapter rejects a stage before launching if any claim is absent.  In
// particular, a grid containing R=0 requires an independently qualified
// peeling coefficient; the metric-curvature point graph correctly reports its
// raw 0/0 quotient as invalid there.
struct Plus2LiveSourceCapability {
  bool transported_curvature_in_common_rk_state = false;
  bool independently_qualified_scri_coefficients = false;
  bool primitive_bianchi_spatial_graph_qualified = false;
  bool angular_projection_graph_qualified = false;
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
  std::uint64_t generation = 0;
};

struct Plus2LiveSourceWriteTarget {
  std::uint64_t generation;
  Plus2SpatialPrimitiveView primitive_value;
  Plus2SpatialPrimitiveView primitive_tangent;
  Plus2ProductionJkDerivativeView jk_derivative_value;
  Plus2ProductionJkDerivativeView jk_derivative_tangent;
  Plus2ProductionQDerivativeView q_derivative_value;
  Plus2LiveStampView primitive_value_stamps;
  Plus2LiveStampView primitive_tangent_stamps;
  Plus2LiveStampView jk_derivative_value_stamps;
  Plus2LiveStampView jk_derivative_tangent_stamps;
  Plus2LiveStampView q_derivative_value_stamps;
};

struct Plus2LiveOuterWriteTarget {
  std::uint64_t generation;
  Plus2SpatialAggregateView projected_sum_value;
  Plus2SpatialOuterDerivativeView outer_derivative_value;
  Plus2LiveStampView projected_sum_value_stamps;
  Plus2LiveStampView outer_derivative_value_stamps;
};

namespace plus2_live_source_detail {

KOKKOS_INLINE_FUNCTION std::size_t flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

struct NormalizeSourceInputsFunctor {
  Complex* primitive_value;
  Complex* primitive_tangent;
  Complex* jk_value;
  Complex* jk_tangent;
  Complex* q_value;
  const Complex* curvature;
  const std::uint64_t* primitive_value_stamps;
  const std::uint64_t* primitive_tangent_stamps;
  const std::uint64_t* jk_value_stamps;
  const std::uint64_t* jk_tangent_stamps;
  const std::uint64_t* q_value_stamps;
  const std::uint64_t* curvature_stamps;
  std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t mode_count;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t primitive_count =
        static_cast<std::size_t>(Plus2SpatialPrimitive::Count);
    constexpr std::size_t jk_count =
        static_cast<std::size_t>(Plus2ProductionJkDerivative::Count);
    constexpr std::size_t q_count =
        static_cast<std::size_t>(Plus2ProductionQDerivative::Count);
    constexpr std::size_t curvature_count =
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = generation != 0;
    for (std::size_t field = 0; field < curvature_count; ++field) {
      valid = valid &&
              curvature_stamps[flat4(mode, field, radial, theta,
                                     curvature_count, radial_count,
                                     theta_count)] == generation;
    }
    // Z0/Z1 values and their analytic tangents are owned by the transported
    // curvature stage.  A primitive producer cannot silently replace them.
    const std::size_t z0 = static_cast<std::size_t>(
        Plus2SpatialPrimitive::Z0);
    const std::size_t z1 = static_cast<std::size_t>(
        Plus2SpatialPrimitive::Z1);
    const auto curvature_at = [&](const Plus2TransportedCurvatureComponent f) {
      return curvature[flat4(mode, static_cast<std::size_t>(f), radial,
                             theta, curvature_count, radial_count,
                             theta_count)];
    };
    primitive_value[flat4(mode, z0, radial, theta, primitive_count,
                          radial_count, theta_count)] =
        curvature_at(Plus2TransportedCurvatureComponent::Z0);
    primitive_value[flat4(mode, z1, radial, theta, primitive_count,
                          radial_count, theta_count)] =
        curvature_at(Plus2TransportedCurvatureComponent::Z1);
    primitive_tangent[flat4(mode, z0, radial, theta, primitive_count,
                            radial_count, theta_count)] =
        curvature_at(Plus2TransportedCurvatureComponent::Z0T);
    primitive_tangent[flat4(mode, z1, radial, theta, primitive_count,
                            radial_count, theta_count)] =
        curvature_at(Plus2TransportedCurvatureComponent::Z1T);

    for (std::size_t field = 0; field < primitive_count; ++field) {
      if (field == z0 || field == z1) continue;
      const std::size_t index = flat4(mode, field, radial, theta,
                                      primitive_count, radial_count,
                                      theta_count);
      valid = valid && primitive_value_stamps[index] == generation &&
              primitive_tangent_stamps[index] == generation;
    }
    for (std::size_t field = 0; field < jk_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta, jk_count,
                                      radial_count, theta_count);
      valid = valid && jk_value_stamps[index] == generation &&
              jk_tangent_stamps[index] == generation;
    }
    for (std::size_t field = 0; field < q_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta, q_count,
                                      radial_count, theta_count);
      valid = valid && q_value_stamps[index] == generation;
    }
    if (!valid) {
      for (std::size_t field = 0; field < primitive_count; ++field) {
        const std::size_t index = flat4(mode, field, radial, theta,
                                        primitive_count, radial_count,
                                        theta_count);
        primitive_value[index] = Complex{};
        primitive_tangent[index] = Complex{};
      }
      for (std::size_t field = 0; field < jk_count; ++field) {
        const std::size_t index = flat4(mode, field, radial, theta, jk_count,
                                        radial_count, theta_count);
        jk_value[index] = Complex{};
        jk_tangent[index] = Complex{};
      }
      for (std::size_t field = 0; field < q_count; ++field) {
        q_value[flat4(mode, field, radial, theta, q_count, radial_count,
                      theta_count)] = Complex{};
      }
    }
    // One readiness bit per (R,theta), strict over the complete signed-mode
    // registry.  Atomic AND avoids accepting a target fed by a stale parent.
    if (!valid) Kokkos::atomic_exchange(&ready[radial * theta_count + theta],
                                         static_cast<std::uint8_t>(0));
  }
};

struct NormalizeOuterInputsFunctor {
  Complex* projected;
  Complex* outer;
  const std::uint64_t* projected_stamps;
  const std::uint64_t* outer_stamps;
  std::uint8_t* ready;
  std::uint64_t generation;
  std::size_t mode_count;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t outer_count =
        static_cast<std::size_t>(Plus2SpatialOuterDerivative::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    bool valid = true;
    for (std::size_t field = 0; field < aggregate_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      aggregate_count, radial_count,
                                      theta_count);
      valid = valid && projected_stamps[index] == generation;
      if (!valid) projected[index] = Complex{};
    }
    for (std::size_t field = 0; field < outer_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta, outer_count,
                                      radial_count, theta_count);
      valid = valid && outer_stamps[index] == generation;
      if (!valid) outer[index] = Complex{};
    }
    if (!valid) Kokkos::atomic_exchange(&ready[radial * theta_count + theta],
                                         static_cast<std::uint8_t>(0));
  }
};

struct GatherForcingFunctor {
  const Complex* source_forcing;
  const std::size_t* target_indices;
  const std::uint8_t* ready;
  Complex* target_forcing;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    const std::size_t plane = radial_count * theta_count;
    const std::size_t target = flat / plane;
    const std::size_t within = flat - target * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t mode = target_indices[target];
    target_forcing[flat] = ready[radial * theta_count + theta]
                               ? source_forcing[(mode * radial_count + radial) *
                                                    theta_count +
                                                theta]
                               : Complex{};
  }
};

struct ZeroForcingFunctor {
  Complex* forcing;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    forcing[i] = Complex{};
  }
};

struct SetReadyFunctor {
  std::uint8_t* ready;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    ready[i] = static_cast<std::uint8_t>(1);
  }
};

static_assert(std::is_trivially_copyable_v<NormalizeSourceInputsFunctor>);
static_assert(std::is_trivially_copyable_v<NormalizeOuterInputsFunctor>);
static_assert(std::is_trivially_copyable_v<GatherForcingFunctor>);
static_assert(std::is_trivially_copyable_v<ZeroForcingFunctor>);
static_assert(std::is_trivially_copyable_v<SetReadyFunctor>);
static_assert(sizeof(NormalizeSourceInputsFunctor) < 1800);
static_assert(sizeof(NormalizeOuterInputsFunctor) < 1800);

}  // namespace plus2_live_source_detail

template <class ExecSpace = ExecutionSpace>
class Plus2LiveSourceComposition {
 public:
  using execution_space = ExecSpace;

  Plus2LiveSourceComposition(
      const execution_space& execution, const ModeRegistry& registry,
      const UniformRadialGrid& radial_grid, const KerrParameters& parameters,
      const int ell_max, const int theta_count,
      const Plus2SpatialThetaView& cos_theta,
      const Plus2SpatialThetaView& sin_theta,
      const std::string& label = "plus2_live_source",
      const RadialDiscretization radial_discretization =
          RadialDiscretization::D105)
      : registry_(registry),
        radial_grid_(radial_grid),
        parameters_(parameters),
        cos_theta_(cos_theta),
        sin_theta_(sin_theta),
        radial_discretization_(radial_discretization),
        linear_(execution, registry, radial_grid.size(), ell_max, theta_count,
                label + "_linear", radial_discretization),
        source_(registry, radial_grid.size(), theta_count, label + "_source"),
        primitive_value_(label + "_primitive_value", registry.size(),
                         static_cast<std::size_t>(
                             Plus2SpatialPrimitive::Count),
                         radial_grid.size(), theta_count),
        primitive_tangent_(label + "_primitive_tangent", registry.size(),
                           static_cast<std::size_t>(
                               Plus2SpatialPrimitive::Count),
                           radial_grid.size(), theta_count),
        jk_value_(label + "_jk_value", registry.size(),
                  static_cast<std::size_t>(
                      Plus2ProductionJkDerivative::Count),
                  radial_grid.size(), theta_count),
        jk_tangent_(label + "_jk_tangent", registry.size(),
                    static_cast<std::size_t>(
                        Plus2ProductionJkDerivative::Count),
                    radial_grid.size(), theta_count),
        q_value_(label + "_q_value", registry.size(),
                 static_cast<std::size_t>(
                     Plus2ProductionQDerivative::Count),
                 radial_grid.size(), theta_count),
        projected_(label + "_projected", registry.size(),
                   static_cast<std::size_t>(Plus2SpatialAggregate::Count),
                   radial_grid.size(), theta_count),
        outer_(label + "_outer", registry.size(),
               static_cast<std::size_t>(
                   Plus2SpatialOuterDerivative::Count),
               radial_grid.size(), theta_count),
        primitive_value_stamps_(Kokkos::view_alloc(
                                    Kokkos::WithoutInitializing,
                                    label + "_primitive_value_stamps"),
                                primitive_value_.layout()),
        primitive_tangent_stamps_(Kokkos::view_alloc(
                                      Kokkos::WithoutInitializing,
                                      label + "_primitive_tangent_stamps"),
                                  primitive_tangent_.layout()),
        jk_value_stamps_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             label + "_jk_value_stamps"),
                         jk_value_.layout()),
        jk_tangent_stamps_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               label + "_jk_tangent_stamps"),
                           jk_tangent_.layout()),
        q_value_stamps_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                            label + "_q_value_stamps"),
                        q_value_.layout()),
        projected_stamps_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                              label + "_projected_stamps"),
                          projected_.layout()),
        outer_stamps_(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                          label + "_outer_stamps"),
                      outer_.layout()),
        readiness_(label + "_readiness", radial_grid.size(), theta_count) {
    if (radial_grid.size() < radial_minimum_points(radial_discretization_) ||
        cos_theta.extent(0) != static_cast<std::size_t>(theta_count) ||
        sin_theta.extent(0) != static_cast<std::size_t>(theta_count)) {
      throw std::invalid_argument("spin +2 live source geometry is invalid");
    }
  }

  [[nodiscard]] RadialDiscretization radial_discretization() const noexcept {
    return radial_discretization_;
  }
  [[nodiscard]] Plus2LiveReadinessView readiness() const { return readiness_; }
  [[nodiscard]] const Plus2LinearPsi0SpatialWorkspace<execution_space>&
  linear_workspace() const { return linear_; }
  [[nodiscard]] const Plus2SourceValueSpatialWorkspace& source_workspace()
      const { return source_; }

  template <class ReconstructionView, class SourceProducer,
            class OuterProducer>
  void evaluate_stage(
      const execution_space& execution, const double stage_time,
      const ReconstructionView& reconstruction,
      const ReconstructionView& tangent,
      const ReconstructionView& second_tangent,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2LiveSourceCapability& capability,
      const SourceActivationState& activation_snapshot,
      const Plus2StageSourceTarget& target,
      SourceProducer&& source_producer, OuterProducer&& outer_producer,
      const Plus2ReconstructionMetricOffsets offsets = {}) {
    validate_stage(capability, activation_snapshot, target, curvature);
    last_generation_ = capability.generation;
    if (!activation_snapshot.active) {
      Kokkos::parallel_for(
          "zero_inactive_plus2_live_forcing",
          Kokkos::RangePolicy<execution_space>(execution, 0,
                                               target.coordinate_forcing.size()),
          plus2_live_source_detail::ZeroForcingFunctor{
              target.coordinate_forcing.data()});
      return;
    }

    linear_.pack_reconstruction_metric(
        execution, radial_grid_, parameters_, sin_theta_, cos_theta_,
        reconstruction, tangent, second_tangent, offsets);
    linear_.evaluate_packed_metric(execution, radial_grid_, parameters_,
                                   sin_theta_, cos_theta_);

    const Plus2LiveSourceWriteTarget source_target{
        capability.generation,
        primitive_value_,
        primitive_tangent_,
        jk_value_,
        jk_tangent_,
        q_value_,
        primitive_value_stamps_,
        primitive_tangent_stamps_,
        jk_value_stamps_,
        jk_tangent_stamps_,
        q_value_stamps_};
    using const_reconstruction_view = typename ReconstructionView::const_type;
    const const_reconstruction_view read_only_reconstruction = reconstruction;
    const const_reconstruction_view read_only_tangent = tangent;
    const const_reconstruction_view read_only_second_tangent = second_tangent;
    source_producer(execution, stage_time, read_only_reconstruction,
                    read_only_tangent, read_only_second_tangent,
                    linear_.z_plus(), linear_.z_plus_valid(), curvature,
                    source_target);

    Kokkos::parallel_for(
        "initialize_plus2_live_readiness",
        Kokkos::RangePolicy<execution_space>(execution, 0, readiness_.size()),
        plus2_live_source_detail::SetReadyFunctor{readiness_.data()});
    const std::size_t total =
        registry_.size() * radial_grid_.size() * sin_theta_.extent(0);
    Kokkos::parallel_for(
        "normalize_plus2_live_source_inputs",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        plus2_live_source_detail::NormalizeSourceInputsFunctor{
            primitive_value_.data(), primitive_tangent_.data(),
            jk_value_.data(), jk_tangent_.data(), q_value_.data(),
            curvature.fields.data(), primitive_value_stamps_.data(),
            primitive_tangent_stamps_.data(), jk_value_stamps_.data(),
            jk_tangent_stamps_.data(), q_value_stamps_.data(),
            curvature.stamps.data(), readiness_.data(), capability.generation,
            registry_.size(), radial_grid_.size(), sin_theta_.extent(0)});

    evaluate_plus2_production_ordered_pair_values(
        execution, radial_grid_, parameters_, cos_theta_, sin_theta_,
        primitive_value_, primitive_tangent_, jk_value_, jk_tangent_, q_value_,
        source_);

    const Plus2LiveOuterWriteTarget outer_target{
        capability.generation, projected_, outer_, projected_stamps_,
        outer_stamps_};
    outer_producer(execution, stage_time, source_.summed_value(),
                   source_.summed_jk_tangent(), outer_target);
    Kokkos::parallel_for(
        "normalize_plus2_live_outer_inputs",
        Kokkos::RangePolicy<execution_space>(execution, 0, total),
        plus2_live_source_detail::NormalizeOuterInputsFunctor{
            projected_.data(), outer_.data(), projected_stamps_.data(),
            outer_stamps_.data(), readiness_.data(), capability.generation,
            registry_.size(), radial_grid_.size(), sin_theta_.extent(0)});
    evaluate_plus2_production_outer_source_value(
        execution, radial_grid_, parameters_, cos_theta_, sin_theta_,
        projected_, outer_, 1.0, source_);
    Kokkos::parallel_for(
        "gather_plus2_live_target_forcing",
        Kokkos::RangePolicy<execution_space>(
            execution, 0, target.coordinate_forcing.size()),
        plus2_live_source_detail::GatherForcingFunctor{
            source_.forcing_value().data(), source_.target_indices().data(),
            readiness_.data(), target.coordinate_forcing.data(),
            radial_grid_.size(), sin_theta_.extent(0)});
  }

 private:
  void validate_stage(const Plus2LiveSourceCapability& capability,
                      const SourceActivationState& activation,
                      const Plus2StageSourceTarget& target,
                      const Plus2TransportedCurvatureStage& curvature) const {
    constexpr std::size_t curvature_count =
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Count);
    const bool has_scri = radial_grid_.lower_radius() == 0.0;
    const bool supports_nested_fourth_order =
        radial_discretization_ == RadialDiscretization::D105;
    if (capability.radial_discretization != radial_discretization_ ||
        !supports_nested_fourth_order ||
        capability.generation == 0 ||
        capability.generation <= last_generation_ ||
        !capability.transported_curvature_in_common_rk_state ||
        !capability.primitive_bianchi_spatial_graph_qualified ||
        !capability.angular_projection_graph_qualified ||
        (has_scri &&
         !capability.independently_qualified_scri_coefficients)) {
      throw std::invalid_argument(
          "spin +2 live source capability/readiness contract is incomplete");
    }
    if (activation.active != target.accepted_activation.active ||
        activation.activation_time !=
            target.accepted_activation.activation_time ||
        activation.consecutive_passes !=
            target.accepted_activation.consecutive_passes ||
        activation.last_eligibility_time !=
            target.accepted_activation.last_eligibility_time) {
      throw std::invalid_argument(
          "spin +2 live source activation snapshot mismatch");
    }
    const auto valid_curvature_shape = [&](const auto& view) {
      return view.extent(0) == registry_.size() &&
             view.extent(1) == curvature_count &&
             view.extent(2) == radial_grid_.size() &&
             view.extent(3) == sin_theta_.extent(0);
    };
    if (!valid_curvature_shape(curvature.fields) ||
        !valid_curvature_shape(curvature.stamps) ||
        target.coordinate_forcing.extent(0) != registry_.targets().size() ||
        target.coordinate_forcing.extent(1) != radial_grid_.size() ||
        target.coordinate_forcing.extent(2) != sin_theta_.extent(0)) {
      throw std::invalid_argument("spin +2 live source stage extent mismatch");
    }
  }

  ModeRegistry registry_;
  UniformRadialGrid radial_grid_;
  KerrParameters parameters_;
  Plus2SpatialThetaView cos_theta_;
  Plus2SpatialThetaView sin_theta_;
  RadialDiscretization radial_discretization_;
  Plus2LinearPsi0SpatialWorkspace<execution_space> linear_;
  Plus2SourceValueSpatialWorkspace source_;
  Plus2SpatialPrimitiveView primitive_value_;
  Plus2SpatialPrimitiveView primitive_tangent_;
  Plus2ProductionJkDerivativeView jk_value_;
  Plus2ProductionJkDerivativeView jk_tangent_;
  Plus2ProductionQDerivativeView q_value_;
  Plus2SpatialAggregateView projected_;
  Plus2SpatialOuterDerivativeView outer_;
  Plus2LiveStampView primitive_value_stamps_;
  Plus2LiveStampView primitive_tangent_stamps_;
  Plus2LiveStampView jk_value_stamps_;
  Plus2LiveStampView jk_tangent_stamps_;
  Plus2LiveStampView q_value_stamps_;
  Plus2LiveStampView projected_stamps_;
  Plus2LiveStampView outer_stamps_;
  Plus2LiveReadinessView readiness_;
  std::uint64_t last_generation_ = 0;
};

}  // namespace teuk
