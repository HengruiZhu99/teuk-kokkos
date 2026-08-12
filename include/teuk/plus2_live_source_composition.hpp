#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "teuk/plus2_companion_pipeline.hpp"
#include "teuk/plus2_linear_spatial.hpp"
#include "teuk/plus2_routeb_curvature_spatial.hpp"
#include "teuk/plus2_source_primitive_spatial.hpp"
#include "teuk/plus2_source_outer_spatial.hpp"
#include "teuk/plus2_source_value_spatial.hpp"
#include "teuk/plus2_transported_curvature.hpp"

namespace teuk {

// This is the narrow live-composition gate between the already qualified
// kernels.  It intentionally does not hide Z0/Z1 as mutable adapter state:
// callers must supply their exact common-RK-stage values and first/second
// tangents. The scientific overload below binds the concrete primitive
// spatial producer; its curvature adapter may come from Route A only for
// validation because the rotating Bianchi transport is weakly hyperbolic.
// Rotating production requires the pending local Route-B curvature graph.
// The callback overload remains a low-level component-test seam.
using Plus2LiveReadinessView =
    Kokkos::View<std::uint8_t**, Kokkos::LayoutRight, MemorySpace>;

// These are scientific capabilities, not optional performance hints.  The
// adapter rejects a stage before launching if any claim is absent.  In
// particular, a grid containing R=0 requires an independently qualified
// peeling coefficient; the metric-curvature point graph correctly reports its
// raw 0/0 quotient as invalid there.
struct Plus2LiveSourceCapability {
  bool curvature_bound_to_common_rk_stage = false;
  bool independently_qualified_scri_coefficients = false;
  bool primitive_bianchi_spatial_graph_qualified = false;
  bool angular_projection_graph_qualified = false;
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
  std::uint64_t generation = 0;
  Plus2SourceNormalization source_normalization =
      plus2_source_normalization;
};

template <class ExecSpace>
struct Plus2RouteBStageSourceInputs {
  Plus2RouteBCurvatureTowerStage tower;
  Plus2RouteBCurvatureSpatialProvider<ExecSpace>* curvature = nullptr;
  Plus2SourcePrimitiveSpatialProducer<ExecSpace>* primitive = nullptr;
  Plus2SourceOuterSpatialProducer<ExecSpace>* outer = nullptr;
  Plus2RouteBCurvatureOffsets offsets{};
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
  const Complex* bianchi_derivatives;
  const std::uint64_t* primitive_value_stamps;
  const std::uint64_t* primitive_tangent_stamps;
  const std::uint64_t* jk_value_stamps;
  const std::uint64_t* jk_tangent_stamps;
  const std::uint64_t* q_value_stamps;
  const std::uint64_t* curvature_stamps;
  const std::uint64_t* bianchi_derivative_stamps;
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
    constexpr std::size_t bianchi_derivative_count =
        static_cast<std::size_t>(Plus2BianchiDerivativeComponent::Count);
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
    for (std::size_t field = 0; field < bianchi_derivative_count; ++field) {
      valid =
          valid &&
          bianchi_derivative_stamps[flat4(
              mode, field, radial, theta, bianchi_derivative_count,
              radial_count, theta_count)] == generation;
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

    // The Bianchi transport owns exactly these four J/K derivative pairs.
    // Always overwrite producer storage before source evaluation, and never
    // use producer stamps as authority for these slots.
    const auto bianchi_derivative_at =
        [&](const Plus2BianchiDerivativeComponent f) {
          return bianchi_derivatives[flat4(
              mode, static_cast<std::size_t>(f), radial, theta,
              bianchi_derivative_count, radial_count, theta_count)];
        };
    const auto copy_bianchi_pair =
        [&](const Plus2ProductionJkDerivative output,
            const Plus2BianchiDerivativeComponent value,
            const Plus2BianchiDerivativeComponent tangent) {
          const std::size_t index = flat4(
              mode, static_cast<std::size_t>(output), radial, theta, jk_count,
              radial_count, theta_count);
          jk_value[index] = bianchi_derivative_at(value);
          jk_tangent[index] = bianchi_derivative_at(tangent);
        };
    copy_bianchi_pair(
        Plus2ProductionJkDerivative::CapitalDelta4Z1,
        Plus2BianchiDerivativeComponent::CapitalDelta4Z1,
        Plus2BianchiDerivativeComponent::CapitalDelta4Z1T);
    copy_bianchi_pair(Plus2ProductionJkDerivative::EthPrime4Z1,
                      Plus2BianchiDerivativeComponent::EthPrime4Z1,
                      Plus2BianchiDerivativeComponent::EthPrime4Z1T);
    copy_bianchi_pair(
        Plus2ProductionJkDerivative::CapitalDelta5Z0,
        Plus2BianchiDerivativeComponent::CapitalDelta5Z0,
        Plus2BianchiDerivativeComponent::CapitalDelta5Z0T);
    copy_bianchi_pair(Plus2ProductionJkDerivative::Eth5Z0,
                      Plus2BianchiDerivativeComponent::Eth5Z0,
                      Plus2BianchiDerivativeComponent::Eth5Z0T);

    for (std::size_t field = 0; field < primitive_count; ++field) {
      if (field == z0 || field == z1) continue;
      const std::size_t index = flat4(mode, field, radial, theta,
                                      primitive_count, radial_count,
                                      theta_count);
      valid = valid && primitive_value_stamps[index] == generation &&
              primitive_tangent_stamps[index] == generation;
    }
    for (std::size_t field = 0; field < jk_count; ++field) {
      const bool transport_owned =
          field == static_cast<std::size_t>(
                       Plus2ProductionJkDerivative::CapitalDelta4Z1) ||
          field == static_cast<std::size_t>(
                       Plus2ProductionJkDerivative::EthPrime4Z1) ||
          field == static_cast<std::size_t>(
                       Plus2ProductionJkDerivative::CapitalDelta5Z0) ||
          field == static_cast<std::size_t>(
                       Plus2ProductionJkDerivative::Eth5Z0);
      if (transport_owned) continue;
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
    // Determine validity completely before mutating any diagnostic.  The
    // source-stage readiness reduction is authoritative: a stale primitive in
    // any signed parent invalidates every exposed target diagnostic at this
    // (R,theta) point even when the outer producer wrote fresh stamps.
    bool valid = ready[radial * theta_count + theta] != 0;
    for (std::size_t field = 0; field < aggregate_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta,
                                      aggregate_count, radial_count,
                                      theta_count);
      valid = valid && projected_stamps[index] == generation;
    }
    for (std::size_t field = 0; field < outer_count; ++field) {
      const std::size_t index = flat4(mode, field, radial, theta, outer_count,
                                      radial_count, theta_count);
      valid = valid && outer_stamps[index] == generation;
    }
    if (!valid) {
      for (std::size_t field = 0; field < aggregate_count; ++field) {
        projected[flat4(mode, field, radial, theta, aggregate_count,
                        radial_count, theta_count)] = Complex{};
      }
      for (std::size_t field = 0; field < outer_count; ++field) {
        outer[flat4(mode, field, radial, theta, outer_count, radial_count,
                    theta_count)] = Complex{};
      }
      Kokkos::atomic_exchange(&ready[radial * theta_count + theta],
                              static_cast<std::uint8_t>(0));
    }
  }
};

// Source workspaces are intentionally inspectable for diagnostics.  Once the
// registry-wide readiness reduction rejects a point, clear every exposed
// pair, target sum, raw S0/R6, regular S0/R7, and evolved-forcing value there.
// This runs after the pair and outer kernels so no invalid intermediate can be
// mistaken for a usable diagnostic merely because final forcing was gated.
struct ClearInvalidSourceDiagnosticsFunctor {
  Complex* pair_families;
  Complex* summed;
  Complex* summed_jk_tangent;
  Complex* source_over_r6;
  Complex* source_over_r7;
  Complex* forcing;
  const std::uint8_t* ready;
  std::size_t pair_count;
  std::size_t mode_count;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t point) const {
    if (ready[point] != 0) return;
    constexpr std::size_t family_count =
        static_cast<std::size_t>(Plus2SpatialPairFamily::Count);
    constexpr std::size_t aggregate_count =
        static_cast<std::size_t>(Plus2SpatialAggregate::Count);
    constexpr std::size_t tangent_count =
        static_cast<std::size_t>(Plus2ProductionJkAggregate::Count);
    const std::size_t radial = point / theta_count;
    const std::size_t theta = point - radial * theta_count;
    for (std::size_t pair = 0; pair < pair_count; ++pair) {
      for (std::size_t field = 0; field < family_count; ++field) {
        pair_families[flat4(pair, field, radial, theta, family_count,
                            radial_count, theta_count)] = Complex{};
      }
    }
    for (std::size_t mode = 0; mode < mode_count; ++mode) {
      for (std::size_t field = 0; field < aggregate_count; ++field) {
        summed[flat4(mode, field, radial, theta, aggregate_count,
                     radial_count, theta_count)] = Complex{};
      }
      for (std::size_t field = 0; field < tangent_count; ++field) {
        summed_jk_tangent[flat4(mode, field, radial, theta, tangent_count,
                                radial_count, theta_count)] = Complex{};
      }
      const std::size_t index =
          (mode * radial_count + radial) * theta_count + theta;
      source_over_r6[index] = Complex{};
      source_over_r7[index] = Complex{};
      forcing[index] = Complex{};
    }
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

struct PackRouteBTowerLevelsFunctor {
  const Complex* tower;
  const std::uint64_t* tower_stamps;
  Complex* value;
  Complex* tangent;
  Complex* second_tangent;
  std::uint64_t* value_stamps;
  std::uint64_t* tangent_stamps;
  std::uint64_t* second_tangent_stamps;
  std::size_t mode_count;
  std::size_t tower_field_count;
  std::size_t radial_count;
  std::size_t theta_count;
  std::size_t h_offset;
  std::size_t pi_offset;
  std::size_t b_offset;
  std::size_t c_offset;
  std::size_t u_offset;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t output_fields = 5;
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t offsets[output_fields]{h_offset, pi_offset, b_offset,
                                             c_offset, u_offset};
    Complex* outputs[3]{value, tangent, second_tangent};
    std::uint64_t* output_stamps[3]{value_stamps, tangent_stamps,
                                    second_tangent_stamps};
    for (std::size_t level = 0; level < 3; ++level) {
      const std::size_t stamp_index =
          ((level * mode_count + mode) * radial_count + radial) *
              theta_count +
          theta;
      for (std::size_t field = 0; field < output_fields; ++field) {
        const std::size_t input_index =
            ((((level * mode_count + mode) * tower_field_count +
               offsets[field]) *
                  radial_count +
              radial) *
                 theta_count +
             theta);
        const std::size_t output_index =
            ((mode * output_fields + field) * radial_count + radial) *
                theta_count +
            theta;
        outputs[level][output_index] = tower[input_index];
        output_stamps[level][output_index] = tower_stamps[stamp_index];
      }
    }
  }
};

static_assert(std::is_trivially_copyable_v<NormalizeSourceInputsFunctor>);
static_assert(std::is_trivially_copyable_v<NormalizeOuterInputsFunctor>);
static_assert(
    std::is_trivially_copyable_v<ClearInvalidSourceDiagnosticsFunctor>);
static_assert(std::is_trivially_copyable_v<GatherForcingFunctor>);
static_assert(std::is_trivially_copyable_v<ZeroForcingFunctor>);
static_assert(std::is_trivially_copyable_v<SetReadyFunctor>);
static_assert(std::is_trivially_copyable_v<PackRouteBTowerLevelsFunctor>);
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
        ell_max_(ell_max),
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
        routeb_value_(label + "_routeb_value", registry.size(), 5,
                      radial_grid.size(), theta_count),
        routeb_tangent_(label + "_routeb_tangent", registry.size(), 5,
                        radial_grid.size(), theta_count),
        routeb_second_tangent_(label + "_routeb_second_tangent",
                               registry.size(), 5, radial_grid.size(),
                               theta_count),
        routeb_value_stamps_(label + "_routeb_value_stamps", registry.size(),
                             5, radial_grid.size(), theta_count),
        routeb_tangent_stamps_(label + "_routeb_tangent_stamps",
                               registry.size(), 5, radial_grid.size(),
                               theta_count),
        routeb_second_tangent_stamps_(
            label + "_routeb_second_tangent_stamps", registry.size(), 5,
            radial_grid.size(), theta_count),
        readiness_(label + "_readiness", radial_grid.size(), theta_count) {
    if (radial_grid.size() < radial_minimum_points(radial_discretization_) ||
        cos_theta.extent(0) != static_cast<std::size_t>(theta_count) ||
        sin_theta.extent(0) != static_cast<std::size_t>(theta_count)) {
      throw std::invalid_argument("spin +2 live source geometry is invalid");
    }
    // WithoutInitializing is intentional for setup cost, but no uninitialized
    // stamp may accidentally equal the first accepted generation.
    Kokkos::deep_copy(execution, primitive_value_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, primitive_tangent_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, jk_value_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, jk_tangent_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, q_value_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, projected_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, outer_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, routeb_value_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, routeb_tangent_stamps_, std::uint64_t{0});
    Kokkos::deep_copy(execution, routeb_second_tangent_stamps_,
                      std::uint64_t{0});
    Kokkos::deep_copy(execution, readiness_, std::uint8_t{0});
  }

  [[nodiscard]] RadialDiscretization radial_discretization() const noexcept {
    return radial_discretization_;
  }
  [[nodiscard]] Plus2LiveReadinessView readiness() const { return readiness_; }
  [[nodiscard]] const Plus2LinearPsi0SpatialWorkspace<execution_space>&
  linear_workspace() const { return linear_; }
  [[nodiscard]] const Plus2SourceValueSpatialWorkspace& source_workspace()
      const { return source_; }

  [[nodiscard]] Plus2SourceProvenanceAuthority
  source_provenance_authority() const {
    return Plus2SourceProvenanceAuthority{this};
  }

  template <class Provider>
  [[nodiscard]] Plus2BoundStageSourceAdapter<execution_space, Provider>
  bind_routeb_stage_source(Provider& provider) {
    return Plus2BoundStageSourceAdapter<execution_space, Provider>{*this,
                                                                   provider};
  }

  template <class PrimaryStage, class Provider>
  void evaluate_bound_routeb_stage(
      const execution_space& execution, const double stage_time,
      const PrimaryStage& primary_stage,
      const Plus2StageSourceTarget& target, Provider& provider) {
    auto inputs = provider.routeb_stage_source_inputs(
        execution, stage_time, primary_stage, target);
    if (inputs.curvature == nullptr || inputs.primitive == nullptr ||
        inputs.outer == nullptr) {
      throw std::invalid_argument(
          "spin +2 bound Route-B source provider is incomplete");
    }
    evaluate_routeb_stage(execution, stage_time, inputs.tower,
                          target.accepted_activation, target,
                          *inputs.curvature, *inputs.primitive, *inputs.outer,
                          inputs.offsets);
  }

  // Complete local Route-B scientific path. The five-level tower is one
  // immutable common-stage object. Curvature is evaluated exactly once, then
  // levels h0/h1/h2 are packed into the concrete primitive producer without a
  // host copy or a second authority. This remains standalone and is not a
  // solver/runtime enablement seam.
  void evaluate_routeb_stage(
      const execution_space& execution, const double stage_time,
      const Plus2RouteBCurvatureTowerStage& tower,
      const SourceActivationState& activation_snapshot,
      const Plus2StageSourceTarget& target,
      Plus2RouteBCurvatureSpatialProvider<execution_space>&
          curvature_provider,
      Plus2SourcePrimitiveSpatialProducer<execution_space>& primitive_producer,
      Plus2SourceOuterSpatialProducer<execution_space>& outer_producer,
      const Plus2RouteBCurvatureOffsets tower_offsets = {}) {
    const Plus2LiveSourceCapability capability{
        true, true, true, true, RadialDiscretization::D105,
        tower.generation};
    if (!curvature_provider.accepts_generation(tower.generation) ||
        !curvature_provider.matches_configuration(
            registry_, radial_grid_, parameters_, ell_max_, cos_theta_,
            sin_theta_) ||
        primitive_producer.radial_discretization() !=
            radial_discretization_ ||
        !primitive_producer.accepts_generation(tower.generation) ||
        !primitive_producer.matches_configuration(
            registry_, radial_grid_, parameters_, ell_max_, cos_theta_,
            sin_theta_) ||
        outer_producer.radial_discretization() != radial_discretization_ ||
        !outer_producer.matches_configuration(
            registry_, radial_grid_, parameters_, ell_max_, cos_theta_,
            sin_theta_)) {
      throw std::invalid_argument(
          "spin +2 Route-B source stage configuration mismatch");
    }
    validate_stage(capability, activation_snapshot, target,
                   curvature_provider.curvature_stage(),
                   curvature_provider.derivative_stage());
    curvature_provider.evaluate(execution, tower, tower_offsets);
    const std::size_t points =
        registry_.size() * radial_grid_.size() * sin_theta_.extent(0);
    Kokkos::parallel_for(
        "pack_plus2_routeb_live_levels",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        plus2_live_source_detail::PackRouteBTowerLevelsFunctor{
            tower.fields.data(), tower.stamps.data(), routeb_value_.data(),
            routeb_tangent_.data(), routeb_second_tangent_.data(),
            routeb_value_stamps_.data(), routeb_tangent_stamps_.data(),
            routeb_second_tangent_stamps_.data(), registry_.size(),
            tower.fields.extent(2), radial_grid_.size(), sin_theta_.extent(0),
            tower_offsets.H, tower_offsets.Pi, tower_offsets.B,
            tower_offsets.C, tower_offsets.U});
    evaluate_stage(
        execution, stage_time,
        Plus2PrimitiveReconstructionStage{
            tower.generation, routeb_value_, routeb_tangent_,
            routeb_second_tangent_, routeb_value_stamps_,
            routeb_tangent_stamps_, routeb_second_tangent_stamps_},
        curvature_provider.curvature_stage(),
        curvature_provider.derivative_stage(), capability,
        activation_snapshot, target, primitive_producer, outer_producer);
  }

  // Scientific path: bind the complete stamped reconstruction stage directly
  // to the concrete primitive producer.  No generic source callback is
  // accepted by this overload.
  void evaluate_stage(
      const execution_space& execution, const double /*stage_time*/,
      const Plus2PrimitiveReconstructionStage& reconstruction,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2BianchiDerivativeStage& bianchi_derivatives,
      const Plus2LiveSourceCapability& capability,
      const SourceActivationState& activation_snapshot,
      const Plus2StageSourceTarget& target,
      Plus2SourcePrimitiveSpatialProducer<execution_space>& primitive_producer,
      Plus2SourceOuterSpatialProducer<execution_space>& outer_producer,
      const Plus2PrimitiveReconstructionOffsets offsets = {}) {
    if (reconstruction.generation != capability.generation ||
        primitive_producer.radial_discretization() !=
            radial_discretization_ ||
        !primitive_producer.accepts_generation(capability.generation) ||
        !primitive_producer.matches_configuration(
            registry_, radial_grid_, parameters_, ell_max_, cos_theta_,
            sin_theta_) ||
        outer_producer.radial_discretization() != radial_discretization_ ||
        !outer_producer.matches_configuration(
            registry_, radial_grid_, parameters_, ell_max_, cos_theta_,
            sin_theta_)) {
      throw std::invalid_argument(
          "spin +2 concrete source stage generation/scheme/band mismatch");
    }
    validate_reconstruction_stage_shape(reconstruction, offsets);
    const Plus2ReconstructionMetricOffsets metric_offsets{
        offsets.B, offsets.C, offsets.U};
    if (!prepare_stage(execution, reconstruction.value,
                       reconstruction.tangent,
                       reconstruction.second_tangent, curvature,
                       bianchi_derivatives, capability, activation_snapshot,
                       target, metric_offsets)) {
      return;
    }
    primitive_producer.evaluate(execution, reconstruction, curvature,
                                bianchi_derivatives,
                                source_write_target(capability.generation),
                                offsets);
    finish_scientific_stage(execution, curvature, bianchi_derivatives,
                            capability, target, outer_producer);
  }

  // Low-level callback seam retained for isolated composition tests.
  template <class ReconstructionView, class SourceProducer,
            class OuterProducer>
  void evaluate_stage(
      const execution_space& execution, const double stage_time,
      const ReconstructionView& reconstruction,
      const ReconstructionView& tangent,
      const ReconstructionView& second_tangent,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2BianchiDerivativeStage& bianchi_derivatives,
      const Plus2LiveSourceCapability& capability,
      const SourceActivationState& activation_snapshot,
      const Plus2StageSourceTarget& target,
      SourceProducer&& source_producer, OuterProducer&& outer_producer,
      const Plus2ReconstructionMetricOffsets offsets = {}) {
    if (!prepare_stage(execution, reconstruction, tangent, second_tangent,
                       curvature, bianchi_derivatives, capability,
                       activation_snapshot, target, offsets)) {
      return;
    }
    const auto source_target = source_write_target(capability.generation);
    using const_reconstruction_view = typename ReconstructionView::const_type;
    const const_reconstruction_view read_only_reconstruction = reconstruction;
    const const_reconstruction_view read_only_tangent = tangent;
    const const_reconstruction_view read_only_second_tangent = second_tangent;
    source_producer(execution, stage_time, read_only_reconstruction,
                    read_only_tangent, read_only_second_tangent,
                    linear_.z_plus(), linear_.z_plus_valid(), curvature,
                    bianchi_derivatives, source_target);
    finish_stage(execution, stage_time, curvature, bianchi_derivatives,
                 capability, target,
                 std::forward<OuterProducer>(outer_producer));
  }

  [[nodiscard]] std::uint64_t last_generation() const {
    return last_generation_;
  }

 private:
  void validate_reconstruction_stage_shape(
      const Plus2PrimitiveReconstructionStage& reconstruction,
      const Plus2PrimitiveReconstructionOffsets offsets) const {
    const std::size_t largest =
        std::max({offsets.H, offsets.Pi, offsets.B, offsets.C, offsets.U});
    const auto valid_shape = [&](const auto& view) {
      return view.extent(0) == registry_.size() &&
             view.extent(1) > largest &&
             view.extent(2) == radial_grid_.size() &&
             view.extent(3) == sin_theta_.extent(0);
    };
    if (!valid_shape(reconstruction.value) ||
        !valid_shape(reconstruction.tangent) ||
        !valid_shape(reconstruction.second_tangent) ||
        !valid_shape(reconstruction.value_stamps) ||
        !valid_shape(reconstruction.tangent_stamps) ||
        !valid_shape(reconstruction.second_tangent_stamps)) {
      throw std::invalid_argument(
          "spin +2 concrete reconstruction stage extent mismatch");
    }
  }

  [[nodiscard]] Plus2LiveSourceWriteTarget source_write_target(
      const std::uint64_t generation) const {
    return {generation,
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
  }

  template <class ReconstructionView>
  bool prepare_stage(
      const execution_space& execution,
      const ReconstructionView& reconstruction,
      const ReconstructionView& tangent,
      const ReconstructionView& second_tangent,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2BianchiDerivativeStage& bianchi_derivatives,
      const Plus2LiveSourceCapability& capability,
      const SourceActivationState& activation_snapshot,
      const Plus2StageSourceTarget& target,
      const Plus2ReconstructionMetricOffsets offsets) {
    validate_stage(capability, activation_snapshot, target, curvature,
                   bianchi_derivatives);
    last_generation_ = capability.generation;
    if (!activation_snapshot.active) {
      Kokkos::parallel_for(
          "zero_inactive_plus2_live_forcing",
          Kokkos::RangePolicy<execution_space>(execution, 0,
                                               target.coordinate_forcing.size()),
          plus2_live_source_detail::ZeroForcingFunctor{
              target.coordinate_forcing.data()});
      return false;
    }
    linear_.pack_reconstruction_metric(
        execution, radial_grid_, parameters_, sin_theta_, cos_theta_,
        reconstruction, tangent, second_tangent, offsets);
    linear_.evaluate_packed_metric(execution, radial_grid_, parameters_,
                                   sin_theta_, cos_theta_);
    return true;
  }

  template <class OuterProducer>
  void finish_stage(
      const execution_space& execution, const double stage_time,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2BianchiDerivativeStage& bianchi_derivatives,
      const Plus2LiveSourceCapability& capability,
      const Plus2StageSourceTarget& target,
      OuterProducer&& outer_producer) {
    prepare_outer_stage(execution, curvature, bianchi_derivatives,
                        capability);
    const Plus2LiveOuterWriteTarget outer_target{
        capability.generation, projected_, outer_, projected_stamps_,
        outer_stamps_};
    outer_producer(execution, stage_time, source_.summed_value(),
                   source_.summed_jk_tangent(), outer_target);
    finish_outer_stage(execution, capability, target);
  }

  void finish_scientific_stage(
      const execution_space& execution,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2BianchiDerivativeStage& bianchi_derivatives,
      const Plus2LiveSourceCapability& capability,
      const Plus2StageSourceTarget& target,
      Plus2SourceOuterSpatialProducer<execution_space>& outer_producer) {
    prepare_outer_stage(execution, curvature, bianchi_derivatives,
                        capability);
    const Plus2LiveOuterWriteTarget outer_target{
        capability.generation, projected_, outer_, projected_stamps_,
        outer_stamps_};
    outer_producer.evaluate(
        execution,
        Plus2OuterSpatialStage{capability.generation, source_.summed_value(),
                               source_.summed_jk_tangent()},
        outer_target);
    finish_outer_stage(execution, capability, target);
  }

  void prepare_outer_stage(
      const execution_space& execution,
      const Plus2TransportedCurvatureStage& curvature,
      const Plus2BianchiDerivativeStage& bianchi_derivatives,
      const Plus2LiveSourceCapability& capability) {
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
            curvature.fields.data(), bianchi_derivatives.fields.data(),
            primitive_value_stamps_.data(),
            primitive_tangent_stamps_.data(), jk_value_stamps_.data(),
            jk_tangent_stamps_.data(), q_value_stamps_.data(),
            curvature.stamps.data(), bianchi_derivatives.stamps.data(),
            readiness_.data(), capability.generation, registry_.size(),
            radial_grid_.size(), sin_theta_.extent(0)});
    evaluate_plus2_production_ordered_pair_values(
        execution, radial_grid_, parameters_, cos_theta_, sin_theta_,
        primitive_value_, primitive_tangent_, jk_value_, jk_tangent_, q_value_,
        source_);
  }

  void finish_outer_stage(const execution_space& execution,
                          const Plus2LiveSourceCapability& capability,
                          const Plus2StageSourceTarget& target) {
    const std::size_t total =
        registry_.size() * radial_grid_.size() * sin_theta_.extent(0);
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
        "clear_invalid_plus2_live_source_diagnostics",
        Kokkos::RangePolicy<execution_space>(execution, 0, readiness_.size()),
        plus2_live_source_detail::ClearInvalidSourceDiagnosticsFunctor{
            source_.pair_family_value().data(), source_.summed_value().data(),
            source_.summed_jk_tangent().data(),
            source_.source_over_r6_value().data(),
            source_.source_over_r7_value().data(),
            source_.forcing_value().data(), readiness_.data(),
            source_.pair_count(), registry_.size(), radial_grid_.size(),
            sin_theta_.extent(0)});
    Kokkos::parallel_for(
        "gather_plus2_live_target_forcing",
        Kokkos::RangePolicy<execution_space>(
            execution, 0, target.coordinate_forcing.size()),
        plus2_live_source_detail::GatherForcingFunctor{
            source_.forcing_value().data(), source_.target_indices().data(),
            readiness_.data(), target.coordinate_forcing.data(),
            radial_grid_.size(), sin_theta_.extent(0)});
  }

  void validate_stage(const Plus2LiveSourceCapability& capability,
                      const SourceActivationState& activation,
                      const Plus2StageSourceTarget& target,
                      const Plus2TransportedCurvatureStage& curvature,
                      const Plus2BianchiDerivativeStage&
                          bianchi_derivatives) const {
    constexpr std::size_t curvature_count =
        static_cast<std::size_t>(Plus2TransportedCurvatureComponent::Count);
    constexpr std::size_t bianchi_derivative_count =
        static_cast<std::size_t>(Plus2BianchiDerivativeComponent::Count);
    const bool has_scri = radial_grid_.lower_radius() == 0.0;
    const bool supports_nested_fourth_order =
        radial_discretization_ == RadialDiscretization::D105;
    if (capability.radial_discretization != radial_discretization_ ||
        !supports_nested_fourth_order ||
        capability.source_normalization != plus2_source_normalization ||
        capability.generation == 0 ||
        capability.generation <= last_generation_ ||
        !capability.curvature_bound_to_common_rk_stage ||
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
    const auto valid_bianchi_derivative_shape = [&](const auto& view) {
      return view.extent(0) == registry_.size() &&
             view.extent(1) == bianchi_derivative_count &&
             view.extent(2) == radial_grid_.size() &&
             view.extent(3) == sin_theta_.extent(0);
    };
    if (!valid_curvature_shape(curvature.fields) ||
        !valid_curvature_shape(curvature.stamps) ||
        !valid_bianchi_derivative_shape(bianchi_derivatives.fields) ||
        !valid_bianchi_derivative_shape(bianchi_derivatives.stamps) ||
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
  int ell_max_;
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
  Plus2PrimitiveConstStageView::non_const_type routeb_value_;
  Plus2PrimitiveConstStageView::non_const_type routeb_tangent_;
  Plus2PrimitiveConstStageView::non_const_type routeb_second_tangent_;
  Plus2LiveStampView routeb_value_stamps_;
  Plus2LiveStampView routeb_tangent_stamps_;
  Plus2LiveStampView routeb_second_tangent_stamps_;
  Plus2LiveReadinessView readiness_;
  std::uint64_t last_generation_ = 0;
};

}  // namespace teuk
