#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "teuk/background.hpp"
#include "teuk/grid.hpp"
#include "teuk/jet.hpp"
#include "teuk/modes.hpp"
#include "teuk/second_order.hpp"
#include "teuk/source_spatial.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Adds analytic common-stage tangents to the deterministic value workspace.
// The nested value workspace owns pair ordering, target offsets, and explicit
// m -> -m lookup; all storage is allocated at construction, never in kernels.
class SpatialInnerSourceTangentWorkspace {
 public:
  SpatialInnerSourceTangentWorkspace(
      const ModeRegistry& registry, const std::size_t radial_points,
      const std::size_t theta_points,
      const std::string& label = "spatial_source_tangent")
      : values_(registry, radial_points, theta_points, label + "_values"),
        summed_tangent_(label + "_summed_tangent", registry.size(),
                        static_cast<std::size_t>(
                            SpatialInnerSourceComponent::Count),
                        radial_points, theta_points),
        per_pair_tangent_(
            label + "_per_pair_tangent", registry.ordered_pairs().size(),
            static_cast<std::size_t>(SpatialInnerSourceComponent::Count),
            radial_points, theta_points) {}

  [[nodiscard]] SpatialInnerSourceWorkspace& values() { return values_; }
  [[nodiscard]] const SpatialInnerSourceWorkspace& values() const {
    return values_;
  }
  [[nodiscard]] SpatialInnerSourceView summed_value() const {
    return values_.summed();
  }
  [[nodiscard]] SpatialPairSourceView per_pair_value() const {
    return values_.per_pair();
  }
  [[nodiscard]] SpatialInnerSourceView summed_tangent() const {
    return summed_tangent_;
  }
  [[nodiscard]] SpatialPairSourceView per_pair_tangent() const {
    return per_pair_tangent_;
  }

 private:
  SpatialInnerSourceWorkspace values_;
  SpatialInnerSourceView summed_tangent_;
  SpatialPairSourceView per_pair_tangent_;
};

// Evaluate B(U,V) and its analytic common-stage tangent
// B(U_dot,V)+B(U,V_dot) for every deterministic ordered pair. Sharp fields use
// conj(X_dot[-m]) as required; the sharp derivative components are already
// explicit caller-supplied derivative fields and carry their own tangents.
template <class ExecutionSpace>
void evaluate_spatial_inner_source_tangent(
    const ExecutionSpace& execution, const UniformRadialGrid& radial_grid,
    const KerrParameters& parameters, const SpatialThetaView& cos_theta,
    const SpatialThetaView& sin_theta, const SpatialSourceFieldView& fields,
    const SpatialSourceFieldView& field_tangents,
    const SpatialSourceDerivativeView& derivatives,
    const SpatialSourceDerivativeView& derivative_tangents,
    SpatialInnerSourceTangentWorkspace& workspace) {
  using J = Jet1<Complex>;
  SpatialInnerSourceWorkspace& value_workspace = workspace.values();
  const std::size_t modes = value_workspace.mode_count();
  const std::size_t radial_points = value_workspace.radial_points();
  const std::size_t theta_points = value_workspace.theta_points();
  const auto valid_fields = [&](const SpatialSourceFieldView& view) {
    return view.extent(0) == modes &&
           view.extent(1) ==
               static_cast<std::size_t>(SpatialSourceField::Count) &&
           view.extent(2) == radial_points &&
           view.extent(3) == theta_points;
  };
  const auto valid_derivatives =
      [&](const SpatialSourceDerivativeView& view) {
        return view.extent(0) == modes &&
               view.extent(1) == static_cast<std::size_t>(
                                      SpatialSourceDerivative::Count) &&
               view.extent(2) == radial_points &&
               view.extent(3) == theta_points;
      };
  if (radial_grid.size() != radial_points ||
      cos_theta.extent(0) != theta_points ||
      sin_theta.extent(0) != theta_points || !valid_fields(fields) ||
      !valid_fields(field_tangents) || !valid_derivatives(derivatives) ||
      !valid_derivatives(derivative_tangents)) {
    throw std::invalid_argument(
        "spatial source tangent view extents do not match");
  }

  const auto pairs = value_workspace.pairs();
  const auto sharp = value_workspace.sharp_indices();
  const auto targets = value_workspace.target_indices();
  const auto offsets = value_workspace.pair_offsets();
  const auto summed_value = workspace.summed_value();
  const auto summed_tangent = workspace.summed_tangent();
  const auto per_pair_value = workspace.per_pair_value();
  const auto per_pair_tangent = workspace.per_pair_tangent();
  Kokkos::deep_copy(execution, summed_value, Complex(0.0, 0.0));
  Kokkos::deep_copy(execution, summed_tangent, Complex(0.0, 0.0));

  const std::size_t target_points =
      value_workspace.target_count() * radial_points * theta_points;
  Kokkos::parallel_for(
      "teuk_spatial_inner_source_tangent",
      Kokkos::RangePolicy<ExecutionSpace>(execution, 0, target_points),
      KOKKOS_LAMBDA(const std::size_t flat) {
        const std::size_t target_plane = radial_points * theta_points;
        const std::size_t target_slot = flat / target_plane;
        const std::size_t within_target = flat - target_slot * target_plane;
        const std::size_t radial = within_target / theta_points;
        const std::size_t theta = within_target - radial * theta_points;
        const std::size_t target = targets(target_slot);
        J total_D;
        J total_T;
        const double radius = radial_grid.coordinate(radial);
        const KerrBackgroundPoint background = kerr_background_point(
            parameters, radius, cos_theta(theta), sin_theta(theta));

        for (std::size_t pair_index = offsets(target_slot);
             pair_index < offsets(target_slot + 1); ++pair_index) {
          const SpatialSourcePair pair = pairs(pair_index);
          const std::size_t m1 = pair.index1;
          const std::size_t m2 = pair.index2;
          const std::size_t sharp1 = sharp(m1);
          const std::size_t sharp2 = sharp(m2);
          const auto field = [&](const std::size_t mode,
                                 const SpatialSourceField component) {
            const std::size_t c = static_cast<std::size_t>(component);
            return J(fields(mode, c, radial, theta),
                     field_tangents(mode, c, radial, theta));
          };
          const auto derivative =
              [&](const std::size_t mode,
                  const SpatialSourceDerivative component) {
                const std::size_t c = static_cast<std::size_t>(component);
                return J(derivatives(mode, c, radial, theta),
                         derivative_tangents(mode, c, radial, theta));
              };
          const OrderedPairFieldsT<J> point_fields{
              field(m1, SpatialSourceField::F),
              field(m1, SpatialSourceField::G),
              field(m1, SpatialSourceField::Lambda),
              field(m1, SpatialSourceField::Pi),
              field(m1, SpatialSourceField::B),
              field(m1, SpatialSourceField::C),
              field(m1, SpatialSourceField::U),
              field(m2, SpatialSourceField::F),
              field(m2, SpatialSourceField::G),
              field(m2, SpatialSourceField::H),
              field(m2, SpatialSourceField::B),
              field(m2, SpatialSourceField::C),
              field(m2, SpatialSourceField::U),
              jet_conj(field(sharp2, SpatialSourceField::U)),
              jet_conj(field(sharp2, SpatialSourceField::C)),
              jet_conj(field(sharp1, SpatialSourceField::C)),
              jet_conj(field(sharp1, SpatialSourceField::B)),
              jet_conj(field(sharp2, SpatialSourceField::Pi)),
              jet_conj(field(sharp2, SpatialSourceField::B))};
          const OrderedPairDerivativesT<J> point_derivatives{
              derivative(m2, SpatialSourceDerivative::Delta1F),
              derivative(m2, SpatialSourceDerivative::Delta3U),
              derivative(m2, SpatialSourceDerivative::Eth2C),
              derivative(m2, SpatialSourceDerivative::EthPrime2CSharp),
              derivative(m2, SpatialSourceDerivative::Eth1B),
              derivative(m2, SpatialSourceDerivative::Delta2C),
              derivative(m2, SpatialSourceDerivative::Delta2G),
              derivative(m2, SpatialSourceDerivative::Eth2G),
              derivative(m2, SpatialSourceDerivative::EthPrime1F),
              derivative(m2, SpatialSourceDerivative::Delta2CSharp),
              derivative(m2, SpatialSourceDerivative::EthPrime1BSharp)};
          const InnerSourceT<J> source = corrected_ordered_pair_source(
              radius, background, point_fields, point_derivatives);
          const std::size_t D =
              static_cast<std::size_t>(SpatialInnerSourceComponent::D);
          const std::size_t T =
              static_cast<std::size_t>(SpatialInnerSourceComponent::T);
          per_pair_value(pair_index, D, radial, theta) = source.D.value;
          per_pair_value(pair_index, T, radial, theta) = source.T.value;
          per_pair_tangent(pair_index, D, radial, theta) = source.D.dt;
          per_pair_tangent(pair_index, T, radial, theta) = source.T.dt;
          total_D += source.D;
          total_T += source.T;
        }
        const std::size_t D =
            static_cast<std::size_t>(SpatialInnerSourceComponent::D);
        const std::size_t T =
            static_cast<std::size_t>(SpatialInnerSourceComponent::T);
        summed_value(target, D, radial, theta) = total_D.value;
        summed_value(target, T, radial, theta) = total_T.value;
        summed_tangent(target, D, radial, theta) = total_D.dt;
        summed_tangent(target, T, radial, theta) = total_T.dt;
      });
}

}  // namespace teuk
