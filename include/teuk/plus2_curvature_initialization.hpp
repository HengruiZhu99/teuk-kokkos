#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "teuk/background.hpp"
#include "teuk/plus2_endpoint_extraction.hpp"
#include "teuk/grid.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/types.hpp"

namespace teuk {

// The triangular Bianchi system has generalized radial characteristic speed
// v=R^2/(2 L^2+4 M R), repeated twice.  In rotating Kerr, eth_4 Z1 contains
// Z1_T, so the principal time matrix is triangular and can have a Jordan
// block; this helper reports direction, not a diagonalizability/stability
// claim.  R increases from scri toward the future horizon.
enum class Plus2BianchiRadialBoundaryKind {
  ScriCharacteristic,
  InteriorTowardHorizon,
  HorizonOutflow,
};

struct Plus2BianchiRadialCharacteristic {
  double coordinate_speed = 0.0;
  Complex jordan_superdiagonal{};
  Plus2BianchiRadialBoundaryKind kind =
      Plus2BianchiRadialBoundaryKind::InteriorTowardHorizon;
  bool strongly_hyperbolic_radial_block = false;
  bool valid = false;
};

KOKKOS_INLINE_FUNCTION Plus2BianchiRadialCharacteristic
plus2_bianchi_radial_characteristic(const double radius,
                                    const double horizon_radius,
                                    const double sin_theta,
                                    const double cos_theta,
                                    const KerrParameters& parameters) {
  if (!Kokkos::isfinite(radius) || !Kokkos::isfinite(horizon_radius) ||
      !Kokkos::isfinite(sin_theta) || !Kokkos::isfinite(cos_theta) ||
      !Kokkos::isfinite(parameters.mass) ||
      !Kokkos::isfinite(parameters.spin) ||
      !Kokkos::isfinite(parameters.compactification_length) || radius < 0.0 ||
      horizon_radius <= 0.0 || radius > horizon_radius ||
      parameters.mass <= 0.0 ||
      Kokkos::abs(parameters.spin) > parameters.mass ||
      parameters.compactification_length <= 0.0) {
    return {};
  }
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const double A = 2.0 + 4.0 * parameters.mass * radius / length2;
  const double BR = radius * radius / length2;
  const double speed = BR / A;
  const Complex imaginary_unit(0.0, 1.0);
  const Complex eth_time_coefficient =
      -imaginary_unit * parameters.spin * sin_theta /
      (Kokkos::sqrt(2.0) *
       Complex(length2, -parameters.spin * radius * cos_theta));
  const Complex jordan = eth_time_coefficient * BR / (A * A);
  const auto kind =
      radius == 0.0
          ? Plus2BianchiRadialBoundaryKind::ScriCharacteristic
          : (radius == horizon_radius
                 ? Plus2BianchiRadialBoundaryKind::HorizonOutflow
                 : Plus2BianchiRadialBoundaryKind::InteriorTowardHorizon);
  return {speed, jordan, kind, Kokkos::abs(jordan) == 0.0, true};
}

enum class Plus2CurvatureFormulaId {
  Unknown,
  OrgRicciPeelingNumeratorsV1,
};

enum class Plus2PeelingEndpointOperatorId {
  Unknown,
  D105NestedLhopitalV1,
  ConstrainedPositiveNodesV2,
};

enum class Plus2RadialBoundaryPolicyId {
  Unknown,
  ContinuumNoIncomingButWeaklyHyperbolicV1,
};

struct Plus2PeelingResolutionSample {
  std::size_t radial_count = 0;
  double lower_radius = 0.0;
  double upper_radius = 0.0;
  double spacing = 0.0;
  double peeling_residual_norm = 0.0;
  std::vector<int> signed_modes;
  std::vector<double> cos_theta_nodes;
  std::string profile_id;
  std::vector<Complex> z0_scri;
  std::vector<Complex> z1_scri;
};

// This certificate has no public constructor.  It can only be obtained from
// the three-resolution gate below, so a caller cannot turn initialization on
// by filling a bundle of booleans.
class Plus2PeelingConvergenceCertificate {
 public:
  [[nodiscard]] const std::string& evidence_id() const noexcept {
    return evidence_id_;
  }
  [[nodiscard]] double minimum_observed_order() const noexcept {
    return minimum_observed_order_;
  }
  [[nodiscard]] const Plus2PeelingResolutionSample& qualified_fine_sample()
      const noexcept {
    return qualified_fine_sample_;
  }
  [[nodiscard]] double maximum_coefficient_change() const noexcept {
    return maximum_coefficient_change_;
  }

 private:
  Plus2PeelingConvergenceCertificate(std::string evidence_id,
                                     const double minimum_observed_order,
                                     Plus2PeelingResolutionSample fine_sample,
                                     const double maximum_coefficient_change)
      : evidence_id_(std::move(evidence_id)),
        minimum_observed_order_(minimum_observed_order),
        qualified_fine_sample_(std::move(fine_sample)),
        maximum_coefficient_change_(maximum_coefficient_change) {}

  std::string evidence_id_;
  double minimum_observed_order_ = 0.0;
  Plus2PeelingResolutionSample qualified_fine_sample_;
  double maximum_coefficient_change_ = 0.0;

  friend Plus2PeelingConvergenceCertificate
  qualify_plus2_peeling_convergence(
      const Plus2PeelingResolutionSample&,
      const Plus2PeelingResolutionSample&,
      const Plus2PeelingResolutionSample&, const std::string&, double);
};

namespace plus2_curvature_initialization_detail {

inline double observed_order(const double coarse, const double medium,
                             const double coarse_spacing,
                             const double medium_spacing) {
  if (coarse == 0.0 && medium == 0.0) {
    return 100.0;
  }
  if (!(coarse > medium && medium > 0.0 && coarse_spacing > medium_spacing)) {
    return -1.0;
  }
  return std::log(coarse / medium) /
         std::log(coarse_spacing / medium_spacing);
}

KOKKOS_INLINE_FUNCTION constexpr std::size_t flat3(
    const std::size_t mode, const std::size_t radial,
    const std::size_t theta, const std::size_t radial_count,
    const std::size_t theta_count) {
  return (mode * radial_count + radial) * theta_count + theta;
}

KOKKOS_INLINE_FUNCTION constexpr std::size_t flat4(
    const std::size_t mode, const std::size_t field,
    const std::size_t radial, const std::size_t theta,
    const std::size_t field_count, const std::size_t radial_count,
    const std::size_t theta_count) {
  return ((mode * field_count + field) * radial_count + radial) *
             theta_count +
         theta;
}

enum class Scratch : std::size_t { Df0, Count };

struct FirstDerivativeFunctor {
  const Complex* f0;
  Complex* scratch;
  std::size_t radial_count;
  std::size_t theta_count;
  double inverse_spacing;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t fields = static_cast<std::size_t>(Scratch::Count);
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t input = flat3(mode, 0, theta, radial_count, theta_count);
    scratch[flat4(mode, static_cast<std::size_t>(Scratch::Df0), radial,
                  theta, fields, radial_count, theta_count)] =
        radial_first_derivative_strided_at(
            RadialDiscretization::D105, f0 + input, radial_count, radial,
            inverse_spacing, theta_count);
  }
};

struct FinalizeFunctor {
  UniformRadialGrid grid;
  const Complex* f0;
  const Complex* f1;
  const Complex* z0_regular;
  const Complex* z1_regular;
  const Complex* scratch;
  Complex* state;
  Complex* endpoint_audit;
  std::size_t radial_count;
  std::size_t theta_count;

  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t flat) const {
    constexpr std::size_t scratch_fields =
        static_cast<std::size_t>(Scratch::Count);
    constexpr std::size_t state_fields = 2;
    constexpr std::size_t audit_fields = 5;
    const std::size_t plane = radial_count * theta_count;
    const std::size_t mode = flat / plane;
    const std::size_t within = flat - mode * plane;
    const std::size_t radial = within / theta_count;
    const std::size_t theta = within - radial * theta_count;
    const std::size_t input = flat3(mode, radial, theta, radial_count,
                                    theta_count);
    const double radius = grid.coordinate(radial);
    const Complex df0 = scratch[flat4(
        mode, static_cast<std::size_t>(Scratch::Df0), radial, theta,
        scratch_fields, radial_count, theta_count)];
    const Complex q0 =
        radius == 0.0
            ? plus2_extract_q0_at_scri(f0 + input, radial_count,
                                       1.0 / grid.spacing(), theta_count)
            : f0[input] / (radius * radius);
    const Complex q1 =
        radius == 0.0
            ? plus2_extract_q1_at_scri(f1 + input, radial_count,
                                       1.0 / grid.spacing(), theta_count)
            : f1[input] / radius;
    state[flat4(mode, 0, radial, theta, state_fields, radial_count,
                theta_count)] = q0 + z0_regular[input];
    state[flat4(mode, 1, radial, theta, state_fields, radial_count,
                theta_count)] = q1 + z1_regular[input];
    if (radial == 0) {
      const Complex audit[audit_fields]{f0[input], df0, f1[input], q0, q1};
      for (std::size_t field = 0; field < audit_fields; ++field) {
        endpoint_audit[(mode * audit_fields + field) * theta_count + theta] =
            audit[field];
      }
    }
  }
};

static_assert(std::is_trivially_copyable_v<FirstDerivativeFunctor>);
static_assert(std::is_trivially_copyable_v<FinalizeFunctor>);

}  // namespace plus2_curvature_initialization_detail

inline Plus2PeelingConvergenceCertificate qualify_plus2_peeling_convergence(
    const Plus2PeelingResolutionSample& coarse,
    const Plus2PeelingResolutionSample& medium,
    const Plus2PeelingResolutionSample& fine,
    const std::string& evidence_id,
    const double required_order = 4.0) {
  const auto finite_sample = [](const Plus2PeelingResolutionSample& sample) {
    const std::size_t endpoint_count =
        sample.signed_modes.size() * sample.cos_theta_nodes.size();
    bool finite_coefficients = sample.z0_scri.size() == endpoint_count &&
                               sample.z1_scri.size() == endpoint_count;
    for (const Complex value : sample.z0_scri) {
      finite_coefficients = finite_coefficients &&
                            std::isfinite(value.real()) &&
                            std::isfinite(value.imag());
    }
    for (const Complex value : sample.z1_scri) {
      finite_coefficients = finite_coefficients &&
                            std::isfinite(value.real()) &&
                            std::isfinite(value.imag());
    }
    bool finite_angles = !sample.cos_theta_nodes.empty();
    for (const double value : sample.cos_theta_nodes) {
      finite_angles = finite_angles && std::isfinite(value) &&
                      value >= -1.0 && value <= 1.0;
    }
    const double expected_spacing =
        (sample.upper_radius - sample.lower_radius) /
        static_cast<double>(sample.radial_count > 1
                                ? sample.radial_count - 1
                                : 1);
    const double spacing_scale =
        std::max({1.0, Kokkos::abs(expected_spacing),
                  Kokkos::abs(sample.spacing)});
    return sample.radial_count >=
               radial_minimum_points(RadialDiscretization::D105) &&
           sample.upper_radius > sample.lower_radius &&
           std::isfinite(sample.lower_radius) &&
           std::isfinite(sample.upper_radius) &&
           std::isfinite(sample.spacing) && sample.spacing > 0.0 &&
           Kokkos::abs(expected_spacing - sample.spacing) <=
               8.0 * std::numeric_limits<double>::epsilon() * spacing_scale &&
           std::isfinite(sample.peeling_residual_norm) &&
           sample.peeling_residual_norm >= 0.0 &&
           !sample.signed_modes.empty() && finite_angles &&
           !sample.profile_id.empty() && finite_coefficients;
  };
  const double coarse_ratio = coarse.spacing / medium.spacing;
  const double fine_ratio = medium.spacing / fine.spacing;
  const double ratio_scale = std::max(coarse_ratio, fine_ratio);
  if (evidence_id.empty() || !std::isfinite(required_order) ||
      required_order < 4.0 || !finite_sample(coarse) ||
      !finite_sample(medium) || !finite_sample(fine) ||
      !(coarse.spacing > medium.spacing && medium.spacing > fine.spacing) ||
      coarse.radial_count >= medium.radial_count ||
      medium.radial_count >= fine.radial_count ||
      coarse.lower_radius != medium.lower_radius ||
      medium.lower_radius != fine.lower_radius ||
      coarse.upper_radius != medium.upper_radius ||
      medium.upper_radius != fine.upper_radius ||
      coarse.signed_modes != medium.signed_modes ||
      medium.signed_modes != fine.signed_modes ||
      coarse.cos_theta_nodes != medium.cos_theta_nodes ||
      medium.cos_theta_nodes != fine.cos_theta_nodes ||
      coarse.profile_id != medium.profile_id ||
      medium.profile_id != fine.profile_id ||
      Kokkos::abs(coarse_ratio - fine_ratio) >
          1.0e-12 * ratio_scale) {
    throw std::invalid_argument("invalid plus2 peeling convergence evidence");
  }
  const double residual_order_cm =
      plus2_curvature_initialization_detail::observed_order(
          coarse.peeling_residual_norm, medium.peeling_residual_norm,
          coarse.spacing, medium.spacing);
  const double residual_order_mf =
      plus2_curvature_initialization_detail::observed_order(
          medium.peeling_residual_norm, fine.peeling_residual_norm,
          medium.spacing, fine.spacing);
  double coefficient_change_cm = 0.0;
  double coefficient_change_mf = 0.0;
  for (std::size_t point = 0; point < fine.z0_scri.size(); ++point) {
    coefficient_change_cm = std::max(
        {coefficient_change_cm,
         Kokkos::abs(coarse.z0_scri[point] - medium.z0_scri[point]),
         Kokkos::abs(coarse.z1_scri[point] - medium.z1_scri[point])});
    coefficient_change_mf = std::max(
        {coefficient_change_mf,
         Kokkos::abs(medium.z0_scri[point] - fine.z0_scri[point]),
         Kokkos::abs(medium.z1_scri[point] - fine.z1_scri[point])});
  }
  const double coefficient_order =
      plus2_curvature_initialization_detail::observed_order(
          coefficient_change_cm, coefficient_change_mf, coarse.spacing,
          medium.spacing);
  const double minimum_order =
      std::min({residual_order_cm, residual_order_mf, coefficient_order});
  if (!(minimum_order >= required_order)) {
    throw std::runtime_error(
        "plus2 peeling residuals or endpoint coefficients do not converge "
        "at fourth order");
  }
  return Plus2PeelingConvergenceCertificate(
      evidence_id, minimum_order, fine, coefficient_change_mf);
}

struct Plus2CurvatureInitializationContract {
  Plus2CurvatureFormulaId formula = Plus2CurvatureFormulaId::Unknown;
  Plus2PeelingEndpointOperatorId endpoint_operator =
      Plus2PeelingEndpointOperatorId::Unknown;
  Plus2RadialBoundaryPolicyId boundary_policy =
      Plus2RadialBoundaryPolicyId::Unknown;
  std::string profile_id;
  std::string metric_curvature_evidence_id;
  Plus2PeelingConvergenceCertificate peeling;
  std::string bianchi_consistency_evidence_id;
};

// Input numerators are the reviewed primitive decomposition
//   Z0=f0/R^2+z0_regular, Z1=f1/R+z1_regular.
// The workspace performs no allocation after construction and never divides
// by R at scri.  The caller remains responsible for producing f0/f1 from the
// same signed-mode h[0..2] stage and for earning the supplied certificate.
template <class ExecSpace = ExecutionSpace>
class Plus2CurvatureInitializationWorkspace {
 public:
  using execution_space = ExecSpace;
  using memory_space = typename execution_space::memory_space;
  using field_view =
      Kokkos::View<Complex***, Kokkos::LayoutRight, memory_space>;
  using state_view =
      Kokkos::View<Complex****, Kokkos::LayoutRight, memory_space>;
  using audit_view =
      Kokkos::View<Complex***, Kokkos::LayoutRight, memory_space>;

  Plus2CurvatureInitializationWorkspace(
      std::vector<int> signed_modes, std::vector<double> cos_theta_nodes,
      const std::size_t radial_count,
      const std::string& label = "plus2_curvature_initialization")
      : signed_modes_(std::move(signed_modes)),
        cos_theta_nodes_(std::move(cos_theta_nodes)),
        mode_count_(signed_modes_.size()),
        radial_count_(radial_count),
        theta_count_(cos_theta_nodes_.size()),
        scratch_(label + "_scratch", mode_count_,
                 static_cast<std::size_t>(
                     plus2_curvature_initialization_detail::Scratch::Count),
                 radial_count, theta_count_),
        candidate_(label + "_candidate", mode_count_, 2, radial_count,
                   theta_count_),
        endpoint_audit_(label + "_endpoint_audit", mode_count_, 5,
                        theta_count_) {
    bool angles_valid = !cos_theta_nodes_.empty();
    for (const double value : cos_theta_nodes_) {
      angles_valid = angles_valid && std::isfinite(value) && value >= -1.0 &&
                     value <= 1.0;
    }
    auto sorted_modes = signed_modes_;
    std::sort(sorted_modes.begin(), sorted_modes.end());
    const bool modes_unique =
        std::adjacent_find(sorted_modes.begin(), sorted_modes.end()) ==
        sorted_modes.end();
    if (mode_count_ == 0 || !modes_unique || !angles_valid ||
        radial_count < radial_minimum_points(RadialDiscretization::D105)) {
      throw std::invalid_argument(
          "plus2 curvature initialization requires D10-5 geometry");
    }
  }

  [[nodiscard]] audit_view endpoint_audit() const { return endpoint_audit_; }

  void initialize(const execution_space& execution,
                  const UniformRadialGrid& grid, const field_view& f0,
                  const field_view& f1, const field_view& z0_regular,
                  const field_view& z1_regular, const state_view& state,
                  const Plus2CurvatureInitializationContract& contract) {
    validate(grid, f0, f1, z0_regular, z1_regular, state, contract);
    using namespace plus2_curvature_initialization_detail;
    const std::size_t points = mode_count_ * radial_count_ * theta_count_;
    Kokkos::parallel_for(
        "plus2_curvature_initial_first_radial",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        FirstDerivativeFunctor{f0.data(), scratch_.data(),
                               radial_count_, theta_count_,
                               1.0 / grid.spacing()});
    Kokkos::parallel_for(
        "plus2_curvature_initial_finalize",
        Kokkos::RangePolicy<execution_space>(execution, 0, points),
        FinalizeFunctor{grid, f0.data(), f1.data(), z0_regular.data(),
                        z1_regular.data(), scratch_.data(), candidate_.data(),
                        endpoint_audit_.data(), radial_count_, theta_count_});
    execution.fence("audit plus2 curvature initialization");
    const auto audit = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, endpoint_audit_);
    double actual_residual = 0.0;
    for (std::size_t mode = 0; mode < mode_count_; ++mode) {
      for (std::size_t theta = 0; theta < theta_count_; ++theta) {
        for (std::size_t field = 0; field < 3; ++field) {
          const Complex value = audit(mode, field, theta);
          if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            throw std::runtime_error(
                "nonfinite plus2 peeling endpoint residual");
          }
          actual_residual = std::max(actual_residual, Kokkos::abs(value));
        }
      }
    }
    const double residual_allowance =
        std::max(1.0e-14,
                 2.0 * contract.peeling.qualified_fine_sample()
                           .peeling_residual_norm);
    if (actual_residual > residual_allowance) {
      throw std::runtime_error(
          "plus2 endpoint residual does not match qualified data");
    }
    const double coefficient_allowance =
        std::max(1.0e-13,
                 2.0 * contract.peeling.maximum_coefficient_change());
    const auto& qualified = contract.peeling.qualified_fine_sample();
    for (std::size_t mode = 0; mode < mode_count_; ++mode) {
      for (std::size_t theta = 0; theta < theta_count_; ++theta) {
        const std::size_t point = mode * theta_count_ + theta;
        if (Kokkos::abs(audit(mode, 3, theta) -
                        qualified.z0_scri[point]) >
                coefficient_allowance ||
            Kokkos::abs(audit(mode, 4, theta) -
                        qualified.z1_scri[point]) >
                coefficient_allowance) {
          throw std::runtime_error(
              "plus2 endpoint coefficients do not match qualified data");
        }
      }
    }
    const auto candidate = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, candidate_);
    for (std::size_t mode = 0; mode < mode_count_; ++mode) {
      for (std::size_t field = 0; field < 2; ++field) {
        for (std::size_t radial = 0; radial < radial_count_; ++radial) {
          for (std::size_t theta = 0; theta < theta_count_; ++theta) {
            const Complex value = candidate(mode, field, radial, theta);
            if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
              throw std::runtime_error(
                  "nonfinite plus2 curvature initialization candidate");
            }
          }
        }
      }
    }
    Kokkos::deep_copy(execution, state, candidate_);
  }

 private:
  void validate(const UniformRadialGrid& grid, const field_view& f0,
                const field_view& f1, const field_view& z0_regular,
                const field_view& z1_regular, const state_view& state,
                const Plus2CurvatureInitializationContract& contract) const {
    const auto field_shape = [&](const field_view& view) {
      return view.extent(0) == mode_count_ &&
             view.extent(1) == radial_count_ &&
             view.extent(2) == theta_count_;
    };
    const auto& qualified = contract.peeling.qualified_fine_sample();
    if (grid.lower_radius() != 0.0 || grid.size() != radial_count_ ||
        !field_shape(f0) || !field_shape(f1) || !field_shape(z0_regular) ||
        !field_shape(z1_regular) || state.extent(0) != mode_count_ ||
        state.extent(1) != 2 || state.extent(2) != radial_count_ ||
        state.extent(3) != theta_count_ ||
        contract.metric_curvature_evidence_id.empty() ||
        contract.peeling.evidence_id().empty() ||
        contract.peeling.minimum_observed_order() < 4.0 ||
        qualified.radial_count != grid.size() ||
        qualified.lower_radius != grid.lower_radius() ||
        qualified.upper_radius != grid.upper_radius() ||
        qualified.spacing != grid.spacing() ||
        qualified.signed_modes != signed_modes_ ||
        qualified.cos_theta_nodes != cos_theta_nodes_ ||
        contract.profile_id.empty() ||
        contract.profile_id != qualified.profile_id ||
        contract.bianchi_consistency_evidence_id.empty() ||
        contract.formula !=
            Plus2CurvatureFormulaId::OrgRicciPeelingNumeratorsV1 ||
        contract.endpoint_operator !=
            Plus2PeelingEndpointOperatorId::ConstrainedPositiveNodesV2 ||
        contract.boundary_policy !=
            Plus2RadialBoundaryPolicyId::
                ContinuumNoIncomingButWeaklyHyperbolicV1) {
      throw std::invalid_argument(
          "plus2 curvature initialization evidence or geometry is "
          "incomplete");
    }
  }

  std::vector<int> signed_modes_;
  std::vector<double> cos_theta_nodes_;
  std::size_t mode_count_;
  std::size_t radial_count_;
  std::size_t theta_count_;
  state_view scratch_;
  state_view candidate_;
  audit_view endpoint_audit_;
};

}  // namespace teuk
