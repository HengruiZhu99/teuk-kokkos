#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/background.hpp"
#include "teuk/modes.hpp"
#include "teuk/second_order.hpp"

namespace teuk::angular_source {

// Everything in this header is an allocation-owning host reference layer. It
// prepares and verifies modal data around the allocation-free point algebra in
// second_order.hpp. Production device kernels consume flattened matrices,
// registry pair tables, and point structs rather than these std::vectors.

class HostModalField {
 public:
  HostModalField(const ModeRegistry& registry, const int spin,
                 const int ell_max)
      : registry_(&registry), spin_(spin), ell_max_(ell_max) {
    for (const int m : registry.modes()) {
      const int ell_min = angular::minimum_ell(spin, m);
      if (ell_max < ell_min) {
        throw std::invalid_argument(
            "field ell_max does not support every registered mode");
      }
      modal_by_index_.emplace_back(
          static_cast<std::size_t>(ell_max - ell_min + 1),
          Complex(0.0, 0.0));
    }
  }

  [[nodiscard]] const ModeRegistry& registry() const noexcept {
    return *registry_;
  }
  [[nodiscard]] int spin() const noexcept { return spin_; }
  [[nodiscard]] int ell_max() const noexcept { return ell_max_; }

  [[nodiscard]] std::vector<Complex>& mode(const int m) {
    return modal_by_index_.at(registry_->index(m));
  }
  [[nodiscard]] const std::vector<Complex>& mode(const int m) const {
    return modal_by_index_.at(registry_->index(m));
  }

  // Modal representation of X_m^sharp = conjugate(X_{-m}). Conjugating a
  // spin-s harmonic changes its spin and contributes the standard
  // (-1)^(s-m) phase in the unit-normalized Condon-Shortley convention.
  [[nodiscard]] std::vector<Complex> sharp_mode(const int m) const {
    const auto& source = modal_by_index_.at(registry_->sharp_index(m));
    const Real phase = ((spin_ - m) & 1) != 0 ? -1.0 : 1.0;
    std::vector<Complex> result(source.size());
    for (std::size_t ell = 0; ell < source.size(); ++ell) {
      result[ell] = phase * Kokkos::conj(source[ell]);
    }
    return result;
  }

  [[nodiscard]] HostModalField sharp_field() const {
    HostModalField result(*registry_, -spin_, ell_max_);
    for (const int m : registry_->modes()) result.mode(m) = sharp_mode(m);
    return result;
  }

 private:
  const ModeRegistry* registry_;
  int spin_;
  int ell_max_;
  std::vector<std::vector<Complex>> modal_by_index_;
};

struct HostPairProductProjection {
  ModePair pair;
  std::vector<Complex> modal;
};

struct HostTargetProductProjection {
  int target;
  std::vector<Complex> modal;
};

struct HostMultimodeProductProjection {
  // Both lists preserve ModeRegistry ordering exactly.
  std::vector<HostPairProductProjection> pair_contributions;
  std::vector<HostTargetProductProjection> targets;
};

enum class HostProductMethod { ExactGaunt, PaddedCollocation };

inline void validate_registry_identity(const ModeRegistry& registry,
                                       const HostModalField& field,
                                       const char* name) {
  if (&field.registry() != &registry) {
    throw std::invalid_argument(std::string(name) +
                                " uses a different ModeRegistry instance");
  }
}

[[nodiscard]] inline HostMultimodeProductProjection project_product(
    const ModeRegistry& registry, const HostModalField& left,
    const HostModalField& right, const int target_ell_max,
    const HostProductMethod method) {
  validate_registry_identity(registry, left, "left field");
  validate_registry_identity(registry, right, "right field");
  const int target_spin = left.spin() + right.spin();

  HostMultimodeProductProjection result;
  result.targets.reserve(registry.targets().size());
  result.pair_contributions.reserve(registry.ordered_pairs().size());
  for (const int target_m : registry.targets()) {
    const angular::SpinWeightedTransform target_transform(
        target_spin, target_m, target_ell_max);
    HostTargetProductProjection target{
        target_m, std::vector<Complex>(target_transform.mode_count(),
                                      Complex(0.0, 0.0))};
    const auto [begin, end] = registry.pair_range(target_m);
    for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
      const ModePair pair = registry.ordered_pairs()[pair_index];
      const angular::SpinWeightedTransform left_transform(
          left.spin(), pair.m1, left.ell_max());
      const angular::SpinWeightedTransform right_transform(
          right.spin(), pair.m2, right.ell_max());
      std::vector<Complex> contribution;
      if (method == HostProductMethod::ExactGaunt) {
        contribution = angular::exact_modal_product(
            left_transform, left.mode(pair.m1), right_transform,
            right.mode(pair.m2), target_transform);
      } else {
        contribution = angular::collocation_product(
            left_transform, left.mode(pair.m1), right_transform,
            right.mode(pair.m2), target_transform);
      }
      for (std::size_t ell = 0; ell < target.modal.size(); ++ell) {
        target.modal[ell] += contribution[ell];
      }
      result.pair_contributions.push_back({pair, std::move(contribution)});
    }
    result.targets.push_back(std::move(target));
  }
  return result;
}

// Raw physical fields used by corrected_ordered_pair_source. Sharp values are
// not aliases: the projection routine obtains them through sharp_mode(m), which
// performs the registry lookup of -m before conjugating.
struct HostSourceFields {
  const HostModalField& F;
  const HostModalField& G;
  const HostModalField& H;
  const HostModalField& Lambda;
  const HostModalField& Pi;
  const HostModalField& B;
  const HostModalField& C;
  const HostModalField& U;
};

// Angular/radial derivative fields are explicit because a sharp derivative is
// generally not interchangeable with conjugating the unsharp derivative after
// a spin-changing operator has acted.
struct HostSourceDerivatives {
  const HostModalField& delta1_F;
  const HostModalField& delta3_U;
  const HostModalField& eth2_C;
  const HostModalField& ethprime2_C_sharp;
  const HostModalField& eth1_B;
  const HostModalField& delta2_C;
  const HostModalField& delta2_G;
  const HostModalField& eth2_G;
  const HostModalField& ethprime1_F;
  const HostModalField& delta2_C_sharp;
  const HostModalField& ethprime1_B_sharp;
};

struct HostPairInnerSourceProjection {
  ModePair pair;
  std::vector<Complex> D;
  std::vector<Complex> T;
};

struct HostTargetInnerSourceProjection {
  int target;
  std::vector<Complex> D;
  std::vector<Complex> T;
};

struct HostMultimodeInnerSourceProjection {
  std::vector<HostPairInnerSourceProjection> pair_contributions;
  std::vector<HostTargetInnerSourceProjection> targets;
};

namespace detail {

inline void require_field(const ModeRegistry& registry,
                          const HostModalField& field, const int expected_spin,
                          const char* name) {
  validate_registry_identity(registry, field, name);
  if (field.spin() != expected_spin) {
    throw std::invalid_argument(std::string(name) + " has the wrong spin");
  }
}

inline std::vector<Complex> synthesize(const HostModalField& field,
                                       const int m, const int node_count) {
  const angular::SpinWeightedTransform transform(field.spin(), m,
                                                 field.ell_max(), node_count);
  return transform.synthesize(field.mode(m));
}

inline std::vector<Complex> synthesize_sharp(const HostModalField& field,
                                             const int m,
                                             const int node_count) {
  const angular::SpinWeightedTransform transform(-field.spin(), m,
                                                 field.ell_max(), node_count);
  return transform.synthesize(field.sharp_mode(m));
}

inline int maximum_ell(const HostSourceFields& fields,
                       const HostSourceDerivatives& derivatives,
                       const int target_ell_max) noexcept {
  return std::max(
      {target_ell_max, fields.F.ell_max(), fields.G.ell_max(),
       fields.H.ell_max(), fields.Lambda.ell_max(), fields.Pi.ell_max(),
       fields.B.ell_max(), fields.C.ell_max(), fields.U.ell_max(),
       derivatives.delta1_F.ell_max(), derivatives.delta3_U.ell_max(),
       derivatives.eth2_C.ell_max(),
       derivatives.ethprime2_C_sharp.ell_max(),
       derivatives.eth1_B.ell_max(), derivatives.delta2_C.ell_max(),
       derivatives.delta2_G.ell_max(), derivatives.eth2_G.ell_max(),
       derivatives.ethprime1_F.ell_max(),
       derivatives.delta2_C_sharp.ell_max(),
       derivatives.ethprime1_B_sharp.ell_max()});
}

}  // namespace detail

inline void validate_source_metadata(
    const ModeRegistry& registry, const HostSourceFields& fields,
    const HostSourceDerivatives& derivatives) {
  detail::require_field(registry, fields.F, -2, "F");
  detail::require_field(registry, fields.G, -1, "G");
  detail::require_field(registry, fields.H, 0, "H");
  detail::require_field(registry, fields.Lambda, -2, "Lambda");
  detail::require_field(registry, fields.Pi, -1, "Pi");
  detail::require_field(registry, fields.B, -2, "B");
  detail::require_field(registry, fields.C, -1, "C");
  detail::require_field(registry, fields.U, 0, "U");

  detail::require_field(registry, derivatives.delta1_F, -2, "delta1_F");
  detail::require_field(registry, derivatives.delta3_U, 0, "delta3_U");
  detail::require_field(registry, derivatives.eth2_C, 0, "eth2_C");
  detail::require_field(registry, derivatives.ethprime2_C_sharp, 0,
                        "ethprime2_C_sharp");
  detail::require_field(registry, derivatives.eth1_B, -1, "eth1_B");
  detail::require_field(registry, derivatives.delta2_C, -1, "delta2_C");
  detail::require_field(registry, derivatives.delta2_G, -1, "delta2_G");
  detail::require_field(registry, derivatives.eth2_G, 0, "eth2_G");
  detail::require_field(registry, derivatives.ethprime1_F, -3,
                        "ethprime1_F");
  detail::require_field(registry, derivatives.delta2_C_sharp, 1,
                        "delta2_C_sharp");
  detail::require_field(registry, derivatives.ethprime1_B_sharp, 1,
                        "ethprime1_B_sharp");
}

// Host collocation assembly of the complete inner source. Every ordered pair is
// synthesized separately, evaluated by the same KOKKOS_INLINE_FUNCTION point
// algebra used in production, projected, recorded, and only then accumulated.
[[nodiscard]] inline HostMultimodeInnerSourceProjection project_inner_source(
    const ModeRegistry& registry, const HostSourceFields& fields,
    const HostSourceDerivatives& derivatives,
    const KerrParameters& parameters, const Real radius,
    const int target_ell_max, int node_count = 0) {
  validate_source_metadata(registry, fields, derivatives);
  if (node_count == 0) {
    // Products and the smooth axisymmetric Kerr coefficients are deliberately
    // overcollocated in this reference path. This is not an exact Gaunt claim
    // for the nonpolynomial background coefficients.
    node_count = 3 * detail::maximum_ell(fields, derivatives,
                                         target_ell_max) +
                 1;
  }

  HostMultimodeInnerSourceProjection result;
  result.targets.reserve(registry.targets().size());
  result.pair_contributions.reserve(registry.ordered_pairs().size());
  for (const int target_m : registry.targets()) {
    const angular::SpinWeightedTransform D_transform(-2, target_m,
                                                     target_ell_max,
                                                     node_count);
    const angular::SpinWeightedTransform T_transform(-1, target_m,
                                                     target_ell_max,
                                                     node_count);
    HostTargetInnerSourceProjection target{
        target_m,
        std::vector<Complex>(D_transform.mode_count(), Complex(0.0, 0.0)),
        std::vector<Complex>(T_transform.mode_count(), Complex(0.0, 0.0))};

    const auto [begin, end] = registry.pair_range(target_m);
    for (std::size_t pair_index = begin; pair_index < end; ++pair_index) {
      const ModePair pair = registry.ordered_pairs()[pair_index];

      const auto F1 = detail::synthesize(fields.F, pair.m1, node_count);
      const auto G1 = detail::synthesize(fields.G, pair.m1, node_count);
      const auto Lambda1 =
          detail::synthesize(fields.Lambda, pair.m1, node_count);
      const auto Pi1 = detail::synthesize(fields.Pi, pair.m1, node_count);
      const auto B1 = detail::synthesize(fields.B, pair.m1, node_count);
      const auto C1 = detail::synthesize(fields.C, pair.m1, node_count);
      const auto U1 = detail::synthesize(fields.U, pair.m1, node_count);

      const auto F2 = detail::synthesize(fields.F, pair.m2, node_count);
      const auto G2 = detail::synthesize(fields.G, pair.m2, node_count);
      const auto H2 = detail::synthesize(fields.H, pair.m2, node_count);
      const auto B2 = detail::synthesize(fields.B, pair.m2, node_count);
      const auto C2 = detail::synthesize(fields.C, pair.m2, node_count);
      const auto U2 = detail::synthesize(fields.U, pair.m2, node_count);

      const auto U2_sharp =
          detail::synthesize_sharp(fields.U, pair.m2, node_count);
      const auto C2_sharp =
          detail::synthesize_sharp(fields.C, pair.m2, node_count);
      const auto C1_sharp =
          detail::synthesize_sharp(fields.C, pair.m1, node_count);
      const auto B1_sharp =
          detail::synthesize_sharp(fields.B, pair.m1, node_count);
      const auto Pi2_sharp =
          detail::synthesize_sharp(fields.Pi, pair.m2, node_count);
      const auto B2_sharp =
          detail::synthesize_sharp(fields.B, pair.m2, node_count);

      const auto delta1_F2 =
          detail::synthesize(derivatives.delta1_F, pair.m2, node_count);
      const auto delta3_U2 =
          detail::synthesize(derivatives.delta3_U, pair.m2, node_count);
      const auto eth2_C2 =
          detail::synthesize(derivatives.eth2_C, pair.m2, node_count);
      const auto ethprime2_C2_sharp = detail::synthesize(
          derivatives.ethprime2_C_sharp, pair.m2, node_count);
      const auto eth1_B2 =
          detail::synthesize(derivatives.eth1_B, pair.m2, node_count);
      const auto delta2_C2 =
          detail::synthesize(derivatives.delta2_C, pair.m2, node_count);
      const auto delta2_G2 =
          detail::synthesize(derivatives.delta2_G, pair.m2, node_count);
      const auto eth2_G2 =
          detail::synthesize(derivatives.eth2_G, pair.m2, node_count);
      const auto ethprime1_F2 = detail::synthesize(
          derivatives.ethprime1_F, pair.m2, node_count);
      const auto delta2_C2_sharp = detail::synthesize(
          derivatives.delta2_C_sharp, pair.m2, node_count);
      const auto ethprime1_B2_sharp = detail::synthesize(
          derivatives.ethprime1_B_sharp, pair.m2, node_count);

      std::vector<Complex> D_nodal(static_cast<std::size_t>(node_count));
      std::vector<Complex> T_nodal(static_cast<std::size_t>(node_count));
      for (int node = 0; node < node_count; ++node) {
        const auto i = static_cast<std::size_t>(node);
        const Real cos_theta = D_transform.grid().x[i];
        const Real sin_theta =
            std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
        const auto background = kerr_background_point(
            parameters, radius, cos_theta, sin_theta);
        const OrderedPairFields point_fields{
            F1[i], G1[i], Lambda1[i], Pi1[i], B1[i], C1[i], U1[i],
            F2[i], G2[i], H2[i], B2[i], C2[i], U2[i], U2_sharp[i],
            C2_sharp[i], C1_sharp[i], B1_sharp[i], Pi2_sharp[i],
            B2_sharp[i]};
        const OrderedPairDerivatives point_derivatives{
            delta1_F2[i], delta3_U2[i], eth2_C2[i],
            ethprime2_C2_sharp[i], eth1_B2[i], delta2_C2[i],
            delta2_G2[i], eth2_G2[i], ethprime1_F2[i],
            delta2_C2_sharp[i], ethprime1_B2_sharp[i]};
        const auto source = corrected_ordered_pair_source(
            radius, background, point_fields, point_derivatives);
        D_nodal[i] = source.D;
        T_nodal[i] = source.T;
      }

      HostPairInnerSourceProjection contribution{
          pair, D_transform.analyze(D_nodal), T_transform.analyze(T_nodal)};
      for (std::size_t ell = 0; ell < target.D.size(); ++ell) {
        target.D[ell] += contribution.D[ell];
      }
      for (std::size_t ell = 0; ell < target.T.size(); ++ell) {
        target.T[ell] += contribution.T[ell];
      }
      result.pair_contributions.push_back(std::move(contribution));
    }
    result.targets.push_back(std::move(target));
  }
  return result;
}

}  // namespace teuk::angular_source
