#pragma once

#include <Kokkos_Complex.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "teuk/background.hpp"
#include "teuk/reconstruction.hpp"

namespace teuk {

struct ReconstructionFieldDerivatives {
  Complex G;
  Complex Lambda;
  Complex H;
  Complex B;
  Complex Pi;
  Complex C;
  Complex U;
};

struct ReconstructionResiduals {
  Complex G;
  Complex Lambda;
  Complex H;
  Complex B;
  Complex Pi;
  Complex C;
  Complex U;
};

KOKKOS_INLINE_FUNCTION
Complex reconstruction_delta_lhs_point(
    const Complex& value, const Complex& time_derivative,
    const Complex& radial_derivative, const int radial_falloff,
    const double radius, const double mass,
    const double compactification_length) {
  const double length2 =
      compactification_length * compactification_length;
  return (2.0 + 4.0 * mass * radius / length2) * time_derivative +
         (radius / length2) *
             (radius * radial_derivative + radial_falloff * value);
}

// Form the strong-form transport-equation consistency residual L_X-R_X for
// each equation Delta_n X=R_X. The right sides are deliberately repeated here
// rather than obtained from reconstruction_delta_rhs, so this can catch an
// implementation mismatch. These are not the independent Bianchi/reality
// constraints of the reconstruction formalism.
KOKKOS_INLINE_FUNCTION
ReconstructionResiduals reconstruction_residuals_point(
    const double radius, const double mass,
    const double compactification_length,
    const KerrBackgroundPoint& background,
    const ReconstructionFields& fields,
    const ReconstructionAngularDerivatives& angular,
    const ReconstructionFieldDerivatives& time_derivatives,
    const ReconstructionFieldDerivatives& radial_derivatives) {
  const Complex mu0_bar = Kokkos::conj(background.mu0);
  const Complex pi0_bar = Kokkos::conj(background.pi0);
  const Complex tau0_bar = Kokkos::conj(background.tau0);
  const double radius2 = radius * radius;

  const Complex rhs_G =
      -4.0 * radius * background.mu0 * fields.G + angular.eth1_F -
      radius * background.tau0 * fields.F;
  const Complex rhs_Lambda =
      -radius * (background.mu0 + mu0_bar) * fields.Lambda - fields.F;
  const Complex rhs_H =
      -3.0 * radius * background.mu0 * fields.H + angular.eth2_G -
      2.0 * radius * background.tau0 * fields.G;
  const Complex rhs_B =
      radius * (background.mu0 - mu0_bar) * fields.B - 2.0 * fields.Lambda;
  const Complex rhs_Pi =
      -fields.G - radius * (pi0_bar + background.tau0) * fields.Lambda +
      0.5 * radius2 * background.mu0 *
          (pi0_bar + background.tau0) * fields.B;
  const Complex rhs_C =
      -radius * mu0_bar * fields.C - 2.0 * fields.Pi -
      radius * background.tau0 * fields.B;

  const Complex rhs_U_transport = -radius * mu0_bar * fields.U;
  const Complex rhs_U_c_eth =
      -radius * background.mu0 * angular.eth2_C;
  const Complex rhs_U_c_connection =
      -radius2 * background.mu0 *
      (pi0_bar + 2.0 * background.tau0) * fields.C;
  const Complex rhs_U_pi_eth = -2.0 * angular.eth2_Pi;
  const Complex rhs_U_pi_connection =
      -2.0 * radius * pi0_bar * fields.Pi;
  const Complex rhs_U_psi2 = -2.0 * fields.H;
  const Complex rhs_U_sharp_pi =
      -2.0 * radius * background.pi0 * fields.Pi_sharp;
  const Complex rhs_U_sharp_b_eth =
      -radius * background.pi0 * angular.ethprime1_B_sharp;
  const Complex rhs_U_sharp_b_connection =
      radius2 * background.pi0 * background.pi0 * fields.B_sharp;
  const Complex rhs_U_sharp_c_eth =
      radius * background.mu0 * angular.ethprime2_C_sharp;
  const Complex rhs_U_sharp_c_connection =
      radius2 * (-3.0 * background.mu0 * background.pi0 +
                 2.0 * mu0_bar * background.pi0 -
                 2.0 * background.mu0 * tau0_bar) *
      fields.C_sharp;
  const Complex rhs_U =
      rhs_U_transport + rhs_U_c_eth + rhs_U_c_connection + rhs_U_pi_eth +
      rhs_U_pi_connection + rhs_U_psi2 + rhs_U_sharp_pi +
      rhs_U_sharp_b_eth + rhs_U_sharp_b_connection + rhs_U_sharp_c_eth +
      rhs_U_sharp_c_connection;

  ReconstructionResiduals residuals;
  residuals.G = reconstruction_delta_lhs_point(
                    fields.G, time_derivatives.G, radial_derivatives.G, 2,
                    radius, mass, compactification_length) -
                rhs_G;
  residuals.Lambda =
      reconstruction_delta_lhs_point(
          fields.Lambda, time_derivatives.Lambda, radial_derivatives.Lambda, 1,
          radius, mass, compactification_length) -
      rhs_Lambda;
  residuals.H = reconstruction_delta_lhs_point(
                    fields.H, time_derivatives.H, radial_derivatives.H, 3,
                    radius, mass, compactification_length) -
                rhs_H;
  residuals.B = reconstruction_delta_lhs_point(
                    fields.B, time_derivatives.B, radial_derivatives.B, 1,
                    radius, mass, compactification_length) -
                rhs_B;
  residuals.Pi = reconstruction_delta_lhs_point(
                     fields.Pi, time_derivatives.Pi, radial_derivatives.Pi, 2,
                     radius, mass, compactification_length) -
                 rhs_Pi;
  residuals.C = reconstruction_delta_lhs_point(
                    fields.C, time_derivatives.C, radial_derivatives.C, 2,
                    radius, mass, compactification_length) -
                rhs_C;
  residuals.U = reconstruction_delta_lhs_point(
                    fields.U, time_derivatives.U, radial_derivatives.U, 3,
                    radius, mass, compactification_length) -
                rhs_U;
  return residuals;
}

struct ResidualNorm {
  double rms = 0.0;
  double maximum = 0.0;
};

struct ReconstructionResidualNorms {
  ResidualNorm G;
  ResidualNorm Lambda;
  ResidualNorm H;
  ResidualNorm B;
  ResidualNorm Pi;
  ResidualNorm C;
  ResidualNorm U;
  ResidualNorm combined;
};

inline ReconstructionResidualNorms reconstruction_residual_norms(
    const std::vector<ReconstructionResiduals>& residuals) {
  if (residuals.empty()) {
    throw std::invalid_argument(
        "reconstruction residual norms require at least one point");
  }

  std::array<double, 7> sums{};
  std::array<double, 7> maxima{};
  double combined_sum = 0.0;
  double combined_maximum = 0.0;
  for (const ReconstructionResiduals& point : residuals) {
    const std::array<Complex, 7> values{point.G, point.Lambda, point.H,
                                        point.B, point.Pi, point.C, point.U};
    for (std::size_t field = 0; field < values.size(); ++field) {
      const double magnitude = Kokkos::abs(values[field]);
      sums[field] += magnitude * magnitude;
      maxima[field] = std::max(maxima[field], magnitude);
      combined_sum += magnitude * magnitude;
      combined_maximum = std::max(combined_maximum, magnitude);
    }
  }

  const double point_count = static_cast<double>(residuals.size());
  std::array<ResidualNorm, 7> norms{};
  for (std::size_t field = 0; field < norms.size(); ++field) {
    norms[field] = {std::sqrt(sums[field] / point_count), maxima[field]};
  }
  const ResidualNorm combined{
      std::sqrt(combined_sum / (7.0 * point_count)), combined_maximum};
  return {norms[0], norms[1], norms[2], norms[3],
          norms[4], norms[5], norms[6], combined};
}

}  // namespace teuk
