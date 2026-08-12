#pragma once

#include <Kokkos_Core.hpp>

#include "teuk/background.hpp"
#include "teuk/plus2_field.hpp"
#include "teuk/types.hpp"

namespace teuk {

// Physical ORG tetrad projections in the code tetrad.  All entries carry the
// same signed azimuthal mode exp(i m phi).
template <class Scalar>
struct Plus2OrgMetricFieldsT {
  Scalar h_ll{};
  Scalar h_lm{};
  Scalar h_mm{};
};

using Plus2OrgMetricFields = Plus2OrgMetricFieldsT<Complex>;

// T0 is second order in the reconstructed metric.  These three values must
// come from one RK stage: h_T is the stage tangent and h_TT is the tangent of
// that tangent.  Endpoint interpolation is not a valid way to fill them.
struct Plus2OrgMetricStage {
  Plus2OrgMetricFields h;
  Plus2OrgMetricFields h_T;
  Plus2OrgMetricFields h_TT;
};

// Coordinate-spatial derivative slots at that same stage.  The angular slots
// are ordinary coordinate derivatives, not eth/eth-prime values.  Explicit
// phi slots keep the local operator independent of the modal representation;
// plus2_fill_modal_azimuthal_derivatives supplies their exact i*m form.
struct Plus2OrgMetricDerivativeSlots {
  Plus2OrgMetricFields h_R;
  Plus2OrgMetricFields h_TR;
  Plus2OrgMetricFields h_RR;
  Plus2OrgMetricFields h_theta;
  Plus2OrgMetricFields h_Ttheta;
  Plus2OrgMetricFields h_Rtheta;
  Plus2OrgMetricFields h_thetatheta;
  Plus2OrgMetricFields h_phi;
  Plus2OrgMetricFields h_Tphi;
  Plus2OrgMetricFields h_Rphi;
  Plus2OrgMetricFields h_thetaphi;
  Plus2OrgMetricFields h_phiphi;
};

KOKKOS_INLINE_FUNCTION void plus2_fill_modal_azimuthal_derivatives(
    const int azimuthal_mode, const Plus2OrgMetricStage& stage,
    Plus2OrgMetricDerivativeSlots& derivatives) {
  const Complex im(0.0, static_cast<double>(azimuthal_mode));
  const double mode = static_cast<double>(azimuthal_mode);
  const double minus_m_squared = -mode * mode;
  derivatives.h_phi = {im * stage.h.h_ll, im * stage.h.h_lm,
                       im * stage.h.h_mm};
  derivatives.h_Tphi = {im * stage.h_T.h_ll, im * stage.h_T.h_lm,
                        im * stage.h_T.h_mm};
  derivatives.h_Rphi = {im * derivatives.h_R.h_ll,
                        im * derivatives.h_R.h_lm,
                        im * derivatives.h_R.h_mm};
  derivatives.h_thetaphi = {im * derivatives.h_theta.h_ll,
                            im * derivatives.h_theta.h_lm,
                            im * derivatives.h_theta.h_mm};
  derivatives.h_phiphi = {
      minus_m_squared * stage.h.h_ll, minus_m_squared * stage.h.h_lm,
      minus_m_squared * stage.h.h_mm};
}

// Conversion from the repository's stored reconstruction variables.  Sharp
// values must already have been obtained as X_m^sharp=conj(X_{-m}); this helper
// intentionally cannot form a same-mode conjugate.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Plus2OrgMetricFieldsT<Scalar>
plus2_org_metric_from_reconstruction(const double radius,
                                     const Scalar& mu0,
                                     const Scalar& B_sharp,
                                     const Scalar& C_sharp,
                                     const Scalar& U) {
  const double radius2 = radius * radius;
  return {radius2 * U / mu0, radius2 * C_sharp, radius * B_sharp};
}

// Physical background coefficients in Ripley et al. arXiv:2010.00162 Eq. (8)
// after the coordinate definition r=L^2/R in Eq. (6).  rho, epsilon, tau and
// pi are reconciled below with kerr_background_point; alpha and beta are the
// two additional coefficients required by ordinary-NP T0.
struct Plus2LinearBackgroundPoint {
  Complex rho{};
  Complex epsilon{};
  Complex alpha{};
  Complex beta{};
  Complex tau{};
  Complex pi{};
};

namespace plus2_linear_detail {

enum CoordinateIndex : int {
  Time = 0,
  Radius = 1,
  Theta = 2,
  Phi = 3,
  Count = 4
};

// Internal fixed-size second coordinate jet.  It exists only to apply product
// and quotient rules to the analytic Kerr tetrad/background.  The public API
// remains the named stage and derivative slots above.
struct PointJet2 {
  Complex value{};
  Complex first[Count]{};
  Complex second[Count][Count]{};
};

KOKKOS_INLINE_FUNCTION PointJet2 constant(const Complex& value) {
  PointJet2 result;
  result.value = value;
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 constant(const double value) {
  return constant(Complex(value, 0.0));
}

KOKKOS_INLINE_FUNCTION PointJet2 coordinate(const double value,
                                             const int index) {
  PointJet2 result = constant(value);
  result.first[index] = Complex(1.0, 0.0);
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator+(const PointJet2& left,
                                           const PointJet2& right) {
  PointJet2 result;
  result.value = left.value + right.value;
  for (int i = 0; i < Count; ++i) {
    result.first[i] = left.first[i] + right.first[i];
    for (int j = 0; j < Count; ++j) {
      result.second[i][j] = left.second[i][j] + right.second[i][j];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator-(const PointJet2& input) {
  PointJet2 result;
  result.value = -input.value;
  for (int i = 0; i < Count; ++i) {
    result.first[i] = -input.first[i];
    for (int j = 0; j < Count; ++j) {
      result.second[i][j] = -input.second[i][j];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator-(const PointJet2& left,
                                           const PointJet2& right) {
  return left + (-right);
}

KOKKOS_INLINE_FUNCTION PointJet2 operator*(const PointJet2& left,
                                           const PointJet2& right) {
  PointJet2 result;
  result.value = left.value * right.value;
  for (int i = 0; i < Count; ++i) {
    result.first[i] =
        left.first[i] * right.value + left.value * right.first[i];
    for (int j = 0; j < Count; ++j) {
      result.second[i][j] =
          left.second[i][j] * right.value +
          left.first[i] * right.first[j] +
          left.first[j] * right.first[i] +
          left.value * right.second[i][j];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 reciprocal(const PointJet2& input) {
  PointJet2 result;
  const Complex inverse = Complex(1.0, 0.0) / input.value;
  const Complex inverse2 = inverse * inverse;
  const Complex inverse3 = inverse2 * inverse;
  result.value = inverse;
  for (int i = 0; i < Count; ++i) {
    result.first[i] = -input.first[i] * inverse2;
    for (int j = 0; j < Count; ++j) {
      result.second[i][j] =
          2.0 * input.first[i] * input.first[j] * inverse3 -
          input.second[i][j] * inverse2;
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator/(const PointJet2& numerator,
                                           const PointJet2& denominator) {
  return numerator * reciprocal(denominator);
}

KOKKOS_INLINE_FUNCTION PointJet2 operator+(const PointJet2& left,
                                           const double right) {
  return left + constant(right);
}

KOKKOS_INLINE_FUNCTION PointJet2 operator+(const double left,
                                           const PointJet2& right) {
  return constant(left) + right;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator-(const PointJet2& left,
                                           const double right) {
  return left - constant(right);
}

KOKKOS_INLINE_FUNCTION PointJet2 operator-(const double left,
                                           const PointJet2& right) {
  return constant(left) - right;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator*(const PointJet2& left,
                                           const double right) {
  return left * constant(right);
}

KOKKOS_INLINE_FUNCTION PointJet2 operator*(const double left,
                                           const PointJet2& right) {
  return constant(left) * right;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator*(const PointJet2& left,
                                           const Complex& right) {
  return left * constant(right);
}

KOKKOS_INLINE_FUNCTION PointJet2 operator*(const Complex& left,
                                           const PointJet2& right) {
  return constant(left) * right;
}

KOKKOS_INLINE_FUNCTION PointJet2 operator/(const PointJet2& left,
                                           const double right) {
  return left / constant(right);
}

KOKKOS_INLINE_FUNCTION PointJet2 operator/(const double left,
                                           const PointJet2& right) {
  return constant(left) / right;
}

KOKKOS_INLINE_FUNCTION PointJet2 jet_conj(const PointJet2& input) {
  PointJet2 result;
  result.value = Kokkos::conj(input.value);
  for (int i = 0; i < Count; ++i) {
    result.first[i] = Kokkos::conj(input.first[i]);
    for (int j = 0; j < Count; ++j) {
      result.second[i][j] = Kokkos::conj(input.second[i][j]);
    }
  }
  return result;
}

struct FirstJet {
  Complex value{};
  Complex first[Count]{};
};

KOKKOS_INLINE_FUNCTION FirstJet first_jet(const PointJet2& input) {
  FirstJet result;
  result.value = input.value;
  for (int i = 0; i < Count; ++i) result.first[i] = input.first[i];
  return result;
}

KOKKOS_INLINE_FUNCTION FirstJet operator+(const FirstJet& left,
                                          const FirstJet& right) {
  FirstJet result;
  result.value = left.value + right.value;
  for (int i = 0; i < Count; ++i) {
    result.first[i] = left.first[i] + right.first[i];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION FirstJet operator-(const FirstJet& input) {
  FirstJet result;
  result.value = -input.value;
  for (int i = 0; i < Count; ++i) result.first[i] = -input.first[i];
  return result;
}

KOKKOS_INLINE_FUNCTION FirstJet operator-(const FirstJet& left,
                                          const FirstJet& right) {
  return left + (-right);
}

KOKKOS_INLINE_FUNCTION FirstJet operator*(const double factor,
                                          const FirstJet& input) {
  FirstJet result;
  result.value = factor * input.value;
  for (int i = 0; i < Count; ++i) result.first[i] = factor * input.first[i];
  return result;
}

KOKKOS_INLINE_FUNCTION FirstJet first_product(const PointJet2& left,
                                              const PointJet2& right) {
  FirstJet result;
  result.value = left.value * right.value;
  for (int i = 0; i < Count; ++i) {
    result.first[i] =
        left.first[i] * right.value + left.value * right.first[i];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 metric_jet(
    const Complex& value, const Complex& dt, const Complex& dtt,
    const Complex& dr, const Complex& dtr, const Complex& drr,
    const Complex& dtheta, const Complex& dttheta,
    const Complex& drtheta, const Complex& dthetatheta) {
  PointJet2 result = constant(value);
  result.first[Time] = dt;
  result.first[Radius] = dr;
  result.first[Theta] = dtheta;
  result.second[Time][Time] = dtt;
  result.second[Time][Radius] = dtr;
  result.second[Radius][Time] = dtr;
  result.second[Radius][Radius] = drr;
  result.second[Time][Theta] = dttheta;
  result.second[Theta][Time] = dttheta;
  result.second[Radius][Theta] = drtheta;
  result.second[Theta][Radius] = drtheta;
  result.second[Theta][Theta] = dthetatheta;
  return result;
}

KOKKOS_INLINE_FUNCTION PointJet2 metric_component(
    const Plus2OrgMetricFields& value, const Plus2OrgMetricFields& dt,
    const Plus2OrgMetricFields& dtt, const Plus2OrgMetricFields& dr,
    const Plus2OrgMetricFields& dtr, const Plus2OrgMetricFields& drr,
    const Plus2OrgMetricFields& dtheta,
    const Plus2OrgMetricFields& dttheta,
    const Plus2OrgMetricFields& drtheta,
    const Plus2OrgMetricFields& dthetatheta,
    const Plus2OrgMetricFields& dphi,
    const Plus2OrgMetricFields& dtphi,
    const Plus2OrgMetricFields& drphi,
    const Plus2OrgMetricFields& dthetaphi,
    const Plus2OrgMetricFields& dphiphi, const int component) {
  const Complex value_components[] = {value.h_ll, value.h_lm, value.h_mm};
  const Complex dt_components[] = {dt.h_ll, dt.h_lm, dt.h_mm};
  const Complex dtt_components[] = {dtt.h_ll, dtt.h_lm, dtt.h_mm};
  const Complex dr_components[] = {dr.h_ll, dr.h_lm, dr.h_mm};
  const Complex dtr_components[] = {dtr.h_ll, dtr.h_lm, dtr.h_mm};
  const Complex drr_components[] = {drr.h_ll, drr.h_lm, drr.h_mm};
  const Complex dtheta_components[] = {dtheta.h_ll, dtheta.h_lm,
                                       dtheta.h_mm};
  const Complex dttheta_components[] = {dttheta.h_ll, dttheta.h_lm,
                                        dttheta.h_mm};
  const Complex drtheta_components[] = {drtheta.h_ll, drtheta.h_lm,
                                        drtheta.h_mm};
  const Complex dthetatheta_components[] = {
      dthetatheta.h_ll, dthetatheta.h_lm, dthetatheta.h_mm};
  PointJet2 result = metric_jet(
      value_components[component], dt_components[component],
      dtt_components[component], dr_components[component],
      dtr_components[component], drr_components[component],
      dtheta_components[component], dttheta_components[component],
      drtheta_components[component], dthetatheta_components[component]);
  const Complex dphi_components[] = {dphi.h_ll, dphi.h_lm, dphi.h_mm};
  const Complex dtphi_components[] = {dtphi.h_ll, dtphi.h_lm, dtphi.h_mm};
  const Complex drphi_components[] = {drphi.h_ll, drphi.h_lm, drphi.h_mm};
  const Complex dthetaphi_components[] = {
      dthetaphi.h_ll, dthetaphi.h_lm, dthetaphi.h_mm};
  const Complex dphiphi_components[] = {dphiphi.h_ll, dphiphi.h_lm,
                                        dphiphi.h_mm};
  result.first[Phi] = dphi_components[component];
  result.second[Time][Phi] = dtphi_components[component];
  result.second[Phi][Time] = dtphi_components[component];
  result.second[Radius][Phi] = drphi_components[component];
  result.second[Phi][Radius] = drphi_components[component];
  result.second[Theta][Phi] = dthetaphi_components[component];
  result.second[Phi][Theta] = dthetaphi_components[component];
  result.second[Phi][Phi] = dphiphi_components[component];
  return result;
}

struct GeometryJets {
  PointJet2 l_T;
  PointJet2 l_R;
  PointJet2 l_phi;
  PointJet2 m_T;
  PointJet2 m_theta;
  PointJet2 m_phi;
  PointJet2 rho;
  PointJet2 epsilon;
  PointJet2 alpha;
  PointJet2 beta;
  PointJet2 tau;
  PointJet2 pi;
};

KOKKOS_INLINE_FUNCTION GeometryJets geometry_jets(
    const KerrParameters& parameters, const double radius_value,
    const double sin_theta_value, const double cos_theta_value) {
  const PointJet2 radius = coordinate(radius_value, Radius);
  PointJet2 sin_theta = constant(sin_theta_value);
  sin_theta.first[Theta] = Complex(cos_theta_value, 0.0);
  sin_theta.second[Theta][Theta] = Complex(-sin_theta_value, 0.0);
  PointJet2 cos_theta = constant(cos_theta_value);
  cos_theta.first[Theta] = Complex(-sin_theta_value, 0.0);
  cos_theta.second[Theta][Theta] = Complex(-cos_theta_value, 0.0);

  const double mass = parameters.mass;
  const double spin = parameters.spin;
  const double length = parameters.compactification_length;
  const double length2 = length * length;
  const double length4 = length2 * length2;
  const Complex imaginary_unit(0.0, 1.0);
  const double sqrt_two = Kokkos::sqrt(2.0);
  const PointJet2 radius2 = radius * radius;
  const PointJet2 denominator =
      length4 + spin * spin * radius2 * cos_theta * cos_theta;
  const PointJet2 radial_polynomial =
      length2 - 2.0 * mass * radius +
      spin * spin * radius2 / length2;
  const PointJet2 dm = length2 - imaginary_unit * spin * radius * cos_theta;
  const PointJet2 dp = length2 + imaginary_unit * spin * radius * cos_theta;

  GeometryJets result;
  result.l_T = radius2 / denominator *
               (2.0 * mass *
                (2.0 * mass - spin * spin * radius / length2));
  result.l_R = -0.5 * radius2 / denominator * radial_polynomial;
  result.l_phi = spin * radius2 / denominator;
  const PointJet2 angular_prefactor = radius / (sqrt_two * dm);
  result.m_T = -imaginary_unit * spin * sin_theta * angular_prefactor;
  result.m_theta = -angular_prefactor;
  result.m_phi = -imaginary_unit * angular_prefactor / sin_theta;

  result.rho =
      -0.5 * radius *
      (spin * spin * radius2 + length4 - 2.0 * length2 * mass * radius) /
      (dm * dm * dp);
  result.epsilon =
      0.5 * radius2 *
      (length2 * mass - spin * spin * radius -
       imaginary_unit * spin * (length2 - mass * radius) * cos_theta) /
      (dm * dm * dp);
  result.alpha = radius * cos_theta /
                 (2.0 * sqrt_two * sin_theta * dp);
  result.beta =
      radius *
      (-length2 * cos_theta / sin_theta +
       imaginary_unit * spin * radius * sin_theta *
           (1.0 / (sin_theta * sin_theta) + 1.0)) /
      (2.0 * sqrt_two * dm * dm);
  result.tau = imaginary_unit * spin * radius2 * sin_theta /
               (sqrt_two * dm * dm);
  result.pi = -imaginary_unit * spin * radius2 * sin_theta /
              (sqrt_two * denominator);
  return result;
}

KOKKOS_INLINE_FUNCTION FirstJet directional_D(
    const GeometryJets& geometry, const PointJet2& field) {
  FirstJet result;
  result.value = geometry.l_T.value * field.first[Time] +
                 geometry.l_R.value * field.first[Radius] +
                 geometry.l_phi.value * field.first[Phi];
  for (int derivative = 0; derivative < Count; ++derivative) {
    result.first[derivative] =
        geometry.l_T.first[derivative] * field.first[Time] +
        geometry.l_T.value * field.second[derivative][Time] +
        geometry.l_R.first[derivative] * field.first[Radius] +
        geometry.l_R.value * field.second[derivative][Radius] +
        geometry.l_phi.first[derivative] * field.first[Phi] +
        geometry.l_phi.value * field.second[derivative][Phi];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION FirstJet directional_delta(
    const GeometryJets& geometry, const PointJet2& field) {
  FirstJet result;
  result.value = geometry.m_T.value * field.first[Time] +
                 geometry.m_theta.value * field.first[Theta] +
                 geometry.m_phi.value * field.first[Phi];
  for (int derivative = 0; derivative < Count; ++derivative) {
    result.first[derivative] =
        geometry.m_T.first[derivative] * field.first[Time] +
        geometry.m_T.value * field.second[derivative][Time] +
        geometry.m_theta.first[derivative] * field.first[Theta] +
        geometry.m_theta.value * field.second[derivative][Theta] +
        geometry.m_phi.first[derivative] * field.first[Phi] +
        geometry.m_phi.value * field.second[derivative][Phi];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION Complex directional_D_value(
    const GeometryJets& geometry, const FirstJet& field) {
  return geometry.l_T.value * field.first[Time] +
         geometry.l_R.value * field.first[Radius] +
         geometry.l_phi.value * field.first[Phi];
}

KOKKOS_INLINE_FUNCTION Complex directional_delta_value(
    const GeometryJets& geometry, const FirstJet& field) {
  return geometry.m_T.value * field.first[Time] +
         geometry.m_theta.value * field.first[Theta] +
         geometry.m_phi.value * field.first[Phi];
}

}  // namespace plus2_linear_detail

KOKKOS_INLINE_FUNCTION Plus2LinearBackgroundPoint
plus2_linear_background_point(const KerrParameters& parameters,
                              const double radius, const double sin_theta,
                              const double cos_theta) {
  const auto geometry = plus2_linear_detail::geometry_jets(
      parameters, radius, sin_theta, cos_theta);
  return {geometry.rho.value, geometry.epsilon.value, geometry.alpha.value,
          geometry.beta.value, geometry.tau.value, geometry.pi.value};
}

struct Plus2LinearPsi0Point {
  Complex sigma1{};
  Complex sigma1_T{};
  Complex kappa1{};
  Complex kappa1_T{};
  Complex psi0_code_tetrad{};
};

// Loutrel et al. arXiv:2008.11770 Eqs. (C1c,C1e), followed by the
// corrected outer identity obtained from its Eq. (A9b) and independently
// printed in Campanelli--Lousto arXiv:gr-qc/9811019 Eq. (A5):
//
// Psi0 = (D-rho-rhobar-3 epsilon+epsilonbar) sigma1
//        -(delta-alphabar-3 beta+pibar-tau) kappa1.
KOKKOS_INLINE_FUNCTION Plus2LinearPsi0Point evaluate_plus2_linear_psi0(
    const KerrParameters& parameters, const double radius,
    const double sin_theta, const double cos_theta,
    const Plus2OrgMetricStage& stage,
    const Plus2OrgMetricDerivativeSlots& derivatives) {
  using namespace plus2_linear_detail;
  const GeometryJets geometry =
      geometry_jets(parameters, radius, sin_theta, cos_theta);
  const PointJet2 h_ll = metric_component(
      stage.h, stage.h_T, stage.h_TT, derivatives.h_R, derivatives.h_TR,
      derivatives.h_RR, derivatives.h_theta, derivatives.h_Ttheta,
      derivatives.h_Rtheta, derivatives.h_thetatheta, derivatives.h_phi,
      derivatives.h_Tphi, derivatives.h_Rphi, derivatives.h_thetaphi,
      derivatives.h_phiphi, 0);
  const PointJet2 h_lm = metric_component(
      stage.h, stage.h_T, stage.h_TT, derivatives.h_R, derivatives.h_TR,
      derivatives.h_RR, derivatives.h_theta, derivatives.h_Ttheta,
      derivatives.h_Rtheta, derivatives.h_thetatheta, derivatives.h_phi,
      derivatives.h_Tphi, derivatives.h_Rphi, derivatives.h_thetaphi,
      derivatives.h_phiphi, 1);
  const PointJet2 h_mm = metric_component(
      stage.h, stage.h_T, stage.h_TT, derivatives.h_R, derivatives.h_TR,
      derivatives.h_RR, derivatives.h_theta, derivatives.h_Ttheta,
      derivatives.h_Rtheta, derivatives.h_thetatheta, derivatives.h_phi,
      derivatives.h_Tphi, derivatives.h_Rphi, derivatives.h_thetaphi,
      derivatives.h_phiphi, 2);

  const PointJet2 rho_bar = jet_conj(geometry.rho);
  const PointJet2 epsilon_bar = jet_conj(geometry.epsilon);
  const PointJet2 alpha_bar = jet_conj(geometry.alpha);
  const PointJet2 pi_bar = jet_conj(geometry.pi);
  const FirstJet sigma1 =
      0.5 * (directional_D(geometry, h_mm) +
             first_product(2.0 * (epsilon_bar - geometry.epsilon) +
                               geometry.rho - rho_bar,
                           h_mm)) -
      first_product(geometry.tau + pi_bar, h_lm);
  const FirstJet kappa1 =
      directional_D(geometry, h_lm) -
      first_product(2.0 * geometry.epsilon + rho_bar, h_lm) -
      0.5 * (directional_delta(geometry, h_ll) +
             first_product(-2.0 * alpha_bar - 2.0 * geometry.beta +
                               pi_bar + geometry.tau,
                           h_ll));

  const Complex psi0 =
      directional_D_value(geometry, sigma1) +
      (-geometry.rho.value - rho_bar.value -
       3.0 * geometry.epsilon.value + epsilon_bar.value) *
          sigma1.value -
      directional_delta_value(geometry, kappa1) -
      (-alpha_bar.value - 3.0 * geometry.beta.value + pi_bar.value -
       geometry.tau.value) *
          kappa1.value;
  return {sigma1.value, sigma1.first[Time], kappa1.value,
          kappa1.first[Time], psi0};
}

struct Plus2RegularizedPsi0Point {
  Complex z_plus{};
  bool valid = false;
};

// Cancellation-safe regularization.  At R=0 the raw result is identically
// zero for a peeling solution and cannot be divided by W_plus.  The caller
// must explicitly provide the independently evaluated coefficient
// lim_{R->0} Psi0/R^5.  Missing data fail closed with valid=false.
KOKKOS_INLINE_FUNCTION Plus2RegularizedPsi0Point
regularize_plus2_linear_psi0(
    const Complex& psi0_code_tetrad, const double radius,
    const double cos_theta, const double spin,
    const double compactification_length,
    const bool has_scri_psi0_over_radius5 = false,
    const Complex& scri_psi0_over_radius5 = Complex()) {
  if (radius < 0.0 || compactification_length <= 0.0) return {};
  if (radius == 0.0) {
    if (!has_scri_psi0_over_radius5) return {};
    const double length2 =
        compactification_length * compactification_length;
    const double length4 = length2 * length2;
    return {length4 * length4 * scri_psi0_over_radius5, true};
  }
  const Complex scaling = plus2_code_tetrad_scaling(
      radius, cos_theta, spin, compactification_length);
  return {psi0_code_tetrad / scaling, true};
}

}  // namespace teuk
