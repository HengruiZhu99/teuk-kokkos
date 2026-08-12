#pragma once

#include <Kokkos_Core.hpp>

#include "teuk/background.hpp"

namespace teuk {

// Background angular coefficients not currently stored in
// KerrBackgroundPoint.  Physical alpha=R*alpha0 and beta=R*beta0.  These are
// the rotated-Kinnersley coefficients in Ripley et al., arXiv:2010.00162,
// Eq. (NP_IEF_HC); their separate pole singularities cancel in weighted
// angular combinations and must not be evaluated at sin(theta)=0 in isolation.
struct Plus2PrimitiveBackground {
  KerrBackgroundPoint kerr;
  Complex alpha0;
  Complex beta0;
};

KOKKOS_INLINE_FUNCTION Plus2PrimitiveBackground
plus2_primitive_background(const KerrParameters& parameters,
                           const double radius, const double cos_theta,
                           const double sin_theta) {
  const double length2 = parameters.compactification_length *
                         parameters.compactification_length;
  const double sqrt_two = Kokkos::sqrt(2.0);
  const Complex imaginary_unit(0.0, 1.0);
  const Complex minus_denominator(length2,
                                  -parameters.spin * radius * cos_theta);
  const Complex plus_denominator(length2,
                                 parameters.spin * radius * cos_theta);
  const Complex alpha0 =
      cos_theta /
      (2.0 * sqrt_two * sin_theta * plus_denominator);
  const Complex beta0 =
      (-length2 * cos_theta / sin_theta +
       imaginary_unit * parameters.spin * radius * sin_theta *
           (1.0 / (sin_theta * sin_theta) + 1.0)) /
      (2.0 * sqrt_two * minus_denominator * minus_denominator);
  return {kerr_background_point(parameters, radius, cos_theta, sin_theta),
          alpha0, beta0};
}

// Existing reconstructed ORG fields for one signed mode.  Sharp values must
// already be selected as X_m^sharp=conj(X_-m), never conj(X_m).
// Their physical metric components are h_ll=R^2 U/mu0, h_lbar=R^2 C,
// h_lm=R^2 Csharp, h_barbar=R B, and h_mm=R Bsharp.  H=Psi2/R^3 and
// Pi=pi^(1)/R^2 are existing reconstruction outputs.
template <class Scalar>
struct Plus2OrgMetricFieldsT {
  Scalar U;
  Scalar Usharp;
  Scalar C;
  Scalar Csharp;
  Scalar B;
  Scalar Bsharp;
  Scalar H;
  Scalar Pi;
};

// Every operator required by the regular primitive formulas is explicit.
// capital_delta_n X=R^(-n) Delta(R^n X), while thorn_n, eth_n, and
// ethprime_n use R^(-n-1) op(R^n X), in the repository's GHP convention.
// The connection formulas are the ordinary-NP expressions in Appendix B of
// Ripley et al., arXiv:2010.00162, converted to these regular operators.
// The two quotient fields are separately named
// because their numerators cancel only for peeling-compatible reconstructed
// data.  A future spatial layer must compute these quotients with a reviewed
// endpoint operator; this point evaluator does not divide by R.
template <class Scalar>
struct Plus2PrimitiveDerivativesT {
  Scalar thorn1_Bsharp;
  Scalar thorn2_Csharp;
  Scalar eth2_V;
  Scalar ethprime2_Csharp;
  Scalar eth2_C;
  Scalar capital_delta2_Csharp;
  Scalar capital_delta2_C;
  Scalar capital_delta2_V;
  Scalar capital_delta2_Vsharp;
  Scalar eth1_B;
  Scalar ethprime1_Bsharp;

  Scalar thorn2_Sig;
  Scalar eth3_Kap;
  Scalar psi0_leading_combination_over_r2;

  Scalar thorn2_Be;
  Scalar eth2_Ep;
  Scalar capital_delta1_beta0;
  Scalar capital_delta2_epsilon0;
  Scalar bardelta2_epsilon0;
  Scalar psi1_leading_combination_over_r;
};

template <class Scalar>
struct Plus2SourcePrimitivesT {
  // The fourteen manifest rows, in manifest order.  The physical fields are
  // Psi0=R^5 Z0, Psi1=R^4 Z1, Psi2=R^3 H, sigma=R^2 Sig,
  // kappa=R^3 Kap, rho1=R^3 Rh, and tau1/alpha1/beta1/epsilon1/pi1
  // equal R^2 times Ta/Al/Be/Ep/Pi respectively.  V,C,B are the three
  // rescaled metric components described above.
  Scalar Z0;
  Scalar Z1;
  Scalar H;
  Scalar Sig;
  Scalar Kap;
  Scalar Rh;
  Scalar Ta;
  Scalar Al;
  Scalar Be;
  Scalar Ep;
  Scalar Pi;
  Scalar V;
  Scalar C;
  Scalar B;

  // Sharp helpers required by the compact source and curvature construction.
  Scalar Rhsharp;
  Scalar Alsharp;
  Scalar Epsharp;
  Scalar Pisharp;

  // Audit outputs.  For consistent reconstructed data Pi_input_residual=0.
  // Peeling requires psi0_leading=R^2*q0 and psi1_leading=R*q1, where q0/q1
  // are the two explicit quotient inputs.
  Scalar Pi_input_residual;
  Scalar psi0_leading;
  Scalar psi0_quotient_residual;
  Scalar psi1_leading;
  Scalar psi1_quotient_residual;
};

using Plus2OrgMetricFields = Plus2OrgMetricFieldsT<Complex>;
using Plus2PrimitiveDerivatives = Plus2PrimitiveDerivativesT<Complex>;
using Plus2SourcePrimitives = Plus2SourcePrimitivesT<Complex>;

template <class Scalar>
KOKKOS_INLINE_FUNCTION Plus2SourcePrimitivesT<Scalar>
plus2_source_primitives(
    const double radius, const Plus2PrimitiveBackground& background,
    const Plus2OrgMetricFieldsT<Scalar>& metric,
    const Plus2PrimitiveDerivativesT<Scalar>& d) {
  const Complex mu0 = background.kerr.mu0;
  const Complex mubar0 = Kokkos::conj(mu0);
  const Complex rho0 = background.kerr.rho0;
  const Complex rhobar0 = Kokkos::conj(rho0);
  const Complex epsilon0 = background.kerr.epsilon0;
  const Complex pibar0 = Kokkos::conj(background.kerr.pi0);
  const Complex taubar0 = Kokkos::conj(background.kerr.tau0);
  const Complex betabar0 = Kokkos::conj(background.beta0);
  const double radius2 = radius * radius;

  const Scalar V = metric.U / mu0;
  const Scalar Vsharp = metric.Usharp / mubar0;

  const Scalar Sig =
      0.5 * d.thorn1_Bsharp + 0.5 * (rho0 - rhobar0) * metric.Bsharp -
      radius2 * (pibar0 + background.kerr.tau0) * metric.Csharp;
  const Scalar Kap =
      d.thorn2_Csharp - rhobar0 * metric.Csharp - 0.5 * d.eth2_V -
      0.5 * radius * (pibar0 + background.kerr.tau0) * V;
  const Scalar Rh =
      0.5 * metric.U + 0.5 * d.ethprime2_Csharp - 0.5 * d.eth2_C -
      0.5 * radius * background.kerr.pi0 * metric.Csharp -
      0.5 * radius * (pibar0 + 2.0 * background.kerr.tau0) * metric.C;
  const Scalar Rhsharp =
      0.5 * metric.Usharp + 0.5 * d.eth2_C -
      0.5 * d.ethprime2_Csharp - 0.5 * radius * pibar0 * metric.C -
      0.5 * radius *
          (background.kerr.pi0 + 2.0 * taubar0) * metric.Csharp;
  const Scalar Ta =
      0.5 * d.capital_delta2_Csharp +
      0.5 * radius * mu0 * metric.Csharp -
      0.5 * radius * background.kerr.pi0 * metric.Bsharp;
  const Scalar Pi_from_metric =
      -0.5 * d.capital_delta2_C -
      0.5 * radius * mubar0 * metric.C -
      0.5 * radius * background.kerr.tau0 * metric.B;
  const Scalar Pisharp =
      -0.5 * d.capital_delta2_Csharp -
      0.5 * radius * mu0 * metric.Csharp -
      0.5 * radius * taubar0 * metric.Bsharp;

  const Scalar Al =
      -0.25 * d.capital_delta2_C +
      0.25 * radius * (2.0 * mu0 - mubar0) * metric.C -
      0.25 * d.eth1_B + 0.5 * background.beta0 * metric.B -
      0.25 * radius * (pibar0 + background.kerr.tau0) * metric.B;
  const Scalar Alsharp =
      -0.25 * d.capital_delta2_Csharp +
      0.25 * radius * (2.0 * mubar0 - mu0) * metric.Csharp -
      0.25 * d.ethprime1_Bsharp + 0.5 * betabar0 * metric.Bsharp -
      0.25 * radius *
          (background.kerr.pi0 + taubar0) * metric.Bsharp;
  const Scalar Be =
      -0.25 * d.capital_delta2_Csharp -
      0.25 * radius * (mu0 + 2.0 * mubar0) * metric.Csharp +
      0.25 * d.ethprime1_Bsharp + 0.5 * background.alpha0 * metric.Bsharp -
      0.25 * radius *
          (background.kerr.pi0 + taubar0) * metric.Bsharp;
  const Scalar Ep =
      -0.25 * d.capital_delta2_V - 0.25 * radius * d.eth2_C +
      0.25 * radius * d.ethprime2_Csharp +
      0.25 * radius * (mu0 - mubar0) * V -
      0.25 * radius2 *
          (pibar0 + 2.0 * background.kerr.tau0) * metric.C -
      0.25 * radius2 *
          (3.0 * background.kerr.pi0 + 2.0 * taubar0) * metric.Csharp;
  const Scalar Epsharp =
      -0.25 * d.capital_delta2_Vsharp +
      0.25 * radius * d.eth2_C -
      0.25 * radius * d.ethprime2_Csharp +
      0.25 * radius * (mubar0 - mu0) * Vsharp -
      0.25 * radius2 *
          (background.kerr.pi0 + 2.0 * taubar0) * metric.Csharp -
      0.25 * radius2 *
          (3.0 * pibar0 + 2.0 * background.kerr.tau0) * metric.C;

  // Corrected Psi0=(thorn-rho-rhobar)sigma-(eth+pibar-tau)kappa.
  // With sigma=R^2 Sig and kappa=R^3 Kap, the first two pieces form the
  // peeling numerator below; its quotient by R^2 is an explicit input.
  const Scalar psi0_leading =
      d.thorn2_Sig - (rho0 + rhobar0) * Sig - radius * d.eth3_Kap;
  const Scalar Z0 =
      d.psi0_leading_combination_over_r2 -
      (pibar0 - background.kerr.tau0) * Kap;

  // Linearization of
  // (D-rhobar+epsilonbar)beta-(delta-alphabar+pibar)epsilon
  // -(alpha+pi)sigma+mu*kappa in the gamma=gamma1=0 ORG tetrad.
  const Scalar psi1_leading =
      d.thorn2_Be - rhobar0 * Be - d.eth2_Ep -
      background.beta0 * Ep - background.alpha0 * Sig -
      0.5 * V * d.capital_delta1_beta0 + Epsharp * background.beta0;
  const Scalar psi1_order4 =
      epsilon0 * Be - pibar0 * Ep - background.kerr.pi0 * Sig + mu0 * Kap +
      metric.Csharp * d.capital_delta2_epsilon0 -
      0.5 * metric.Bsharp * d.bardelta2_epsilon0 -
      Rhsharp * background.beta0 + Alsharp * epsilon0 - Pisharp * epsilon0;
  const Scalar Z1 = d.psi1_leading_combination_over_r + psi1_order4;

  return {Z0,
          Z1,
          metric.H,
          Sig,
          Kap,
          Rh,
          Ta,
          Al,
          Be,
          Ep,
          metric.Pi,
          V,
          metric.C,
          metric.B,
          Rhsharp,
          Alsharp,
          Epsharp,
          Pisharp,
          metric.Pi - Pi_from_metric,
          psi0_leading,
          psi0_leading - radius2 * d.psi0_leading_combination_over_r2,
          psi1_leading,
          psi1_leading - radius * d.psi1_leading_combination_over_r};
}

}  // namespace teuk
