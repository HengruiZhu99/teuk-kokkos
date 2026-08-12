#pragma once

#include <Kokkos_Core.hpp>

#include "teuk/background.hpp"

namespace teuk {

// Exact first-order vacuum Bianchi closures in the repository's ORG and
// rotated-Kinnersley tetrad.  Scalar may be Complex or Jet1<Complex>; the
// latter gives F and F_T without a separate hand-written tangent formula.
//
// Loutrel et al. arXiv:2008.11770, bianchi-5/6 and delta-1:
//   Delta_5 Z0 = eth_4 Z1 - R mu0 Z0 - 4 R tau0 Z1 + 3 Sig H,
//   Delta_4 Z1 = eth_3 H + R[-2 mu0 Z1 - 3 tau0 H
//                    - Csharp Delta_3 psi20
//                    + 1/2 Bsharp ethprime_3 psi20 - 3 Ta psi20].
// The metric-term signs are fixed by
// delta^(1)=-h_lm Delta+(1/2)h_mm bardelta in ORG.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar plus2_bianchi_delta4_z1(
    const double radius, const KerrBackgroundPoint& background,
    const Scalar& z1, const Scalar& h, const Scalar& c_sharp,
    const Scalar& b_sharp, const Scalar& tau1,
    const Scalar& eth3_h, const Complex& delta3_psi20,
    const Complex& ethprime3_psi20) {
  return eth3_h +
         radius * (-2.0 * background.mu0 * z1 -
                   3.0 * background.tau0 * h -
                   c_sharp * delta3_psi20 +
                   0.5 * b_sharp * ethprime3_psi20 -
                   3.0 * tau1 * background.psi20);
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar plus2_bianchi_delta5_z0(
    const double radius, const KerrBackgroundPoint& background,
    const Scalar& z0, const Scalar& z1, const Scalar& sigma1,
    const Scalar& h, const Scalar& eth4_z1) {
  return eth4_z1 - radius * background.mu0 * z0 -
         4.0 * radius * background.tau0 * z1 + 3.0 * sigma1 * h;
}

// Invert the exact repository Delta_n point formula.  This is regular at
// scri: A=2 and the radial coefficient vanishes at R=0.  M>0 and L>0 imply
// A>0 throughout the compact exterior.
template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar plus2_invert_capital_delta_n(
    const Scalar& capital_delta_n, const Scalar& value,
    const Scalar& radial_derivative, const int radial_falloff,
    const double radius, const double mass,
    const double compactification_length) {
  const double length2 =
      compactification_length * compactification_length;
  const double time_coefficient = 2.0 + 4.0 * mass * radius / length2;
  const double radial_coefficient = radius / length2;
  return (capital_delta_n -
          radial_coefficient *
              (radius * radial_derivative + radial_falloff * value)) /
         time_coefficient;
}

// Explicit endpoint contract for the two peeling cancellations used by the
// local metric-curvature Z0/Z1 route.  Interior points use numerator/R^power.
// At R=0 the caller must supply the independently qualified limit.  A later
// spatial gate may provide one-sided fourth-order coefficients; this helper
// deliberately does not extrapolate or substitute zero.
template <class Scalar>
struct Plus2PeelingQuotientResultT {
  Scalar value{};
  bool valid = false;
};

template <class Scalar>
KOKKOS_INLINE_FUNCTION Plus2PeelingQuotientResultT<Scalar>
plus2_peeling_quotient(const Scalar& numerator, const int radial_power,
                       const double radius,
                       const bool has_scri_coefficient = false,
                       const Scalar& scri_coefficient = Scalar{}) {
  if (!Kokkos::isfinite(radius) || radius < 0.0 || radial_power <= 0) {
    return {};
  }
  if (radius == 0.0) {
    if (!has_scri_coefficient) return {};
    return {scri_coefficient, true};
  }
  double radius_power = 1.0;
  for (int exponent = 0; exponent < radial_power; ++exponent) {
    radius_power *= radius;
  }
  return {numerator / radius_power, true};
}

}  // namespace teuk
