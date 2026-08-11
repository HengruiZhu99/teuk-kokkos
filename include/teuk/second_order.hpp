#pragma once

#include <Kokkos_Complex.hpp>

#include <complex>

#include "teuk/background.hpp"

namespace teuk {

template <class ComplexType>
struct OrderedPairFieldsT {
  // First factor, carrying m1.
  ComplexType F1;
  ComplexType G1;
  ComplexType Lambda1;
  ComplexType Pi1;
  ComplexType B1;
  ComplexType C1;
  ComplexType U1;

  // Second factor, carrying m2.
  ComplexType F2;
  ComplexType G2;
  ComplexType H2;
  ComplexType B2;
  ComplexType C2;
  ComplexType U2;

  // X_m^sharp = conjugate(X_{-m}); these are explicit to prevent the common
  // but incorrect replacement by conjugate(X_m).
  ComplexType U2_sharp;
  ComplexType C2_sharp;
  ComplexType C1_sharp;
  ComplexType B1_sharp;
  ComplexType Pi2_sharp;
  ComplexType B2_sharp;
};

template <class ComplexType>
struct OrderedPairDerivativesT {
  ComplexType delta1_F2;
  ComplexType delta3_U2;
  ComplexType eth2_C2;
  ComplexType ethprime2_C2_sharp;
  ComplexType eth1_B2;
  ComplexType delta2_C2;
  ComplexType delta2_G2;
  ComplexType eth2_G2;

  ComplexType ethprime1_F2;
  ComplexType delta2_C2_sharp;
  ComplexType ethprime1_B2_sharp;
};

template <class ComplexType>
struct InnerSourceT {
  ComplexType D;
  ComplexType T;
};

using OrderedPairFields = OrderedPairFieldsT<Complex>;
using OrderedPairDerivatives = OrderedPairDerivativesT<Complex>;
using InnerSource = InnerSourceT<Complex>;

using ReferenceOrderedPairFields = OrderedPairFieldsT<std::complex<double>>;
using ReferenceOrderedPairDerivatives =
    OrderedPairDerivativesT<std::complex<double>>;
using ReferenceInnerSource = InnerSourceT<std::complex<double>>;

// Correct compact source for one ordered pair (m1,m2)->m1+m2. The caller is
// responsible for deterministic enumeration and summing both orderings.
KOKKOS_INLINE_FUNCTION
InnerSource corrected_ordered_pair_source(
    const double radius, const KerrBackgroundPoint& background,
    const OrderedPairFields& fields,
    const OrderedPairDerivatives& derivatives) {
  const Complex mu0_bar = Kokkos::conj(background.mu0);
  const Complex pi0_bar = Kokkos::conj(background.pi0);
  const Complex tau0_bar = Kokkos::conj(background.tau0);
  const double radius2 = radius * radius;

  // D terms sd01--sd15 in SOURCE_TERM_LEDGER.csv.
  const Complex sd01_sd02 =
      0.5 * fields.U1 *
      (derivatives.delta1_F2 / background.mu0 + radius * fields.F2);

  const Complex sd03 =
      fields.F1 * (0.5 * radius * derivatives.eth2_C2);
  // Critical correction: 0.5 multiplies the connection term as well as eth.
  const Complex sd04 =
      fields.F1 *
      (0.5 * radius2 * (pi0_bar + 2.0 * background.tau0) * fields.C2);
  const Complex sd05 =
      fields.F1 *
      (derivatives.delta3_U2 / background.mu0 +
       radius * fields.U2_sharp);
  const Complex sd06 =
      fields.F1 * (-0.5 * radius * derivatives.ethprime2_C2_sharp);
  const Complex sd07 =
      fields.F1 *
      (0.5 * radius2 * (5.0 * background.pi0 + 4.0 * tau0_bar) *
       fields.C2_sharp);

  const Complex sd08 =
      -0.5 * fields.G1 * (radius * derivatives.eth1_B2);
  const Complex sd09 =
      -0.5 * fields.G1 *
      (radius2 * (pi0_bar + background.tau0) * fields.B2);
  const Complex sd10 =
      -0.5 * fields.G1 * (radius * derivatives.delta2_C2);
  const Complex sd11 =
      -0.5 * fields.G1 *
      (radius2 * (-2.0 * background.mu0 + mu0_bar) * fields.C2);
  const Complex sd12 =
      -radius * fields.C1 * derivatives.delta2_G2;
  const Complex sd13 =
      0.5 * radius * fields.B1 * derivatives.eth2_G2;
  const Complex sd14 = 4.0 * radius * fields.Pi1 * fields.G2;
  const Complex sd15 = -3.0 * radius * fields.Lambda1 * fields.H2;

  // T terms st01--st07 in SOURCE_TERM_LEDGER.csv.
  const Complex st01_st02 =
      -fields.C1_sharp *
      (derivatives.delta1_F2 +
       radius * (background.mu0 + 2.0 * mu0_bar) * fields.F2);
  const Complex st03 =
      0.5 * fields.B1_sharp * derivatives.ethprime1_F2;
  const Complex st04 = fields.F1 * fields.Pi2_sharp;
  const Complex st05 = -fields.F1 * derivatives.delta2_C2_sharp;
  const Complex st06 = fields.F1 * derivatives.ethprime1_B2_sharp;
  const Complex st07 =
      fields.F1 *
      (-0.5 * radius * (background.pi0 + tau0_bar) * fields.B2_sharp);

  InnerSource source;
  source.D = sd01_sd02 + sd03 + sd04 + sd05 + sd06 + sd07 + sd08 +
             sd09 + sd10 + sd11 + sd12 + sd13 + sd14 + sd15;
  source.T = st01_st02 + st03 + st04 + st05 + st06 + st07;
  return source;
}

struct OuterSourceDerivatives {
  Complex delta3_D;
  Complex ethprime3_T;
};

KOKKOS_INLINE_FUNCTION
Complex outer_source_over_r3(const double radius,
                             const KerrBackgroundPoint& background,
                             const InnerSource& summed_inner_source,
                             const OuterSourceDerivatives& derivatives) {
  const Complex s01 = derivatives.delta3_D;
  const Complex s02 =
      radius * (4.0 * background.mu0 + Kokkos::conj(background.mu0)) *
      summed_inner_source.D;
  const Complex s03 = radius * derivatives.ethprime3_T;
  const Complex s04 =
      radius * radius *
      (4.0 * background.pi0 - Kokkos::conj(background.tau0)) *
      summed_inner_source.T;
  return s01 + s02 + s03 + s04;
}

KOKKOS_INLINE_FUNCTION
Complex coordinate_second_order_forcing(const double radius,
                                         const double cos_theta,
                                         const double kerr_spin,
                                         const double compactification_length,
                                         const Complex& source_over_r3) {
  const double length2 =
      compactification_length * compactification_length;
  const double length4 = length2 * length2;
  const double normalization =
      2.0 * (length4 + kerr_spin * kerr_spin * radius * radius * cos_theta *
                           cos_theta);
  return normalization * source_over_r3;
}

// Host-only scalar oracle. This intentionally repeats the complete expression
// with std::complex so tests can compare the device-compatible path against a
// transparent, debugger-friendly reference calculation.
inline ReferenceInnerSource corrected_ordered_pair_source_reference(
    const double radius, const std::complex<double>& mu0,
    const std::complex<double>& tau0, const std::complex<double>& pi0,
    const ReferenceOrderedPairFields& f,
    const ReferenceOrderedPairDerivatives& d) {
  const std::complex<double> mu0_bar = std::conj(mu0);
  const std::complex<double> pi0_bar = std::conj(pi0);
  const std::complex<double> tau0_bar = std::conj(tau0);
  const double radius2 = radius * radius;

  const std::complex<double> sd01_sd02 =
      0.5 * f.U1 * (d.delta1_F2 / mu0 + radius * f.F2);
  const std::complex<double> sd03 = f.F1 * (0.5 * radius * d.eth2_C2);
  const std::complex<double> sd04 =
      f.F1 * (0.5 * radius2 * (pi0_bar + 2.0 * tau0) * f.C2);
  const std::complex<double> sd05 =
      f.F1 * (d.delta3_U2 / mu0 + radius * f.U2_sharp);
  const std::complex<double> sd06 =
      f.F1 * (-0.5 * radius * d.ethprime2_C2_sharp);
  const std::complex<double> sd07 =
      f.F1 *
      (0.5 * radius2 * (5.0 * pi0 + 4.0 * tau0_bar) * f.C2_sharp);
  const std::complex<double> sd08 =
      -0.5 * f.G1 * (radius * d.eth1_B2);
  const std::complex<double> sd09 =
      -0.5 * f.G1 * (radius2 * (pi0_bar + tau0) * f.B2);
  const std::complex<double> sd10 =
      -0.5 * f.G1 * (radius * d.delta2_C2);
  const std::complex<double> sd11 =
      -0.5 * f.G1 * (radius2 * (-2.0 * mu0 + mu0_bar) * f.C2);
  const std::complex<double> sd12 = -radius * f.C1 * d.delta2_G2;
  const std::complex<double> sd13 = 0.5 * radius * f.B1 * d.eth2_G2;
  const std::complex<double> sd14 = 4.0 * radius * f.Pi1 * f.G2;
  const std::complex<double> sd15 = -3.0 * radius * f.Lambda1 * f.H2;

  const std::complex<double> st01_st02 =
      -f.C1_sharp *
      (d.delta1_F2 + radius * (mu0 + 2.0 * mu0_bar) * f.F2);
  const std::complex<double> st03 =
      0.5 * f.B1_sharp * d.ethprime1_F2;
  const std::complex<double> st04 = f.F1 * f.Pi2_sharp;
  const std::complex<double> st05 = -f.F1 * d.delta2_C2_sharp;
  const std::complex<double> st06 = f.F1 * d.ethprime1_B2_sharp;
  const std::complex<double> st07 =
      f.F1 * (-0.5 * radius * (pi0 + tau0_bar) * f.B2_sharp);

  return {sd01_sd02 + sd03 + sd04 + sd05 + sd06 + sd07 + sd08 + sd09 +
              sd10 + sd11 + sd12 + sd13 + sd14 + sd15,
          st01_st02 + st03 + st04 + st05 + st06 + st07};
}

}  // namespace teuk
