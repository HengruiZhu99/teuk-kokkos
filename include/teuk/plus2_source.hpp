#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <stdexcept>

#include "teuk/background.hpp"
#include "teuk/modes.hpp"

namespace teuk {

inline constexpr std::size_t plus2_compact_source_ledger_term_count = 51;

// Standalone compact raw fixed-tetrad spin +2 source.  The cancellation-safe
// constructions of Z0=Psi0/R^5 and Z1=Psi1/R^4 are deliberately outside this
// boundary: callers must provide those peeling-regular primitives and their
// derivatives explicitly.  Scalar may be Complex or Jet1<Complex>.
template <class Scalar>
struct Plus2OrderedPairFieldsT {
  // First factor, carrying signed mode m1.
  Scalar V1;
  Scalar C1;
  Scalar Csharp1;
  Scalar B1;
  Scalar Bsharp1;
  Scalar Sig1;
  Scalar Kap1;
  Scalar Rh1;
  Scalar Rhsharp1;
  Scalar Ep1;
  Scalar Epsharp1;

  // Second factor, carrying signed mode m2.
  Scalar Z0_2;
  Scalar Z1_2;
  Scalar H2;
  Scalar Sig2;
  Scalar Kap2;
};

template <class Scalar>
struct Plus2OrderedPairDerivativesT {
  Scalar capital_delta4_Z1_2;
  Scalar ethprime4_Z1_2;
  Scalar capital_delta2_Csharp1;
  Scalar ethprime1_Bsharp1;

  Scalar capital_delta5_Z0_2;
  Scalar eth5_Z0_2;
  Scalar capital_delta2_C1;
  Scalar eth1_B1;

  Scalar capital_delta2_V1;
  Scalar eth2_C1;
  Scalar ethprime2_Csharp1;

  Scalar capital_delta2_Sig2;
  Scalar capital_delta3_Kap2;
  Scalar ethprime3_Kap2;
};

// Production forcing values need analytic tangents only through J and K.
// Keep their derivative contract separate from the value-only Q contract so
// callers are never forced to manufacture Delta(Sigma)_T or Delta(Kappa)_T.
template <class Scalar>
struct Plus2OrderedPairJkDerivativesT {
  Scalar capital_delta4_Z1_2;
  Scalar ethprime4_Z1_2;
  Scalar capital_delta2_Csharp1;
  Scalar ethprime1_Bsharp1;
  Scalar capital_delta5_Z0_2;
  Scalar eth5_Z0_2;
  Scalar capital_delta2_C1;
  Scalar eth1_B1;
  Scalar capital_delta2_V1;
  Scalar eth2_C1;
  Scalar ethprime2_Csharp1;
};

template <class Scalar>
struct Plus2OrderedPairQDerivativesT {
  Scalar ethprime1_Bsharp1;
  Scalar capital_delta2_Sig2;
  Scalar capital_delta3_Kap2;
  Scalar ethprime3_Kap2;
};

template <class Scalar>
struct Plus2C12TermsT {
  Scalar c01, c02, c03, c04, c05, c06, c07, c08;
  KOKKOS_INLINE_FUNCTION Scalar total() const {
    return c01 + c02 + c03 + c04 + c05 + c06 + c07 + c08;
  }
};

template <class Scalar>
struct Plus2B12TermsT {
  Scalar b01, b02, b03, b04, b05, b06, b07, b08;
  KOKKOS_INLINE_FUNCTION Scalar total() const {
    return b01 + b02 + b03 + b04 + b05 + b06 + b07 + b08;
  }
};

template <class Scalar>
struct Plus2D12TermsT {
  Scalar d01, d02, d03, d04, d05, d06, d07, d08, d09, d10;
  KOKKOS_INLINE_FUNCTION Scalar total() const {
    return d01 + d02 + d03 + d04 + d05 + d06 + d07 + d08 + d09 +
           d10;
  }
};

template <class Scalar>
struct Plus2Er12TermsT {
  Scalar er01, er02, er03, er04, er05;
  KOKKOS_INLINE_FUNCTION Scalar total() const {
    return er01 + er02 + er03 + er04 + er05;
  }
};

template <class Scalar>
struct Plus2Et12TermsT {
  Scalar et01, et02, et03, et04, et05, et06;
  KOKKOS_INLINE_FUNCTION Scalar total() const {
    return et01 + et02 + et03 + et04 + et05 + et06;
  }
};

template <class Scalar>
struct Plus2J12TermsT {
  Scalar j01, j02;
  KOKKOS_INLINE_FUNCTION Scalar total() const { return j01 + j02; }
};

template <class Scalar>
struct Plus2K12TermsT {
  Scalar k01, k02, k03;
  KOKKOS_INLINE_FUNCTION Scalar total() const { return k01 + k02 + k03; }
};

template <class Scalar>
struct Plus2Q12TermsT {
  Scalar q01, q02;
  KOKKOS_INLINE_FUNCTION Scalar total() const { return q01 + q02; }
};

template <class Scalar>
struct Plus2OrderedPairSourceT {
  Plus2C12TermsT<Scalar> C12;
  Plus2B12TermsT<Scalar> B12;
  Plus2D12TermsT<Scalar> D12;
  Plus2Er12TermsT<Scalar> Er12;
  Plus2Et12TermsT<Scalar> Et12;
  Plus2J12TermsT<Scalar> J12;
  Plus2K12TermsT<Scalar> K12;
  Plus2Q12TermsT<Scalar> Q12;
};

template <class Scalar>
struct Plus2OrderedPairJkSourceT {
  Plus2C12TermsT<Scalar> C12;
  Plus2B12TermsT<Scalar> B12;
  Plus2D12TermsT<Scalar> D12;
  Plus2J12TermsT<Scalar> J12;
  Plus2K12TermsT<Scalar> K12;
};

template <class Scalar>
struct Plus2OrderedPairQSourceT {
  Plus2Er12TermsT<Scalar> Er12;
  Plus2Et12TermsT<Scalar> Et12;
  Plus2Q12TermsT<Scalar> Q12;
};

using Plus2OrderedPairFields = Plus2OrderedPairFieldsT<Complex>;
using Plus2OrderedPairDerivatives = Plus2OrderedPairDerivativesT<Complex>;
using Plus2OrderedPairSource = Plus2OrderedPairSourceT<Complex>;

template <class Scalar>
KOKKOS_INLINE_FUNCTION Plus2OrderedPairJkSourceT<Scalar>
plus2_compact_ordered_pair_jk_source(
    const double radius, const KerrBackgroundPoint& background,
    const Plus2OrderedPairFieldsT<Scalar>& f,
    const Plus2OrderedPairJkDerivativesT<Scalar>& d) {
  const Complex mubar0 = Kokkos::conj(background.mu0);
  const Complex pibar0 = Kokkos::conj(background.pi0);
  const Complex taubar0 = Kokkos::conj(background.tau0);
  const double radius2 = radius * radius;

  Plus2OrderedPairJkSourceT<Scalar> result;
  result.C12 = {
      -f.Csharp1 * d.capital_delta4_Z1_2,
      0.5 * f.Bsharp1 * d.ethprime4_Z1_2,
      -1.5 * d.capital_delta2_Csharp1 * f.Z1_2,
      -0.5 * d.ethprime1_Bsharp1 * f.Z1_2,
      -1.5 * radius * background.mu0 * f.Csharp1 * f.Z1_2,
      radius * mubar0 * f.Csharp1 * f.Z1_2,
      2.5 * radius * background.pi0 * f.Bsharp1 * f.Z1_2,
      0.5 * radius * taubar0 * f.Bsharp1 * f.Z1_2};

  result.B12 = {
      -f.C1 * d.capital_delta5_Z0_2,
      0.5 * f.B1 * d.eth5_Z0_2,
      0.5 * d.capital_delta2_C1 * f.Z0_2,
      d.eth1_B1 * f.Z0_2,
      -2.0 * radius * background.mu0 * f.C1 * f.Z0_2,
      0.5 * radius * mubar0 * f.C1 * f.Z0_2,
      radius * pibar0 * f.B1 * f.Z0_2,
      0.5 * radius * background.tau0 * f.B1 * f.Z0_2};

  result.D12 = {
      -0.5 * f.V1 * d.capital_delta4_Z1_2,
      0.5 * d.capital_delta2_V1 * f.Z1_2,
      2.5 * radius * d.eth2_C1 * f.Z1_2,
      -2.5 * radius * d.ethprime2_Csharp1 * f.Z1_2,
      -2.5 * radius * background.mu0 * f.V1 * f.Z1_2,
      0.5 * radius * mubar0 * f.V1 * f.Z1_2,
      2.5 * radius2 * pibar0 * f.C1 * f.Z1_2,
      5.0 * radius2 * background.tau0 * f.C1 * f.Z1_2,
      3.5 * radius2 * background.pi0 * f.Csharp1 * f.Z1_2,
      radius2 * taubar0 * f.Csharp1 * f.Z1_2};

  result.J12 = {radius * result.C12.total(), 3.0 * f.Sig1 * f.H2};
  result.K12 = {radius * result.B12.total(), -result.D12.total(),
                -3.0 * f.Kap1 * f.H2};
  return result;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Plus2OrderedPairQSourceT<Scalar>
plus2_compact_ordered_pair_q_source(
    const double radius, const KerrBackgroundPoint& background,
    const Plus2OrderedPairFieldsT<Scalar>& f,
    const Plus2OrderedPairQDerivativesT<Scalar>& d) {
  const Complex mubar0 = Kokkos::conj(background.mu0);
  const Complex taubar0 = Kokkos::conj(background.tau0);

  Plus2OrderedPairQSourceT<Scalar> result;
  result.Er12 = {
      -0.5 * f.V1 * d.capital_delta2_Sig2,
      -3.0 * f.Ep1 * f.Sig2,
      f.Epsharp1 * f.Sig2,
      -radius * f.Rh1 * f.Sig2,
      -radius * f.Rhsharp1 * f.Sig2};

  result.Et12 = {
      -f.Csharp1 * d.capital_delta3_Kap2,
      0.5 * f.Bsharp1 * d.ethprime3_Kap2,
      -0.5 * d.ethprime1_Bsharp1 * f.Kap2,
      radius * mubar0 * f.Csharp1 * f.Kap2,
      1.5 * radius * background.pi0 * f.Bsharp1 * f.Kap2,
      0.5 * radius * taubar0 * f.Bsharp1 * f.Kap2};

  result.Q12 = {result.Er12.total(), -radius * result.Et12.total()};
  return result;
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Plus2OrderedPairSourceT<Scalar>
plus2_compact_ordered_pair_source(
    const double radius, const KerrBackgroundPoint& background,
    const Plus2OrderedPairFieldsT<Scalar>& f,
    const Plus2OrderedPairDerivativesT<Scalar>& d) {
  const auto jk = plus2_compact_ordered_pair_jk_source(
      radius, background, f,
      Plus2OrderedPairJkDerivativesT<Scalar>{
          d.capital_delta4_Z1_2, d.ethprime4_Z1_2,
          d.capital_delta2_Csharp1, d.ethprime1_Bsharp1,
          d.capital_delta5_Z0_2, d.eth5_Z0_2, d.capital_delta2_C1,
          d.eth1_B1, d.capital_delta2_V1, d.eth2_C1,
          d.ethprime2_Csharp1});
  const auto q = plus2_compact_ordered_pair_q_source(
      radius, background, f,
      Plus2OrderedPairQDerivativesT<Scalar>{
          d.ethprime1_Bsharp1, d.capital_delta2_Sig2,
          d.capital_delta3_Kap2, d.ethprime3_Kap2});
  return {jk.C12, jk.B12, jk.D12, q.Er12, q.Et12,
          jk.J12, jk.K12, q.Q12};
}

template <class Scalar>
struct Plus2OuterDerivativesT {
  Scalar thorn5_J;
  Scalar eth6_K;
};

template <class Scalar>
struct Plus2OuterSourceTermsT {
  Scalar s01, s02, s03, s04, s05, s06, s07;
  KOKKOS_INLINE_FUNCTION Scalar total() const {
    return s01 + s02 + s03 + s04 + s05 + s06 + s07;
  }
};

template <class Scalar>
KOKKOS_INLINE_FUNCTION Plus2OuterSourceTermsT<Scalar>
plus2_compact_outer_source_over_r6(
    const double radius, const KerrBackgroundPoint& background,
    const Scalar& summed_J, const Scalar& summed_K, const Scalar& summed_Q,
    const Plus2OuterDerivativesT<Scalar>& derivatives) {
  const double radius2 = radius * radius;
  return {
      derivatives.thorn5_J,
      -4.0 * background.rho0 * summed_J,
      -Kokkos::conj(background.rho0) * summed_J,
      radius * derivatives.eth6_K,
      -4.0 * radius2 * background.tau0 * summed_K,
      radius2 * Kokkos::conj(background.pi0) * summed_K,
      -3.0 * radius * background.psi20 * summed_Q};
}

template <class Scalar>
KOKKOS_INLINE_FUNCTION Scalar plus2_coordinate_forcing_from_source_over_r6(
    const double radius, const double cos_theta, const double kerr_spin,
    const double compactification_length, const Scalar& source_over_r6) {
  const double radius2 = radius * radius;
  const double radius3 = radius2 * radius;
  const double length2 =
      compactification_length * compactification_length;
  const double length4 = length2 * length2;
  return 2.0 *
         (length4 + kerr_spin * kerr_spin * radius2 * cos_theta * cos_theta) *
         radius3 * source_over_r6;
}

// Setup-time provenance for explicit X_m^sharp=conj(X_-m) lookup.  The source
// evaluator accepts already-selected sharp values so its device path performs
// no search or allocation.  This helper validates the indices once when a
// standalone source workspace is constructed.
struct Plus2PairLookup {
  int m1;
  int m2;
  int target;
  std::size_t index1;
  std::size_t index2;
  std::size_t target_index;
  std::size_t sharp1;
  std::size_t sharp2;
};

inline Plus2PairLookup make_plus2_pair_lookup(const ModeRegistry& registry,
                                               const ModePair& pair) {
  if (static_cast<long long>(pair.m1) + static_cast<long long>(pair.m2) !=
          static_cast<long long>(pair.target) ||
      pair.index1 != registry.index(pair.m1) ||
      pair.index2 != registry.index(pair.m2) ||
      pair.target_index != registry.index(pair.target)) {
    throw std::invalid_argument("spin +2 source received inconsistent mode pair");
  }
  return {pair.m1, pair.m2, pair.target, pair.index1, pair.index2,
          pair.target_index, registry.sharp_index(pair.m1),
          registry.sharp_index(pair.m2)};
}

}  // namespace teuk
