#pragma once

#include <algorithm>
#include <cmath>

#include "teuk/angular.hpp"
#include "teuk/background.hpp"
#include "teuk/ghp.hpp"
#include "teuk/jet.hpp"
#include "teuk/reconstruction.hpp"
#include "teuk/second_order.hpp"
#include "teuk/teukolsky.hpp"

namespace teuk {

struct ReconstructionState {
  Complex G = 0.0;
  Complex Lambda = 0.0;
  Complex H = 0.0;
  Complex B = 0.0;
  Complex Pi = 0.0;
  Complex C = 0.0;
  Complex U = 0.0;

  KOKKOS_INLINE_FUNCTION ReconstructionState& operator+=(
      const ReconstructionState& other) {
    G += other.G;
    Lambda += other.Lambda;
    H += other.H;
    B += other.B;
    Pi += other.Pi;
    C += other.C;
    U += other.U;
    return *this;
  }
};

KOKKOS_INLINE_FUNCTION ReconstructionState operator+(
    ReconstructionState left, const ReconstructionState& right) {
  left += right;
  return left;
}

KOKKOS_INLINE_FUNCTION ReconstructionState operator*(
    const double scale, const ReconstructionState& state) {
  return {scale * state.G,      scale * state.Lambda, scale * state.H,
          scale * state.B,      scale * state.Pi,     scale * state.C,
          scale * state.U};
}

KOKKOS_INLINE_FUNCTION ReconstructionState operator*(
    const ReconstructionState& state, const double scale) {
  return scale * state;
}

struct PointPipelineState {
  TeukolskyState first;
  ReconstructionState reconstruction;
  TeukolskyState second;

  KOKKOS_INLINE_FUNCTION PointPipelineState& operator+=(
      const PointPipelineState& other) {
    first += other.first;
    reconstruction += other.reconstruction;
    second += other.second;
    return *this;
  }
};

KOKKOS_INLINE_FUNCTION PointPipelineState operator+(
    PointPipelineState left, const PointPipelineState& right) {
  left += right;
  return left;
}

KOKKOS_INLINE_FUNCTION PointPipelineState operator*(
    const double scale, const PointPipelineState& state) {
  return {scale * state.first, scale * state.reconstruction,
          scale * state.second};
}

KOKKOS_INLINE_FUNCTION PointPipelineState operator*(
    const PointPipelineState& state, const double scale) {
  return scale * state;
}

// A one-point vertical slice used as the readable common-stage correctness
// oracle. Spatial production code supplies derivatives from SBP/angular
// matrices; here radial derivatives vanish and a single (ell,m=0) angular
// shape supplies exact raising/lowering ratios. Every continuum RHS and source
// operator is still the same function used by the spatial kernels.
struct PointPipelineParameters {
  KerrParameters background;
  double radius = 0.4;
  double theta = 0.9;
  int ell = 2;
  double reduction_damping = 0.1;
};

struct PointPipelineDiagnostics {
  InnerSource inner_source;
  Complex source_over_r3 = 0.0;
  Complex forcing = 0.0;
};

inline PointPipelineState make_point_pipeline_seed(const double amplitude) {
  PointPipelineState state;
  state.first.P = amplitude * Complex(0.8, -0.2);
  state.first.Q = 0.0;
  state.first.psi = amplitude * Complex(0.3, 0.4);
  state.reconstruction =
      amplitude * ReconstructionState{
                      {0.12, -0.04}, {-0.08, 0.03}, {0.05, 0.07},
                      {-0.02, 0.09}, {0.04, -0.06}, {0.03, 0.02},
                      {-0.07, -0.01}};
  return state;
}

namespace detail {

inline Complex harmonic_ratio(const int ell, const int m,
                              const int source_spin,
                              const int target_spin,
                              const double theta) {
  const double source = angular::spin_weighted_harmonic_theta(
      ell, m, source_spin, theta);
  if (std::abs(source) < 1.0e-13) {
    throw std::invalid_argument("angular oracle point lies on a harmonic zero");
  }
  const double target = angular::spin_weighted_harmonic_theta(
      ell, m, target_spin, theta);
  return Complex(target / source, 0.0);
}

inline Complex raised_value(const Complex value, const int ell, const int m,
                            const int spin, const double theta) {
  if (spin >= ell) return 0.0;
  return angular::raising_factor(ell, spin) *
         harmonic_ratio(ell, m, spin, spin + 1, theta) * value;
}

inline Complex lowered_value(const Complex value, const int ell, const int m,
                             const int spin, const double theta) {
  if (spin <= -ell) return 0.0;
  return angular::lowering_factor(ell, spin) *
         harmonic_ratio(ell, m, spin, spin - 1, theta) * value;
}

inline Complex eth(const Complex value, const Complex dt_value,
                   const int spin, const int boost, const double radius,
                   const PointPipelineParameters& parameters) {
  return eth_n_point(
      value, dt_value,
      raised_value(value, parameters.ell, 0, spin, parameters.theta), spin,
      boost, radius, std::sin(parameters.theta), std::cos(parameters.theta),
      parameters.background.spin,
      parameters.background.compactification_length);
}

inline Complex ethprime(const Complex value, const Complex dt_value,
                        const int spin, const int boost, const double radius,
                        const PointPipelineParameters& parameters) {
  return ethprime_n_point(
      value, dt_value,
      lowered_value(value, parameters.ell, 0, spin, parameters.theta), spin,
      boost, radius, std::sin(parameters.theta), std::cos(parameters.theta),
      parameters.background.spin,
      parameters.background.compactification_length);
}

struct ReconstructionEvaluation {
  ReconstructionState dt;
  ReconstructionDeltaRhs delta;
  ReconstructionAngularDerivatives angular;
};

inline ReconstructionFields reconstruction_fields(
    const ReconstructionState& state, const Complex F) {
  return {F,
          state.G,
          state.H,
          state.Lambda,
          state.Pi,
          state.B,
          state.C,
          state.U,
          Kokkos::conj(state.Pi),
          Kokkos::conj(state.B),
          Kokkos::conj(state.C)};
}

inline ReconstructionEvaluation evaluate_reconstruction(
    const ReconstructionState& state, const Complex F, const Complex F_dt,
    const PointPipelineParameters& parameters,
    const KerrBackgroundPoint& background) {
  const double radius = parameters.radius;
  ReconstructionAngularDerivatives angular{};
  angular.eth1_F = eth(F, F_dt, -2, -2, radius, parameters);
  const ReconstructionFields fields = reconstruction_fields(state, F);

  ReconstructionDeltaRhs delta =
      reconstruction_delta_rhs(radius, background, fields, angular);
  ReconstructionState dt;
  dt.G = reconstruction_time_derivative(
      state.G, 0.0, delta.G, 2, radius, parameters.background.mass,
      parameters.background.compactification_length);
  dt.Lambda = reconstruction_time_derivative(
      state.Lambda, 0.0, delta.Lambda, 1, radius,
      parameters.background.mass,
      parameters.background.compactification_length);

  angular.eth2_G = eth(state.G, dt.G, -1, -1, radius, parameters);
  delta = reconstruction_delta_rhs(radius, background, fields, angular);
  dt.H = reconstruction_time_derivative(
      state.H, 0.0, delta.H, 3, radius, parameters.background.mass,
      parameters.background.compactification_length);
  dt.B = reconstruction_time_derivative(
      state.B, 0.0, delta.B, 1, radius, parameters.background.mass,
      parameters.background.compactification_length);
  dt.Pi = reconstruction_time_derivative(
      state.Pi, 0.0, delta.Pi, 2, radius, parameters.background.mass,
      parameters.background.compactification_length);
  dt.C = reconstruction_time_derivative(
      state.C, 0.0, delta.C, 2, radius, parameters.background.mass,
      parameters.background.compactification_length);

  angular.eth2_C = eth(state.C, dt.C, -1, 1, radius, parameters);
  angular.eth2_Pi = eth(state.Pi, dt.Pi, -1, 0, radius, parameters);
  angular.ethprime1_B_sharp = ethprime(
      Kokkos::conj(state.B), Kokkos::conj(dt.B), 2, 0, radius, parameters);
  angular.ethprime2_C_sharp = ethprime(
      Kokkos::conj(state.C), Kokkos::conj(dt.C), 1, 1, radius, parameters);
  delta = reconstruction_delta_rhs(radius, background, fields, angular);
  dt.U = reconstruction_time_derivative(
      state.U, 0.0, delta.U, 3, radius, parameters.background.mass,
      parameters.background.compactification_length);
  return {dt, delta, angular};
}

inline TeukolskyState homogeneous_teukolsky_rhs(
    const TeukolskyState& state, const PointPipelineParameters& parameters,
    const Complex forcing) {
  TeukolskyParameters teukolsky;
  teukolsky.mass = parameters.background.mass;
  teukolsky.spin = parameters.background.spin;
  teukolsky.compactification_length =
      parameters.background.compactification_length;
  teukolsky.spin_weight = -2;
  teukolsky.azimuthal_mode = 0;
  teukolsky.reduction_damping = parameters.reduction_damping;
  const auto coefficients = teukolsky_coefficients(
      teukolsky, parameters.radius, parameters.theta);
  const Complex psi_dt = teukolsky_psi_rhs(coefficients, state);
  const Complex angular =
      angular::spin_weighted_laplacian_eigenvalue(parameters.ell, -2) *
      state.psi;
  return {teukolsky_p_rhs(coefficients, state, 0.0, angular, forcing),
          teukolsky_q_rhs(0.0, state.Q, parameters.reduction_damping), psi_dt};
}

inline InnerSourceT<Jet1<Complex>> evaluate_inner_source_jet(
    const PointPipelineState& state, const PointPipelineState& dt,
    const PointPipelineState& ddt,
    const ReconstructionEvaluation& reconstruction,
    const ReconstructionEvaluation& reconstruction_tangent,
    const PointPipelineParameters& parameters,
    const KerrBackgroundPoint& background) {
  using J = Jet1<Complex>;
  const double radius = parameters.radius;
  const auto make_jet = [](const Complex value, const Complex tangent) {
    return J(value, tangent);
  };

  const Complex delta1_F = delta_n_point(
      state.first.psi, dt.first.psi, 0.0, 1, radius,
      parameters.background.mass,
      parameters.background.compactification_length);
  const Complex delta1_F_dt = delta_n_point(
      dt.first.psi, ddt.first.psi, 0.0, 1, radius,
      parameters.background.mass,
      parameters.background.compactification_length);
  const Complex eth1_B = eth(state.reconstruction.B, dt.reconstruction.B,
                             -2, 0, radius, parameters);
  const Complex eth1_B_dt =
      eth(dt.reconstruction.B, ddt.reconstruction.B, -2, 0, radius,
          parameters);
  const Complex ethprime1_F =
      ethprime(state.first.psi, dt.first.psi, -2, -2, radius, parameters);
  const Complex ethprime1_F_dt =
      ethprime(dt.first.psi, ddt.first.psi, -2, -2, radius, parameters);

  const ReconstructionState& r = state.reconstruction;
  const ReconstructionState& rt = dt.reconstruction;
  OrderedPairFieldsT<J> fields{
      make_jet(state.first.psi, dt.first.psi), make_jet(r.G, rt.G),
      make_jet(r.Lambda, rt.Lambda),          make_jet(r.Pi, rt.Pi),
      make_jet(r.B, rt.B),                    make_jet(r.C, rt.C),
      make_jet(r.U, rt.U),                    make_jet(state.first.psi, dt.first.psi),
      make_jet(r.G, rt.G),                    make_jet(r.H, rt.H),
      make_jet(r.B, rt.B),                    make_jet(r.C, rt.C),
      make_jet(r.U, rt.U),
      make_jet(Kokkos::conj(r.U), Kokkos::conj(rt.U)),
      make_jet(Kokkos::conj(r.C), Kokkos::conj(rt.C)),
      make_jet(Kokkos::conj(r.C), Kokkos::conj(rt.C)),
      make_jet(Kokkos::conj(r.B), Kokkos::conj(rt.B)),
      make_jet(Kokkos::conj(r.Pi), Kokkos::conj(rt.Pi)),
      make_jet(Kokkos::conj(r.B), Kokkos::conj(rt.B))};

  OrderedPairDerivativesT<J> derivatives{
      make_jet(delta1_F, delta1_F_dt),
      make_jet(reconstruction.delta.U, reconstruction_tangent.delta.U),
      make_jet(reconstruction.angular.eth2_C,
               reconstruction_tangent.angular.eth2_C),
      make_jet(reconstruction.angular.ethprime2_C_sharp,
               reconstruction_tangent.angular.ethprime2_C_sharp),
      make_jet(eth1_B, eth1_B_dt),
      make_jet(reconstruction.delta.C, reconstruction_tangent.delta.C),
      make_jet(reconstruction.delta.G, reconstruction_tangent.delta.G),
      make_jet(reconstruction.angular.eth2_G,
               reconstruction_tangent.angular.eth2_G),
      make_jet(ethprime1_F, ethprime1_F_dt),
      make_jet(Kokkos::conj(reconstruction.delta.C),
               Kokkos::conj(reconstruction_tangent.delta.C)),
      make_jet(reconstruction.angular.ethprime1_B_sharp,
               reconstruction_tangent.angular.ethprime1_B_sharp)};
  return corrected_ordered_pair_source(radius, background, fields, derivatives);
}

}  // namespace detail

inline PointPipelineState evaluate_point_pipeline_rhs(
    const PointPipelineState& state,
    const PointPipelineParameters& parameters,
    PointPipelineDiagnostics* diagnostics = nullptr) {
  const KerrBackgroundPoint background = kerr_background_point(
      parameters.background, parameters.radius, parameters.theta);

  PointPipelineState rhs;
  rhs.first = detail::homogeneous_teukolsky_rhs(state.first, parameters, 0.0);
  const TeukolskyState first_ddt =
      detail::homogeneous_teukolsky_rhs(rhs.first, parameters, 0.0);

  const auto reconstruction = detail::evaluate_reconstruction(
      state.reconstruction, state.first.psi, rhs.first.psi, parameters,
      background);
  rhs.reconstruction = reconstruction.dt;
  const auto reconstruction_tangent = detail::evaluate_reconstruction(
      rhs.reconstruction, rhs.first.psi, first_ddt.psi, parameters,
      background);

  PointPipelineState tangent_rhs;
  tangent_rhs.first = first_ddt;
  tangent_rhs.reconstruction = reconstruction_tangent.dt;
  const auto source_jet = detail::evaluate_inner_source_jet(
      state, rhs, tangent_rhs, reconstruction, reconstruction_tangent,
      parameters, background);
  const InnerSource inner{source_jet.D.value, source_jet.T.value};
  const Complex delta3_D = delta_n_point(
      source_jet.D.value, source_jet.D.dt, 0.0, 3, parameters.radius,
      parameters.background.mass,
      parameters.background.compactification_length);
  const Complex ethprime3_T = detail::ethprime(
      source_jet.T.value, source_jet.T.dt, -1, -2, parameters.radius,
      parameters);
  const Complex source_over_r3 = outer_source_over_r3(
      parameters.radius, background, inner,
      OuterSourceDerivatives{delta3_D, ethprime3_T});
  const Complex forcing = coordinate_second_order_forcing(
      parameters.radius, std::cos(parameters.theta),
      parameters.background.spin,
      parameters.background.compactification_length, source_over_r3);
  rhs.second =
      detail::homogeneous_teukolsky_rhs(state.second, parameters, forcing);

  if (diagnostics != nullptr) {
    diagnostics->inner_source = inner;
    diagnostics->source_over_r3 = source_over_r3;
    diagnostics->forcing = forcing;
  }
  return rhs;
}

}  // namespace teuk
