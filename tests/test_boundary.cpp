#include "test_harness.hpp"

#include <array>
#include <cmath>

#include "teuk/boundary.hpp"

TEST_CASE("Teukolsky principal speeds solve the full characteristic polynomial") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.72;
  parameters.compactification_length = 1.6;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = 3;
  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, 0.43, 0.81);
  const auto characteristics =
      teuk::radial_principal_characteristics(coefficients);
  for (const double lambda : {characteristics.lambda_minus,
                              characteristics.lambda_zero,
                              characteristics.lambda_plus}) {
    const double polynomial =
        lambda *
        (lambda * lambda -
         2.0 * coefficients.radial_advection / coefficients.time * lambda -
         coefficients.radial_principal / coefficients.time);
    CHECK_NEAR(polynomial, 0.0, 2.0e-14);
  }
  CHECK_NEAR(characteristics.coordinate_velocity_minus,
             -characteristics.lambda_minus, 0.0);
  CHECK_NEAR(characteristics.coordinate_velocity_plus,
             -characteristics.lambda_plus, 0.0);
}
TEST_CASE("scri and horizon have no incoming propagating characteristic") {
  for (const double spin : {0.0, 0.7, 0.999999, 1.0, -0.8}) {
    teuk::TeukolskyParameters parameters;
    parameters.mass = 1.0;
    parameters.spin = spin;
    parameters.compactification_length = 1.7;
    parameters.spin_weight = -2;
    parameters.azimuthal_mode = -2;
    for (const double theta : {0.2, 0.9, 1.5}) {
      const auto scri = teuk::radial_principal_characteristics(
          parameters, 0.0, theta);
      const auto scri_classification = teuk::classify_radial_boundary(
          scri, teuk::RadialBoundarySide::ScriLower);
      CHECK(scri_classification.incoming == 0);
      CHECK(scri_classification.outgoing == 1);
      CHECK(scri_classification.stationary == 2);

      const double horizon =
          teuk::compactified_outer_horizon_radius(parameters);
      const auto horizon_characteristics =
          teuk::radial_principal_characteristics(parameters, horizon, theta);
      const auto horizon_classification = teuk::classify_radial_boundary(
          horizon_characteristics, teuk::RadialBoundarySide::HorizonUpper,
          2.0e-11);
      CHECK(horizon_classification.incoming == 0);
      CHECK(horizon_classification.outgoing == 1);
      CHECK(horizon_classification.stationary == 2);
    }
  }
}

TEST_CASE("endpoint coefficient identities fix the outward speed signs") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.3;
  parameters.spin = 0.91;
  parameters.compactification_length = 2.2;
  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const auto scri_coefficients =
      teuk::teukolsky_coefficients(parameters, 0.0, 1.1);
  const auto horizon_coefficients =
      teuk::teukolsky_coefficients(parameters, horizon, 1.1);

  CHECK_NEAR(scri_coefficients.radial_advection,
             parameters.compactification_length *
                 parameters.compactification_length,
             2.0e-14);
  CHECK_NEAR(scri_coefficients.radial_principal, 0.0, 2.0e-14);
  CHECK_NEAR(horizon_coefficients.radial_advection,
             -2.0 * parameters.mass * horizon, 2.0e-13);
  CHECK_NEAR(horizon_coefficients.radial_principal, 0.0, 2.0e-13);
  CHECK(scri_coefficients.time > 0.0);
  CHECK(horizon_coefficients.time > 0.0);
}

TEST_CASE("natural propagating symmetrizer is positive inside and degenerate at endpoints") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.6;
  parameters.compactification_length = 1.5;
  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const auto scri = teuk::teukolsky_coefficients(parameters, 0.0, 0.7);
  const auto interior =
      teuk::teukolsky_coefficients(parameters, 0.5 * horizon, 0.7);
  const auto outer =
      teuk::teukolsky_coefficients(parameters, horizon, 0.7);
  const auto scri_weights = teuk::propagating_symmetrizer_weights(scri);
  const auto interior_weights =
      teuk::propagating_symmetrizer_weights(interior);
  const auto outer_weights = teuk::propagating_symmetrizer_weights(outer);
  CHECK_NEAR(scri_weights.Q, 0.0, 2.0e-14);
  CHECK(interior_weights.Q > 0.0);
  CHECK_NEAR(outer_weights.Q, 0.0, 2.0e-13);
  CHECK_NEAR(interior_weights.P * interior.radial_principal,
             interior_weights.Q / interior.time, 2.0e-14);
}

TEST_CASE("reduction constraint is stationary and damped rather than incoming") {
  const teuk::Complex constraint(0.4, -0.7);
  constexpr double gamma = 1.3;
  const teuk::Complex derivative_velocity(-0.2, 0.6);
  const teuk::Complex q_rhs =
      teuk::teukolsky_q_rhs(derivative_velocity, constraint, gamma);
  CHECK_COMPLEX_NEAR(q_rhs - derivative_velocity,
                     teuk::reduction_constraint_time_derivative(constraint,
                                                                 gamma),
                     1.0e-15);
}

TEST_CASE("characteristic endpoint analysis executes in active Kokkos space") {
  teuk::TeukolskyParameters parameters;
  parameters.spin = 0.8;
  parameters.compactification_length = 1.4;
  Kokkos::View<double*> result("boundary_characteristics", 4);
  Kokkos::parallel_for(
      "boundary_characteristics_kernel", 1, KOKKOS_LAMBDA(const int) {
        const auto scri = teuk::radial_principal_characteristics(
            parameters, 0.0, 0.9);
        const double horizon =
            teuk::compactified_outer_horizon_radius(parameters);
        const auto outer = teuk::radial_principal_characteristics(
            parameters, horizon, 0.9);
        result(0) = scri.coordinate_velocity_plus;
        result(1) = scri.coordinate_velocity_minus;
        result(2) = outer.coordinate_velocity_plus;
        result(3) = outer.coordinate_velocity_minus;
      });
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  CHECK(host(0) < 0.0);
  CHECK_NEAR(host(1), 0.0, 1.0e-14);
  CHECK_NEAR(host(2), 0.0, 2.0e-13);
  CHECK(host(3) > 0.0);
}
