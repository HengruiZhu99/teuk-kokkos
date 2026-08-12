#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "ripley_eq22_oracle.hpp"
#include "teuk/angular.hpp"
#include "teuk/boundary.hpp"
#include "teuk/grid.hpp"
#include "teuk/plus2_field.hpp"
#include "teuk/rk4.hpp"
#include "teuk/teukolsky.hpp"

namespace {

using teuk::Complex;
using teuk::TeukolskyState;

struct PolynomialProfile {
  Complex value;
  Complex first_derivative;
  Complex second_derivative;
};

PolynomialProfile validation_profile(const double radius) {
  const Complex amplitude(0.73, -0.29);
  const double radius2 = radius * radius;
  const double radius3 = radius2 * radius;
  const double radius4 = radius2 * radius2;
  return {
      amplitude * (1.0 + 0.17 * radius - 0.09 * radius2 +
                   0.031 * radius3 - 0.008 * radius4),
      amplitude *
          (0.17 - 0.18 * radius + 0.093 * radius2 - 0.032 * radius3),
      amplitude * (-0.18 + 0.186 * radius - 0.096 * radius2),
  };
}

Complex exponential(const Complex rate, const double time) {
  const std::complex<double> value = std::exp(
      std::complex<double>(rate.real() * time, rate.imag() * time));
  return {value.real(), value.imag()};
}

teuk::test::ripley_eq22::Parameters oracle_parameters(
    const teuk::TeukolskyParameters& parameters) {
  return {parameters.mass, parameters.spin,
          parameters.compactification_length,
          parameters.azimuthal_mode};
}

Complex as_teuk(const std::complex<double> value) {
  return {value.real(), value.imag()};
}

std::vector<TeukolskyState> evolve_source_free_plus2(const int steps) {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.71;
  parameters.compactification_length = 1.5;
  parameters.spin_weight = 2;
  parameters.azimuthal_mode = 1;
  parameters.reduction_damping = 0.23;
  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const teuk::UniformRadialGrid grid(25, 0.0, horizon);
  constexpr double theta = 0.83;
  // Independent Ripley lower-after-raise value -(3-2)(3+2+1).
  constexpr double angular_eigenvalue = -6.0;
  const Complex initial_rate(-0.21, 0.37);

  std::vector<TeukolskyState> state(grid.size());
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    const double radius = grid.coordinate(radial);
    const auto profile = validation_profile(radius);
    const auto coefficients = teuk::teukolsky_coefficients(
        parameters, radius, theta);
    state[radial] = {
        coefficients.time * initial_rate * profile.value -
            2.0 * coefficients.radial_advection * profile.first_derivative +
            coefficients.definition * profile.value,
        profile.first_derivative,
        profile.value,
    };
  }

  std::vector<Complex> angular(grid.size());
  std::vector<Complex> zero_forcing(grid.size(), Complex(0.0, 0.0));
  std::vector<TeukolskyState> rhs(grid.size());
  teuk::TeukolskyRadialWorkspace radial_workspace(grid.size());
  teuk::RK4Workspace<TeukolskyState> rk_workspace(grid.size());
  const auto evaluate_rhs =
      [&](const double, const std::vector<TeukolskyState>& input,
          std::vector<TeukolskyState>& output) {
        for (std::size_t radial = 0; radial < grid.size(); ++radial) {
          angular[radial] = angular_eigenvalue * input[radial].psi;
        }
        teuk::evaluate_teukolsky_radial_line_rhs(
            grid, parameters, theta, input, angular, zero_forcing,
            teuk::ReductionEvolution::FreeDamped, radial_workspace, output);
      };

  constexpr double final_time = 0.08;
  const double time_step = final_time / static_cast<double>(steps);
  double time = 0.0;
  for (int step = 0; step < steps; ++step) {
    teuk::classical_rk4_step(state, time, time_step, evaluate_rhs,
                             rk_workspace);
    time += time_step;
  }
  return state;
}

double state_distance(const std::vector<TeukolskyState>& left,
                      const std::vector<TeukolskyState>& right) {
  double maximum = 0.0;
  for (std::size_t point = 0; point < left.size(); ++point) {
    maximum = std::max(maximum, Kokkos::abs(left[point].P - right[point].P));
    maximum = std::max(maximum, Kokkos::abs(left[point].Q - right[point].Q));
    maximum =
        std::max(maximum, Kokkos::abs(left[point].psi - right[point].psi));
  }
  return maximum;
}

struct ConstraintEvolutionResult {
  double state_error;
  double constraint_error;
};

ConstraintEvolutionResult evolve_damped_plus2_constraint(const int steps) {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.53;
  parameters.compactification_length = 1.8;
  parameters.spin_weight = 2;
  parameters.azimuthal_mode = -2;
  parameters.reduction_damping = 1.4;
  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const teuk::UniformRadialGrid grid(17, 0.0, horizon);
  constexpr double theta = 1.07;
  constexpr double final_time = 0.11;
  constexpr double angular_eigenvalue = -6.0;
  const Complex initial_constraint(0.19, -0.13);

  const auto exact_state = [&](const double radius, const double time) {
    const auto profile = validation_profile(radius);
    const Complex constraint =
        exponential(Complex(-parameters.reduction_damping, 0.0), time) *
        initial_constraint;
    const Complex q = profile.first_derivative + constraint;
    const auto expected = teuk::test::ripley_eq22::coefficients(
        oracle_parameters(parameters), radius, theta);
    const Complex p = -2.0 * expected.radial_advection * q +
                      as_teuk(expected.definition) * profile.value;
    return TeukolskyState{p, q, profile.value};
  };
  const auto validation_forcing = [&](const double radius,
                                      const double time) {
    const auto profile = validation_profile(radius);
    const auto exact = exact_state(radius, time);
    const Complex constraint =
        exponential(Complex(-parameters.reduction_damping, 0.0), time) *
        initial_constraint;
    const auto expected = teuk::test::ripley_eq22::coefficients(
        oracle_parameters(parameters), radius, theta);
    const Complex p_t = 2.0 * expected.radial_advection *
                        parameters.reduction_damping * constraint;
    return p_t - expected.radial_principal * profile.second_derivative -
           as_teuk(expected.q) * exact.Q -
           as_teuk(expected.psi) * exact.psi -
           angular_eigenvalue * exact.psi;
  };

  std::vector<TeukolskyState> state(grid.size());
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    state[radial] = exact_state(grid.coordinate(radial), 0.0);
  }
  std::vector<Complex> angular(grid.size());
  std::vector<Complex> forcing(grid.size());
  std::vector<TeukolskyState> rhs(grid.size());
  teuk::TeukolskyRadialWorkspace radial_workspace(grid.size());
  teuk::RK4Workspace<TeukolskyState> rk_workspace(grid.size());
  const auto evaluate_rhs =
      [&](const double stage_time,
          const std::vector<TeukolskyState>& input,
          std::vector<TeukolskyState>& output) {
        for (std::size_t radial = 0; radial < grid.size(); ++radial) {
          angular[radial] = angular_eigenvalue * input[radial].psi;
          forcing[radial] =
              validation_forcing(grid.coordinate(radial), stage_time);
        }
        teuk::evaluate_teukolsky_radial_line_rhs(
            grid, parameters, theta, input, angular, forcing,
            teuk::ReductionEvolution::FreeDamped, radial_workspace, output);
      };

  const double time_step = final_time / static_cast<double>(steps);
  double time = 0.0;
  for (int step = 0; step < steps; ++step) {
    teuk::classical_rk4_step(state, time, time_step, evaluate_rhs,
                             rk_workspace);
    time += time_step;
  }

  double state_error = 0.0;
  std::vector<Complex> psi(grid.size());
  std::vector<Complex> derivative(grid.size());
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    const auto exact = exact_state(grid.coordinate(radial), final_time);
    state_error = std::max(state_error, Kokkos::abs(state[radial].P - exact.P));
    state_error = std::max(state_error, Kokkos::abs(state[radial].Q - exact.Q));
    state_error =
        std::max(state_error, Kokkos::abs(state[radial].psi - exact.psi));
    psi[radial] = state[radial].psi;
  }
  teuk::fourth_order_radial_derivative(grid, psi, derivative);
  const Complex expected_constraint =
      exponential(Complex(-parameters.reduction_damping, 0.0), final_time) *
      initial_constraint;
  double constraint_error = 0.0;
  for (std::size_t radial = 0; radial < grid.size(); ++radial) {
    constraint_error = std::max(
        constraint_error,
        Kokkos::abs(state[radial].Q - derivative[radial] -
                    expected_constraint));
  }
  return {state_error, constraint_error};
}

}  // namespace

TEST_CASE("source-free homogeneous spin plus2 evolution self-converges at RK4 order") {
  const auto steps_8 = evolve_source_free_plus2(8);
  const auto steps_16 = evolve_source_free_plus2(16);
  const auto steps_32 = evolve_source_free_plus2(32);
  const auto steps_64 = evolve_source_free_plus2(64);
  const double difference_8_16 = state_distance(steps_8, steps_16);
  const double difference_16_32 = state_distance(steps_16, steps_32);
  const double difference_32_64 = state_distance(steps_32, steps_64);
  CHECK(difference_8_16 / difference_16_32 > 13.5);
  CHECK(difference_16_32 / difference_32_64 > 13.5);
  CHECK(difference_32_64 < 2.0e-8);
}

TEST_CASE("spin plus2 reduction constraint follows exact damped evolution") {
  const auto coarse = evolve_damped_plus2_constraint(2);
  const auto medium = evolve_damped_plus2_constraint(4);
  const auto fine = evolve_damped_plus2_constraint(8);
  CHECK(coarse.state_error / medium.state_error > 13.5);
  CHECK(medium.state_error / fine.state_error > 13.5);
  CHECK(coarse.constraint_error / medium.constraint_error > 13.5);
  CHECK(medium.constraint_error / fine.constraint_error > 13.5);
  CHECK(fine.constraint_error < 2.0e-9);
}

TEST_CASE("spin plus2 principal characteristics equal spin minus2 everywhere") {
  for (const double black_hole_spin : {0.0, 0.71, 0.999, -0.84}) {
    teuk::TeukolskyParameters plus;
    plus.mass = 1.0;
    plus.spin = black_hole_spin;
    plus.compactification_length = 1.7;
    plus.spin_weight = 2;
    plus.azimuthal_mode = 2;
    auto minus = plus;
    minus.spin_weight = -2;
    const double horizon = teuk::compactified_outer_horizon_radius(plus);
    for (const double theta : {0.0, 0.42, 1.3, teuk::angular::pi}) {
      for (const double fraction : {0.0, 0.37, 1.0}) {
        const double radius = fraction * horizon;
        const auto plus_characteristics =
            teuk::radial_principal_characteristics(plus, radius, theta);
        const auto minus_characteristics =
            teuk::radial_principal_characteristics(minus, radius, theta);
        CHECK_NEAR(plus_characteristics.lambda_minus,
                   minus_characteristics.lambda_minus, 0.0);
        CHECK_NEAR(plus_characteristics.lambda_zero,
                   minus_characteristics.lambda_zero, 0.0);
        CHECK_NEAR(plus_characteristics.lambda_plus,
                   minus_characteristics.lambda_plus, 0.0);
      }
      const auto scri = teuk::classify_radial_boundary(
          teuk::radial_principal_characteristics(plus, 0.0, theta),
          teuk::RadialBoundarySide::ScriLower);
      const auto outer = teuk::classify_radial_boundary(
          teuk::radial_principal_characteristics(plus, horizon, theta),
          teuk::RadialBoundarySide::HorizonUpper, 2.0e-11);
      CHECK(scri.incoming == 0);
      CHECK(scri.outgoing == 1);
      CHECK(scri.stationary == 2);
      CHECK(outer.incoming == 0);
      CHECK(outer.outgoing == 1);
      CHECK(outer.stationary == 2);
    }
  }
}

TEST_CASE("device plus2 scaling and coefficients match independent host equations") {
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.999;
  parameters.compactification_length = 2.0;
  parameters.spin_weight = 2;
  parameters.azimuthal_mode = -2;
  constexpr std::size_t point_count = 5;
  const double horizon = teuk::compactified_outer_horizon_radius(parameters);
  const std::array<double, point_count> host_radius{
      0.0, 0.13 * horizon, 0.47 * horizon, 0.81 * horizon, horizon};
  const std::array<double, point_count> host_theta{
      0.0, 0.39, 1.17, 2.41, teuk::angular::pi};

  Kokkos::View<double*> radius("plus2_parity_radius", point_count);
  Kokkos::View<double*> theta("plus2_parity_theta", point_count);
  Kokkos::View<teuk::TeukolskyCoefficients*> coefficients(
      "plus2_parity_coefficients", point_count);
  Kokkos::View<Complex*> scaling("plus2_parity_scaling", point_count);
  auto radius_mirror = Kokkos::create_mirror_view(radius);
  auto theta_mirror = Kokkos::create_mirror_view(theta);
  for (std::size_t point = 0; point < point_count; ++point) {
    radius_mirror(point) = host_radius[point];
    theta_mirror(point) = host_theta[point];
  }
  Kokkos::deep_copy(radius, radius_mirror);
  Kokkos::deep_copy(theta, theta_mirror);
  Kokkos::parallel_for(
      "plus2_coefficient_device_parity", point_count,
      KOKKOS_LAMBDA(const std::size_t point) {
        coefficients(point) = teuk::teukolsky_coefficients(
            parameters, radius(point), theta(point));
        scaling(point) = teuk::plus2_code_tetrad_scaling(
            parameters, radius(point), Kokkos::cos(theta(point)));
      });
  const auto device_coefficients = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, coefficients);
  const auto device_scaling = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, scaling);

  for (std::size_t point = 0; point < point_count; ++point) {
    const auto expected = teuk::test::ripley_eq22::coefficients(
        oracle_parameters(parameters), host_radius[point], host_theta[point]);
    CHECK_NEAR(device_coefficients(point).time, expected.time, 8.0e-13);
    CHECK_NEAR(device_coefficients(point).radial_advection,
               expected.radial_advection, 8.0e-13);
    CHECK_NEAR(device_coefficients(point).radial_principal,
               expected.radial_principal, 8.0e-13);
    CHECK_COMPLEX_NEAR(device_coefficients(point).definition,
                       as_teuk(expected.definition), 8.0e-13);
    CHECK_COMPLEX_NEAR(device_coefficients(point).q, as_teuk(expected.q),
                       8.0e-13);
    CHECK_COMPLEX_NEAR(device_coefficients(point).psi,
                       as_teuk(expected.psi), 8.0e-13);

    const double length2 = parameters.compactification_length *
                           parameters.compactification_length;
    const std::complex<double> denominator(
        length2, -parameters.spin * host_radius[point] *
                     std::cos(host_theta[point]));
    const std::complex<double> expected_scaling =
        std::pow(host_radius[point], 5) / std::pow(denominator, 4);
    CHECK_COMPLEX_NEAR(device_scaling(point), as_teuk(expected_scaling),
                       8.0e-13);
  }
}
