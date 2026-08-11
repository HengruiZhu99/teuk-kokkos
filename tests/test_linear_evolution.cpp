#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "teuk/grid.hpp"
#include "teuk/rk4.hpp"
#include "teuk/teukolsky.hpp"

namespace {

using teuk::Complex;
using teuk::TeukolskyState;

struct ManufacturedProfile {
  Complex value;
  Complex first_derivative;
  Complex second_derivative;
};

Complex time_factor(const Complex& rate, const double time) {
  const std::complex<double> exponent(rate.real() * time,
                                      rate.imag() * time);
  const std::complex<double> value = std::exp(exponent);
  return {value.real(), value.imag()};
}

ManufacturedProfile polynomial_profile(const double radius) {
  const Complex amplitude(0.8, -0.3);
  const double r2 = radius * radius;
  const double r3 = r2 * radius;
  const double r4 = r2 * r2;
  const double value = 1.0 + 0.2 * radius - 0.11 * r2 + 0.07 * r3 -
                       0.025 * r4;
  const double first = 0.2 - 0.22 * radius + 0.21 * r2 - 0.1 * r3;
  const double second = -0.22 + 0.42 * radius - 0.3 * r2;
  return {amplitude * value, amplitude * first, amplitude * second};
}

ManufacturedProfile trigonometric_profile(const double radius) {
  const Complex amplitude(0.8, -0.3);
  constexpr double wave_number = 1.7;
  const double value = 1.0 + 0.2 * radius + std::sin(wave_number * radius);
  const double first = 0.2 + wave_number * std::cos(wave_number * radius);
  const double second =
      -wave_number * wave_number * std::sin(wave_number * radius);
  return {amplitude * value, amplitude * first, amplitude * second};
}

using ProfileFunction = ManufacturedProfile (*)(double);

TeukolskyState exact_state(const teuk::TeukolskyParameters& parameters,
                           const double theta, const double radius,
                           const double time, const Complex& growth_rate,
                           const ProfileFunction profile_function) {
  const ManufacturedProfile profile = profile_function(radius);
  const Complex factor = time_factor(growth_rate, time);
  const Complex psi = factor * profile.value;
  const Complex q = factor * profile.first_derivative;
  const Complex psi_t = growth_rate * psi;
  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, radius, theta);
  const Complex p = coefficients.time * psi_t -
                    2.0 * coefficients.radial_advection * q +
                    coefficients.definition * psi;
  return {p, q, psi};
}

Complex exact_forcing(const teuk::TeukolskyParameters& parameters,
                      const double theta, const double radius,
                      const double time, const Complex& growth_rate,
                      const double angular_eigenvalue,
                      const ProfileFunction profile_function) {
  const ManufacturedProfile profile = profile_function(radius);
  const Complex factor = time_factor(growth_rate, time);
  const TeukolskyState exact = exact_state(
      parameters, theta, radius, time, growth_rate, profile_function);
  const Complex p_t = growth_rate * exact.P;
  const Complex q_r = factor * profile.second_derivative;
  const Complex angular = angular_eigenvalue * exact.psi;
  const auto coefficients =
      teuk::teukolsky_coefficients(parameters, radius, theta);
  return p_t - coefficients.radial_principal * q_r -
         coefficients.q * exact.Q - coefficients.psi * exact.psi - angular;
}

struct EvolutionResult {
  double state_error;
  double constraint_rms;
};

EvolutionResult evolve_manufactured_solution(
    const std::size_t point_count, const int step_count,
    const ProfileFunction profile_function) {
  const teuk::UniformRadialGrid grid(point_count, 0.0, 0.8);
  teuk::TeukolskyParameters parameters;
  parameters.mass = 1.0;
  parameters.spin = 0.37;
  parameters.compactification_length = 2.0;
  parameters.spin_weight = -2;
  parameters.azimuthal_mode = -1;
  parameters.reduction_damping = 0.0;
  constexpr double theta = 0.83;
  constexpr double final_time = 0.08;
  constexpr double angular_eigenvalue = -4.0;
  const Complex growth_rate(-0.31, 0.43);

  std::vector<TeukolskyState> state(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    state[i] = exact_state(parameters, theta, grid.coordinate(i), 0.0,
                           growth_rate, profile_function);
  }

  std::vector<Complex> angular(point_count);
  std::vector<Complex> forcing(point_count);
  std::vector<TeukolskyState> rhs(point_count);
  teuk::TeukolskyRadialWorkspace radial_workspace(point_count);
  teuk::RK4Workspace<TeukolskyState> rk_workspace(point_count);
  const auto evaluate_rhs =
      [&](const double stage_time, const std::vector<TeukolskyState>& input,
          std::vector<TeukolskyState>& output) {
        for (std::size_t i = 0; i < point_count; ++i) {
          const double radius = grid.coordinate(i);
          angular[i] = angular_eigenvalue * input[i].psi;
          forcing[i] = exact_forcing(parameters, theta, radius, stage_time,
                                     growth_rate, angular_eigenvalue,
                                     profile_function);
        }
        teuk::evaluate_teukolsky_radial_line_rhs(
            grid, parameters, theta, input, angular, forcing,
            teuk::ReductionEvolution::FreeDamped, radial_workspace,
            output);
      };

  const double dt = final_time / static_cast<double>(step_count);
  double time = 0.0;
  for (int step = 0; step < step_count; ++step) {
    teuk::classical_rk4_step(state, time, dt, evaluate_rhs, rk_workspace);
    time += dt;
  }

  double maximum_error = 0.0;
  for (std::size_t i = 0; i < point_count; ++i) {
    const TeukolskyState exact = exact_state(
        parameters, theta, grid.coordinate(i), final_time, growth_rate,
        profile_function);
    maximum_error = std::max(maximum_error, Kokkos::abs(state[i].P - exact.P));
    maximum_error = std::max(maximum_error, Kokkos::abs(state[i].Q - exact.Q));
    maximum_error =
        std::max(maximum_error, Kokkos::abs(state[i].psi - exact.psi));
  }
  std::vector<Complex> psi_buffer(point_count);
  std::vector<Complex> derivative_buffer(point_count);
  return {maximum_error, teuk::reduction_constraint_rms(
                             grid, state, psi_buffer, derivative_buffer)};
}

}  // namespace

TEST_CASE("manufactured linear Teukolsky evolution converges fourth order in time") {
  // The quartic profile and its derivatives are exact under every radial
  // closure, isolating repeated full-PDE RK stages from spatial truncation.
  const EvolutionResult coarse =
      evolve_manufactured_solution(17, 2, polynomial_profile);
  const EvolutionResult medium =
      evolve_manufactured_solution(17, 4, polynomial_profile);
  const EvolutionResult fine =
      evolve_manufactured_solution(17, 8, polynomial_profile);
  CHECK(coarse.state_error / medium.state_error > 14.0);
  CHECK(medium.state_error / fine.state_error > 14.0);
  CHECK(fine.constraint_rms < 2.0e-12);
}

TEST_CASE("manufactured evolution preserves fourth-order reduction convergence") {
  // This non-polynomial profile exercises both one-sided closures. The
  // reduction residual has the fourth-order convergence expected from the
  // explicit reference derivative, even though a production spatial gate also
  // requires SBP/SAT boundary evolution that is outside this test.
  const EvolutionResult coarse =
      evolve_manufactured_solution(17, 32, trigonometric_profile);
  const EvolutionResult medium =
      evolve_manufactured_solution(33, 64, trigonometric_profile);
  const EvolutionResult fine =
      evolve_manufactured_solution(65, 128, trigonometric_profile);
  CHECK(coarse.constraint_rms / medium.constraint_rms > 14.0);
  CHECK(medium.constraint_rms / fine.constraint_rms > 14.0);
  CHECK(coarse.state_error > medium.state_error);
  CHECK(medium.state_error > fine.state_error);
  CHECK(fine.constraint_rms < 2.0e-5);
}
