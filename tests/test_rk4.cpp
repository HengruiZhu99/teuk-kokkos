#include "test_harness.hpp"

#include <cmath>
#include <vector>

#include "teuk/rk4.hpp"

namespace {

double exponential_error(const int steps) {
  std::vector<double> state{1.0};
  teuk::RK4Workspace<double> workspace(state.size());
  const double dt = 1.0 / static_cast<double>(steps);
  double time = 0.0;
  const auto rhs = [](double, const std::vector<double>& input,
                      std::vector<double>& output) { output[0] = input[0]; };
  for (int step = 0; step < steps; ++step) {
    teuk::classical_rk4_step(state, time, dt, rhs, workspace);
    time += dt;
  }
  return std::abs(state[0] - std::exp(1.0));
}

double coupled_error(const int steps) {
  // x'=x generates the time-dependent source y'=x^2.  Evaluating x and y as
  // one vector forces every source evaluation to use the same RK stage.
  std::vector<double> state{1.0, 0.0};
  teuk::RK4Workspace<double> workspace(state.size());
  const double dt = 1.0 / static_cast<double>(steps);
  double time = 0.0;
  const auto rhs = [](double, const std::vector<double>& input,
                      std::vector<double>& output) {
    output[0] = input[0];
    output[1] = input[0] * input[0];
  };
  for (int step = 0; step < steps; ++step) {
    teuk::classical_rk4_step(state, time, dt, rhs, workspace);
    time += dt;
  }
  const double x_error = std::abs(state[0] - std::exp(1.0));
  const double y_error =
      std::abs(state[1] - 0.5 * (std::exp(2.0) - 1.0));
  return std::max(x_error, y_error);
}

}  // namespace

TEST_CASE("classical RK4 has fourth-order autonomous convergence") {
  const double error_10 = exponential_error(10);
  const double error_20 = exponential_error(20);
  const double error_40 = exponential_error(40);
  CHECK(error_10 / error_20 > 14.0);
  CHECK(error_20 / error_40 > 14.5);
}

TEST_CASE("classical RK4 evaluates explicitly forced stages at stage times") {
  std::vector<double> state{0.0};
  teuk::RK4Workspace<double> workspace(state.size());
  const auto rhs = [](const double time, const std::vector<double>&,
                      std::vector<double>& output) {
    output[0] = 2.0 * time;
  };
  teuk::classical_rk4_step(state, 0.0, 0.7, rhs, workspace);
  CHECK_NEAR(state[0], 0.49, 2.0e-15);
}

TEST_CASE("coupled source-generating RK4 retains fourth order") {
  const double error_10 = coupled_error(10);
  const double error_20 = coupled_error(20);
  const double error_40 = coupled_error(40);
  CHECK(error_10 / error_20 > 13.0);
  CHECK(error_20 / error_40 > 14.0);
}

TEST_CASE("RK4 exposes reusable stage and accumulation primitives") {
  const std::vector<double> initial{1.0, -2.0};
  const std::vector<double> slope{3.0, 4.0};
  std::vector<double> stage(2);
  teuk::rk4_form_stage(initial, slope, 0.25, stage);
  CHECK_NEAR(stage[0], 1.75, 1.0e-15);
  CHECK_NEAR(stage[1], -1.0, 1.0e-15);

  std::vector<double> accumulated = initial;
  teuk::rk4_accumulate(0.6, slope, slope, slope, slope, accumulated);
  CHECK_NEAR(accumulated[0], 2.8, 1.0e-15);
  CHECK_NEAR(accumulated[1], 0.4, 1.0e-15);
}
