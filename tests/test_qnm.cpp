#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "teuk/rk4.hpp"
#include "teuk/teukolsky.hpp"

namespace {

struct RingdownFit {
  double frequency;
  double damping;
  double relative_recurrence_residual;
};

RingdownFit schwarzschild_quadrupole_ringdown() {
  constexpr std::size_t point_count = 129;
  const teuk::UniformRadialGrid grid(point_count, 0.0, 2.0);
  teuk::TeukolskyParameters parameters{1.0, 0.0, 2.0, -2, 2, 0.1};
  std::vector<teuk::TeukolskyState> state(point_count);
  std::vector<teuk::TeukolskyState> rhs(point_count);
  std::vector<teuk::Complex> angular(point_count);
  std::vector<teuk::Complex> forcing(point_count,
                                      teuk::Complex(0.0, 0.0));
  teuk::TeukolskyRadialWorkspace radial_workspace(point_count);
  teuk::RK4Workspace<teuk::TeukolskyState> rk_workspace(point_count);

  // Compact Gaussian with partial_T Psi4=0. The pure ell=m=2 angular action
  // is supplied independently as -4 Psi4.
  for (std::size_t radial = 0; radial < point_count; ++radial) {
    const double coordinate = grid.coordinate(radial);
    const double normalized = (coordinate - 1.35) / 0.16;
    const double psi = std::exp(-normalized * normalized);
    const double q = -2.0 * normalized * psi / 0.16;
    const auto coefficients =
        teuk::teukolsky_coefficients(parameters, coordinate, 1.0);
    state[radial].psi = teuk::Complex(psi, 0.0);
    state[radial].Q = teuk::Complex(q, 0.0);
    state[radial].P =
        -2.0 * coefficients.radial_advection * state[radial].Q +
        coefficients.definition * state[radial].psi;
  }
  const auto evaluate_rhs =
      [&](const double, const std::vector<teuk::TeukolskyState>& input,
          std::vector<teuk::TeukolskyState>& output) {
        for (std::size_t radial = 0; radial < point_count; ++radial) {
          angular[radial] = -4.0 * input[radial].psi;
        }
        teuk::evaluate_teukolsky_radial_line_rhs(
            grid, parameters, 1.0, input, angular, forcing,
            teuk::ReductionEvolution::FreeDamped, radial_workspace, output);
      };

  constexpr double time_step = 0.01;
  constexpr double sample_step = 0.05;
  constexpr int sample_every = 5;
  constexpr int step_count = 6500;
  constexpr std::size_t observation_index = 64;
  std::vector<double> times;
  std::vector<double> signal;
  for (int step = 0; step <= step_count; ++step) {
    if (step % sample_every == 0) {
      times.push_back(static_cast<double>(step) * time_step);
      signal.push_back(state[observation_index].psi.real());
    }
    if (step != step_count) {
      teuk::classical_rk4_step(state, static_cast<double>(step) * time_step,
                               time_step, evaluate_rhs, rk_workspace);
    }
  }

  // A real damped sinusoid obeys y[n+2]=c1*y[n+1]+c2*y[n]. Fit that
  // two-pole recurrence over the fundamental-mode window 45M--65M.
  constexpr double fit_start = 45.0;
  constexpr double fit_end = 65.0;
  double s11 = 0.0;
  double s12 = 0.0;
  double s22 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double signal_squared = 0.0;
  for (std::size_t sample = 0; sample + 2 < signal.size(); ++sample) {
    if (times[sample] < fit_start || times[sample + 2] > fit_end) continue;
    const double y0 = signal[sample];
    const double y1 = signal[sample + 1];
    const double y2 = signal[sample + 2];
    s11 += y1 * y1;
    s12 += y1 * y0;
    s22 += y0 * y0;
    b1 += y1 * y2;
    b2 += y0 * y2;
    signal_squared += y2 * y2;
  }
  const double determinant = s11 * s22 - s12 * s12;
  const double c1 = (b1 * s22 - b2 * s12) / determinant;
  const double c2 = (s11 * b2 - s12 * b1) / determinant;
  const double radius = std::sqrt(-c2);
  const double phase = std::acos(
      std::clamp(c1 / (2.0 * radius), -1.0, 1.0));
  double residual_squared = 0.0;
  for (std::size_t sample = 0; sample + 2 < signal.size(); ++sample) {
    if (times[sample] < fit_start || times[sample + 2] > fit_end) continue;
    const double residual = signal[sample + 2] - c1 * signal[sample + 1] -
                            c2 * signal[sample];
    residual_squared += residual * residual;
  }
  return {phase / sample_step, -std::log(radius) / sample_step,
          std::sqrt(residual_squared / signal_squared)};
}

}  // namespace

TEST_CASE("Schwarzschild quadrupole ringdown matches the trusted fundamental QNM") {
  // Fundamental gravitational Schwarzschild value in units M=1:
  // M omega = 0.37367 - 0.08896 i (standard tabulated QNM spectrum).
  const RingdownFit fit = schwarzschild_quadrupole_ringdown();
  CHECK(std::abs(fit.frequency - 0.37367168) < 8.0e-4);
  CHECK(std::abs(fit.damping - 0.08896232) < 8.0e-4);
  CHECK(fit.relative_recurrence_residual < 1.0e-5);
}
