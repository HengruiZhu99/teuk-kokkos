#include "test_harness.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "teuk/diagnostics.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/rk4.hpp"
#include "teuk/spatial_pipeline.hpp"
#include "teuk/teukolsky.hpp"

namespace {

struct RingdownFit {
  double frequency;
  double damping;
  double relative_recurrence_residual;
};

struct ProductionRingdownSetup {
  std::size_t radial_points;
  int ell_max;
  int theta_nodes;
  double spin;
  double fit_start;
  double fit_end;
};

RingdownFit fit_real_damped_ringdown(const std::vector<double>& times,
                                     const std::vector<double>& signal,
                                     const double sample_step,
                                     const double fit_start,
                                     const double fit_end) {
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
  const double phase =
      std::acos(std::clamp(c1 / (2.0 * radius), -1.0, 1.0));
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

RingdownFit fit_complex_two_pole_ringdown(
    const std::vector<double>& times,
    const std::vector<std::complex<double>>& signal, const double sample_step,
    const double fit_start, const double fit_end) {
  std::complex<double> h11(0.0, 0.0);
  std::complex<double> h12(0.0, 0.0);
  std::complex<double> h22(0.0, 0.0);
  std::complex<double> g1(0.0, 0.0);
  std::complex<double> g2(0.0, 0.0);
  double signal_squared = 0.0;
  for (std::size_t sample = 0; sample + 2 < signal.size(); ++sample) {
    if (times[sample] < fit_start || times[sample + 2] > fit_end) continue;
    const std::complex<double> y0 = signal[sample];
    const std::complex<double> y1 = signal[sample + 1];
    const std::complex<double> y2 = signal[sample + 2];
    h11 += std::conj(y1) * y1;
    h12 += std::conj(y1) * y0;
    h22 += std::conj(y0) * y0;
    g1 += std::conj(y1) * y2;
    g2 += std::conj(y0) * y2;
    signal_squared += std::norm(y2);
  }
  const std::complex<double> determinant =
      h11 * h22 - h12 * std::conj(h12);
  if (std::abs(determinant) <
      100.0 * std::numeric_limits<double>::epsilon() *
          std::abs(h11 * h22)) {
    throw std::runtime_error("QNM two-pole fit is rank deficient");
  }
  const std::complex<double> c1 = (g1 * h22 - h12 * g2) / determinant;
  const std::complex<double> c2 =
      (h11 * g2 - std::conj(h12) * g1) / determinant;
  const std::complex<double> discriminant =
      std::sqrt(c1 * c1 + 4.0 * c2);
  const std::array<std::complex<double>, 2> roots{
      0.5 * (c1 + discriminant), 0.5 * (c1 - discriminant)};

  // With the repository's exp(i m phi) convention, the physical positive
  // frequency branch has a negative phase per forward time sample.
  std::size_t selected = 0;
  double selected_frequency = -std::arg(roots[0]) / sample_step;
  const double second_frequency = -std::arg(roots[1]) / sample_step;
  if (selected_frequency <= 0.0 && second_frequency > 0.0) {
    selected = 1;
    selected_frequency = second_frequency;
  } else if (selected_frequency > 0.0 && second_frequency > 0.0) {
    const double first_damping = -std::log(std::abs(roots[0])) / sample_step;
    const double second_damping = -std::log(std::abs(roots[1])) / sample_step;
    if (second_damping < first_damping) {
      selected = 1;
      selected_frequency = second_frequency;
    }
  }
  const double damping = -std::log(std::abs(roots[selected])) / sample_step;

  double residual_squared = 0.0;
  for (std::size_t sample = 0; sample + 2 < signal.size(); ++sample) {
    if (times[sample] < fit_start || times[sample + 2] > fit_end) continue;
    const std::complex<double> residual =
        signal[sample + 2] - c1 * signal[sample + 1] - c2 * signal[sample];
    residual_squared += std::norm(residual);
  }
  return {selected_frequency, damping,
          std::sqrt(residual_squared / signal_squared)};
}

RingdownFit production_pipeline_ringdown(
    const ProductionRingdownSetup& setup) {
  constexpr double mass = 1.0;
  constexpr double compactification_length = 2.0;
  constexpr double time_step = 0.01;
  constexpr double sample_step = 0.05;
  constexpr int sample_every = 5;
  const double final_time = setup.fit_end;
  const int step_count = static_cast<int>(std::llround(final_time / time_step));
  const double horizon = compactification_length * compactification_length /
                         teuk::outer_horizon_radius(mass, setup.spin);

  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2}, {-2, 2}, {-2, 2});
  const teuk::UniformRadialGrid grid(setup.radial_points, 0.0, horizon);
  const teuk::KerrParameters background{mass, setup.spin,
                                        compactification_length};
  teuk::SpatialPipeline pipeline(
      execution, registry, grid,
      teuk::PipelineAngularBands{setup.ell_max, setup.ell_max},
      setup.theta_nodes, background, 0.1, 0.005,
      teuk::ReductionEvolution::FreeDamped, "production_qnm",
      teuk::SecondOrderSourcePolicy::disabled());
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.675 * horizon;
  pulse.width = 0.08 * horizon;
  pulse.modes = {{2, 2, teuk::Complex(1.0, 0.0)}};
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, setup.ell_max, background, pulse);

  const std::size_t mode_index = registry.index(2);
  const std::size_t field_index =
      static_cast<std::size_t>(teuk::PipelineField::FirstPsi);
  const std::size_t observation_index =
      static_cast<std::size_t>(std::llround(
          (1.0 / horizon) * static_cast<double>(setup.radial_points - 1)));
  const teuk::angular::SpinWeightedTransform transform(
      -2, 2, setup.ell_max, setup.theta_nodes);
  Kokkos::View<teuk::Complex*, Kokkos::HostSpace> host_line(
      "production_qnm_host_line", static_cast<std::size_t>(setup.theta_nodes));
  std::vector<teuk::Complex> nodal(
      static_cast<std::size_t>(setup.theta_nodes));
  std::vector<double> times;
  std::vector<std::complex<double>> signal;
  times.reserve(static_cast<std::size_t>(step_count / sample_every + 1));
  signal.reserve(times.capacity());
  for (int step = 0; step <= step_count; ++step) {
    const double time = static_cast<double>(step) * time_step;
    if (step % sample_every == 0) {
      const auto device_line = Kokkos::subview(
          pipeline.storage().state(), mode_index, field_index,
          observation_index, Kokkos::ALL);
      Kokkos::deep_copy(host_line, device_line);
      execution.fence("sample production QNM angular line");
      for (std::size_t node = 0; node < nodal.size(); ++node) {
        nodal[node] = host_line(node);
      }
      const auto modal = transform.analyze(nodal);
      times.push_back(time);
      signal.emplace_back(modal.front().real(), modal.front().imag());
    }
    if (step != step_count) pipeline.step(execution, time, time_step);
  }
  execution.fence("complete production QNM evolution");
  if (setup.spin == 0.0) {
    std::vector<double> real_signal(signal.size());
    std::transform(signal.begin(), signal.end(), real_signal.begin(),
                   [](const auto value) { return value.real(); });
    return fit_real_damped_ringdown(times, real_signal, sample_step,
                                    setup.fit_start, setup.fit_end);
  }
  return fit_complex_two_pole_ringdown(times, signal, sample_step,
                                       setup.fit_start, setup.fit_end);
}

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

TEST_CASE("production pipeline QNMs converge in radial and angular resolution") {
  constexpr double schwarzschild_frequency = 0.37367168;
  constexpr double schwarzschild_damping = 0.08896232;
  // Independent Leaver-solver value for s=-2, ell=m=2, n=0, a/M=0.7:
  //   M omega = 0.5326002435510186 - 0.08079287315500745 i.
  // Sources and data provenance:
  //   https://pages.jh.edu/eberti2/ringdown/
  //   https://arxiv.org/abs/gr-qc/0512160
  //   https://arxiv.org/abs/1908.10377
  // The last source's qnm 0.4.4 Leaver implementation was evaluated at
  // exactly a/M=0.7; no fit coefficient is used as the regression target.
  constexpr double kerr_frequency = 0.5326002435510186;
  constexpr double kerr_damping = 0.08079287315500745;
  const std::array<ProductionRingdownSetup, 7> setups{{
      {17, 3, 5, 0.0, 45.0, 65.0},
      {25, 3, 5, 0.0, 45.0, 65.0},
      {33, 3, 5, 0.0, 45.0, 65.0},
      {33, 4, 7, 0.0, 45.0, 65.0},
      {65, 3, 5, 0.7, 55.0, 80.0},
      {65, 4, 7, 0.7, 55.0, 80.0},
      {65, 5, 8, 0.7, 55.0, 80.0},
  }};
  std::array<RingdownFit, setups.size()> fits{};
  for (std::size_t index = 0; index < setups.size(); ++index) {
    const auto& setup = setups[index];
    fits[index] = production_pipeline_ringdown(setup);
    const RingdownFit fit = fits[index];
    std::cout << "[QNM] a=" << setup.spin << " nr=" << setup.radial_points
              << " ellmax=" << setup.ell_max
              << " ntheta=" << setup.theta_nodes
              << " omega=" << fit.frequency
              << " damping=" << fit.damping
              << " residual=" << fit.relative_recurrence_residual << '\n';
    CHECK(std::isfinite(fit.frequency));
    CHECK(std::isfinite(fit.damping));
    CHECK(std::isfinite(fit.relative_recurrence_residual));
  }

  const auto frequency_error = [](const RingdownFit fit,
                                  const double trusted_frequency,
                                  const double trusted_damping) {
    return std::hypot(fit.frequency - trusted_frequency,
                      fit.damping - trusted_damping);
  };
  const double schwarzschild_coarse = frequency_error(
      fits[0], schwarzschild_frequency, schwarzschild_damping);
  const double schwarzschild_medium = frequency_error(
      fits[1], schwarzschild_frequency, schwarzschild_damping);
  const double schwarzschild_fine = frequency_error(
      fits[2], schwarzschild_frequency, schwarzschild_damping);
  CHECK(schwarzschild_medium < 0.25 * schwarzschild_coarse);
  CHECK(schwarzschild_fine < 0.70 * schwarzschild_medium);
  CHECK(schwarzschild_fine < 3.0e-4);
  CHECK(fits[2].relative_recurrence_residual <
        fits[1].relative_recurrence_residual);
  CHECK(fits[1].relative_recurrence_residual <
        fits[0].relative_recurrence_residual);

  // A pure spherical ell=2 Schwarzschild mode is exactly represented by both
  // retained angular configurations; changing (ellmax,ntheta)=(3,5)->(4,7)
  // must leave the measured production waveform unchanged.
  CHECK_NEAR(fits[3].frequency, fits[2].frequency, 1.0e-8);
  CHECK_NEAR(fits[3].damping, fits[2].damping, 1.0e-8);
  CHECK(frequency_error(fits[3], schwarzschild_frequency,
                        schwarzschild_damping) < 3.0e-4);

  // Kerr's spheroidal mode requires a spherical-harmonic sequence.  Demand
  // self-convergence of the complete production path and agreement of the
  // converged complex frequency with the independent Leaver value.
  const double kerr_coarse_to_medium =
      std::hypot(fits[5].frequency - fits[4].frequency,
                 fits[5].damping - fits[4].damping);
  const double kerr_medium_to_fine =
      std::hypot(fits[6].frequency - fits[5].frequency,
                 fits[6].damping - fits[5].damping);
  CHECK(kerr_medium_to_fine < 0.05 * kerr_coarse_to_medium);
  CHECK(frequency_error(fits[6], kerr_frequency, kerr_damping) < 7.0e-4);
  CHECK(std::abs(fits[6].frequency - kerr_frequency) < 5.0e-4);
  CHECK(std::abs(fits[6].damping - kerr_damping) < 5.0e-4);
  CHECK(fits[6].relative_recurrence_residual < 3.0e-6);
}
