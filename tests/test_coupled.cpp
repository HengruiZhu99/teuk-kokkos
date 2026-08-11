#include <algorithm>
#include <vector>

#include "test_harness.hpp"
#include "teuk/coupled.hpp"
#include "teuk/rk4.hpp"

namespace {

teuk::PointPipelineState initial_state(const double amplitude) {
  teuk::PointPipelineState state;
  state.first.P = amplitude * teuk::Complex(0.8, -0.2);
  state.first.Q = 0.0;
  state.first.psi = amplitude * teuk::Complex(0.3, 0.4);
  state.reconstruction =
      amplitude * teuk::ReconstructionState{
                      {0.12, -0.04}, {-0.08, 0.03}, {0.05, 0.07},
                      {-0.02, 0.09}, {0.04, -0.06}, {0.03, 0.02},
                      {-0.07, -0.01}};
  return state;
}

teuk::PointPipelineParameters parameters() {
  teuk::PointPipelineParameters result;
  result.background = {1.0, 0.73, 1.4};
  result.radius = 0.37;
  result.theta = 0.91;
  result.ell = 2;
  result.reduction_damping = 0.2;
  return result;
}

teuk::PointPipelineState evolve(const double amplitude, const int steps) {
  auto state = initial_state(amplitude);
  std::vector<teuk::PointPipelineState> storage{state};
  teuk::RK4Workspace<teuk::PointPipelineState> workspace(1);
  const auto configured = parameters();
  const auto rhs = [&](double, const std::vector<teuk::PointPipelineState>& in,
                       std::vector<teuk::PointPipelineState>& out) {
    out[0] = teuk::evaluate_point_pipeline_rhs(in[0], configured);
  };
  constexpr double final_time = 0.08;
  const double dt = final_time / static_cast<double>(steps);
  double time = 0.0;
  for (int step = 0; step < steps; ++step) {
    teuk::classical_rk4_step(storage, time, dt, rhs, workspace);
    time += dt;
  }
  return storage[0];
}

double distance(const teuk::PointPipelineState& left,
                const teuk::PointPipelineState& right) {
  return std::max(
      {Kokkos::abs(left.first.P - right.first.P),
       Kokkos::abs(left.first.psi - right.first.psi),
       Kokkos::abs(left.reconstruction.G - right.reconstruction.G),
       Kokkos::abs(left.reconstruction.U - right.reconstruction.U),
       Kokkos::abs(left.second.P - right.second.P),
       Kokkos::abs(left.second.psi - right.second.psi)});
}

}  // namespace

TEST_CASE("point pipeline evaluates reconstruction source and second order together") {
  auto state = initial_state(1.0);
  teuk::PointPipelineDiagnostics diagnostics;
  const auto rhs =
      teuk::evaluate_point_pipeline_rhs(state, parameters(), &diagnostics);
  CHECK(Kokkos::abs(rhs.first.psi) > 1.0e-6);
  CHECK(Kokkos::abs(rhs.reconstruction.G) > 1.0e-6);
  CHECK(Kokkos::abs(diagnostics.inner_source.D) > 1.0e-6);
  CHECK(Kokkos::abs(diagnostics.forcing) > 1.0e-6);
  CHECK_COMPLEX_NEAR(rhs.second.P, diagnostics.forcing, 1.0e-14);
}

TEST_CASE("complete common-stage second-order pipeline converges at RK4 order") {
  const auto reference = evolve(1.0, 2048);
  const double error_2 = distance(evolve(1.0, 2), reference);
  const double error_4 = distance(evolve(1.0, 4), reference);
  const double error_8 = distance(evolve(1.0, 8), reference);
  CHECK(error_2 / error_4 > 15.0);
  CHECK(error_4 / error_8 > 15.0);
}

TEST_CASE("quadratic pipeline scales source and response as amplitude squared") {
  const auto unit = initial_state(1.0);
  const auto doubled = initial_state(2.0);
  teuk::PointPipelineDiagnostics unit_diagnostics;
  teuk::PointPipelineDiagnostics doubled_diagnostics;
  (void)teuk::evaluate_point_pipeline_rhs(unit, parameters(),
                                           &unit_diagnostics);
  (void)teuk::evaluate_point_pipeline_rhs(doubled, parameters(),
                                           &doubled_diagnostics);
  CHECK_COMPLEX_NEAR(doubled_diagnostics.forcing,
                     4.0 * unit_diagnostics.forcing, 2.0e-12);

  const auto evolved_unit = evolve(1.0, 256);
  const auto evolved_double = evolve(2.0, 256);
  CHECK_COMPLEX_NEAR(evolved_double.second.P, 4.0 * evolved_unit.second.P,
                     2.0e-11);
  CHECK_COMPLEX_NEAR(evolved_double.second.psi,
                     4.0 * evolved_unit.second.psi, 2.0e-11);
}
