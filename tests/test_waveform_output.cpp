#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/modes.hpp"
#include "teuk/pipeline_storage.hpp"
#include "teuk/waveform_output.hpp"

TEST_CASE("endpoint waveform sampler recovers complex first and second modes") {
  constexpr int theta_nodes = 12;
  const teuk::ModeRegistry registry({-4, -2, 0, 2, 4}, {-2, 2}, {-4, 0, 4});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  teuk::SpatialPipelineStorage storage(
      registry, radial_grid, teuk::angular::gauss_legendre(theta_nodes),
      "waveform_test_storage");
  auto host = Kokkos::create_mirror_view(storage.state());
  Kokkos::deep_copy(host, teuk::Complex(0.0, 0.0));

  const auto fill = [&](const int order, const int m, const int ell,
                        const teuk::Complex scri,
                        const teuk::Complex horizon) {
    const int ell_max = order == 1 ? 4 : 6;
    const teuk::angular::SpinWeightedTransform transform(
        -2, m, ell_max, theta_nodes);
    std::vector<teuk::Complex> modal(transform.mode_count(),
                                     teuk::Complex(0.0, 0.0));
    const std::size_t field = static_cast<std::size_t>(
        order == 1 ? teuk::PipelineField::FirstPsi
                   : teuk::PipelineField::SecondPsi);
    for (const auto& [radial, coefficient] :
         {std::pair<std::size_t, teuk::Complex>{0, scri},
          {radial_grid.size() - 1, horizon}}) {
      modal[static_cast<std::size_t>(ell - transform.ell_min())] = coefficient;
      const auto nodal = transform.synthesize(modal);
      for (std::size_t theta = 0; theta < nodal.size(); ++theta) {
        host(registry.index(m), field, radial, theta) = nodal[theta];
      }
      modal[static_cast<std::size_t>(ell - transform.ell_min())] =
          teuk::Complex(0.0, 0.0);
    }
  };
  fill(1, 2, 3, teuk::Complex(0.4, -0.2),
       teuk::Complex(-0.1, 0.3));
  fill(2, 0, 4, teuk::Complex(-0.05, 0.07),
       teuk::Complex(0.08, 0.02));
  Kokkos::deep_copy(storage.state(), host);

  const teuk::ExecutionSpace execution;
  teuk::EndpointWaveformSampler sampler(
      registry, {4, 6}, theta_nodes, 1.5, radial_grid.upper_radius());
  const auto records = sampler.sample(execution, storage.state(), 17, 2.5);
  const auto find = [&](const int order, const int m, const int ell,
                        const teuk::WaveformBoundary boundary) {
    return std::find_if(records.begin(), records.end(), [&](const auto& value) {
      return value.perturbative_order == order && value.m == m &&
             value.ell == ell && value.boundary == boundary;
    });
  };
  const auto first_scri = find(1, 2, 3, teuk::WaveformBoundary::Scri);
  const auto first_horizon =
      find(1, 2, 3, teuk::WaveformBoundary::Horizon);
  const auto second_scri = find(2, 0, 4, teuk::WaveformBoundary::Scri);
  CHECK(first_scri != records.end());
  CHECK(first_horizon != records.end());
  CHECK(second_scri != records.end());
  CHECK_COMPLEX_NEAR(first_scri->rescaled, teuk::Complex(0.4, -0.2),
                     2.0e-12);
  CHECK_COMPLEX_NEAR(first_scri->endpoint_waveform,
                     2.25 * teuk::Complex(0.4, -0.2), 4.0e-12);
  CHECK_COMPLEX_NEAR(first_horizon->endpoint_waveform,
                     0.8 * teuk::Complex(-0.1, 0.3), 3.0e-12);
  CHECK_COMPLEX_NEAR(second_scri->rescaled,
                     teuk::Complex(-0.05, 0.07), 2.0e-12);

  std::ostringstream output;
  teuk::write_endpoint_waveform_header(output);
  teuk::write_endpoint_waveform_records(output, records);
  CHECK(output.str().find("step,time,boundary,order,m,ell") == 0);
  CHECK(output.str().find("17,2.5,scri,1,2,3") != std::string::npos);
}
