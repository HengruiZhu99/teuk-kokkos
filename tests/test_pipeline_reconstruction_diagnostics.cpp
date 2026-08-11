#include "test_harness.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "teuk/pipeline_reconstruction_diagnostics.hpp"

TEST_CASE("full pipeline transport-equation consistency residuals close") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid grid(9, 0.0, 0.8);
  teuk::SpatialPipeline pipeline(
      execution, registry, grid, 3, 5,
      teuk::KerrParameters{1.0, 0.0, 1.5}, 0.1, 0.0);
  auto host = Kokkos::create_mirror_view(pipeline.storage().state());
  constexpr std::array<int, teuk::point_pipeline_field_count> spins{
      -2, -2, -2, -1, -2, 0, -2, -1, -1, 0, -2, -2, -2};
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
         ++field) {
      const teuk::angular::SpinWeightedTransform transform(
          spins[field], registry.modes()[mode], 3, 5);
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        std::vector<teuk::Complex> modal(transform.mode_count(),
                                         teuk::Complex(0.0, 0.0));
        const double seed = 1.0 + 0.1 * mode + 0.03 * field +
                            0.02 * radial;
        modal[static_cast<std::size_t>(3 - transform.ell_min())] =
            teuk::Complex(1.0e-4 * seed, -6.0e-5 * seed);
        const auto nodal = transform.synthesize(modal);
        for (std::size_t theta = 0; theta < 5; ++theta) {
          host(mode, field, radial, theta) =
              nodal[theta];
        }
      }
    }
  }
  Kokkos::deep_copy(execution, pipeline.storage().state(), host);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  teuk::PipelineTransportEquationDiagnostics diagnostic(registry.size(),
                                                          grid.size(), 5);
  const auto closed = diagnostic.sample(execution, pipeline);
  CHECK(closed.combined.maximum < 3.0e-12);

  const auto rhs = pipeline.storage().rhs();
  Kokkos::parallel_for(
      "corrupt_one_reconstruction_rhs", 1, KOKKOS_LAMBDA(const int) {
        rhs(1, static_cast<std::size_t>(teuk::PipelineField::G), 4, 2) +=
            teuk::Complex(0.25, -0.1);
      });
  const auto opened = diagnostic.sample(execution, pipeline);
  CHECK(opened.G.maximum > 0.1);
  CHECK(opened.Lambda.maximum < 3.0e-12);
}
