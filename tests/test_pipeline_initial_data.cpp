#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "test_harness.hpp"
#include "teuk/angular.hpp"
#include "teuk/pipeline_initial_data.hpp"
#include "teuk/sbp.hpp"
#include "teuk/spatial_pipeline.hpp"

namespace {

int initial_data_copies = 0;
std::uint64_t initial_data_state_bytes = 0;

void count_initial_data_copy(Kokkos::Tools::SpaceHandle, const char*,
                             const void*, Kokkos::Tools::SpaceHandle,
                             const char*, const void*, std::uint64_t bytes) {
  if (bytes == initial_data_state_bytes) ++initial_data_copies;
}

}  // namespace

TEST_CASE("pipeline Gaussian initial data are bandlimited consistent and scalable") {
  constexpr int ell_max = 3;
  constexpr int theta_nodes = 6;
  const teuk::ModeRegistry registry({2, -2, 0});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.6, 1.2};
  const teuk::ExecutionSpace execution;
  teuk::SpatialPipeline pipeline(execution, registry, radial_grid, ell_max,
                                 theta_nodes, background, 0.1, 0.0,
                                 teuk::ReductionEvolution::FreeDamped,
                                 "initial_data_pipeline");
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.14;
  pulse.modes = {{3, 2, teuk::Complex(0.7, -0.2)},
                 {3, -2, teuk::Complex(-0.35, 0.5)}};

  initial_data_copies = 0;
  initial_data_state_bytes =
      pipeline.storage().value_count() * sizeof(teuk::Complex);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_initial_data_copy);
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, ell_max, background, pulse);
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  CHECK(initial_data_copies == 1);
  // Use an owning host snapshot even on Serial, where
  // create_mirror_view_and_copy may legally return an alias.
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      initial("initial_data_snapshot", registry.size(),
              teuk::point_pipeline_field_count, radial_grid.size(),
              theta_nodes);
  Kokkos::deep_copy(initial, pipeline.storage().state());

  const auto psi = static_cast<std::size_t>(teuk::PipelineField::FirstPsi);
  const auto q = static_cast<std::size_t>(teuk::PipelineField::FirstQ);
  const auto p = static_cast<std::size_t>(teuk::PipelineField::FirstP);
  const std::size_t center_index = 4;
  for (std::size_t mode_index = 0; mode_index < registry.size(); ++mode_index) {
    const int m = registry.modes()[mode_index];
    const teuk::angular::SpinWeightedTransform transform(
        -2, m, ell_max, theta_nodes);
    std::vector<teuk::Complex> nodal(theta_nodes);
    for (int node = 0; node < theta_nodes; ++node) {
      nodal[static_cast<std::size_t>(node)] =
          initial(mode_index, psi, center_index,
                  static_cast<std::size_t>(node));
    }
    const auto modal = transform.analyze(nodal);
    for (int ell = transform.ell_min(); ell <= ell_max; ++ell) {
      teuk::Complex expected(0.0, 0.0);
      for (const auto& seed : pulse.modes) {
        if (seed.m == m && seed.ell == ell) expected = seed.amplitude;
      }
      CHECK_COMPLEX_NEAR(
          modal[static_cast<std::size_t>(ell - transform.ell_min())], expected,
          2e-12);
    }

    for (const std::size_t field : {p, q, psi}) {
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (int node = 0; node < theta_nodes; ++node) {
          nodal[static_cast<std::size_t>(node)] = initial(
              mode_index, field, radial, static_cast<std::size_t>(node));
        }
        const auto projected = transform.synthesize(transform.analyze(nodal));
        for (int node = 0; node < theta_nodes; ++node) {
          CHECK_COMPLEX_NEAR(nodal[static_cast<std::size_t>(node)],
                             projected[static_cast<std::size_t>(node)],
                             3e-12);
        }
      }
    }

    for (int node = 0; node < theta_nodes; ++node) {
      const auto theta = static_cast<std::size_t>(node);
      std::vector<teuk::Complex> psi_line(radial_grid.size());
      std::vector<teuk::Complex> derivative(radial_grid.size());
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        psi_line[radial] = initial(mode_index, psi, radial, theta);
      }
      teuk::d42_first_derivative(radial_grid, psi_line, derivative);
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        CHECK_COMPLEX_NEAR(initial(mode_index, q, radial, theta),
                           derivative[radial], 2e-14);
      }
    }

    // In Kerr, the pointwise time derivative may contain discarded angular
    // content. Its retained Galerkin coefficients must vanish instead.
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      for (int node = 0; node < theta_nodes; ++node) {
        const auto theta = static_cast<std::size_t>(node);
        teuk::TeukolskyParameters parameters;
        parameters.mass = background.mass;
        parameters.spin = background.spin;
        parameters.compactification_length =
            background.compactification_length;
        parameters.spin_weight = -2;
        parameters.azimuthal_mode = m;
        const auto coefficients = teuk::teukolsky_coefficients(
            parameters, radial_grid.coordinate(radial),
            teuk::angular::gauss_legendre(theta_nodes).theta(theta));
        nodal[theta] = teuk::teukolsky_psi_rhs(
            coefficients,
            {initial(mode_index, p, radial, theta),
             initial(mode_index, q, radial, theta),
             initial(mode_index, psi, radial, theta)});
      }
      const auto modal_dt = transform.analyze(nodal);
      for (const auto coefficient : modal_dt) {
        CHECK_COMPLEX_NEAR(coefficient, teuk::Complex(0.0, 0.0), 8e-13);
      }
    }
  }

  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field =
             static_cast<std::size_t>(teuk::PipelineField::G);
         field <= static_cast<std::size_t>(teuk::PipelineField::SecondPsi);
         ++field) {
      if (field == static_cast<std::size_t>(teuk::PipelineField::SecondP) ||
          field == static_cast<std::size_t>(teuk::PipelineField::SecondQ) ||
          field == static_cast<std::size_t>(teuk::PipelineField::SecondPsi) ||
          (field >= static_cast<std::size_t>(teuk::PipelineField::G) &&
           field <= static_cast<std::size_t>(teuk::PipelineField::U))) {
        for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
          for (int node = 0; node < theta_nodes; ++node) {
            CHECK_COMPLEX_NEAR(
                initial(mode, field, radial, static_cast<std::size_t>(node)),
                teuk::Complex(0.0, 0.0), 0.0);
          }
        }
      }
    }
  }

  // Reinitialize in place with scaled amplitudes and optional nonlinear seeds.
  const double amplitude_scale = 2.5;
  for (auto& seed : pulse.modes) seed.amplitude *= amplitude_scale;
  pulse.second_order_scale = teuk::Complex(0.4, -0.1);
  pulse.reconstruction_scales[static_cast<std::size_t>(
      teuk::ReconstructionField::B)] = teuk::Complex(0.2, 0.0);
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, ell_max, background, pulse);
  const auto scaled = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().state());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      for (int node = 0; node < theta_nodes; ++node) {
        const auto theta = static_cast<std::size_t>(node);
        for (const std::size_t first_field : {p, q, psi}) {
          CHECK_COMPLEX_NEAR(
              scaled(mode, first_field, radial, theta),
              amplitude_scale *
                  initial(mode, first_field, radial, theta),
              8e-13);
        }
        CHECK_COMPLEX_NEAR(
            scaled(mode,
                   static_cast<std::size_t>(teuk::PipelineField::SecondPsi),
                   radial, theta),
            pulse.second_order_scale * scaled(mode, psi, radial, theta),
            3e-13);
      }
    }
  }
  CHECK(Kokkos::abs(scaled(
            registry.index(2),
            static_cast<std::size_t>(teuk::PipelineField::B), center_index,
            0)) > 1e-6);
}

TEST_CASE("pipeline Gaussian initial data reject invalid modes and sharp storage") {
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.4, 1.1};
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.1;
  pulse.modes = {{2, 2, teuk::Complex(1.0, 0.0)}};

  bool sharp_rejected = false;
  try {
    const teuk::ModeRegistry incomplete({0, 2});
    teuk::SpatialPipelineStorage storage(
        incomplete, radial_grid, teuk::angular::gauss_legendre(6));
    static_cast<void>(storage);
  } catch (const std::invalid_argument&) {
    sharp_rejected = true;
  }
  CHECK(sharp_rejected);

  const teuk::ModeRegistry registry({-2, 0, 2});
  const teuk::ExecutionSpace execution;
  teuk::SpatialPipeline pipeline(execution, registry, radial_grid, 3, 6,
                                 background, 0.1, 0.0,
                                 teuk::ReductionEvolution::FreeDamped,
                                 "invalid_initial_data_pipeline");
  pulse.modes[0].ell = 4;
  bool ell_rejected = false;
  try {
    teuk::initialize_compactified_gaussian_pulse(
        execution, pipeline, registry, 3, background, pulse);
  } catch (const std::invalid_argument&) {
    ell_rejected = true;
  }
  CHECK(ell_rejected);
}

TEST_CASE("D8-4 pipeline uses one radial scheme for Gaussian Q and reconstruction") {
  constexpr int ell_max = 3;
  constexpr int theta_nodes = 6;
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid grid(17, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.35, 1.2};
  teuk::SpatialPipeline pipeline(
      execution, registry, grid, ell_max, theta_nodes, background, 0.1, 0.0,
      teuk::ReductionEvolution::FreeDamped, "d84_initial_data_pipeline", {},
      teuk::RadialDiscretization::D84);
  CHECK(pipeline.radial_discretization() == teuk::RadialDiscretization::D84);

  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.17;
  pulse.modes = {{2, 2, teuk::Complex(0.4, -0.1)}};
  pulse.reconstruction_scales[static_cast<std::size_t>(
      teuk::ReconstructionField::G)] = teuk::Complex(0.25, 0.05);
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, ell_max, background, pulse);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  execution.fence("sample D8-4 initial and reconstruction derivatives");

  const auto state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().state());
  const auto reconstruction = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.reconstruction_radial_derivatives());
  constexpr std::size_t psi =
      static_cast<std::size_t>(teuk::PipelineField::FirstPsi);
  constexpr std::size_t q =
      static_cast<std::size_t>(teuk::PipelineField::FirstQ);
  constexpr std::size_t g = static_cast<std::size_t>(teuk::PipelineField::G);
  constexpr std::size_t reconstruction_g = 0;
  const double inverse_spacing = 1.0 / grid.spacing();
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t theta = 0; theta < theta_nodes; ++theta) {
      for (std::size_t radial = 0; radial < grid.size(); ++radial) {
        const auto psi_dr = teuk::radial_first_derivative_strided_at(
            teuk::RadialDiscretization::D84,
            &state(mode, psi, 0, theta), grid.size(), radial, inverse_spacing,
            state.stride(2));
        const auto g_dr = teuk::radial_first_derivative_strided_at(
            teuk::RadialDiscretization::D84,
            &state(mode, g, 0, theta), grid.size(), radial, inverse_spacing,
            state.stride(2));
        CHECK_COMPLEX_NEAR(state(mode, q, radial, theta), psi_dr, 1.0e-13);
        CHECK_COMPLEX_NEAR(
            reconstruction(mode, reconstruction_g, radial, theta), g_dr,
            2.0e-14);
      }
    }
  }
}

TEST_CASE("compact Gaussian pulse is smooth inside and exactly support limited") {
  constexpr int ell_max = 3;
  constexpr int theta_nodes = 6;
  const teuk::ModeRegistry registry({-2, 2});
  const teuk::UniformRadialGrid radial_grid(17, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.5, 1.2};
  const teuk::ExecutionSpace execution;
  teuk::SpatialPipeline pipeline(execution, registry, radial_grid, ell_max,
                                 theta_nodes, background, 0.1, 0.0,
                                 teuk::ReductionEvolution::FreeDamped,
                                 "compact_initial_data_pipeline");
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.2;
  pulse.compact_support = true;
  pulse.modes = {{2, 2, teuk::Complex(0.7, -0.1)}};
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, ell_max, background, pulse);
  const auto state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().state());
  const std::size_t psi =
      static_cast<std::size_t>(teuk::PipelineField::FirstPsi);
  const std::size_t mode = registry.index(2);

  for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
    const double coordinate = radial_grid.coordinate(radial);
    const bool outside = coordinate <= pulse.center - pulse.width ||
                         coordinate >= pulse.center + pulse.width;
    for (int theta = 0; theta < theta_nodes; ++theta) {
      if (outside) {
        CHECK_COMPLEX_NEAR(
            state(mode, psi, radial, static_cast<std::size_t>(theta)),
            teuk::Complex(0.0, 0.0), 0.0);
      }
    }
  }
  double center_maximum = 0.0;
  for (int theta = 0; theta < theta_nodes; ++theta) {
    center_maximum = std::max(
        center_maximum,
        static_cast<double>(Kokkos::abs(
            state(mode, psi, 8, static_cast<std::size_t>(theta)))));
  }
  CHECK(center_maximum > 1.0e-3);
  CHECK(teuk::pipeline_initial_data_detail::radial_profile(pulse, 0.4) ==
        1.0);
  CHECK(teuk::pipeline_initial_data_detail::radial_profile(pulse, 0.2) ==
        0.0);
  CHECK(teuk::pipeline_initial_data_detail::radial_profile(pulse, 0.6) ==
        0.0);
}
