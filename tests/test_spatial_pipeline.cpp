#include "test_harness.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <cmath>
#include <vector>

#include "teuk/pipeline_initial_data.hpp"
#include "teuk/spatial_pipeline.hpp"

namespace {

int pipeline_deep_copies = 0;
int pipeline_allocations = 0;

void count_pipeline_deep_copy(Kokkos::Tools::SpaceHandle, const char*,
                              const void*, Kokkos::Tools::SpaceHandle,
                              const char*, const void*, std::uint64_t) {
  ++pipeline_deep_copies;
}

void count_pipeline_allocation(Kokkos::Tools::SpaceHandle, const char*,
                               const void*, std::uint64_t) {
  ++pipeline_allocations;
}

constexpr std::array<int, teuk::point_pipeline_field_count> pipeline_spins{
    -2, -2, -2, -1, -2, 0, -2, -1, -1, 0, -2, -2, -2};

template <class HostView>
double maximum_relative_off_band(const HostView& host,
                                 const teuk::ModeRegistry& registry,
                                 const int ell_max,
                                 const std::size_t theta_count) {
  double maximum_residual = 0.0;
  std::vector<teuk::Complex> nodal(theta_count);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
         ++field) {
      const teuk::angular::SpinWeightedTransform transform(
          pipeline_spins[field], registry.modes()[mode], ell_max,
          static_cast<int>(theta_count));
      for (std::size_t radial = 0; radial < host.extent(2); ++radial) {
        double line_maximum = 0.0;
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          nodal[theta] = host(mode, field, radial, theta);
          line_maximum =
              std::max(line_maximum,
                       static_cast<double>(Kokkos::abs(nodal[theta])));
        }
        const auto projected = transform.synthesize(transform.analyze(nodal));
        double line_residual = 0.0;
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          line_residual = std::max(
              line_residual,
              static_cast<double>(Kokkos::abs(nodal[theta] - projected[theta])));
        }
        maximum_residual = std::max(
            maximum_residual,
            line_residual / std::max(line_maximum, 1.0e-300));
      }
    }
  }
  return maximum_residual;
}

}  // namespace

TEST_CASE("composed spatial pipeline injects and scales its quadratic source") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const teuk::KerrParameters parameters{1.0, 0.37, 1.6};
  teuk::SpatialPipeline pipeline(execution, registry, radial_grid, 3, 6,
                                 parameters, 0.12, 0.01,
                                 teuk::ReductionEvolution::FreeDamped,
                                 "explicit_unrestricted_source_test",
                                 teuk::SecondOrderSourcePolicy::unrestricted());

  auto state_host = Kokkos::create_mirror_view(pipeline.storage().state());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
         ++field) {
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < pipeline.storage().theta_count();
             ++theta) {
          const double seed = 1.0 + 0.13 * static_cast<double>(mode) +
                              0.07 * static_cast<double>(field) +
                              0.03 * static_cast<double>(radial) +
                              0.02 * static_cast<double>(theta);
          state_host(mode, field, radial, theta) =
              field < static_cast<std::size_t>(teuk::PipelineField::SecondP)
                  ? teuk::Complex(2.0e-4 * seed, -1.0e-4 * seed)
                  : teuk::Complex(0.0, 0.0);
        }
      }
    }
  }
  Kokkos::deep_copy(execution, pipeline.storage().state(), state_host);

  pipeline_allocations = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_pipeline_allocation);
  {
    teuk::FullSpatialValueView positive_control(
        "pipeline_allocation_positive_control", 1, 1, 1);
    execution.fence("observe pipeline allocation positive control");
  }
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(pipeline_allocations > 0);

  pipeline_deep_copies = 0;
  pipeline_allocations = 0;
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(
      count_pipeline_deep_copy);
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_pipeline_allocation);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  execution.fence("composed spatial pipeline test");
  Kokkos::Tools::Experimental::set_begin_deep_copy_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  CHECK(pipeline_deep_copies == 0);
  CHECK(pipeline_allocations == 0);

  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      rhs_host("pipeline_rhs_snapshot", registry.size(),
               teuk::point_pipeline_field_count, radial_grid.size(),
               pipeline.storage().theta_count());
  Kokkos::View<teuk::Complex***, Kokkos::LayoutRight, Kokkos::HostSpace>
      forcing_host("pipeline_forcing_snapshot", registry.size(),
                   radial_grid.size(), pipeline.storage().theta_count());
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      source_host("pipeline_source_snapshot", registry.size(), 2,
                  radial_grid.size(), pipeline.storage().theta_count());
  Kokkos::deep_copy(rhs_host, pipeline.storage().rhs());
  Kokkos::deep_copy(forcing_host, pipeline.forcing());
  Kokkos::deep_copy(source_host, pipeline.inner_source());
  double maximum_forcing = 0.0;
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < pipeline.storage().theta_count();
           ++theta) {
        const auto forcing = forcing_host(mode, radial, theta);
        maximum_forcing =
            std::max(maximum_forcing, static_cast<double>(Kokkos::abs(forcing)));
        std::vector<teuk::Complex> forcing_line(
            pipeline.storage().theta_count());
        for (std::size_t angle = 0; angle < pipeline.storage().theta_count();
             ++angle) {
          forcing_line[angle] = forcing_host(mode, radial, angle);
        }
        const teuk::angular::SpinWeightedTransform forcing_transform(
            -2, registry.modes()[mode], 3,
            static_cast<int>(pipeline.storage().theta_count()));
        const auto projected_forcing = forcing_transform.synthesize(
            forcing_transform.analyze(forcing_line));
        CHECK_COMPLEX_NEAR(
            rhs_host(mode,
                     static_cast<std::size_t>(teuk::PipelineField::SecondP),
                     radial, theta),
            projected_forcing[theta], 2.0e-12);
        CHECK_COMPLEX_NEAR(
            rhs_host(mode,
                     static_cast<std::size_t>(teuk::PipelineField::SecondQ),
                     radial, theta),
            teuk::Complex(0.0, 0.0), 2.0e-12);
        CHECK_COMPLEX_NEAR(
            rhs_host(mode,
                     static_cast<std::size_t>(teuk::PipelineField::SecondPsi),
                     radial, theta),
            teuk::Complex(0.0, 0.0), 2.0e-12);
      }
    }
  }
  CHECK(maximum_forcing > 1.0e-12);

  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0;
         field < static_cast<std::size_t>(teuk::PipelineField::SecondP);
         ++field) {
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < pipeline.storage().theta_count();
             ++theta) {
          state_host(mode, field, radial, theta) *= 2.0;
        }
      }
    }
  }
  Kokkos::deep_copy(execution, pipeline.storage().state(), state_host);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  execution.fence("scaled composed spatial pipeline test");
  const auto scaled_rhs_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().rhs());
  const auto scaled_forcing_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.forcing());
  const auto scaled_source_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.inner_source());
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < pipeline.storage().theta_count();
           ++theta) {
        for (std::size_t field = 0;
             field < static_cast<std::size_t>(teuk::PipelineField::SecondP);
             ++field) {
          CHECK_COMPLEX_NEAR(scaled_rhs_host(mode, field, radial, theta),
                             2.0 * rhs_host(mode, field, radial, theta),
                             2.0e-10);
        }
        CHECK_COMPLEX_NEAR(scaled_forcing_host(mode, radial, theta),
                           4.0 * forcing_host(mode, radial, theta), 2.0e-10);
        CHECK_COMPLEX_NEAR(
            scaled_rhs_host(
                mode, static_cast<std::size_t>(teuk::PipelineField::SecondP),
                radial, theta),
            4.0 * rhs_host(
                      mode,
                      static_cast<std::size_t>(teuk::PipelineField::SecondP),
                      radial, theta),
            2.0e-10);
        for (std::size_t component = 0; component < 2; ++component) {
          CHECK_COMPLEX_NEAR(
              scaled_source_host(mode, component, radial, theta),
              4.0 * source_host(mode, component, radial, theta), 2.0e-10);
        }
      }
    }
  }
}

TEST_CASE("full pipeline RHS and RK updates stay in every retained spin band") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-4, -3, -2, -1, 0, 1, 2, 3, 4});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  constexpr int ell_max = 4;
  constexpr std::size_t theta_count = 7;
  teuk::SpatialPipeline pipeline(
      execution, registry, radial_grid, ell_max,
      static_cast<int>(theta_count),
      teuk::KerrParameters{1.0, 0.71, 1.4}, 0.09, 0.003);

  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      initial("bandlimited_pipeline_initial", registry.size(),
              teuk::point_pipeline_field_count, radial_grid.size(),
              theta_count);
  for (std::size_t mode = 0; mode < registry.size(); ++mode) {
    for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
         ++field) {
      const teuk::angular::SpinWeightedTransform transform(
          pipeline_spins[field], registry.modes()[mode], ell_max,
          static_cast<int>(theta_count));
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        std::vector<teuk::Complex> modal(transform.mode_count());
        for (std::size_t ell = 0; ell < modal.size(); ++ell) {
          const double seed = 1.0 + 0.07 * static_cast<double>(mode) +
                              0.03 * static_cast<double>(field) +
                              0.02 * static_cast<double>(radial) +
                              0.01 * static_cast<double>(ell);
          modal[ell] = teuk::Complex(2.0e-6 * seed, -1.0e-6 * seed);
        }
        const auto nodal = transform.synthesize(modal);
        for (std::size_t theta = 0; theta < theta_count; ++theta) {
          initial(mode, field, radial, theta) = nodal[theta];
        }
      }
    }
  }
  Kokkos::deep_copy(execution, pipeline.storage().state(), initial);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  execution.fence("evaluate bandlimited pipeline RHS");
  const auto host_rhs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().rhs());
  CHECK(maximum_relative_off_band(host_rhs, registry, ell_max, theta_count) <
        2.0e-12);

  teuk::FieldView stage("bandlimited_pipeline_stage", registry.size(),
                        teuk::point_pipeline_field_count, radial_grid.size(),
                        theta_count);
  const auto state = pipeline.storage().state();
  const auto rhs = pipeline.storage().rhs();
  Kokkos::parallel_for(
      "form_one_bandlimited_rk_stage", pipeline.storage().value_count(),
      KOKKOS_LAMBDA(const std::size_t i) {
        stage.data()[i] = state.data()[i] + 1.0e-4 * rhs.data()[i];
      });
  execution.fence("form one bandlimited RK stage");
  const auto host_stage =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, stage);
  CHECK(maximum_relative_off_band(host_stage, registry, ell_max, theta_count) <
        2.0e-12);

  pipeline.step(execution, 0.0, 1.0e-4);
  execution.fence("advance one bandlimited RK step");
  const auto host_advanced = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().state());
  CHECK(maximum_relative_off_band(host_advanced, registry, ell_max,
                                  theta_count) < 3.0e-12);
}

TEST_CASE("composed spatial pipeline rejects an unpadded nonlinear grid") {
  const teuk::ExecutionSpace execution;
  bool threw = false;
  try {
    const teuk::SpatialPipeline pipeline(
        execution, teuk::ModeRegistry({-1, 0, 1}),
        teuk::UniformRadialGrid(9, 0.0, 0.8), 4, 6,
        teuk::KerrParameters{1.0, 0.2, 1.4});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

TEST_CASE("composed nonlinear spatial pipeline retains RK4 stage order") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({0});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  teuk::SpatialPipeline pipeline(
      execution, registry, radial_grid, 3, 5,
      teuk::KerrParameters{1.0, 0.31, 1.5}, 0.08, 0.005,
      teuk::ReductionEvolution::FreeDamped,
      "explicit_unrestricted_temporal_test",
      teuk::SecondOrderSourcePolicy::unrestricted());
  Kokkos::View<teuk::Complex****, Kokkos::LayoutRight, Kokkos::HostSpace>
      initial("pipeline_rk_initial", 1, teuk::point_pipeline_field_count,
              radial_grid.size(), pipeline.storage().theta_count());
  for (std::size_t field = 0;
       field < static_cast<std::size_t>(teuk::PipelineField::SecondP);
       ++field) {
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < pipeline.storage().theta_count();
           ++theta) {
        const double seed = 1.0 + 0.04 * static_cast<double>(field) +
                            0.02 * static_cast<double>(radial) +
                            0.015 * static_cast<double>(theta);
        initial(0, field, radial, theta) =
            teuk::Complex(3.0e-5 * seed, -2.0e-5 * seed);
      }
    }
  }
  for (std::size_t field =
           static_cast<std::size_t>(teuk::PipelineField::SecondP);
       field < teuk::point_pipeline_field_count; ++field) {
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < pipeline.storage().theta_count();
           ++theta) {
        initial(0, field, radial, theta) = teuk::Complex(0.0, 0.0);
      }
    }
  }

  const auto integrate = [&](const int steps) {
    Kokkos::deep_copy(execution, pipeline.storage().state(), initial);
    constexpr double final_time = 5.0e-2;
    const double time_step = final_time / static_cast<double>(steps);
    for (int step = 0; step < steps; ++step) {
      pipeline.step(execution, static_cast<double>(step) * time_step,
                    time_step);
    }
    execution.fence("finish composed spatial RK trajectory");
    Kokkos::View<teuk::Complex*, Kokkos::HostSpace> host(
        "pipeline_rk_snapshot", pipeline.storage().value_count());
    Kokkos::deep_copy(host, pipeline.storage().flat_state());
    std::vector<teuk::Complex> result(host.extent(0));
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
    return result;
  };
  const auto coarse = integrate(1);
  const auto medium = integrate(2);
  const auto fine = integrate(4);
  const auto reference = integrate(32);
  const auto error = [&](const std::vector<teuk::Complex>& values) {
    double squared = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
      const double difference = Kokkos::abs(values[i] - reference[i]);
      squared += difference * difference;
    }
    return std::sqrt(squared / static_cast<double>(values.size()));
  };
  const double coarse_error = error(coarse);
  const double medium_error = error(medium);
  const double fine_error = error(fine);
  CHECK(coarse_error / medium_error > 12.0);
  CHECK(medium_error / fine_error > 12.0);
}

TEST_CASE("pipeline supports distinct parent target modes and angular bandlimits") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-4, -2, 0, 2, 4}, {-2, 2},
                                    {-4, 0, 4});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const teuk::KerrParameters background{1.0, 0.4, 1.5};
  teuk::SpatialPipeline pipeline(
      execution, registry, radial_grid, teuk::PipelineAngularBands{3, 4}, 6,
      background, 0.1, 0.0, teuk::ReductionEvolution::FreeDamped,
      "separate_parent_target_bands",
      teuk::SecondOrderSourcePolicy::unrestricted());
  teuk::PipelineGaussianPulse pulse;
  pulse.center = 0.4;
  pulse.width = 0.14;
  pulse.modes = {{2, -2, teuk::Complex(1.2e-4, -0.4e-4)},
                 {2, 2, teuk::Complex(-0.9e-4, 0.7e-4)}};
  for (std::size_t field = 0; field < pulse.reconstruction_scales.size();
       ++field) {
    pulse.reconstruction_scales[field] =
        teuk::Complex(0.08 + 0.01 * static_cast<double>(field), -0.03);
  }
  teuk::initialize_compactified_gaussian_pulse(
      execution, pipeline, registry, 3, background, pulse);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  execution.fence("evaluate separate parent target bands");
  const auto state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().state());
  const auto rhs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().rhs());

  for (const int daughter_only : {-4, 4}) {
    const std::size_t mode = registry.index(daughter_only);
    for (std::size_t field = 0;
         field < static_cast<std::size_t>(teuk::PipelineField::SecondP);
         ++field) {
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < 6; ++theta) {
          CHECK_COMPLEX_NEAR(state(mode, field, radial, theta),
                             teuk::Complex(0.0, 0.0), 0.0);
          CHECK_COMPLEX_NEAR(rhs(mode, field, radial, theta),
                             teuk::Complex(0.0, 0.0), 0.0);
        }
      }
    }
  }
  for (const int parent_only : {-2, 2}) {
    const std::size_t mode = registry.index(parent_only);
    for (std::size_t field =
             static_cast<std::size_t>(teuk::PipelineField::SecondP);
         field <= static_cast<std::size_t>(teuk::PipelineField::SecondPsi);
         ++field) {
      for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
        for (std::size_t theta = 0; theta < 6; ++theta) {
          CHECK_COMPLEX_NEAR(rhs(mode, field, radial, theta),
                             teuk::Complex(0.0, 0.0), 0.0);
        }
      }
    }
  }
  double daughter_rhs_maximum = 0.0;
  for (const int target : {-4, 4}) {
    const std::size_t mode = registry.index(target);
    for (std::size_t radial = 0; radial < radial_grid.size(); ++radial) {
      for (std::size_t theta = 0; theta < 6; ++theta) {
        daughter_rhs_maximum = std::max(
            daughter_rhs_maximum,
            static_cast<double>(Kokkos::abs(rhs(
                mode, static_cast<std::size_t>(teuk::PipelineField::SecondP),
                radial, theta))));
      }
    }
  }
  CHECK(daughter_rhs_maximum > 1.0e-16);
}
