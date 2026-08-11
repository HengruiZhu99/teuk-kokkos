#include "test_harness.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

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

}  // namespace

TEST_CASE("composed spatial pipeline injects and scales its quadratic source") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const teuk::KerrParameters parameters{1.0, 0.37, 1.6};
  teuk::SpatialPipeline pipeline(execution, registry, radial_grid, 3, 6,
                                 parameters, 0.12, 0.01);

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
        CHECK_COMPLEX_NEAR(
            rhs_host(mode,
                     static_cast<std::size_t>(teuk::PipelineField::SecondP),
                     radial, theta),
            forcing, 2.0e-12);
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
      teuk::KerrParameters{1.0, 0.31, 1.5}, 0.08, 0.005);
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
