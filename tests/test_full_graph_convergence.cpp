#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "teuk/angular.hpp"
#include "teuk/pipeline_bands.hpp"
#include "teuk/spatial_pipeline.hpp"

namespace {

constexpr teuk::PipelineAngularBands graph_bands{3, 6};
constexpr int graph_theta_count = 7;

struct GraphSnapshot {
  std::size_t radial_count = 0;
  std::size_t theta_count = 0;
  std::vector<teuk::Complex> rhs;
  std::vector<teuk::Complex> forcing;
  std::vector<teuk::Complex> inner_source;
  std::vector<teuk::Complex> inner_source_tangent;
  std::vector<teuk::Complex> constraints;
};

std::size_t field_index(const std::size_t field, const std::size_t radial,
                        const std::size_t theta,
                        const GraphSnapshot& snapshot) {
  return (field * snapshot.radial_count + radial) * snapshot.theta_count +
         theta;
}

std::size_t component_index(const std::size_t component,
                            const std::size_t radial,
                            const std::size_t theta,
                            const GraphSnapshot& snapshot) {
  return (component * snapshot.radial_count + radial) * snapshot.theta_count +
         theta;
}

teuk::Complex modal_radial_profile(const std::size_t field,
                                   const std::size_t ell_slot,
                                   const double radius) {
  const double field_scale = 2.0e-5 * (1.0 + 0.07 * field);
  const double modal_scale = 1.0 + 0.11 * ell_slot;
  const double alpha = 0.35 + 0.025 * field + 0.04 * ell_slot;
  return field_scale * modal_scale *
         teuk::Complex(std::exp(alpha * radius),
                       0.37 * std::sin((0.8 + 0.03 * field) * radius) +
                           0.19 * std::cos((0.55 + 0.02 * ell_slot) * radius));
}

template <class HostView>
void fill_exact_in_band_state(const HostView& state,
                              const teuk::UniformRadialGrid& grid) {
  for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
       ++field) {
    const int ell_max = teuk::pipeline_field_ell_max(field, graph_bands);
    const teuk::angular::SpinWeightedTransform transform(
        teuk::pipeline_field_spins[field], 0, ell_max, graph_theta_count);
    for (std::size_t radial = 0; radial < grid.size(); ++radial) {
      std::vector<teuk::Complex> modal(transform.mode_count());
      for (std::size_t ell = 0; ell < modal.size(); ++ell) {
        modal[ell] = modal_radial_profile(field, ell, grid.coordinate(radial));
      }
      const auto nodal = transform.synthesize(modal);
      for (std::size_t theta = 0; theta < nodal.size(); ++theta) {
        state(0, field, radial, theta) = nodal[theta];
      }
    }
  }
}

teuk::SpatialPipeline make_graph_pipeline(const teuk::ExecutionSpace& execution,
                                          const std::size_t radial_count,
                                          const std::string& label,
                                          const teuk::RadialDiscretization
                                              discretization =
                                                  teuk::RadialDiscretization::D84,
                                          const double dissipation = 0.0) {
  return teuk::SpatialPipeline(
      execution, teuk::ModeRegistry({0}),
      teuk::UniformRadialGrid(radial_count, 0.0, 0.64), graph_bands,
      graph_theta_count, teuk::KerrParameters{1.0, 0.41, 1.5}, 0.17,
      dissipation,
      teuk::ReductionEvolution::FreeDamped, label,
      teuk::SecondOrderSourcePolicy::unrestricted(),
      discretization);
}

GraphSnapshot evaluate_graph(
    const std::size_t radial_count,
    const teuk::RadialDiscretization discretization =
        teuk::RadialDiscretization::D84,
    const double dissipation = 0.0) {
  const teuk::ExecutionSpace execution;
  auto pipeline = make_graph_pipeline(execution, radial_count,
                                      "full_graph_spatial", discretization,
                                      dissipation);
  const auto grid = pipeline.storage().radial_grid();
  auto initial = Kokkos::create_mirror_view(pipeline.storage().state());
  fill_exact_in_band_state(initial, grid);
  Kokkos::deep_copy(execution, pipeline.storage().state(), initial);
  pipeline.evaluate_rhs(execution, pipeline.storage().state(),
                        pipeline.storage().rhs());
  execution.fence("finish full graph spatial evaluation");

  const auto rhs = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().rhs());
  const auto forcing = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.forcing());
  const auto inner = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.inner_source());
  const auto tangent = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.inner_source_tangent());
  const auto constraints = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.independent_reconstruction_constraints());

  GraphSnapshot result;
  result.radial_count = radial_count;
  result.theta_count = graph_theta_count;
  result.rhs.resize(teuk::point_pipeline_field_count * radial_count *
                    graph_theta_count);
  result.forcing.resize(radial_count * graph_theta_count);
  result.inner_source.resize(2 * radial_count * graph_theta_count);
  result.inner_source_tangent.resize(2 * radial_count * graph_theta_count);
  result.constraints.resize(3 * radial_count * graph_theta_count);
  for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
       ++field) {
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t theta = 0; theta < graph_theta_count; ++theta) {
        result.rhs[field_index(field, radial, theta, result)] =
            rhs(0, field, radial, theta);
      }
    }
  }
  for (std::size_t radial = 0; radial < radial_count; ++radial) {
    for (std::size_t theta = 0; theta < graph_theta_count; ++theta) {
      result.forcing[radial * graph_theta_count + theta] =
          forcing(0, radial, theta);
      for (std::size_t component = 0; component < 2; ++component) {
        const auto index = component_index(component, radial, theta, result);
        result.inner_source[index] = inner(0, component, radial, theta);
        result.inner_source_tangent[index] =
            tangent(0, component, radial, theta);
      }
      for (std::size_t component = 0; component < 3; ++component) {
        result.constraints[component_index(component, radial, theta, result)] =
            constraints(0, component, radial, theta);
      }
    }
  }
  return result;
}

struct DifferenceNorm {
  double rms = 0.0;
  double endpoint_maximum = 0.0;
  double magnitude_maximum = 0.0;
};

DifferenceNorm rhs_difference(const GraphSnapshot& coarse,
                              const GraphSnapshot& fine,
                              const std::size_t first_field,
                              const std::size_t field_count) {
  double squared = 0.0;
  std::size_t count = 0;
  DifferenceNorm result;
  for (std::size_t field = first_field; field < first_field + field_count;
       ++field) {
    for (std::size_t radial = 0; radial < coarse.radial_count; ++radial) {
      for (std::size_t theta = 0; theta < coarse.theta_count; ++theta) {
        const auto coarse_value =
            coarse.rhs[field_index(field, radial, theta, coarse)];
        const auto fine_value =
            fine.rhs[field_index(field, 2 * radial, theta, fine)];
        const double difference = Kokkos::abs(coarse_value - fine_value);
        squared += difference * difference;
        ++count;
        result.magnitude_maximum = std::max(
            result.magnitude_maximum,
            static_cast<double>(Kokkos::abs(coarse_value)));
        if (radial == 0 || radial + 1 == coarse.radial_count) {
          result.endpoint_maximum =
              std::max(result.endpoint_maximum, difference);
        }
      }
    }
  }
  result.rms = std::sqrt(squared / static_cast<double>(count));
  return result;
}

DifferenceNorm component_difference(const GraphSnapshot& coarse,
                                    const GraphSnapshot& fine,
                                    const std::vector<teuk::Complex>& coarse_v,
                                    const std::vector<teuk::Complex>& fine_v,
                                    const std::size_t component_count) {
  double squared = 0.0;
  std::size_t count = 0;
  DifferenceNorm result;
  for (std::size_t component = 0; component < component_count; ++component) {
    for (std::size_t radial = 0; radial < coarse.radial_count; ++radial) {
      for (std::size_t theta = 0; theta < coarse.theta_count; ++theta) {
        const auto coarse_value =
            coarse_v[component_index(component, radial, theta, coarse)];
        const auto fine_value =
            fine_v[component_index(component, 2 * radial, theta, fine)];
        const double difference = Kokkos::abs(coarse_value - fine_value);
        squared += difference * difference;
        ++count;
        result.magnitude_maximum = std::max(
            result.magnitude_maximum,
            static_cast<double>(Kokkos::abs(coarse_value)));
        if (radial == 0 || radial + 1 == coarse.radial_count) {
          result.endpoint_maximum =
              std::max(result.endpoint_maximum, difference);
        }
      }
    }
  }
  result.rms = std::sqrt(squared / static_cast<double>(count));
  return result;
}

DifferenceNorm forcing_difference(const GraphSnapshot& coarse,
                                  const GraphSnapshot& fine) {
  double squared = 0.0;
  std::size_t count = 0;
  DifferenceNorm result;
  for (std::size_t radial = 0; radial < coarse.radial_count; ++radial) {
    for (std::size_t theta = 0; theta < coarse.theta_count; ++theta) {
      const auto coarse_value = coarse.forcing[radial * coarse.theta_count + theta];
      const auto fine_value = fine.forcing[2 * radial * fine.theta_count + theta];
      const double difference = Kokkos::abs(coarse_value - fine_value);
      squared += difference * difference;
      ++count;
      result.magnitude_maximum = std::max(
          result.magnitude_maximum,
          static_cast<double>(Kokkos::abs(coarse_value)));
      if (radial == 0 || radial + 1 == coarse.radial_count) {
        result.endpoint_maximum = std::max(result.endpoint_maximum, difference);
      }
    }
  }
  result.rms = std::sqrt(squared / static_cast<double>(count));
  return result;
}

struct TemporalSnapshot {
  std::vector<teuk::Complex> state;
  std::size_t radial_count = 0;
  std::size_t theta_count = 0;
};

TemporalSnapshot evolve_graph(
    const int step_count,
    const teuk::RadialDiscretization discretization =
        teuk::RadialDiscretization::D84,
    const std::size_t radial_count = 17,
    const double final_time = 0.024,
    const double dissipation = 0.0) {
  const teuk::ExecutionSpace execution;
  auto pipeline = make_graph_pipeline(execution, radial_count,
                                      "full_graph_temporal", discretization,
                                      dissipation);
  auto initial = Kokkos::create_mirror_view(pipeline.storage().state());
  fill_exact_in_band_state(initial, pipeline.storage().radial_grid());
  for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
       ++field) {
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t theta = 0; theta < graph_theta_count; ++theta) {
        const bool reconstruction_field = field >= 3 && field < 10;
        initial(0, field, radial, theta) *=
            (reconstruction_field ? 1.0e5 : 1000.0) *
            (1.0 + 0.35 * std::sin((9.0 + field) *
                                    pipeline.storage().radial_grid().coordinate(
                                        radial)));
      }
    }
  }
  Kokkos::deep_copy(execution, pipeline.storage().state(), initial);
  const double time_step = final_time / static_cast<double>(step_count);
  for (int step = 0; step < step_count; ++step) {
    pipeline.step(execution, step * time_step, time_step);
  }
  execution.fence("finish full graph temporal trajectory");
  const auto state = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, pipeline.storage().state());
  TemporalSnapshot result;
  result.radial_count = radial_count;
  result.theta_count = graph_theta_count;
  result.state.resize(teuk::point_pipeline_field_count * radial_count *
                      graph_theta_count);
  for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
       ++field) {
    for (std::size_t radial = 0; radial < radial_count; ++radial) {
      for (std::size_t theta = 0; theta < graph_theta_count; ++theta) {
        const std::size_t index =
            (field * radial_count + radial) * graph_theta_count + theta;
        result.state[index] = state(0, field, radial, theta);
      }
    }
  }
  return result;
}

double temporal_sector_error(const TemporalSnapshot& values,
                             const TemporalSnapshot& reference,
                             const std::size_t first_field,
                             const std::size_t field_count) {
  double squared = 0.0;
  std::size_t count = 0;
  for (std::size_t field = first_field; field < first_field + field_count;
       ++field) {
    for (std::size_t radial = 0; radial < values.radial_count; ++radial) {
      for (std::size_t theta = 0; theta < values.theta_count; ++theta) {
        const std::size_t index =
            (field * values.radial_count + radial) * values.theta_count +
            theta;
        const double difference =
            Kokkos::abs(values.state[index] - reference.state[index]);
        squared += difference * difference;
        ++count;
      }
    }
  }
  return std::sqrt(squared / static_cast<double>(count));
}

}  // namespace

TEST_CASE("complete D84 graph exposes the nested source endpoint order gap") {
  const auto coarse = evaluate_graph(49);
  const auto medium = evaluate_graph(97);
  const auto fine = evaluate_graph(193);

  const auto first_cm = rhs_difference(coarse, medium, 0, 3);
  const auto first_mf = rhs_difference(medium, fine, 0, 3);
  const auto reconstruction_cm = rhs_difference(coarse, medium, 3, 7);
  const auto reconstruction_mf = rhs_difference(medium, fine, 3, 7);
  const auto second_cm = rhs_difference(coarse, medium, 10, 3);
  const auto second_mf = rhs_difference(medium, fine, 10, 3);
  const auto forcing_cm = forcing_difference(coarse, medium);
  const auto forcing_mf = forcing_difference(medium, fine);
  const auto source_cm = component_difference(
      coarse, medium, coarse.inner_source, medium.inner_source, 2);
  const auto source_mf =
      component_difference(medium, fine, medium.inner_source,
                           fine.inner_source, 2);
  const auto tangent_cm = component_difference(
      coarse, medium, coarse.inner_source_tangent,
      medium.inner_source_tangent, 2);
  const auto tangent_mf = component_difference(
      medium, fine, medium.inner_source_tangent, fine.inner_source_tangent, 2);
  const auto constraint_cm = component_difference(
      coarse, medium, coarse.constraints, medium.constraints, 3);
  const auto constraint_mf = component_difference(
      medium, fine, medium.constraints, fine.constraints, 3);

  const std::array<std::pair<DifferenceNorm, DifferenceNorm>, 7> pairs{
      {{first_cm, first_mf},
       {reconstruction_cm, reconstruction_mf},
       {second_cm, second_mf},
       {forcing_cm, forcing_mf},
       {source_cm, source_mf},
       {tangent_cm, tangent_mf},
       {constraint_cm, constraint_mf}}};
  for (const auto& pair : pairs) {
    CHECK(pair.first.magnitude_maximum > 1.0e-14);
  }

  // The direct first-derivative sectors retain fourth-order endpoint closure.
  for (const std::size_t index : {0U, 1U, 2U, 4U, 6U}) {
    CHECK(pairs[index].first.rms / pairs[index].second.rms > 12.0);
    CHECK(pairs[index].first.endpoint_maximum /
              pairs[index].second.endpoint_maximum >
          12.0);
  }

  // The inner-source tangent and final forcing each contain a nested radial
  // derivative. D8-4's O(h^4) boundary error is differentiated once more, so
  // the endpoint approaches O(h^3), not O(h^4). Preserve this measurement as
  // an explicit blocker rather than silently weakening an overall-order gate.
  const double tangent_endpoint_ratio =
      tangent_cm.endpoint_maximum / tangent_mf.endpoint_maximum;
  const double forcing_endpoint_ratio =
      forcing_cm.endpoint_maximum / forcing_mf.endpoint_maximum;
  CHECK(tangent_cm.rms / tangent_mf.rms > 12.0);
  CHECK(tangent_endpoint_ratio > 6.0);
  CHECK(tangent_endpoint_ratio < 10.0);
  CHECK(forcing_cm.rms / forcing_mf.rms > 10.0);
  CHECK(forcing_cm.rms / forcing_mf.rms < 14.0);
  CHECK(forcing_endpoint_ratio > 6.0);
  CHECK(forcing_endpoint_ratio < 10.0);
}

TEST_CASE("complete zero-dissipation D105 graph closes the endpoint order gap") {
  constexpr double dissipation = 0.0;
  const auto coarse = evaluate_graph(
      49, teuk::RadialDiscretization::D105, dissipation);
  const auto medium = evaluate_graph(
      97, teuk::RadialDiscretization::D105, dissipation);
  const auto fine = evaluate_graph(
      193, teuk::RadialDiscretization::D105, dissipation);

  const auto first_cm = rhs_difference(coarse, medium, 0, 3);
  const auto first_mf = rhs_difference(medium, fine, 0, 3);
  const auto reconstruction_cm = rhs_difference(coarse, medium, 3, 7);
  const auto reconstruction_mf = rhs_difference(medium, fine, 3, 7);
  const auto second_cm = rhs_difference(coarse, medium, 10, 3);
  const auto second_mf = rhs_difference(medium, fine, 10, 3);
  const auto forcing_cm = forcing_difference(coarse, medium);
  const auto forcing_mf = forcing_difference(medium, fine);
  const auto source_cm = component_difference(
      coarse, medium, coarse.inner_source, medium.inner_source, 2);
  const auto source_mf = component_difference(
      medium, fine, medium.inner_source, fine.inner_source, 2);
  const auto tangent_cm = component_difference(
      coarse, medium, coarse.inner_source_tangent,
      medium.inner_source_tangent, 2);
  const auto tangent_mf = component_difference(
      medium, fine, medium.inner_source_tangent,
      fine.inner_source_tangent, 2);
  const auto constraint_cm = component_difference(
      coarse, medium, coarse.constraints, medium.constraints, 3);
  const auto constraint_mf = component_difference(
      medium, fine, medium.constraints, fine.constraints, 3);

  const std::array<std::pair<DifferenceNorm, DifferenceNorm>, 7> pairs{
      {{first_cm, first_mf},
       {reconstruction_cm, reconstruction_mf},
       {second_cm, second_mf},
       {forcing_cm, forcing_mf},
       {source_cm, source_mf},
       {tangent_cm, tangent_mf},
       {constraint_cm, constraint_mf}}};
  const std::array<const char*, 7> names{
      "first RHS", "reconstruction RHS", "second RHS", "forcing",
      "inner source", "inner-source tangent", "constraints"};
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    const auto& pair = pairs[index];
    std::cout << "D105 full-graph epsilon=" << dissipation << " "
              << names[index] << " RMS/endpoint ratios "
              << pair.first.rms / pair.second.rms << " "
              << pair.first.endpoint_maximum / pair.second.endpoint_maximum
              << " raw " << pair.first.rms << " " << pair.second.rms << " "
              << pair.first.endpoint_maximum << " "
              << pair.second.endpoint_maximum
              << '\n';
    CHECK(pair.first.magnitude_maximum > 1.0e-14);
    const double rms_ratio = pair.first.rms / pair.second.rms;
    const double endpoint_ratio =
        pair.first.endpoint_maximum / pair.second.endpoint_maximum;
    CHECK(std::isfinite(rms_ratio));
    CHECK(std::isfinite(endpoint_ratio));
    CHECK(rms_ratio > 15.0);
    CHECK(endpoint_ratio > 15.0);
  }
}

TEST_CASE("complete D84 graph has fourth order RK4 convergence at fixed space") {
  const auto coarse = evolve_graph(16);
  const auto medium = evolve_graph(32);
  const auto fine = evolve_graph(64);
  const auto reference = evolve_graph(512);
  for (const auto& sector :
       std::array<std::pair<std::size_t, std::size_t>, 3>{
           {{0, 3}, {3, 7}, {10, 3}}}) {
    const double coarse_error = temporal_sector_error(
        coarse, reference, sector.first, sector.second);
    const double medium_error = temporal_sector_error(
        medium, reference, sector.first, sector.second);
    const double fine_error = temporal_sector_error(
        fine, reference, sector.first, sector.second);
    CHECK(coarse_error / medium_error > 12.0);
    CHECK(medium_error / fine_error > 12.0);
  }
}

TEST_CASE("complete D105 graph has fourth order RK4 convergence at fixed space") {
  const auto coarse = evolve_graph(
      4, teuk::RadialDiscretization::D105, 25, 0.24);
  const auto medium = evolve_graph(
      8, teuk::RadialDiscretization::D105, 25, 0.24);
  const auto fine = evolve_graph(
      16, teuk::RadialDiscretization::D105, 25, 0.24);
  const auto reference = evolve_graph(
      256, teuk::RadialDiscretization::D105, 25, 0.24);
  for (const auto& sector :
       std::array<std::pair<std::size_t, std::size_t>, 3>{
           {{0, 3}, {3, 7}, {10, 3}}}) {
    const double coarse_error = temporal_sector_error(
        coarse, reference, sector.first, sector.second);
    const double medium_error = temporal_sector_error(
        medium, reference, sector.first, sector.second);
    const double fine_error = temporal_sector_error(
        fine, reference, sector.first, sector.second);
    std::cout << "D105 fixed-space RK4 sector " << sector.first
              << " errors " << coarse_error << " " << medium_error << " "
              << fine_error << " ratios " << coarse_error / medium_error
              << " " << medium_error / fine_error << '\n';
    const double coarse_medium_ratio = coarse_error / medium_error;
    const double medium_fine_ratio = medium_error / fine_error;
    CHECK(std::isfinite(coarse_error));
    CHECK(std::isfinite(medium_error));
    CHECK(std::isfinite(fine_error));
    CHECK(std::isfinite(coarse_medium_ratio));
    CHECK(std::isfinite(medium_fine_ratio));
    CHECK(coarse_medium_ratio > 12.0);
    CHECK(medium_fine_ratio > 12.0);
  }
}
