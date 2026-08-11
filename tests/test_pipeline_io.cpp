#include "test_harness.hpp"

#include <filesystem>
#include <vector>

#include "teuk/coupled.hpp"
#include "teuk/io.hpp"
#include "teuk/pipeline_io.hpp"
#include "teuk/rk4.hpp"

namespace {

void evolve(std::vector<teuk::PointPipelineState>& state, double& time,
            const int steps, const double time_step,
            const teuk::PointPipelineParameters& parameters) {
  teuk::RK4Workspace<teuk::PointPipelineState> workspace(state.size());
  const auto rhs = [&](const double,
                       const std::vector<teuk::PointPipelineState>& input,
                       std::vector<teuk::PointPipelineState>& output) {
    for (std::size_t i = 0; i < input.size(); ++i) {
      output[i] = teuk::evaluate_point_pipeline_rhs(input[i], parameters);
    }
  };
  for (int step = 0; step < steps; ++step) {
    teuk::classical_rk4_step(state, time, time_step, rhs, workspace);
    time += time_step;
  }
}

void check_state_near(const teuk::PointPipelineState& actual,
                      const teuk::PointPipelineState& expected) {
  for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
       ++field) {
    CHECK_COMPLEX_NEAR(teuk::point_pipeline_component(actual, field),
                       teuk::point_pipeline_component(expected, field), 0.0);
  }
}

}  // namespace

TEST_CASE("coupled snapshot packing preserves documented field ordering") {
  const teuk::SnapshotShape shape{2, teuk::point_pipeline_field_count, 3, 4};
  std::vector<teuk::PointPipelineState> points(2 * 3 * 4);
  for (std::size_t point = 0; point < points.size(); ++point) {
    for (std::size_t field = 0; field < teuk::point_pipeline_field_count;
         ++field) {
      teuk::set_point_pipeline_component(
          points[point], field,
          teuk::Complex(100.0 * field + point, -10.0 * field - point));
    }
  }
  const auto packed = teuk::pack_point_pipeline_snapshot(points, shape);
  const auto unpacked = teuk::unpack_point_pipeline_snapshot(packed, shape);
  CHECK(unpacked.size() == points.size());
  for (std::size_t point = 0; point < points.size(); ++point) {
    check_state_near(unpacked[point], points[point]);
  }
}

TEST_CASE("restarted coupled trajectory equals uninterrupted RK4 trajectory") {
  teuk::PointPipelineParameters parameters;
  parameters.background = {1.0, 0.73, 1.2};
  parameters.radius = 0.41;
  parameters.theta = 0.87;
  std::vector<teuk::PointPipelineState> uninterrupted{
      teuk::make_point_pipeline_seed(2.0e-3)};
  std::vector<teuk::PointPipelineState> interrupted = uninterrupted;
  constexpr double time_step = 2.0e-4;
  double uninterrupted_time = 0.0;
  evolve(uninterrupted, uninterrupted_time, 10, time_step, parameters);

  double interrupted_time = 0.0;
  evolve(interrupted, interrupted_time, 4, time_step, parameters);
  const teuk::SnapshotShape shape{1, teuk::point_pipeline_field_count, 1, 1};
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "teuk-kokkos-restart-test";
  teuk::RunProvenance provenance;
  provenance.time = interrupted_time;
  provenance.step = 4;
  provenance.time_step = time_step;
  provenance.spin = parameters.background.spin;
  teuk::write_metadata(directory, shape, {0}, "test", provenance);
  teuk::write_complex_snapshot(
      directory / "snapshot_000004.bin",
      teuk::pack_point_pipeline_snapshot(interrupted, shape));

  const auto metadata = teuk::read_metadata(directory);
  const auto packed = teuk::read_complex_snapshot(
      directory / "snapshot_000004.bin",
      teuk::checked_snapshot_value_count(metadata.shape));
  auto restarted =
      teuk::unpack_point_pipeline_snapshot(packed, metadata.shape);
  double restarted_time = metadata.provenance.time;
  evolve(restarted, restarted_time, 6, metadata.provenance.time_step,
         parameters);
  CHECK_NEAR(restarted_time, uninterrupted_time, 0.0);
  check_state_near(restarted.front(), uninterrupted.front());
  std::filesystem::remove_all(directory);
}

TEST_CASE("coupled snapshot shape validation fails closed") {
  bool rejected_fields = false;
  try {
    (void)teuk::pack_point_pipeline_snapshot(
        {teuk::PointPipelineState{}}, {1, 12, 1, 1});
  } catch (const std::invalid_argument&) {
    rejected_fields = true;
  }
  CHECK(rejected_fields);

  bool rejected_zero = false;
  try {
    (void)teuk::checked_snapshot_value_count({1, 13, 0, 1});
  } catch (const std::invalid_argument&) {
    rejected_zero = true;
  }
  CHECK(rejected_zero);
}
