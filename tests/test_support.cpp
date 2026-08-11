#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "test_harness.hpp"
#include "teuk/config.hpp"
#include "teuk/diagnostics.hpp"
#include "teuk/io.hpp"

TEST_CASE("surface gravity has Schwarzschild and extremal limits") {
  CHECK_NEAR(teuk::surface_gravity(1.0, 0.0), 0.25, 1.0e-15);
  CHECK_NEAR(teuk::surface_gravity(1.0, 1.0), 0.0, 1.0e-15);
  CHECK(teuk::surface_gravity(1.0, 0.999) > 0.0);
}

TEST_CASE("runtime configuration parses typed values modes and overrides") {
  const std::string text = R"cfg(
    # a complete nonlinear run
    config_version = 1
    mass = 1.0
    spin = 7.0e-1
    compactification_length = 1.5
    nr = 33
    ntheta = 8
    ellmax_first = 4
    ellmax_second = 5
    first_order_modes = -2, 2
    second_order_modes = -4, 0, 4
    final_time = 2.5e-2
    steps = 64
    cfl = 1.0e-1
    reduction_damping = 0.2
    dissipation = 5e-3
    reduction_mode = stage_constrained
    initial_data.type = gaussian
    initial_data.seed_ell = 3
    initial_data.seed_m = 2
    initial_data.amplitude_real = 2e-4
    initial_data.amplitude_imag = -3e-5
    initial_data.mode.1.ell = 4
    initial_data.mode.1.m = -2
    initial_data.mode.1.amplitude_real = -1e-4
    initial_data.mode.1.amplitude_imag = 4e-5
    initial_data.add_sharp_partner = false
    second_order.enabled = true
    second_order.source_mode = constraint_aware
    second_order.source_start_time = 0.75
    second_order.constraint_tolerance = 2e-8
    second_order.required_consecutive_passes = 2
    second_order.allow_truncated_daughter_modes = false
    output.directory = test-run
    output.diagnostic_every = 4
    output.checkpoint_every = 8
  )cfg";
  const auto parameters = teuk::parse_run_configuration_text(
      text, {"spin=0.6", "nr=65", "output.directory=override-run"});
  CHECK_NEAR(parameters.background.spin, 0.6, 1.0e-15);
  CHECK(parameters.grid.radial_points == 65);
  CHECK(parameters.grid.ell_max_first == 4);
  CHECK(parameters.grid.ell_max_second == 5);
  CHECK(parameters.grid.first_order_modes == std::vector<int>({-2, 2}));
  CHECK(parameters.grid.second_order_modes ==
        std::vector<int>({-4, 0, 4}));
  CHECK(parameters.method.reduction ==
        teuk::ReductionEvolution::StageConstrained);
  CHECK(parameters.initial_data.modes.size() == 2);
  CHECK(parameters.initial_data.modes[0].ell == 3);
  CHECK(parameters.initial_data.modes[0].m == 2);
  CHECK_COMPLEX_NEAR(parameters.initial_data.modes[0].amplitude,
                     teuk::Complex(2e-4, -3e-5), 0.0);
  CHECK(parameters.initial_data.modes[1].ell == 4);
  CHECK(parameters.initial_data.modes[1].m == -2);
  CHECK(parameters.second_order.enabled);
  CHECK(parameters.second_order.required_consecutive_passes == 2);
  CHECK_NEAR(parameters.second_order.normalized_constraint_tolerance, 2e-8,
             0.0);
  CHECK(parameters.output.directory == "override-run");
}

TEST_CASE("metadata documents binary snapshot ordering") {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() / "teuk-kokkos-io-test";
  teuk::RunProvenance provenance;
  provenance.time = 1.25;
  provenance.step = 17;
  provenance.time_step = 0.025;
  provenance.spin = 0.7;
  provenance.equation_bundle_sha256 = "abc123";
  provenance.git_commit = "deadbeef";
  provenance.device = "test-device";
  teuk::write_metadata(directory, {2, 3, 4, 5}, {-2, 2}, "Serial",
                       provenance);
  const std::vector<teuk::Complex> original{
      teuk::Complex(1.0, -2.0), teuk::Complex(3.0, 4.0)};
  teuk::write_complex_snapshot(directory / "snapshot_000000.bin", original);

  std::ifstream metadata_stream(directory / "metadata.txt");
  const std::string text{std::istreambuf_iterator<char>(metadata_stream), {}};
  CHECK(text.find("ordering=mode,field,radial,theta") != std::string::npos);
  CHECK(text.find("equation_bundle_sha256=abc123") != std::string::npos);
  CHECK(std::filesystem::file_size(directory / "snapshot_000000.bin") ==
        4 * sizeof(double));

  const teuk::SnapshotMetadata metadata = teuk::read_metadata(directory);
  CHECK(metadata.shape.modes == 2);
  CHECK(metadata.shape.fields == 3);
  CHECK(metadata.shape.radial == 4);
  CHECK(metadata.shape.theta == 5);
  CHECK(metadata.modes == std::vector<int>({-2, 2}));
  CHECK(metadata.backend == "Serial");
  CHECK_NEAR(metadata.provenance.time, 1.25, 1.0e-15);
  CHECK(metadata.provenance.step == 17);
  CHECK(metadata.provenance.git_commit == "deadbeef");
  const auto restarted = teuk::read_complex_snapshot(
      directory / "snapshot_000000.bin", original.size());
  CHECK(restarted == original);

  bool rejected_wrong_shape = false;
  try {
    (void)teuk::read_complex_snapshot(directory / "snapshot_000000.bin", 3);
  } catch (const std::runtime_error&) {
    rejected_wrong_shape = true;
  }
  CHECK(rejected_wrong_shape);

  std::filesystem::remove_all(directory);
}
