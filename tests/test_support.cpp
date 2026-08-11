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

TEST_CASE("key value parameters are strict and validated") {
  teuk::Parameters parameters;
  teuk::apply_key_value(parameters, "spin=0.99");
  teuk::apply_key_value(parameters, "nr=256");
  teuk::apply_key_value(parameters, "steps=64");
  teuk::validate(parameters);
  CHECK_NEAR(parameters.spin, 0.99, 1.0e-15);
  CHECK(parameters.radial_points == 256);
  CHECK(parameters.steps == 64);

  bool rejected = false;
  try {
    teuk::apply_key_value(parameters, "mystery=1");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
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
