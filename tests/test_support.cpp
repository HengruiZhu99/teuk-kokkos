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
  teuk::write_metadata(directory, {2, 3, 4, 5}, {-2, 2}, "Serial");
  teuk::write_complex_snapshot(directory / "snapshot_000000.bin",
                               {teuk::Complex(1.0, -2.0), teuk::Complex(3.0, 4.0)});

  std::ifstream metadata(directory / "metadata.txt");
  const std::string text{std::istreambuf_iterator<char>(metadata), {}};
  CHECK(text.find("ordering=mode,field,radial,theta") != std::string::npos);
  CHECK(std::filesystem::file_size(directory / "snapshot_000000.bin") ==
        4 * sizeof(double));

  std::filesystem::remove_all(directory);
}
