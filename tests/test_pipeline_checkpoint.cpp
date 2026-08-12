#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "teuk/pipeline_checkpoint.hpp"
#include "teuk/spatial_pipeline.hpp"

namespace {

class TemporaryTestDirectory {
 public:
  TemporaryTestDirectory() {
    static std::atomic<std::uint64_t> sequence{0};
    const auto tick = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    for (std::uint64_t attempt = 0; attempt < 100; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("teuk-pipeline-checkpoint-test-" + std::to_string(tick) +
               "-" + std::to_string(sequence.fetch_add(1)));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) return;
    }
    throw std::runtime_error("cannot create checkpoint test directory");
  }

  TemporaryTestDirectory(const TemporaryTestDirectory&) = delete;
  TemporaryTestDirectory& operator=(const TemporaryTestDirectory&) = delete;

  ~TemporaryTestDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

teuk::PipelineCheckpointConfiguration checkpoint_configuration() {
  return {{1.0, 0.27, 1.5},
          3,
          3,
          6,
          0.08,
          0.004,
          teuk::ReductionEvolution::FreeDamped,
          2.0e-5,
          {teuk::SecondOrderSourceMode::Unrestricted, 3.0e-5, 0.0, 1}};
}

void initialize_pipeline(const teuk::ExecutionSpace& execution,
                         teuk::SpatialPipeline& pipeline,
                         const teuk::ModeRegistry& registry,
                         const int ell_max) {
  auto host = Kokkos::create_mirror_view(pipeline.storage().state());
  for (std::size_t mode = 0; mode < host.extent(0); ++mode) {
    for (std::size_t field = 0; field < host.extent(1); ++field) {
      for (std::size_t radial = 0; radial < host.extent(2); ++radial) {
        for (std::size_t theta = 0; theta < host.extent(3); ++theta) {
          const double seed = 1.0 + 0.11 * static_cast<double>(mode) +
                              0.031 * static_cast<double>(field) +
                              0.007 * static_cast<double>(radial) +
                              0.003 * static_cast<double>(theta);
          host(mode, field, radial, theta) =
              field < static_cast<std::size_t>(teuk::PipelineField::SecondP)
                  ? teuk::Complex(2.0e-5 * seed, -1.3e-5 * seed)
                  : teuk::Complex(0.0, 0.0);
        }
      }
    }
  }
  teuk::project_pipeline_state_to_retained_bands(
      host, registry, {ell_max, ell_max});
  Kokkos::deep_copy(execution, pipeline.storage().state(), host);
}

std::vector<teuk::Complex> pipeline_state(
    const teuk::ExecutionSpace& execution,
    const teuk::SpatialPipeline& pipeline) {
  Kokkos::View<teuk::Complex*, Kokkos::HostSpace> host(
      "checkpoint_test_state", pipeline.storage().value_count());
  Kokkos::deep_copy(execution, host, pipeline.storage().flat_state());
  execution.fence("copy checkpoint test state");
  std::vector<teuk::Complex> result(host.extent(0));
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = host(i);
  return result;
}

void set_pipeline_state(const teuk::ExecutionSpace& execution,
                        teuk::SpatialPipeline& pipeline,
                        const teuk::Complex value) {
  Kokkos::deep_copy(execution, pipeline.storage().state(), value);
  execution.fence("set checkpoint test state");
}

template <class Function>
bool rejects(Function&& function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

void copy_checkpoint(const std::filesystem::path& source,
                     const std::filesystem::path& destination) {
  std::filesystem::copy(
      source, destination,
      std::filesystem::copy_options::recursive |
          std::filesystem::copy_options::copy_symlinks);
}

void replace_metadata_line(const std::filesystem::path& path,
                           const std::string& key,
                           const std::string& replacement) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read test metadata");
  std::vector<std::string> lines;
  std::string line;
  bool replaced = false;
  while (std::getline(input, line)) {
    if (line.starts_with(key + "=")) {
      line = key + "=" + replacement;
      replaced = true;
    }
    lines.push_back(line);
  }
  if (!replaced) throw std::runtime_error("test metadata key not found");
  input.close();
  std::ofstream output(path, std::ios::trunc);
  for (const std::string& output_line : lines) output << output_line << '\n';
  if (!output) throw std::runtime_error("cannot write test metadata");
}

void remove_metadata_line(const std::filesystem::path& path,
                          const std::string& key) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot read test metadata");
  std::vector<std::string> lines;
  std::string line;
  bool removed = false;
  while (std::getline(input, line)) {
    if (line.starts_with(key + "=")) {
      removed = true;
    } else {
      lines.push_back(line);
    }
  }
  if (!removed) throw std::runtime_error("test metadata key not found");
  input.close();
  std::ofstream output(path, std::ios::trunc);
  for (const std::string& output_line : lines) output << output_line << '\n';
  if (!output) throw std::runtime_error("cannot write test metadata");
}

std::string serialize_double_list(const std::vector<double>& values) {
  std::ostringstream output;
  output << std::setprecision(std::numeric_limits<double>::max_digits10);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) output << ',';
    output << values[i];
  }
  return output.str();
}

}  // namespace

TEST_CASE("full spatial pipeline checkpoint resumes the real RK4 trajectory") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({1, -1, 0}, {1, 0, -1});
  const teuk::UniformRadialGrid radial_grid(23, 0.0, 0.8);
  auto configuration = checkpoint_configuration();
  configuration.radial_discretization = teuk::RadialDiscretization::D105;
  teuk::SpatialPipeline uninterrupted(
      execution, registry, radial_grid,
      teuk::PipelineAngularBands{configuration.ell_max_first,
                                 configuration.ell_max_second},
      configuration.theta_nodes, configuration.background,
      configuration.reduction_damping, configuration.dissipation,
      configuration.reduction, "checkpoint_uninterrupted",
      configuration.source_policy, configuration.radial_discretization);
  teuk::SpatialPipeline interrupted(
      execution, registry, radial_grid,
      teuk::PipelineAngularBands{configuration.ell_max_first,
                                 configuration.ell_max_second},
      configuration.theta_nodes, configuration.background,
      configuration.reduction_damping, configuration.dissipation,
      configuration.reduction, "checkpoint_interrupted",
      configuration.source_policy, configuration.radial_discretization);
  initialize_pipeline(execution, uninterrupted, registry,
                      configuration.ell_max_first);
  initialize_pipeline(execution, interrupted, registry,
                      configuration.ell_max_first);

  constexpr std::uint64_t total_steps = 4;
  constexpr std::uint64_t checkpoint_step = 2;
  for (std::uint64_t step = 0; step < total_steps; ++step) {
    uninterrupted.step(execution,
                       static_cast<double>(step) * configuration.time_step,
                       configuration.time_step);
  }
  for (std::uint64_t step = 0; step < checkpoint_step; ++step) {
    interrupted.step(execution,
                     static_cast<double>(step) * configuration.time_step,
                     configuration.time_step);
  }
  execution.fence("finish checkpoint prefix trajectories");
  const auto state_at_checkpoint = pipeline_state(execution, interrupted);

  TemporaryTestDirectory temporary;
  const auto checkpoint = temporary.path() / "checkpoint-000002";
  const auto written_identity = teuk::write_pipeline_checkpoint(
      execution, checkpoint, interrupted, registry, configuration,
      {checkpoint_step * configuration.time_step, checkpoint_step});
  teuk::validate_primary_checkpoint_content_identity(
      written_identity.content_identity());
  CHECK(written_identity.content_identity() ==
        teuk::inspect_pipeline_checkpoint_content_identity(checkpoint));
  CHECK(std::filesystem::is_regular_file(
      checkpoint / teuk::pipeline_checkpoint_metadata_file));
  CHECK(std::filesystem::is_regular_file(
      checkpoint / teuk::pipeline_checkpoint_state_file));
  CHECK(rejects([&] {
    teuk::write_pipeline_checkpoint(
        execution, checkpoint, interrupted, registry, configuration,
        {checkpoint_step * configuration.time_step, checkpoint_step});
  }));

  const auto metadata = teuk::read_pipeline_checkpoint_metadata(checkpoint);
  CHECK(metadata.version == teuk::pipeline_checkpoint_format_version);
  CHECK(metadata.shape.modes == 3);
  CHECK(metadata.shape.fields == teuk::point_pipeline_field_count);
  CHECK(metadata.shape.radial == radial_grid.size());
  CHECK(metadata.shape.theta ==
        static_cast<std::uint64_t>(configuration.theta_nodes));
  CHECK(metadata.modes == std::vector<int>({-1, 0, 1}));
  CHECK(metadata.targets == std::vector<int>({-1, 0, 1}));
  CHECK(metadata.parents == std::vector<int>({-1, 0, 1}));
  CHECK(metadata.radial_coordinates.front() == radial_grid.lower_radius());
  CHECK_NEAR(metadata.radial_coordinates.back(), radial_grid.upper_radius(),
             2.0e-16);
  CHECK(metadata.configuration.background.mass ==
        configuration.background.mass);
  CHECK(metadata.configuration.background.spin ==
        configuration.background.spin);
  CHECK(metadata.configuration.reduction == configuration.reduction);
  CHECK(metadata.configuration.radial_discretization ==
        teuk::RadialDiscretization::D105);
  CHECK(metadata.configuration.time_step == configuration.time_step);
  CHECK(metadata.configuration.source_policy.mode ==
        configuration.source_policy.mode);
  CHECK(metadata.configuration.source_policy.source_start_time ==
        configuration.source_policy.source_start_time);
  CHECK(metadata.configuration.source_policy.normalized_constraint_tolerance ==
        configuration.source_policy.normalized_constraint_tolerance);
  CHECK(metadata.configuration.source_policy.required_consecutive_passes ==
        configuration.source_policy.required_consecutive_passes);
  CHECK(metadata.progress.step == checkpoint_step);
  CHECK(metadata.progress.time ==
        checkpoint_step * configuration.time_step);
  CHECK(metadata.source_activation.active);
  CHECK_NEAR(metadata.source_activation.activation_time, 3.0e-5, 0.0);
  CHECK(metadata.source_activation.consecutive_passes == 1);

  const teuk::ModeRegistry restored_registry(metadata.modes, metadata.parents,
                                             metadata.targets);
  const teuk::UniformRadialGrid restored_grid(
      metadata.radial_coordinates.size(), metadata.radial_coordinates.front(),
      metadata.radial_coordinates.back());
  teuk::SpatialPipeline restarted(
      execution, restored_registry, restored_grid,
      teuk::PipelineAngularBands{metadata.configuration.ell_max_first,
                                 metadata.configuration.ell_max_second},
      metadata.configuration.theta_nodes,
      metadata.configuration.background,
      metadata.configuration.reduction_damping,
      metadata.configuration.dissipation, metadata.configuration.reduction,
      "checkpoint_restarted", metadata.configuration.source_policy,
      metadata.configuration.radial_discretization);
  const auto loaded = teuk::load_pipeline_checkpoint_verified(
      execution, checkpoint, restarted, restored_registry,
      metadata.configuration);
  CHECK(loaded.receipt == written_identity);
  const auto loaded_state = pipeline_state(execution, restarted);
  CHECK(loaded_state == state_at_checkpoint);
  CHECK(restarted.source_activation_state().active);
  CHECK_NEAR(restarted.source_activation_state().activation_time, 3.0e-5,
             0.0);

  for (std::uint64_t step = loaded.metadata.progress.step; step < total_steps;
       ++step) {
    restarted.step(execution, static_cast<double>(step) *
                                  loaded.metadata.configuration.time_step,
                   loaded.metadata.configuration.time_step);
  }
  execution.fence("finish restarted checkpoint trajectory");
  const auto expected = pipeline_state(execution, uninterrupted);
  const auto actual = pipeline_state(execution, restarted);
  CHECK(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i) {
    CHECK_COMPLEX_NEAR(actual[i], expected[i], 2.0e-14);
  }
}

TEST_CASE("pipeline checkpoint rejects malformed truncated and mismatched data") {
  const teuk::ExecutionSpace execution;
  const teuk::ModeRegistry registry({-1, 0, 1}, {-1, 0, 1});
  const teuk::UniformRadialGrid radial_grid(9, 0.0, 0.8);
  const auto configuration = checkpoint_configuration();
  teuk::SpatialPipeline pipeline(
      execution, registry, radial_grid,
      teuk::PipelineAngularBands{configuration.ell_max_first,
                                 configuration.ell_max_second},
      configuration.theta_nodes, configuration.background,
      configuration.reduction_damping, configuration.dissipation,
      configuration.reduction, "checkpoint_rejection",
      configuration.source_policy);
  initialize_pipeline(execution, pipeline, registry,
                      configuration.ell_max_first);
  const auto valid_state = pipeline_state(execution, pipeline);

  TemporaryTestDirectory temporary;
  const auto valid = temporary.path() / "valid";
  teuk::write_pipeline_checkpoint(execution, valid, pipeline, registry,
                                  configuration, {0.0, 0});

  // IntelLLVM may reassociate lower+h*i while the validator recovers h from
  // the two serialized endpoints. One-ULP coordinate differences are valid
  // binary64 uniform grids; a material perturbation must still fail closed.
  const auto valid_metadata =
      teuk::read_pipeline_checkpoint_metadata(valid);
  const auto one_ulp = temporary.path() / "one-ulp-uniform";
  copy_checkpoint(valid, one_ulp);
  auto one_ulp_coordinates = valid_metadata.radial_coordinates;
  one_ulp_coordinates[4] = std::nextafter(
      one_ulp_coordinates[4], std::numeric_limits<double>::infinity());
  replace_metadata_line(one_ulp / teuk::pipeline_checkpoint_metadata_file,
                        "radial_coordinates",
                        serialize_double_list(one_ulp_coordinates));
  const auto accepted_one_ulp =
      teuk::read_pipeline_checkpoint_metadata(one_ulp);
  CHECK(accepted_one_ulp.radial_coordinates == one_ulp_coordinates);

  const auto nonuniform = temporary.path() / "nonuniform-grid";
  copy_checkpoint(valid, nonuniform);
  auto nonuniform_coordinates = valid_metadata.radial_coordinates;
  nonuniform_coordinates[4] += 1.0e-8;
  replace_metadata_line(nonuniform / teuk::pipeline_checkpoint_metadata_file,
                        "radial_coordinates",
                        serialize_double_list(nonuniform_coordinates));
  CHECK(rejects([&] {
    (void)teuk::read_pipeline_checkpoint_metadata(nonuniform);
  }));

  const auto legacy = temporary.path() / "legacy-v4";
  copy_checkpoint(valid, legacy);
  replace_metadata_line(legacy / teuk::pipeline_checkpoint_metadata_file,
                        "version", "4");
  remove_metadata_line(legacy / teuk::pipeline_checkpoint_metadata_file,
                       "radial_discretization");
  const auto legacy_metadata =
      teuk::read_pipeline_checkpoint_metadata(legacy);
  CHECK(legacy_metadata.version ==
        teuk::pipeline_checkpoint_legacy_d42_version);
  CHECK(legacy_metadata.configuration.radial_discretization ==
        teuk::RadialDiscretization::D42);
  set_pipeline_state(execution, pipeline, teuk::Complex(-0.2, 0.3));
  static_cast<void>(teuk::load_pipeline_checkpoint(
      execution, legacy, pipeline, registry, configuration));
  CHECK(pipeline_state(execution, pipeline) == valid_state);

  const teuk::UniformRadialGrid d84_grid(17, 0.0, 0.8);
  teuk::SpatialPipeline d84_pipeline(
      execution, registry, d84_grid,
      teuk::PipelineAngularBands{configuration.ell_max_first,
                                 configuration.ell_max_second},
      configuration.theta_nodes, configuration.background,
      configuration.reduction_damping, configuration.dissipation,
      configuration.reduction, "legacy_v4_d84_rejection",
      configuration.source_policy, teuk::RadialDiscretization::D84);
  const teuk::Complex d84_sentinel(0.41, -0.19);
  set_pipeline_state(execution, d84_pipeline, d84_sentinel);
  const auto d84_before = pipeline_state(execution, d84_pipeline);
  auto d84_configuration = configuration;
  d84_configuration.radial_discretization =
      teuk::RadialDiscretization::D84;
  CHECK(rejects([&] {
    (void)teuk::load_pipeline_checkpoint(
        execution, legacy, d84_pipeline, registry, d84_configuration);
  }));
  CHECK(pipeline_state(execution, d84_pipeline) == d84_before);

  const teuk::UniformRadialGrid d105_grid(23, 0.0, 0.8);
  teuk::SpatialPipeline d105_pipeline(
      execution, registry, d105_grid,
      teuk::PipelineAngularBands{configuration.ell_max_first,
                                 configuration.ell_max_second},
      configuration.theta_nodes, configuration.background,
      configuration.reduction_damping, configuration.dissipation,
      configuration.reduction, "legacy_v4_d105_rejection",
      configuration.source_policy, teuk::RadialDiscretization::D105);
  const teuk::Complex d105_sentinel(-0.37, 0.23);
  set_pipeline_state(execution, d105_pipeline, d105_sentinel);
  const auto d105_before = pipeline_state(execution, d105_pipeline);
  auto d105_configuration = configuration;
  d105_configuration.radial_discretization =
      teuk::RadialDiscretization::D105;
  CHECK(rejects([&] {
    (void)teuk::load_pipeline_checkpoint(
        execution, legacy, d105_pipeline, registry, d105_configuration);
  }));
  CHECK(pipeline_state(execution, d105_pipeline) == d105_before);

  const auto off_band = temporary.path() / "off-band";
  auto off_band_host = Kokkos::create_mirror_view(pipeline.storage().state());
  Kokkos::deep_copy(off_band_host, pipeline.storage().state());
  off_band_host(0, static_cast<std::size_t>(teuk::PipelineField::FirstP), 0,
                0) += teuk::Complex(0.25, -0.1);
  Kokkos::deep_copy(execution, pipeline.storage().state(), off_band_host);
  teuk::write_pipeline_checkpoint(execution, off_band, pipeline, registry,
                                  configuration, {0.0, 0});

  const auto malformed = temporary.path() / "malformed";
  copy_checkpoint(valid, malformed);
  replace_metadata_line(
      malformed / teuk::pipeline_checkpoint_metadata_file, "version", "1");
  CHECK(rejects(
      [&] { (void)teuk::read_pipeline_checkpoint_metadata(malformed); }));

  const auto mismatched = temporary.path() / "mismatched";
  copy_checkpoint(valid, mismatched);
  replace_metadata_line(
      mismatched / teuk::pipeline_checkpoint_metadata_file, "radial", "10");
  CHECK(rejects(
      [&] { (void)teuk::read_pipeline_checkpoint_metadata(mismatched); }));

  const teuk::Complex sentinel(0.125, -0.375);
  set_pipeline_state(execution, pipeline, sentinel);
  const auto before_rejections = pipeline_state(execution, pipeline);

  CHECK(rejects([&] {
    (void)teuk::load_pipeline_checkpoint(execution, off_band, pipeline,
                                         registry, configuration);
  }));
  CHECK(pipeline_state(execution, pipeline) == before_rejections);

  const auto truncated = temporary.path() / "truncated";
  copy_checkpoint(valid, truncated);
  const auto truncated_state =
      truncated / teuk::pipeline_checkpoint_state_file;
  const auto original_size = std::filesystem::file_size(truncated_state);
  std::filesystem::resize_file(truncated_state, original_size - sizeof(double));
  CHECK(rejects([&] {
    (void)teuk::load_pipeline_checkpoint(execution, truncated, pipeline,
                                         registry, configuration);
  }));
  CHECK(pipeline_state(execution, pipeline) == before_rejections);

  auto wrong_configuration = configuration;
  wrong_configuration.dissipation += 0.01;
  CHECK(rejects([&] {
    (void)teuk::load_pipeline_checkpoint(execution, valid, pipeline, registry,
                                         wrong_configuration);
  }));
  CHECK(pipeline_state(execution, pipeline) == before_rejections);

  wrong_configuration = configuration;
  wrong_configuration.source_policy.source_start_time += 0.01;
  CHECK(rejects([&] {
    (void)teuk::load_pipeline_checkpoint(execution, valid, pipeline, registry,
                                         wrong_configuration);
  }));
  CHECK(pipeline_state(execution, pipeline) == before_rejections);

  const auto corrupt = temporary.path() / "corrupt";
  copy_checkpoint(valid, corrupt);
  std::fstream binary(corrupt / teuk::pipeline_checkpoint_state_file,
                      std::ios::binary | std::ios::in | std::ios::out);
  char byte = 0;
  binary.read(&byte, 1);
  byte ^= 0x1;
  binary.seekp(0);
  binary.write(&byte, 1);
  binary.close();
  CHECK(rejects([&] {
    (void)teuk::load_pipeline_checkpoint(execution, corrupt, pipeline,
                                         registry, configuration);
  }));
  CHECK(pipeline_state(execution, pipeline) == before_rejections);
}
