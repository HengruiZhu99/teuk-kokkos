#include "test_harness.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "teuk/config.hpp"
#include "teuk/initial_data_factory.hpp"

namespace {

template <class Function>
bool config_rejects(Function&& function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE("runtime configuration rejects duplicate unknown and malformed values") {
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nconfig_version = 1\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nspni = 0.7\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nspin = 0.7oops\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 2\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\ninitial_data.add_sharp_partner = maybe\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\ninitial_data.mode.1.ell = 3\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\n", {"spin=0.2", "spin=0.3"});
  }));
}

TEST_CASE("runtime configuration defaults fallback and resolved round trip") {
  const auto defaults =
      teuk::parse_run_configuration_text("config_version = 1\n");
  CHECK(defaults.background.mass == 1.0);
  CHECK(defaults.grid.ell_max_first == 4);
  CHECK(defaults.grid.ell_max_second == defaults.grid.ell_max_first);
  CHECK(!defaults.second_order.enabled);
  CHECK(!defaults.plus2.enabled);
  CHECK(defaults.plus2.mode == teuk::Plus2RunMode::Disabled);
  CHECK(defaults.plus2.ell_max_first == defaults.grid.ell_max_first);
  CHECK(defaults.plus2.ell_max_second == defaults.grid.ell_max_second);
  CHECK(defaults.initial_data.modes.size() == 1);
  CHECK(defaults.method.radial_discretization ==
        teuk::RadialDiscretization::D42);

  const auto configured = teuk::parse_run_configuration_text(R"cfg(
    config_version = 1
    spin = 0.55
    ellmax_first = 5
    ntheta = 9
    first_order_modes = -2,2
    second_order_modes = -4,0,4
    initial_data.mode.0.ell = 4
    initial_data.mode.0.m = 2
    initial_data.mode.0.amplitude_real = 1.25e-4
    initial_data.mode.0.amplitude_imag = -2.5e-5
    initial_data.compact_support = true
    radial_discretization = d8-4
    nr = 17
    second_order.enabled = true
    second_order.source_mode = unrestricted
    second_order.source_start_time = 0.125
    second_order.constraint_tolerance = 3e-9
    second_order.required_consecutive_passes = 3
    plus2.enabled = true
    plus2.mode = concurrent
    plus2.linear.method = both
    plus2.linear.evolve_validation = true
    plus2.second.method = sourced_companion
    plus2.second.initial_policy = checkpoint
    plus2.second.checkpoint = companion-start.bin
    plus2.ell_max_first = 5
    plus2.ell_max_second = 5
    plus2.output.regularized = false
    plus2.output.physical_tetrad_field = true
    plus2.output.source_families = true
    plus2.output.ordered_pairs = true
    output.directory = roundtrip-output
  )cfg");
  CHECK(configured.grid.ell_max_second == 5);
  const std::string resolved = teuk::resolved_configuration_text(configured);
  const auto roundtrip = teuk::parse_run_configuration_text(resolved);
  CHECK(roundtrip.background.spin == configured.background.spin);
  CHECK(roundtrip.grid.ell_max_first == configured.grid.ell_max_first);
  CHECK(roundtrip.grid.ell_max_second == configured.grid.ell_max_second);
  CHECK(roundtrip.grid.first_order_modes ==
        configured.grid.first_order_modes);
  CHECK(roundtrip.grid.second_order_modes ==
        configured.grid.second_order_modes);
  CHECK(roundtrip.initial_data.modes.size() == 1);
  CHECK(roundtrip.initial_data.compact_support);
  CHECK(roundtrip.method.radial_discretization ==
        teuk::RadialDiscretization::D84);
  CHECK_COMPLEX_NEAR(roundtrip.initial_data.modes[0].amplitude,
                     configured.initial_data.modes[0].amplitude, 0.0);
  CHECK(roundtrip.second_order.source_mode ==
        teuk::SecondOrderSourceMode::Unrestricted);
  CHECK(roundtrip.second_order.required_consecutive_passes == 3);
  CHECK(roundtrip.second_order.normalized_constraint_tolerance == 3e-9);
  CHECK(roundtrip.plus2.enabled);
  CHECK(roundtrip.plus2.mode == teuk::Plus2RunMode::Concurrent);
  CHECK(roundtrip.plus2.linear_method == teuk::Plus2LinearMethod::Both);
  CHECK(roundtrip.plus2.linear_evolve_validation);
  CHECK(roundtrip.plus2.second_method ==
        teuk::Plus2SecondMethod::SourcedCompanion);
  CHECK(roundtrip.plus2.second_initial_policy ==
        teuk::Plus2InitialPolicy::Checkpoint);
  CHECK(roundtrip.plus2.second_checkpoint == "companion-start.bin");
  CHECK(roundtrip.plus2.ell_max_first == 5);
  CHECK(roundtrip.plus2.ell_max_second == 5);
  CHECK(!roundtrip.plus2.output.regularized);
  CHECK(roundtrip.plus2.output.physical_tetrad_field);
  CHECK(roundtrip.plus2.output.source_families);
  CHECK(roundtrip.plus2.output.ordered_pairs);
}

TEST_CASE("runtime radial discretization is strict and enforces its grid") {
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nradial_discretization = d6-3\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nradial_discretization = d8-4\nnr = 15\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nradial_discretization = d10-5\nnr = 21\n");
  }));
  const auto parameters = teuk::parse_run_configuration_text(
      "config_version = 1\nradial_discretization = d8-4\nnr = 16\n");
  CHECK(parameters.method.radial_discretization ==
        teuk::RadialDiscretization::D84);
  CHECK(teuk::resolved_configuration_text(parameters).find(
            "radial_discretization = d8-4\n") != std::string::npos);
  const auto d105 = teuk::parse_run_configuration_text(
      "config_version = 1\nradial_discretization = d10-5\nnr = 22\n");
  CHECK(d105.method.radial_discretization ==
        teuk::RadialDiscretization::D105);
  CHECK(teuk::resolved_configuration_text(d105).find(
            "radial_discretization = d10-5\n") != std::string::npos);
}

TEST_CASE("runtime radial dissipation bound rejects unsafe D10-5 steps") {
  const std::string near_extremal =
      "config_version = 1\n"
      "spin = 0.999\n"
      "nr = 513\n"
      "ntheta = 9\n"
      "ellmax_first = 4\n"
      "ellmax_second = 4\n"
      "final_time = 200\n"
      "steps = 400000\n"
      "dissipation = 0.005\n";
  const auto d84 = teuk::parse_run_configuration_text(
      near_extremal + "radial_discretization = d8-4\n");
  CHECK(d84.method.radial_discretization ==
        teuk::RadialDiscretization::D84);
  CHECK(teuk::resolved_configuration_text(d84).find(
            "radial_discretization = d8-4\n") != std::string::npos);
  CHECK(config_rejects([&] {
    (void)teuk::parse_run_configuration_text(
        near_extremal + "radial_discretization = d10-5\n");
  }));
  const auto safe_d105 = teuk::parse_run_configuration_text(
      near_extremal + "radial_discretization = d10-5\n",
      {"steps=1600000"});
  CHECK(safe_d105.method.radial_discretization ==
        teuk::RadialDiscretization::D105);
  CHECK(teuk::resolved_configuration_text(safe_d105).find(
            "steps = 1600000\n") != std::string::npos);

  CHECK_NEAR(teuk::radial_dissipation_spectral_radius_bound(
                 teuk::RadialDiscretization::D42),
             64.0, 0.0);
  CHECK_NEAR(teuk::radial_dissipation_spectral_radius_bound(
                 teuk::RadialDiscretization::D84),
             1371.0, 0.0);
  CHECK_NEAR(teuk::radial_dissipation_spectral_radius_bound(
                 teuk::RadialDiscretization::D105),
             5500.0, 0.0);
}

TEST_CASE("plus2 runtime settings reject unknown and incompatible combinations") {
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nplus2.mode = concurrent\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nplus2.enabled = true\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nplus2.linear.method = curvatureish\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nplus2.second.method = transformed_psi4\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nplus2.second.initial_policy = checkpoint\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nplus2.second.checkpoint = state.bin\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(R"cfg(
      config_version = 1
      plus2.second.initial_policy = checkpoint
      plus2.second.checkpoint = state.bin
    )cfg");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(R"cfg(
      config_version = 1
      plus2.enabled = true
      plus2.mode = diagnostic_only
      plus2.second.initial_policy = checkpoint
      plus2.second.checkpoint = state.bin
    )cfg");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(R"cfg(
      config_version = 1
      plus2.enabled = true
      plus2.mode = concurrent
    )cfg");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(R"cfg(
      config_version = 1
      plus2.enabled = true
      plus2.mode = diagnostic_only
      plus2.output.source_families = false
      plus2.output.ordered_pairs = true
    )cfg");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(R"cfg(
      config_version = 1
      plus2.enabled = true
      plus2.mode = diagnostic_only
      plus2.output.regularized = false
      plus2.output.physical_tetrad_field = false
    )cfg");
  }));
}

TEST_CASE("plus2 runtime settings validate inherited registries and bands") {
  const auto valid = teuk::parse_run_configuration_text(R"cfg(
    config_version = 1
    plus2.ell_max_first = 2
    plus2.ell_max_second = 2
  )cfg");
  CHECK(valid.plus2.ell_max_first == 2);
  CHECK(valid.plus2.ell_max_second == 2);

  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(
        "config_version = 1\nplus2.ell_max_first = 1\n");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(R"cfg(
      config_version = 1
      second_order_modes = -3,3
      plus2.ell_max_second = 2
    )cfg");
  }));
  CHECK(config_rejects([] {
    (void)teuk::parse_run_configuration_text(R"cfg(
      config_version = 1
      plus2.enabled = true
      plus2.mode = diagnostic_only
      plus2.ell_max_first = 6
      plus2.ell_max_second = 6
    )cfg");
  }));
}

TEST_CASE("initial-data factory adds sharp partners only when requested") {
  teuk::InitialDataParameters initial;
  initial.modes = {{3, 2, teuk::Complex(0.4, -0.3)},
                   {4, 0, teuk::Complex(-0.2, 0.1)}};
  initial.add_sharp_partner = false;
  const auto explicit_only = teuk::expanded_gaussian_modes(initial);
  CHECK(explicit_only.size() == 2);
  initial.add_sharp_partner = true;
  const auto with_partner = teuk::expanded_gaussian_modes(initial);
  CHECK(with_partner.size() == 3);
  CHECK(with_partner[0].m == -2);
  CHECK_COMPLEX_NEAR(with_partner[0].amplitude,
                     Kokkos::conj(teuk::Complex(0.4, -0.3)), 0.0);
}
