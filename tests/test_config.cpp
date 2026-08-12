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
  CHECK(defaults.initial_data.modes.size() == 1);

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
    second_order.enabled = true
    second_order.source_mode = unrestricted
    second_order.source_start_time = 0.125
    second_order.constraint_tolerance = 3e-9
    second_order.required_consecutive_passes = 3
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
  CHECK_COMPLEX_NEAR(roundtrip.initial_data.modes[0].amplitude,
                     configured.initial_data.modes[0].amplitude, 0.0);
  CHECK(roundtrip.second_order.source_mode ==
        teuk::SecondOrderSourceMode::Unrestricted);
  CHECK(roundtrip.second_order.required_consecutive_passes == 3);
  CHECK(roundtrip.second_order.normalized_constraint_tolerance == 3e-9);
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
