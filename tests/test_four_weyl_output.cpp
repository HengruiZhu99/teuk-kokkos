#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/four_weyl_output.hpp"
#include "teuk/types.hpp"

namespace {

using View = Kokkos::View<teuk::Complex***, Kokkos::LayoutRight,
                          teuk::MemorySpace>;

int output_allocations = 0;
int output_launches = 0;

void count_output_allocation(Kokkos::Tools::SpaceHandle, const char*,
                             const void*, std::uint64_t) {
  ++output_allocations;
}

void count_output_launch(const char*, std::uint32_t, std::uint64_t*) {
  ++output_launches;
}

teuk::FourWeylOutputMetadata metadata() {
  teuk::FourWeylOutputMetadata result;
  result.git_commit = "0123456789abcdef";
  result.runtime_config_schema_version = 1;
  result.linear_method = teuk::FourWeylLinearMethod::Both;
  result.source_method = teuk::FourWeylSourceMethod::RawOrgSourcedCompanion;
  result.initial_policy = teuk::FourWeylInitialPolicy::Checkpoint;
  result.ell_max_first = 3;
  result.ell_max_second = 4;
  result.parent_modes = {-2, 2};
  result.target_modes = {-4, 0, 4};
  result.output_cadence_steps = 5;
  result.radial_discretization = teuk::RadialDiscretization::D84;
  return result;
}

teuk::FourWeylOutputPacker packer() {
  return teuk::FourWeylOutputPacker::enabled(
      metadata(), {0.0, 0.2, 0.47}, {-0.6, 0.4}, 0.91, 1.7, 0.47);
}

std::array<View, 4> fields(const double amplitude = 1.0) {
  std::array<View, 4> result{
      View("four_weyl_test_f1", 5, 3, 2),
      View("four_weyl_test_z1", 5, 3, 2),
      View("four_weyl_test_f2", 5, 3, 2),
      View("four_weyl_test_z2", 5, 3, 2),
  };
  for (std::size_t field = 0; field < result.size(); ++field) {
    auto host = Kokkos::create_mirror_view(result[field]);
    const double scaling = field < 2 ? amplitude : amplitude * amplitude;
    for (std::size_t mode = 0; mode < 5; ++mode) {
      for (std::size_t radial = 0; radial < 3; ++radial) {
        for (std::size_t theta = 0; theta < 2; ++theta) {
          const double base = 1.0 + 100.0 * static_cast<double>(field) +
                              10.0 * static_cast<double>(mode) +
                              2.0 * static_cast<double>(radial) +
                              static_cast<double>(theta);
          host(mode, radial, theta) =
              scaling * teuk::Complex(base, -0.25 * base);
        }
      }
    }
    Kokkos::deep_copy(result[field], host);
  }
  return result;
}

std::vector<teuk::FourWeylOutputRecord> pack(
    const teuk::FourWeylOutputPacker& output,
    const std::array<View, 4>& input, const std::uint64_t step = 10,
    const double time = 1.25) {
  const teuk::ExecutionSpace execution;
  return output.pack_nodal(execution, step, time, input[0], input[1], input[2],
                           input[3]);
}

std::vector<teuk::Complex> copy_values(const View& input) {
  const auto host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, input);
  std::vector<teuk::Complex> values;
  values.reserve(input.size());
  for (std::size_t mode = 0; mode < input.extent(0); ++mode) {
    for (std::size_t radial = 0; radial < input.extent(1); ++radial) {
      for (std::size_t theta = 0; theta < input.extent(2); ++theta) {
        values.push_back(host(mode, radial, theta));
      }
    }
  }
  return values;
}

}  // namespace

TEST_CASE("disabled four-Weyl output allocates and launches nothing") {
  output_allocations = 0;
  output_launches = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_output_allocation);
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      count_output_launch);
  const teuk::FourWeylOutputPacker disabled;
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  CHECK(!disabled.is_enabled());
  CHECK(!disabled.should_emit(0));
  CHECK(disabled.packed_capacity() == 0);
  CHECK(output_allocations == 0);
  CHECK(output_launches == 0);
}

TEST_CASE("four-Weyl metadata has exact roundtrip and physical warnings") {
  const auto expected = metadata();
  std::ostringstream output;
  teuk::write_four_weyl_metadata(output, expected);
  const std::string serialized = output.str();
  CHECK(serialized.find(teuk::four_weyl_minus2_scaling) != std::string::npos);
  CHECK(serialized.find(teuk::four_weyl_plus2_scaling) != std::string::npos);
  CHECK(serialized.find("is-not-a-flux-observable") != std::string::npos);
  CHECK(serialized.find("no-factorial") != std::string::npos);
  CHECK(serialized.find("metric_curvature") == std::string::npos);
  CHECK(serialized.find("both") != std::string::npos);

  std::istringstream input(serialized);
  const auto actual = teuk::read_four_weyl_metadata(input);
  CHECK(actual == expected);
  std::ostringstream second;
  teuk::write_four_weyl_metadata(second, actual);
  CHECK(second.str() == serialized);
}

TEST_CASE("four-Weyl nodal packing preserves signed ordering and scalings") {
  const auto output = packer();
  const auto input = fields();
  const auto records = pack(output, input);
  CHECK(records.size() == 60);
  CHECK(records.front().field == teuk::FourWeylField::Psi4Order1);
  CHECK(records.front().m == -2);
  CHECK(records.front().radial_index == 0);
  CHECK(records.front().theta_index == 0);
  CHECK(records[6].m == 2);
  CHECK(records[12].field == teuk::FourWeylField::Psi0Order1);
  CHECK(records[24].field == teuk::FourWeylField::Psi4Order2);
  CHECK(records[24].m == -4);
  CHECK(records[30].m == 0);
  CHECK(records[36].m == 4);
  CHECK(records[42].field == teuk::FourWeylField::Psi0Order2);
  for (const auto& record : records) {
    CHECK(record.step == 10);
    CHECK(record.time == 1.25);
    CHECK(record.ell == -1);
    if (record.boundary == teuk::FourWeylBoundary::Scri) {
      CHECK_COMPLEX_NEAR(record.raw_code_tetrad, teuk::Complex(), 0.0);
      const int expected_power = teuk::four_weyl_spin(record.field) == -2
                                     ? 1
                                     : 5;
      CHECK(record.asymptotic_power == expected_power);
      const double coefficient = expected_power == 1
                                     ? 1.0
                                     : teuk::plus2_scri_scaling_coefficient(1.7);
      CHECK_COMPLEX_NEAR(record.asymptotic_coefficient,
                         coefficient * record.regularized, 2.0e-14);
    } else {
      CHECK(record.asymptotic_power == 0);
      CHECK_COMPLEX_NEAR(record.asymptotic_coefficient, teuk::Complex(), 0.0);
      if (teuk::four_weyl_spin(record.field) == -2) {
        CHECK_COMPLEX_NEAR(record.raw_code_tetrad,
                           record.radius * record.regularized, 2.0e-14);
      } else {
        const auto scaling = teuk::plus2_code_tetrad_scaling(
            record.radius, record.cos_theta, 0.91, 1.7);
        CHECK_COMPLEX_NEAR(record.raw_code_tetrad,
                           scaling * record.regularized, 2.0e-14);
      }
    }
  }
}

TEST_CASE("four-Weyl packing has linear and quadratic amplitude scaling") {
  const auto output = packer();
  const auto unit = pack(output, fields(1.0));
  const auto scaled = pack(output, fields(0.25));
  CHECK(unit.size() == scaled.size());
  for (std::size_t i = 0; i < unit.size(); ++i) {
    const double factor = teuk::four_weyl_order(unit[i].field) == 1
                              ? 0.25
                              : 0.0625;
    CHECK_COMPLEX_NEAR(scaled[i].regularized,
                       factor * unit[i].regularized, 2.0e-13);
    CHECK_COMPLEX_NEAR(scaled[i].raw_code_tetrad,
                       factor * unit[i].raw_code_tetrad, 2.0e-13);
    CHECK_COMPLEX_NEAR(scaled[i].asymptotic_coefficient,
                       factor * unit[i].asymptotic_coefficient, 2.0e-13);
  }
}

TEST_CASE("four-Weyl device packing equals independent host packing") {
  const auto output = packer();
  const auto input = fields();
  const auto records = pack(output, input);
  const std::array<int, 5> stored_modes{-4, -2, 0, 2, 4};
  const std::array<double, 3> radii{0.0, 0.2, 0.47};
  const std::array<double, 2> cosines{-0.6, 0.4};
  for (const auto& record : records) {
    const auto mode = std::find(stored_modes.begin(), stored_modes.end(),
                                record.m);
    CHECK(mode != stored_modes.end());
    const std::size_t mode_index =
        static_cast<std::size_t>(std::distance(stored_modes.begin(), mode));
    const std::size_t field = static_cast<std::size_t>(record.field);
    const double base = 1.0 + 100.0 * static_cast<double>(field) +
                        10.0 * static_cast<double>(mode_index) +
                        2.0 * static_cast<double>(record.radial_index) +
                        static_cast<double>(record.theta_index);
    const teuk::Complex regularized(base, -0.25 * base);
    CHECK_COMPLEX_NEAR(record.regularized, regularized, 0.0);
    if (record.radial_index == 0) continue;
    teuk::Complex expected;
    if (teuk::four_weyl_spin(record.field) == -2) {
      expected = radii[static_cast<std::size_t>(record.radial_index)] *
                 regularized;
    } else {
      const teuk::Complex denominator(
          1.7 * 1.7,
          -0.91 * radii[static_cast<std::size_t>(record.radial_index)] *
              cosines[static_cast<std::size_t>(record.theta_index)]);
      const teuk::Complex denominator2 = denominator * denominator;
      expected =
          std::pow(radii[static_cast<std::size_t>(record.radial_index)], 5) /
          (denominator2 * denominator2) * regularized;
    }
    CHECK_COMPLEX_NEAR(record.raw_code_tetrad, expected, 2.0e-14);
  }
}

TEST_CASE("four-Weyl output cadence does not mutate the trajectory") {
  const auto output = packer();
  auto input = fields();
  const auto before0 = copy_values(input[0]);
  const auto before1 = copy_values(input[1]);
  const auto before2 = copy_values(input[2]);
  const auto before3 = copy_values(input[3]);
  CHECK(!output.should_emit(9));
  CHECK(output.should_emit(10));
  const auto records = pack(output, input, 10, 2.0);
  CHECK(!records.empty());
  CHECK(copy_values(input[0]) == before0);
  CHECK(copy_values(input[1]) == before1);
  CHECK(copy_values(input[2]) == before2);
  CHECK(copy_values(input[3]) == before3);
}

TEST_CASE("four-Weyl packing rejects missing nonfinite and inconsistent data") {
  const auto output = packer();
  auto input = fields();
  View wrong("four_weyl_wrong_extent", 4, 3, 2);
  bool rejected_extent = false;
  try {
    const teuk::ExecutionSpace execution;
    (void)output.pack_nodal(execution, 0, 0.0, wrong, input[1], input[2],
                            input[3]);
  } catch (const std::invalid_argument&) {
    rejected_extent = true;
  }
  CHECK(rejected_extent);

  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, input[2]);
  host(0, 0, 0) = teuk::Complex(
      std::numeric_limits<double>::quiet_NaN(), 0.0);
  Kokkos::deep_copy(input[2], host);
  bool rejected_nonfinite = false;
  try {
    (void)pack(output, input);
  } catch (const std::runtime_error&) {
    rejected_nonfinite = true;
  }
  CHECK(rejected_nonfinite);

  bool rejected_time = false;
  try {
    (void)pack(output, fields(), 0,
               std::numeric_limits<double>::infinity());
  } catch (const std::invalid_argument&) {
    rejected_time = true;
  }
  CHECK(rejected_time);
}

TEST_CASE("four-Weyl scri modal input is complete ordered and explicit") {
  const auto output = packer();
  teuk::FourWeylScriModalInput input;
  for (int field_index = 0;
       field_index < static_cast<int>(teuk::FourWeylField::Count);
       ++field_index) {
    const auto field = static_cast<teuk::FourWeylField>(field_index);
    const auto& modes = teuk::four_weyl_order(field) == 1
                            ? output.metadata().parent_modes
                            : output.metadata().target_modes;
    const int ell_max = teuk::four_weyl_order(field) == 1
                            ? output.metadata().ell_max_first
                            : output.metadata().ell_max_second;
    for (const int mode : modes) {
      for (int ell = std::max(2, std::abs(mode)); ell <= ell_max; ++ell) {
        input.fields[static_cast<std::size_t>(field_index)].push_back(
            {ell, mode,
             teuk::Complex(10.0 * field_index + ell, 0.5 * mode)});
      }
    }
  }
  const auto records = output.pack_scri_modal(15, 3.5, input);
  CHECK(!records.empty());
  for (const auto& record : records) {
    CHECK(record.step == 15);
    CHECK(record.time == 3.5);
    CHECK(record.representation ==
          teuk::FourWeylAngularRepresentation::SpinWeightedSphericalModalAtScri);
    CHECK(record.boundary == teuk::FourWeylBoundary::Scri);
    CHECK(record.ell >= std::max(2, std::abs(record.m)));
    CHECK(record.radial_index == -1);
    CHECK(record.theta_index == -1);
    CHECK_COMPLEX_NEAR(record.raw_code_tetrad, teuk::Complex(), 0.0);
  }

  input.fields[0].pop_back();
  bool rejected_missing = false;
  try {
    (void)output.pack_scri_modal(15, 3.5, input);
  } catch (const std::invalid_argument&) {
    rejected_missing = true;
  }
  CHECK(rejected_missing);
}

TEST_CASE("four-Weyl CSV has exact binary64 roundtrip") {
  const auto output = packer();
  const auto expected = pack(output, fields(), 20, 4.25);
  std::ostringstream serialized;
  teuk::write_four_weyl_csv_header(serialized);
  teuk::write_four_weyl_csv_records(serialized, expected);
  std::istringstream input(serialized.str());
  const auto actual = teuk::read_four_weyl_csv_records(input);
  CHECK(actual == expected);
  std::ostringstream second;
  teuk::write_four_weyl_csv_header(second);
  teuk::write_four_weyl_csv_records(second, actual);
  CHECK(second.str() == serialized.str());
}

TEST_CASE("four-Weyl sharp lookup uses conjugate opposite signed mode") {
  const std::vector<int> modes{-2, 0, 2};
  const std::vector<teuk::Complex> values{
      teuk::Complex(1.0, 2.0), teuk::Complex(3.0, -4.0),
      teuk::Complex(5.0, 6.0)};
  CHECK_COMPLEX_NEAR(teuk::four_weyl_sharp_value(modes, values, 2),
                     teuk::Complex(1.0, -2.0), 0.0);
  CHECK_COMPLEX_NEAR(teuk::four_weyl_sharp_value(modes, values, -2),
                     teuk::Complex(5.0, -6.0), 0.0);
  CHECK_COMPLEX_NEAR(teuk::four_weyl_sharp_value(modes, values, 0),
                     teuk::Complex(3.0, 4.0), 0.0);
}
