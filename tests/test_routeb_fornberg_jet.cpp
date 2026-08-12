#include "test_harness.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "teuk/routeb_fornberg.hpp"
#include "teuk/routeb_radial_taylor_jet.hpp"

namespace {

using C = teuk::Complex;

int routeb_allocations = 0;
int routeb_fences = 0;

void count_routeb_allocation(Kokkos::Tools::SpaceHandle, const char*,
                             const void*, std::uint64_t) {
  ++routeb_allocations;
}

void count_routeb_fence(const char*, std::uint32_t, std::uint64_t*) {
  ++routeb_fences;
}

template <class Function>
bool throws_invalid_argument(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

long double factorial(const int value) {
  long double result = 1.0L;
  for (int factor = 2; factor <= value; ++factor) result *= factor;
  return result;
}

TEST_CASE("Route-B generated Fornberg table has pinned exact moments") {
  CHECK(std::string(teuk::routeb_fornberg_weights_sha256) ==
        "b6114660d48020bc2d18313f0efc561bd05ed5c5a4386d4b6e129e118d8b5c5a");
  for (int derivative = 1; derivative <= 4; ++derivative) {
    for (std::size_t row = 0; row < teuk::routeb_fornberg_window; ++row) {
      for (int power = 0; power < 9; ++power) {
        long double moment = 0.0L;
        long double absolute_sum = 0.0L;
        for (std::size_t column = 0; column < teuk::routeb_fornberg_window;
             ++column) {
          const long double offset = static_cast<long double>(column) -
                                     static_cast<long double>(row);
          const long double term =
              static_cast<long double>(
                  teuk::routeb_fornberg_weight(derivative - 1, row, column)) *
              std::pow(offset, power);
          moment += term;
          absolute_sum += std::abs(term);
        }
        const long double expected =
            power == derivative ? factorial(derivative) : 0.0L;
        // The generator's Fraction oracle checks these identities exactly.
        // This companion check bounds cancellation after the rational entries
        // have intentionally been emitted as binary64 divisions.
        const long double roundoff_bound =
            64.0L * std::numeric_limits<double>::epsilon() *
            std::max(1.0L, absolute_sum);
        CHECK(std::abs(moment - expected) <= roundoff_bound);
      }
    }
  }
}

TEST_CASE("Route-B Fornberg coefficient noise norms are pinned") {
  const std::array<double, 4> expected_l1{
      78.0190476190476261, 357.587301587301567,
      955.733333333333348, 1740.79999999999995};
  const std::array<double, 4> expected_l2{
      32.7203410300103954, 154.320425365676925,
      418.116481851449294, 767.217558304574595};
  for (std::size_t derivative = 0; derivative < 4; ++derivative) {
    double maximum_l1 = 0.0;
    double maximum_l2 = 0.0;
    for (std::size_t row = 0; row < 9; ++row) {
      double l1 = 0.0;
      double l2_squared = 0.0;
      for (std::size_t column = 0; column < 9; ++column) {
        const double weight =
            teuk::routeb_fornberg_weight(derivative, row, column);
        l1 += std::abs(weight);
        l2_squared += weight * weight;
      }
      maximum_l1 = std::max(maximum_l1, l1);
      maximum_l2 = std::max(maximum_l2, std::sqrt(l2_squared));
    }
    CHECK_NEAR(maximum_l1,
               teuk::routeb_fornberg_maximum_l1_norm[derivative], 5.0e-13);
    CHECK_NEAR(maximum_l2,
               teuk::routeb_fornberg_maximum_l2_norm[derivative], 2.0e-13);
    CHECK_NEAR(maximum_l1, expected_l1[derivative], 2.0e-12);
    CHECK_NEAR(maximum_l2, expected_l2[derivative], 2.0e-12);
  }
}

double derivative_error(const std::size_t points, const int derivative) {
  constexpr double left = -0.23;
  constexpr double right = 0.71;
  constexpr double alpha = 1.37;
  constexpr double beta = 2.41;
  const double spacing = (right - left) / static_cast<double>(points - 1);
  std::vector<C> values(points);
  for (std::size_t index = 0; index < points; ++index) {
    const double x = left + spacing * static_cast<double>(index);
    values[index] = C(std::exp(alpha * x), std::sin(beta * x));
  }
  double maximum = 0.0;
  for (std::size_t index = 0; index < points; ++index) {
    const double x = left + spacing * static_cast<double>(index);
    const C expected(
        std::pow(alpha, derivative) * std::exp(alpha * x),
        std::pow(beta, derivative) *
            std::sin(beta * x + derivative * std::acos(-1.0) / 2.0));
    const C actual = teuk::routeb_fornberg_direct_derivative_at(
        derivative, values.data(), points, index, 1.0 / spacing);
    maximum = std::max(maximum, Kokkos::abs(actual - expected));
  }
  return maximum;
}

TEST_CASE("Route-B direct D1 through D4 converge endpoint-inclusively") {
  for (int derivative = 1; derivative <= 4; ++derivative) {
    const double coarse = derivative_error(17, derivative);
    const double medium = derivative_error(33, derivative);
    const double fine = derivative_error(65, derivative);
    std::cout << "Route-B direct D" << derivative << " errors " << coarse
              << ' ' << medium << ' ' << fine << " ratios "
              << coarse / medium << ' ' << medium / fine << '\n';
    CHECK(coarse / medium > 15.0);
    CHECK(medium / fine > 15.0);
  }
}

template <class View>
void fill_device_input(const View& input) {
  auto host = Kokkos::create_mirror_view(input);
  for (std::size_t mode = 0; mode < input.extent(0); ++mode) {
    for (std::size_t field = 0; field < input.extent(1); ++field) {
      for (std::size_t radial = 0; radial < input.extent(2); ++radial) {
        for (std::size_t theta = 0; theta < input.extent(3); ++theta) {
          const double x = -0.2 + 0.05 * radial;
          host(mode, field, radial, theta) =
              (1.0 + 0.1 * mode + 0.03 * field + 0.02 * theta) *
              C(std::exp(0.7 * x), std::sin(1.3 * x));
        }
      }
    }
  }
  Kokkos::deep_copy(input, host);
}

TEST_CASE("Route-B Fornberg device graph has right and stride parity") {
  constexpr std::size_t modes = 2;
  constexpr std::size_t fields = 3;
  constexpr std::size_t radial = 17;
  constexpr std::size_t theta = 2;
  constexpr double spacing = 0.05;
  const teuk::ExecutionSpace execution;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> right_input(
      "routeb_right_input", modes, fields, radial, theta);
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> right_output(
      "routeb_right_output", modes, 4, fields, radial, theta);
  using Stride4 = Kokkos::View<C****, Kokkos::LayoutStride,
                               teuk::MemorySpace>;
  using Stride5 = Kokkos::View<C*****, Kokkos::LayoutStride,
                               teuk::MemorySpace>;
  Stride4 stride_input(
      "routeb_stride_input",
      Kokkos::LayoutStride(modes, fields * (radial * (theta + 1) + 3) + 11,
                           fields, radial * (theta + 1) + 3, radial,
                           theta + 1, theta, 1));
  Stride5 stride_output(
      "routeb_stride_output",
      Kokkos::LayoutStride(
          modes, 4 * (fields * (radial * (theta + 1) + 3) + 5) + 17, 4,
          fields * (radial * (theta + 1) + 3) + 5, fields,
          radial * (theta + 1) + 3, radial, theta + 1, theta, 1));
  fill_device_input(right_input);
  auto right_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                         right_input);
  auto stride_host = Kokkos::create_mirror_view(stride_input);
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t field = 0; field < fields; ++field) {
      for (std::size_t r = 0; r < radial; ++r) {
        for (std::size_t t = 0; t < theta; ++t) {
          stride_host(mode, field, r, t) = right_host(mode, field, r, t);
        }
      }
    }
  }
  Kokkos::deep_copy(stride_input, stride_host);
  teuk::evaluate_routeb_fornberg_derivatives(execution, right_input,
                                              right_output, spacing);
  teuk::evaluate_routeb_fornberg_derivatives(execution, stride_input,
                                              stride_output, spacing);
  execution.fence("finish Route-B layout parity");
  const auto right_result = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, right_output);
  const auto stride_result = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, stride_output);
  for (std::size_t mode = 0; mode < modes; ++mode) {
    for (std::size_t derivative = 0; derivative < 4; ++derivative) {
      for (std::size_t field = 0; field < fields; ++field) {
        for (std::size_t r = 0; r < radial; ++r) {
          for (std::size_t t = 0; t < theta; ++t) {
            CHECK_COMPLEX_NEAR(
                right_result(mode, derivative, field, r, t),
                stride_result(mode, derivative, field, r, t), 1.0e-12);
            const C expected = teuk::routeb_fornberg_direct_derivative_at(
                static_cast<int>(derivative + 1),
                &right_host(mode, field, 0, t), radial, r, 1.0 / spacing,
                right_host.stride(2));
            CHECK_COMPLEX_NEAR(
                right_result(mode, derivative, field, r, t), expected,
                2.0e-12);
          }
        }
      }
    }
  }
}

TEST_CASE("Route-B Fornberg graph rejects shapes aliases and stays hot") {
  const teuk::ExecutionSpace execution;
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> input(
      "routeb_contract_input", 1, 2, 17, 2);
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> output(
      "routeb_contract_output", 1, 4, 2, 17, 2);
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> wrong(
      "routeb_contract_wrong", 1, 3, 2, 17, 2);
  fill_device_input(input);
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(execution, input, wrong, 0.1);
  }));
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(execution, input, output, 0.0);
  }));
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(
        execution, input, output, std::numeric_limits<double>::denorm_min());
  }));
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(execution, input, output,
                                                1.0e-100);
  }));
  Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace> empty_input(
      "routeb_contract_empty", 0, 2, 17, 2);
  Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace> empty_output(
      "routeb_contract_empty_output", 0, 4, 2, 17, 2);
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(
        execution, empty_input, empty_output, 0.1);
  }));
  using NullInput =
      Kokkos::View<C****, Kokkos::LayoutRight, teuk::MemorySpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  NullInput null_input(static_cast<C*>(nullptr), 1, 2, 17, 2);
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(
        execution, null_input, output, 0.1);
  }));
  using StrideOutput =
      Kokkos::View<C*****, Kokkos::LayoutStride, teuk::MemorySpace>;
  StrideOutput internally_aliased_output(
      "routeb_contract_internal_alias",
      Kokkos::LayoutStride(1, 512, 4, 1, 2, 128, 17, 4, 2, 1));
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(
        execution, input, internally_aliased_output, 0.1);
  }));
  using AliasOutput =
      Kokkos::View<C*****, Kokkos::LayoutRight, teuk::MemorySpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  AliasOutput alias(input.data(), 1, 4, 2, 17, 2);
  CHECK(throws_invalid_argument([&] {
    teuk::evaluate_routeb_fornberg_derivatives(execution, input, alias, 0.1);
  }));

  teuk::evaluate_routeb_fornberg_derivatives(execution, input, output, 0.1);
  execution.fence("warm Route-B direct derivative graph");
  routeb_allocations = 0;
  routeb_fences = 0;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      count_routeb_allocation);
  Kokkos::Tools::Experimental::set_begin_fence_callback(count_routeb_fence);
  teuk::evaluate_routeb_fornberg_derivatives(execution, input, output, 0.1);
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
  execution.fence("finish hot Route-B direct derivative graph");
  CHECK(routeb_allocations == 0);
  CHECK(routeb_fences == 0);
}

TEST_CASE("Route-B radial Taylor jets obey product quotient and derivative") {
  constexpr std::size_t degree = 4;
  const C alpha(0.7, -0.2);
  const C beta(-0.3, 0.4);
  Kokkos::Array<C, degree + 1> f_derivatives{};
  Kokkos::Array<C, degree + 1> g_derivatives{};
  C f_power(1.0, 0.0);
  C g_power(1.0, 0.0);
  for (std::size_t order = 0; order <= degree; ++order) {
    f_derivatives[order] = f_power;
    g_derivatives[order] = g_power;
    f_power *= alpha;
    g_power *= beta;
  }
  const auto f = teuk::routeb_radial_jet_from_derivatives(f_derivatives);
  const auto g = teuk::routeb_radial_jet_from_derivatives(g_derivatives);
  const auto product = f * g;
  const auto recovered = product / g;
  const auto differentiated = teuk::routeb_radial_jet_derivative(f);
  C combined_power(1.0, 0.0);
  for (std::size_t order = 0; order <= degree; ++order) {
    CHECK_COMPLEX_NEAR(teuk::routeb_radial_jet_derivative(product, order),
                       combined_power, 2.0e-14);
    CHECK_COMPLEX_NEAR(recovered[order], f[order], 2.0e-14);
    combined_power *= alpha + beta;
    if (order < degree) {
      CHECK_COMPLEX_NEAR(differentiated[order],
                         static_cast<double>(order + 1) * f[order + 1], 0.0);
    }
  }
  const auto truncated = product.template truncate<2>();
  CHECK_COMPLEX_NEAR(truncated[2], product[2], 0.0);
}

TEST_CASE("Route-B radial Taylor jet algebra has device parity") {
  using Jet = teuk::RouteBRadialTaylorJet<4, C>;
  const Kokkos::Array<C, 5> f_derivatives{
      C(1.2, 0.1), C(-0.3, 0.4), C(0.7, -0.2), C(0.11, 0.08),
      C(-0.05, 0.03)};
  const Kokkos::Array<C, 5> g_derivatives{
      C(0.9, -0.2), C(0.2, 0.1), C(-0.4, 0.3), C(0.07, -0.02),
      C(0.03, 0.01)};
  const Jet expected =
      teuk::routeb_radial_jet_from_derivatives(f_derivatives) /
      teuk::routeb_radial_jet_from_derivatives(g_derivatives);
  Kokkos::View<Jet*, teuk::MemorySpace> result("routeb_jet_device", 1);
  Kokkos::parallel_for(
      "routeb_jet_device_parity", 1, KOKKOS_LAMBDA(const int) {
        result(0) = teuk::routeb_radial_jet_from_derivatives(f_derivatives) /
                    teuk::routeb_radial_jet_from_derivatives(g_derivatives);
      });
  const auto copied = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                           result);
  for (std::size_t order = 0; order <= 4; ++order) {
    CHECK_COMPLEX_NEAR(copied(0)[order], expected[order], 2.0e-14);
  }
}

}  // namespace
