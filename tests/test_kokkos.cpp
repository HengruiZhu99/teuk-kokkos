#include <Kokkos_Core.hpp>

#include <cstddef>

#include "test_harness.hpp"
#include "teuk/types.hpp"

TEST_CASE("active Kokkos execution space runs a complex field kernel") {
  constexpr int count = 4096;
  teuk::View<teuk::Complex*> values("values", count);

  Kokkos::parallel_for(
      "initialize_complex_field", Kokkos::RangePolicy<teuk::ExecutionSpace>(0, count),
      KOKKOS_LAMBDA(const int i) {
        const double x = static_cast<double>(i) / static_cast<double>(count);
        values(i) = teuk::Complex(x, -2.0 * x);
      });

  double error = 0.0;
  Kokkos::parallel_reduce(
      "verify_complex_field", Kokkos::RangePolicy<teuk::ExecutionSpace>(0, count),
      KOKKOS_LAMBDA(const int i, double& local_error) {
        const double x = static_cast<double>(i) / static_cast<double>(count);
        const teuk::Complex difference = values(i) - teuk::Complex(x, -2.0 * x);
        local_error += difference.real() * difference.real() +
                       difference.imag() * difference.imag();
      },
      error);
  Kokkos::fence();

  CHECK_NEAR(error, 0.0, 1.0e-28);
}
