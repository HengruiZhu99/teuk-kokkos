#include <Kokkos_Core.hpp>

#include <iostream>

#include "test_harness.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int failures = 0;
  {
    for (const auto& test : teuk::test::registry()) {
      try {
        test.body();
        std::cout << "[PASS] " << test.name << '\n';
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
      }
    }
  }
  Kokkos::finalize();
  std::cout << (teuk::test::registry().size() - static_cast<std::size_t>(failures))
            << '/' << teuk::test::registry().size() << " tests passed\n";
  return failures == 0 ? 0 : 1;
}

