#include <Kokkos_Core.hpp>

#include <iostream>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "teuk-kokkos 0.1.0\n";
    std::cout << "Kokkos execution space: "
              << Kokkos::DefaultExecutionSpace::name() << '\n';
    std::cout << "The end-to-end solver is under active construction.\n";
  }
  Kokkos::finalize();
  return 0;
}
