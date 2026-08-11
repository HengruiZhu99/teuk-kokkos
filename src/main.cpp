#include <Kokkos_Core.hpp>

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "teuk/config.hpp"
#include "teuk/run_parameters.hpp"
#include "teuk/solver_driver.hpp"

namespace {

void print_help() {
  std::cout
      << "Usage:\n"
      << "  teuk_solver --config run.cfg [key=value ...]\n"
      << "  teuk_solver backend\n"
      << "  teuk_solver --help\n\n"
      << "Configuration files use strict key = value assignments and # comments.\n"
      << "Command-line key=value overrides take precedence over file values.\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    const std::string_view command = argc > 1 ? argv[1] : "--help";
    if (command == "backend") {
      std::cout << "teuk-kokkos " << teuk::solver_executable_version << '\n'
                << "Kokkos execution space: "
                << Kokkos::DefaultExecutionSpace::name() << '\n';
    } else if (command == "--config") {
      if (argc < 3) {
        throw std::invalid_argument("--config requires a configuration path");
      }
      std::vector<std::string> overrides;
      for (int i = 3; i < argc; ++i) overrides.emplace_back(argv[i]);
      const teuk::RunParameters parameters =
          teuk::parse_run_configuration(argv[2], overrides);
      status = teuk::run_solver(parameters);
    } else if (command == "help" || command == "--help" || command == "-h") {
      print_help();
    } else {
      throw std::invalid_argument("unknown command: " + std::string(command));
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    status = 2;
  }
  Kokkos::finalize();
  return status;
}
