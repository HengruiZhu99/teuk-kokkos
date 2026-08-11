cmake_minimum_required(VERSION 3.24)

project(teuk_kokkos VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
option(TEUK_ENABLE_SYMBOLIC_AUDIT
  "Run the development-only Python/SymPy mathematical audit under CTest" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos/CMakeLists.txt")
  message(FATAL_ERROR
    "Kokkos submodule is missing. Run: git submodule update --init --recursive")
endif()

# Serial is the portable baseline. Additional Kokkos backends are selected at
# configure time, for example -DKokkos_ENABLE_OPENMP=ON or
# -DKokkos_ENABLE_SYCL=ON. Physics sources do not depend on that selection.
if(NOT DEFINED Kokkos_ENABLE_SERIAL)
  set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Enable the Kokkos Serial backend")
endif()
add_subdirectory(external/kokkos)

add_library(teuk INTERFACE)
target_include_directories(teuk INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(teuk INTERFACE Kokkos::kokkos)
target_compile_features(teuk INTERFACE cxx_std_20)

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE TEUK_GIT_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT TEUK_GIT_COMMIT)
  set(TEUK_GIT_COMMIT "unknown")
endif()
target_compile_definitions(teuk INTERFACE
  TEUK_GIT_COMMIT="${TEUK_GIT_COMMIT}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  target_compile_options(teuk INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

add_executable(teuk_solver src/main.cpp)
target_link_libraries(teuk_solver PRIVATE teuk)

foreach(example IN ITEMS linear_gaussian second_order_mode_coupling near_extremal)
  add_executable(teuk_example_${example} examples/${example}.cpp)
  target_link_libraries(teuk_example_${example} PRIVATE teuk)
endforeach()

if(BUILD_TESTING)
  file(GLOB TEUK_TEST_SOURCES CONFIGURE_DEPENDS tests/test_*.cpp)
  list(REMOVE_ITEM TEUK_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_qnm.cpp")
  add_executable(teuk_tests ${TEUK_TEST_SOURCES})
  target_include_directories(teuk_tests PRIVATE tests)
  target_link_libraries(teuk_tests PRIVATE teuk)
  add_test(NAME teuk.unit COMMAND teuk_tests)

  add_executable(teuk_qnm_tests tests/test_main.cpp tests/test_qnm.cpp)
  target_include_directories(teuk_qnm_tests PRIVATE tests)
  target_link_libraries(teuk_qnm_tests PRIVATE teuk)
  add_test(NAME teuk.qnm COMMAND teuk_qnm_tests)

  add_test(
    NAME teuk.config_integration
    COMMAND ${CMAKE_COMMAND}
      -DTEUK_SOLVER=$<TARGET_FILE:teuk_solver>
      -DTEUK_INTEGRATION_DIR=${CMAKE_CURRENT_BINARY_DIR}/config-integration
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_config_integration.cmake)

  if(TEUK_ENABLE_SYMBOLIC_AUDIT)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
      NAME teuk.symbolic_audit
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/verify_second_order_teukolsky.py
        --json-out
        ${CMAKE_CURRENT_BINARY_DIR}/verify_second_order_teukolsky_output.local.json)
  endif()
endif()
