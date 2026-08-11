#pragma once

#include <cmath>
#include <complex>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "teuk/types.hpp"

namespace teuk::test {

struct Case {
  std::string name;
  std::function<void()> body;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

struct Register {
  Register(std::string name, std::function<void()> body) {
    registry().push_back({std::move(name), std::move(body)});
  }
};

[[noreturn]] inline void fail(const char* expression, const char* file,
                              int line, const std::string& detail = {}) {
  std::ostringstream message;
  message << file << ':' << line << ": check failed: " << expression;
  if (!detail.empty()) message << " (" << detail << ')';
  throw std::runtime_error(message.str());
}

inline void check_near(double actual, double expected, double tolerance,
                       const char* expression, const char* file, int line) {
  if (std::abs(actual - expected) > tolerance) {
    std::ostringstream detail;
    detail << "actual=" << actual << ", expected=" << expected
           << ", tolerance=" << tolerance;
    fail(expression, file, line, detail.str());
  }
}

inline void check_complex_near(Complex actual, Complex expected,
                               double tolerance, const char* expression,
                               const char* file, int line) {
  if (Kokkos::abs(actual - expected) > tolerance) {
    std::ostringstream detail;
    detail << "actual=(" << actual.real() << ',' << actual.imag()
           << "), expected=(" << expected.real() << ',' << expected.imag()
           << "), tolerance=" << tolerance;
    fail(expression, file, line, detail.str());
  }
}

}  // namespace teuk::test

#define TEUK_CONCAT_INNER(a, b) a##b
#define TEUK_CONCAT(a, b) TEUK_CONCAT_INNER(a, b)
#define TEST_CASE(name)                                                        \
  static void TEUK_CONCAT(teuk_test_, __LINE__)();                            \
  [[maybe_unused]] static ::teuk::test::Register                              \
      TEUK_CONCAT(teuk_register_, __LINE__)(                                  \
      name, TEUK_CONCAT(teuk_test_, __LINE__));                               \
  static void TEUK_CONCAT(teuk_test_, __LINE__)()
#define CHECK(expression)                                                      \
  do {                                                                         \
    if (!(expression))                                                         \
      ::teuk::test::fail(#expression, __FILE__, __LINE__);                    \
  } while (false)
#define CHECK_NEAR(actual, expected, tolerance)                                \
  ::teuk::test::check_near((actual), (expected), (tolerance),                 \
                           #actual " ~= " #expected, __FILE__, __LINE__)
#define CHECK_COMPLEX_NEAR(actual, expected, tolerance)                        \
  ::teuk::test::check_complex_near((actual), (expected), (tolerance),         \
                                   #actual " ~= " #expected, __FILE__,       \
                                   __LINE__)
