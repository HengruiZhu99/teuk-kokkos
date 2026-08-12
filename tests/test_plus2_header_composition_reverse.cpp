#include "test_harness.hpp"

#include <type_traits>

#include "teuk/plus2_source_primitives.hpp"
#include "teuk/plus2_linear_psi0.hpp"

namespace {

static_assert(!std::is_same_v<teuk::Plus2ReconstructionPrimitiveInputs,
                              teuk::Plus2OrgMetricFields>);

TEST_CASE("plus2 linear and primitive headers compose primitives first") {
  teuk::Plus2ReconstructionPrimitiveInputs primitive_input{};
  teuk::Plus2OrgMetricFields physical_metric{};
  primitive_input.U = teuk::Complex(0.1, 0.2);
  physical_metric.h_ll = primitive_input.U;
  CHECK_COMPLEX_NEAR(physical_metric.h_ll, primitive_input.U, 0.0);
}

}  // namespace
