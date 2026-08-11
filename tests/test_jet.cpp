#include <array>

#include "test_harness.hpp"
#include "teuk/jet.hpp"
#include "teuk/second_order.hpp"

namespace {

using C = teuk::Complex;
using J = teuk::Jet1<C>;

C value(const int i) {
  return C(0.07 * static_cast<double>(i + 1),
           -0.03 * static_cast<double>(2 * i + 1));
}

C tangent(const int i) {
  return C(-0.02 * static_cast<double>(i + 2),
           0.05 * static_cast<double>(i + 1));
}

teuk::OrderedPairFields fields_at(const double epsilon) {
  return {
      value(0) + epsilon * tangent(0),   value(1) + epsilon * tangent(1),
      value(2) + epsilon * tangent(2),   value(3) + epsilon * tangent(3),
      value(4) + epsilon * tangent(4),   value(5) + epsilon * tangent(5),
      value(6) + epsilon * tangent(6),   value(7) + epsilon * tangent(7),
      value(8) + epsilon * tangent(8),   value(9) + epsilon * tangent(9),
      value(10) + epsilon * tangent(10), value(11) + epsilon * tangent(11),
      value(12) + epsilon * tangent(12), value(13) + epsilon * tangent(13),
      value(14) + epsilon * tangent(14), value(15) + epsilon * tangent(15),
      value(16) + epsilon * tangent(16), value(17) + epsilon * tangent(17),
      value(18) + epsilon * tangent(18)};
}

teuk::OrderedPairDerivatives derivatives_at(const double epsilon) {
  return {
      value(20) + epsilon * tangent(20), value(21) + epsilon * tangent(21),
      value(22) + epsilon * tangent(22), value(23) + epsilon * tangent(23),
      value(24) + epsilon * tangent(24), value(25) + epsilon * tangent(25),
      value(26) + epsilon * tangent(26), value(27) + epsilon * tangent(27),
      value(28) + epsilon * tangent(28), value(29) + epsilon * tangent(29),
      value(30) + epsilon * tangent(30)};
}

teuk::OrderedPairFieldsT<J> jet_fields() {
  return {{value(0), tangent(0)},   {value(1), tangent(1)},
          {value(2), tangent(2)},   {value(3), tangent(3)},
          {value(4), tangent(4)},   {value(5), tangent(5)},
          {value(6), tangent(6)},   {value(7), tangent(7)},
          {value(8), tangent(8)},   {value(9), tangent(9)},
          {value(10), tangent(10)}, {value(11), tangent(11)},
          {value(12), tangent(12)}, {value(13), tangent(13)},
          {value(14), tangent(14)}, {value(15), tangent(15)},
          {value(16), tangent(16)}, {value(17), tangent(17)},
          {value(18), tangent(18)}};
}

teuk::OrderedPairDerivativesT<J> jet_derivatives() {
  return {{value(20), tangent(20)},   {value(21), tangent(21)},
           {value(22), tangent(22)},  {value(23), tangent(23)},
           {value(24), tangent(24)},  {value(25), tangent(25)},
           {value(26), tangent(26)},  {value(27), tangent(27)},
           {value(28), tangent(28)},  {value(29), tangent(29)},
           {value(30), tangent(30)}};
}

}  // namespace

TEST_CASE("Jet1 product and quotient rules are stage local") {
  const J left{C(0.7, -0.2), C(-0.3, 0.5)};
  const J right{C(-0.4, 0.8), C(0.6, 0.1)};
  const J product = left * right;
  CHECK_COMPLEX_NEAR(product.value, left.value * right.value, 1.0e-15);
  CHECK_COMPLEX_NEAR(product.dt,
                     left.dt * right.value + left.value * right.dt,
                     1.0e-15);

  const J quotient = left / right;
  CHECK_COMPLEX_NEAR(quotient.value, left.value / right.value, 1.0e-15);
  CHECK_COMPLEX_NEAR(
      quotient.dt,
      (left.dt * right.value - left.value * right.dt) /
          (right.value * right.value),
      1.0e-15);
}

TEST_CASE("analytic ordered-pair source tangent matches a directional oracle") {
  const double radius = 0.41;
  const teuk::KerrBackgroundPoint background{
      C(-0.8, 0.13), C(0.04, 0.09), C(-0.02, -0.07), C(-0.6, 0.2)};
  const auto jet_source = teuk::corrected_ordered_pair_source(
      radius, background, jet_fields(), jet_derivatives());

  constexpr double epsilon = 2.0e-7;
  const auto plus = teuk::corrected_ordered_pair_source(
      radius, background, fields_at(epsilon), derivatives_at(epsilon));
  const auto minus = teuk::corrected_ordered_pair_source(
      radius, background, fields_at(-epsilon), derivatives_at(-epsilon));
  const C finite_difference_D = (plus.D - minus.D) / (2.0 * epsilon);
  const C finite_difference_T = (plus.T - minus.T) / (2.0 * epsilon);

  const auto direct_value = teuk::corrected_ordered_pair_source(
      radius, background, fields_at(0.0), derivatives_at(0.0));
  CHECK_COMPLEX_NEAR(jet_source.D.value, direct_value.D, 1.0e-14);
  CHECK_COMPLEX_NEAR(jet_source.T.value, direct_value.T, 1.0e-14);
  CHECK_COMPLEX_NEAR(jet_source.D.dt, finite_difference_D, 2.0e-8);
  CHECK_COMPLEX_NEAR(jet_source.T.dt, finite_difference_T, 2.0e-8);
}
