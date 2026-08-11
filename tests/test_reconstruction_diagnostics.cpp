#include "test_harness.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "teuk/background.hpp"
#include "teuk/grid.hpp"
#include "teuk/radial.hpp"
#include "teuk/reconstruction.hpp"
#include "teuk/reconstruction_diagnostics.hpp"

namespace {

using teuk::Complex;

struct Profile {
  Complex value;
  Complex radial_derivative;
};

Profile profile(const double radius, const double wave_number,
                const Complex& amplitude, const double offset) {
  const double shape =
      offset + std::sin(wave_number * radius) + 0.13 * radius * radius;
  const double radial_shape =
      wave_number * std::cos(wave_number * radius) + 0.26 * radius;
  return {amplitude * shape, amplitude * radial_shape};
}

teuk::ReconstructionFieldDerivatives explicit_time_derivatives(
    const double radius, const double mass, const double length,
    const teuk::ReconstructionFields& fields,
    const teuk::ReconstructionFieldDerivatives& radial,
    const teuk::ReconstructionDeltaRhs& rhs) {
  const double length2 = length * length;
  const double denominator = 2.0 + 4.0 * mass * radius / length2;
  const auto solve = [&](const Complex& value, const Complex& dr,
                         const Complex& source, const int falloff) {
    return (source - radius * radius * dr / length2 -
            static_cast<double>(falloff) * radius * value / length2) /
           denominator;
  };
  return {solve(fields.G, radial.G, rhs.G, 2),
          solve(fields.Lambda, radial.Lambda, rhs.Lambda, 1),
          solve(fields.H, radial.H, rhs.H, 3),
          solve(fields.B, radial.B, rhs.B, 1),
          solve(fields.Pi, radial.Pi, rhs.Pi, 2),
          solve(fields.C, radial.C, rhs.C, 2),
          solve(fields.U, radial.U, rhs.U, 3)};
}

struct ManufacturedPoint {
  teuk::KerrBackgroundPoint background;
  teuk::ReconstructionFields fields;
  teuk::ReconstructionAngularDerivatives angular;
  teuk::ReconstructionFieldDerivatives exact_radial;
  teuk::ReconstructionFieldDerivatives exact_time;
};

ManufacturedPoint manufactured_point(const double radius) {
  constexpr double mass = 1.0;
  constexpr double spin = 0.61;
  constexpr double length = 1.7;
  constexpr double theta = 0.91;
  const teuk::KerrParameters parameters{mass, spin, length};
  const auto background =
      teuk::kerr_background_point(parameters, radius, theta);

  const Profile F = profile(radius, 0.71, Complex(0.8, -0.2), 0.5);
  const Profile G = profile(radius, 0.83, Complex(-0.3, 0.7), 0.6);
  const Profile H = profile(radius, 0.97, Complex(0.5, 0.4), -0.2);
  const Profile Lambda = profile(radius, 1.09, Complex(-0.6, -0.1), 0.4);
  const Profile Pi = profile(radius, 1.21, Complex(0.2, -0.8), -0.3);
  const Profile B = profile(radius, 1.37, Complex(0.9, 0.3), 0.2);
  const Profile C = profile(radius, 1.51, Complex(-0.4, 0.6), 0.7);
  const Profile U = profile(radius, 1.69, Complex(0.7, -0.5), -0.1);
  const Profile Pi_sharp =
      profile(radius, 1.13, Complex(-0.2, 0.9), 0.35);
  const Profile B_sharp =
      profile(radius, 1.43, Complex(0.3, -0.7), -0.45);
  const Profile C_sharp =
      profile(radius, 1.73, Complex(-0.8, -0.4), 0.55);

  const teuk::ReconstructionFields fields{
      F.value, G.value, H.value, Lambda.value, Pi.value, B.value,
      C.value, U.value, Pi_sharp.value, B_sharp.value, C_sharp.value};
  const teuk::ReconstructionAngularDerivatives angular{
      profile(radius, 0.77, Complex(0.1, 0.5), 0.1).value,
      profile(radius, 0.89, Complex(-0.7, 0.2), -0.2).value,
      profile(radius, 1.03, Complex(0.4, -0.6), 0.3).value,
      profile(radius, 1.17, Complex(-0.5, -0.3), -0.4).value,
      profile(radius, 1.31, Complex(0.6, 0.8), 0.5).value,
      profile(radius, 1.47, Complex(-0.9, 0.1), -0.6).value};
  const teuk::ReconstructionFieldDerivatives radial{
      G.radial_derivative,      Lambda.radial_derivative,
      H.radial_derivative,      B.radial_derivative,
      Pi.radial_derivative,     C.radial_derivative,
      U.radial_derivative};

  // Evaluate the seven dependencies in their production order, then invert
  // Delta_n explicitly rather than through reconstruction_time_derivative.
  const auto rhs =
      teuk::reconstruction_delta_rhs(radius, background, fields, angular);
  const auto time = explicit_time_derivatives(radius, mass, length, fields,
                                               radial, rhs);
  return {background, fields, angular, radial, time};
}

teuk::ReconstructionResidualNorms manufactured_residual_norms(
    const std::size_t point_count, const bool use_exact_radial_derivatives) {
  constexpr double mass = 1.0;
  constexpr double length = 1.7;
  const teuk::UniformRadialGrid grid(point_count, 0.0, 0.9);
  std::vector<ManufacturedPoint> points;
  points.reserve(point_count);
  std::array<std::vector<Complex>, 7> values;
  std::array<std::vector<Complex>, 7> numerical_derivatives;
  for (std::size_t field = 0; field < values.size(); ++field) {
    values[field].resize(point_count);
    numerical_derivatives[field].resize(point_count);
  }

  for (std::size_t i = 0; i < point_count; ++i) {
    points.push_back(manufactured_point(grid.coordinate(i)));
    const auto& fields = points.back().fields;
    values[0][i] = fields.G;
    values[1][i] = fields.Lambda;
    values[2][i] = fields.H;
    values[3][i] = fields.B;
    values[4][i] = fields.Pi;
    values[5][i] = fields.C;
    values[6][i] = fields.U;
  }
  for (std::size_t field = 0; field < values.size(); ++field) {
    teuk::fourth_order_radial_derivative(
        grid, values[field], numerical_derivatives[field]);
  }

  std::vector<teuk::ReconstructionResiduals> residuals(point_count);
  for (std::size_t i = 0; i < point_count; ++i) {
    teuk::ReconstructionFieldDerivatives radial{
        numerical_derivatives[0][i], numerical_derivatives[1][i],
        numerical_derivatives[2][i], numerical_derivatives[3][i],
        numerical_derivatives[4][i], numerical_derivatives[5][i],
        numerical_derivatives[6][i]};
    if (use_exact_radial_derivatives) radial = points[i].exact_radial;
    residuals[i] = teuk::reconstruction_residuals_point(
        grid.coordinate(i), mass, length, points[i].background,
        points[i].fields, points[i].angular, points[i].exact_time, radial);
  }
  return teuk::reconstruction_residual_norms(residuals);
}

std::array<teuk::ResidualNorm, 7> field_norms(
    const teuk::ReconstructionResidualNorms& norms) {
  return {norms.G, norms.Lambda, norms.H, norms.B,
          norms.Pi, norms.C, norms.U};
}

}  // namespace

TEST_CASE("independent reconstruction residuals vanish for exact derivatives") {
  const auto norms = manufactured_residual_norms(17, true);
  for (const auto& field : field_norms(norms)) {
    CHECK(field.rms < 2.0e-13);
    CHECK(field.maximum < 7.0e-13);
  }
  CHECK(norms.combined.rms < 2.0e-13);
}

TEST_CASE("all seven reconstruction residuals converge fourth order in space") {
  const auto coarse = manufactured_residual_norms(17, false);
  const auto medium = manufactured_residual_norms(33, false);
  const auto fine = manufactured_residual_norms(65, false);
  const auto coarse_fields = field_norms(coarse);
  const auto medium_fields = field_norms(medium);
  const auto fine_fields = field_norms(fine);
  for (std::size_t field = 0; field < coarse_fields.size(); ++field) {
    CHECK(coarse_fields[field].rms / medium_fields[field].rms > 11.0);
    CHECK(medium_fields[field].rms / fine_fields[field].rms > 11.0);
    CHECK(fine_fields[field].maximum > 0.0);
  }
  CHECK(coarse.combined.rms / medium.combined.rms > 11.0);
  CHECK(medium.combined.rms / fine.combined.rms > 11.0);
}

TEST_CASE("reconstruction residual norms report per-field and combined scales") {
  const std::vector<teuk::ReconstructionResiduals> residuals{
      {Complex(3.0, 4.0), Complex(0.0, 0.0), Complex(0.0, 0.0),
       Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0),
       Complex(0.0, 0.0)},
      {Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0),
       Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0),
       Complex(0.0, 12.0)}};
  const auto norms = teuk::reconstruction_residual_norms(residuals);
  CHECK_NEAR(norms.G.maximum, 5.0, 1.0e-14);
  CHECK_NEAR(norms.G.rms, 5.0 / std::sqrt(2.0), 1.0e-14);
  CHECK_NEAR(norms.U.maximum, 12.0, 1.0e-14);
  CHECK_NEAR(norms.U.rms, 12.0 / std::sqrt(2.0), 1.0e-14);
  CHECK_NEAR(norms.combined.maximum, 12.0, 1.0e-14);
  CHECK_NEAR(norms.combined.rms, 13.0 / std::sqrt(14.0), 1.0e-14);
}

TEST_CASE("reconstruction residual norms reject an empty sample") {
  bool threw = false;
  try {
    static_cast<void>(teuk::reconstruction_residual_norms({}));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}
