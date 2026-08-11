#pragma once

#include <cmath>
#include <limits>

namespace teuk {

inline double outer_horizon_radius(const double mass, const double spin) {
  const double discriminant = mass * mass - spin * spin;
  return mass + std::sqrt(discriminant > 0.0 ? discriminant : 0.0);
}

inline double inner_horizon_radius(const double mass, const double spin) {
  const double discriminant = mass * mass - spin * spin;
  return mass - std::sqrt(discriminant > 0.0 ? discriminant : 0.0);
}

inline double surface_gravity(const double mass, const double spin) {
  if (mass <= 0.0 || std::abs(spin) > mass) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double r_plus = outer_horizon_radius(mass, spin);
  const double r_minus = inner_horizon_radius(mass, spin);
  return (r_plus - r_minus) /
         (2.0 * (r_plus * r_plus + spin * spin));
}

}  // namespace teuk

