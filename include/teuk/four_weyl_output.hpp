#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "teuk/plus2_field.hpp"
#include "teuk/plus2_source.hpp"
#include "teuk/radial_discretization.hpp"
#include "teuk/types.hpp"

namespace teuk {

inline constexpr std::uint32_t four_weyl_output_schema_version = 3;
inline constexpr const char* four_weyl_output_schema =
    "teuk.four-weyl-field-output";
inline constexpr const char* four_weyl_gauge_id =
    "outgoing-radiation-gauge";
inline constexpr const char* four_weyl_tetrad_id =
    "Ripley-rotated-Kinnersley-horizon-regular-code-tetrad";
inline constexpr const char* four_weyl_perturbative_convention =
    "A=A0+epsilon*A1+epsilon^2*A2+O(epsilon^3);no-factorial";
inline constexpr const char* four_weyl_minus2_scaling =
    "Psi4_raw_fixed_tetrad=R*F";
inline constexpr const char* four_weyl_plus2_scaling =
    "Psi0_raw_fixed_tetrad=(R^5/(L^2-i*a*R*cos(theta))^4)*Z_plus";
inline constexpr const char* four_weyl_sharp_semantics =
    "X_m^sharp=conjugate(X_-m);never-conjugate-same-m";
inline constexpr const char* four_weyl_scri_semantics =
    "raw-code-tetrad-value-is-zero;store-explicit-leading-R-power-coefficient";
inline constexpr const char* four_weyl_horizon_semantics =
    "raw-code-tetrad-value-is-convention-fixed-and-is-not-a-flux-observable";

enum class FourWeylField : int {
  Psi4Order1 = 0,
  Psi0Order1 = 1,
  Psi4Order2 = 2,
  Psi0Order2 = 3,
  Count = 4
};

enum class FourWeylAngularRepresentation : int {
  SignedAzimuthalModalPolarNodal = 0,
  SpinWeightedSphericalModalAtScri = 1
};

enum class FourWeylBoundary : int { Interior = 0, Scri = 1, Horizon = 2 };
enum class FourWeylLinearMethod : int {
  MetricCurvature = 0,
  TeukolskyStarobinsky = 1,
  Both = 2
};
enum class FourWeylSourceMethod : int { RawOrgSourcedCompanion = 0 };
enum class FourWeylInitialPolicy : int { Zero = 0, Checkpoint = 1 };

inline const char* four_weyl_field_name(const FourWeylField field) {
  switch (field) {
    case FourWeylField::Psi4Order1: return "psi4_order1";
    case FourWeylField::Psi0Order1: return "psi0_order1";
    case FourWeylField::Psi4Order2: return "psi4_order2";
    case FourWeylField::Psi0Order2: return "psi0_order2";
    case FourWeylField::Count: break;
  }
  throw std::invalid_argument("unsupported four-Weyl field");
}

inline FourWeylField parse_four_weyl_field(const std::string& name) {
  for (int field = 0; field < static_cast<int>(FourWeylField::Count); ++field) {
    const auto value = static_cast<FourWeylField>(field);
    if (name == four_weyl_field_name(value)) return value;
  }
  throw std::invalid_argument("unknown four-Weyl field: " + name);
}

KOKKOS_INLINE_FUNCTION int four_weyl_spin(const FourWeylField field) {
  return field == FourWeylField::Psi4Order1 ||
                 field == FourWeylField::Psi4Order2
             ? -2
             : 2;
}

KOKKOS_INLINE_FUNCTION int four_weyl_order(const FourWeylField field) {
  return field == FourWeylField::Psi4Order1 ||
                 field == FourWeylField::Psi0Order1
             ? 1
             : 2;
}

inline const char* four_weyl_representation_name(
    const FourWeylAngularRepresentation representation) {
  switch (representation) {
    case FourWeylAngularRepresentation::SignedAzimuthalModalPolarNodal:
      return "signed-m-polar-nodal";
    case FourWeylAngularRepresentation::SpinWeightedSphericalModalAtScri:
      return "spin-weighted-spherical-modal-at-scri";
  }
  throw std::invalid_argument("unsupported four-Weyl representation");
}

inline FourWeylAngularRepresentation parse_four_weyl_representation(
    const std::string& name) {
  if (name == "signed-m-polar-nodal") {
    return FourWeylAngularRepresentation::SignedAzimuthalModalPolarNodal;
  }
  if (name == "spin-weighted-spherical-modal-at-scri") {
    return FourWeylAngularRepresentation::SpinWeightedSphericalModalAtScri;
  }
  throw std::invalid_argument("unknown four-Weyl representation: " + name);
}

inline const char* four_weyl_boundary_name(const FourWeylBoundary boundary) {
  switch (boundary) {
    case FourWeylBoundary::Interior: return "interior";
    case FourWeylBoundary::Scri: return "scri";
    case FourWeylBoundary::Horizon: return "horizon";
  }
  throw std::invalid_argument("unsupported four-Weyl boundary");
}

inline FourWeylBoundary parse_four_weyl_boundary(const std::string& name) {
  if (name == "interior") return FourWeylBoundary::Interior;
  if (name == "scri") return FourWeylBoundary::Scri;
  if (name == "horizon") return FourWeylBoundary::Horizon;
  throw std::invalid_argument("unknown four-Weyl boundary: " + name);
}

inline const char* four_weyl_linear_method_name(
    const FourWeylLinearMethod method) {
  switch (method) {
    case FourWeylLinearMethod::MetricCurvature: return "metric_curvature";
    case FourWeylLinearMethod::TeukolskyStarobinsky: return "tsi";
    case FourWeylLinearMethod::Both: return "both";
  }
  throw std::invalid_argument("unsupported linear Psi0 method");
}

inline FourWeylLinearMethod parse_four_weyl_linear_method(
    const std::string& name) {
  if (name == "metric_curvature") return FourWeylLinearMethod::MetricCurvature;
  if (name == "tsi") return FourWeylLinearMethod::TeukolskyStarobinsky;
  if (name == "both") return FourWeylLinearMethod::Both;
  throw std::invalid_argument("unknown linear Psi0 method: " + name);
}

inline const char* four_weyl_source_method_name(
    const FourWeylSourceMethod method) {
  if (method == FourWeylSourceMethod::RawOrgSourcedCompanion) {
    return "raw_org_sourced_companion";
  }
  throw std::invalid_argument("unsupported second-order Psi0 source method");
}

inline FourWeylSourceMethod parse_four_weyl_source_method(
    const std::string& name) {
  if (name == "raw_org_sourced_companion") {
    return FourWeylSourceMethod::RawOrgSourcedCompanion;
  }
  throw std::invalid_argument("unknown second-order Psi0 source method: " +
                              name);
}

inline const char* four_weyl_initial_policy_name(
    const FourWeylInitialPolicy policy) {
  switch (policy) {
    case FourWeylInitialPolicy::Zero: return "zero";
    case FourWeylInitialPolicy::Checkpoint: return "checkpoint";
  }
  throw std::invalid_argument("unsupported companion initial policy");
}

inline FourWeylInitialPolicy parse_four_weyl_initial_policy(
    const std::string& name) {
  if (name == "zero") return FourWeylInitialPolicy::Zero;
  if (name == "checkpoint") return FourWeylInitialPolicy::Checkpoint;
  throw std::invalid_argument("unknown companion initial policy: " + name);
}

struct FourWeylOutputMetadata {
  std::string schema = four_weyl_output_schema;
  std::uint32_t version = four_weyl_output_schema_version;
  std::string git_commit = TEUK_GIT_COMMIT;
  int runtime_config_schema_version = 1;
  std::string gauge_id = four_weyl_gauge_id;
  std::string tetrad_id = four_weyl_tetrad_id;
  std::string perturbative_convention = four_weyl_perturbative_convention;
  std::string psi4_scaling = four_weyl_minus2_scaling;
  std::string psi0_scaling = four_weyl_plus2_scaling;
  FourWeylLinearMethod linear_method = FourWeylLinearMethod::MetricCurvature;
  FourWeylSourceMethod source_method =
      FourWeylSourceMethod::RawOrgSourcedCompanion;
  std::uint32_t source_normalization_version =
      plus2_source_normalization_version;
  std::string source_normalization_name = plus2_source_normalization_name;
  FourWeylInitialPolicy initial_policy = FourWeylInitialPolicy::Zero;
  int ell_max_first = 0;
  int ell_max_second = 0;
  std::vector<int> parent_modes;
  std::vector<int> target_modes;
  std::uint64_t output_cadence_steps = 0;
  RadialDiscretization radial_discretization = RadialDiscretization::D42;
  std::string sharp_semantics = four_weyl_sharp_semantics;
  std::string scri_semantics = four_weyl_scri_semantics;
  std::string horizon_semantics = four_weyl_horizon_semantics;

  bool operator==(const FourWeylOutputMetadata&) const = default;
};

namespace four_weyl_detail {

inline bool finite(const Complex value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

inline void require_string(const std::string& value, const char* label) {
  if (value.empty() || value.find('\n') != std::string::npos ||
      value.find('\r') != std::string::npos) {
    throw std::invalid_argument(std::string("invalid four-Weyl ") + label);
  }
}

inline void require_registry(const std::vector<int>& modes, const int ell_max,
                             const char* label) {
  if (ell_max < 2 || modes.empty() ||
      !std::is_sorted(modes.begin(), modes.end()) ||
      std::adjacent_find(modes.begin(), modes.end()) != modes.end()) {
    throw std::invalid_argument(std::string(label) +
                                " registry must be nonempty sorted unique");
  }
  for (const int mode : modes) {
    if (std::abs(mode) > ell_max ||
        !std::binary_search(modes.begin(), modes.end(), -mode)) {
      throw std::invalid_argument(std::string(label) +
                                  " registry must be bandlimited and sharp closed");
    }
  }
}

inline void validate_metadata(const FourWeylOutputMetadata& metadata) {
  if (metadata.schema != four_weyl_output_schema ||
      metadata.version != four_weyl_output_schema_version ||
      metadata.runtime_config_schema_version <= 0 ||
      metadata.output_cadence_steps == 0 ||
      metadata.psi4_scaling != four_weyl_minus2_scaling ||
      metadata.psi0_scaling != four_weyl_plus2_scaling ||
      metadata.sharp_semantics != four_weyl_sharp_semantics ||
      metadata.scri_semantics != four_weyl_scri_semantics ||
      metadata.horizon_semantics != four_weyl_horizon_semantics) {
    throw std::invalid_argument(
        "invalid four-Weyl schema, cadence, or fixed convention metadata");
  }
  require_string(metadata.git_commit, "git commit");
  require_string(metadata.gauge_id, "gauge identifier");
  require_string(metadata.tetrad_id, "tetrad identifier");
  require_string(metadata.perturbative_convention,
                 "perturbative convention");
  if (metadata.source_normalization_version !=
          plus2_source_normalization_version ||
      metadata.source_normalization_name != plus2_source_normalization_name) {
    throw std::invalid_argument(
        "invalid four-Weyl spin plus2 source normalization metadata");
  }
  require_registry(metadata.parent_modes, metadata.ell_max_first, "parent");
  require_registry(metadata.target_modes, metadata.ell_max_second, "target");
  (void)radial_discretization_name(metadata.radial_discretization);
}

inline std::vector<int> stored_modes(const FourWeylOutputMetadata& metadata) {
  std::vector<int> result = metadata.parent_modes;
  result.insert(result.end(), metadata.target_modes.begin(),
                metadata.target_modes.end());
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

inline std::string encode_modes(const std::vector<int>& modes) {
  std::ostringstream output;
  for (std::size_t i = 0; i < modes.size(); ++i) {
    if (i != 0) output << ',';
    output << modes[i];
  }
  return output.str();
}

inline int parse_int(const std::string& text, const char* label) {
  std::size_t consumed = 0;
  const long value = std::stol(text, &consumed);
  if (consumed != text.size() || value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + label);
  }
  return static_cast<int>(value);
}

inline std::uint64_t parse_uint64(const std::string& text,
                                  const char* label) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string("invalid ") + label);
  }
  std::size_t consumed = 0;
  const auto value = std::stoull(text, &consumed);
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string("invalid ") + label);
  }
  return static_cast<std::uint64_t>(value);
}

inline double parse_finite_double(const std::string& text,
                                  const char* label) {
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string("invalid nonfinite ") + label);
  }
  return value;
}

inline std::vector<int> decode_modes(const std::string& text) {
  std::vector<int> result;
  std::stringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    if (item.empty()) throw std::invalid_argument("empty signed mode");
    result.push_back(parse_int(item, "signed mode"));
  }
  if (result.empty()) throw std::invalid_argument("empty signed-mode registry");
  return result;
}

inline std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream input(line);
  std::string field;
  while (std::getline(input, field, ',')) fields.push_back(field);
  if (!line.empty() && line.back() == ',') fields.emplace_back();
  return fields;
}

struct FieldModeEntry {
  int field = 0;
  int mode_index = 0;
};

struct PackedValue {
  Complex regularized{};
  Complex raw_code_tetrad{};
  Complex asymptotic_coefficient{};
  int asymptotic_power = 0;
};

KOKKOS_INLINE_FUNCTION Complex raw_from_regularized(
    const FourWeylField field, const Complex regularized, const double radius,
    const double cos_theta, const double spin,
    const double compactification_length) {
  if (four_weyl_spin(field) == -2) return radius * regularized;
  return plus2_code_tetrad_scaling(radius, cos_theta, spin,
                                   compactification_length) *
         regularized;
}

KOKKOS_INLINE_FUNCTION Complex scri_coefficient_from_regularized(
    const FourWeylField field, const Complex regularized,
    const double compactification_length) {
  if (four_weyl_spin(field) == -2) return regularized;
  return plus2_scri_scaling_coefficient(compactification_length) *
         regularized;
}

}  // namespace four_weyl_detail

struct FourWeylOutputRecord {
  std::uint64_t step = 0;
  double time = 0.0;
  FourWeylAngularRepresentation representation =
      FourWeylAngularRepresentation::SignedAzimuthalModalPolarNodal;
  FourWeylField field = FourWeylField::Psi4Order1;
  int m = 0;
  int ell = -1;
  int radial_index = -1;
  int theta_index = -1;
  double radius = 0.0;
  double cos_theta = 0.0;
  FourWeylBoundary boundary = FourWeylBoundary::Interior;
  Complex regularized{};
  Complex raw_code_tetrad{};
  int asymptotic_power = 0;
  Complex asymptotic_coefficient{};

  bool operator==(const FourWeylOutputRecord&) const = default;
};

struct FourWeylModalValue {
  int ell = 2;
  int m = 0;
  Complex regularized{};
};

struct FourWeylScriModalInput {
  std::array<std::vector<FourWeylModalValue>,
             static_cast<std::size_t>(FourWeylField::Count)>
      fields;
};

inline Complex four_weyl_sharp_value(const std::vector<int>& signed_modes,
                                     const std::vector<Complex>& values,
                                     const int mode) {
  if (signed_modes.size() != values.size() ||
      !std::is_sorted(signed_modes.begin(), signed_modes.end())) {
    throw std::invalid_argument("invalid signed-mode values for sharp lookup");
  }
  const auto iterator =
      std::lower_bound(signed_modes.begin(), signed_modes.end(), -mode);
  if (iterator == signed_modes.end() || *iterator != -mode) {
    throw std::invalid_argument("signed-mode registry is not sharp closed");
  }
  return Kokkos::conj(values[static_cast<std::size_t>(
      std::distance(signed_modes.begin(), iterator))]);
}

class FourWeylOutputPacker {
 public:
  FourWeylOutputPacker() = default;

  static FourWeylOutputPacker enabled(
      FourWeylOutputMetadata metadata, const std::vector<double>& radii,
      const std::vector<double>& cos_theta, const double kerr_spin,
      const double compactification_length, const double horizon_radius) {
    four_weyl_detail::validate_metadata(metadata);
    if (radii.size() < 2 || cos_theta.empty() ||
        !std::isfinite(kerr_spin) || !(compactification_length > 0.0) ||
        !std::isfinite(compactification_length) || !(horizon_radius > 0.0) ||
        !std::isfinite(horizon_radius) || radii.front() != 0.0 ||
        radii.back() != horizon_radius ||
        !std::is_sorted(radii.begin(), radii.end())) {
      throw std::invalid_argument("invalid four-Weyl output grid geometry");
    }
    if (std::adjacent_find(radii.begin(), radii.end()) != radii.end()) {
      throw std::invalid_argument("four-Weyl radii must be strictly increasing");
    }
    for (const double radius : radii) {
      if (!std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument("nonfinite four-Weyl radius");
      }
    }
    for (const double cosine : cos_theta) {
      if (!std::isfinite(cosine) || cosine < -1.0 || cosine > 1.0) {
        throw std::invalid_argument("invalid four-Weyl cos(theta)");
      }
    }

    FourWeylOutputPacker result;
    result.enabled_ = true;
    result.metadata_ = std::move(metadata);
    result.radii_ = radii;
    result.cos_theta_ = cos_theta;
    result.kerr_spin_ = kerr_spin;
    result.compactification_length_ = compactification_length;
    result.horizon_radius_ = horizon_radius;
    result.stored_modes_ = four_weyl_detail::stored_modes(result.metadata_);

    for (int field_index = 0;
         field_index < static_cast<int>(FourWeylField::Count); ++field_index) {
      const auto field = static_cast<FourWeylField>(field_index);
      const auto& registry = four_weyl_order(field) == 1
                                 ? result.metadata_.parent_modes
                                 : result.metadata_.target_modes;
      for (const int mode : registry) {
        const auto found = std::lower_bound(result.stored_modes_.begin(),
                                            result.stored_modes_.end(), mode);
        result.entries_.push_back(
            {field_index, static_cast<int>(std::distance(
                              result.stored_modes_.begin(), found))});
      }
    }

    result.device_radii_ = decltype(result.device_radii_)(
        "four_weyl_output_radii", result.radii_.size());
    result.device_cos_theta_ = decltype(result.device_cos_theta_)(
        "four_weyl_output_cos_theta", result.cos_theta_.size());
    result.device_entries_ = decltype(result.device_entries_)(
        "four_weyl_output_entries", result.entries_.size());
    const std::size_t count = result.entries_.size() * result.radii_.size() *
                              result.cos_theta_.size();
    result.device_packed_ = decltype(result.device_packed_)(
        "four_weyl_output_packed", count);

    auto host_radii = Kokkos::create_mirror_view(result.device_radii_);
    auto host_cos = Kokkos::create_mirror_view(result.device_cos_theta_);
    auto host_entries = Kokkos::create_mirror_view(result.device_entries_);
    for (std::size_t i = 0; i < radii.size(); ++i) host_radii(i) = radii[i];
    for (std::size_t i = 0; i < cos_theta.size(); ++i) {
      host_cos(i) = cos_theta[i];
    }
    for (std::size_t i = 0; i < result.entries_.size(); ++i) {
      host_entries(i) = result.entries_[i];
    }
    Kokkos::deep_copy(result.device_radii_, host_radii);
    Kokkos::deep_copy(result.device_cos_theta_, host_cos);
    Kokkos::deep_copy(result.device_entries_, host_entries);
    return result;
  }

  [[nodiscard]] bool is_enabled() const noexcept { return enabled_; }
  [[nodiscard]] bool should_emit(const std::uint64_t step) const noexcept {
    return enabled_ && step % metadata_.output_cadence_steps == 0;
  }
  [[nodiscard]] const FourWeylOutputMetadata& metadata() const {
    if (!enabled_) throw std::logic_error("four-Weyl output is disabled");
    return metadata_;
  }
  [[nodiscard]] std::size_t packed_capacity() const noexcept {
    return device_packed_.extent(0);
  }

  template <class Execution, class View>
  std::vector<FourWeylOutputRecord> pack_nodal(
      const Execution& execution, const std::uint64_t step, const double time,
      const View& psi4_order1_f, const View& psi0_order1_z,
      const View& psi4_order2_f, const View& psi0_order2_z) const {
    static_assert(View::rank == 3,
                  "four-Weyl nodal inputs must have rank three");
    if (!enabled_) throw std::logic_error("four-Weyl output is disabled");
    if (!std::isfinite(time) || time < 0.0) {
      throw std::invalid_argument("four-Weyl output time must be finite");
    }
    const std::array<std::array<std::size_t, 3>, 4> extents{{
        {psi4_order1_f.extent(0), psi4_order1_f.extent(1),
         psi4_order1_f.extent(2)},
        {psi0_order1_z.extent(0), psi0_order1_z.extent(1),
         psi0_order1_z.extent(2)},
        {psi4_order2_f.extent(0), psi4_order2_f.extent(1),
         psi4_order2_f.extent(2)},
        {psi0_order2_z.extent(0), psi0_order2_z.extent(1),
         psi0_order2_z.extent(2)},
    }};
    const std::array<std::size_t, 3> expected{
        stored_modes_.size(), radii_.size(), cos_theta_.size()};
    for (const auto& extent : extents) {
      if (extent != expected) {
        throw std::invalid_argument(
            "missing or inconsistent four-Weyl nodal input extents");
      }
    }

    const auto entries = device_entries_;
    const auto radii = device_radii_;
    const auto cos_theta = device_cos_theta_;
    const auto packed = device_packed_;
    const std::size_t radial_count = radii_.size();
    const std::size_t theta_count = cos_theta_.size();
    const double spin = kerr_spin_;
    const double length = compactification_length_;
    Kokkos::parallel_for(
        "pack_four_weyl_fields",
        Kokkos::RangePolicy<Execution>(execution, 0, packed.extent(0)),
        KOKKOS_LAMBDA(const std::size_t flat) {
          const std::size_t angular_radial = radial_count * theta_count;
          const std::size_t entry_index = flat / angular_radial;
          const std::size_t remainder = flat - entry_index * angular_radial;
          const std::size_t radial = remainder / theta_count;
          const std::size_t theta = remainder - radial * theta_count;
          const auto entry = entries(entry_index);
          const auto field = static_cast<FourWeylField>(entry.field);
          Complex regularized;
          switch (field) {
            case FourWeylField::Psi4Order1:
              regularized = psi4_order1_f(entry.mode_index, radial, theta);
              break;
            case FourWeylField::Psi0Order1:
              regularized = psi0_order1_z(entry.mode_index, radial, theta);
              break;
            case FourWeylField::Psi4Order2:
              regularized = psi4_order2_f(entry.mode_index, radial, theta);
              break;
            case FourWeylField::Psi0Order2:
              regularized = psi0_order2_z(entry.mode_index, radial, theta);
              break;
            case FourWeylField::Count: return;
          }
          four_weyl_detail::PackedValue value;
          value.regularized = regularized;
          value.raw_code_tetrad = four_weyl_detail::raw_from_regularized(
              field, regularized, radii(radial), cos_theta(theta), spin,
              length);
          if (radii(radial) == 0.0) {
            value.asymptotic_power = four_weyl_spin(field) == -2 ? 1 : 5;
            value.asymptotic_coefficient =
                four_weyl_detail::scri_coefficient_from_regularized(
                    field, regularized, length);
          }
          packed(flat) = value;
        });
    auto host = Kokkos::create_mirror_view(device_packed_);
    Kokkos::deep_copy(execution, host, device_packed_);
    execution.fence("copy four-Weyl output seam");

    std::vector<FourWeylOutputRecord> result;
    result.reserve(device_packed_.extent(0));
    for (std::size_t flat = 0; flat < device_packed_.extent(0); ++flat) {
      const std::size_t angular_radial = radial_count * theta_count;
      const std::size_t entry_index = flat / angular_radial;
      const std::size_t remainder = flat - entry_index * angular_radial;
      const std::size_t radial = remainder / theta_count;
      const std::size_t theta = remainder - radial * theta_count;
      const auto entry = entries_[entry_index];
      const auto packed_value = host(flat);
      if (!four_weyl_detail::finite(packed_value.regularized) ||
          !four_weyl_detail::finite(packed_value.raw_code_tetrad) ||
          !four_weyl_detail::finite(packed_value.asymptotic_coefficient)) {
        throw std::runtime_error("nonfinite four-Weyl packed value");
      }
      FourWeylBoundary boundary = FourWeylBoundary::Interior;
      if (radial == 0) boundary = FourWeylBoundary::Scri;
      if (radial + 1 == radial_count) boundary = FourWeylBoundary::Horizon;
      result.push_back(
          {step,
           time,
           FourWeylAngularRepresentation::SignedAzimuthalModalPolarNodal,
           static_cast<FourWeylField>(entry.field),
           stored_modes_[static_cast<std::size_t>(entry.mode_index)],
           -1,
           static_cast<int>(radial),
           static_cast<int>(theta),
           radii_[radial],
           cos_theta_[theta],
           boundary,
           packed_value.regularized,
           packed_value.raw_code_tetrad,
           packed_value.asymptotic_power,
           packed_value.asymptotic_coefficient});
    }
    return result;
  }

  std::vector<FourWeylOutputRecord> pack_scri_modal(
      const std::uint64_t step, const double time,
      const FourWeylScriModalInput& input) const {
    if (!enabled_) throw std::logic_error("four-Weyl output is disabled");
    if (!std::isfinite(time) || time < 0.0) {
      throw std::invalid_argument("four-Weyl output time must be finite");
    }
    std::vector<FourWeylOutputRecord> result;
    for (int field_index = 0;
         field_index < static_cast<int>(FourWeylField::Count); ++field_index) {
      const auto field = static_cast<FourWeylField>(field_index);
      const auto& registry = four_weyl_order(field) == 1
                                 ? metadata_.parent_modes
                                 : metadata_.target_modes;
      const int ell_max = four_weyl_order(field) == 1
                              ? metadata_.ell_max_first
                              : metadata_.ell_max_second;
      const auto& values = input.fields[static_cast<std::size_t>(field_index)];
      std::size_t expected_count = 0;
      for (const int mode : registry) {
        expected_count += static_cast<std::size_t>(
            ell_max - std::max(2, std::abs(mode)) + 1);
      }
      if (values.size() != expected_count) {
        throw std::invalid_argument("missing four-Weyl scri modal values");
      }
      std::size_t index = 0;
      for (const int mode : registry) {
        for (int ell = std::max(2, std::abs(mode)); ell <= ell_max;
             ++ell, ++index) {
          const auto& value = values[index];
          if (value.m != mode || value.ell != ell ||
              !four_weyl_detail::finite(value.regularized)) {
            throw std::invalid_argument(
                "unordered, inconsistent, or nonfinite scri modal input");
          }
          const int power = four_weyl_spin(field) == -2 ? 1 : 5;
          result.push_back(
              {step,
               time,
               FourWeylAngularRepresentation::SpinWeightedSphericalModalAtScri,
               field,
               mode,
               ell,
               -1,
               -1,
               0.0,
               0.0,
               FourWeylBoundary::Scri,
               value.regularized,
               Complex(0.0, 0.0),
               power,
               four_weyl_detail::scri_coefficient_from_regularized(
                   field, value.regularized, compactification_length_)});
        }
      }
    }
    return result;
  }

 private:
  bool enabled_ = false;
  FourWeylOutputMetadata metadata_;
  std::vector<double> radii_;
  std::vector<double> cos_theta_;
  std::vector<int> stored_modes_;
  std::vector<four_weyl_detail::FieldModeEntry> entries_;
  double kerr_spin_ = 0.0;
  double compactification_length_ = 0.0;
  double horizon_radius_ = 0.0;
  Kokkos::View<double*, MemorySpace> device_radii_;
  Kokkos::View<double*, MemorySpace> device_cos_theta_;
  Kokkos::View<four_weyl_detail::FieldModeEntry*, MemorySpace> device_entries_;
  mutable Kokkos::View<four_weyl_detail::PackedValue*, MemorySpace>
      device_packed_;
};

inline void write_four_weyl_metadata(
    std::ostream& output, const FourWeylOutputMetadata& metadata) {
  four_weyl_detail::validate_metadata(metadata);
  output.imbue(std::locale::classic());
  const auto line = [&output](const char* key, const auto& value) {
    output << key << '=' << std::quoted(value) << '\n';
  };
  line("schema", metadata.schema);
  output << "version=" << metadata.version << '\n';
  line("git_commit", metadata.git_commit);
  output << "runtime_config_schema_version="
         << metadata.runtime_config_schema_version << '\n';
  line("gauge_id", metadata.gauge_id);
  line("tetrad_id", metadata.tetrad_id);
  line("perturbative_convention", metadata.perturbative_convention);
  line("psi4_scaling", metadata.psi4_scaling);
  line("psi0_scaling", metadata.psi0_scaling);
  line("linear_method",
       std::string(four_weyl_linear_method_name(metadata.linear_method)));
  line("source_method",
       std::string(four_weyl_source_method_name(metadata.source_method)));
  output << "source_normalization_version="
         << metadata.source_normalization_version << '\n';
  line("source_normalization_name", metadata.source_normalization_name);
  line("initial_policy",
       std::string(four_weyl_initial_policy_name(metadata.initial_policy)));
  output << "ell_max_first=" << metadata.ell_max_first << '\n';
  output << "ell_max_second=" << metadata.ell_max_second << '\n';
  line("parent_modes", four_weyl_detail::encode_modes(metadata.parent_modes));
  line("target_modes", four_weyl_detail::encode_modes(metadata.target_modes));
  output << "output_cadence_steps=" << metadata.output_cadence_steps << '\n';
  line("radial_discretization",
       std::string(radial_discretization_name(
           metadata.radial_discretization)));
  line("sharp_semantics", metadata.sharp_semantics);
  line("scri_semantics", metadata.scri_semantics);
  line("horizon_semantics", metadata.horizon_semantics);
}

inline FourWeylOutputMetadata read_four_weyl_metadata(std::istream& input) {
  input.imbue(std::locale::classic());
  const auto read_value = [&input](const char* expected_key,
                                   const bool quoted) {
    std::string line;
    if (!std::getline(input, line)) {
      throw std::runtime_error("truncated four-Weyl metadata");
    }
    const auto split = line.find('=');
    if (split == std::string::npos || line.substr(0, split) != expected_key) {
      throw std::runtime_error(std::string("missing four-Weyl metadata key ") +
                               expected_key);
    }
    std::string value = line.substr(split + 1);
    if (quoted) {
      std::istringstream parser(value);
      std::string decoded;
      parser >> std::quoted(decoded);
      parser >> std::ws;
      if (!parser || !parser.eof()) {
        throw std::runtime_error("invalid quoted four-Weyl metadata value");
      }
      return decoded;
    }
    return value;
  };

  FourWeylOutputMetadata result;
  result.schema = read_value("schema", true);
  result.version = static_cast<std::uint32_t>(four_weyl_detail::parse_uint64(
      read_value("version", false), "metadata version"));
  result.git_commit = read_value("git_commit", true);
  result.runtime_config_schema_version = four_weyl_detail::parse_int(
      read_value("runtime_config_schema_version", false),
      "runtime config schema version");
  result.gauge_id = read_value("gauge_id", true);
  result.tetrad_id = read_value("tetrad_id", true);
  result.perturbative_convention =
      read_value("perturbative_convention", true);
  result.psi4_scaling = read_value("psi4_scaling", true);
  result.psi0_scaling = read_value("psi0_scaling", true);
  result.linear_method = parse_four_weyl_linear_method(
      read_value("linear_method", true));
  result.source_method = parse_four_weyl_source_method(
      read_value("source_method", true));
  result.source_normalization_version = static_cast<std::uint32_t>(
      four_weyl_detail::parse_uint64(
          read_value("source_normalization_version", false),
          "source normalization version"));
  result.source_normalization_name =
      read_value("source_normalization_name", true);
  result.initial_policy = parse_four_weyl_initial_policy(
      read_value("initial_policy", true));
  result.ell_max_first = four_weyl_detail::parse_int(
      read_value("ell_max_first", false), "ell_max_first");
  result.ell_max_second = four_weyl_detail::parse_int(
      read_value("ell_max_second", false), "ell_max_second");
  result.parent_modes = four_weyl_detail::decode_modes(
      read_value("parent_modes", true));
  result.target_modes = four_weyl_detail::decode_modes(
      read_value("target_modes", true));
  result.output_cadence_steps = four_weyl_detail::parse_uint64(
      read_value("output_cadence_steps", false), "output cadence");
  result.radial_discretization = parse_radial_discretization(
      read_value("radial_discretization", true));
  result.sharp_semantics = read_value("sharp_semantics", true);
  result.scri_semantics = read_value("scri_semantics", true);
  result.horizon_semantics = read_value("horizon_semantics", true);
  std::string trailing;
  if (std::getline(input, trailing)) {
    throw std::runtime_error("trailing data in four-Weyl metadata");
  }
  four_weyl_detail::validate_metadata(result);
  return result;
}

inline void write_four_weyl_csv_header(std::ostream& output) {
  output << "schema_version,step,time,representation,field,spin,order,m,ell,"
            "radial_index,theta_index,radius,cos_theta,boundary,regularized_re,"
            "regularized_im,raw_code_tetrad_re,raw_code_tetrad_im,"
            "asymptotic_power,asymptotic_coefficient_re,"
            "asymptotic_coefficient_im\n";
}

inline void write_four_weyl_csv_records(
    std::ostream& output, const std::vector<FourWeylOutputRecord>& records) {
  output.imbue(std::locale::classic());
  output << std::setprecision(17);
  for (const auto& record : records) {
    if (!std::isfinite(record.time) || !std::isfinite(record.radius) ||
        !std::isfinite(record.cos_theta) ||
        !four_weyl_detail::finite(record.regularized) ||
        !four_weyl_detail::finite(record.raw_code_tetrad) ||
        !four_weyl_detail::finite(record.asymptotic_coefficient)) {
      throw std::invalid_argument("cannot write nonfinite four-Weyl record");
    }
    output << four_weyl_output_schema_version << ',' << record.step << ','
           << record.time << ','
           << four_weyl_representation_name(record.representation) << ','
           << four_weyl_field_name(record.field) << ','
           << four_weyl_spin(record.field) << ','
           << four_weyl_order(record.field) << ',' << record.m << ','
           << record.ell << ',' << record.radial_index << ','
           << record.theta_index << ',' << record.radius << ','
           << record.cos_theta << ','
           << four_weyl_boundary_name(record.boundary) << ','
           << record.regularized.real() << ',' << record.regularized.imag()
           << ',' << record.raw_code_tetrad.real() << ','
           << record.raw_code_tetrad.imag() << ',' << record.asymptotic_power
           << ',' << record.asymptotic_coefficient.real() << ','
           << record.asymptotic_coefficient.imag() << '\n';
  }
}

inline std::vector<FourWeylOutputRecord> read_four_weyl_csv_records(
    std::istream& input) {
  input.imbue(std::locale::classic());
  std::ostringstream expected_header;
  write_four_weyl_csv_header(expected_header);
  std::string header;
  if (!std::getline(input, header) || header + '\n' != expected_header.str()) {
    throw std::runtime_error("invalid four-Weyl CSV header");
  }
  std::vector<FourWeylOutputRecord> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) throw std::runtime_error("blank four-Weyl CSV row");
    const auto columns = four_weyl_detail::split_csv(line);
    if (columns.size() != 21) {
      throw std::runtime_error("invalid four-Weyl CSV column count");
    }
    const auto version = four_weyl_detail::parse_uint64(
        columns[0], "CSV schema version");
    if (version != four_weyl_output_schema_version) {
      throw std::runtime_error("unsupported four-Weyl CSV schema version");
    }
    FourWeylOutputRecord record;
    record.step = four_weyl_detail::parse_uint64(columns[1], "step");
    record.time = four_weyl_detail::parse_finite_double(columns[2], "time");
    record.representation = parse_four_weyl_representation(columns[3]);
    record.field = parse_four_weyl_field(columns[4]);
    if (four_weyl_detail::parse_int(columns[5], "spin") !=
            four_weyl_spin(record.field) ||
        four_weyl_detail::parse_int(columns[6], "order") !=
            four_weyl_order(record.field)) {
      throw std::runtime_error("inconsistent four-Weyl CSV field metadata");
    }
    record.m = four_weyl_detail::parse_int(columns[7], "m");
    record.ell = four_weyl_detail::parse_int(columns[8], "ell");
    record.radial_index =
        four_weyl_detail::parse_int(columns[9], "radial index");
    record.theta_index =
        four_weyl_detail::parse_int(columns[10], "theta index");
    record.radius =
        four_weyl_detail::parse_finite_double(columns[11], "radius");
    record.cos_theta =
        four_weyl_detail::parse_finite_double(columns[12], "cos theta");
    record.boundary = parse_four_weyl_boundary(columns[13]);
    record.regularized =
        Complex(four_weyl_detail::parse_finite_double(columns[14],
                                                       "regularized real"),
                four_weyl_detail::parse_finite_double(columns[15],
                                                       "regularized imag"));
    record.raw_code_tetrad =
        Complex(four_weyl_detail::parse_finite_double(columns[16], "raw real"),
                four_weyl_detail::parse_finite_double(columns[17], "raw imag"));
    record.asymptotic_power =
        four_weyl_detail::parse_int(columns[18], "asymptotic power");
    record.asymptotic_coefficient = Complex(
        four_weyl_detail::parse_finite_double(columns[19],
                                               "asymptotic real"),
        four_weyl_detail::parse_finite_double(columns[20],
                                               "asymptotic imag"));
    if ((record.boundary == FourWeylBoundary::Scri) !=
            (record.asymptotic_power == 1 || record.asymptotic_power == 5) ||
        (record.boundary == FourWeylBoundary::Scri &&
         Kokkos::abs(record.raw_code_tetrad) != 0.0) ||
        (record.boundary != FourWeylBoundary::Scri &&
         record.asymptotic_power != 0)) {
      throw std::runtime_error("inconsistent four-Weyl boundary semantics");
    }
    result.push_back(record);
  }
  return result;
}

}  // namespace teuk
