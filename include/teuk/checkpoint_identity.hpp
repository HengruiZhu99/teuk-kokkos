#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "teuk/sha256.hpp"

namespace teuk {

class PipelineCheckpointCodec;

inline constexpr const char* checkpoint_content_hash_algorithm = "sha256";
inline constexpr const char* primary_checkpoint_state_schema =
    "teuk.pipeline-state-layout-v1";
inline constexpr const char* primary_checkpoint_content_domain =
    "teuk.pipeline-checkpoint-content.v1";
inline constexpr const char* primary_checkpoint_state_domain =
    "teuk.pipeline-checkpoint-state.v1";

struct PrimaryCheckpointContentIdentity {
  std::string algorithm;
  std::string state_schema;
  Sha256Digest digest{};

  friend bool operator==(const PrimaryCheckpointContentIdentity&,
                         const PrimaryCheckpointContentIdentity&) = default;
};

inline void validate_primary_checkpoint_content_identity(
    const PrimaryCheckpointContentIdentity& identity) {
  if (identity.algorithm != checkpoint_content_hash_algorithm ||
      identity.state_schema != primary_checkpoint_state_schema) {
    throw std::invalid_argument(
        "unsupported primary checkpoint content identity");
  }
}

inline PrimaryCheckpointContentIdentity primary_checkpoint_identity_from_hex(
    const std::string_view digest) {
  return {checkpoint_content_hash_algorithm, primary_checkpoint_state_schema,
          parse_sha256_hex(digest)};
}

// This receipt is deliberately not aggregate-constructible.  A syntactically
// valid SHA-256 string is checkpoint metadata, not evidence that the matching
// primary state was validated and loaded.  Only the primary checkpoint codec
// may issue a receipt after validating the exact metadata/state byte buffers.
class VerifiedPrimaryCheckpointReceipt {
 public:
  VerifiedPrimaryCheckpointReceipt() = delete;

  [[nodiscard]] PrimaryCheckpointContentIdentity content_identity() const {
    return {checkpoint_content_hash_algorithm, primary_checkpoint_state_schema,
            content_digest_};
  }
  [[nodiscard]] const Sha256Digest& state_digest() const {
    return state_digest_;
  }
  [[nodiscard]] double time() const { return time_; }
  [[nodiscard]] std::uint64_t step() const { return step_; }
  [[nodiscard]] std::size_t state_value_count() const {
    return state_value_count_;
  }

  friend bool operator==(const VerifiedPrimaryCheckpointReceipt&,
                         const VerifiedPrimaryCheckpointReceipt&) = default;

 private:
  VerifiedPrimaryCheckpointReceipt(const Sha256Digest content_digest,
                                   const Sha256Digest state_digest,
                                   const double time,
                                   const std::uint64_t step,
                                   const std::size_t state_value_count)
      : content_digest_(content_digest),
        state_digest_(state_digest),
        time_(time),
        step_(step),
        state_value_count_(state_value_count) {}

  Sha256Digest content_digest_{};
  Sha256Digest state_digest_{};
  double time_ = 0.0;
  std::uint64_t step_ = 0;
  std::size_t state_value_count_ = 0;

  friend class PipelineCheckpointCodec;
};

namespace checkpoint_identity_detail {

inline void add_u64_little_endian(Sha256& hash, const std::uint64_t value) {
  std::array<std::uint8_t, 8> bytes{};
  for (std::size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
  }
  hash.update(bytes);
}

inline std::uint64_t checked_file_size(const std::filesystem::path& path) {
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error || size > std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("cannot determine checkpoint content size");
  }
  return static_cast<std::uint64_t>(size);
}

inline void add_file(Sha256& hash, const std::filesystem::path& path,
                     const std::string_view logical_name) {
  add_u64_little_endian(hash, logical_name.size());
  hash.update(logical_name);
  add_u64_little_endian(hash, checked_file_size(path));
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open checkpoint content file");
  std::array<std::uint8_t, 65536> buffer{};
  while (input) {
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      hash.update(std::span<const std::uint8_t>(
          buffer.data(), static_cast<std::size_t>(count)));
    }
  }
  if (!input.eof()) throw std::runtime_error("failed reading checkpoint content");
}

inline void add_named_bytes(Sha256& hash,
                            const std::span<const std::uint8_t> bytes,
                            const std::string_view logical_name) {
  add_u64_little_endian(hash, logical_name.size());
  hash.update(logical_name);
  add_u64_little_endian(hash, bytes.size());
  hash.update(bytes);
}

inline Sha256Digest checkpoint_state_digest(
    const std::span<const std::uint8_t> state) {
  Sha256 hash;
  hash.update(primary_checkpoint_state_domain);
  const std::uint8_t terminator = 0;
  hash.update(std::span<const std::uint8_t>(&terminator, 1));
  add_u64_little_endian(hash, state.size());
  hash.update(state);
  return hash.finalize();
}

}  // namespace checkpoint_identity_detail

inline PrimaryCheckpointContentIdentity make_primary_checkpoint_content_identity(
    const std::span<const std::uint8_t> metadata,
    const std::span<const std::uint8_t> state) {
  Sha256 hash;
  hash.update(primary_checkpoint_content_domain);
  const std::uint8_t terminator = 0;
  hash.update(std::span<const std::uint8_t>(&terminator, 1));
  checkpoint_identity_detail::add_named_bytes(hash, metadata, "metadata.txt");
  checkpoint_identity_detail::add_named_bytes(hash, state, "state.bin");
  return {checkpoint_content_hash_algorithm, primary_checkpoint_state_schema,
          hash.finalize()};
}

inline PrimaryCheckpointContentIdentity make_primary_checkpoint_content_identity(
    const std::filesystem::path& metadata,
    const std::filesystem::path& state) {
  Sha256 hash;
  hash.update(primary_checkpoint_content_domain);
  const std::uint8_t terminator = 0;
  hash.update(std::span<const std::uint8_t>(&terminator, 1));
  checkpoint_identity_detail::add_file(hash, metadata, "metadata.txt");
  checkpoint_identity_detail::add_file(hash, state, "state.bin");
  return {checkpoint_content_hash_algorithm, primary_checkpoint_state_schema,
          hash.finalize()};
}

}  // namespace teuk
