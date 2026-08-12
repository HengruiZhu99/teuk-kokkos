#include "test_harness.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "teuk/checkpoint_identity.hpp"
#include "teuk/sha256.hpp"

namespace {

teuk::Sha256Digest hash_text(const std::string& text) {
  teuk::Sha256 hash;
  hash.update(text);
  return hash.finalize();
}

class TemporaryIdentityDirectory {
 public:
  TemporaryIdentityDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("teuk-checkpoint-identity-" +
             std::to_string(std::chrono::steady_clock::now()
                                .time_since_epoch()
                                .count()));
    std::filesystem::create_directory(path_);
  }
  ~TemporaryIdentityDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void write_bytes(const std::filesystem::path& path,
                 const std::vector<std::uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) throw std::runtime_error("failed writing SHA-256 test file");
}

}  // namespace

TEST_CASE("SHA-256 matches NIST vectors and chunk boundaries") {
  CHECK(teuk::sha256_hex(hash_text("")) ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(teuk::sha256_hex(hash_text("abc")) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::string multi =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  CHECK(teuk::sha256_hex(hash_text(multi)) ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

  teuk::Sha256 chunked;
  for (const char value : multi) {
    const auto byte = static_cast<std::uint8_t>(value);
    chunked.update(std::span<const std::uint8_t>(&byte, 1));
  }
  CHECK(chunked.finalize() == hash_text(multi));
  const auto parsed = teuk::parse_sha256_hex(teuk::sha256_hex(hash_text(multi)));
  CHECK(parsed == hash_text(multi));
  bool uppercase_rejected = false;
  try {
    static_cast<void>(teuk::parse_sha256_hex(
        "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"));
  } catch (const std::invalid_argument&) {
    uppercase_rejected = true;
  }
  CHECK(uppercase_rejected);
}

TEST_CASE("primary checkpoint identity binds names sizes and exact bytes") {
  TemporaryIdentityDirectory temporary;
  const auto metadata = temporary.path() / "metadata.txt";
  const auto state = temporary.path() / "state.bin";
  write_bytes(metadata, {'m', 'e', 't', 'a'});
  write_bytes(state, {0, 1, 2, 3, 4, 5});
  const auto baseline =
      teuk::make_primary_checkpoint_content_identity(metadata, state);
  teuk::validate_primary_checkpoint_content_identity(baseline);
  CHECK(baseline.algorithm == teuk::checkpoint_content_hash_algorithm);
  CHECK(baseline.state_schema == teuk::primary_checkpoint_state_schema);
  teuk::Sha256 independent;
  independent.update(teuk::primary_checkpoint_content_domain);
  const std::uint8_t terminator = 0;
  independent.update(std::span<const std::uint8_t>(&terminator, 1));
  const auto add_u64 = [&](const std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
      bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
    }
    independent.update(bytes);
  };
  const auto add = [&](const std::string& name,
                       const std::vector<std::uint8_t>& bytes) {
    add_u64(name.size());
    independent.update(name);
    add_u64(bytes.size());
    independent.update(bytes);
  };
  add("metadata.txt", {'m', 'e', 't', 'a'});
  add("state.bin", {0, 1, 2, 3, 4, 5});
  CHECK(independent.finalize() == baseline.digest);

  std::vector<std::uint8_t> large_state(65537);
  for (std::size_t index = 0; index < large_state.size(); ++index) {
    large_state[index] = static_cast<std::uint8_t>(index % 251U);
  }
  write_bytes(state, large_state);
  const auto large =
      teuk::make_primary_checkpoint_content_identity(metadata, state);
  // Independently generated with Python hashlib and explicit little-endian
  // framing.  This freezes both the domain encoding and the 64 KiB read split.
  CHECK(teuk::sha256_hex(large.digest) ==
        "017f1460116817386c0bb3b7df7a7a97b4f476389243d5bbea004061cb66df6d");

  write_bytes(state, {0, 1, 2, 3, 4, 4});
  const auto state_mutated =
      teuk::make_primary_checkpoint_content_identity(metadata, state);
  CHECK(state_mutated != baseline);
  write_bytes(state, {0, 1, 2, 3, 4, 5});
  write_bytes(metadata, {'m', 'e', 't', 'b'});
  const auto metadata_mutated =
      teuk::make_primary_checkpoint_content_identity(metadata, state);
  CHECK(metadata_mutated != baseline);
  write_bytes(metadata, {'m', 'e', 't', 'a'});
  write_bytes(state, {0, 1, 2, 3, 4});
  const auto truncated =
      teuk::make_primary_checkpoint_content_identity(metadata, state);
  CHECK(truncated != baseline);
  write_bytes(metadata, {0, 1, 2, 3, 4, 5});
  write_bytes(state, {'m', 'e', 't', 'a'});
  const auto swapped =
      teuk::make_primary_checkpoint_content_identity(metadata, state);
  CHECK(swapped != baseline);
}
