#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace teuk {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
 public:
  Sha256() = default;

  void update(const std::span<const std::uint8_t> bytes) {
    if (finalized_) throw std::logic_error("cannot update finalized SHA-256");
    if (bytes.size() > UINT64_MAX - total_bytes_) {
      throw std::overflow_error("SHA-256 input length overflow");
    }
    total_bytes_ += static_cast<std::uint64_t>(bytes.size());
    for (const std::uint8_t byte : bytes) {
      block_[block_size_++] = byte;
      if (block_size_ == block_.size()) {
        transform(block_);
        block_size_ = 0;
      }
    }
  }

  void update(const std::string_view text) {
    update(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
  }

  Sha256Digest finalize() {
    if (finalized_) throw std::logic_error("SHA-256 already finalized");
    if (total_bytes_ > UINT64_MAX / 8U) {
      throw std::overflow_error("SHA-256 bit length overflow");
    }
    const std::uint64_t bit_length = total_bytes_ * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56) {
      while (block_size_ < block_.size()) block_[block_size_++] = 0U;
      transform(block_);
      block_size_ = 0;
    }
    while (block_size_ < 56) block_[block_size_++] = 0U;
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[block_size_++] =
          static_cast<std::uint8_t>((bit_length >> shift) & 0xffU);
    }
    transform(block_);
    finalized_ = true;

    Sha256Digest digest{};
    for (std::size_t word = 0; word < state_.size(); ++word) {
      digest[4 * word] = static_cast<std::uint8_t>(state_[word] >> 24U);
      digest[4 * word + 1] =
          static_cast<std::uint8_t>(state_[word] >> 16U);
      digest[4 * word + 2] =
          static_cast<std::uint8_t>(state_[word] >> 8U);
      digest[4 * word + 3] = static_cast<std::uint8_t>(state_[word]);
    }
    return digest;
  }

 private:
  static constexpr std::array<std::uint32_t, 64> round_constants_{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

  static constexpr std::uint32_t rotate_right(const std::uint32_t value,
                                               const int shift) {
    return (value >> shift) | (value << (32 - shift));
  }

  void transform(const std::array<std::uint8_t, 64>& block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i) {
      words[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24U) |
                 (static_cast<std::uint32_t>(block[4 * i + 1]) << 16U) |
                 (static_cast<std::uint32_t>(block[4 * i + 2]) << 8U) |
                 static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (std::size_t i = 16; i < words.size(); ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                               rotate_right(words[i - 15], 18) ^
                               (words[i - 15] >> 3U);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                               rotate_right(words[i - 2], 19) ^
                               (words[i - 2] >> 10U);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t i = 0; i < words.size(); ++i) {
      const std::uint32_t sigma1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                   rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary1 =
          h + sigma1 + choice + round_constants_[i] + words[i];
      const std::uint32_t sigma0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                   rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::uint64_t total_bytes_ = 0;
  std::size_t block_size_ = 0;
  bool finalized_ = false;
};

inline std::string sha256_hex(const Sha256Digest& digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(64, '0');
  for (std::size_t i = 0; i < digest.size(); ++i) {
    result[2 * i] = digits[digest[i] >> 4U];
    result[2 * i + 1] = digits[digest[i] & 0x0fU];
  }
  return result;
}

inline Sha256Digest parse_sha256_hex(const std::string_view text) {
  if (text.size() != 64) {
    throw std::invalid_argument("SHA-256 digest must have 64 hex digits");
  }
  const auto nibble = [](const char value) -> std::uint8_t {
    if (value >= '0' && value <= '9') {
      return static_cast<std::uint8_t>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<std::uint8_t>(10 + value - 'a');
    }
    throw std::invalid_argument(
        "SHA-256 digest must use canonical lowercase hex");
  };
  Sha256Digest result{};
  for (std::size_t i = 0; i < result.size(); ++i) {
    result[i] = static_cast<std::uint8_t>((nibble(text[2 * i]) << 4U) |
                                          nibble(text[2 * i + 1]));
  }
  return result;
}

}  // namespace teuk
