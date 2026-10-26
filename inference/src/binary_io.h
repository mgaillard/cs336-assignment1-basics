#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs336 {

// Little-endian reader on top of FILE*. Offsets use fseeko (64-bit with -D_FILE_OFFSET_BITS=64) so
// the 32-bit build can open files larger than 2 GB.
class BinaryReader {
 public:
  explicit BinaryReader(const std::string& path) : path_(path) {
    file_ = std::fopen(path.c_str(), "rb");
    if (!file_) throw std::runtime_error("cannot open file: " + path);
  }
  ~BinaryReader() {
    if (file_) std::fclose(file_);
  }
  BinaryReader(const BinaryReader&) = delete;
  BinaryReader& operator=(const BinaryReader&) = delete;

  void read_bytes(void* dst, std::size_t n) {
    if (n != 0 && std::fread(dst, 1, n, file_) != n) throw std::runtime_error("unexpected end of file: " + path_);
  }
  std::uint16_t read_u16() {
    unsigned char b[2];
    read_bytes(b, 2);
    return static_cast<std::uint16_t>(b[0] | (b[1] << 8));
  }
  std::uint32_t read_u32() {
    unsigned char b[4];
    read_bytes(b, 4);
    return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
  }
  std::uint64_t read_u64() {
    const std::uint64_t lo = read_u32();
    const std::uint64_t hi = read_u32();
    return lo | (hi << 32);
  }
  float read_f32() {
    const std::uint32_t bits = read_u32();
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
  }
  double read_f64() {
    const std::uint64_t bits = read_u64();
    double value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
  }
  std::string read_string(std::size_t n) {
    std::string s(n, '\0');
    read_bytes(&s[0], n);
    return s;
  }
  void seek(std::uint64_t offset) {
    if (fseeko(file_, static_cast<off_t>(offset), SEEK_SET) != 0) throw std::runtime_error("seek failed: " + path_);
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  std::FILE* file_ = nullptr;
};

inline bool host_is_little_endian() {
  const std::uint32_t one = 1;
  unsigned char first;
  std::memcpy(&first, &one, 1);
  return first == 1;
}

}  // namespace cs336
