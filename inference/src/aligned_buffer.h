#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cs336 {

// Heap array of T aligned to 64 bytes (portable on 32-bit and 64-bit targets), zero-initialized.
template <typename T>
class AlignedBuffer {
 public:
  static constexpr std::size_t kAlignment = 64;

  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t size) : size_(size) {
    raw_.reset(new unsigned char[size * sizeof(T) + kAlignment]());
    std::uintptr_t p = reinterpret_cast<std::uintptr_t>(raw_.get());
    p = (p + (kAlignment - 1)) & ~static_cast<std::uintptr_t>(kAlignment - 1);
    data_ = reinterpret_cast<T*>(p);
  }

  AlignedBuffer(AlignedBuffer&& other) noexcept
      : raw_(std::move(other.raw_)), data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
    raw_ = std::move(other.raw_);
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }
  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }

 private:
  std::unique_ptr<unsigned char[]> raw_;
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace cs336
