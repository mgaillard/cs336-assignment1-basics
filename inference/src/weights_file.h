#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "binary_io.h"

namespace cs336 {

struct TensorInfo {
  std::vector<std::uint32_t> shape;
  std::uint64_t offset = 0;
  std::uint64_t nbytes = 0;
  std::size_t numel() const;
};

// Reader of model.bin (written by convert_checkpoint.py). Opening the file parses the header and
// the tensor table and checks the stored architecture against the compile-time constants of
// model_config.h; tensors are then read on demand straight into the destination buffers.
class WeightsFile {
 public:
  explicit WeightsFile(const std::string& path);

  // Reads tensor `name` (float32) into `dst` after checking that its shape is `expected_shape`.
  void read_tensor(const std::string& name, const std::vector<std::uint32_t>& expected_shape, float* dst);

 private:
  BinaryReader reader_;
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace cs336
