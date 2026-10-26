#include "weights_file.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

#include "model_config.h"

namespace cs336 {

namespace {

constexpr char kMagic[8] = {'C', 'S', '3', '3', '6', 'L', 'M', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kDtypeF32 = 0;

std::string shape_to_string(const std::vector<std::uint32_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
  os << "]";
  return os.str();
}

template <typename T>
void check_equal(const char* what, T file_value, T compiled_value) {
  if (file_value != compiled_value) {
    std::ostringstream os;
    os << "model.bin was exported with " << what << "=" << file_value << " but this binary was compiled with "
       << what << "=" << compiled_value << " (see src/model_config.h)";
    throw std::runtime_error(os.str());
  }
}

}  // namespace

std::size_t TensorInfo::numel() const {
  std::size_t n = 1;
  for (std::uint32_t s : shape) n *= s;
  return n;
}

WeightsFile::WeightsFile(const std::string& path) : reader_(path) {
  if (!host_is_little_endian()) throw std::runtime_error("big-endian hosts are not supported");

  char magic[8];
  reader_.read_bytes(magic, sizeof magic);
  if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error(path + " is not a model.bin file");
  const std::uint32_t version = reader_.read_u32();
  if (version != kVersion) throw std::runtime_error("unsupported model.bin version " + std::to_string(version));
  const std::uint32_t dtype = reader_.read_u32();
  if (dtype != kDtypeF32) throw std::runtime_error("unsupported model.bin dtype " + std::to_string(dtype));

  check_equal<std::uint32_t>("vocab_size", reader_.read_u32(), VOCAB_SIZE);
  check_equal<std::uint32_t>("num_layers", reader_.read_u32(), NUM_LAYERS);
  check_equal<std::uint32_t>("d_model", reader_.read_u32(), D_MODEL);
  check_equal<std::uint32_t>("num_heads", reader_.read_u32(), NUM_HEADS);
  check_equal<std::uint32_t>("d_ff", reader_.read_u32(), D_FF_CONFIG);
  check_equal<std::uint32_t>("max_seq_len", reader_.read_u32(), MAX_SEQ_LEN);
  check_equal<std::uint32_t>("tie_embeddings", reader_.read_u32(), TIE_EMBEDDINGS ? 1u : 0u);
  const float eps = reader_.read_f32();
  if (std::fabs(eps - EPS) > 1e-3f * EPS) check_equal("eps", eps, EPS);
  const double theta = reader_.read_f64();
  if (std::fabs(theta - THETA) > 1e-9 * THETA) check_equal("theta", theta, THETA);

  const std::uint32_t count = reader_.read_u32();
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint16_t name_len = reader_.read_u16();
    const std::string name = reader_.read_string(name_len);
    TensorInfo info;
    const std::uint32_t ndim = reader_.read_u32();
    for (std::uint32_t d = 0; d < ndim; ++d) info.shape.push_back(reader_.read_u32());
    info.offset = reader_.read_u64();
    info.nbytes = reader_.read_u64();
    if (info.nbytes != info.numel() * sizeof(float)) throw std::runtime_error("corrupt tensor table entry: " + name);
    tensors_[name] = info;
  }
}

void WeightsFile::read_tensor(const std::string& name, const std::vector<std::uint32_t>& expected_shape, float* dst) {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("model.bin has no tensor " + name);
  if (it->second.shape != expected_shape) {
    throw std::runtime_error("tensor " + name + " has shape " + shape_to_string(it->second.shape) +
                             " but the model expects " + shape_to_string(expected_shape));
  }
  reader_.seek(it->second.offset);
  reader_.read_bytes(dst, static_cast<std::size_t>(it->second.nbytes));
}

}  // namespace cs336
