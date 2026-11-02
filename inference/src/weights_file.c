#include "weights_file.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model_config.h"

#define MAX_TENSORS 100000u

static const char kMagic[8] = {'C', 'S', '3', '3', '6', 'L', 'M', '\0'};
static const uint32_t kVersion = 1;
static const uint32_t kDtypeF32 = 0;

static int check_u32(const char *what, uint32_t file_value, uint32_t compiled_value) {
  if (file_value == compiled_value) return 0;
  cs_set_error("model.bin was exported with %s=%lu but this binary was compiled with %s=%lu (see src/model_config.h)",
               what, (unsigned long)file_value, what, (unsigned long)compiled_value);
  return -1;
}

static void format_shape(char *out, size_t size, const uint32_t *shape, uint32_t ndim) {
  size_t pos = 0;
  pos += (size_t)snprintf(out + pos, size - pos, "[");
  for (uint32_t i = 0; i < ndim && pos < size; ++i) {
    pos += (size_t)snprintf(out + pos, size - pos, "%s%lu", i ? ", " : "", (unsigned long)shape[i]);
  }
  if (pos < size) snprintf(out + pos, size - pos, "]");
}

void weights_file_close(WeightsFile *file) {
  if (file->tensors) {
    for (uint32_t i = 0; i < file->count; ++i) free(file->tensors[i].name);
    free(file->tensors);
  }
  file->tensors = NULL;
  file->count = 0;
  binary_reader_close(&file->reader);
}

int weights_file_open(WeightsFile *file, const char *path) {
  memset(file, 0, sizeof *file);
  if (!host_is_little_endian()) {
    cs_set_error("big-endian hosts are not supported");
    return -1;
  }
  BinaryReader *r = &file->reader;
  if (binary_reader_open(r, path) != 0) return -1;

  char magic[8];
  binary_reader_read_bytes(r, magic, sizeof magic);
  if (!binary_reader_ok(r)) goto fail;
  if (memcmp(magic, kMagic, sizeof magic) != 0) {
    cs_set_error("%s is not a model.bin file", path);
    goto fail;
  }
  const uint32_t version = binary_reader_read_u32(r);
  if (binary_reader_ok(r) && version != kVersion) {
    cs_set_error("unsupported model.bin version %lu", (unsigned long)version);
    goto fail;
  }
  const uint32_t dtype = binary_reader_read_u32(r);
  if (binary_reader_ok(r) && dtype != kDtypeF32) {
    cs_set_error("unsupported model.bin dtype %lu", (unsigned long)dtype);
    goto fail;
  }

  const uint32_t vocab_size = binary_reader_read_u32(r);
  const uint32_t num_layers = binary_reader_read_u32(r);
  const uint32_t d_model = binary_reader_read_u32(r);
  const uint32_t num_heads = binary_reader_read_u32(r);
  const uint32_t d_ff = binary_reader_read_u32(r);
  const uint32_t max_seq_len = binary_reader_read_u32(r);
  const uint32_t tie_embeddings = binary_reader_read_u32(r);
  const float eps = binary_reader_read_f32(r);
  const double theta = binary_reader_read_f64(r);
  const uint32_t count = binary_reader_read_u32(r);
  if (!binary_reader_ok(r)) goto fail;

  if (check_u32("vocab_size", vocab_size, VOCAB_SIZE) || check_u32("num_layers", num_layers, NUM_LAYERS) ||
      check_u32("d_model", d_model, D_MODEL) || check_u32("num_heads", num_heads, NUM_HEADS) ||
      check_u32("d_ff", d_ff, D_FF_CONFIG) || check_u32("max_seq_len", max_seq_len, MAX_SEQ_LEN) ||
      check_u32("tie_embeddings", tie_embeddings, TIE_EMBEDDINGS ? 1u : 0u)) {
    goto fail;
  }
  if (fabsf(eps - EPS) > 1e-3f * EPS) {
    cs_set_error("model.bin was exported with eps=%g but this binary was compiled with eps=%g (see src/model_config.h)",
                 (double)eps, (double)EPS);
    goto fail;
  }
  if (fabs(theta - THETA) > 1e-9 * THETA) {
    cs_set_error("model.bin was exported with theta=%g but this binary was compiled with theta=%g (see src/model_config.h)",
                 theta, THETA);
    goto fail;
  }
  if (count > MAX_TENSORS) {
    cs_set_error("corrupt tensor table (%lu tensors)", (unsigned long)count);
    goto fail;
  }

  file->tensors = (TensorInfo *)calloc(count ? count : 1, sizeof(TensorInfo));
  if (!file->tensors) {
    cs_set_error("out of memory");
    goto fail;
  }
  for (uint32_t i = 0; i < count; ++i) {
    TensorInfo *info = &file->tensors[i];
    const uint16_t name_len = binary_reader_read_u16(r);
    info->name = (char *)malloc((size_t)name_len + 1);
    if (!info->name) {
      cs_set_error("out of memory");
      file->count = i;
      goto fail;
    }
    file->count = i + 1;  // from here weights_file_close() releases the name
    binary_reader_read_bytes(r, info->name, name_len);
    info->name[name_len] = '\0';
    info->ndim = binary_reader_read_u32(r);
    if (binary_reader_ok(r) && info->ndim > WEIGHTS_MAX_NDIM) {
      cs_set_error("corrupt tensor table entry: %s", info->name);
      goto fail;
    }
    uint64_t numel = 1;
    for (uint32_t d = 0; d < info->ndim; ++d) {
      info->shape[d] = binary_reader_read_u32(r);
      numel *= info->shape[d];
    }
    info->offset = binary_reader_read_u64(r);
    info->nbytes = binary_reader_read_u64(r);
    if (!binary_reader_ok(r)) goto fail;
    if (info->nbytes != numel * sizeof(float)) {
      cs_set_error("corrupt tensor table entry: %s", info->name);
      goto fail;
    }
  }
  return 0;

fail:
  weights_file_close(file);
  return -1;
}

int weights_file_read_tensor(WeightsFile *file, const char *name, const uint32_t *expected_shape, int ndim, float *dst) {
  const TensorInfo *info = NULL;
  for (uint32_t i = 0; i < file->count; ++i) {
    if (strcmp(file->tensors[i].name, name) == 0) {
      info = &file->tensors[i];
      break;
    }
  }
  if (!info) {
    cs_set_error("model.bin has no tensor %s", name);
    return -1;
  }
  int same = info->ndim == (uint32_t)ndim;
  for (int d = 0; same && d < ndim; ++d) same = info->shape[d] == expected_shape[d];
  if (!same) {
    char have[96], want[96];
    format_shape(have, sizeof have, info->shape, info->ndim);
    format_shape(want, sizeof want, expected_shape, (uint32_t)ndim);
    cs_set_error("tensor %s has shape %s but the model expects %s", name, have, want);
    return -1;
  }
  if (binary_reader_seek(&file->reader, info->offset) != 0) return -1;
  binary_reader_read_bytes(&file->reader, dst, (size_t)info->nbytes);
  return binary_reader_ok(&file->reader) ? 0 : -1;
}
