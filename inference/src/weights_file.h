#ifndef CS336_WEIGHTS_FILE_H
#define CS336_WEIGHTS_FILE_H

#include <stdint.h>

#include "binary_io.h"

#define WEIGHTS_MAX_NDIM 4

typedef struct {
  char *name;
  uint32_t ndim;
  uint32_t shape[WEIGHTS_MAX_NDIM];
  uint64_t offset;
  uint64_t nbytes;
} TensorInfo;

// Reader of model.bin (written by convert_checkpoint.py). Opening the file parses the header and
// the tensor table and checks the stored architecture against the compile-time constants of
// model_config.h; tensors are then read on demand straight into the destination buffers.
typedef struct {
  BinaryReader reader;
  TensorInfo *tensors;
  uint32_t count;
} WeightsFile;

int weights_file_open(WeightsFile *file, const char *path);
void weights_file_close(WeightsFile *file);

// Reads tensor `name` (float32) into `dst` after checking that its shape is `expected_shape`.
int weights_file_read_tensor(WeightsFile *file, const char *name, const uint32_t *expected_shape, int ndim, float *dst);

#endif
