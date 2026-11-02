#ifndef CS336_BINARY_IO_H
#define CS336_BINARY_IO_H

#include <stdint.h>

#include "common.h"
#include "platform.h"

// Little-endian reader. Errors are sticky: after the first failure (message set) every read returns 0
// and binary_reader_ok() is false, so callers can parse a whole header and check once.
typedef struct {
  CsFile *file;
  int failed;
} BinaryReader;

int binary_reader_open(BinaryReader *reader, const char *path);
void binary_reader_close(BinaryReader *reader);
int binary_reader_ok(const BinaryReader *reader);

void binary_reader_read_bytes(BinaryReader *reader, void *dst, size_t n);
uint16_t binary_reader_read_u16(BinaryReader *reader);
uint32_t binary_reader_read_u32(BinaryReader *reader);
uint64_t binary_reader_read_u64(BinaryReader *reader);
float binary_reader_read_f32(BinaryReader *reader);
double binary_reader_read_f64(BinaryReader *reader);
int binary_reader_seek(BinaryReader *reader, uint64_t offset);

int host_is_little_endian(void);

#endif
