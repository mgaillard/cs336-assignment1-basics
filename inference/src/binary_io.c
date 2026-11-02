#include "binary_io.h"

#include <string.h>

int binary_reader_open(BinaryReader *reader, const char *path) {
  reader->failed = 0;
  reader->file = cs_file_open_read(path);
  return reader->file ? 0 : -1;
}

void binary_reader_close(BinaryReader *reader) {
  cs_file_close(reader->file);
  reader->file = NULL;
}

int binary_reader_ok(const BinaryReader *reader) { return !reader->failed; }

void binary_reader_read_bytes(BinaryReader *reader, void *dst, size_t n) {
  if (reader->failed) {
    memset(dst, 0, n);
    return;
  }
  if (cs_file_read(reader->file, dst, n) != 0) {
    reader->failed = 1;
    memset(dst, 0, n);
  }
}

uint16_t binary_reader_read_u16(BinaryReader *reader) {
  unsigned char b[2];
  binary_reader_read_bytes(reader, b, 2);
  return (uint16_t)(b[0] | (b[1] << 8));
}

uint32_t binary_reader_read_u32(BinaryReader *reader) {
  unsigned char b[4];
  binary_reader_read_bytes(reader, b, 4);
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

uint64_t binary_reader_read_u64(BinaryReader *reader) {
  const uint64_t lo = binary_reader_read_u32(reader);
  const uint64_t hi = binary_reader_read_u32(reader);
  return lo | (hi << 32);
}

float binary_reader_read_f32(BinaryReader *reader) {
  const uint32_t bits = binary_reader_read_u32(reader);
  float value;
  memcpy(&value, &bits, sizeof value);
  return value;
}

double binary_reader_read_f64(BinaryReader *reader) {
  const uint64_t bits = binary_reader_read_u64(reader);
  double value;
  memcpy(&value, &bits, sizeof value);
  return value;
}

int binary_reader_seek(BinaryReader *reader, uint64_t offset) {
  if (reader->failed) return -1;
  if (cs_file_seek(reader->file, offset) != 0) {
    reader->failed = 1;
    return -1;
  }
  return 0;
}

int host_is_little_endian(void) {
  const uint32_t one = 1;
  unsigned char first;
  memcpy(&first, &one, 1);
  return first == 1;
}
