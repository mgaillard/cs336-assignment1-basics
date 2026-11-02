#ifndef _WIN32

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "platform.h"

#if defined(__i386__) && defined(__SSE__)
#include <cpuid.h>
#endif

struct CsFile {
  FILE *fp;
  const char *path;
};

CsFile *cs_file_open_read(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    cs_set_error("cannot open file: %s", path);
    return NULL;
  }
  CsFile *file = (CsFile *)malloc(sizeof *file);
  if (!file) {
    fclose(fp);
    cs_set_error("out of memory");
    return NULL;
  }
  file->fp = fp;
  file->path = path;
  return file;
}

int cs_file_read(CsFile *file, void *dst, size_t n) {
  if (n != 0 && fread(dst, 1, n, file->fp) != n) {
    cs_set_error("unexpected end of file: %s", file->path);
    return -1;
  }
  return 0;
}

int cs_file_seek(CsFile *file, uint64_t offset) {
  if (fseeko(file->fp, (off_t)offset, SEEK_SET) != 0) {
    cs_set_error("seek failed: %s", file->path);
    return -1;
  }
  return 0;
}

void cs_file_close(CsFile *file) {
  if (!file) return;
  fclose(file->fp);
  free(file);
}

double cs_now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

unsigned cs_time_seed(void) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (unsigned)ts.tv_sec * 1000003u ^ (unsigned)ts.tv_nsec;
}

int cs_cpu_supported(void) {
#if defined(__i386__) && defined(__SSE__)
  unsigned eax, ebx, ecx, edx;
  return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (edx & (1u << 25));
#else
  return 1;
#endif
}

#else

typedef int cs_platform_posix_is_empty_here;  // keeps the translation unit non-empty (-pedantic)

#endif
