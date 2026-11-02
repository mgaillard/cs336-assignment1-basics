#ifndef CS336_PLATFORM_H
#define CS336_PLATFORM_H

// The only OS dependent interface. Implemented by platform_win32.c (plain Win32 calls available since
// Windows 95) and platform_posix.c; each file is empty on the other OS.

#include <stdint.h>

#include "common.h"

typedef struct CsFile CsFile;

// Read-only file access with 64-bit offsets. NULL / -1 on failure (error message set).
CsFile *cs_file_open_read(const char *path);
int cs_file_read(CsFile *file, void *dst, size_t n);
int cs_file_seek(CsFile *file, uint64_t offset);
void cs_file_close(CsFile *file);

double cs_now_seconds(void);  // monotonic-ish clock for timing
unsigned cs_time_seed(void);  // different on every run

// 1 if the CPU can run the SSE code this binary was compiled with (always 1 where SSE is not the
// baseline), else 0. Must be called before any floating point work on 32-bit builds.
int cs_cpu_supported(void);

#endif
