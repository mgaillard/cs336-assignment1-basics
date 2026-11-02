#ifdef _WIN32

// Only kernel32 functions that exist on Windows 95/98 are used (not the CRT 64-bit seek functions).

#include <stdlib.h>
#include <windows.h>

#include "platform.h"

#if defined(__i386__) && defined(__SSE__)
#include <cpuid.h>
#endif

struct CsFile {
  HANDLE handle;
  const char *path;
};

CsFile *cs_file_open_read(const char *path) {
  HANDLE handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (handle == INVALID_HANDLE_VALUE) {
    cs_set_error("cannot open file: %s", path);
    return NULL;
  }
  CsFile *file = (CsFile *)malloc(sizeof *file);
  if (!file) {
    CloseHandle(handle);
    cs_set_error("out of memory");
    return NULL;
  }
  file->handle = handle;
  file->path = path;
  return file;
}

int cs_file_read(CsFile *file, void *dst, size_t n) {
  unsigned char *out = (unsigned char *)dst;
  while (n > 0) {
    const DWORD chunk = n > (size_t)(16u << 20) ? (DWORD)(16u << 20) : (DWORD)n;
    DWORD got = 0;
    if (!ReadFile(file->handle, out, chunk, &got, NULL) || got != chunk) {
      cs_set_error("unexpected end of file: %s", file->path);
      return -1;
    }
    out += chunk;
    n -= chunk;
  }
  return 0;
}

int cs_file_seek(CsFile *file, uint64_t offset) {
  LONG high = (LONG)(offset >> 32);
  const DWORD low = SetFilePointer(file->handle, (LONG)(offset & 0xFFFFFFFFu), &high, FILE_BEGIN);
  if (low == INVALID_SET_FILE_POINTER && GetLastError() != NO_ERROR) {
    cs_set_error("seek failed: %s", file->path);
    return -1;
  }
  return 0;
}

void cs_file_close(CsFile *file) {
  if (!file) return;
  CloseHandle(file->handle);
  free(file);
}

double cs_now_seconds(void) {
  LARGE_INTEGER freq, now;
  if (QueryPerformanceFrequency(&freq) && freq.QuadPart > 0 && QueryPerformanceCounter(&now)) {
    return (double)now.QuadPart / (double)freq.QuadPart;
  }
  return (double)GetTickCount() / 1000.0;
}

unsigned cs_time_seed(void) { return (unsigned)GetTickCount() * 2654435761u ^ (unsigned)GetCurrentProcessId(); }

int cs_cpu_supported(void) {
#if defined(__i386__) && defined(__SSE__)
  unsigned eax, ebx, ecx, edx;
  return __get_cpuid(1, &eax, &ebx, &ecx, &edx) && (edx & (1u << 25));
#else
  return 1;
#endif
}

#else

typedef int cs_platform_win32_is_empty_here;  // keeps the translation unit non-empty (-pedantic)

#endif
