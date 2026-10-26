# 0012 - Windows 32-bit SSE target

Status: Accepted

**Context.** The Pentium III-class, SSE-only, single-core build (ADR 0003) is also needed on Windows.

**Decision.** Add Makefile target `x86_sse_win`, producing `build/x86_sse_win/cs336_infer.exe`. It reuses the
ISA flags of `x86_sse` (`-march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd`, no `-m32` needed as
the compiler is already i686) and is cross-compiled with MinGW-w64 (`WIN32_CXX`, default
`i686-w64-mingw32-g++`). It links with `-static` so no MinGW runtime DLLs are required, and no OpenMP runtime is
linked, so it stays single-threaded. The Makefile gives each target its own compiler (`<target>_CXX`) and
executable suffix (`<target>_EXT`). `main.cpp` calls `SetConsoleOutputCP(CP_UTF8)` under `_WIN32` because
the app prints UTF-8 bytes. Source code needed no other change: `fseeko`/`off_t` come from MinGW with
`-D_FILE_OFFSET_BITS=64`, and `AlignedBuffer` does not use platform allocators.

**Consequences.** `make all` now needs a MinGW-w64 C++ cross-compiler; build one target alone with
`make x86_sse` or `make x64_avx2` otherwise. `WIN32_CXXFLAGS`/`WIN32_LDFLAGS` allow locally extracted toolchains.
The exe compiles warning-free and its objects contain no SSE2+ instructions, but it has not been run
(no Wine/Windows here) and `validate.py` does not cover it. Command-line arguments use the ANSI code
page on Windows, so non-ASCII prompts may be mangled; reading UTF-8 from a file would be the fix.
