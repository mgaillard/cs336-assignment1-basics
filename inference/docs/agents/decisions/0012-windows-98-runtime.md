# 0012 - Windows 98 SE runtime

Status: Accepted

**Context.** The 32-bit Windows build (Pentium III-class, SSE-only, single core, see ADR 0003) must run on Windows 98 SE,
not only on modern Windows. Win98 has no UTF-8 console, an ANSI `argv`, a 4-byte aligned stack, a `kernel32.dll`
without the XP+ functions and an `msvcrt.dll` without 64-bit file seeks.

**Decision.** Makefile target `x86_sse_win` produces `build/x86_sse_win/cs336_infer.exe`: a plain MinGW-w64 C99 build
(`WIN32_CC`, default `i686-w64-mingw32-gcc`) with the normal CRT start-up (`main`), linked `-static` so no MinGW runtime
DLLs are needed and without an OpenMP runtime, so it is single-threaded. The Makefile gives each target its own compiler
(`<target>_CC`) and executable suffix (`<target>_EXT`). `-nostartfiles` with a custom entry point is only a fallback if
the CRT start-up ever needs to be avoided.
- Flags: ISA of ADR 0003, `-mstackrealign` (GCC assumes a 16-byte aligned stack for SSE spills, Win9x gives 4),
  `-DWINVER=0x0400 -D_WIN32_WINNT=0x0400`, PE OS and subsystem version 4.0.
- `platform_win32.c` uses only Win95-era `kernel32` calls: `CreateFileA`, `ReadFile` (16 MB chunks), `SetFilePointer`
  with a high dword (64-bit offsets without CRT help), `QueryPerformanceCounter` with a `GetTickCount` fallback,
  `GetTickCount`/`GetCurrentProcessId` for the seed.
- `main()` first calls `cs_cpu_supported()` (CPUID SSE bit on 32-bit SSE builds) and exits with a message on a CPU without SSE.
- `tools/check_imports.py` (`make check_win98`) parses the PE file and fails unless it is i386, console, OS/subsystem
  version <= 4.0 and imports only allow-listed functions of `kernel32.dll` and `msvcrt.dll`. The allow-list is the reviewed
  set the current exe uses; a new import has to be checked against Win98 SE before being added. `GetModuleHandleW` (a stub
  on Win9x, called by the MinGW CRT) is reported as a warning.
- The console and `argv` are not UTF-8 on Win98, so the app does not touch the console code page and `--prompt-file` /
  `--output-file` read and write raw UTF-8 bytes (a UTF-8 BOM in the prompt file is skipped).

**Consequences.** The audit passes (26 kernel32 + 48 msvcrt imports, version 4.0, no SSE2 instructions) and the exe has
been run successfully on Windows 98 SE; `validate.py` does not cover it. `make all` needs a MinGW-w64 C cross-compiler; build
one target alone with `make x86_sse` or `make x64_avx2` otherwise. `WIN32_CFLAGS`/`WIN32_LDFLAGS` allow locally extracted
toolchains. Command-line arguments use the ANSI code page, so non-ASCII prompts should come from `--prompt-file`.
Checklist for re-testing on a Win98 SE machine / 86Box / QEMU (`-cpu pentium3`) with >= 256 MB RAM (the model is 153 MB):
1. `cs336_infer.exe --print-tokens --prompt "Once upon a time"` prints `7454 2402 257 640` (model not needed).
2. `cs336_infer.exe --prompt-file p.txt --output-file out.txt --max_steps 8` with `models\` next to it.
3. Greedy ids equal the Linux ones (`--show-ids`).
If it fails to start after a change, check the loader error for the missing DLL function first (`make check_win98`).
