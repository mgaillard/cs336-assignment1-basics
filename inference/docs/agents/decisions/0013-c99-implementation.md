# 0013 - plain C99 with a small platform layer

Status: Accepted

**Context.** The app must have a very small, simple runtime so that it runs on Windows 98 SE and any
32-bit or 64-bit target without a language runtime library.

**Decision.** The whole app is plain C99 (`-std=c99 -pedantic`, `restrict`, `stdint.h`, no VLAs, no C11), one struct
plus `<module>_<method>` functions per Python class (see `architecture.md`).
- Errors are `int` returns (0 / -1, pointers: NULL) with a message stored by `cs_set_error()`; `main.c` prints it.
  There are no exceptions; `*_free` functions are safe after a failed `*_init`.
- Growable data uses a small byte buffer `CsBuf` (`common.h`); big float arrays come from `cs_alloc_floats()`
  (calloc + 64-byte alignment, portable on every target).
- The tokenizer finds tokens through an open-addressing hash table over slices of one byte pool (no allocation per
  lookup); weight tensors (~40) are found by a linear table scan.
- Callbacks are a function pointer plus `void *user`; compile-time checks use a typedef-based `CS_STATIC_ASSERT`.
- Everything OS specific is isolated in `platform.h` (`platform_win32.c`, `platform_posix.c`); the rest is ISO C.
  The 32-bit Linux build needs only `gcc-multilib`.

**Consequences.** Builds are warning-free with `-Wall -Wextra -pedantic` on all three targets and the 32-bit objects contain
no SSE2 instructions. Validation (`make validate`): token ids of 184 strings equal tiktoken, logits are within ~8e-6 of
PyTorch, greedy ids are identical on both Linux targets; decode speed is ~105 tok/s on a single thread (x86_64). ASan/UBSan
runs were clean. The static Windows exe is ~310 KB.
