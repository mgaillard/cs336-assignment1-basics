# C99 inference app for the cs336_basics TransformerLM.
#
#   make              build every target
#   make x86_sse      32-bit, single core, SSE only (no SSE2)      -> build/x86_sse/cs336_infer
#   make x86_sse_win  32-bit Windows 98 SE+ (MinGW-w64), single core, SSE only -> build/x86_sse_win/cs336_infer.exe
#   make x64_avx2     64-bit, multi core (OpenMP), SSE4.2 + AVX2   -> build/x64_avx2/cs336_infer
#   make vecreport    summarize the loops the compiler vectorized
#   make check_win98  list the DLL imports of the Windows exe and check them against the Win98 SE allow-list
#   make convert      create models/model.bin and models/tokenizer.bin from the checkpoint
#   make validate     compare the binaries with PyTorch / tiktoken
#   make clean
#
# SIMD comes from the compiler's auto-vectorizer (no intrinsics): -O3 plus the target -m flags.

CC       ?= gcc
PYTHON   ?= ../.venv/bin/python
CHECKPOINT ?= ../outputs/2026-10-25/17-13-45/checkpoint_best_model.pt
CONFIG   ?= ../configs/gpt_small.yaml

SRC_DIR  := src
BUILD    := build
SOURCES  := $(wildcard $(SRC_DIR)/*.c)
TARGETS  := x86_sse x86_sse_win x64_avx2

COMMON_FLAGS := -std=c99 -O3 -Wall -Wextra -pedantic -ffp-contract=fast -fno-math-errno -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64

# 32-bit, SSE1 only (Pentium III class). -fopenmp-simd enables `#pragma omp simd` but no threads.
# Needs only the 32-bit C runtime (gcc-multilib / libc6-dev-i386).
x86_sse_FLAGS  := -m32 -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd
x86_sse_LDFLAGS := -m32 -lm

# Windows 98 SE and later, same ISA flags as x86_sse, cross-compiled with MinGW-w64 as a plain C99 program
# (normal CRT start-up, msvcrt.dll). Win98 specifics: PE OS/subsystem version 4.0, API level 0x0400, and
# -mstackrealign because Windows 9x only guarantees a 4-byte aligned stack while GCC assumes 16 for SSE spills.
# Fully static so the .exe needs no MinGW DLLs. Without root, point WIN32_CC at an extracted toolchain.
WIN32_CC ?= i686-w64-mingw32-gcc
WIN32_CFLAGS ?=
WIN32_LDFLAGS ?=
x86_sse_win_CC := $(WIN32_CC)
x86_sse_win_EXT := .exe
x86_sse_win_FLAGS := -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd -mstackrealign \
                     -DWINVER=0x0400 -D_WIN32_WINNT=0x0400 -fno-asynchronous-unwind-tables -fno-ident $(WIN32_CFLAGS)
x86_sse_win_LDFLAGS := -static -Wl,--major-os-version=4 -Wl,--minor-os-version=0 \
                       -Wl,--major-subsystem-version=4 -Wl,--minor-subsystem-version=0 $(WIN32_LDFLAGS)

# 64-bit, SSE4.2 + AVX2 + FMA, OpenMP threads.
x64_avx2_FLAGS := -m64 -msse4.2 -mavx2 -mfma -fopenmp
x64_avx2_LDFLAGS := -m64 -fopenmp -lm

.PHONY: all $(TARGETS) clean convert validate vecreport check_win98

all: $(TARGETS)

define TARGET_RULES
$(1)_CC ?= $$(CC)
$(1)_EXT ?=

$(1): $(BUILD)/$(1)/cs336_infer$$($(1)_EXT)

$(BUILD)/$(1)/cs336_infer$$($(1)_EXT): $(patsubst $(SRC_DIR)/%.c,$(BUILD)/$(1)/%.o,$(SOURCES))
	$$($(1)_CC) -o $$@ $$^ $$($(1)_LDFLAGS)

$(BUILD)/$(1)/%.o: $(SRC_DIR)/%.c | $(BUILD)/$(1)
	$$($(1)_CC) $$(COMMON_FLAGS) $$($(1)_FLAGS) -MMD -MP -fopt-info-vec-optimized=$$(BUILD)/$(1)/$$*.vec.log -c $$< -o $$@

$(BUILD)/$(1):
	mkdir -p $$@
endef
$(foreach t,$(TARGETS),$(eval $(call TARGET_RULES,$(t))))

-include $(foreach t,$(TARGETS),$(wildcard $(BUILD)/$(t)/*.d))

vecreport: all
	@for t in $(TARGETS); do \
	  echo "== $$t: vectorized loops per source file"; \
	  for f in attention linear rmsnorm sampler rope; do \
	    printf '  %-28s %s\n' $$f.c "$$(grep -c 'optimized: loop vectorized' $(BUILD)/$$t/$$f.vec.log 2>/dev/null)"; \
	  done; \
	  echo "  loops in kernels.h vectorized:"; \
	  cat $(BUILD)/$$t/*.vec.log | grep 'kernels.h' | grep 'loop vectorized' | sed 's/^/    /' | sort -u; \
	done

check_win98: x86_sse_win
	$(PYTHON) tools/check_imports.py $(BUILD)/x86_sse_win/cs336_infer.exe

convert:
	mkdir -p models
	$(PYTHON) convert_checkpoint.py $(CHECKPOINT) models/model.bin --config $(CONFIG)
	$(PYTHON) export_tokenizer.py models/tokenizer.bin

validate: all
	$(PYTHON) validate.py --checkpoint $(CHECKPOINT) --config $(CONFIG)

clean:
	rm -rf $(BUILD)
