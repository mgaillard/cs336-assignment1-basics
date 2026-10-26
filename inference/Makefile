# C++17 inference app for the cs336_basics TransformerLM.
#
#   make              build every target
#   make x86_sse      32-bit, single core, SSE only (no SSE2)      -> build/x86_sse/cs336_infer
#   make x86_sse_win 32-bit Windows (MinGW-w64), single core, SSE only -> build/x86_sse_win/cs336_infer.exe
#   make x64_avx2     64-bit, multi core (OpenMP), SSE4.2 + AVX2   -> build/x64_avx2/cs336_infer
#   make vecreport    summarize the loops the compiler vectorized
#   make convert      create models/model.bin and models/tokenizer.bin from the checkpoint
#   make validate     compare the binaries with PyTorch / tiktoken
#   make clean
#
# SIMD comes from the compiler's auto-vectorizer (no intrinsics): -O3 plus the target -m flags.

CXX      ?= g++
PYTHON   ?= ../.venv/bin/python
CHECKPOINT ?= ../outputs/2026-10-25/17-13-45/checkpoint_best_model.pt
CONFIG   ?= ../configs/gpt_small.yaml

SRC_DIR  := src
BUILD    := build
SOURCES  := $(wildcard $(SRC_DIR)/*.cpp)
TARGETS  := x86_sse x86_sse_win x64_avx2

COMMON_FLAGS := -std=c++17 -O3 -Wall -Wextra -ffp-contract=fast -fno-math-errno -D_FILE_OFFSET_BITS=64

# 32-bit, SSE1 only (Pentium III class). -fopenmp-simd enables `#pragma omp simd` but no threads.
# Without root access to install g++-multilib, extract lib32stdc++-13-dev into a directory D and pass
#   M32_CXXFLAGS="-isystem D/usr/include/x86_64-linux-gnu/c++/13/32"
#   M32_LDFLAGS="-L D/usr/lib/gcc/x86_64-linux-gnu/13/32 -static-libstdc++"
M32_CXXFLAGS ?=
M32_LDFLAGS ?=
x86_sse_FLAGS  := -m32 -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd $(M32_CXXFLAGS)
x86_sse_LDFLAGS := -m32 $(M32_LDFLAGS)

# Windows 32-bit, same ISA flags as x86_sse, cross-compiled with MinGW-w64 (posix or win32 threads
# model is irrelevant: no threads are used). Fully static so the .exe needs no MinGW DLLs.
# Without root, extract g++-mingw-w64-i686-win32 into a directory and point WIN32_CXX at its driver,
# adding -B/-L flags through WIN32_CXXFLAGS / WIN32_LDFLAGS.
WIN32_CXX ?= i686-w64-mingw32-g++
WIN32_CXXFLAGS ?=
WIN32_LDFLAGS ?=
x86_sse_win_CXX := $(WIN32_CXX)
x86_sse_win_EXT := .exe
x86_sse_win_FLAGS := -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd $(WIN32_CXXFLAGS)
x86_sse_win_LDFLAGS := -static $(WIN32_LDFLAGS)

# 64-bit, SSE4.2 + AVX2 + FMA, OpenMP threads.
x64_avx2_FLAGS := -m64 -msse4.2 -mavx2 -mfma -fopenmp
x64_avx2_LDFLAGS := -m64 -fopenmp

.PHONY: all $(TARGETS) clean convert validate vecreport

all: $(TARGETS)

define TARGET_RULES
$(1)_CXX ?= $$(CXX)
$(1)_EXT ?=

$(1): $(BUILD)/$(1)/cs336_infer$$($(1)_EXT)

$(BUILD)/$(1)/cs336_infer$$($(1)_EXT): $(patsubst $(SRC_DIR)/%.cpp,$(BUILD)/$(1)/%.o,$(SOURCES))
	$$($(1)_CXX) $$($(1)_LDFLAGS) -o $$@ $$^

$(BUILD)/$(1)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)/$(1)
	$$($(1)_CXX) $$(COMMON_FLAGS) $$($(1)_FLAGS) -MMD -MP -fopt-info-vec-optimized=$$(BUILD)/$(1)/$$*.vec.log -c $$< -o $$@

$(BUILD)/$(1):
	mkdir -p $$@
endef
$(foreach t,$(TARGETS),$(eval $(call TARGET_RULES,$(t))))

-include $(foreach t,$(TARGETS),$(wildcard $(BUILD)/$(t)/*.d))

vecreport: all
	@for t in $(TARGETS); do \
	  echo "== $$t: vectorized loops per source file"; \
	  for f in attention linear rmsnorm sampler rope; do \
	    printf '  %-28s %s\n' $$f.cpp "$$(grep -c 'optimized: loop vectorized' $(BUILD)/$$t/$$f.vec.log 2>/dev/null)"; \
	  done; \
	  echo "  loops in kernels.h vectorized:"; \
	  cat $(BUILD)/$$t/*.vec.log | grep 'kernels.h' | grep 'loop vectorized' | sed 's/^/    /' | sort -u; \
	done

convert:
	mkdir -p models
	$(PYTHON) convert_checkpoint.py $(CHECKPOINT) models/model.bin --config $(CONFIG)
	$(PYTHON) export_tokenizer.py models/tokenizer.bin

validate: all
	$(PYTHON) validate.py --checkpoint $(CHECKPOINT) --config $(CONFIG)

clean:
	rm -rf $(BUILD)
