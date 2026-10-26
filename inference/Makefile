# C++17 inference app for the cs336_basics TransformerLM.
#
#   make              build every target
#   make x86_sse      32-bit, single core, SSE only (no SSE2)      -> build/x86_sse/cs336_infer
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
TARGETS  := x86_sse x64_avx2

COMMON_FLAGS := -std=c++17 -O3 -Wall -Wextra -ffp-contract=fast -fno-math-errno -D_FILE_OFFSET_BITS=64

# 32-bit, SSE1 only (Pentium III class). -fopenmp-simd enables `#pragma omp simd` but no threads.
# Without root access to install g++-multilib, extract lib32stdc++-13-dev into a directory D and pass
#   M32_CXXFLAGS="-isystem D/usr/include/x86_64-linux-gnu/c++/13/32"
#   M32_LDFLAGS="-L D/usr/lib/gcc/x86_64-linux-gnu/13/32 -static-libstdc++"
M32_CXXFLAGS ?=
M32_LDFLAGS ?=
x86_sse_FLAGS  := -m32 -march=pentium3 -msse -mno-sse2 -mfpmath=sse,387 -fopenmp-simd $(M32_CXXFLAGS)
x86_sse_LDFLAGS := -m32 $(M32_LDFLAGS)

# 64-bit, SSE4.2 + AVX2 + FMA, OpenMP threads.
x64_avx2_FLAGS := -m64 -msse4.2 -mavx2 -mfma -fopenmp
x64_avx2_LDFLAGS := -m64 -fopenmp

.PHONY: all $(TARGETS) clean convert validate vecreport

all: $(TARGETS)

define TARGET_RULES
$(1): $(BUILD)/$(1)/cs336_infer

$(BUILD)/$(1)/cs336_infer: $(patsubst $(SRC_DIR)/%.cpp,$(BUILD)/$(1)/%.o,$(SOURCES))
	$$(CXX) $$($(1)_LDFLAGS) -o $$@ $$^

$(BUILD)/$(1)/%.o: $(SRC_DIR)/%.cpp | $(BUILD)/$(1)
	$$(CXX) $$(COMMON_FLAGS) $$($(1)_FLAGS) -MMD -MP -fopt-info-vec-optimized=$$(BUILD)/$(1)/$$*.vec.log -c $$< -o $$@

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
