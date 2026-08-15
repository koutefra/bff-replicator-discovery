CUDA ?= auto
CXX ?= g++
NVCC ?= nvcc
PYTHON ?= python3

BIN_DIR := bin
SRC_DIR := src

BROTLI_CFLAGS := $(shell pkg-config --cflags libbrotlienc libbrotlicommon)
BROTLI_LIBS := $(shell pkg-config --libs libbrotlienc libbrotlicommon)
COMMON_FLAGS := -std=c++17 -O3 $(BROTLI_CFLAGS)

ifeq ($(CUDA),auto)
  CUDA := $(if $(shell command -v $(NVCC) 2>/dev/null),1,0)
endif

ifeq ($(CUDA),1)
  BUILD_DIR := build/cuda
  COMPILER := $(NVCC)
  COMPILE_FLAGS := -x cu $(COMMON_FLAGS) -arch=sm_75 -Xcompiler=-Wall,-fPIC,-fopenmp
  LINK_FLAGS := -arch=sm_75 -Xcompiler=-fopenmp $(BROTLI_LIBS)
  REPLAY_BINS := $(BIN_DIR)/fixed_partner_replay $(BIN_DIR)/pre_replicator_availability_replay
else
  BUILD_DIR := build/cpu
  COMPILER := $(CXX)
  COMPILE_FLAGS := -x c++ $(COMMON_FLAGS) -Wall -fPIC -fopenmp
  LINK_FLAGS := -fopenmp $(BROTLI_LIBS)
  REPLAY_BINS :=
endif

CORE_OBJECTS := \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/common.o \
	$(BUILD_DIR)/bff_noheads.o

.PHONY: all simulator replays smoke figures validate website clean FORCE

all: simulator $(REPLAY_BINS)

simulator: $(BIN_DIR)/main

replays:
ifeq ($(CUDA),1)
	$(MAKE) $(REPLAY_BINS) CUDA=1
else
	@echo "Replay executables require CUDA/nvcc; build with CUDA=1." >&2
	@exit 2
endif

$(BIN_DIR)/main: $(CORE_OBJECTS) FORCE | $(BIN_DIR)
	$(COMPILER) $(CORE_OBJECTS) $(LINK_FLAGS) -o $@

$(BIN_DIR)/fixed_partner_replay: $(BUILD_DIR)/fixed_partner_replay.o $(BUILD_DIR)/common.o FORCE | $(BIN_DIR)
	$(NVCC) $(BUILD_DIR)/fixed_partner_replay.o $(BUILD_DIR)/common.o $(LINK_FLAGS) -o $@

$(BIN_DIR)/pre_replicator_availability_replay: $(BUILD_DIR)/pre_replicator_availability_replay.o $(BUILD_DIR)/common.o FORCE | $(BIN_DIR)
	$(NVCC) $(BUILD_DIR)/pre_replicator_availability_replay.o $(BUILD_DIR)/common.o $(LINK_FLAGS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc $(SRC_DIR)/common.h $(SRC_DIR)/common_language.h $(SRC_DIR)/bff.inc.h | $(BUILD_DIR)
	$(COMPILER) $(COMPILE_FLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR)/bff_noheads.o: $(SRC_DIR)/bff_noheads.cu $(SRC_DIR)/common.h $(SRC_DIR)/common_language.h $(SRC_DIR)/bff.inc.h | $(BUILD_DIR)
	$(COMPILER) $(COMPILE_FLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

smoke: simulator
	$(PYTHON) experiments/run_regimes.py --profile smoke --output runs/smoke --resume
	$(PYTHON) tests/validate_smoke.py --run-root runs/smoke

figures:
	$(PYTHON) analysis/make_figures.py

validate:
	$(PYTHON) tests/validate_paper_data.py

website:
	$(PYTHON) web/scripts/build_site.py

clean:
	rm -rf build $(BIN_DIR)


FORCE:
