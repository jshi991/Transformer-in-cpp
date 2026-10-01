CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -I.

SRC_DIR := src
BUILD_DIR := build
TARGET := $(BUILD_DIR)/transformer

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

NVCC := nvcc
NVCCFLAGS := -std=c++17 -O2 -Iinclude -I.
CUDA_TARGET := $(BUILD_DIR)/transformer_cuda
# train.cu, quantize_main.cu, infer_main.cu each have their own main().
CUDA_SRCS := $(filter-out src/train.cu src/quantize/quantize_main.cu src/quantize/infer_main.cu,$(shell find $(SRC_DIR) database -name '*.cu' 2>/dev/null))
CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(CUDA_SRCS))

UNAME_S := $(shell uname -s)

BENCH_MATMUL_SRCS := benchmarks/matmul/naive_vs_cublas.cu src/mytorch/jensor.cu
BENCH_MATMUL_TARGET := $(BUILD_DIR)/bench_matmul

TRAIN_TARGET := $(BUILD_DIR)/train
TRAIN_SRCS := $(filter-out src/main.cu,$(CUDA_SRCS)) src/train.cu
TRAIN_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(TRAIN_SRCS))

QUANTIZE_LIB_SRCS := $(filter-out src/main.cu,$(CUDA_SRCS))

QUANTIZE_TARGET := $(BUILD_DIR)/quantize_awq
QUANTIZE_SRCS := $(QUANTIZE_LIB_SRCS) src/quantize/quantize_main.cu
QUANTIZE_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(QUANTIZE_SRCS))

INFER_TARGET := $(BUILD_DIR)/infer
INFER_SRCS := $(QUANTIZE_LIB_SRCS) src/quantize/infer_main.cu
INFER_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(INFER_SRCS))

CORPUS := src/database/dataset/wikitext-2-raw-train.txt
CHECKPOINT := build/checkpoint.bin
CHECKPOINT_AWQ := build/checkpoint_awq.bin

.PHONY: all linux macos cuda bench-matmul train setup-model quantize infer clean

all:
ifeq ($(UNAME_S),Darwin)
	$(MAKE) macos
else
	$(MAKE) linux
endif

linux: CXX := g++
linux: $(TARGET)

macos: CXX := clang++
macos: $(TARGET)

cuda: $(CUDA_TARGET)

bench-matmul: $(BENCH_MATMUL_TARGET)

train: $(TRAIN_TARGET)

# End-to-end AWQ experiment:
#   make setup-model  -> trains the 5-layer decoder-only transformer, saves $(CHECKPOINT)
#   make quantize     -> AWQ-quantizes it (int8 by default), saves $(CHECKPOINT_AWQ),
#                        prints per-layer error + before/after loss
#   make infer        -> generates text from the quantized checkpoint
setup-model: $(TRAIN_TARGET)
	./$(TRAIN_TARGET) $(CORPUS) 1000 4 $(CHECKPOINT)

quantize: $(QUANTIZE_TARGET)
	./$(QUANTIZE_TARGET) $(CORPUS) $(CHECKPOINT) $(CHECKPOINT_AWQ) 8 64

infer: $(INFER_TARGET)
	./$(INFER_TARGET) $(CORPUS) $(CHECKPOINT_AWQ)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -lcublas: jensor.cu's Backend::CuBLAS path calls into cuBLAS (agents.md/02).
# -lcurand: embeddings.cu seeds weights directly on the GPU via cuRAND.
$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) -o $@ $^ -lcublas -lcurand

$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BENCH_MATMUL_TARGET): $(BENCH_MATMUL_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lcublas

$(TRAIN_TARGET): $(TRAIN_OBJS)
	$(NVCC) -o $@ $^ -lcublas -lcurand

$(QUANTIZE_TARGET): $(QUANTIZE_OBJS)
	$(NVCC) -o $@ $^ -lcublas -lcurand

$(INFER_TARGET): $(INFER_OBJS)
	$(NVCC) -o $@ $^ -lcublas -lcurand

clean:
	rm -rf $(BUILD_DIR)
