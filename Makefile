CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude -I.

SRC_DIR := src
BUILD_DIR := build
TARGET := $(BUILD_DIR)/transformer

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

NVCC := nvcc
NVCCFLAGS := -std=c++17 -Iinclude -I.
CUDA_TARGET := $(BUILD_DIR)/transformer_cuda
CUDA_SRCS := $(shell find $(SRC_DIR) database -name '*.cu' 2>/dev/null)
CUDA_OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.cu.o,$(CUDA_SRCS))

UNAME_S := $(shell uname -s)

BENCH_MATMUL_SRCS := benchmarks/matmul/naive_vs_cublas.cu src/mytorch/jensor.cu
BENCH_MATMUL_TARGET := $(BUILD_DIR)/bench_matmul

.PHONY: all linux macos cuda bench-matmul clean

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

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -lcublas: jensor.cu's Backend::CuBLAS path calls into cuBLAS (agents.md/02).
$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) -o $@ $^ -lcublas

$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BENCH_MATMUL_TARGET): $(BENCH_MATMUL_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lcublas

clean:
	rm -rf $(BUILD_DIR)
