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

.PHONY: all linux macos cuda clean

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

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CUDA_TARGET): $(CUDA_OBJS)
	$(NVCC) -o $@ $^

$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
