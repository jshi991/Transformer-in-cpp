CXXFLAGS := -std=c++17 -Wall -Wextra -Iinclude

SRC_DIR := src
BUILD_DIR := build
TARGET := $(BUILD_DIR)/transformer

SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

UNAME_S := $(shell uname -s)

.PHONY: all linux macos clean

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

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
