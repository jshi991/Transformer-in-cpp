#ifndef MYTORCH_GPU_POOL_H
#define MYTORCH_GPU_POOL_H

#include <cstddef>

namespace mytorch {

// Caching allocator for device memory: free() returns the block to a
// size-bucketed free list instead of calling cudaFree, so the
// alloc-a-fresh-buffer-per-op style used throughout this codebase doesn't
// pay cudaMalloc/cudaFree's driver-level cost on every op.
void* pool_alloc(size_t bytes);
void pool_free(void* ptr);

}  // namespace mytorch
#endif  // MYTORCH_GPU_POOL_H
