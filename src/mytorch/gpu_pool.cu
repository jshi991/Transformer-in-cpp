#include "mytorch/gpu_pool.h"

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

namespace mytorch {

namespace {

constexpr size_t kAlignment = 256;

size_t round_up(size_t bytes) { return ((bytes + kAlignment - 1) / kAlignment) * kAlignment; }

struct Pool {
    std::unordered_map<size_t, std::vector<void*>> freeList;
    std::unordered_map<void*, size_t> liveSizes;

    ~Pool() {
        for (auto& bucket : freeList)
            for (void* p : bucket.second) cudaFree(p);
    }
};

Pool& pool() {
    static Pool p;
    return p;
}

}  // namespace

void* pool_alloc(size_t bytes) {
    size_t rounded = round_up(bytes);
    Pool& p = pool();

    auto& bucket = p.freeList[rounded];
    if (!bucket.empty()) {
        void* ptr = bucket.back();
        bucket.pop_back();
        p.liveSizes[ptr] = rounded;
        return ptr;
    }

    void* ptr = nullptr;
    cudaMalloc(&ptr, rounded);
    p.liveSizes[ptr] = rounded;
    return ptr;
}

void pool_free(void* ptr) {
    if (!ptr) return;
    Pool& p = pool();
    auto it = p.liveSizes.find(ptr);
    size_t size = it != p.liveSizes.end() ? it->second : 0;
    if (it != p.liveSizes.end()) p.liveSizes.erase(it);
    p.freeList[size].push_back(ptr);
}

}  // namespace mytorch
