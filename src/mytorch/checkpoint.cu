#include "mytorch/checkpoint.h"

#include <cuda_runtime.h>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace mytorch {

void save_checkpoint(const std::string& path, const std::vector<Jensor<float>*>& params) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_checkpoint: could not open " + path);

    uint32_t numParams = (uint32_t)params.size();
    f.write(reinterpret_cast<const char*>(&numParams), sizeof(numParams));

    for (Jensor<float>* p : params) {
        const auto& shape = p->shape();
        uint32_t rank = (uint32_t)shape.size();
        f.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        f.write(reinterpret_cast<const char*>(shape.data()), sizeof(uint16_t) * rank);

        long long n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<long long>());
        std::vector<float> host(n);
        cudaMemcpy(host.data(), p->data(), sizeof(float) * n, cudaMemcpyDeviceToHost);
        f.write(reinterpret_cast<const char*>(host.data()), sizeof(float) * n);
    }
}

void load_checkpoint(const std::string& path, const std::vector<Jensor<float>*>& params) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_checkpoint: could not open " + path);

    uint32_t numParams = 0;
    f.read(reinterpret_cast<char*>(&numParams), sizeof(numParams));
    if (numParams != params.size())
        throw std::runtime_error("load_checkpoint: parameter count mismatch");

    for (Jensor<float>* p : params) {
        uint32_t rank = 0;
        f.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        std::vector<uint16_t> shape(rank);
        f.read(reinterpret_cast<char*>(shape.data()), sizeof(uint16_t) * rank);
        if (shape != p->shape())
            throw std::runtime_error("load_checkpoint: shape mismatch");

        long long n = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<long long>());
        std::vector<float> host(n);
        f.read(reinterpret_cast<char*>(host.data()), sizeof(float) * n);
        cudaMemcpy(p->data(), host.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    }
}

}  // namespace mytorch
