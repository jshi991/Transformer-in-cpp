#include "mytorch/embeddings.h"

#include "mytorch/gpu_pool.h"

#include <atomic>
#include <cuda_runtime.h>
#include <curand.h>
#include <memory>
#include <random>

namespace mytorch {

namespace {

unsigned long long next_seed() {
    static std::atomic<unsigned long long> counter{std::random_device{}()};
    return counter.fetch_add(0x9E3779B97F4A7C15ULL);
}

__global__ void gather(float* out, const float* weight, const int32_t* ids, int seqLen, int dModel) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)seqLen * dModel;

    for (long long i = index; i < total; i += stride) {
        int row = i / dModel;
        int col = i % dModel;
        out[i] = weight[(long long)ids[row] * dModel + col];
    }
}

__global__ void gather_bwd(float* weightGrad, const float* outGrad, const int32_t* ids, int seqLen, int dModel) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)seqLen * dModel;

    for (long long i = index; i < total; i += stride) {
        int row = i / dModel;
        int col = i % dModel;
        atomicAdd(&weightGrad[(long long)ids[row] * dModel + col], outGrad[i]);
    }
}

__global__ void sinusoidal(float* out, int seqLen, int dModel) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)seqLen * dModel;

    for (long long i = index; i < total; i += stride) {
        int pos = i / dModel;
        int dim = i % dModel;
        float freq = powf(10000.0f, -2.0f * (dim / 2) / dModel);
        float angle = pos * freq;
        out[i] = (dim % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

__global__ void sinusoidal_batched(float* out, int batch, int seqLen, int dModel) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)batch * seqLen * dModel;

    for (long long i = index; i < total; i += stride) {
        long long rem = i % ((long long)seqLen * dModel);
        int pos = rem / dModel;
        int dim = rem % dModel;
        float freq = powf(10000.0f, -2.0f * (dim / 2) / dModel);
        float angle = pos * freq;
        out[i] = (dim % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

__global__ void sinusoidal_at(float* out, int pos, int dModel) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (long long dim = index; dim < dModel; dim += stride) {
        float freq = powf(10000.0f, -2.0f * (dim / 2) / dModel);
        float angle = pos * freq;
        out[dim] = (dim % 2 == 0) ? sinf(angle) : cosf(angle);
    }
}

}

InputEmbeddings::InputEmbeddings(uint16_t vocab_size, uint16_t d_model)
    : vocab_size_(vocab_size), d_model_(d_model), weight_({vocab_size, d_model}, AllocateOnGpu) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, next_seed());
    curandGenerateNormal(gen, weight_.data(), (size_t)vocab_size * d_model, 0.0f, 0.02f);
    curandDestroyGenerator(gen);

    weight_.requires_grad(true);
}

std::vector<Jensor<float>*> InputEmbeddings::parameters() { return {&weight_}; }
Jensor<float>& InputEmbeddings::weight() { return weight_; }

Jensor<float> InputEmbeddings::forward(const std::vector<int32_t>& token_ids) {
    uint16_t seqLen = static_cast<uint16_t>(token_ids.size());
    Jensor<float> out = forward(token_ids, 1, seqLen);
    out.reshape({seqLen, d_model_});
    return out;
}

Jensor<float> InputEmbeddings::forward(const std::vector<int32_t>& flat_ids, uint16_t batch, uint16_t seq_len) {
    long long rows = (long long)batch * seq_len;

    int32_t* idsDevice = (int32_t*)pool_alloc(sizeof(int32_t) * rows);
    cudaMemcpy(idsDevice, flat_ids.data(), sizeof(int32_t) * rows, cudaMemcpyHostToDevice);

    Jensor<float> out({batch, seq_len, d_model_}, AllocateOnGpu);

    int threadsPerBlock = 256;
    long long total = rows * d_model_;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    gather<<<numBlocks, threadsPerBlock>>>(out.data(), weight_.data(), idsDevice, (int)rows, d_model_);

    if (weight_.grad_node()) {
        out.set_grad_node(make_grad_node<float>(out.shape(), out.on_gpu()));
        out.grad_node()->parents.push_back(weight_.grad_node());

        float* selfGrad = out.grad_node()->grad.get();
        auto weightGradNode = weight_.grad_node();
        auto idsPtr = std::shared_ptr<int32_t>(idsDevice, [](int32_t* p) { pool_free(p); });
        uint16_t dModel = d_model_;

        out.grad_node()->backward_fn = [selfGrad, weightGradNode, idsPtr, rows, dModel]() {
            long long total = rows * dModel;
            int tpb = 256, nb = (total + tpb - 1) / tpb;
            gather_bwd<<<nb, tpb>>>(weightGradNode->grad.get(), selfGrad, idsPtr.get(), (int)rows, dModel);
        };
    } else {
        pool_free(idsDevice);
    }

    return out;
}

Jensor<float> positional_encoding(uint16_t seq_len, uint16_t d_model) {
    Jensor<float> pe({seq_len, d_model}, AllocateOnGpu);

    int threadsPerBlock = 256;
    long long total = (long long)seq_len * d_model;
    int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;
    sinusoidal<<<numBlocks, threadsPerBlock>>>(pe.data(), seq_len, d_model);

    return pe;
}

Jensor<float> positional_encoding(uint16_t seq_len, uint16_t d_model, uint16_t batch) {
    Jensor<float> pe({batch, seq_len, d_model}, AllocateOnGpu);

    long long total = (long long)batch * seq_len * d_model;
    int tpb = 256, nb = (total + tpb - 1) / tpb;
    sinusoidal_batched<<<nb, tpb>>>(pe.data(), batch, seq_len, d_model);

    return pe;
}

Jensor<float> positional_encoding_at(uint16_t position, uint16_t d_model) {
    Jensor<float> pe({(uint16_t)1, d_model}, AllocateOnGpu);

    int tpb = 256, nb = (d_model + tpb - 1) / tpb;
    sinusoidal_at<<<nb, tpb>>>(pe.data(), position, d_model);

    return pe;
}

}
