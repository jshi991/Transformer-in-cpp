#include "mytorch/linear.h"

#include "mytorch/gpu_pool.h"

#include <atomic>
#include <cuda_runtime.h>
#include <curand.h>
#include <random>

namespace mytorch {

namespace {

unsigned long long next_seed() {
    static std::atomic<unsigned long long> counter{std::random_device{}()};
    return counter.fetch_add(0x9E3779B97F4A7C15ULL);
}

// Weight-only int8 GEMM: weight codes stay packed in device memory (1 byte
// per value instead of 4) and are dequantized inline per multiply-add,
// rather than ever materializing a full fp32 weight copy.
__global__ void int8_weightonly_matmul_k(const float* a, const int8_t* wCodes, const float* groupScale,
                                          const float* channelScale, int groupSize, float* c,
                                          int M, int N, int K) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)M * N;

    for (long long i = index; i < total; i += stride) {
        int row = i / N;
        int col = i % N;

        float sum = 0.f;
        for (int k = 0; k < K; ++k) {
            float wscale = groupScale[k / groupSize] / channelScale[k];
            sum += a[row * K + k] * ((float)wCodes[k * N + col] * wscale);
        }
        c[i] = sum;
    }
}

__global__ void add_bias_fwd_k(float* out, const float* in, const float* bias, int rows, int cols) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)rows * cols;
    for (long long i = index; i < total; i += stride) out[i] = in[i] + bias[i % cols];
}

__global__ void bias_bwd_reduce_k(float* biasGrad, const float* outGrad, int rows, int cols) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long c = index; c < cols; c += stride) {
        float sum = 0.f;
        for (int r = 0; r < rows; ++r) sum += outGrad[r * cols + c];
        biasGrad[c] += sum;
    }
}

}  // namespace

Linear::Linear(uint16_t in_features, uint16_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias),
      weight_({in_features, out_features}, AllocateOnGpu),
      bias_({(uint16_t)(has_bias_ ? out_features : 1)}, AllocateOnGpu) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, next_seed());
    float stddev = 1.0f / sqrtf((float)in_features);
    curandGenerateNormal(gen, weight_.data(), (size_t)in_features * out_features, 0.0f, stddev);
    curandDestroyGenerator(gen);

    weight_.requires_grad(true);
    if (has_bias_) bias_.requires_grad(true);
}

Jensor<float>& Linear::weight() { return weight_; }
Jensor<float>* Linear::bias() { return has_bias_ ? &bias_ : nullptr; }

std::vector<Jensor<float>*> Linear::parameters() {
    if (has_bias_) return {&weight_, &bias_};
    return {&weight_};
}

void Linear::set_observer(ActivationObserver* observer) { observer_ = observer; }

void Linear::load_int8_weights(const std::vector<uint8_t>& codes, const std::vector<float>& groupScale,
                                const std::vector<float>& channelScale, int groupSize) {
    groupSize_ = groupSize;

    int8_t* codesRaw = (int8_t*)pool_alloc(codes.size());
    cudaMemcpy(codesRaw, codes.data(), codes.size(), cudaMemcpyHostToDevice);
    qCodes_ = std::shared_ptr<int8_t>(codesRaw, [](int8_t* p) { pool_free(p); });

    float* groupRaw = (float*)pool_alloc(sizeof(float) * groupScale.size());
    cudaMemcpy(groupRaw, groupScale.data(), sizeof(float) * groupScale.size(), cudaMemcpyHostToDevice);
    qGroupScale_ = std::shared_ptr<float>(groupRaw, [](float* p) { pool_free(p); });

    float* channelRaw = (float*)pool_alloc(sizeof(float) * channelScale.size());
    cudaMemcpy(channelRaw, channelScale.data(), sizeof(float) * channelScale.size(), cudaMemcpyHostToDevice);
    qChannelScale_ = std::shared_ptr<float>(channelRaw, [](float* p) { pool_free(p); });

    quantized_ = true;
}

Jensor<float> Linear::forward(const Jensor<float>& x) {
    if (observer_) observer_->observe(x);

    const auto& xShape = x.shape();
    bool flatten = xShape.size() > 2;

    Jensor<float> flatX = x;
    if (flatten) {
        long long rows = 1;
        for (size_t i = 0; i + 1 < xShape.size(); ++i) rows *= xShape[i];
        flatX.reshape({(uint16_t)rows, in_features_});
    }

    Jensor<float> y = [&]() -> Jensor<float> {
        if (!quantized_) return flatX.matmul(weight_);

        int rows = flatX.shape()[0];
        Jensor<float> out({(uint16_t)rows, out_features_}, AllocateOnGpu);
        long long total = (long long)rows * out_features_;
        int tpb = 256, nb = (total + tpb - 1) / tpb;
        int8_weightonly_matmul_k<<<nb, tpb>>>(flatX.data(), qCodes_.get(), qGroupScale_.get(), qChannelScale_.get(),
                                               groupSize_, out.data(), rows, out_features_, in_features_);
        return out;
    }();

    auto restore_shape = [&](Jensor<float> t) {
        if (flatten) {
            std::vector<uint16_t> outShape(xShape.begin(), xShape.end() - 1);
            outShape.push_back(out_features_);
            t.reshape(outShape);
        }
        return t;
    };

    if (!has_bias_) return restore_shape(y);

    int rows = y.shape()[0], cols = y.shape()[1];
    Jensor<float> out({(uint16_t)rows, (uint16_t)cols}, AllocateOnGpu);
    long long n = (long long)rows * cols;
    int tpb = 256, nb = (n + tpb - 1) / tpb;
    add_bias_fwd_k<<<nb, tpb>>>(out.data(), y.data(), bias_.data(), rows, cols);

    if (y.grad_node() || bias_.grad_node()) {
        out.set_grad_node(make_grad_node<float>(out.shape(), out.on_gpu()));
        if (y.grad_node()) out.grad_node()->parents.push_back(y.grad_node());
        if (bias_.grad_node()) out.grad_node()->parents.push_back(bias_.grad_node());

        float* selfGrad = out.grad_node()->grad.get();
        auto yGradNode = y.grad_node();
        auto biasGradNode = bias_.grad_node();

        out.grad_node()->backward_fn = [selfGrad, yGradNode, biasGradNode, rows, cols, n]() {
            if (yGradNode) accumulate_into(yGradNode->grad.get(), selfGrad, n, true);
            if (biasGradNode) {
                int tpb = 256, nb = (cols + tpb - 1) / tpb;
                bias_bwd_reduce_k<<<nb, tpb>>>(biasGradNode->grad.get(), selfGrad, rows, cols);
            }
        };
    }

    return restore_shape(out);
}

}  // namespace mytorch
