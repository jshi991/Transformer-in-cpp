#include "mytorch/layernorm.h"

#include "mytorch/gpu_pool.h"

#include <cuda_runtime.h>
#include <memory>
#include <numeric>
#include <vector>

namespace mytorch {

namespace {

__global__ void layernorm_fwd_k(float* out, float* xhat, float* rstd,
                                 const float* x, const float* gamma, const float* beta,
                                 int rows, int D, float eps) {
    long long r = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (; r < rows; r += stride) {
        float mean = 0.f;
        for (int c = 0; c < D; ++c) mean += x[r * D + c];
        mean /= D;

        float var = 0.f;
        for (int c = 0; c < D; ++c) {
            float d = x[r * D + c] - mean;
            var += d * d;
        }
        var /= D;

        float rs = rsqrtf(var + eps);
        rstd[r] = rs;
        for (int c = 0; c < D; ++c) {
            float xh = (x[r * D + c] - mean) * rs;
            xhat[r * D + c] = xh;
            out[r * D + c] = gamma[c] * xh + beta[c];
        }
    }
}

__global__ void layernorm_bwd_k(float* xGrad, float* gammaGrad, float* betaGrad,
                                 const float* dOut, const float* xhat, const float* rstd,
                                 const float* gamma, int rows, int D) {
    long long r = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (; r < rows; r += stride) {
        float sumDxhat = 0.f, sumDxhatXhat = 0.f;
        for (int c = 0; c < D; ++c) {
            float dxh = dOut[r * D + c] * gamma[c];
            sumDxhat += dxh;
            sumDxhatXhat += dxh * xhat[r * D + c];
        }

        float invD = 1.0f / D;
        for (int c = 0; c < D; ++c) {
            if (xGrad) {
                float dxh = dOut[r * D + c] * gamma[c];
                float dx = rstd[r] * (dxh - invD * sumDxhat - xhat[r * D + c] * invD * sumDxhatXhat);
                xGrad[r * D + c] += dx;
            }
            if (gammaGrad) atomicAdd(&gammaGrad[c], dOut[r * D + c] * xhat[r * D + c]);
            if (betaGrad) atomicAdd(&betaGrad[c], dOut[r * D + c]);
        }
    }
}

}  // namespace

LayerNorm::LayerNorm(uint16_t d_model, float eps)
    : d_model_(d_model), eps_(eps), gamma_({d_model}, AllocateOnGpu), beta_({d_model}, AllocateOnGpu) {
    std::vector<float> ones(d_model, 1.0f);
    cudaMemcpy(gamma_.data(), ones.data(), sizeof(float) * d_model, cudaMemcpyHostToDevice);

    gamma_.requires_grad(true);
    beta_.requires_grad(true);
}

Jensor<float>& LayerNorm::gamma() { return gamma_; }
Jensor<float>& LayerNorm::beta() { return beta_; }
std::vector<Jensor<float>*> LayerNorm::parameters() { return {&gamma_, &beta_}; }

Jensor<float> LayerNorm::forward(const Jensor<float>& x) {
    int D = x.shape().back();
    int rows = std::accumulate(x.shape().begin(), x.shape().end() - 1, 1LL, std::multiplies<long long>());
    Jensor<float> out(x.shape(), AllocateOnGpu);

    float* xhatRaw = (float*)pool_alloc(sizeof(float) * rows * D);
    float* rstdRaw = (float*)pool_alloc(sizeof(float) * rows);

    int tpb = 256, nb = (rows + tpb - 1) / tpb;
    layernorm_fwd_k<<<nb, tpb>>>(out.data(), xhatRaw, rstdRaw, x.data(), gamma_.data(), beta_.data(), rows, D, eps_);

    if (x.grad_node() || gamma_.grad_node() || beta_.grad_node()) {
        out.set_grad_node(make_grad_node<float>(out.shape(), out.on_gpu()));
        if (x.grad_node()) out.grad_node()->parents.push_back(x.grad_node());
        if (gamma_.grad_node()) out.grad_node()->parents.push_back(gamma_.grad_node());
        if (beta_.grad_node()) out.grad_node()->parents.push_back(beta_.grad_node());

        float* selfGrad = out.grad_node()->grad.get();
        auto xGradNode = x.grad_node();
        auto gammaGradNode = gamma_.grad_node();
        auto betaGradNode = beta_.grad_node();
        Jensor<float> gammaCopy = gamma_.detach();
        auto xhatPtr = std::shared_ptr<float>(xhatRaw, [](float* p) { pool_free(p); });
        auto rstdPtr = std::shared_ptr<float>(rstdRaw, [](float* p) { pool_free(p); });

        out.grad_node()->backward_fn = [selfGrad, xGradNode, gammaGradNode, betaGradNode,
                                         gammaCopy, xhatPtr, rstdPtr, rows, D]() {
            int tpb = 256, nb = (rows + tpb - 1) / tpb;
            layernorm_bwd_k<<<nb, tpb>>>(xGradNode ? xGradNode->grad.get() : nullptr,
                                         gammaGradNode ? gammaGradNode->grad.get() : nullptr,
                                         betaGradNode ? betaGradNode->grad.get() : nullptr,
                                         selfGrad, xhatPtr.get(), rstdPtr.get(), gammaCopy.data(), rows, D);
        };
    } else {
        pool_free(xhatRaw);
        pool_free(rstdRaw);
    }

    return out;
}

}  // namespace mytorch
