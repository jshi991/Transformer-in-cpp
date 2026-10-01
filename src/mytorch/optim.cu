#include "mytorch/optim.h"

#include "mytorch/gpu_pool.h"

#include <cuda_runtime.h>
#include <numeric>

namespace mytorch {

namespace {

__global__ void adam_step_k(float* param, const float* grad, float* m, float* v,
                             float lr, float beta1, float beta2, float eps,
                             float biasCorr1, float biasCorr2, long long n) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (long long i = index; i < n; i += stride) {
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float mhat = m[i] / biasCorr1;
        float vhat = v[i] / biasCorr2;
        param[i] -= lr * mhat / (sqrtf(vhat) + eps);
    }
}

long long numel(const Jensor<float>& t) {
    return std::accumulate(t.shape().begin(), t.shape().end(), 1LL, std::multiplies<long long>());
}

}  // namespace

Adam::Adam(std::vector<Jensor<float>*> params, float lr, float beta1, float beta2, float eps)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (Jensor<float>* p : params_) {
        long long n = numel(*p);
        float* mRaw = (float*)pool_alloc(sizeof(float) * n);
        cudaMemset(mRaw, 0, sizeof(float) * n);
        float* vRaw = (float*)pool_alloc(sizeof(float) * n);
        cudaMemset(vRaw, 0, sizeof(float) * n);
        m_.push_back(std::shared_ptr<float[]>(mRaw, CudaDeleter<float>()));
        v_.push_back(std::shared_ptr<float[]>(vRaw, CudaDeleter<float>()));
    }
}

void Adam::step() {
    t_++;
    float biasCorr1 = 1.0f - powf(beta1_, (float)t_);
    float biasCorr2 = 1.0f - powf(beta2_, (float)t_);

    for (size_t i = 0; i < params_.size(); ++i) {
        Jensor<float>* p = params_[i];
        if (!p->grad_node()) continue;

        long long n = numel(*p);
        int tpb = 256, nb = (n + tpb - 1) / tpb;
        adam_step_k<<<nb, tpb>>>(p->data(), p->grad(), m_[i].get(), v_[i].get(),
                                 lr_, beta1_, beta2_, eps_, biasCorr1, biasCorr2, n);
    }
}

void Adam::zero_grad() {
    for (Jensor<float>* p : params_) p->zero_grad();
}

}  // namespace mytorch
