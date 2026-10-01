#include "mytorch/ops.h"

#include "mytorch/gpu_pool.h"

#include <cuda_runtime.h>
#include <memory>
#include <numeric>
#include <vector>

namespace mytorch {

namespace {

long long flatten_rows(const std::vector<uint16_t>& shape) {
    return std::accumulate(shape.begin(), shape.end() - 1, 1LL, std::multiplies<long long>());
}

std::vector<uint16_t> with_last_dim(std::vector<uint16_t> shape, uint16_t newLast) {
    shape.back() = newLast;
    return shape;
}

__global__ void scale_k(float* out, const float* in, float factor, long long n) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = index; i < n; i += stride) out[i] = in[i] * factor;
}

__global__ void scaled_accumulate_k(float* dst, const float* src, float factor, long long n) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = index; i < n; i += stride) dst[i] += src[i] * factor;
}

__global__ void relu_fwd_k(float* out, const float* in, long long n) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = index; i < n; i += stride) out[i] = in[i] > 0.f ? in[i] : 0.f;
}

__global__ void relu_bwd_k(float* xGrad, const float* outGrad, const float* x, long long n) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = index; i < n; i += stride)
        if (x[i] > 0.f) xGrad[i] += outGrad[i];
}

__global__ void slice_cols_k(float* out, const float* in, int rows, int cols, int start, int count) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)rows * count;
    for (long long i = index; i < total; i += stride) {
        int r = i / count;
        int c = i % count;
        out[i] = in[r * cols + start + c];
    }
}

__global__ void slice_cols_bwd_k(float* xGrad, const float* outGrad, int rows, int cols, int start, int count) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)rows * count;
    for (long long i = index; i < total; i += stride) {
        int r = i / count;
        int c = i % count;
        xGrad[r * cols + start + c] += outGrad[i];
    }
}

// batch independent causal softmax: row r of each (seqLen x seqLen) block only attends to cols [0, r].
__global__ void causal_softmax_fwd_k(float* out, const float* in, int batch, int seqLen) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long totalRows = (long long)batch * seqLen;

    for (long long gr = index; gr < totalRows; gr += stride) {
        long long b = gr / seqLen;
        int r = gr % seqLen;
        long long base = b * seqLen * seqLen + (long long)r * seqLen;
        const float* inRow = in + base;
        float* outRow = out + base;

        int validLen = r + 1;
        float maxVal = inRow[0];
        for (int c = 1; c < validLen; ++c) maxVal = fmaxf(maxVal, inRow[c]);

        float sum = 0.f;
        for (int c = 0; c < validLen; ++c) {
            float e = expf(inRow[c] - maxVal);
            outRow[c] = e;
            sum += e;
        }
        for (int c = 0; c < validLen; ++c) outRow[c] /= sum;
        for (int c = validLen; c < seqLen; ++c) outRow[c] = 0.f;
    }
}

__global__ void causal_softmax_bwd_k(float* xGrad, const float* outGrad, const float* y, int batch, int seqLen) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long totalRows = (long long)batch * seqLen;

    for (long long gr = index; gr < totalRows; gr += stride) {
        long long b = gr / seqLen;
        int r = gr % seqLen;
        long long base = b * seqLen * seqLen + (long long)r * seqLen;
        const float* outGradRow = outGrad + base;
        const float* yRow = y + base;
        float* xGradRow = xGrad + base;

        int validLen = r + 1;
        float dot = 0.f;
        for (int c = 0; c < validLen; ++c) dot += outGradRow[c] * yRow[c];
        for (int c = 0; c < validLen; ++c) xGradRow[c] += yRow[c] * (outGradRow[c] - dot);
    }
}

__global__ void cross_entropy_fwd_k(float* probs, float* rowLoss, const float* logits,
                                     const int32_t* targets, long long rows, int vocab) {
    long long r = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (; r < rows; r += stride) {
        float maxVal = logits[r * vocab];
        for (int c = 1; c < vocab; ++c) maxVal = fmaxf(maxVal, logits[r * vocab + c]);

        float sum = 0.f;
        for (int c = 0; c < vocab; ++c) {
            float e = expf(logits[r * vocab + c] - maxVal);
            probs[r * vocab + c] = e;
            sum += e;
        }
        for (int c = 0; c < vocab; ++c) probs[r * vocab + c] /= sum;

        int t = targets[r];
        rowLoss[r] = -logf(fmaxf(probs[r * vocab + t], 1e-12f));
    }
}

__global__ void cross_entropy_bwd_k(float* logitsGrad, const float* probs, const int32_t* targets,
                                     long long rows, int vocab, float scale) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = rows * vocab;

    for (long long i = index; i < total; i += stride) {
        long long r = i / vocab;
        int c = i % vocab;
        float g = probs[i] - (c == targets[r] ? 1.0f : 0.0f);
        logitsGrad[i] += scale * g / rows;
    }
}

__global__ void softmax_rows_k(float* out, const float* in, int rows, int cols) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for (long long r = index; r < rows; r += stride) {
        const float* inRow = in + r * cols;
        float* outRow = out + r * cols;

        float maxVal = inRow[0];
        for (int c = 1; c < cols; ++c) maxVal = fmaxf(maxVal, inRow[c]);

        float sum = 0.f;
        for (int c = 0; c < cols; ++c) {
            float e = expf(inRow[c] - maxVal);
            outRow[c] = e;
            sum += e;
        }
        for (int c = 0; c < cols; ++c) outRow[c] /= sum;
    }
}

}  // namespace

Jensor<float> transposed(const Jensor<float>& x) {
    const auto& shape = x.shape();
    int batch = shape.size() == 3 ? shape[0] : 1;
    int rows = shape[shape.size() - 2], cols = shape[shape.size() - 1];

    std::vector<uint16_t> outShape = shape;
    std::swap(outShape[outShape.size() - 2], outShape[outShape.size() - 1]);
    Jensor<float> ret(outShape, AllocateOnGpu);

    batched_transpose_launch<float>(ret.data(), x.data(), batch, rows, cols);

    if (x.grad_node()) {
        ret.set_grad_node(make_grad_node<float>(ret.shape(), ret.on_gpu()));
        ret.grad_node()->parents.push_back(x.grad_node());
        float* selfGrad = ret.grad_node()->grad.get();
        auto xGradNode = x.grad_node();

        ret.grad_node()->backward_fn = [selfGrad, xGradNode, batch, rows, cols]() {
            long long n = (long long)batch * rows * cols;
            float* tmp = (float*)pool_alloc(sizeof(float) * n);
            batched_transpose_launch<float>(tmp, selfGrad, batch, cols, rows);
            accumulate_into(xGradNode->grad.get(), tmp, n, true);
            pool_free(tmp);
        };
    }

    return ret;
}

Jensor<float> scale(const Jensor<float>& x, float factor) {
    Jensor<float> ret(x.shape(), AllocateOnGpu);
    long long n = std::accumulate(x.shape().begin(), x.shape().end(), 1LL, std::multiplies<long long>());
    int tpb = 256, nb = (n + tpb - 1) / tpb;
    scale_k<<<nb, tpb>>>(ret.data(), x.data(), factor, n);

    if (x.grad_node()) {
        ret.set_grad_node(make_grad_node<float>(ret.shape(), ret.on_gpu()));
        ret.grad_node()->parents.push_back(x.grad_node());
        float* selfGrad = ret.grad_node()->grad.get();
        auto xGradNode = x.grad_node();

        ret.grad_node()->backward_fn = [selfGrad, xGradNode, factor, n]() {
            int tpb = 256, nb = (n + tpb - 1) / tpb;
            scaled_accumulate_k<<<nb, tpb>>>(xGradNode->grad.get(), selfGrad, factor, n);
        };
    }

    return ret;
}

Jensor<float> relu(const Jensor<float>& x) {
    Jensor<float> ret(x.shape(), AllocateOnGpu);
    long long n = std::accumulate(x.shape().begin(), x.shape().end(), 1LL, std::multiplies<long long>());
    int tpb = 256, nb = (n + tpb - 1) / tpb;
    relu_fwd_k<<<nb, tpb>>>(ret.data(), x.data(), n);

    if (x.grad_node()) {
        ret.set_grad_node(make_grad_node<float>(ret.shape(), ret.on_gpu()));
        ret.grad_node()->parents.push_back(x.grad_node());
        float* selfGrad = ret.grad_node()->grad.get();
        auto xGradNode = x.grad_node();
        Jensor<float> xCopy = x.detach();

        ret.grad_node()->backward_fn = [selfGrad, xGradNode, xCopy, n]() {
            int tpb = 256, nb = (n + tpb - 1) / tpb;
            relu_bwd_k<<<nb, tpb>>>(xGradNode->grad.get(), selfGrad, xCopy.data(), n);
        };
    }

    return ret;
}

Jensor<float> col_slice(const Jensor<float>& x, uint16_t start, uint16_t count) {
    long long rows = flatten_rows(x.shape());
    int cols = x.shape().back();
    Jensor<float> ret(with_last_dim(x.shape(), count), AllocateOnGpu);

    long long n = rows * count;
    int tpb = 256, nb = (n + tpb - 1) / tpb;
    slice_cols_k<<<nb, tpb>>>(ret.data(), x.data(), (int)rows, cols, start, count);

    if (x.grad_node()) {
        ret.set_grad_node(make_grad_node<float>(ret.shape(), ret.on_gpu()));
        ret.grad_node()->parents.push_back(x.grad_node());
        float* selfGrad = ret.grad_node()->grad.get();
        auto xGradNode = x.grad_node();

        ret.grad_node()->backward_fn = [selfGrad, xGradNode, rows, cols, start, count, n]() {
            int tpb = 256, nb = (n + tpb - 1) / tpb;
            slice_cols_bwd_k<<<nb, tpb>>>(xGradNode->grad.get(), selfGrad, (int)rows, cols, start, count);
        };
    }

    return ret;
}

Jensor<float> causal_softmax(const Jensor<float>& scores) {
    const auto& shape = scores.shape();
    int batch = shape.size() == 3 ? shape[0] : 1;
    int seqLen = shape[shape.size() - 1];
    Jensor<float> ret(shape, AllocateOnGpu);

    int tpb = 256, nb = (batch * seqLen + tpb - 1) / tpb;
    causal_softmax_fwd_k<<<nb, tpb>>>(ret.data(), scores.data(), batch, seqLen);

    if (scores.grad_node()) {
        ret.set_grad_node(make_grad_node<float>(ret.shape(), ret.on_gpu()));
        ret.grad_node()->parents.push_back(scores.grad_node());
        float* selfGrad = ret.grad_node()->grad.get();
        auto xGradNode = scores.grad_node();
        Jensor<float> yCopy = ret.detach();

        ret.grad_node()->backward_fn = [selfGrad, xGradNode, yCopy, batch, seqLen]() {
            int tpb = 256, nb = (batch * seqLen + tpb - 1) / tpb;
            causal_softmax_bwd_k<<<nb, tpb>>>(xGradNode->grad.get(), selfGrad, yCopy.data(), batch, seqLen);
        };
    }

    return ret;
}

Jensor<float> cross_entropy(const Jensor<float>& logits, const std::vector<int32_t>& targets) {
    long long rows = flatten_rows(logits.shape());
    int vocab = logits.shape().back();

    int32_t* targetsRaw = (int32_t*)pool_alloc(sizeof(int32_t) * rows);
    cudaMemcpy(targetsRaw, targets.data(), sizeof(int32_t) * rows, cudaMemcpyHostToDevice);

    float* probsRaw = (float*)pool_alloc(sizeof(float) * rows * vocab);
    float* rowLossRaw = (float*)pool_alloc(sizeof(float) * rows);

    int tpb = 256, nb = (rows + tpb - 1) / tpb;
    cross_entropy_fwd_k<<<nb, tpb>>>(probsRaw, rowLossRaw, logits.data(), targetsRaw, rows, vocab);

    std::vector<float> rowLossHost(rows);
    cudaMemcpy(rowLossHost.data(), rowLossRaw, sizeof(float) * rows, cudaMemcpyDeviceToHost);
    float meanLoss = 0.f;
    for (float v : rowLossHost) meanLoss += v;
    meanLoss /= rows;

    Jensor<float> loss({1, 1}, AllocateOnGpu);
    cudaMemcpy(loss.data(), &meanLoss, sizeof(float), cudaMemcpyHostToDevice);
    pool_free(rowLossRaw);

    if (logits.grad_node()) {
        loss.set_grad_node(make_grad_node<float>(loss.shape(), loss.on_gpu()));
        loss.grad_node()->parents.push_back(logits.grad_node());

        float* selfGrad = loss.grad_node()->grad.get();
        auto logitsGradNode = logits.grad_node();
        auto probsPtr = std::shared_ptr<float>(probsRaw, [](float* p) { pool_free(p); });
        auto targetsPtr = std::shared_ptr<int32_t>(targetsRaw, [](int32_t* p) { pool_free(p); });

        loss.grad_node()->backward_fn = [selfGrad, logitsGradNode, probsPtr, targetsPtr, rows, vocab]() {
            float scale = 1.f;
            cudaMemcpy(&scale, selfGrad, sizeof(float), cudaMemcpyDeviceToHost);
            long long total = rows * vocab;
            int tpb = 256, nb = (total + tpb - 1) / tpb;
            cross_entropy_bwd_k<<<nb, tpb>>>(logitsGradNode->grad.get(), probsPtr.get(), targetsPtr.get(), rows, vocab, scale);
        };
    } else {
        pool_free(probsRaw);
        pool_free(targetsRaw);
    }

    return loss;
}

Jensor<float> softmax_rows(const Jensor<float>& scores) {
    long long rows = flatten_rows(scores.shape());
    int cols = scores.shape().back();
    Jensor<float> ret(scores.shape(), AllocateOnGpu);

    int tpb = 256, nb = (rows + tpb - 1) / tpb;
    softmax_rows_k<<<nb, tpb>>>(ret.data(), scores.data(), (int)rows, cols);

    return ret;
}

}  // namespace mytorch
