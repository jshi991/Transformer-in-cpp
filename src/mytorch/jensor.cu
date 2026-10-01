#include "mytorch/jensor.h"

#include "mytorch/gpu_pool.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace mytorch {

template <typename T>
void CudaDeleter<T>::operator()(T* ptr) const { pool_free(ptr); }

template <typename T>
__global__ void transpose(T* outPtr, const T* inPtr, int inRows, int inCols) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)inRows * inCols;

    for(long long i = index; i < total; i += stride) {
        int r = i / inCols;
        int c = i % inCols;
        outPtr[c * inRows + r] = inPtr[r * inCols + c];
    }
}

template <typename T>
__global__ void add(T* a, const T* b, const T* c, long long n) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for(long long i = index; i < n; i += stride) {
        a[i] = b[i] + c[i];
    }
}

//a M X N
//b N X V
template <typename T>
__global__ void matmul_gpu(const T* a, const T* b, T* c, int finalRow, int finalCol, int N) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)finalRow * finalCol;

    for(long long i = index; i < total; i += stride) {
        int row = i / finalCol;
        int col = i % finalCol;

        T totalSum{};
        for(int k = 0; k < N; k++) {
            totalSum += a[row * N + k] * b[k * finalCol + col];
        }

        c[i] = totalSum;
    }
}

template <typename T>
__global__ void batched_transpose_k(T* out, const T* in, int batch, int rows, int cols) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)batch * rows * cols;

    for (long long i = index; i < total; i += stride) {
        long long b = i / (rows * cols);
        long long rem = i % (rows * cols);
        int r = rem / cols;
        int c = rem % cols;
        out[b * cols * rows + c * rows + r] = in[b * rows * cols + r * cols + c];
    }
}

template <typename T>
__global__ void batched_matmul_k(const T* a, const T* b, T* c, int batch, int M, int N, int K) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long total = (long long)batch * M * N;

    for (long long i = index; i < total; i += stride) {
        long long bIdx = i / (M * N);
        long long rem = i % (M * N);
        int row = rem / N;
        int col = rem % N;

        const T* aB = a + bIdx * M * K;
        const T* bB = b + bIdx * K * N;

        T sum{};
        for (int k = 0; k < K; ++k) sum += aB[row * K + k] * bB[k * N + col];
        c[i] = sum;
    }
}

template <typename T>
void batched_transpose_launch(T* out, const T* in, int batch, int rows, int cols) {
    long long total = (long long)batch * rows * cols;
    int tpb = 256, nb = (total + tpb - 1) / tpb;
    batched_transpose_k<T><<<nb, tpb>>>(out, in, batch, rows, cols);
}

template <typename T>
void batched_matmul_launch(const T* a, const T* b, T* c, int batch, int M, int N, int K) {
    long long total = (long long)batch * M * N;
    int tpb = 256, nb = (total + tpb - 1) / tpb;
    batched_matmul_k<T><<<nb, tpb>>>(a, b, c, batch, M, N, K);
}

template <typename T>
//M x N, N x Z
__global__ void concat_gpu(const T* a, const T* b, const T* c, const uint16_t* aDims, const uint16_t* aStride, const uint16_t* bDims, const uint16_t* bStrides, const long long& finalElems) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for(long long i = index; i < finalElems; i+=stride) {

    }
}

template <typename T>
__global__ void accumulate(T* dst, const T* src, long long n) {
    long long index = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;

    for(long long i = index; i < n; i += stride) {
        dst[i] += src[i];
    }
}

template <typename T>
static std::shared_ptr<T[]> zeroed_buffer(long long n, bool on_gpu) {
    if (on_gpu) {
        T* raw = nullptr;
        raw = (T*)pool_alloc(sizeof(T) * n);
        cudaMemset(raw, 0, sizeof(T) * n);
        return std::shared_ptr<T[]>(raw, CudaDeleter<T>());
    }
    T* raw = new T[n]();
    return std::shared_ptr<T[]>(raw);
}

template <typename T>
std::shared_ptr<GradNode<T>> make_grad_node(const std::vector<uint16_t>& dims, bool on_gpu) {
    auto node = std::make_shared<GradNode<T>>();
    node->dims = dims;
    node->is_on_gpu = on_gpu;
    long long n = std::accumulate(dims.begin(), dims.end(), 1LL, std::multiplies<long long>());
    node->grad = zeroed_buffer<T>(n, on_gpu);
    return node;
}

template <typename T>
void accumulate_into(T* dst, const T* src, long long n, bool on_gpu) {
    if (on_gpu) {
        int threadsPerBlock = 256;
        int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
        accumulate<T><<<numBlocks, threadsPerBlock>>>(dst, src, n);
    } else {
        for (long long i = 0; i < n; ++i) dst[i] += src[i];
    }
}
template <typename T>
void Jensor<T>::transpose() {
    assert(dims_.size() == 2 && "transpose is only supported for 2D Jensors currently");

    uint16_t inRows = dims_[0];
    uint16_t inCols = dims_[1];
    auto grad_node = grad_node_;
    bool req = requires_grad_;

    if (is_on_gpu_) {
        Jensor<T> ret({inCols, inRows}, AllocateOnGpu);

        int threadsPerBlock = 256;
        int numBlocks = (inRows * inCols + threadsPerBlock - 1) / threadsPerBlock;

        mytorch::transpose<T><<<numBlocks, threadsPerBlock>>>(ret.data(), data(), inRows, inCols);

        if (grad_node) {
            T* newGrad = nullptr;
            newGrad = (T*)pool_alloc(sizeof(T) * inRows * inCols);
            mytorch::transpose<T><<<numBlocks, threadsPerBlock>>>(newGrad, grad_node->grad.get(), inRows, inCols);
            grad_node->grad = std::shared_ptr<T[]>(newGrad, CudaDeleter<T>());
            grad_node->dims = ret.dims_;
        }

        *this = ret;
    } else {
        Jensor<T> ret({inCols, inRows}, AllocateOnCpu);

        for (uint16_t r = 0; r < inRows; ++r) {
            for (uint16_t c = 0; c < inCols; ++c) {
                ret.data()[c * inRows + r] = data()[r * inCols + c];
            }
        }

        if (grad_node) {
            long long n = static_cast<long long>(inRows) * inCols;
            T* newGrad = new T[n];
            for (uint16_t r = 0; r < inRows; ++r) {
                for (uint16_t c = 0; c < inCols; ++c) {
                    newGrad[c * inRows + r] = grad_node->grad.get()[r * inCols + c];
                }
            }
            grad_node->grad = std::shared_ptr<T[]>(newGrad);
            grad_node->dims = ret.dims_;
        }

        *this = ret;
    }

    grad_node_ = grad_node;
    requires_grad_ = req;
}

//will not be doing cuBLAS benchmarking for this
template <typename T>
Jensor<T> Jensor<T>::operator+(const Jensor& other) const {
    assert((dims_ == other.dims_) && "Matrix shapes must match!");
    assert((is_on_gpu_ == other.is_on_gpu_) && "both Jensors must be on the same device to add");

    Jensor<T> ret = is_on_gpu_ ? Jensor<T>(dims_, AllocateOnGpu)
                                : Jensor<T>(dims_, AllocateOnCpu);
    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());

    if (is_on_gpu_) {
        int threadsPerBlock = 256;
        int numBlocks = (total + threadsPerBlock - 1) / threadsPerBlock;

        add<<<numBlocks, threadsPerBlock>>>(ret.data(), data(), other.data(), total);
    } else {
        for (long long i = 0; i < total; ++i) {
            ret.data()[i] = data()[i] + other.data()[i];
        }
    }

    if (grad_node_ || other.grad_node_) {
        ret.requires_grad_ = true;
        ret.grad_node_ = make_grad_node<T>(ret.dims_, ret.is_on_gpu_);
        if (grad_node_) ret.grad_node_->parents.push_back(grad_node_);
        if (other.grad_node_) ret.grad_node_->parents.push_back(other.grad_node_);

        T* selfGrad = ret.grad_node_->grad.get();
        auto aGradNode = grad_node_;
        auto bGradNode = other.grad_node_;
        bool onGpu = ret.is_on_gpu_;

        ret.grad_node_->backward_fn = [selfGrad, total, aGradNode, bGradNode, onGpu]() {
            if (aGradNode) accumulate_into(aGradNode->grad.get(), selfGrad, total, onGpu);
            if (bGradNode) accumulate_into(bGradNode->grad.get(), selfGrad, total, onGpu);
        };
    }

    return ret;
}

//matmul: 2D x 2D, or batched 3D x 3D (independent matmul per leading batch index)
template <typename T>
Jensor<T> Jensor<T>::matmul(const Jensor& other, Backend backend) const {
    if (backend == Backend::Naive && dims_.size() == 3 && other.dims_.size() == 3) {
        assert(is_on_gpu_ && other.is_on_gpu_ && "Jensors must be ON GPU to multiply");
        assert(dims_[0] == other.dims_[0] && dims_[2] == other.dims_[1] &&
               "batched matmul: batch size and inner dims must match");

        int batch = dims_[0], M = dims_[1], K = dims_[2], N = other.dims_[2];
        Jensor<T> ret({(uint16_t)batch, (uint16_t)M, (uint16_t)N}, AllocateOnGpu);
        batched_matmul_launch<T>(data(), other.data(), ret.data(), batch, M, N, K);

        if (grad_node_ || other.grad_node_) {
            ret.requires_grad_ = true;
            ret.grad_node_ = make_grad_node<T>(ret.dims_, ret.is_on_gpu_);
            if (grad_node_) ret.grad_node_->parents.push_back(grad_node_);
            if (other.grad_node_) ret.grad_node_->parents.push_back(other.grad_node_);

            T* selfGrad = ret.grad_node_->grad.get();
            auto aGradNode = grad_node_;
            auto bGradNode = other.grad_node_;
            Jensor<T> aCopy = *this;
            Jensor<T> bCopy = other;

            ret.grad_node_->backward_fn = [selfGrad, aGradNode, bGradNode, aCopy, bCopy, batch, M, K, N]() {
                if (aGradNode) {
                    T* bT = nullptr;
                    bT = (T*)pool_alloc(sizeof(T) * (long long)batch * N * K);
                    batched_transpose_launch<T>(bT, bCopy.data(), batch, K, N);

                    T* dA = nullptr;
                    dA = (T*)pool_alloc(sizeof(T) * (long long)batch * M * K);
                    batched_matmul_launch<T>(selfGrad, bT, dA, batch, M, K, N);

                    accumulate_into(aGradNode->grad.get(), dA, (long long)batch * M * K, true);
                    pool_free(bT);
                    pool_free(dA);
                }

                if (bGradNode) {
                    T* aT = nullptr;
                    aT = (T*)pool_alloc(sizeof(T) * (long long)batch * K * M);
                    batched_transpose_launch<T>(aT, aCopy.data(), batch, M, K);

                    T* dB = nullptr;
                    dB = (T*)pool_alloc(sizeof(T) * (long long)batch * K * N);
                    batched_matmul_launch<T>(aT, selfGrad, dB, batch, K, N, M);

                    accumulate_into(bGradNode->grad.get(), dB, (long long)batch * K * N, true);
                    pool_free(aT);
                    pool_free(dB);
                }
            };
        }

        return ret;
    }

    assert((dims_.size() == 2 && other.dims_.size() == 2) && "high dimension matrix mult unsupported");

    switch (backend) {
        case Backend::Naive: {
            assert(is_on_gpu_ && other.is_on_gpu_ && "Jensors must be ON GPU to multiply");

            Jensor<T> ret = is_on_gpu_ ? Jensor<T>({dims_[0], other.dims_[1]}, AllocateOnGpu)
                                        : Jensor<T>({dims_[0], other.dims_[1]}, AllocateOnCpu);
            int M = dims_[0], K = dims_[1], N = other.dims_[1];
            int threadsPerBlock = 256;
            int numBlocks = (M * N + threadsPerBlock - 1) / threadsPerBlock;

            matmul_gpu<<<numBlocks, threadsPerBlock>>>(data(), other.data(), ret.data(), M, N, K);

            if (grad_node_ || other.grad_node_) {
                ret.requires_grad_ = true;
                ret.grad_node_ = make_grad_node<T>(ret.dims_, ret.is_on_gpu_);
                if (grad_node_) ret.grad_node_->parents.push_back(grad_node_);
                if (other.grad_node_) ret.grad_node_->parents.push_back(other.grad_node_);

                T* selfGrad = ret.grad_node_->grad.get();
                auto aGradNode = grad_node_;
                auto bGradNode = other.grad_node_;
                Jensor<T> aCopy = *this;
                Jensor<T> bCopy = other;

                ret.grad_node_->backward_fn = [selfGrad, aGradNode, bGradNode, aCopy, bCopy, M, K, N]() {
                    int threadsPerBlock = 256;

                    if (aGradNode) {
                        T* bT = nullptr;
                        bT = (T*)pool_alloc(sizeof(T) * N * K);
                        int nb = (K * N + threadsPerBlock - 1) / threadsPerBlock;
                        mytorch::transpose<T><<<nb, threadsPerBlock>>>(bT, bCopy.data(), K, N);

                        T* dA = nullptr;
                        dA = (T*)pool_alloc(sizeof(T) * M * K);
                        int nb2 = (M * K + threadsPerBlock - 1) / threadsPerBlock;
                        matmul_gpu<T><<<nb2, threadsPerBlock>>>(selfGrad, bT, dA, M, K, N);

                        accumulate_into(aGradNode->grad.get(), dA, (long long)M * K, true);
                        pool_free(bT);
                        pool_free(dA);
                    }

                    if (bGradNode) {
                        T* aT = nullptr;
                        aT = (T*)pool_alloc(sizeof(T) * K * M);
                        int nb = (M * K + threadsPerBlock - 1) / threadsPerBlock;
                        mytorch::transpose<T><<<nb, threadsPerBlock>>>(aT, aCopy.data(), M, K);

                        T* dB = nullptr;
                        dB = (T*)pool_alloc(sizeof(T) * K * N);
                        int nb2 = (K * N + threadsPerBlock - 1) / threadsPerBlock;
                        matmul_gpu<T><<<nb2, threadsPerBlock>>>(aT, selfGrad, dB, K, N, M);

                        accumulate_into(bGradNode->grad.get(), dB, (long long)K * N, true);
                        pool_free(aT);
                        pool_free(dB);
                    }
                };
            }

            return ret;
        }

        case Backend::CuBLAS: {
            assert(is_on_gpu_ && other.is_on_gpu_ &&
                   "cuBLAS matmul requires both operands to already be on the GPU");

            uint16_t M = dims_[0];
            uint16_t K = dims_[1];
            uint16_t N = other.dims_[1];
            assert(K == other.dims_[0] && "inner dims must match: A's columns == B's rows");

            Jensor<T> result({M, N}, AllocateOnGpu);

            cublasHandle_t handle;
            cublasCreate(&handle);

            const float alpha = 1.0f;
            const float beta = 0.0f;

            cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        other.data(), N,   // reinterpreted as B^T (N x K)
                        this->data(), K,   // reinterpreted as A^T (K x M)
                        &beta,
                        result.data(), N); // writes C^T (N x M) == C row-major (M x N)

            cublasDestroy(handle);
            return result;
        }
    }

    return *this; 
}

template <typename T>
void Jensor<T>::move_device() {
    T* newRaw = nullptr;
    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    
    cudaMemcpyKind type = is_on_gpu_ ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

    cudaError_t e = cudaMemcpy(newRaw, buf_.get(), total, type);
    if(e) {
        throw e;
    }

    if(is_on_gpu_)
        buf_ = std::shared_ptr<T[]>(newRaw, CudaDeleter<T>());
    else
        buf_ = std::shared_ptr<T[]>(newRaw);
}

template <typename T>
Jensor<T>::Jensor() = default;

template <typename T>
Jensor<T>::Jensor(std::initializer_list<uint16_t> shape, AllocateOnCpu_t)
    : is_on_gpu_(false), dims_(shape), offset_(0) {
    assert(dims_.size() > 0 && dims_.size() <= 255 && "Jensor dimensions must be between 1 and 255");

    stride.resize(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), stride.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());
    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    T* raw = new T[total];
    std::fill(raw, raw + total, T(0));
    buf_ = std::shared_ptr<T[]>(raw);
}

template <typename T>
Jensor<T>::Jensor(std::initializer_list<uint16_t> shape, AllocateOnGpu_t)
    : is_on_gpu_(true), dims_(shape), offset_(0) {
    assert(dims_.size() > 0 && dims_.size() <= 255 && "Jensor dimensions must be between 1 and 255");

    stride.resize(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), stride.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());
    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    T* raw = nullptr;
    raw = (T*)pool_alloc(sizeof(T) * total);
    cudaMemset(raw, 0, sizeof(T) * total);
    buf_ = std::shared_ptr<T[]>(raw, CudaDeleter<T>());
}

template <typename T>
Jensor<T>::Jensor(const std::vector<uint16_t>& shape, AllocateOnCpu_t)
    : is_on_gpu_(false), dims_(shape), offset_(0) {
    assert(dims_.size() > 0 && dims_.size() <= 255 && "Jensor dimensions must be between 1 and 255");

    stride.resize(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), stride.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());
    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    T* raw = new T[total];
    std::fill(raw, raw + total, T(0));
    buf_ = std::shared_ptr<T[]>(raw);
}

template <typename T>
Jensor<T>::Jensor(const std::vector<uint16_t>& shape, AllocateOnGpu_t)
    : is_on_gpu_(true), dims_(shape), offset_(0) {
    assert(dims_.size() > 0 && dims_.size() <= 255 && "Jensor dimensions must be between 1 and 255");

    stride.resize(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), stride.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());
    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    T* raw = nullptr;
    raw = (T*)pool_alloc(sizeof(T) * total);
    cudaMemset(raw, 0, sizeof(T) * total);
    buf_ = std::shared_ptr<T[]>(raw, CudaDeleter<T>());
}

template <typename T>
Jensor<T> Jensor<T>::operator[](uint16_t idx) const {
    assert(!dims_.empty() && "cannot index a 0-dimensional Jensor");

    std::vector<uint16_t> new_dims(dims_.begin() + 1, dims_.end());
    std::vector<uint16_t> new_stride(stride.begin() + 1, stride.end());

    Jensor view;
    view.is_on_gpu_ = is_on_gpu_;
    view.dims_ = std::move(new_dims);
    view.stride = std::move(new_stride);
    view.buf_ = buf_;
    view.offset_ = offset_ + static_cast<long long>(idx) * stride[0];
    return view;
}

//concats this tensor with other along dim
template <typename T>
Jensor<T> Jensor<T>::concat(const Jensor& other, uint8_t dim) {
    assert(dims_.size() == other.dims_.size() && "concat requires matching rank");
    assert(is_on_gpu_ == other.is_on_gpu_ && "both Jensors must be on the same device to concat");
    for (size_t i = 0; i < dims_.size(); ++i) {
        assert((i == dim || dims_[i] == other.dims_[i]) && "concat requires matching dims outside the concat axis");
    }

    std::vector<uint16_t> new_dims = dims_;
    new_dims[dim] = dims_[dim] + other.dims_[dim];

    long long outerSize = std::accumulate(dims_.begin(), dims_.begin() + dim, 1LL, std::multiplies<long long>());
    long long innerSize = std::accumulate(dims_.begin() + dim + 1, dims_.end(), 1LL, std::multiplies<long long>());
    long long thisChunk = static_cast<long long>(dims_[dim]) * innerSize;
    long long otherChunk = static_cast<long long>(other.dims_[dim]) * innerSize;
    long long newChunk = thisChunk + otherChunk;
    long long total = outerSize * newChunk;

    Jensor<T> ret;
    ret.is_on_gpu_ = is_on_gpu_;
    ret.dims_ = new_dims;
    ret.stride.resize(new_dims.size());
    std::exclusive_scan(new_dims.rbegin(), new_dims.rend(), ret.stride.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());

    if (is_on_gpu_) {
        T* raw = nullptr;
        raw = (T*)pool_alloc(sizeof(T) * total);
        ret.buf_ = std::shared_ptr<T[]>(raw, CudaDeleter<T>());
        for (long long o = 0; o < outerSize; ++o) {
            cudaMemcpy(raw + o * newChunk, data() + o * thisChunk, sizeof(T) * thisChunk, cudaMemcpyDeviceToDevice);
            cudaMemcpy(raw + o * newChunk + thisChunk, other.data() + o * otherChunk, sizeof(T) * otherChunk, cudaMemcpyDeviceToDevice);
        }
    } else {
        T* raw = new T[total];
        ret.buf_ = std::shared_ptr<T[]>(raw);
        for (long long o = 0; o < outerSize; ++o) {
            std::copy(data() + o * thisChunk, data() + o * thisChunk + thisChunk, raw + o * newChunk);
            std::copy(other.data() + o * otherChunk, other.data() + o * otherChunk + otherChunk, raw + o * newChunk + thisChunk);
        }
    }

    if (grad_node_ || other.grad_node_) {
        ret.requires_grad_ = true;
        ret.grad_node_ = make_grad_node<T>(ret.dims_, ret.is_on_gpu_);
        if (grad_node_) ret.grad_node_->parents.push_back(grad_node_);
        if (other.grad_node_) ret.grad_node_->parents.push_back(other.grad_node_);

        T* selfGrad = ret.grad_node_->grad.get();
        auto aGradNode = grad_node_;
        auto bGradNode = other.grad_node_;
        bool onGpu = ret.is_on_gpu_;

        ret.grad_node_->backward_fn = [selfGrad, aGradNode, bGradNode, onGpu, outerSize, thisChunk, otherChunk, newChunk]() {
            for (long long o = 0; o < outerSize; ++o) {
                if (aGradNode) accumulate_into(aGradNode->grad.get() + o * thisChunk, selfGrad + o * newChunk, thisChunk, onGpu);
                if (bGradNode) accumulate_into(bGradNode->grad.get() + o * otherChunk, selfGrad + o * newChunk + thisChunk, otherChunk, onGpu);
            }
        };
    }

    return ret;
}

template <typename T>
Jensor<T> Jensor<T>::reshape(std::initializer_list<uint16_t> new_shape) {
    long long old_total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    long long new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1LL, std::multiplies<long long>());
    assert(old_total == new_total && "reshape doesnt add new elements");

    std::vector<uint16_t> canonical(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), canonical.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());
    assert(canonical == stride && "failed recheck");

    dims_ = new_shape;
    stride.resize(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), stride.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());

    return *this;
}

template <typename T>
Jensor<T> Jensor<T>::reshape(const std::vector<uint16_t>& new_shape) {
    long long old_total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    long long new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1LL, std::multiplies<long long>());
    assert(old_total == new_total && "reshape doesnt add new elements");

    std::vector<uint16_t> canonical(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), canonical.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());
    assert(canonical == stride && "failed recheck");

    dims_ = new_shape;
    stride.resize(dims_.size());
    std::exclusive_scan(dims_.rbegin(), dims_.rend(), stride.rbegin(), uint16_t{1}, std::multiplies<uint16_t>());

    return *this;
}

template <typename T>
const std::vector<uint16_t>& Jensor<T>::shape() const { return dims_; }

template <typename T>
bool Jensor<T>::on_gpu() const { return is_on_gpu_; }

template <typename T>
T* Jensor<T>::data() { return buf_.get() + offset_; }

template <typename T>
const T* Jensor<T>::data() const { return buf_.get() + offset_; }

template <typename T>
void Jensor<T>::requires_grad(bool flag) {
    requires_grad_ = flag;
    if (flag && !grad_node_) grad_node_ = make_grad_node<T>(dims_, is_on_gpu_);
}

template <typename T>
bool Jensor<T>::requires_grad() const { return requires_grad_; }

template <typename T>
T* Jensor<T>::grad() { return grad_node_ ? grad_node_->grad.get() : nullptr; }

template <typename T>
void Jensor<T>::zero_grad() {
    if (!grad_node_) return;
    long long n = std::accumulate(grad_node_->dims.begin(), grad_node_->dims.end(), 1LL, std::multiplies<long long>());
    if (grad_node_->is_on_gpu) cudaMemset(grad_node_->grad.get(), 0, sizeof(T) * n);
    else std::fill(grad_node_->grad.get(), grad_node_->grad.get() + n, T(0));
}

template <typename T>
void Jensor<T>::backward() {
    assert(grad_node_ && "backward() called on a Jensor with no grad node; call requires_grad(true) on a leaf first");

    long long n = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    if (is_on_gpu_) {
        std::vector<T> ones(n, T(1));
        cudaMemcpy(grad_node_->grad.get(), ones.data(), sizeof(T) * n, cudaMemcpyHostToDevice);
    } else {
        std::fill(grad_node_->grad.get(), grad_node_->grad.get() + n, T(1));
    }

    std::vector<std::shared_ptr<GradNode<T>>> order;
    std::unordered_set<GradNode<T>*> visited;
    std::function<void(const std::shared_ptr<GradNode<T>>&)> dfs = [&](const std::shared_ptr<GradNode<T>>& node) {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get());
        for (auto& p : node->parents) dfs(p);
        order.push_back(node);
    };
    dfs(grad_node_);

    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
}

template <typename T>
std::shared_ptr<GradNode<T>> Jensor<T>::grad_node() const { return grad_node_; }

template <typename T>
void Jensor<T>::set_grad_node(std::shared_ptr<GradNode<T>> node) {
    grad_node_ = std::move(node);
    requires_grad_ = (bool)grad_node_;
}

template <typename T>
Jensor<T> Jensor<T>::detach() const {
    Jensor<T> copy = *this;
    copy.grad_node_ = nullptr;
    copy.requires_grad_ = false;
    return copy;
}

template struct CudaDeleter<float>;
template class Jensor<float>;
template std::shared_ptr<GradNode<float>> make_grad_node<float>(const std::vector<uint16_t>&, bool);
template void accumulate_into<float>(float*, const float*, long long, bool);
template void batched_transpose_launch<float>(float*, const float*, int, int, int);
template void batched_matmul_launch<float>(const float*, const float*, float*, int, int, int, int);

}  
