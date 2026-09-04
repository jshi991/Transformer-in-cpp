#include "mytorch/jensor.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>
#include <numeric>
#include <vector>

namespace mytorch {



template <typename T>
void CudaDeleter<T>::operator()(T* ptr) const { cudaFree(ptr); }


//a M X N 
//b N X V
template <typename T>
__global__ void matmul_gpu(const T* a, const T* b, T* c, int finalRow, int finalCol, int N, int V) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if(row < finalRow && col < finalCol) {
        T totalSum{};
        for(int i = 0; i < innderDim; i++) {
            totalSum += a[row * N + i] * b[i * V + col];
        }

        c[row * V + col] = totalSum;
    }
}

template <typename T>
Jensor<T> Jensor<T>::matmul(const Jensor& other, Backend backend) const {
    assert((dims_.size() == 2 && other.dims_.size() == 2) && "high dimension matrix mult unsupported");

    switch (backend) {
        case Backend::Naive: {
            assert((is_on_gpu_ ^ other.is_on_gpu_) && "Jensors must be ON GPU to multiply");

            Jensor<T> ret([dims_[0], other.dims_[1]], is_on_gpu_ ?? AllocateOnGpu_t : AllocateOnCpu_t);
            if(is_on_gpu_) {
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks((dims_[0] + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (dims_[1] + threadsPerBlock.y - 1) / threadsPerBlock.y);

                matmul_gpu<<<threadsPerBlock, numBlocks>>>(buf_, other.buf_, ret.buf_, ret.dims_[0], ret.dims_[1], dims_[1], other.dims_[1]);
            } else {
                //not yet implemented
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

            Jensor<T> result({M, N}, is_on_gpu_);

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

            cudaDeviceSynchronize();
            cublasDestroy(handle);
            return result;
        }
    }

    return *this; 
}

template <typename T>
void Jensor<T>::move_device() {
//move device need to think about offset situation
}

template <typename T>
__global__ void cuda_add(const T* a, const T* b, T* c, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

template <typename T>
__global__ void cuda_sub(const T* a, const T* b, T* c, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

template <typename T>
__global__ void cuda_mul(const T* a, const T* b, T* c, long long n) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

template <typename T>
Jensor<T>::Jensor() = default;

template <typename T>
Jensor<T>::Jensor(std::initializer_list<uint16_t> shape, AllocateOnCpu_t)
    : Jensor(std::vector<uint16_t>(shape), false) {}

template <typename T>
Jensor<T>::Jensor(std::initializer_list<uint16_t> shape, AllocateOnGpu_t)
    : Jensor(std::vector<uint16_t>(shape), true) {}

template <typename T>
Jensor<T> Jensor<T>::operator[](uint16_t idx) const {
    assert(!dims_.empty() && "cannot index a 0-dimensional Jensor");

    std::vector<uint16_t> new_dims(dims_.begin() + 1, dims_.end());
    long long slice_elems =
        std::accumulate(new_dims.begin(), new_dims.end(), 1LL, std::multiplies<long long>());

    Jensor view;
    view.is_on_gpu_ = is_on_gpu_;
    view.dims_ = std::move(new_dims);
    view.buf_ = buf_;
    view.offset_ = offset_ + static_cast<long long>(idx) * slice_elems;
    return view;
}

template <typename T>
Jensor<T> Jensor<T>::operator+(const Jensor& other) const { return elementwise(other, Op::Add); }

template <typename T>
Jensor<T> Jensor<T>::operator-(const Jensor& other) const { return elementwise(other, Op::Sub); }

template <typename T>
Jensor<T> Jensor<T>::operator*(const Jensor& other) const { return elementwise(other, Op::Mul); }

template <typename T>
Jensor<T> Jensor<T>::concat(const Jensor& other) {
    (void)other;
    assert(false && "Jensor::concat is not implemented yet");
    return *this;
}

template <typename T>
Jensor<T> Jensor<T>::reshape(std::initializer_list<uint16_t> new_shape) {
    (void)new_shape;
    assert(false && "Jensor::reshape is not implemented yet");
    return *this;
}

template <typename T>
Jensor<T>::Jensor(std::vector<uint16_t> shape, bool gpu)
    : is_on_gpu_(gpu), dims_(std::move(shape)), offset_(0) {
    assert(dims_.size() > 0 && dims_.size() <= 255 && "Jensor dimensions must be between 1 and 255");

    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    if (is_on_gpu_) {
        T* raw = nullptr;
        cudaMalloc(&raw, sizeof(T) * total);
        cudaMemset(raw, 0, sizeof(T) * total);
        buf_ = std::shared_ptr<T[]>(raw, CudaDeleter<T>());
    } else {
        T* raw = new T[total];
        std::fill(raw, raw + total, T(0));
        buf_ = std::shared_ptr<T[]>(raw);
    }
}

template <typename T>
Jensor<T> Jensor<T>::elementwise(const Jensor& other, Op op) const {
    assert(dims_ == other.dims_ && "Jensors must be the same shape");
    assert(is_on_gpu_ == other.is_on_gpu_ && "Jensors must be on the same device");

    Jensor result(dims_, is_on_gpu_);
    long long n = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    const T* a = buf_.get() + offset_;
    const T* b = other.buf_.get() + other.offset_;
    T* c = result.buf_.get();

    if (is_on_gpu_) {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        switch (op) {
            case Op::Add: cuda_add<<<blocks, threads>>>(a, b, c, n); break;
            case Op::Sub: cuda_sub<<<blocks, threads>>>(a, b, c, n); break;
            case Op::Mul: cuda_mul<<<blocks, threads>>>(a, b, c, n); break;
        }
        cudaDeviceSynchronize();
    } else {
        for (long long i = 0; i < n; ++i) {
            switch (op) {
                case Op::Add: c[i] = a[i] + b[i]; break;
                case Op::Sub: c[i] = a[i] - b[i]; break;
                case Op::Mul: c[i] = a[i] * b[i]; break;
            }
        }
    }
    return result;
}

template <typename T>
const std::vector<uint16_t>& Jensor<T>::shape() const { return dims_; }

template <typename T>
bool Jensor<T>::on_gpu() const { return is_on_gpu_; }

template <typename T>
T* Jensor<T>::data() { return buf_.get() + offset_; }

template <typename T>
const T* Jensor<T>::data() const { return buf_.get() + offset_; }

template struct CudaDeleter<float>;
template class Jensor<float>;

}  
