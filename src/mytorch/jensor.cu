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

//REFACTOR LATER -> naive implementation 
template <typename T>
__global__ void transpose(const T* outPtr, const T* inPtr, int finalRow, int finalCol) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < finalRow && col < finalCol) {
        outPtr[i * finalCol + row] = inPtr[i * finalRow + col];   
    }
}

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
void Jensor<T>::transpose() {
    
}


//matmul is only supported for 2 by 2 case -> refactors may be needed for further support 
template <typename T>
Jensor<T> Jensor<T>::matmul(const Jensor& other, Backend backend) const {
    assert((dims_.size() == 2 && other.dims_.size() == 2) && "high dimension matrix mult unsupported");

    switch (backend) {
        case Backend::Naive: {
            assert((is_on_gpu_ ^ other.is_on_gpu_) && "Jensors must be ON GPU to multiply");

            Jensor<T> ret = is_on_gpu_ ? Jensor<T>({dims_[0], other.dims_[1]}, AllocateOnGpu)
                                        : Jensor<T>({dims_[0], other.dims_[1]}, AllocateOnCpu);
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

            cudaDeviceSynchronize();
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

    cudaError_t e = cudaMemcpy(newRaw, buf_, total, type);
    if(e) {
        throw e;
    }
    
    if(is_on_gpu_) 
        buf_ = std::shared_ptr<T[]>(newRaw, myCudaDeleter<T>());
    else 
        buf_ = std::shared_ptr<T[]>(newRaw);
}

template <typename T>
Jensor<T>::Jensor() = default;

template <typename T>
Jensor<T>::Jensor(std::initializer_list<uint16_t> shape, AllocateOnCpu_t)
    : is_on_gpu_(false), dims_(shape), offset_(0) {
    assert(dims_.size() > 0 && dims_.size() <= 255 && "Jensor dimensions must be between 1 and 255");

    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    T* raw = new T[total];
    std::fill(raw, raw + total, T(0));
    buf_ = std::shared_ptr<T[]>(raw);
}

template <typename T>
Jensor<T>::Jensor(std::initializer_list<uint16_t> shape, AllocateOnGpu_t)
    : is_on_gpu_(true), dims_(shape), offset_(0) {
    assert(dims_.size() > 0 && dims_.size() <= 255 && "Jensor dimensions must be between 1 and 255");

    long long total = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<long long>());
    T* raw = nullptr;
    cudaMalloc(&raw, sizeof(T) * total);
    cudaMemset(raw, 0, sizeof(T) * total);
    buf_ = std::shared_ptr<T[]>(raw, CudaDeleter<T>());
}

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
