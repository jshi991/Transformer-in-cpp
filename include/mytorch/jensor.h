// JENSOR is shorthand for Justin Tensor — the effective equivalent of PyTorch's tensors.
#ifndef JENSOR_H
#define JENSOR_H

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

namespace mytorch {

// PUBLIC TAG TYPES — pick the device at construction time.
struct AllocateOnCpu_t { explicit AllocateOnCpu_t() = default; };
struct AllocateOnGpu_t { explicit AllocateOnGpu_t() = default; };

inline constexpr AllocateOnCpu_t AllocateOnCpu{};
inline constexpr AllocateOnGpu_t AllocateOnGpu{};

template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const;
};

template <typename T>
struct GradNode {
    std::vector<std::shared_ptr<GradNode<T>>> parents;
    std::function<void()> backward_fn;
    std::shared_ptr<T[]> grad;
    std::vector<uint16_t> dims;
    bool is_on_gpu = false;
};

template <typename T>
class Jensor {
    static_assert(std::is_arithmetic<T>::value, "Jensor element type must be arithmetic");

    public:
        Jensor(std::initializer_list<uint16_t> shape, AllocateOnCpu_t);
        Jensor(std::initializer_list<uint16_t> shape, AllocateOnGpu_t);
        Jensor(const std::vector<uint16_t>& shape, AllocateOnCpu_t);
        Jensor(const std::vector<uint16_t>& shape, AllocateOnGpu_t);

        enum class Backend { Naive, CuBLAS };
        void transpose();
        Jensor matmul(const Jensor& other, Backend backend = Backend::Naive) const;

        Jensor operator+(const Jensor& other) const;
        Jensor operator[](uint16_t idx) const;

        Jensor concat(const Jensor& other, uint8_t dim);
        Jensor reshape(std::initializer_list<uint16_t> new_shape);
        Jensor reshape(const std::vector<uint16_t>& new_shape);
        void move_device();

        const std::vector<uint16_t>& shape() const;
        bool on_gpu() const;
        T* data();
        const T* data() const;

        void requires_grad(bool flag);
        bool requires_grad() const;
        T* grad();
        void backward();
        void zero_grad();

        std::shared_ptr<GradNode<T>> grad_node() const;
        void set_grad_node(std::shared_ptr<GradNode<T>> node);
        Jensor detach() const;

    private:
        Jensor();

        bool is_on_gpu_ = false;
        std::vector<uint16_t> dims_;
        std::vector<uint16_t> stride;
        long long offset_ = 0;
        std::shared_ptr<T[]> buf_;

        bool requires_grad_ = false;
        std::shared_ptr<GradNode<T>> grad_node_;
};

extern template class Jensor<float>;

template <typename T>
std::shared_ptr<GradNode<T>> make_grad_node(const std::vector<uint16_t>& dims, bool on_gpu);

template <typename T>
void accumulate_into(T* dst, const T* src, long long n, bool on_gpu);

// out (batch, cols, rows) <- transpose of in (batch, rows, cols), last two dims only.
template <typename T>
void batched_transpose_launch(T* out, const T* in, int batch, int rows, int cols);

// c (batch, M, N) <- a (batch, M, K) @ b (batch, K, N), independent matmul per batch.
template <typename T>
void batched_matmul_launch(const T* a, const T* b, T* c, int batch, int M, int N, int K);

extern template std::shared_ptr<GradNode<float>> make_grad_node<float>(const std::vector<uint16_t>&, bool);
extern template void accumulate_into<float>(float*, const float*, long long, bool);
extern template void batched_transpose_launch<float>(float*, const float*, int, int, int);
extern template void batched_matmul_launch<float>(const float*, const float*, float*, int, int, int, int);

}
#endif  // JENSOR_H
