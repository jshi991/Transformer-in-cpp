// JENSOR is shorthand for Justin Tensor — the effective equivalent of PyTorch's tensors.
#ifndef JENSOR_H
#define JENSOR_H

#include <cstdint>
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
class Jensor {
    static_assert(std::is_arithmetic<T>::value, "Jensor element type must be arithmetic");

    public:
        Jensor(std::initializer_list<uint16_t> shape, AllocateOnCpu_t);
        Jensor(std::initializer_list<uint16_t> shape, AllocateOnGpu_t);

        enum class Backend { Naive, CuBLAS };
        void transpose();
        Jensor matmul(const Jensor& other, Backend backend = Backend::Naive) const;

        Jensor operator[](uint16_t idx) const;

        Jensor concat(const Jensor& other);
        Jensor reshape(std::initializer_list<uint16_t> new_shape);
        void move_device();

        const std::vector<uint16_t>& shape() const;
        bool on_gpu() const;
        T* data();
        const T* data() const;

    private:
        Jensor();

        bool is_on_gpu_ = false;
        std::vector<uint16_t> dims_;
        long long offset_ = 0;
        std::shared_ptr<T[]> buf_;
};

extern template class Jensor<float>;

}  
#endif  // JENSOR_H
