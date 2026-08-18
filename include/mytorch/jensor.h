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

        // Views share storage with their parent (bumps the refcount, no copy);
        // still returns a Jensor (even 0-dimensional) rather than a raw T so
        // that indexing stays an operation you can eventually differentiate through.
        Jensor operator[](uint16_t idx) const;

        Jensor operator+(const Jensor& other) const;
        Jensor operator-(const Jensor& other) const;
        Jensor operator*(const Jensor& other) const;

        Jensor concat(const Jensor& other);
        Jensor reshape(std::initializer_list<uint16_t> new_shape);

        const std::vector<uint16_t>& shape() const;
        bool on_gpu() const;

    private:
        enum class Op { Add, Sub, Mul };
        Jensor();
        Jensor(std::vector<uint16_t> shape, bool gpu);

        Jensor elementwise(const Jensor& other, Op op) const;

        bool is_on_gpu_ = false;
        std::vector<uint16_t> dims_;
        long long offset_ = 0;
        std::shared_ptr<T[]> buf_;
};

extern template class Jensor<float>;

}  // namespace mytorch
#endif  // JENSOR_H
