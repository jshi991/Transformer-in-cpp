#include "include/mytorch/jensor.h"
#include <cassert>
#include <cstring>


template <typename T>
Jensor<T>::Jensor(std::initializer_list<uint16_t> args) : dims(args.size()) {
    static_assert(std::is_arithmetic<uint16_t>::value, "Jensor can only be constructed from arithmetic types");
    assert(args.size() > 0 && uint8_t(args.size()) <= uint8_t(255) && "Jensor dimensions must be between 1 and 255");
    
    long long TOTAL_SIZE = std::accumulate(args.begin(), args.end(), 1, std::multiplies<uint16_t>());
    arr = std::make_unique<T[]>(TOTAL_SIZE);
    std::fill(arr.get(), arr.get() + TOTAL_SIZE, T(0));
}
