//JENSOR is shorthand for Justin Tensor this is the effective equivalent of pytorch's tensors
#ifndef JENSOR_H
#define JENSOR_H

#include <concepts>
#include <iostream>

template <typename T>
// concept AllowedTypes = std::same_as<T, float>
//                         || std::same_as<T, int>; //add 
// template <AllowedTypes T>
//NOTE: upgarde to g++ 23+
class Jensor {

    public:
        Jensor()=delete;
        Jensor(std::initializer_list<uint16_t> args);

        Jensor operator[]();
        T operator[](int16_t);

        Jensor operator+(const Jensor&);
        Jensor operator-(const Jensor&);
        Jensor operator*(const Jensor&);

        Jensor concat(const Jensor&);
        Jensor reshape(std::initializer_list<uint16_t> new_shape);
    private:
        std::unique_ptr<T[]> arr;
        uint8_t dims;  
};  

#endif // JENSOR_H