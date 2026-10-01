#ifndef MYTORCH_LAYERNORM_H
#define MYTORCH_LAYERNORM_H

#include "mytorch/jensor.h"

#include <vector>

namespace mytorch {

class LayerNorm {
    public:
        explicit LayerNorm(uint16_t d_model, float eps = 1e-5f);

        Jensor<float> forward(const Jensor<float>& x);

        Jensor<float>& gamma();
        Jensor<float>& beta();
        std::vector<Jensor<float>*> parameters();

    private:
        uint16_t d_model_;
        float eps_;
        Jensor<float> gamma_;
        Jensor<float> beta_;
};

}  // namespace mytorch
#endif  // MYTORCH_LAYERNORM_H
