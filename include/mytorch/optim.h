#ifndef MYTORCH_OPTIM_H
#define MYTORCH_OPTIM_H

#include "mytorch/jensor.h"

#include <memory>
#include <vector>

namespace mytorch {

class Adam {
    public:
        explicit Adam(std::vector<Jensor<float>*> params, float lr = 1e-3f,
                      float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

        void step();
        void zero_grad();

    private:
        std::vector<Jensor<float>*> params_;
        float lr_, beta1_, beta2_, eps_;
        int t_ = 0;
        std::vector<std::shared_ptr<float[]>> m_;
        std::vector<std::shared_ptr<float[]>> v_;
};

}  // namespace mytorch
#endif  // MYTORCH_OPTIM_H
