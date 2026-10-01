#ifndef MYTORCH_ATTENTION_H
#define MYTORCH_ATTENTION_H

#include "mytorch/jensor.h"
#include "mytorch/linear.h"

#include <optional>

namespace mytorch {

class MultiHeadSelfAttention {
    public:
        MultiHeadSelfAttention(uint16_t d_model, uint16_t num_heads);

        Jensor<float> forward(const Jensor<float>& x);
        // xNew: the single new token's residual stream, shape (1, d_model).
        // kCache/vCache: this layer's running K/V cache (shape (cache_len,
        // d_model)), grown by one row and updated in place.
        Jensor<float> forward_incremental(const Jensor<float>& xNew, std::optional<Jensor<float>>& kCache,
                                           std::optional<Jensor<float>>& vCache);
        std::vector<Jensor<float>*> parameters();
        std::vector<Linear*> linears();

    private:
        uint16_t d_model_;
        uint16_t num_heads_;
        uint16_t head_dim_;
        Linear q_proj_;
        Linear k_proj_;
        Linear v_proj_;
        Linear o_proj_;
};

}  // namespace mytorch
#endif  // MYTORCH_ATTENTION_H
