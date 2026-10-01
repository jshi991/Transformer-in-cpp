#ifndef MYTORCH_DECODER_H
#define MYTORCH_DECODER_H

#include "mytorch/attention.h"
#include "mytorch/jensor.h"
#include "mytorch/layernorm.h"
#include "mytorch/linear.h"

namespace mytorch {

class DecoderLayer {
    public:
        DecoderLayer(uint16_t d_model, uint16_t num_heads, uint16_t d_ff);

        Jensor<float> forward(const Jensor<float>& x);
        Jensor<float> forward_incremental(const Jensor<float>& xNew, std::optional<Jensor<float>>& kCache,
                                           std::optional<Jensor<float>>& vCache);
        std::vector<Jensor<float>*> parameters();
        std::vector<Linear*> linears();

    private:
        LayerNorm ln1_;
        LayerNorm ln2_;
        MultiHeadSelfAttention attn_;
        Linear ff1_;
        Linear ff2_;
};

}  // namespace mytorch
#endif  // MYTORCH_DECODER_H
