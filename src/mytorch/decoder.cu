#include "mytorch/decoder.h"

#include "mytorch/ops.h"

namespace mytorch {

DecoderLayer::DecoderLayer(uint16_t d_model, uint16_t num_heads, uint16_t d_ff)
    : ln1_(d_model), ln2_(d_model), attn_(d_model, num_heads),
      ff1_(d_model, d_ff), ff2_(d_ff, d_model) {}

Jensor<float> DecoderLayer::forward(const Jensor<float>& x) {
    Jensor<float> attnOut = attn_.forward(ln1_.forward(x));
    Jensor<float> resid1 = x + attnOut;

    Jensor<float> ffOut = ff2_.forward(relu(ff1_.forward(ln2_.forward(resid1))));
    return resid1 + ffOut;
}

Jensor<float> DecoderLayer::forward_incremental(const Jensor<float>& xNew, std::optional<Jensor<float>>& kCache,
                                                 std::optional<Jensor<float>>& vCache) {
    Jensor<float> attnOut = attn_.forward_incremental(ln1_.forward(xNew), kCache, vCache);
    Jensor<float> resid1 = xNew + attnOut;

    Jensor<float> ffOut = ff2_.forward(relu(ff1_.forward(ln2_.forward(resid1))));
    return resid1 + ffOut;
}

std::vector<Jensor<float>*> DecoderLayer::parameters() {
    std::vector<Jensor<float>*> params;
    auto extend = [&](std::vector<Jensor<float>*> p) { params.insert(params.end(), p.begin(), p.end()); };
    extend(ln1_.parameters());
    extend(attn_.parameters());
    extend(ln2_.parameters());
    extend(ff1_.parameters());
    extend(ff2_.parameters());
    return params;
}

std::vector<Linear*> DecoderLayer::linears() {
    std::vector<Linear*> ls = attn_.linears();
    ls.push_back(&ff1_);
    ls.push_back(&ff2_);
    return ls;
}

}  // namespace mytorch
