#include "mytorch/attention.h"

#include "mytorch/ops.h"

#include <cassert>
#include <cmath>

namespace mytorch {

MultiHeadSelfAttention::MultiHeadSelfAttention(uint16_t d_model, uint16_t num_heads)
    : d_model_(d_model), num_heads_(num_heads), head_dim_(d_model / num_heads),
      q_proj_(d_model, d_model), k_proj_(d_model, d_model),
      v_proj_(d_model, d_model), o_proj_(d_model, d_model) {
    assert(d_model % num_heads == 0 && "d_model must be divisible by num_heads");
}

Jensor<float> MultiHeadSelfAttention::forward(const Jensor<float>& x) {
    Jensor<float> Q = q_proj_.forward(x);
    Jensor<float> K = k_proj_.forward(x);
    Jensor<float> V = v_proj_.forward(x);

    float invSqrtDk = 1.0f / sqrtf((float)head_dim_);

    auto head_output = [&](uint16_t h) {
        Jensor<float> Qh = col_slice(Q, h * head_dim_, head_dim_);
        Jensor<float> Kh = col_slice(K, h * head_dim_, head_dim_);
        Jensor<float> Vh = col_slice(V, h * head_dim_, head_dim_);

        Jensor<float> scores = scale(Qh.matmul(transposed(Kh)), invSqrtDk);
        Jensor<float> probs = causal_softmax(scores);
        return probs.matmul(Vh);
    };

    uint8_t lastDim = (uint8_t)(x.shape().size() - 1);
    Jensor<float> merged = head_output(0);
    for (uint16_t h = 1; h < num_heads_; ++h) merged = merged.concat(head_output(h), lastDim);

    return o_proj_.forward(merged);
}

Jensor<float> MultiHeadSelfAttention::forward_incremental(const Jensor<float>& xNew, std::optional<Jensor<float>>& kCache,
                                                           std::optional<Jensor<float>>& vCache) {
    Jensor<float> Qn = q_proj_.forward(xNew);
    Jensor<float> Kn = k_proj_.forward(xNew);
    Jensor<float> Vn = v_proj_.forward(xNew);

    if (!kCache) {
        kCache = Kn.detach();
        vCache = Vn.detach();
    } else {
        kCache = kCache->concat(Kn, 0).detach();
        vCache = vCache->concat(Vn, 0).detach();
    }

    float invSqrtDk = 1.0f / sqrtf((float)head_dim_);

    auto head_output = [&](uint16_t h) {
        Jensor<float> Qh = col_slice(Qn, h * head_dim_, head_dim_);
        Jensor<float> Kh = col_slice(*kCache, h * head_dim_, head_dim_);
        Jensor<float> Vh = col_slice(*vCache, h * head_dim_, head_dim_);

        Jensor<float> scores = scale(Qh.matmul(transposed(Kh)), invSqrtDk);
        Jensor<float> probs = softmax_rows(scores);
        return probs.matmul(Vh);
    };

    Jensor<float> merged = head_output(0);
    for (uint16_t h = 1; h < num_heads_; ++h) merged = merged.concat(head_output(h), 1);

    return o_proj_.forward(merged);
}

std::vector<Jensor<float>*> MultiHeadSelfAttention::parameters() {
    std::vector<Jensor<float>*> params;
    for (Linear* proj : {&q_proj_, &k_proj_, &v_proj_, &o_proj_}) {
        auto p = proj->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

std::vector<Linear*> MultiHeadSelfAttention::linears() {
    return {&q_proj_, &k_proj_, &v_proj_, &o_proj_};
}

}  // namespace mytorch
