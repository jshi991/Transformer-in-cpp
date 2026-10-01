#include "mytorch/model.h"

#include "mytorch/ops.h"

namespace mytorch {

namespace {

// Weight-tied output head: logits = h @ embedWeight^T, reusing the input
// embedding matrix instead of a separate lm_head parameter. Mirrors
// Linear::forward's flatten/restore trick since matmul is 2D-only.
Jensor<float> tied_lm_head(const Jensor<float>& h, const Jensor<float>& embedWeight) {
    const auto& hShape = h.shape();
    bool flatten = hShape.size() > 2;
    uint16_t dModel = embedWeight.shape()[1];

    Jensor<float> flatH = h;
    if (flatten) {
        long long rows = 1;
        for (size_t i = 0; i + 1 < hShape.size(); ++i) rows *= hShape[i];
        flatH.reshape({(uint16_t)rows, dModel});
    }

    Jensor<float> logits = flatH.matmul(transposed(embedWeight));

    if (flatten) {
        std::vector<uint16_t> outShape(hShape.begin(), hShape.end() - 1);
        outShape.push_back(logits.shape()[1]);
        logits.reshape(outShape);
    }

    return logits;
}

}  // namespace

DecoderOnlyTransformer::DecoderOnlyTransformer(uint16_t vocab_size, uint16_t d_model, uint16_t num_heads,
                                                uint16_t d_ff, uint16_t num_layers)
    : d_model_(d_model), embed_(vocab_size, d_model), ln_f_(d_model) {
    for (uint16_t i = 0; i < num_layers; ++i) layers_.emplace_back(d_model, num_heads, d_ff);
}

Jensor<float> DecoderOnlyTransformer::forward(const std::vector<int32_t>& token_ids) {
    uint16_t seqLen = static_cast<uint16_t>(token_ids.size());
    Jensor<float> h = embed_.forward(token_ids) + positional_encoding(seqLen, d_model_);

    for (auto& layer : layers_) h = layer.forward(h);

    h = ln_f_.forward(h);
    return tied_lm_head(h, embed_.weight());
}

Jensor<float> DecoderOnlyTransformer::forward(const std::vector<int32_t>& flat_ids, uint16_t batch, uint16_t seq_len) {
    Jensor<float> h = embed_.forward(flat_ids, batch, seq_len) + positional_encoding(seq_len, d_model_, batch);

    for (auto& layer : layers_) h = layer.forward(h);

    h = ln_f_.forward(h);
    return tied_lm_head(h, embed_.weight());
}

Jensor<float> DecoderOnlyTransformer::forward_incremental(int32_t token_id, uint16_t position, KVCache& cache) {
    if (cache.keys.empty()) {
        cache.keys.resize(layers_.size());
        cache.values.resize(layers_.size());
    }

    Jensor<float> h = embed_.forward({token_id}) + positional_encoding_at(position, d_model_);

    for (size_t i = 0; i < layers_.size(); ++i) h = layers_[i].forward_incremental(h, cache.keys[i], cache.values[i]);

    h = ln_f_.forward(h);
    return tied_lm_head(h, embed_.weight());
}

std::vector<Jensor<float>*> DecoderOnlyTransformer::parameters() {
    std::vector<Jensor<float>*> params;
    auto extend = [&](std::vector<Jensor<float>*> p) { params.insert(params.end(), p.begin(), p.end()); };
    extend(embed_.parameters());
    for (auto& layer : layers_) extend(layer.parameters());
    extend(ln_f_.parameters());
    return params;
}

std::vector<Linear*> DecoderOnlyTransformer::linears() {
    std::vector<Linear*> ls;
    for (auto& layer : layers_) {
        auto l = layer.linears();
        ls.insert(ls.end(), l.begin(), l.end());
    }
    return ls;
}

}  // namespace mytorch
