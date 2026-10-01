#ifndef MYTORCH_MODEL_H
#define MYTORCH_MODEL_H

#include "mytorch/decoder.h"
#include "mytorch/embeddings.h"
#include "mytorch/jensor.h"
#include "mytorch/layernorm.h"
#include "mytorch/linear.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mytorch {

// One entry per decoder layer; grows by one row per forward_incremental() call.
struct KVCache {
    std::vector<std::optional<Jensor<float>>> keys;
    std::vector<std::optional<Jensor<float>>> values;
};

class DecoderOnlyTransformer {
    public:
        DecoderOnlyTransformer(uint16_t vocab_size, uint16_t d_model, uint16_t num_heads,
                               uint16_t d_ff, uint16_t num_layers);

        Jensor<float> forward(const std::vector<int32_t>& token_ids);
        // flat_ids.size() must equal batch * seq_len; produces logits (batch, seq_len, vocab_size).
        Jensor<float> forward(const std::vector<int32_t>& flat_ids, uint16_t batch, uint16_t seq_len);
        // Single-token incremental decode step: O(1) work per step instead of
        // recomputing the whole context, using/growing `cache` in place.
        Jensor<float> forward_incremental(int32_t token_id, uint16_t position, KVCache& cache);
        std::vector<Jensor<float>*> parameters();
        std::vector<Linear*> linears();

    private:
        uint16_t d_model_;
        InputEmbeddings embed_;
        std::vector<DecoderLayer> layers_;
        LayerNorm ln_f_;
};

}  // namespace mytorch
#endif  // MYTORCH_MODEL_H
