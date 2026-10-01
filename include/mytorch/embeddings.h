#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "mytorch/jensor.h"

#include <cstdint>
#include <vector>

namespace mytorch {

constexpr uint16_t kModelDim = 512;

class InputEmbeddings {
    public:
        explicit InputEmbeddings(uint16_t vocab_size, uint16_t d_model = kModelDim);

        Jensor<float> forward(const std::vector<int32_t>& token_ids);
        // flat_ids.size() must equal batch * seq_len; produces shape (batch, seq_len, d_model).
        Jensor<float> forward(const std::vector<int32_t>& flat_ids, uint16_t batch, uint16_t seq_len);
        std::vector<Jensor<float>*> parameters();
        Jensor<float>& weight();

    private:
        uint16_t vocab_size_;
        uint16_t d_model_;
        Jensor<float> weight_;
};

Jensor<float> positional_encoding(uint16_t seq_len, uint16_t d_model = kModelDim);
Jensor<float> positional_encoding(uint16_t seq_len, uint16_t d_model, uint16_t batch);
// Single-row positional encoding at an absolute position, for KV-cache
// incremental decoding (shape (1, d_model)).
Jensor<float> positional_encoding_at(uint16_t position, uint16_t d_model = kModelDim);

}  // namespace mytorch
#endif  // EMBEDDINGS_H
