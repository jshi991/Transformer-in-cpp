#ifndef MYTORCH_OPS_H
#define MYTORCH_OPS_H

#include "mytorch/jensor.h"

#include <cstdint>
#include <vector>

namespace mytorch {

Jensor<float> transposed(const Jensor<float>& x);
Jensor<float> scale(const Jensor<float>& x, float factor);
Jensor<float> relu(const Jensor<float>& x);
Jensor<float> col_slice(const Jensor<float>& x, uint16_t start, uint16_t count);
Jensor<float> causal_softmax(const Jensor<float>& scores);
Jensor<float> cross_entropy(const Jensor<float>& logits, const std::vector<int32_t>& targets);

// Plain (non-causal) row-wise softmax, forward-only (no autograd) — used for
// KV-cache incremental decoding, where every cached key is already valid to
// attend to, so no masking is needed.
Jensor<float> softmax_rows(const Jensor<float>& scores);

}  // namespace mytorch
#endif  // MYTORCH_OPS_H
