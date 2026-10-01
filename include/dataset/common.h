#ifndef APP_COMMON_H
#define APP_COMMON_H

#include "dataset/dataset.h"
#include "mytorch/model.h"

#include <cuda_runtime.h>
#include <string>
#include <utility>
#include <vector>

namespace app {

constexpr char kDefaultCorpusPath[] = "src/database/dataset/wikitext-2-raw-train.txt";

constexpr uint16_t kSeqLen = 128;
constexpr uint16_t kNumHeads = 8;
constexpr uint16_t kFeedForwardDim = 2048;
constexpr uint16_t kNumLayers = 5;

struct Corpus {
    std::string text;
    database::CharTokenizer tok;
    std::vector<int32_t> ids;

    Corpus(std::string t, database::CharTokenizer tokenizer, std::vector<int32_t> i)
        : text(std::move(t)), tok(std::move(tokenizer)), ids(std::move(i)) {}
};

inline Corpus load_corpus(const std::string& path) {
    std::string text = database::load_dataset_text(path);
    database::CharTokenizer tok(text);
    std::vector<int32_t> ids = tok.encode(text);
    return Corpus(std::move(text), std::move(tok), std::move(ids));
}

inline mytorch::DecoderOnlyTransformer make_model(uint16_t vocab_size) {
    return mytorch::DecoderOnlyTransformer(vocab_size, mytorch::kModelDim, kNumHeads, kFeedForwardDim, kNumLayers);
}

inline std::vector<int32_t> greedy_generate(mytorch::DecoderOnlyTransformer& model, const database::CharTokenizer& tok,
                                             std::vector<int32_t> context, int numNewTokens, uint16_t seqLen = kSeqLen) {
    int vocab = (int)tok.vocab_size();
    std::vector<float> lastLogits(vocab);

    for (int step = 0; step < numNewTokens; ++step) {
        std::vector<int32_t> window = context.size() > seqLen
            ? std::vector<int32_t>(context.end() - seqLen, context.end())
            : context;

        mytorch::Jensor<float> logits = model.forward(window);
        int lastRow = (int)window.size() - 1;
        cudaMemcpy(lastLogits.data(), logits.data() + (long long)lastRow * vocab,
                   sizeof(float) * vocab, cudaMemcpyDeviceToHost);

        int best = 0;
        for (int c = 1; c < vocab; ++c)
            if (lastLogits[c] > lastLogits[best]) best = c;
        context.push_back(best);
    }

    return context;
}

// Same greedy decoding as greedy_generate, but each step costs O(1) work via
// a per-layer KV cache instead of recomputing the whole context window.
inline std::vector<int32_t> kv_cache_generate(mytorch::DecoderOnlyTransformer& model, const database::CharTokenizer& tok,
                                               std::vector<int32_t> context, int numNewTokens) {
    mytorch::KVCache cache;
    int vocab = (int)tok.vocab_size();
    std::vector<float> lastLogits(vocab);

    auto argmax = [&](mytorch::Jensor<float>& logits) {
        cudaMemcpy(lastLogits.data(), logits.data(), sizeof(float) * vocab, cudaMemcpyDeviceToHost);
        int best = 0;
        for (int c = 1; c < vocab; ++c)
            if (lastLogits[c] > lastLogits[best]) best = c;
        return best;
    };

    int next = 0;
    for (size_t i = 0; i < context.size(); ++i) {
        mytorch::Jensor<float> logits = model.forward_incremental(context[i], (uint16_t)i, cache);
        if (i + 1 == context.size()) next = argmax(logits);
    }

    for (int step = 0; step < numNewTokens; ++step) {
        context.push_back(next);
        mytorch::Jensor<float> logits = model.forward_incremental(next, (uint16_t)(context.size() - 1), cache);
        next = argmax(logits);
    }

    return context;
}

}  // namespace app
#endif  // APP_COMMON_H
