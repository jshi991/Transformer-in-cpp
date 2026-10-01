#include "dataset/common.h"
#include "mytorch/checkpoint.h"
#include "mytorch/ops.h"
#include "mytorch/optim.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string corpusPath = argc > 1 ? argv[1] : app::kDefaultCorpusPath;
    int numIters = argc > 2 ? std::atoi(argv[2]) : 1000;
    int batchSize = argc > 3 ? std::atoi(argv[3]) : 4;
    std::string checkpointPath = argc > 4 ? argv[4] : "build/checkpoint.bin";

    app::Corpus corpus = app::load_corpus(corpusPath);
    std::printf("corpus: %zu chars, vocab: %zu, batch_size: %d\n",
                corpus.text.size(), corpus.tok.vocab_size(), batchSize);

    mytorch::DecoderOnlyTransformer model = app::make_model((uint16_t)corpus.tok.vocab_size());
    mytorch::Adam optimizer(model.parameters(), 3e-4f);

    if (std::ifstream(checkpointPath).good()) {
        mytorch::load_checkpoint(checkpointPath, model.parameters());
        std::printf("resumed from %s\n", checkpointPath.c_str());
    }

    const auto& ids = corpus.ids;
    uint16_t seqLen = app::kSeqLen;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> pick(0, ids.size() - seqLen - 2);

    for (int iter = 0; iter < numIters; ++iter) {
        std::vector<int32_t> flatInput(batchSize * seqLen), flatTarget(batchSize * seqLen);
        for (int b = 0; b < batchSize; ++b) {
            size_t start = pick(rng);
            std::copy(ids.begin() + start, ids.begin() + start + seqLen, flatInput.begin() + b * seqLen);
            std::copy(ids.begin() + start + 1, ids.begin() + start + seqLen + 1, flatTarget.begin() + b * seqLen);
        }

        mytorch::Jensor<float> logits = model.forward(flatInput, (uint16_t)batchSize, seqLen);
        mytorch::Jensor<float> loss = mytorch::cross_entropy(logits, flatTarget);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (iter % 10 == 0 || iter == numIters - 1) {
            float lossVal = 0.f;
            cudaMemcpy(&lossVal, loss.data(), sizeof(float), cudaMemcpyDeviceToHost);
            std::printf("iter %d: loss %.4f\n", iter, lossVal);
        }

        if (iter % 50 == 0 && iter > 0) {
            mytorch::save_checkpoint(checkpointPath, model.parameters());
            std::printf("checkpoint saved to %s\n", checkpointPath.c_str());
        }
    }

    mytorch::save_checkpoint(checkpointPath, model.parameters());
    std::printf("checkpoint saved to %s\n", checkpointPath.c_str());

    std::vector<int32_t> context = corpus.tok.encode(corpus.text.substr(0, seqLen));
    context = app::kv_cache_generate(model, corpus.tok, context, 100);
    std::printf("--- generated ---\n%s\n", corpus.tok.decode(context).c_str());
    return 0;
}
