#include "dataset/common.h"
#include "mytorch/checkpoint.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string corpusPath = argc > 1 ? argv[1] : app::kDefaultCorpusPath;
    std::string checkpointPath = argc > 2 ? argv[2] : "build/checkpoint_awq.bin";
    int numTokens = argc > 3 ? std::atoi(argv[3]) : 100;

    app::Corpus corpus = app::load_corpus(corpusPath);
    uint16_t seqLen = app::kSeqLen;

    mytorch::DecoderOnlyTransformer model = app::make_model((uint16_t)corpus.tok.vocab_size());
    mytorch::load_checkpoint(checkpointPath, model.parameters());
    std::printf("loaded %s\n", checkpointPath.c_str());

    std::vector<int32_t> context = corpus.tok.encode(corpus.text.substr(0, seqLen));
    context = app::kv_cache_generate(model, corpus.tok, context, numTokens);

    std::printf("--- generated (quantized model) ---\n%s\n", corpus.tok.decode(context).c_str());
    return 0;
}
