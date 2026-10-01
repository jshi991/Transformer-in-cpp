#include "quantize/awq.h"

#include "dataset/common.h"
#include "mytorch/checkpoint.h"
#include "mytorch/ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

namespace {

// Evaluated on the same fixed set of windows both before and after
// quantization, so the comparison isolates the quantization effect rather
// than sampling variance across the corpus.
float eval_loss(mytorch::DecoderOnlyTransformer& model, const std::vector<int32_t>& ids,
                 const std::vector<size_t>& windowStarts, uint16_t seqLen) {
    float total = 0.f;
    for (size_t start : windowStarts) {
        std::vector<int32_t> input(ids.begin() + start, ids.begin() + start + seqLen);
        std::vector<int32_t> target(ids.begin() + start + 1, ids.begin() + start + seqLen + 1);

        auto logits = model.forward(input);
        auto loss = mytorch::cross_entropy(logits, target);
        float v = 0.f;
        cudaMemcpy(&v, loss.data(), sizeof(float), cudaMemcpyDeviceToHost);
        total += v;
    }
    return total / windowStarts.size();
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpusPath = argc > 1 ? argv[1] : app::kDefaultCorpusPath;
    std::string inputCheckpoint = argc > 2 ? argv[2] : "build/checkpoint.bin";
    std::string outputCheckpoint = argc > 3 ? argv[3] : "build/checkpoint_awq.bin";
    int bits = argc > 4 ? std::atoi(argv[4]) : 8;
    int groupSize = argc > 5 ? std::atoi(argv[5]) : 64;

    app::Corpus corpus = app::load_corpus(corpusPath);
    const auto& tok = corpus.tok;
    const auto& ids = corpus.ids;
    uint16_t seqLen = app::kSeqLen;

    mytorch::DecoderOnlyTransformer model = app::make_model((uint16_t)tok.vocab_size());
    mytorch::load_checkpoint(inputCheckpoint, model.parameters());
    std::printf("loaded %s (vocab=%zu, bits=int%d, group_size=%d)\n",
                inputCheckpoint.c_str(), tok.vocab_size(), bits, groupSize);

    std::vector<mytorch::Linear*> linears = model.linears();
    std::vector<quantize::CalibrationRecorder> recorders;
    recorders.reserve(linears.size());
    for (auto* lin : linears) recorders.emplace_back(lin->weight().shape()[0]);
    for (size_t i = 0; i < linears.size(); ++i) linears[i]->set_observer(&recorders[i]);

    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> pick(0, ids.size() - seqLen - 2);
    int numCalibBatches = 8;
    for (int b = 0; b < numCalibBatches; ++b) {
        size_t start = pick(rng);
        std::vector<int32_t> input(ids.begin() + start, ids.begin() + start + seqLen);
        model.forward(input);
    }
    for (auto* lin : linears) lin->set_observer(nullptr);
    std::printf("calibration done (%d batches, %zu Linear layers)\n", numCalibBatches, linears.size());

    std::vector<size_t> evalWindows;
    for (int i = 0; i < 8; ++i) evalWindows.push_back(pick(rng));
    float lossBefore = eval_loss(model, ids, evalWindows, seqLen);

    size_t totalOrigBytes = 0, totalPackedBytes = 0;
    double worstRelErr = 0.0;
    std::vector<quantize::QuantizedWeight> quantizedWeights;
    for (size_t i = 0; i < linears.size(); ++i) {
        mytorch::Jensor<float>& w = linears[i]->weight();
        int rows = w.shape()[0], cols = w.shape()[1];
        std::vector<float> hostW((size_t)rows * cols);
        cudaMemcpy(hostW.data(), w.data(), sizeof(float) * hostW.size(), cudaMemcpyDeviceToHost);

        quantize::QuantizedWeight qw = quantize::awq_quantize(hostW, rows, cols, recorders[i], bits, groupSize);
        std::vector<float> dequant = quantize::awq_dequantize(qw);
        cudaMemcpy(w.data(), dequant.data(), sizeof(float) * dequant.size(), cudaMemcpyHostToDevice);

        double relErr = quantize::relative_error(hostW, dequant);
        worstRelErr = std::max(worstRelErr, relErr);
        totalOrigBytes += hostW.size() * sizeof(float);
        totalPackedBytes += quantize::packed_size_bytes(qw);
        std::printf("  linear %2zu: %4dx%-4d  rel error %.3f%%\n", i, rows, cols, relErr * 100.0);
        quantizedWeights.push_back(std::move(qw));
    }

    float lossAfter = eval_loss(model, ids, evalWindows, seqLen);

    std::printf("\nloss before quantization: %.4f\n", lossBefore);
    std::printf("loss after  quantization: %.4f (int%d, group_size=%d, dequantized fp32)\n", lossAfter, bits, groupSize);
    std::printf("worst-layer relative weight error: %.3f%%\n", worstRelErr * 100.0);
    std::printf("weight bytes: %zu -> %zu (%.2fx smaller)\n", totalOrigBytes, totalPackedBytes,
                (double)totalOrigBytes / (double)totalPackedBytes);

    mytorch::save_checkpoint(outputCheckpoint, model.parameters());
    std::printf("saved quantized (dequantized fp32) checkpoint to %s\n", outputCheckpoint.c_str());

    if (bits == 8) {
        mytorch::DecoderOnlyTransformer kernelModel = app::make_model((uint16_t)tok.vocab_size());
        mytorch::load_checkpoint(inputCheckpoint, kernelModel.parameters());
        std::vector<mytorch::Linear*> kernelLinears = kernelModel.linears();
        for (size_t i = 0; i < kernelLinears.size(); ++i) {
            const auto& qw = quantizedWeights[i];
            kernelLinears[i]->load_int8_weights(qw.codes, qw.groupScale, qw.channelScale, qw.groupSize);
        }

        float lossKernel = eval_loss(kernelModel, ids, evalWindows, seqLen);
        std::printf("loss via real int8 GEMM kernel: %.4f (matches dequantized fp32 math: %s)\n",
                    lossKernel, std::fabs(lossKernel - lossAfter) < 1e-2f ? "yes" : "NO");
    }

    return 0;
}
