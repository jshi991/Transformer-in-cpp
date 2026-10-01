#include "dataset/common.h"
#include "mytorch/attention.h"
#include "mytorch/checkpoint.h"
#include "mytorch/decoder.h"
#include "mytorch/embeddings.h"
#include "mytorch/jensor.h"
#include "mytorch/layernorm.h"
#include "mytorch/linear.h"
#include "mytorch/model.h"
#include "mytorch/ops.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

using mytorch::Jensor;

static void upload(Jensor<float>& t, const float* host, size_t n) {
    cudaMemcpy(t.data(), host, n * sizeof(float), cudaMemcpyHostToDevice);
}

static void download(const Jensor<float>& t, float* host, size_t n) {
    cudaMemcpy(host, t.data(), n * sizeof(float), cudaMemcpyDeviceToHost);
}

static bool expect(const char* name, float got, float want) {
    bool ok = got == want;
    std::printf("%s: %s (got %f, want %f)\n", name, ok ? "OK" : "FAIL", got, want);
    return ok;
}

static bool expect_close(const char* name, float got, float want, float tol = 1e-3f) {
    bool ok = std::fabs(got - want) < tol;
    std::printf("%s: %s (got %f, want %f)\n", name, ok ? "OK" : "FAIL", got, want);
    return ok;
}

int main() {
    Jensor<float> a({2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> b({2, 2}, mytorch::AllocateOnGpu);

    float aHost[4] = {1, 2, 3, 4};
    float bHost[4] = {1, 1, 1, 1};
    upload(a, aHost, 4);
    upload(b, bHost, 4);

    bool ok = true;

    Jensor<float> sum = a + b;
    float sumHost[4];
    download(sum, sumHost, 4);
    ok &= expect("add", sumHost[0], 2.0f);

    Jensor<float> prod = a.matmul(b);
    float prodHost[4];
    download(prod, prodHost, 4);
    ok &= expect("matmul", prodHost[0], 3.0f);

    a.transpose();
    float aTransHost[4];
    download(a, aTransHost, 4);
    ok &= expect("transpose", aTransHost[1], 3.0f);

    Jensor<float> cat = sum.concat(prod, 0);
    float catHost[8];
    download(cat, catHost, 8);
    ok &= expect("concat", catHost[4], 3.0f);

    Jensor<float> x({2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> y({2, 2}, mytorch::AllocateOnGpu);
    x.requires_grad(true);
    y.requires_grad(true);
    upload(x, aHost, 4);
    upload(y, bHost, 4);

    Jensor<float> xy = x + y;
    xy.backward();
    float xGradHost[4], yGradHost[4];
    cudaMemcpy(xGradHost, x.grad(), sizeof(xGradHost), cudaMemcpyDeviceToHost);
    cudaMemcpy(yGradHost, y.grad(), sizeof(yGradHost), cudaMemcpyDeviceToHost);
    ok &= expect("add grad x", xGradHost[0], 1.0f);
    ok &= expect("add grad y", yGradHost[0], 1.0f);

    Jensor<float> p({2, 3}, mytorch::AllocateOnGpu);
    Jensor<float> q({3, 2}, mytorch::AllocateOnGpu);
    p.requires_grad(true);
    q.requires_grad(true);
    float pHost[6] = {1, 2, 3, 4, 5, 6};
    float qHost[6] = {1, 0, 0, 1, 1, 1};
    upload(p, pHost, 6);
    upload(q, qHost, 6);

    Jensor<float> pq = p.matmul(q);
    float pqHost[4];
    download(pq, pqHost, 4);
    ok &= expect("nonsquare matmul", pqHost[0], 4.0f);

    pq.backward();
    float pGradHost[6], qGradHost[6];
    cudaMemcpy(pGradHost, p.grad(), sizeof(pGradHost), cudaMemcpyDeviceToHost);
    cudaMemcpy(qGradHost, q.grad(), sizeof(qGradHost), cudaMemcpyDeviceToHost);
    ok &= expect("matmul grad p", pGradHost[0], 1.0f);
    ok &= expect("matmul grad q", qGradHost[0], 5.0f);

    Jensor<float> m({2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> n({2, 2}, mytorch::AllocateOnGpu);
    m.requires_grad(true);
    n.requires_grad(true);
    upload(m, aHost, 4);
    upload(n, bHost, 4);

    Jensor<float> mn = m.concat(n, 0);
    mn.backward();
    float mGradHost[4], nGradHost[4];
    cudaMemcpy(mGradHost, m.grad(), sizeof(mGradHost), cudaMemcpyDeviceToHost);
    cudaMemcpy(nGradHost, n.grad(), sizeof(nGradHost), cudaMemcpyDeviceToHost);
    ok &= expect("concat grad m", mGradHost[0], 1.0f);
    ok &= expect("concat grad n", nGradHost[0], 1.0f);

    Jensor<float> bmA({2, 2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> bmB({2, 2, 2}, mytorch::AllocateOnGpu);
    bmA.requires_grad(true);
    bmB.requires_grad(true);
    float bmAHost[8] = {1, 2, 3, 4, 1, 1, 1, 1};
    float bmBHost[8] = {1, 0, 0, 1, 2, 0, 0, 2};
    upload(bmA, bmAHost, 8);
    upload(bmB, bmBHost, 8);

    Jensor<float> bmC = bmA.matmul(bmB);
    float bmCHost[8];
    download(bmC, bmCHost, 8);
    ok &= (bmC.shape()[0] == 2 && bmC.shape()[1] == 2 && bmC.shape()[2] == 2);
    ok &= expect("batched matmul fwd b0", bmCHost[1], 2.0f);
    ok &= expect("batched matmul fwd b1", bmCHost[4], 2.0f);

    bmC.backward();
    float bmAGrad[8], bmBGrad[8];
    cudaMemcpy(bmAGrad, bmA.grad(), sizeof(bmAGrad), cudaMemcpyDeviceToHost);
    cudaMemcpy(bmBGrad, bmB.grad(), sizeof(bmBGrad), cudaMemcpyDeviceToHost);
    ok &= expect("batched matmul grad A b0", bmAGrad[0], 1.0f);
    ok &= expect("batched matmul grad A b1", bmAGrad[4], 2.0f);
    ok &= expect("batched matmul grad B b0", bmBGrad[0], 4.0f);
    ok &= expect("batched matmul grad B b1", bmBGrad[4], 2.0f);

    Jensor<float> x2({2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> y2({2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> w2({2, 2}, mytorch::AllocateOnGpu);
    x2.requires_grad(true);
    y2.requires_grad(true);
    w2.requires_grad(true);
    upload(x2, aHost, 4);
    upload(y2, bHost, 4);
    float identity[4] = {1, 0, 0, 1};
    upload(w2, identity, 4);

    Jensor<float> z = (x2 + y2).matmul(w2);
    z.backward();
    float x2Grad[4], y2Grad[4], w2Grad[4];
    cudaMemcpy(x2Grad, x2.grad(), sizeof(x2Grad), cudaMemcpyDeviceToHost);
    cudaMemcpy(y2Grad, y2.grad(), sizeof(y2Grad), cudaMemcpyDeviceToHost);
    cudaMemcpy(w2Grad, w2.grad(), sizeof(w2Grad), cudaMemcpyDeviceToHost);
    ok &= expect("chain grad x", x2Grad[0], 1.0f);
    ok &= expect("chain grad y", y2Grad[0], 1.0f);
    ok &= expect("chain grad w", w2Grad[0], 6.0f);

    Jensor<float> r({2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> r2({2, 2}, mytorch::AllocateOnGpu);
    Jensor<float> w3({2, 2}, mytorch::AllocateOnGpu);
    r.requires_grad(true);
    r2.requires_grad(true);
    upload(r, aHost, 4);
    upload(r2, bHost, 4);
    upload(w3, identity, 4);

    Jensor<float> s1 = r.matmul(w3);
    Jensor<float> s2 = r + r2;
    Jensor<float> out = s1 + s2;
    out.backward();
    float rGrad[4];
    cudaMemcpy(rGrad, r.grad(), sizeof(rGrad), cudaMemcpyDeviceToHost);
    ok &= expect("residual grad r", rGrad[0], 2.0f);

    mytorch::InputEmbeddings embed(100, 16);
    Jensor<float> tokEmb = embed.forward({1, 2, 3, 4});
    ok &= (tokEmb.shape()[0] == 4 && tokEmb.shape()[1] == 16);
    std::printf("input embeddings: %s\n", ok ? "OK" : "FAIL");

    Jensor<float> pe = mytorch::positional_encoding(4, 16);
    float peHost[64];
    download(pe, peHost, 64);
    ok &= expect("positional encoding pe[0][0]", peHost[0], 0.0f);
    ok &= expect("positional encoding pe[0][1]", peHost[1], 1.0f);

    Jensor<float> tx({2, 2}, mytorch::AllocateOnGpu);
    tx.requires_grad(true);
    upload(tx, aHost, 4);
    Jensor<float> ty = mytorch::transposed(tx);
    float tyHost[4];
    download(ty, tyHost, 4);
    ok &= expect("transposed fwd", tyHost[1], 3.0f);
    ty.backward();
    float txGrad[4];
    cudaMemcpy(txGrad, tx.grad(), sizeof(txGrad), cudaMemcpyDeviceToHost);
    ok &= expect("transposed grad", txGrad[0], 1.0f);

    Jensor<float> sx({2, 2}, mytorch::AllocateOnGpu);
    sx.requires_grad(true);
    upload(sx, aHost, 4);
    Jensor<float> sy = mytorch::scale(sx, 2.0f);
    float syHost[4];
    download(sy, syHost, 4);
    ok &= expect("scale fwd", syHost[0], 2.0f);
    sy.backward();
    float sxGrad[4];
    cudaMemcpy(sxGrad, sx.grad(), sizeof(sxGrad), cudaMemcpyDeviceToHost);
    ok &= expect("scale grad", sxGrad[0], 2.0f);

    Jensor<float> rx({2, 2}, mytorch::AllocateOnGpu);
    rx.requires_grad(true);
    float rxHost[4] = {-1, 2, -3, 4};
    upload(rx, rxHost, 4);
    Jensor<float> ry = mytorch::relu(rx);
    float ryHost[4];
    download(ry, ryHost, 4);
    ok &= expect("relu fwd", ryHost[0], 0.0f);
    ok &= expect("relu fwd pos", ryHost[1], 2.0f);
    ry.backward();
    float rxGrad[4];
    cudaMemcpy(rxGrad, rx.grad(), sizeof(rxGrad), cudaMemcpyDeviceToHost);
    ok &= expect("relu grad masked", rxGrad[0], 0.0f);
    ok &= expect("relu grad pass", rxGrad[1], 1.0f);

    Jensor<float> slx({2, 4}, mytorch::AllocateOnGpu);
    slx.requires_grad(true);
    float slxHost[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    upload(slx, slxHost, 8);
    Jensor<float> sly = mytorch::col_slice(slx, 1, 2);
    float slyHost[4];
    download(sly, slyHost, 4);
    ok &= expect("col_slice fwd", slyHost[0], 2.0f);
    ok &= expect("col_slice fwd2", slyHost[1], 3.0f);
    sly.backward();
    float slxGrad[8];
    cudaMemcpy(slxGrad, slx.grad(), sizeof(slxGrad), cudaMemcpyDeviceToHost);
    ok &= expect("col_slice grad in-range", slxGrad[1], 1.0f);
    ok &= expect("col_slice grad out-of-range", slxGrad[0], 0.0f);

    Jensor<float> csx({3, 3}, mytorch::AllocateOnGpu);
    csx.requires_grad(true);
    Jensor<float> csy = mytorch::causal_softmax(csx);
    float csyHost[9];
    download(csy, csyHost, 9);
    ok &= expect("causal_softmax row0", csyHost[0], 1.0f);
    ok &= expect("causal_softmax row1", csyHost[3], 0.5f);
    ok &= expect("causal_softmax masked", csyHost[2], 0.0f);
    csy.backward();
    float csxGrad[9];
    cudaMemcpy(csxGrad, csx.grad(), sizeof(csxGrad), cudaMemcpyDeviceToHost);
    ok &= expect("causal_softmax grad uniform seed", csxGrad[3], 0.0f);

    mytorch::Linear lin(2, 2, true);
    float identW[4] = {1, 0, 0, 1};
    upload(lin.weight(), identW, 4);
    float biasVals[2] = {10, 20};
    upload(*lin.bias(), biasVals, 2);

    Jensor<float> lx({2, 2}, mytorch::AllocateOnGpu);
    lx.requires_grad(true);
    upload(lx, aHost, 4);

    Jensor<float> ly = lin.forward(lx);
    float lyHost[4];
    download(ly, lyHost, 4);
    ok &= expect("linear fwd", lyHost[0], 11.0f);
    ok &= expect("linear fwd2", lyHost[3], 24.0f);

    ly.backward();
    float lxGrad[4], lwGrad[4], lbGrad[2];
    cudaMemcpy(lxGrad, lx.grad(), sizeof(lxGrad), cudaMemcpyDeviceToHost);
    cudaMemcpy(lwGrad, lin.weight().grad(), sizeof(lwGrad), cudaMemcpyDeviceToHost);
    cudaMemcpy(lbGrad, lin.bias()->grad(), sizeof(lbGrad), cudaMemcpyDeviceToHost);
    ok &= expect("linear grad x", lxGrad[0], 1.0f);
    ok &= expect("linear grad w", lwGrad[0], 4.0f);
    ok &= expect("linear grad b", lbGrad[0], 2.0f);

    mytorch::LayerNorm ln(4);
    Jensor<float> nx({1, 4}, mytorch::AllocateOnGpu);
    nx.requires_grad(true);
    float nxHost[4] = {1, 2, 3, 4};
    upload(nx, nxHost, 4);

    Jensor<float> ny = ln.forward(nx);
    float nyHost[4];
    download(ny, nyHost, 4);
    ok &= expect_close("layernorm fwd", nyHost[0], -1.341641f);
    ok &= expect_close("layernorm fwd sym", nyHost[3], 1.341641f);

    ny.backward();
    float nxGrad[4], ngGrad[4], nbGrad[4];
    cudaMemcpy(nxGrad, nx.grad(), sizeof(nxGrad), cudaMemcpyDeviceToHost);
    cudaMemcpy(ngGrad, ln.gamma().grad(), sizeof(ngGrad), cudaMemcpyDeviceToHost);
    cudaMemcpy(nbGrad, ln.beta().grad(), sizeof(nbGrad), cudaMemcpyDeviceToHost);
    ok &= expect_close("layernorm grad x", nxGrad[0], 0.0f);
    ok &= expect_close("layernorm grad gamma", ngGrad[0], -1.341641f);
    ok &= expect_close("layernorm grad beta", nbGrad[0], 1.0f);

    mytorch::MultiHeadSelfAttention mha(8, 2);
    Jensor<float> ax1({3, 8}, mytorch::AllocateOnGpu);
    Jensor<float> ax2({3, 8}, mytorch::AllocateOnGpu);
    float ax1Host[24], ax2Host[24];
    for (int i = 0; i < 24; ++i) ax1Host[i] = ax2Host[i] = (float)(i % 7) - 3.0f;
    for (int i = 16; i < 24; ++i) ax2Host[i] = ax1Host[i] + 100.0f;
    upload(ax1, ax1Host, 24);
    upload(ax2, ax2Host, 24);

    Jensor<float> aOut1 = mha.forward(ax1);
    Jensor<float> aOut2 = mha.forward(ax2);
    ok &= (aOut1.shape()[0] == 3 && aOut1.shape()[1] == 8);
    float aOut1Host[24], aOut2Host[24];
    download(aOut1, aOut1Host, 24);
    download(aOut2, aOut2Host, 24);

    bool causal = true;
    for (int i = 0; i < 16; ++i) causal &= (std::fabs(aOut1Host[i] - aOut2Host[i]) < 1e-4f);
    bool changed = std::fabs(aOut1Host[16] - aOut2Host[16]) > 1e-4f;
    std::printf("attention causal mask: %s\n", (causal && changed) ? "OK" : "FAIL");
    ok &= causal && changed;

    ax1.requires_grad(true);
    Jensor<float> aOut3 = mha.forward(ax1);
    aOut3.backward();
    float ax1Grad[24];
    cudaMemcpy(ax1Grad, ax1.grad(), sizeof(ax1Grad), cudaMemcpyDeviceToHost);
    bool hasNonzeroGrad = false;
    for (int i = 0; i < 24; ++i) hasNonzeroGrad |= (ax1Grad[i] != 0.0f);
    std::printf("attention backward nonzero grad: %s\n", hasNonzeroGrad ? "OK" : "FAIL");
    ok &= hasNonzeroGrad;

    {
        mytorch::MultiHeadSelfAttention bmha(8, 2);
        int batch = 2, seq = 3, dModel = 8;
        Jensor<float> bx({(uint16_t)batch, (uint16_t)seq, (uint16_t)dModel}, mytorch::AllocateOnGpu);
        std::vector<float> bxHost(batch * seq * dModel);
        for (int s = 0; s < seq; ++s) {
            for (int d = 0; d < dModel; ++d) {
                float v = (float)((s * dModel + d) % 7) - 3.0f;
                bxHost[0 * seq * dModel + s * dModel + d] = v;
                bxHost[1 * seq * dModel + s * dModel + d] = (s == seq - 1) ? v + 100.0f : v;
            }
        }
        upload(bx, bxHost.data(), bxHost.size());

        Jensor<float> bOut = bmha.forward(bx);
        ok &= (bOut.shape()[0] == (uint16_t)batch && bOut.shape()[1] == (uint16_t)seq && bOut.shape()[2] == (uint16_t)dModel);
        std::vector<float> bOutHost(batch * seq * dModel);
        download(bOut, bOutHost.data(), bOutHost.size());

        bool batchedCausal = true;
        for (int s = 0; s < seq - 1; ++s) {
            for (int d = 0; d < dModel; ++d) {
                int i0 = 0 * seq * dModel + s * dModel + d;
                int i1 = 1 * seq * dModel + s * dModel + d;
                batchedCausal &= (std::fabs(bOutHost[i0] - bOutHost[i1]) < 1e-4f);
            }
        }
        bool batchedChanged = std::fabs(bOutHost[0 * seq * dModel + (seq - 1) * dModel] -
                                        bOutHost[1 * seq * dModel + (seq - 1) * dModel]) > 1e-4f;
        std::printf("batched attention causal mask: %s\n", (batchedCausal && batchedChanged) ? "OK" : "FAIL");
        ok &= batchedCausal && batchedChanged;
    }

    {
        uint16_t vocabSize = 100, seqLen = 32, dModel = mytorch::kModelDim, numHeads = 8, dFf = 2048, numLayers = 5;
        mytorch::InputEmbeddings tokEmbeds(vocabSize, dModel);
        std::vector<int32_t> ids;
        for (uint16_t i = 0; i < seqLen; ++i) ids.push_back(i % vocabSize);

        Jensor<float> h = tokEmbeds.forward(ids) + mytorch::positional_encoding(seqLen, dModel);
        h.requires_grad(true);

        std::vector<mytorch::DecoderLayer> layers;
        for (uint16_t i = 0; i < numLayers; ++i) layers.emplace_back(dModel, numHeads, dFf);

        for (auto& layer : layers) h = layer.forward(h);

        ok &= (h.shape()[0] == seqLen && h.shape()[1] == dModel);
        h.backward();

        size_t freeB, totalB;
        cudaMemGetInfo(&freeB, &totalB);
        std::printf("5-layer decoder forward+backward: %s (seq_len=%u, d_model=%u, VRAM used %.1f/%.1f MB)\n",
                    ok ? "OK" : "FAIL", (unsigned)seqLen, (unsigned)dModel,
                    (totalB - freeB) / 1e6, totalB / 1e6);
    }

    {
        uint16_t vocabSize = 100, seqLen = 32, dModel = 128, numHeads = 8, dFf = 512, numLayers = 3;
        uint16_t batch = 4;
        mytorch::DecoderOnlyTransformer model(vocabSize, dModel, numHeads, dFf, numLayers);

        std::vector<int32_t> flatIds(batch * seqLen);
        for (int i = 0; i < batch * seqLen; ++i) flatIds[i] = i % vocabSize;

        Jensor<float> logits = model.forward(flatIds, batch, seqLen);
        ok &= (logits.shape()[0] == batch && logits.shape()[1] == seqLen && logits.shape()[2] == vocabSize);

        std::vector<int32_t> targets(batch * seqLen);
        for (int i = 0; i < batch * seqLen; ++i) targets[i] = (i + 1) % vocabSize;
        Jensor<float> loss = mytorch::cross_entropy(logits, targets);
        loss.backward();

        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 3; ++rep) {
            Jensor<float> l2 = model.forward(flatIds, batch, seqLen);
            cudaDeviceSynchronize();
        }
        auto t1 = std::chrono::steady_clock::now();
        for (int rep = 0; rep < 3 * batch; ++rep) {
            std::vector<int32_t> oneIds(flatIds.begin(), flatIds.begin() + seqLen);
            Jensor<float> l2 = model.forward(oneIds);
            cudaDeviceSynchronize();
        }
        auto t2 = std::chrono::steady_clock::now();

        double batchedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double sequentialMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("batched model forward: %s (batch=%u logits shape OK, batched %.1fms vs %ux-sequential %.1fms)\n",
                    ok ? "OK" : "FAIL", (unsigned)batch, batchedMs, (unsigned)batch, sequentialMs);
    }

    {
        database::CharTokenizer tok("the quick brown fox jumps over the lazy dog");
        mytorch::DecoderOnlyTransformer modelA((uint16_t)tok.vocab_size(), 64, 4, 128, 2);
        mytorch::DecoderOnlyTransformer modelB((uint16_t)tok.vocab_size(), 64, 4, 128, 2);
        mytorch::save_checkpoint("build/kv_test.bin", modelA.parameters());
        mytorch::load_checkpoint("build/kv_test.bin", modelB.parameters());

        std::vector<int32_t> prompt = tok.encode("the quick");
        std::vector<int32_t> viaFull = app::greedy_generate(modelA, tok, prompt, 10, 128);
        std::vector<int32_t> viaCache = app::kv_cache_generate(modelB, tok, prompt, 10);

        bool kvMatches = viaFull.size() == viaCache.size();
        for (size_t i = 0; kvMatches && i < viaFull.size(); ++i) kvMatches &= (viaFull[i] == viaCache[i]);
        std::printf("kv cache matches full recompute: %s\n", kvMatches ? "OK" : "FAIL");
        ok &= kvMatches;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("cuda error: %s\n", cudaGetErrorString(err));
        ok = false;
    }

    return ok ? 0 : 1;
}
