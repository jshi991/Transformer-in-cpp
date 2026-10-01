#include "quantize/awq.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace quantize {

CalibrationRecorder::CalibrationRecorder(int in_features, int max_snapshots)
    : inFeatures_(in_features), maxSnapshots_(max_snapshots) {}

void CalibrationRecorder::observe(const mytorch::Jensor<float>& x) {
    long long rows = 1;
    for (size_t i = 0; i + 1 < x.shape().size(); ++i) rows *= x.shape()[i];
    int inFeatures = x.shape().back();

    std::vector<float> host((size_t)rows * inFeatures);
    cudaMemcpy(host.data(), x.data(), sizeof(float) * host.size(), cudaMemcpyDeviceToHost);

    if (channelAbsMean_.empty()) channelAbsMean_.assign(inFeatures, 0.f);
    for (long long r = 0; r < rows; ++r) {
        ++rowsSeen_;
        for (int c = 0; c < inFeatures; ++c) {
            float v = std::fabs(host[r * inFeatures + c]);
            channelAbsMean_[c] += (v - channelAbsMean_[c]) / (float)rowsSeen_;
        }
    }

    if ((int)snapshots_.size() < maxSnapshots_) {
        snapshots_.push_back(std::move(host));
        snapshotRows_.push_back((int)rows);
    }
}

namespace {

QuantizedWeight quantize_with_channel_scale(const std::vector<float>& weight, int rows, int cols,
                                             const std::vector<float>& channelScale, int bits, int groupSize) {
    QuantizedWeight q;
    q.rows = rows;
    q.cols = cols;
    q.bits = bits;
    q.groupSize = groupSize;
    q.channelScale = channelScale;

    int numGroups = q.num_groups();
    q.groupScale.assign(numGroups, 0.f);

    std::vector<float> scaled((size_t)rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) scaled[(size_t)r * cols + c] = weight[(size_t)r * cols + c] * channelScale[r];

    int qmax = (1 << (bits - 1)) - 1;
    std::vector<int32_t> codesInt((size_t)rows * cols);

    for (int g = 0; g < numGroups; ++g) {
        int r0 = g * groupSize, r1 = std::min(r0 + groupSize, rows);
        float maxAbs = 1e-8f;
        for (int r = r0; r < r1; ++r)
            for (int c = 0; c < cols; ++c) maxAbs = std::max(maxAbs, std::fabs(scaled[(size_t)r * cols + c]));

        float groupScale = maxAbs / qmax;
        q.groupScale[g] = groupScale;

        for (int r = r0; r < r1; ++r) {
            for (int c = 0; c < cols; ++c) {
                size_t idx = (size_t)r * cols + c;
                int code = (int)std::lround(scaled[idx] / groupScale);
                code = std::max(-qmax, std::min(qmax, code));
                codesInt[idx] = code;
            }
        }
    }

    size_t n = codesInt.size();
    if (bits == 8) {
        q.codes.resize(n);
        for (size_t i = 0; i < n; ++i) q.codes[i] = (uint8_t)(int8_t)codesInt[i];
    } else {
        q.codes.assign((n + 1) / 2, 0);
        for (size_t i = 0; i < n; i += 2) {
            int lo = codesInt[i] & 0xF;
            int hi = (i + 1 < n ? codesInt[i + 1] : 0) & 0xF;
            q.codes[i / 2] = (uint8_t)(lo | (hi << 4));
        }
    }

    return q;
}

double reconstruction_error_gpu(const mytorch::Jensor<float>& xg, const mytorch::Jensor<float>& yg,
                                 const std::vector<float>& approx, int rows, int cols) {
    mytorch::Jensor<float> wag({(uint16_t)rows, (uint16_t)cols}, mytorch::AllocateOnGpu);
    cudaMemcpy(wag.data(), approx.data(), sizeof(float) * approx.size(), cudaMemcpyHostToDevice);

    mytorch::Jensor<float> yhat = xg.matmul(wag);

    size_t n = (size_t)xg.shape()[0] * cols;
    std::vector<float> yHost(n), yhatHost(n);
    cudaMemcpy(yHost.data(), yg.data(), sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(yhatHost.data(), yhat.data(), sizeof(float) * n, cudaMemcpyDeviceToHost);

    double err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)yHost[i] - yhatHost[i];
        err += d * d;
    }
    return err;
}

}  // namespace

QuantizedWeight awq_quantize(const std::vector<float>& weight, int rows, int cols,
                             const CalibrationRecorder& calib, int bits, int groupSize) {
    const auto& actAbs = calib.channel_abs_mean();

    mytorch::Jensor<float> wg({(uint16_t)rows, (uint16_t)cols}, mytorch::AllocateOnGpu);
    cudaMemcpy(wg.data(), weight.data(), sizeof(float) * weight.size(), cudaMemcpyHostToDevice);

    std::vector<mytorch::Jensor<float>> xgs, ygs;
    for (size_t i = 0; i < calib.snapshots().size(); ++i) {
        int xRows = calib.snapshot_rows()[i];
        mytorch::Jensor<float> xg({(uint16_t)xRows, (uint16_t)rows}, mytorch::AllocateOnGpu);
        cudaMemcpy(xg.data(), calib.snapshots()[i].data(), sizeof(float) * (size_t)xRows * rows, cudaMemcpyHostToDevice);
        mytorch::Jensor<float> yg = xg.matmul(wg);
        xgs.push_back(std::move(xg));
        ygs.push_back(std::move(yg));
    }

    const int numAlphas = 11;
    double bestErr = -1.0;
    std::vector<float> bestScale(rows, 1.0f);

    for (int ai = 0; ai < numAlphas; ++ai) {
        float alpha = ai / (float)(numAlphas - 1);
        std::vector<float> s(rows, 1.0f);
        if (!actAbs.empty()) {
            for (int r = 0; r < rows; ++r) s[r] = std::pow(std::max(actAbs[r], 1e-5f), alpha);
            float meanS = 0.f;
            for (float v : s) meanS += v;
            meanS = std::max(meanS / rows, 1e-8f);
            for (float& v : s) v /= meanS;
        }

        QuantizedWeight cand = quantize_with_channel_scale(weight, rows, cols, s, bits, groupSize);
        std::vector<float> approx = awq_dequantize(cand);

        double err = 0.0;
        for (size_t i = 0; i < xgs.size(); ++i) err += reconstruction_error_gpu(xgs[i], ygs[i], approx, rows, cols);

        if (bestErr < 0.0 || err < bestErr) {
            bestErr = err;
            bestScale = s;
        }
    }

    return quantize_with_channel_scale(weight, rows, cols, bestScale, bits, groupSize);
}

std::vector<float> awq_dequantize(const QuantizedWeight& q) {
    std::vector<float> out((size_t)q.rows * q.cols);

    auto unpack = [&](size_t i) -> int {
        if (q.bits == 8) return (int)(int8_t)q.codes[i];
        uint8_t byte = q.codes[i / 2];
        int nibble = (i % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
        return nibble > 7 ? nibble - 16 : nibble;
    };

    for (int r = 0; r < q.rows; ++r) {
        int g = r / q.groupSize;
        float scale = q.groupScale[g] / q.channelScale[r];
        for (int c = 0; c < q.cols; ++c) {
            size_t idx = (size_t)r * q.cols + c;
            out[idx] = unpack(idx) * scale;
        }
    }

    return out;
}

double relative_error(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        num += d * d;
        den += (double)a[i] * a[i];
    }
    return std::sqrt(num / std::max(den, 1e-12));
}

size_t packed_size_bytes(const QuantizedWeight& q) {
    return q.codes.size() + q.groupScale.size() * sizeof(float) + q.channelScale.size() * sizeof(float);
}

}  // namespace quantize
