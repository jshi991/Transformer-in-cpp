#ifndef QUANTIZE_AWQ_H
#define QUANTIZE_AWQ_H

#include "mytorch/jensor.h"
#include "mytorch/linear.h"

#include <cstdint>
#include <vector>

namespace quantize {

class CalibrationRecorder : public mytorch::ActivationObserver {
    public:
        explicit CalibrationRecorder(int in_features, int max_snapshots = 4);

        void observe(const mytorch::Jensor<float>& x) override;

        const std::vector<float>& channel_abs_mean() const { return channelAbsMean_; }
        const std::vector<std::vector<float>>& snapshots() const { return snapshots_; }
        const std::vector<int>& snapshot_rows() const { return snapshotRows_; }

    private:
        int inFeatures_;
        int maxSnapshots_;
        long long rowsSeen_ = 0;
        std::vector<float> channelAbsMean_;
        std::vector<std::vector<float>> snapshots_;
        std::vector<int> snapshotRows_;
};

struct QuantizedWeight {
    int rows = 0, cols = 0;
    int bits = 8;
    int groupSize = 64;
    std::vector<float> channelScale; 
    std::vector<float> groupScale;    
    std::vector<uint8_t> codes;

    int num_groups() const { return (rows + groupSize - 1) / groupSize; }
};

QuantizedWeight awq_quantize(const std::vector<float>& weight, int rows, int cols,
                             const CalibrationRecorder& calib, int bits, int groupSize);

std::vector<float> awq_dequantize(const QuantizedWeight& q);

double relative_error(const std::vector<float>& a, const std::vector<float>& b);

size_t packed_size_bytes(const QuantizedWeight& q);

}  // namespace quantize
#endif  // QUANTIZE_AWQ_H
