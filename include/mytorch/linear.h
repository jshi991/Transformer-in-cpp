#ifndef MYTORCH_LINEAR_H
#define MYTORCH_LINEAR_H

#include "mytorch/jensor.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mytorch {

// Optional forward hook: Linear::forward passes its (pre-matmul) input to
// observer_->observe() when one is attached. Used by calibration tooling
// (e.g. quantize/) without the core library knowing about quantization.
class ActivationObserver {
    public:
        virtual void observe(const Jensor<float>& x) = 0;
        virtual ~ActivationObserver() = default;
};

class Linear {
    public:
        Linear(uint16_t in_features, uint16_t out_features, bool bias = true);

        Jensor<float> forward(const Jensor<float>& x);

        Jensor<float>& weight();
        Jensor<float>* bias();
        std::vector<Jensor<float>*> parameters();
        void set_observer(ActivationObserver* observer);

        // Switches forward() to a weight-only int8 GEMM: the packed int8
        // codes are dequantized inline inside the matmul kernel (never
        // materializing a full fp32 weight copy), instead of the regular
        // fp32 matmul against weight(). Inference-only — backward() through
        // a quantized Linear is not supported. codes/groupScale/channelScale
        // use the same layout as quantize::QuantizedWeight.
        void load_int8_weights(const std::vector<uint8_t>& codes, const std::vector<float>& groupScale,
                                const std::vector<float>& channelScale, int groupSize);
        bool is_quantized() const { return quantized_; }

    private:
        uint16_t in_features_;
        uint16_t out_features_;
        bool has_bias_;
        Jensor<float> weight_;
        Jensor<float> bias_;
        ActivationObserver* observer_ = nullptr;

        bool quantized_ = false;
        int groupSize_ = 0;
        std::shared_ptr<int8_t> qCodes_;
        std::shared_ptr<float> qGroupScale_;
        std::shared_ptr<float> qChannelScale_;
};

}  // namespace mytorch
#endif  // MYTORCH_LINEAR_H
