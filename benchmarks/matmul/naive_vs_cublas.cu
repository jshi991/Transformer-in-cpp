// Benchmark: Jensor::matmul, Backend::Naive vs Backend::CuBLAS, same shapes.
// See agents.md/01, agents.md/02, and benchmarks/README.md.
//
// Flip kRunNaive to true once the naive kernel (exercise 1) is wired up —
// it's off by default because that branch isn't finished yet and currently
// hits an assert. With it off, this only records cuBLAS numbers.
//
// Build:  make bench-matmul
// Run:    ./build/bench_matmul >> benchmarks/results.md
//         (or just run it and paste the printed rows in by hand)

#include "mytorch/jensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <ctime>
#include <random>
#include <vector>

namespace {

constexpr bool kRunNaive = false;  // see file comment above

std::string today() {
    std::time_t t = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&t));
    return buf;
}

std::string device_name() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    return prop.name;
}

// Fills a GPU Jensor with random values via a host staging buffer —
// Jensor::data() is documented as the way callers copy data in/out.
mytorch::Jensor<float> random_gpu_jensor(uint16_t rows, uint16_t cols, std::mt19937& rng) {
    mytorch::Jensor<float> t({rows, cols}, mytorch::AllocateOnGpu);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> host(static_cast<size_t>(rows) * cols);
    for (auto& v : host) v = dist(rng);
    cudaMemcpy(t.data(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    return t;
}

std::vector<float> to_host(const mytorch::Jensor<float>& t) {
    long long n = 1;
    for (auto d : t.shape()) n *= d;
    std::vector<float> host(static_cast<size_t>(n));
    cudaMemcpy(host.data(), t.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost);
    return host;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

// Times `run` (already-launched-synchronously-by-the-time-it-returns work)
// using CUDA events, per benchmarks/README.md ("timing should use CUDA
// events... not wall-clock").
template <typename Fn>
float time_ms(Fn&& run) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    run();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

void report_row(const std::string& impl, uint16_t M, uint16_t K, uint16_t N,
                 const std::string& device, float ms, const std::string& notes) {
    double flops = 2.0 * M * N * K;
    double gflops = flops / (ms / 1000.0) / 1e9;
    std::printf("| %s | matmul | %s | %ux%ux%u | %s | %.3f ms | %.2f GFLOPS | %s |\n",
                today().c_str(), impl.c_str(), M, K, N, device.c_str(), ms, gflops, notes.c_str());
}

}  // namespace

int main() {
    using mytorch::Jensor;
    std::mt19937 rng(42);
    std::string device = device_name();

    // A few shapes worth comparing; add more as needed.
    struct Shape { uint16_t M, K, N; };
    const Shape shapes[] = {{256, 256, 256}, {512, 512, 512}, {1024, 1024, 1024}};

    for (auto s : shapes) {
        Jensor<float> a = random_gpu_jensor(s.M, s.K, rng);
        Jensor<float> b = random_gpu_jensor(s.K, s.N, rng);

        Jensor<float> c_cublas({0, 0}, mytorch::AllocateOnGpu);  // reassigned below
        float cublas_ms = time_ms([&] {
            c_cublas = a.matmul(b, Jensor<float>::Backend::CuBLAS);
        });
        report_row("cuBLAS", s.M, s.K, s.N, device, cublas_ms, "");

        if constexpr (kRunNaive) {
            Jensor<float> c_naive({0, 0}, mytorch::AllocateOnGpu);
            float naive_ms = time_ms([&] {
                c_naive = a.matmul(b, Jensor<float>::Backend::Naive);
            });

            float diff = max_abs_diff(to_host(c_naive), to_host(c_cublas));
            std::string notes = diff < 1e-2f ? "matches cuBLAS" : "MISMATCH vs cuBLAS!";
            report_row("naive", s.M, s.K, s.N, device, naive_ms, notes);
        }
    }

    return 0;
}
