# 01 — naive matmul

**Goal:** add `Jensor::matmul` with a hand-written CUDA kernel (2D case
first; batched/strided version can wait for exercise 3's attention needs).
No tiling, no shared memory yet — just correct, one-thread-per-output-element.

**AI assistance level: hands-off.** Claude should not write the kernel or
the `matmul` method body. Fine for Claude to: review code after it's
written, explain a CUDA concept (coalescing, grid/block sizing) on request,
or sanity-check the math. Not fine: writing the `__global__` kernel,
picking block/thread dimensions, or filling in the method — that's the part
this exercise is for.

**Done when:** `matmul` produces correct results (checked against a CPU
reference loop) and a naive benchmark harness under `benchmarks/matmul/`
records its time/GFLOPS for a few shapes in `benchmarks/results.md`.
