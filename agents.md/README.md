# agents.md — exercise ramp

This folder is the source of truth for how much AI assistance is appropriate
at each stage of building the CUDA-heavy parts of Jensor (matmul, attention).
It exists so the ramp survives across sessions instead of living only in one
conversation — see the root `CLAUDE.md`, which points here.

The idea: early exercises are hand-written with little AI help, because the
whole point is building your own understanding of the kernel. Later
exercises — once you're wiring in vendor libraries (cuBLAS, cuDNN) — lean on
AI more for boilerplate/glue, matching how this is actually done in industry:
engineers hand-write the fundamentals once, then mostly call (and let tooling
help them call) a vendor's optimized implementation rather than re-deriving
it every time.

Each exercise file has:
- **Goal** — what gets built
- **AI assistance level** — what Claude should and shouldn't write
- **Done when** — the benchmark result that closes it out (see `../benchmarks/`)

## Exercises

1. [`01-naive-matmul.md`](01-naive-matmul.md) — hand-written matmul kernel
2. [`02-cublas-matmul.md`](02-cublas-matmul.md) — cuBLAS-backed matmul, benchmarked against #1
3. [`03-naive-attention.md`](03-naive-attention.md) — hand-written QK^T → softmax → @V
4. [`04-cudnn-fused-attention.md`](04-cudnn-fused-attention.md) — cuDNN fused MHA, benchmarked against #3

When picking up work in this repo: check which exercise is in progress, read
its assistance level before writing any code, and default to the more
conservative (hands-off) level if it's unclear which exercise applies.
