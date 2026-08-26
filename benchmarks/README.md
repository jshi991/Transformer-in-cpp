# benchmarks

Compares hand-written Jensor kernels against vendor libraries (cuBLAS,
cuDNN) on the same ops/shapes. Each exercise in `../agents.md/` that adds an
implementation should end with a result recorded here.

- `matmul/` — naive kernel vs. cuBLAS (`agents.md/01`, `agents.md/02`)
- `attention/` — naive 3-kernel attention vs. cuDNN fused MHA
  (`agents.md/03`, `agents.md/04`)
- `results.md` — running log: one row per (op, implementation, shape) run,
  appended over time rather than overwritten, so old numbers stay comparable
  as the hardware/build changes.

Nothing runs locally yet (no `nvcc` on the Mac dev machine, same constraint
as the rest of the CUDA build — see `colab/colab_build.ipynb`). Benchmark
harnesses go under `matmul/` and `attention/` once the corresponding Jensor
ops exist; timing should use CUDA events (`cudaEventRecord`) around the
kernel/library call, not wall-clock around the whole process. Where
possible, cross-check with Nsight Compute for memory-throughput numbers
rather than trusting wall time alone (see `agents.md/README.md`).
