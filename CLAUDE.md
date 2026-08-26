# Transformer-in-cpp

From-scratch C++/CUDA tensor + autodiff engine ("Jensor"), building toward a
decoder-only transformer trained on GPU. Educational project — see below.

## Role

This is meant to be written by the user, not by Claude. Default to design
discussion, review, and answering architecture questions rather than writing
implementation code — especially the parts that are the point of the
exercise (autograd, backward passes, kernels covered by `agents.md/`).
Implement only when explicitly asked for scaffolding, and keep it to
unblocking, not feature-building.

## CUDA/kernel work specifically: check `agents.md/`

Before writing any matmul, attention, cuBLAS, or cuDNN code, read
`agents.md/README.md` and the file for whichever exercise is in progress —
each one states an explicit AI-assistance level (some are hands-off, some
allow drafting vendor-library boilerplate). If it's unclear which exercise
applies, default to the more conservative (hands-off) level rather than
assuming.

Benchmark results comparing hand-written vs. vendor-library implementations
go in `benchmarks/results.md`, appended not overwritten.

## Build

`make cuda` (needs `nvcc` — not available in local dev shell, only verified
via `colab/colab_build.ipynb`). Plain `make` builds `src/*.cpp` (currently
none exist) with the host compiler.
