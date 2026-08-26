# 04 — cuDNN fused attention

**Goal:** swap exercise 3's three-kernel attention for cuDNN's fused
scaled-dot-product-attention op (`cudnn_frontend` graph API, cuDNN ≥8.9).
Requires linking `-lcudnn` and checking it's actually installed/linkable in
the build environment (Colab usually has it via its PyTorch install — verify
before assuming).

**AI assistance level: high — this is the "use the vendor's fused kernel"
step, industry-realistic to lean on tooling for.** Claude can draft the
`cudnn_frontend` graph construction (this API is verbose and mostly
boilerplate — tensor descriptors, op graph nodes, engine heuristics). Focus
your own attention on: understanding *why* it's fused (no materialized N×N
matrix — the memory-bound cost from exercise 3 is what this removes) and
verifying that's actually true by comparing memory traffic, not just wall
time.

**Done when:** output matches exercise 3 on the same inputs, and
`benchmarks/results.md` shows the fused path's time and (if measurable via
Nsight Compute) memory traffic against exercise 3 — the intermediate score
matrix should visibly disappear from the memory trace.
