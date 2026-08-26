# 03 — naive attention

**Goal:** hand-written scaled dot-product attention as three explicit steps
on Jensor: `Q @ K^T` (scaled), softmax over the last dim, `@ V`. Uses the
exercise 1 matmul kernel (or a batched variant of it) plus a new hand-written
softmax kernel. Materializes the full attention-score matrix — that's the
point, it's what exercise 4 removes.

**AI assistance level: hands-off**, same as exercise 1 — this is where you
build the intuition for why attention is memory-bound (the N×N score matrix
you're about to write and re-read is the whole reason FlashAttention/cuDNN's
fused kernel exist). Claude reviews and explains on request; doesn't write
the softmax kernel or wire the three steps together.

**Done when:** output matches a CPU reference implementation, and
`benchmarks/attention/` has a naive-path entry in `benchmarks/results.md`
including memory traffic/time for the intermediate N×N matrix specifically —
that number is what exercise 4 should visibly beat.
