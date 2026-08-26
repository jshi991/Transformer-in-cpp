# 02 — cuBLAS matmul

**Goal:** add a second GPU code path for `Jensor::matmul` that calls
`cublasSgemm` (or `cublasGemmEx`) instead of the exercise 1 kernel, selected
by a flag/build option so both stay comparable. Requires linking `-lcublas`
in the `cuda` target.

**AI assistance level: moderate — glue is fine, understanding isn't
optional.** This is vendor-library integration, not algorithm design, so
Claude can draft the cuBLAS handle setup, the `cublasSgemm` call and its
(notoriously fiddly, column-major) argument order, and the Makefile link
flag. Before accepting that code: read the cuBLAS docs for the call being
used and be able to explain why the arguments are what they are (cuBLAS
expects column-major — this trips everyone up the first time and is worth
actually understanding, not pattern-matching past).

**Done when:** cuBLAS path matches exercise 1's output on the same inputs,
and `benchmarks/results.md` has both numbers side by side for the same
shapes — this is the comparison the whole ramp is building toward.
