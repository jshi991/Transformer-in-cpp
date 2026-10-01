# Transformer-in-cpp

A decoder-only transformer (GPT-style) written from scratch in CUDA/C++ —
no PyTorch, no libtorch. `Jensor` is a small hand-rolled tensor type with its
own reverse-mode autograd, on top of which the embeddings, attention,
feed-forward, training loop, and an AWQ weight quantizer are all built
directly against raw CUDA kernels.

## Requirements

- NVIDIA GPU + driver
- `nvcc` (CUDA toolkit) on `PATH`
- `g++` (or `clang++` on macOS) for the small CPU-only `transformer` target

Everything that actually builds a model is a CUDA target; the plain
`transformer`/`transformer_cuda` targets below are smoke tests, not the
model binary.

## Quickstart: train → quantize → infer

```sh
make setup-model   # trains the 5-layer decoder-only model, saves build/checkpoint.bin
make quantize      # AWQ-quantizes it to int8, saves build/checkpoint_awq.bin
make infer         # greedy-decodes text from the quantized checkpoint
```

`setup-model` downloads nothing itself — it expects a text corpus at
`src/database/dataset/wikitext-2-raw-train.txt` (see **Dataset** below).

Each of these is a thin wrapper around a real binary with its own CLI args,
so you can run them directly for more control:

```sh
./build/train       <corpus.txt> <iters> <batch_size> <checkpoint_out>
./build/quantize_awq <corpus.txt> <checkpoint_in> <checkpoint_out> <bits: 8|4> <group_size>
./build/infer        <corpus.txt> <checkpoint_in> <num_tokens>
```

`train` auto-resumes if `<checkpoint_out>` already exists, and saves a
checkpoint every 50 iterations plus at the end. All three binaries must be
pointed at the *same* corpus file — the vocabulary (`CharTokenizer`) is
rebuilt from the corpus text every run, not stored in the checkpoint, so a
different corpus means a different token-id mapping and a checkpoint that
silently no longer lines up with it.

## Other Makefile targets

| Target | What it builds/runs |
|---|---|
| `make cuda` | `build/transformer_cuda` — the real correctness suite: every `Jensor` op, autograd path, and model component checked against hand-computed expected values (run it directly to see all checks) |
| `make bench-matmul` | naive kernel vs. cuBLAS matmul benchmark |
| `make train` / `setup-model` | the training loop (`src/train.cu`) |
| `make quantize` | the AWQ quantizer (`src/quantize/quantize_main.cu`) |
| `make infer` | greedy generation from a checkpoint (`src/quantize/infer_main.cu`) |
| `make clean` | removes `build/` |

## Architecture

Fixed in `include/dataset/common.h` (`app::k*` constants) and
`mytorch::kModelDim`:

- `d_model = 512`, 8 attention heads, `d_ff = 2048`, 5 decoder layers,
  `seq_len = 128` — sized for a single consumer GPU (tested on a 6GB card).
- Pre-norm (GPT-style) decoder block: `x = x + attn(ln1(x))`,
  `x = x + ffn(ln2(x))`.
- Adam optimizer, character-level tokenizer, cross-entropy loss.
- `Jensor` supports 2D and 3D (batch, seq, feature) tensors; batching across
  sequences runs as real parallel GPU work, not a Python-style loop.

## Dataset

`src/database/dataset/` is gitignored — nothing is committed there. Put a
plain-text corpus at `src/database/dataset/wikitext-2-raw-train.txt`, or
pass a different path as the first argument to `train`/`quantize_awq`/`infer`.
The default corpus used across this project is the WikiText-2 (raw) train
split from `Salesforce/wikitext` on Hugging Face.

## Known limitations

- **Unbatched-by-default naive allocator**: most ops allocate a fresh GPU
  buffer per call rather than reusing memory, so wall-clock time is
  dominated by `cudaMalloc`/`cudaFree` churn, not compute. Fine for a 5-layer
  model on a single GPU; would need a memory pool to scale further.
- **No embedding-table sharing / weight tying** between the input embedding
  and the output `lm_head`.
- **No KV cache**: `app::greedy_generate` reruns the full forward pass over
  the entire context window at every single decode step and throws away
  everything except the last token's logits. A real implementation would
  cache each layer's K/V projections across steps so each new token costs
  O(1) incremental work instead of recomputing the whole window every time.
- **AWQ quantization dequantizes back to fp32** for inference — it proves
  the quantization math and measures real accuracy loss (see the per-layer
  error and before/after loss printed by `make quantize`), but it does not
  run an actual low-bit (int4/int8) GEMM kernel at inference time.
- The model trains very little by default (`setup-model` runs 200 iterations
  on random windows); increase the iteration count in the Makefile or by
  calling `./build/train` directly for a less undertrained model.
