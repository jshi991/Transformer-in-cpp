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

- `d_model = 512`, 8 attention heads, `d_ff = 2048`, 5 decoder layers,
  `seq_len = 128` — sized for a single consumer GPU (tested on a 6GB card).
- Pre-norm (GPT-style) decoder block: `x = x + attn(ln1(x))`,
  `x = x + ffn(ln2(x))`.
- Adam optimizer, character-level tokenizer, cross-entropy loss.
- `Jensor` supports 2D and 3D (batch, seq, feature) tensors; batching across
  sequences runs as real parallel GPU work, not a Python-style loop.
