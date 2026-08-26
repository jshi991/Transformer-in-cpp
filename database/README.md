# database

Data pipeline for the decoder-only model: load raw text, tokenize it
(char-level `stoi`/`itos`), and land the token ids on the GPU as a
`mytorch::Jensor`.

- `dataset/` — where the raw on-device text file(s) live. Nothing is
  downloaded automatically yet; put a `.txt` file here and pass its path to
  `load_dataset_to_gpu`.
  **TODO(refactor):** pull the dataset from the Hugging Face Hub (e.g. the
  `datasets` library or `hf_hub_download`) instead of expecting a
  pre-downloaded local file.
- `dataset.h` / `dataset.cu` — `CharTokenizer` (builds `stoi`/`itos` from the
  corpus, first-seen order) and `load_dataset_to_gpu`, which reads the file,
  encodes it, and copies the ids into a GPU-resident `Jensor<float>` via
  `cudaMemcpy`.

Built by `make cuda` (needs `nvcc`; not available in local dev, verified on
Colab — see `colab/colab_build.ipynb`). Nothing here runs on the Mac dev
machine yet, per the "allocate on GPU for now" ask — CPU/local testing is a
later step.

Known rough edges, left as-is for now:
- Tokenizer is character-level — the simplest thing that works. Swap for a
  subword/BPE tokenizer later if needed.
- Token ids are stored as `float` in the `Jensor` because `Jensor<float>` is
  the only explicit template instantiation that currently exists
  (`src/mytorch/jensor.cu`). Fine for vocab sizes far under 2^24, but ids are
  conceptually integers — add a `Jensor<int32_t>` instantiation if that
  matters later.
- `Jensor` shape dims are `uint16_t` (max 65535 per dimension), so a corpus
  longer than that can't be loaded as one flat `(1, N)` Jensor as-is. Chunking
  into `(num_blocks, block_size)` for training will need to happen before or
  during loading — not implemented here yet.
