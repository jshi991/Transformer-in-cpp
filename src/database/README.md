# database

Data pipeline for the decoder-only model: load raw text and tokenize it
(char-level `stoi`/`itos`).

- `dataset/` — raw text corpora live here (gitignored). `wikitext-2-raw-train.txt`
  (WikiText-2, raw split, pulled from `Salesforce/wikitext` on Hugging Face)
  is the default corpus used by `make setup-model` / `make quantize` / `make infer`.
- `dataset.h` / `dataset.cu` — `CharTokenizer` (builds `stoi`/`itos` from the
  corpus, first-seen order) and `load_dataset_text` (reads a file into a
  `std::string`). Callers (`train.cu`, `quantize/`) encode the whole corpus
  once with `CharTokenizer::encode` and slice fixed-length windows out of the
  resulting `std::vector<int32_t>` themselves.

Known rough edges, left as-is for now:
- Tokenizer is character-level. Swap for a subword/BPE tokenizer later if needed.
- Token ids are `int32_t` on the host; `Jensor<float>` stores them as floats
  once they're copied to the GPU (`Jensor<float>` is the only instantiated
  `Jensor` type right now).
