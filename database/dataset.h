#ifndef DATABASE_DATASET_H
#define DATABASE_DATASET_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "mytorch/jensor.h"

namespace database {

// Character-level tokenizer for a decoder-only LM: every distinct byte seen
// in the training corpus gets an id, assigned in first-seen order.
//
// TODO: this is the bare-minimum tokenizer. Swap for a subword/BPE tokenizer
// once database/dataset/ is refactored to pull from the Hugging Face Hub.
class CharTokenizer {
public:
    explicit CharTokenizer(const std::string& text);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    size_t vocab_size() const { return itos_.size(); }
    const std::unordered_map<char, int32_t>& stoi() const { return stoi_; }
    const std::vector<char>& itos() const { return itos_; }

private:
    std::unordered_map<char, int32_t> stoi_;
    std::vector<char> itos_;
};

// Reads the whole file at `path` into a string.
//
// TODO: this is the "on-device" path — refactor to fetch dataset shards from
// the Hugging Face Hub instead of expecting a pre-downloaded file under
// database/dataset/.
std::string load_dataset_text(const std::string& path);

// Reads `path`, encodes it with `tok`, and copies the resulting token ids
// onto the GPU as a Jensor<float> of shape (1, num_tokens).
//
// Ids are stored as float since Jensor<float> is the only instantiated
// Jensor type right now (see src/mytorch/jensor.cu) — revisit if an integer
// Jensor gets instantiated later. Note num_tokens must fit in a uint16_t
// (Jensor's shape dim type); long corpora aren't chunked yet.
mytorch::Jensor<float> load_dataset_to_gpu(const std::string& path, const CharTokenizer& tok);

}  // namespace database

#endif  // DATABASE_DATASET_H
