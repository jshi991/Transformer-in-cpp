#ifndef DATABASE_DATASET_H
#define DATABASE_DATASET_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace database {

// Character-level tokenizer: every distinct byte seen in the corpus gets an
// id, assigned in first-seen order.
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

std::string load_dataset_text(const std::string& path);

}  // namespace database

#endif  // DATABASE_DATASET_H
