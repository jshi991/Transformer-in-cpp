#include "include/dataset/dataset.h"

#include <cuda_runtime.h>

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace database {

CharTokenizer::CharTokenizer(const std::string& text) {
    for (char c : text) {
        if (stoi_.find(c) == stoi_.end()) {
            stoi_[c] = static_cast<int32_t>(itos_.size());
            itos_.push_back(c);
        }
    }
}

std::vector<int32_t> CharTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (char c : text) {
        auto it = stoi_.find(c);
        if (it == stoi_.end()) {
            throw std::out_of_range("CharTokenizer::encode: character not in vocab");
        }
        ids.push_back(it->second);
    }
    return ids;
}

std::string CharTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string text;
    text.reserve(ids.size());
    for (int32_t id : ids) {
        text.push_back(itos_.at(static_cast<size_t>(id)));
    }
    return text;
}

std::string load_dataset_text(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("load_dataset_text: could not open " + path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

mytorch::Jensor<float> load_dataset_to_gpu(const std::string& path, const CharTokenizer& tok) {
    std::string text = load_dataset_text(path);
    std::vector<int32_t> ids = tok.encode(text);
    assert(ids.size() <= 0xFFFF && "load_dataset_to_gpu: corpus too long for a uint16_t Jensor dim (chunking not implemented yet)");

    std::vector<float> ids_f(ids.begin(), ids.end());

    mytorch::Jensor<float> out({static_cast<uint16_t>(1), static_cast<uint16_t>(ids_f.size())},
                                mytorch::AllocateOnGpu);
    cudaMemcpy(out.data(), ids_f.data(), ids_f.size() * sizeof(float), cudaMemcpyHostToDevice);
    return out;
}

}  
