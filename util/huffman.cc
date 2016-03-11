#include "util/huffman.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "document/vocabulary.h"

namespace deeplearning {
namespace embedding {
namespace util {
namespace {

void BuildHuffmanTree(const std::vector<Vocabulary::Item*>& items,
                      std::vector<size_t>* left_index_vec,
                      std::vector<size_t>* right_index_vec) {
  left_index_vec->resize(items.size() - 1);
  right_index_vec->resize(items.size() - 1);

  std::vector<size_t> count(items.size() * 2);
  for (size_t i = 0; i < items.size(); ++i) {
    count[i] = items[i]->count();
  }
  std::vector<size_t> index_vec(items.size());
  std::iota(index_vec.begin(), index_vec.end(), 0);
  auto item_index_greater = [&count](size_t i, size_t j) {
    return count[i] > count[j];
  };
  std::make_heap(index_vec.begin(), index_vec.end(), item_index_greater);

  for (size_t i = 0; i + 1 < items.size(); ++i) {
    size_t u = index_vec[0];
    std::pop_heap(index_vec.begin(), index_vec.end(), item_index_greater);
    index_vec.pop_back();
    size_t v = index_vec[0];
    std::pop_heap(index_vec.begin(), index_vec.end(), item_index_greater);
    index_vec.pop_back();

    size_t k = items.size() + i;
    count[k] = count[u] + count[v];
    (*left_index_vec)[i] = u;
    (*right_index_vec)[i] = v;
    index_vec.push_back(k);
    std::push_heap(index_vec.begin(), index_vec.end(), item_index_greater);
  }
}

}  // namespace

void Huffman::Build(const std::vector<Vocabulary::Item*>& items) {
  CHECK(items.size() >= 2) << "items should be more than 2.";
  CHECK(codes_.empty() && points_.empty()) << "Huffman should be empty.";

  std::vector<size_t> left_index_vec, right_index_vec;
  BuildHuffmanTree(items, &left_index_vec, &right_index_vec);

  std::vector<bool> code_vec;
  std::vector<size_t> point_vec;
  codes_.resize(items.size());
  points_.resize(items.size());
  GenerateCodesAndPoints(left_index_vec, right_index_vec,
                         items.size() + left_index_vec.size() - 1, &code_vec,
                         &point_vec);
}

void Huffman::Write(std::ostream* out) const {
}

void Huffman::Read(std::istream* in, Huffman* huffman) {
}

std::string Huffman::ToString() const {
  std::string ans;
  for (size_t i = 0; i < codes_.size(); ++i) {
    for (bool code : codes_[i]) {
      ans += code ? "1" : "0";
    }
    ans += ": ";
    for (size_t point : points_[i]) {
      ans += std::to_string(point) + ",";
    }
    ans += "\n";
  }
  return ans;
}

void Huffman::GenerateCodesAndPoints(const std::vector<size_t>& left_index_vec,
                                     const std::vector<size_t>& right_index_vec,
                                     size_t index, std::vector<bool>* code_vec,
                                     std::vector<size_t>* point_vec) {
  const size_t items_size = left_index_vec.size() + 1;
  if (index < items_size) {
    codes_[index] = *code_vec;
    points_[index] = *point_vec;
    return;
  }
  size_t smaller_index = index - items_size;
  point_vec->push_back(smaller_index);
  code_vec->push_back(true);
  GenerateCodesAndPoints(left_index_vec, right_index_vec,
                         left_index_vec[smaller_index], code_vec,
                         point_vec);
  code_vec->back() = false;
  GenerateCodesAndPoints(left_index_vec, right_index_vec,
                         right_index_vec[smaller_index], code_vec,
                         point_vec);

  code_vec->pop_back();
  point_vec->pop_back();
}

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning
