#ifndef UTIL_HUFFMAN_H_
#define UTIL_HUFFMAN_H_

#include <iostream>
#include <string>
#include <vector>

#include "document/vocabulary.h"

namespace deeplearning {
namespace embedding {
namespace util {

class Huffman final {
 public:
   void Build(const std::vector<Vocabulary::Item*>& items);

   const std::vector<bool>& codes(size_t i) const { return codes_[i]; }
   const std::vector<size_t>& points(size_t i) const { return points_[i]; }

   void Write(std::ostream* out) const;
   static void Read(std::istream* in, Huffman* huffman);

   std::string ToString() const;

 private:
  void GenerateCodesAndPoints(const std::vector<size_t>& left_index_vec,
                              const std::vector<size_t>& right_index_vec,
                              size_t index, std::vector<bool>* code_vec,
                              std::vector<size_t>* point_vec);

 private:
   std::vector<std::vector<bool>> codes_;
   std::vector<std::vector<size_t>> points_;
};

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

#endif  // UTIL_HUFFMAN_H_
