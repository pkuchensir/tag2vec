#ifndef DOCUMENT_WORD_H_
#define DOCUMENT_WORD_H_

#include <iostream>
#include <string>
#include <vector>

namespace deeplearning {
namespace embedding {

class Word {
 public:
  size_t index_;
  size_t count_;
  float sample_probability_;
  std::string text_;

  // std::vector<bool>* codes_ = nullptr;
  // std::vector<size_t>* points_ = nullptr;

 public:
  Word() {}
  Word(size_t index, size_t count, std::string text)
      : index_(index), count_(count), text_(text) {}

  ~Word() {}

  /*
  std::vector<bool>* codes() { return codes_; }
  const std::vector<bool>* codes() const { return codes_; }

  std::vector<size_t>* points() { return points_; }
  const std::vector<size_t>* points() const { return points_; }
  */

  void Write(std::ostream* out);
  static void Read(std::istream* in, Word* word);
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // DOCUMENT_WORD_H_
