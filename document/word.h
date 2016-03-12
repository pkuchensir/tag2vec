#ifndef DOCUMENT_WORD_H_
#define DOCUMENT_WORD_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "document/vocabulary.h"

namespace deeplearning {
namespace embedding {

class Word final : public Vocabulary::Item {
 public:
  Word() = default;
  Word(const std::string& text) : Word(0, 1, text) {}
  Word(std::string&& text) : Word(0, 1, std::move(text)) {}
  Word(size_t index, size_t count, const std::string& text)
      : Word(index, count, text, 1) {}
  Word(size_t index, size_t count, std::string&& text)
      : Word(index, count, std::move(text), 1) {}
  Word(size_t index, size_t count, const std::string& text,
       float probability)
      : Vocabulary::Item(index, count, std::move(text)),
        probability_(probability) {}
  Word(size_t index, size_t count, std::string&& text, float probability)
      : Vocabulary::Item(index, count, text),
        probability_(probability) {}

  float probability() const { return probability_; }
  void set_probability(float probability) {
    probability_ = probability;
  }

  void Write(std::ostream* out) const override;
  static void Read(std::istream* in, Word* word);

  std::string ToString() const override;

 private:
  size_t index_;
  size_t count_ = 1;
  std::string text_;
  float probability_ = 1.0;
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // DOCUMENT_WORD_H_
