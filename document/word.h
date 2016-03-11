#ifndef DOCUMENT_WORD_H_
#define DOCUMENT_WORD_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "document/vocabulary.h"

namespace deeplearning {
namespace embedding {

class Word : public Vocabulary::Item {
 public:
  Word() = default;
  Word(const std::string& text) : Word(0, 1, text) {}
  Word(std::string&& text) : Word(0, 1, std::move(text)) {}
  Word(size_t index, size_t count, const std::string& text)
      : Word(index, count, text, 1) {}
  Word(size_t index, size_t count, std::string&& text)
      : Word(index, count, std::move(text), 1) {}
  Word(size_t index, size_t count, const std::string& text,
       float sample_probability)
      : Vocabulary::Item(index, count, std::move(text)),
        sample_probability_(sample_probability) {}
  Word(size_t index, size_t count, std::string&& text, float sample_probability)
      : Vocabulary::Item(index, count, text),
        sample_probability_(sample_probability) {}

  ~Word();

  float sample_probability() const { return sample_probability_; }
  void set_sample_probability(float sample_probability) {
    sample_probability_ = sample_probability;
  }

  std::vector<bool>* codes() { return codes_; }
  const std::vector<bool>* codes() const { return codes_; }

  std::vector<size_t>* points() { return points_; }
  const std::vector<size_t>* points() const { return points_; }

  void InitHsNode();

  void Write(std::ostream* out);
  static void Read(std::istream* in, Word* word);

 private:
  size_t index_;
  size_t count_;
  std::string text_;
  float sample_probability_;

  std::vector<bool>* codes_ = nullptr;
  std::vector<size_t>* points_ = nullptr;
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // DOCUMENT_WORD_H_
