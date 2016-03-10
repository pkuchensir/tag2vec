#ifndef DOCUMENT_WORD_H_
#define DOCUMENT_WORD_H_

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace deeplearning {
namespace embedding {

class Word {
 public:
  size_t index_;
  size_t count_;
  std::string text_;
  float sample_probability_;

  std::vector<bool>* codes_ = nullptr;
  std::vector<size_t>* points_ = nullptr;

 public:
  Word() = default;
  Word(size_t index, size_t count, const std::string& text)
      : Word(index, count, text, 0) {}
  Word(size_t index, size_t count, std::string&& text)
      : Word(index, count, text, 0) {}
  Word(size_t index, size_t count, const std::string& text, float sample_probability)
      : index_(index),
        count_(count),
        text_(text),
        sample_probability_(sample_probability) {}
  Word(size_t index, size_t count, std::string&& text, float sample_probability)
      : index_(index),
        count_(count),
        text_(std::move(text)),
        sample_probability_(sample_probability) {}

  ~Word();

  size_t index() const { return index_; }
  void set_index(size_t index) { index_ = index; }

  size_t count() const { return count_; }
  void set_count(size_t count) { count_ = count; }

  const std::string& text() const { return text_; }
  void set_text(const std::string& text) { text_ = text; }
  void set_text(std::string&& text) { text_ = std::move(text); }

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
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // DOCUMENT_WORD_H_
