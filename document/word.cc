#include "document/word.h"

#include <iostream>
#include <string>
#include <vector>

#include "document/vocabulary.h"
#include "util/io.h"

namespace deeplearning {
namespace embedding {

void Word::Write(std::ostream* out) const {
  Vocabulary::Item::Write(out);
  util::WriteBasicItem(out, probability_);
}

void Word::Read(std::istream* in, Word* word) {
  Vocabulary::Item::Read(in, word);
  util::ReadBasicItem(in, &word->probability_);
}

std::string Word::ToString() const {
  return Vocabulary::Item::ToString() + "," + std::to_string(probability_);
}

}  // namespace embedding
}  // namespace deeplearning
