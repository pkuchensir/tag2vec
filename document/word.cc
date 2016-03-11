#include "document/word.h"

#include <iostream>
#include <vector>

#include "document/vocabulary.h"
#include "util/io.h"

namespace deeplearning {
namespace embedding {

Word::~Word() {
  if (codes_) delete codes_;
  if (points_) delete points_;
}

void Word::InitHsNode() {
  codes_ = new std::vector<bool>;
  points_ = new std::vector<size_t>;
}

void Word::Write(std::ostream* out) {
  Vocabulary::Item::Write(out);
  util::WriteBasicItem(out, sample_probability_);

  bool has_codes = codes_;
  util::WriteBasicItem(out, has_codes);
  if (has_codes) {
    util::WriteBasicItemVector(out, *codes_);
  }

  bool has_points = points_;
  util::WriteBasicItem(out, has_points);
  if (has_points) {
    util::WriteBasicItemVector(out, *points_);
  }
}

void Word::Read(std::istream* in, Word* word) {
  Vocabulary::Item::Read(in, word);
  util::ReadBasicItem(in, &word->sample_probability_);

  bool has_codes;
  util::ReadBasicItem(in, &has_codes);
  if (has_codes) {
    word->codes_ = new std::vector<bool>;
    util::ReadBasicItemVector(in, word->codes_);
  }

  bool has_points;
  util::ReadBasicItem(in, &has_points);
  if (has_points) {
    word->points_ = new std::vector<size_t>;
    util::ReadBasicItemVector(in, word->points_);
  }
}


}  // namespace embedding
}  // namespace deeplearning
