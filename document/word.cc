#include "document/word.h"

#include <iostream>
#include <vector>

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
  util::WriteBasicItem(out, index_);
  util::WriteBasicItem(out, count_);
  util::WriteBasicItem(out, sample_probability_);
  util::WriteString(out, text_);

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
  util::ReadBasicItem(in, &word->index_);
  util::ReadBasicItem(in, &word->count_);
  util::ReadBasicItem(in, &word->sample_probability_);
  util::ReadString(in, &word->text_);

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
