#include "document/word.h"

#include "util/io.h"

namespace deeplearning {
namespace embedding {

/*
Word::~Word() {
  if (codes_) delete codes_;
  if (points_) delete points_;
}
*/

void Word::Write(std::ostream* out) {
  util::WriteBasicItem(out, index_);
  util::WriteBasicItem(out, count_);
  util::WriteBasicItem(out, sample_probability_);
  out->write(text_.c_str(), text_.size() + 1);
}

void Word::Read(std::istream* in, Word* word) {
  in->read((char*)&word->index_, sizeof(word->index_));
  in->read((char*)&word->count_, sizeof(word->count_));
  in->read((char*)&word->sample_probability_,
           sizeof(word->sample_probability_));
  getline(*in, word->text_, '\0');
}


}  // namespace embedding
}  // namespace deeplearning
