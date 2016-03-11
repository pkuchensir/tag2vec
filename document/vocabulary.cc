#include "document/vocabulary.h"

#include <iostream>

#include "util/io.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {

void Vocabulary::Write(std::ostream* out) const {
  size_t items_size = items_.size();
  util::WriteBasicItem(out, items_size);
  for (Item* item : items_) {
    item->Write(out);
  }
}

void Vocabulary::Read(std::istream* in, Vocabulary* vocabulary) {
  CHECK_EQ(0, vocabulary->items_.size()) << "items_ should be empty.";
  CHECK_EQ(0, vocabulary->item_hash_.size()) << "item_hash_ should be empty.";
  size_t items_size;
  util::ReadBasicItem(in, &items_size);
  vocabulary->items_.resize(items_size);
  for (size_t i = 0; i < items_size; ++i) {
    Item::Read(in, vocabulary->items_[i]);
    vocabulary->item_hash_[vocabulary->items_[i]->text()] =
        vocabulary->items_[i];
  }
}

void Vocabulary::Item::Write(std::ostream* out) const {
  util::WriteBasicItem(out, index_);
  util::WriteBasicItem(out, count_);
  util::WriteString(out, text_);
}

void Vocabulary::Item::Read(std::istream* in, Vocabulary::Item* item) {
  util::ReadBasicItem(in, &item->index_);
  util::ReadBasicItem(in, &item->count_);
  util::ReadString(in, &item->text_);
}

}  // namespace embedding
}  // namespace deeplearning
