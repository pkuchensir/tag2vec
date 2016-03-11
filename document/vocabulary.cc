#include "document/vocabulary.h"

#include <algorithm>
#include <iostream>

#include "util/io.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {

void Vocabulary::Build(size_t min_count) {
  CHECK(!has_built) << "This vocabulary has already been built.";
  has_built = true;

  size_t num_original_items = 0;
  std::vector<Item*> to_remove_items;
  for (const auto& key_item : item_hash_) {
    Item* item = key_item.second;
    to_remove_items.push_back(item);
    num_original_items += item->count();
  }

  size_t num_to_remove = 0;
  for (Item* item : to_remove_items) {
    item_hash_.erase(item->text());
    num_to_remove += item->count();
    delete item;
  }

  total_items_ = num_original_items - num_to_remove;

  for (const auto& key_item : item_hash_) {
    Item* item = key_item.second;
    items_.push_back(item);
  }
  std::sort(items_.begin(), items_.end(),
            [](const Item* item1, const Item* item2) {
              return item1->count() > item2->count();
            });
  for (size_t i = 0; i < items_.size(); ++i) {
    items_[i]->set_index(i);
  }
}

void Vocabulary::Write(std::ostream* out) const {
  util::WriteBasicItem(out, total_items_);
  size_t items_size = items_.size();
  util::WriteBasicItem(out, items_size);
  for (Item* item : items_) {
    item->Write(out);
  }
}

void Vocabulary::Read(std::istream* in, Vocabulary* vocabulary) {
  CHECK(vocabulary->CheckEmpty()) << "vocabulary should be empty.";
  util::ReadBasicItem(in, &vocabulary->total_items_);
  size_t items_size;
  util::ReadBasicItem(in, &items_size);
  vocabulary->items_.resize(items_size);
  for (size_t i = 0; i < items_size; ++i) {
    Item::Read(in, vocabulary->items_[i]);
    vocabulary->item_hash_[vocabulary->items_[i]->text()] =
        vocabulary->items_[i];
  }
}

bool Vocabulary::CheckEmpty() const {
  return total_items_ == 0 && items_.empty() && item_hash_.empty() &&
         !has_built;
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
