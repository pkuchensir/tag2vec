#include "document/vocabulary.h"

#include <istream>
#include <string>

#include "util/io.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {

template <class ItemSubClass>
void Vocabulary::AddItem(const std::string& item) {
  CHECK(!has_built_) << "Vocabulary has already been built.";
  auto it = item_hash_.find(item);
  if (it != item_hash_.end()) {
    Item* item_ptr = it->second;
    item_ptr->set_count(item_ptr->count() + 1);
  } else {
    item_hash_[item] = new ItemSubClass(item);
  }
}

template <class ItemSubClass>
void Vocabulary::Read(std::istream* in, Vocabulary* vocabulary) {
  CHECK(vocabulary->CheckEmpty()) << "Vocabulary should be empty.";
  util::ReadBasicItem(in, &vocabulary->total_items_);
  size_t items_size;
  util::ReadBasicItem(in, &items_size);
  vocabulary->items_.resize(items_size);
  for (size_t i = 0; i < items_size; ++i) {
    vocabulary->items_[i] = new ItemSubClass();
    ItemSubClass::Read(in, vocabulary->items_[i]);
    vocabulary->item_hash_[vocabulary->items_[i]->text()] =
        vocabulary->items_[i];
  }
  vocabulary->has_built_ = true;
}

}  // namespace embedding
}  // namespace deeplearning
