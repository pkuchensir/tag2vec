#include "document/vocabulary.h"

#include <string>

#include "util/logging.h"

namespace deeplearning {
namespace embedding {

template <class ItemSubType>
void Vocabulary::AddItem(const std::string& item) {
  CHECK(!has_built) << "This vocabulary has already been built.";
  auto it = item_hash_.find(item);
  if (it != item_hash_.end()) {
    Item* item_ptr = it->second;
    item_ptr->set_count(item_ptr->count() + 1);
  } else {
    item_hash_[item] = new ItemSubType(item);
  }
}

}  // namespace embedding
}  // namespace deeplearning
