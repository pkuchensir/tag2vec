#include "document/memory_document_iterator.h"

#include <istream>
#include <string>

#include "util/util.h"

namespace deeplearning {
namespace embedding {

Document* MemoryDocumentIterator::NextDocument() {
  if (index_ < documents_->size()) {
    return new Document((*documents_)[index_++]);
  }
  return nullptr;
}

void MemoryDocumentIterator::Reset() {
  index_ = 0;
}

}  // namespace embedding
}  // namespace deepleraning
