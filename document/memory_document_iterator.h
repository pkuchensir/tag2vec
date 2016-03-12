#ifndef DOCUMENT_MEMORY_DOCUMENT_ITERATOR_H_
#define DOCUMENT_MEMORY_DOCUMENT_ITERATOR_H_

#include <vector>

#include "document/document.h"

namespace deeplearning {
namespace embedding {

class MemoryDocumentIterator final : public DocumentIterator {
 public:
  MemoryDocumentIterator(const std::vector<Document>& documents)
      : documents_(&documents) {}

  Document* NextDocument() override;

  void Reset() override;

 private:
  const std::vector<Document>* const documents_;  // NOT OWNED
  size_t index_ = 0;
};

}  // namespace embedding
}  // namespace deepleraning

#endif  // DOCUMENT_MEMORY_DOCUMENT_ITERATOR_H_
