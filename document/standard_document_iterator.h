#ifndef DOCUMENT_STANDARD_DOCUMENT_ITERATOR_H_
#define DOCUMENT_STANDARD_DOCUMENT_ITERATOR_H_

#include <istream>
#include <string>

#include "document/document.h"

namespace deeplearning {
namespace embedding {

class StandardDocumentIterator final : public DocumentIterator {
 public:
  StandardDocumentIterator(std::istream* in) : in_(in) {}

  Document* NextDocument() override;

  void Reset();

 private:
  std::istream* in_;
};

}  // namespace embedding
}  // namespace deepleraning

#endif  // DOCUMENT_STANDARD_DOCUMENT_ITERATOR_H_
