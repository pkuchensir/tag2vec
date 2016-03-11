#include "document/standard_document_iterator.h"

#include <istream>
#include <string>

#include "util/util.h"

namespace deeplearning {
namespace embedding {

Document* StandardDocumentIterator::NextDocument() {
  std::string words;
  bool has_next = std::getline(*in_, words);
  if (has_next) {
    std::string tags;
    has_next = std::getline(*in_, tags);
    Document* document = new Document(util::Split(words, " \t\n\r"),
                                      util::Split(tags, " \t\n\r"));
    return document;
  }
  return nullptr;
}

void StandardDocumentIterator::Reset() {
  in_->seekg(0);
}

}  // namespace embedding
}  // namespace deepleraning
