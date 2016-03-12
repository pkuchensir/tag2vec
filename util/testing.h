#ifndef UTIL_TESTING_H_
#define UTIL_TESTING_H_

#include <vector>

#include "document/document.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace util {

void CheckEqDocument(const Document& expect, const Document& actual);

void CheckDocumentIterator(DocumentIterator* iterator,
                           const std::vector<Document>& documents);

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

#endif  // UTIL_TESTING_H_
