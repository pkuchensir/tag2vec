#ifndef UTIL_EVALUATE_H_
#define UTIL_EVALUATE_H_

#include <vector>

#include "document/document.h"
#include "model/model.h"

namespace deeplearning {
namespace embedding {
namespace util {

void Evaluate(Model* model, DocumentIterator* test_data);

void Evaluate(Model* model, const std::vector<Document>& test_data);

}  // namespace util
}  // namespace embedding
}  // namespace deeplearning

#endif  // UTIL_EVALUATE_H_
