#include "document/memory_document_iterator.h"

#include <vector>

#include "document.h"
#include "util/logging.h"
#include "util/testing.h"

namespace deeplearning {
namespace embedding {
namespace {

void TestMemoryDocumentIterator() {
  std::vector<Document> expect_documents = {
      Document({"a", "b", "c"}, {"1", "2"}), Document({"b", "c", "a"}, {"2"})};
  MemoryDocumentIterator iterator(expect_documents);
  util::CheckDocumentIterator(&iterator, expect_documents);

  iterator.Reset();
  util::CheckDocumentIterator(&iterator, expect_documents);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::TestMemoryDocumentIterator();
  LOG(INFO) << "PASS";
  return 0;
}
