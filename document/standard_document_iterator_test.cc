#include "document/standard_document_iterator.h"

#include <string>
#include <vector>

#include "document.h"
#include "util/logging.h"
#include "util/testing.h"

namespace deeplearning {
namespace embedding {
namespace {

void TestStandardDocumentIterator() {
  std::string text =
    "a b  c \n"
    "  1 2\n"
    " b  c a \n"
    "2\n";
  std::istringstream iss(text);
  StandardDocumentIterator iterator(&iss);
  std::vector<Document> expect_documents = {
      Document({"a", "b", "c"}, {"1", "2"}), Document({"b", "c", "a"}, {"2"})};
  util::CheckDocumentIterator(&iterator, expect_documents);

  iterator.Reset();
  util::CheckDocumentIterator(&iterator, expect_documents);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::TestStandardDocumentIterator();
  LOG(INFO) << "PASS";
  return 0;
}
