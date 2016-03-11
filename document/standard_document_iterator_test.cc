#include "document/standard_document_iterator.h"

#include <string>
#include <vector>

#include "document.h"
#include "util/logging.h"

namespace deeplearning {
namespace embedding {
namespace {

void CheckEqDocument(const Document& expect, const Document& actual) {
  CHECK_EQ(expect.words().size(), actual.words().size())
      << "Size of words mismathced.";
  for (size_t i = 0; i < expect.words().size(); ++i) {
    CHECK_EQ(expect.words()[i], actual.words()[i]) << "words[ " << i
                                                   << "] mismatched.";
  }
  CHECK_EQ(expect.tags().size(), actual.tags().size())
      << "Size of tags mismathced.";
  for (size_t i = 0; i < expect.tags().size(); ++i) {
    CHECK_EQ(expect.tags()[i], actual.tags()[i]) << "tags[ " << i
                                                   << "] mismatched.";
  }
}

void CheckDocumentIterator(DocumentIterator* iterator,
                           const std::vector<Document>& documents) {
  size_t i;
  for (i = 0; i < documents.size(); ++i) {
    const Document* expect_doc = iterator->NextDocument();
    if (expect_doc) {
      CheckEqDocument(*expect_doc, documents[i]);
      delete expect_doc;
    } else {
      LOG(FATAL) << "iterator exhausted.";
    }
  }
  const Document* doc = iterator->NextDocument();
  if (doc) {
    delete doc;
    LOG(FATAL) << "Too much documents in iterator.";
  }
}

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
  CheckDocumentIterator(&iterator, expect_documents);

  iterator.Reset();
  CheckDocumentIterator(&iterator, expect_documents);
}

}  // namespace
}  // namespace embedding
}  // namespace deeplearning

int main(int argc, char** argv) {
  deeplearning::embedding::TestStandardDocumentIterator();
  LOG(INFO) << "PASS";
  return 0;
}
