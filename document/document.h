#ifndef DOCUMENT_DOCUMENT_H_
#define DOCUMENT_DOCUMENT_H_

#include <string>
#include <utility>
#include <vector>

namespace deeplearning {
namespace embedding {

class Document final {
 public:
   Document() = default;
   Document(const std::vector<std::string>& words,
            const std::vector<std::string>& tags)
       : words_(words), tags_(tags) {}
   Document(std::vector<std::string>&& words, std::vector<std::string>&& tags)
       : words_(std::move(words)), tags_(std::move(tags)) {}

   std::vector<std::string>& words() { return words_; }
   const std::vector<std::string>& words() const { return words_; }

   std::vector<std::string>& tags() { return tags_; }
   const std::vector<std::string>& tags() const { return tags_; }

 private:
  std::vector<std::string> words_;
  std::vector<std::string> tags_;
};

class DocumentIterator {
 public:
  virtual ~DocumentIterator() = default;

  // If no more document, returns nullptr.
  //
  // The caller will take the ownership of return value.
  virtual Document* NextDocument() = 0;
};

}  // namespace embedding
}  // namespace deeplearning

#endif  // DOCUMENT_DOCUMENT_H_
